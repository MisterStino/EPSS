#!/usr/bin/env python
"""
Ray Tune Hyperparameter Optimization for EPSS LSTM Forecaster
Following the exact walk-through pattern for production-ready tuning
"""

import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
import sys

# Ray Tune imports - following official docs pattern
import ray
from ray import tune, air
from ray.air import session
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.schedulers import ASHAScheduler
import ray.train.torch

# Import our streaming dataset and model components
from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed

# ───────────────────────────── MODEL DEFINITIONS ──────────────────────────────────────
class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_num: int, n_bool: int, cat_sizes,
                 horizon: int = 30, hidden: int = 512,
                 layers: int = 3, emb_dim: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        # Handle case where no categorical columns exist
        # Build the embedding with the real required size, not len(dict)
        self.emb = nn.ModuleList([nn.Embedding(s, emb_dim, padding_idx=0) for s in cat_sizes])
        in_dim   = n_num + n_bool + emb_dim * len(cat_sizes)

        self.lstm = nn.LSTM(in_dim, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, num, boo, cat):
        # Handle case where no categorical embeddings exist  
        if len(self.emb):
            cat = cat.long()
            # Safety check: clamp any out-of-bounds indices to UNK (0)
            cat_sizes = [emb.num_embeddings for emb in self.emb]
            for i, vocab_size in enumerate(cat_sizes):
                out_of_bounds = cat[..., i] >= vocab_size
                if out_of_bounds.any():
                    print(f"WARNING: Found {out_of_bounds.sum()} out-of-bounds indices in column {i}, clamping to UNK")
                    cat[..., i] = torch.clamp(cat[..., i], 0, vocab_size - 1)
            e = torch.cat([emb(cat[..., i]) for i, emb in enumerate(self.emb)], dim=-1)
            x = torch.cat([num, boo.float(), e], dim=-1)
        else:
            x = torch.cat([num, boo.float()], dim=-1)
        h, _ = self.lstm(x)
        return self.head(h)

def masked_mse(pred, true, m_t, m_h, m_eval):
    """Masked MSE loss for variable-length sequences"""
    mask = m_t * m_eval                        # [B,L]
    err  = (pred - true) ** 2                  # [B,L,H]
    return (err * mask.unsqueeze(-1) * m_h).sum() / m_h.sum()

# ───────────────────────────── DETERMINISTIC TRAIN FUNCTION ────────────────────────────
def train_tune(config):
    """
    Deterministic training function for Ray Tune
    Follows official Ray docs pattern with reproducibility, mixed precision, and checkpointing
    """
    # ------------- 1. Reproducibility (Ray docs recommended pattern) -----------------
    ray.train.torch.enable_reproducibility()  # Official Ray reproducibility helper
    seed = config.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed)
    
    # CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")  # Tensor-Core optimization
    
    print(f"[TRIAL] Starting with seed={seed}, hidden={config['hidden_size']}, batch={config['batch_size']}")

    # ------------- 2. Resource Constraint Check -----------------
    # Reject configs that would cause OOM (as recommended in walk-through)
    memory_requirement = config["hidden_size"] * config["batch_size"]
    if memory_requirement > 8192 * 1536:  # Conservative GPU memory limit
        print(f"[TRIAL] Skipping due to memory constraint: {memory_requirement} > {8192 * 1536}")
        session.report({"epoch": 0, "train_loss": float('inf'), "val_loss": float('inf')}, skip=True)
        return

    # ------------- 3. Data Loading (Memory-efficient streaming) ----------------------------
    print("[TRIAL] Creating streaming datasets...")
    ds_train = CVEIterableDatasetFixed(config["arrow_path"], horizon=30)
    ds_val = CVEIterableDatasetFixed(config["arrow_path"], horizon=30)
    
    # Training DataLoader with tuned parameters
    dl_train = DataLoader(
        ds_train,
        batch_size=config["batch_size"],
        shuffle=False,  # IterableDataset doesn't support shuffle
        collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=30),
        num_workers=config["num_workers"],
        pin_memory=config["num_workers"] > 0,
        persistent_workers=config["num_workers"] > 0,
        prefetch_factor=1  # Keep low for memory efficiency
    )
    
    # Validation DataLoader (single worker for deterministic order)
    dl_val = DataLoader(
        ds_val,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=partial(pad_and_mask_fixed, flag_kind="val", horizon=30),
        num_workers=0,  # Single worker for validation consistency
        pin_memory=False,
        prefetch_factor=1
    )
    
    print(f"[TRIAL] Data loaders created with batch_size={config['batch_size']}, num_workers={config['num_workers']}")

    # ------------- 4. Model Setup (Dynamic dimensions from data) ---------------------------
    print("[TRIAL] Setting up model with dynamic dimensions...")
    
    # Get dimensions from first batch
    sample_batch = next(iter(dl_train))
    n_num = sample_batch[0].shape[-1]   # Numeric features dimension
    n_bool = sample_batch[1].shape[-1]  # Boolean features dimension  
    n_cat = sample_batch[2].shape[-1]   # Categorical features dimension
    
    # Load vocabulary for categorical embeddings
    with open(config["vocab_path"], 'r') as f:
        vocab = json.load(f)
    cat_sizes = [len(vocab[col]) for col in vocab.keys()]
    
    print(f"[TRIAL] Detected dimensions: {n_num} numeric, {n_bool} boolean, {n_cat} categorical")
    print(f"[TRIAL] Categorical vocab sizes: {cat_sizes}")
    
    # Create model with tuned hyperparameters
    model = Seq2SeqLSTM(
        n_num=n_num,
        n_bool=n_bool,
        cat_sizes=cat_sizes,
        horizon=30,
        hidden=config["hidden_size"],
        layers=config["lstm_layers"],
        emb_dim=config["emb_dim"],
        dropout=config["dropout"]
    ).cuda()
    
    # Optimizer with tuned parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    
    # Mixed precision scaler (official PyTorch pattern)
    scaler = torch.cuda.amp.GradScaler()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TRIAL] Model ready: {total_params:,} parameters")

    # ------------- 5. Training Loop with Early Reporting -------------------
    print(f"[TRIAL] Starting training for max {config['max_epochs']} epochs...")
    
    for epoch in range(1, config["max_epochs"] + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch in dl_train:
            # Unpack batch (9 elements including lengths tensor)
            num, boo, cat, Y, mT, mH, mE, date_pad, lengths = batch
            
            # Move to GPU (cat synchronously to avoid embedding race conditions)
            num = num.cuda(non_blocking=True)
            boo = boo.cuda(non_blocking=True)
            cat = cat.cuda(non_blocking=False)  # Synchronous for index tensor
            Y = Y.cuda(non_blocking=True)
            mT = mT.cuda(non_blocking=True)
            mH = mH.cuda(non_blocking=True)
            mE = mE.cuda(non_blocking=True)
            
            # Forward pass with mixed precision
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            with torch.cuda.amp.autocast():
                loss = masked_mse(model(num, boo, cat), Y, mT, mH, mE)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            num_batches += 1
            
        train_loss = running_loss / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in dl_val:
                # Unpack batch
                num, boo, cat, Y, mT, mH, mE, date_pad, lengths = batch
                
                # Move to GPU
                num = num.cuda(non_blocking=True)
                boo = boo.cuda(non_blocking=True)
                cat = cat.cuda(non_blocking=False)
                Y = Y.cuda(non_blocking=True) 
                mT = mT.cuda(non_blocking=True)
                mH = mH.cuda(non_blocking=True)
                mE = mE.cuda(non_blocking=True)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    loss = masked_mse(model(num, boo, cat), Y, mT, mH, mE)
                
                val_loss += loss.item()
                val_batches += 1
                
        val_loss = val_loss / val_batches
        
        print(f"[TRIAL] Epoch {epoch:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # ------------- 6. Ray Tune Reporting & Checkpointing ----------
        # Save checkpoint (Ray AIR best practice pattern)
        with session.get_checkpoint_dir(epoch) as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "train_loss": train_loss,
                "val_loss": val_loss
            }, checkpoint_path)
        
        # Report metrics to Ray Tune
        session.report({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
    
    print(f"[TRIAL] Training completed. Final val_loss: {val_loss:.4f}")

# ───────────────────────────── MAIN TUNING SCRIPT ────────────────────────────
def main():
    """Main Ray Tune orchestration following official patterns"""
    
    print("🚀 STARTING RAY TUNE HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # FIXED: Standardized paths for cross-platform consistency
    WORK_DIR = Path("ml_pipeline/work")
    ARROW_PATH = WORK_DIR / "epss_stage1.arrow"
    VOCAB_PATH = WORK_DIR / "vocab.json"
    
    # Validate required files exist
    if not ARROW_PATH.exists():
        raise FileNotFoundError(f"Arrow file not found: {ARROW_PATH}")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError(f"Vocab file not found: {VOCAB_PATH}")
    
    print(f"[SETUP] Using Arrow file: {ARROW_PATH}")
    print(f"[SETUP] Using vocabulary: {VOCAB_PATH}")
    
    # ------------- 2. Search Space Definition (Following Ray docs patterns) -------------
    search_space = {
        # Tunable hyperparameters
        "seed": tune.randint(0, 10000),
        "hidden_size": tune.qloguniform(384, 1024, 64),  # Power-of-2 quantized log space
        "lstm_layers": tune.choice([2, 3, 4]),
        "emb_dim": tune.choice([8, 16]),  # Embedding dimension
        "dropout": tune.uniform(0.1, 0.5),
        "lr": tune.loguniform(1e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-5, 1e-2),
        "clip_grad": tune.choice([0.5, 1.0, 2.0]),
        "batch_size": tune.choice([448, 640, 768, 896]),  # GPU memory optimized sizes
        "num_workers": tune.choice([4, 6, 8]),
        
        # Fixed constants
        "arrow_path": str(ARROW_PATH),
        "vocab_path": str(VOCAB_PATH),
        "max_epochs": 12,
    }
    
    print("[SETUP] Search space configured with BayesOpt-friendly distributions")
    
    # ------------- 3. Algorithm & Scheduler (Ray docs recommendations) -------------
    # BayesOpt for efficient search in continuous spaces
    search_algorithm = BayesOptSearch(
        metric="val_loss",
        mode="min"
    )
    
    # ASHA for aggressive early stopping (Ray's default recommendation)
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        grace_period=1,        # Let all trials run at least 1 epoch
        max_t=12,             # Maximum epochs
        reduction_factor=3    # Keep 1/3 of trials at each rung
    )
    
    print("[SETUP] Using BayesOptSearch + ASHAScheduler (Ray recommended)")
    print("[SETUP] ASHA rungs: 1 → 3 → 9 → 12 epochs")
    
    # ------------- 4. Resource-Aware Trainable -------------
    # Fractional GPU allocation for concurrent trials
    trainable_with_resources = tune.with_resources(
        train_tune,
        {"gpu": 0.85, "cpu": 3}  # Allow 2 trials per 48GB GPU
    )
    
    print("[SETUP] Resource allocation: 0.85 GPU + 3 CPU per trial")
    print("[SETUP] Supports ~2 concurrent trials on 48GB GPU")
    
    # ------------- 5. Launch Tuner (Ray AIR pattern) -------------
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=60,           # ~1 GPU-day budget
            search_alg=search_algorithm,
            scheduler=scheduler,
            metric="val_loss",
            mode="min"
        ),
        run_config=air.RunConfig(
            name="epss_lstm_asha_bayesopt",
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=3,                              # Keep only best 3 checkpoints
                checkpoint_score_attribute="val_loss",      # Rank by validation loss
                checkpoint_score_order="min"                # Lower is better
            ),
            verbose=1,  # Show trial progress
            # Optional: failure tolerance
            failure_config=air.FailureConfig(max_failures=5)
        )
    )
    
    print("[SETUP] Tuner configured for 60 trials with failure tolerance")
    print("=" * 60)
    
    # ------------- 6. Execute Hyperparameter Search -------------
    print("🎯 LAUNCHING HYPERPARAMETER SEARCH...")
    results = tuner.fit()
    
    # ------------- 7. Extract Best Results -------------
    print("\n" + "=" * 60)
    print("📊 HYPERPARAMETER SEARCH COMPLETED")
    print("=" * 60)
    
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    print(f"🏆 BEST VALIDATION LOSS: {best_result.metrics['val_loss']:.4f}")
    print(f"📁 BEST CHECKPOINT: {best_result.checkpoint.path}")
    print("\n🔧 BEST HYPERPARAMETERS:")
    for key, value in best_result.config.items():
        if key not in ["arrow_path", "vocab_path"]:  # Skip path constants
            print(f"   {key:15s}: {value}")
    
    # Save best config for production use
    best_config_path = "ml_pipeline/results/best_config.json"
    Path("ml_pipeline/results").mkdir(parents=True, exist_ok=True)
    
    with open(best_config_path, 'w') as f:
        json.dump(best_result.config, f, indent=2, default=str)
    
    print(f"\n💾 Best configuration saved to: {best_config_path}")
    print(f"💾 Best checkpoint available at: {best_result.checkpoint.path}")
    
    print("\n" + "=" * 60)
    print("✅ READY FOR PRODUCTION TRAINING")
    print("   1. Use the best config for final train+val retraining")
    print("   2. Load checkpoint with: torch.load(checkpoint_path)")
    print("   3. Test on held-out set exactly once")
    print("=" * 60)
    
    return results, best_result

if __name__ == "__main__":
    # Initialize Ray (local cluster)
    ray.init(
        num_gpus=1,
        num_cpus=8,
        object_store_memory=2_000_000_000,  # 2GB object store
        ignore_reinit_error=True
    )
    
    try:
        results, best_result = main()
        print("\n🎉 Hyperparameter optimization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during hyperparameter optimization: {e}")
        raise
    finally:
        ray.shutdown() 