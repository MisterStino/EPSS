#!/usr/bin/env python
"""
Final Model Retraining with Best Ray Tune Configuration
Loads the optimal hyperparameters and trains on train+val for final evaluation
"""

import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import xarray as xr

# Import our components
from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed

# ───────────────────────────── MODEL DEFINITIONS ──────────────────────────────────────
class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_num: int, n_bool: int, cat_sizes,
                 horizon: int = 30, hidden: int = 512,
                 layers: int = 3, emb_dim: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(s, emb_dim, padding_idx=0) for s in cat_sizes])
        in_dim   = n_num + n_bool + emb_dim * len(cat_sizes)

        self.lstm = nn.LSTM(in_dim, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, num, boo, cat):
        if len(self.emb):
            cat = cat.long()
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

def load_best_config(config_path: str = "ml_pipeline/results/best_config.json"):
    """Load the best configuration from Ray Tune results"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    """Final training with optimized hyperparameters"""
    
    print("🚀 FINAL MODEL TRAINING WITH OPTIMAL HYPERPARAMETERS")
    print("=" * 60)
    
    # ------------- 1. Load Best Configuration -------------
    config_path = "ml_pipeline/results/best_config.json"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Best config not found: {config_path}")
    
    best_config = load_best_config(config_path)
    print(f"[SETUP] Loaded best configuration from: {config_path}")
    print("\n🔧 OPTIMAL HYPERPARAMETERS:")
    for key, value in best_config.items():
        if key not in ["arrow_path", "vocab_path"]:
            print(f"   {key:15s}: {value}")
    
    # ------------- 2. Setup Deterministic Training -------------
    torch.manual_seed(best_config["seed"])
    np.random.seed(best_config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[SETUP] Using device: {device}")
    
    # ------------- 3. Create Final Training Dataset (Train + Val) -------------
    print("\n[DATA] Creating final training dataset (merging train + val)...")
    
    # Create a special dataset that uses both train and val splits
    class FinalTrainingDataset(CVEIterableDatasetFixed):
        """Uses both train and val data for final training"""
        pass
    
    final_train_ds = FinalTrainingDataset(best_config["arrow_path"], horizon=30)
    test_ds = CVEIterableDatasetFixed(best_config["arrow_path"], horizon=30)
    
    # Create DataLoaders with optimal configuration
    final_train_loader = DataLoader(
        final_train_ds,
        batch_size=best_config["batch_size"],
        shuffle=False,
        collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=30),  # Mix train+val
        num_workers=best_config["num_workers"],
        pin_memory=best_config["num_workers"] > 0,
        persistent_workers=best_config["num_workers"] > 0,
        prefetch_factor=1
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=best_config["batch_size"],
        shuffle=False,
        collate_fn=partial(pad_and_mask_fixed, flag_kind="test", horizon=30),
        num_workers=0,  # Single worker for test consistency
        pin_memory=False,
        prefetch_factor=1
    )
    
    print(f"[DATA] Final training loader: batch_size={best_config['batch_size']}")
    
    # ------------- 4. Model Setup with Optimal Architecture -------------
    print("\n[MODEL] Setting up model with optimal architecture...")
    
    # Get dimensions from data
    sample_batch = next(iter(final_train_loader))
    n_num = sample_batch[0].shape[-1]
    n_bool = sample_batch[1].shape[-1]
    n_cat = sample_batch[2].shape[-1]
    
    # Load vocabulary
    with open(best_config["vocab_path"], 'r') as f:
        vocab = json.load(f)
    cat_sizes = [len(vocab[col]) for col in vocab.keys()]
    
    # Create model with optimal hyperparameters
    model = Seq2SeqLSTM(
        n_num=n_num,
        n_bool=n_bool,
        cat_sizes=cat_sizes,
        horizon=30,
        hidden=best_config["hidden_size"],
        layers=best_config["lstm_layers"],
        emb_dim=best_config["emb_dim"],
        dropout=best_config["dropout"]
    ).to(device)
    
    # Optimal optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_config["lr"],
        weight_decay=best_config["weight_decay"]
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Model created: {total_params:,} parameters")
    print(f"[MODEL] Architecture: {best_config['lstm_layers']} layers, {best_config['hidden_size']} hidden")
    
    # ------------- 5. Final Training (Full Budget) -------------
    print(f"\n[TRAINING] Starting final training for {best_config['max_epochs']} epochs...")
    print("=" * 60)
    
    history = {"epoch": [], "train_loss": []}
    
    for epoch in range(1, best_config["max_epochs"] + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        # Training loop with progress bar
        pbar = tqdm(final_train_loader, desc=f"Epoch {epoch:2d}/{best_config['max_epochs']}")
        for batch in pbar:
            # Unpack batch
            num, boo, cat, Y, mT, mH, mE, date_pad, lengths = batch
            
            # Move to device
            num = num.to(device, non_blocking=True)
            boo = boo.to(device, non_blocking=True)
            cat = cat.to(device, non_blocking=False)
            Y = Y.to(device, non_blocking=True)
            mT = mT.to(device, non_blocking=True)
            mH = mH.to(device, non_blocking=True)
            mE = mE.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = masked_mse(model(num, boo, cat), Y, mT, mH, mE)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), best_config["clip_grad"])
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        epoch_loss = running_loss / num_batches
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        
        print(f"Epoch {epoch:2d}: train_loss = {epoch_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("🎯 FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    # ------------- 6. Final Test Evaluation (Once Only!) -------------
    model.eval()
    test_mse = test_mae = test_n = 0.0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            num, boo, cat, Y, mT, mH, mE, date_pad, lengths = batch
            
            # Move to device
            num = num.to(device, non_blocking=True)
            boo = boo.to(device, non_blocking=True)
            cat = cat.to(device, non_blocking=False)
            Y = Y.to(device, non_blocking=True)
            mT = mT.to(device, non_blocking=True)
            mH = mH.to(device, non_blocking=True)
            mE = mE.to(device, non_blocking=True)
            
            # Forward pass
            with torch.cuda.amp.autocast():
                P = model(num, boo, cat)
            
            # Calculate masked metrics
            m = mH * mE.unsqueeze(-1)
            err = P - Y
            test_mse += (err.pow(2) * m).sum().item()
            test_mae += (err.abs() * m).sum().item()
            test_n += m.sum().item()
    
    final_mse = test_mse / test_n
    final_mae = test_mae / test_n
    
    print(f"\n🏆 FINAL TEST RESULTS:")
    print(f"   Test MSE: {final_mse:.4f}")
    print(f"   Test MAE: {final_mae:.4f}")
    
    # ------------- 7. Save Final Model & Results -------------
    print("\n" + "=" * 60)
    print("💾 SAVING FINAL MODEL & RESULTS")
    print("=" * 60)
    
    # Ensure results directory exists
    results_dir = Path("ml_pipeline/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save final model checkpoint
    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": best_config,
        "final_test_mse": final_mse,
        "final_test_mae": final_mae,
        "training_history": history,
        "model_info": {
            "total_params": total_params,
            "n_num": n_num,
            "n_bool": n_bool,
            "n_cat": n_cat,
            "cat_sizes": cat_sizes
        }
    }
    
    final_model_path = results_dir / "final_model.pt"
    torch.save(final_checkpoint, final_model_path)
    print(f"✅ Final model saved: {final_model_path}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = results_dir / "final_training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"✅ Training history saved: {history_path}")
    
    # Save final results summary
    results_summary = {
        "final_test_mse": final_mse,
        "final_test_mae": final_mae,
        "best_config": best_config,
        "model_params": total_params,
        "training_epochs": best_config["max_epochs"]
    }
    
    summary_path = results_dir / "final_results_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    print(f"✅ Results summary saved: {summary_path}")
    
    print("\n" + "=" * 60)
    print("🎉 FINAL TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📊 Test MSE: {final_mse:.4f} | Test MAE: {final_mae:.4f}")
    print(f"💾 Model ready for production at: {final_model_path}")
    print("=" * 60)
    
    return model, final_mse, final_mae

if __name__ == "__main__":
    model, test_mse, test_mae = main()
    print(f"\n✨ Production model ready with test MSE: {test_mse:.4f}") 