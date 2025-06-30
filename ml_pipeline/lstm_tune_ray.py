#!/usr/bin/env python
"""
Ray Tune-enabled EPSS LSTM Forecaster with Advanced HPO
Comprehensive hyperparameter optimization for the streaming LSTM model using Ray Tune
with spike-aware metrics, early reporting, Bayesian optimization, and refined search spaces.
"""

# ───────────────────────────── RAY TUNE IMPORTS ──────────────────────
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
import ray.air
import optuna

# ───────────────────────────── ORIGINAL IMPORTS & SETUP ──────────────────────
import json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from pathlib import Path
import pytorch_lightning as pl
import sys
import os
import time
import math

# FIXED: Always use module imports regardless of execution context
sys.path.append('training')
from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed

# ───────────────────────────── HELPER FUNCTIONS ──────────────────────

def get_device():
    """Get the best available device and enable optimizations."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        return device
    else:
        print("WARNING: CUDA not available, falling back to CPU")
        return torch.device("cpu")

def find_project_root():
    """Find the project root directory containing ml_pipeline."""
    current = Path.cwd()
    
    # Try current directory first
    if (current / "ml_pipeline").exists():
        return current
    
    # Try parent directories
    for parent in current.parents:
        if (parent / "ml_pipeline").exists():
            return parent
    
    # Try common paths for remote execution
    for path in ["/notebooks/EPSS", "/notebooks", Path.home() / "EPSS"]:
        path = Path(path)
        if path.exists() and (path / "ml_pipeline").exists():
            return path
    
    raise FileNotFoundError("Could not find project root with ml_pipeline directory")

def load_vocab_and_paths():
    """Load vocabulary and verify paths exist."""
    # Find project root dynamically
    project_root = find_project_root()
    
    # Try multiple possible locations
    arrow_candidates = [
        project_root / "ml_pipeline" / "work" / "epss_stage1.arrow",
        project_root / "ml_pipeline" / "data_prep" / "work" / "epss_stage1.arrow",
    ]
    
    vocab_candidates = [
        project_root / "ml_pipeline" / "work" / "vocab.json",
        project_root / "ml_pipeline" / "data_prep" / "work" / "vocab.json",
    ]
    
    # Find Arrow file
    arrow_path = None
    for candidate in arrow_candidates:
        if candidate.exists():
            arrow_path = candidate
            break
    
    if arrow_path is None:
        raise FileNotFoundError(f"Arrow file not found in any of: {arrow_candidates}")
    
    # Find vocab file
    vocab_path = None
    for candidate in vocab_candidates:
        if candidate.exists():
            vocab_path = candidate
            break
    
    if vocab_path is None:
        # Try to generate vocab.json if missing
        print("vocab.json not found. Run: python -m ml_pipeline.generate_missing_vocab")
        raise FileNotFoundError(f"Vocabulary file not found in any of: {vocab_candidates}")
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    print(f"Using project root: {project_root}")
    print(f"Using Arrow file: {arrow_path}")
    print(f"Using vocab file: {vocab_path}")
    
    return vocab, arrow_path

def define_spike_threshold(epss_scores, percentile=95):
    """
    Define spike threshold based on EPSS score distribution.
    A spike is defined as an EPSS score above the 95th percentile.
    """
    return np.percentile(epss_scores, percentile)

def calculate_spike_recall(y_true, y_pred, spike_threshold):
    """
    Calculate spike recall: proportion of actual spikes correctly predicted as spikes.
    """
    # Identify actual spikes
    actual_spikes = y_true >= spike_threshold
    predicted_spikes = y_pred >= spike_threshold
    
    if actual_spikes.sum() == 0:
        return 0.0  # No actual spikes to recall
    
    # True positives: correctly predicted spikes
    true_positives = (actual_spikes & predicted_spikes).sum()
    
    # Spike recall = TP / (TP + FN)
    spike_recall = true_positives / actual_spikes.sum()
    return float(spike_recall)

# ───────────────────────────── MODEL DEFINITION ──────────────────────

class Seq2SeqLSTM(nn.Module):
    """
    Sequence-to-sequence LSTM for EPSS forecasting.
    Handles mixed input types: numeric, boolean, and categorical.
    """
    
    def __init__(self, vocab, num_features, bool_features, cat_features, 
                 hidden_size=768, layers=2, dropout=0.1, emb_dim=64, horizon=30):
        super().__init__()
        
        self.vocab = vocab
        self.num_features = num_features
        self.bool_features = bool_features
        self.cat_features = cat_features
        self.hidden_size = hidden_size
        self.layers = layers
        self.horizon = horizon
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        for col in cat_features:
            vocab_size = len(vocab[col])
            self.embeddings[col] = nn.Embedding(vocab_size, emb_dim)
        
        # Calculate input dimension
        input_dim = num_features + bool_features + len(cat_features) * emb_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, horizon)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, num, boo, cat):
        batch_size, seq_len = num.shape[:2]
        
        # Process embeddings
        embedded_cats = []
        for i, col in enumerate(self.cat_features):
            emb = self.embeddings[col](cat[:, :, i])
            embedded_cats.append(emb)
        
        # Concatenate all features
        if embedded_cats:
            cat_combined = torch.cat(embedded_cats, dim=-1)
            combined = torch.cat([num, boo, cat_combined], dim=-1)
        else:
            combined = torch.cat([num, boo], dim=-1)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(combined)
        lstm_out = self.dropout(lstm_out)
        
        # Project to horizon dimension
        output = self.output_proj(lstm_out)
        
        return output

# ───────────────────────────── LOSS FUNCTION ──────────────────────

def masked_mse(pred, target, m_t, m_eval, m_h):
    """
    Compute MSE loss only where all masks permit.
    """
    # Combine all masks
    combined_mask = m_t.unsqueeze(-1) * m_eval.unsqueeze(-1) * m_h.unsqueeze(0).unsqueeze(0)
    
    # Apply mask and compute MSE
    masked_pred = pred * combined_mask
    masked_target = target * combined_mask
    
    # Compute MSE only on valid positions
    valid_positions = combined_mask.sum()
    if valid_positions > 0:
        loss = ((masked_pred - masked_target) ** 2).sum() / valid_positions
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss

# ───────────────────────────── MAIN TRAINING FUNCTION ──────────────────────

def train_lstm_with_tune(config):
    """
    Train LSTM with hyperparameters from Ray Tune.
    Includes spike-aware metrics and early reporting.
    """
    
    # Extract hyperparameters
    hidden_size = config["hidden_size"]
    layers = config["layers"]
    dropout = config["dropout"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    
    # Constants
    HORIZON = 30
    REPORT_EVERY_N_BATCHES = 100  # Early metric reporting
    
    # Setup device and paths
    device = get_device()
    vocab, arrow_path = load_vocab_and_paths()
    
    # Debug GPU availability in Ray worker
    print(f"[Ray Worker] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Ray Worker] CUDA available: {torch.cuda.is_available()}")
        print(f"[Ray Worker] CUDA device count: {torch.cuda.device_count()}")
        print(f"[Ray Worker] Current CUDA device: {torch.cuda.current_device()}")
        print(f"[Ray Worker] CUDA device name: {torch.cuda.get_device_name()}")
    else:
        print("[Ray Worker] WARNING: CUDA not available - running on CPU only!")
    
    try:
        # Create datasets (CVEIterableDatasetFixed only takes arrow_path and horizon)
        train_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
        val_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
        test_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
        
        # Create dataloaders with proper collate function parameters
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=HORIZON),
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            collate_fn=partial(pad_and_mask_fixed, flag_kind="val", horizon=HORIZON),
            num_workers=0
        )
        
        # Get feature dimensions from first batch
        first_batch = next(iter(train_loader))
        num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = first_batch
        
        num_features = num_tensor.shape[-1]
        bool_features = boo_tensor.shape[-1]
        cat_features_list = list(vocab.keys())
        
        # Debug: Check data ranges
        print(f"Numeric tensor range: [{num_tensor.min():.6f}, {num_tensor.max():.6f}]")
        print(f"Target tensor range: [{target_tensor.min():.6f}, {target_tensor.max():.6f}]")
        print(f"Boolean tensor range: [{boo_tensor.min():.6f}, {boo_tensor.max():.6f}]")
        print(f"Categorical tensor range: [{cat_tensor.min()}, {cat_tensor.max()}]")
        
        # Check for any initial NaN/Inf in the data
        if torch.isnan(num_tensor).any():
            print("WARNING: NaN found in numeric features!")
        if torch.isnan(target_tensor).any():
            print("WARNING: NaN found in targets!")
        if torch.isinf(num_tensor).any():
            print("WARNING: Inf found in numeric features!")
        if torch.isinf(target_tensor).any():
            print("WARNING: Inf found in targets!")
        
        # Define spike threshold from training data
        # Sample some batches to estimate EPSS distribution
        epss_samples = []
        for i, batch in enumerate(train_loader):
            if i >= 10:  # Sample first 10 batches
                break
            _, _, _, targets, _, _, _, _, _ = batch
            epss_samples.append(targets.cpu().numpy())
        
        epss_samples = np.concatenate(epss_samples, axis=0)
        spike_threshold = define_spike_threshold(epss_samples.flatten())
        
        # Initialize model
        model = Seq2SeqLSTM(
            vocab=vocab,
            num_features=num_features,
            bool_features=bool_features,
            cat_features=cat_features_list,
            hidden_size=hidden_size,
            layers=layers,
            dropout=dropout,
            horizon=HORIZON
        ).to(device)
        
        # Initialize weights properly to prevent NaN/Inf
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[(n//4):(n//2)].fill_(1)
        
        model.apply(init_weights)
        
        # Optimizer and scheduler with improved stability
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=0.01,
            eps=1e-8,  # Numerical stability
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        model.train()
        global_step = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_spike_recall = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = batch
                
                # Move to device
                num_tensor = num_tensor.to(device)
                boo_tensor = boo_tensor.to(device)
                cat_tensor = cat_tensor.to(device)
                target_tensor = target_tensor.to(device)
                m_t = m_t.to(device)
                m_eval = m_eval.to(device)
                m_h = m_h.to(device)
                
                # Input validation - check for NaN/Inf
                if torch.isnan(num_tensor).any() or torch.isinf(num_tensor).any():
                    print(f"NaN or Inf found in num_tensor")
                    continue
                if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
                    print(f"NaN or Inf found in target_tensor")
                    continue
                
                # Forward pass
                pred = model(num_tensor, boo_tensor, cat_tensor)
                
                # Check predictions for NaN/Inf
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"NaN or Inf found in predictions")
                    continue
                
                loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
                
                # Check loss for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf found in loss: {loss.item()}")
                    continue
                
                # Calculate spike recall for this batch
                with torch.no_grad():
                    # Flatten predictions and targets for spike calculation
                    pred_flat = pred.cpu().numpy().flatten()
                    target_flat = target_tensor.cpu().numpy().flatten()
                    batch_spike_recall = calculate_spike_recall(target_flat, pred_flat, spike_threshold)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Check gradients for NaN/Inf before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"NaN or Inf found in gradients, skipping update")
                    continue
                
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_spike_recall += batch_spike_recall
                num_batches += 1
                global_step += 1
                
                # Early reporting every N batches
                if global_step % REPORT_EVERY_N_BATCHES == 0 and num_batches > 0:
                    avg_loss = epoch_loss / num_batches
                    avg_spike_recall = epoch_spike_recall / num_batches
                    
                    # Only report if metrics are valid
                    if not (math.isnan(avg_loss) or math.isinf(avg_loss)):
                        session.report({
                            "train_loss": avg_loss,
                            "train_spike_recall": avg_spike_recall,
                            "epoch": epoch + (batch_idx / len(train_loader)),
                            "global_step": global_step
                        })
            
            # Validation at end of epoch
            model.eval()
            val_loss = 0.0
            val_spike_recall = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for val_batch in val_loader:
                    num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = val_batch
                    
                    # Move to device
                    num_tensor = num_tensor.to(device)
                    boo_tensor = boo_tensor.to(device)
                    cat_tensor = cat_tensor.to(device)
                    target_tensor = target_tensor.to(device)
                    m_t = m_t.to(device)
                    m_eval = m_eval.to(device)
                    m_h = m_h.to(device)
                    
                    # Input validation
                    if torch.isnan(num_tensor).any() or torch.isinf(num_tensor).any():
                        continue
                    if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
                        continue
                    
                    # Forward pass
                    pred = model(num_tensor, boo_tensor, cat_tensor)
                    
                    # Check predictions
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        continue
                    
                    loss = masked_mse(pred, target_tensor, m_t, m_eval, m_h)
                    
                    # Check loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    # Calculate spike recall
                    pred_flat = pred.cpu().numpy().flatten()
                    target_flat = target_tensor.cpu().numpy().flatten()
                    batch_spike_recall = calculate_spike_recall(target_flat, pred_flat, spike_threshold)
                    
                    val_loss += loss.item()
                    val_spike_recall += batch_spike_recall
                    val_batches += 1
            
            # Calculate epoch averages
            if num_batches > 0:
                avg_train_loss = epoch_loss / num_batches
                avg_train_spike_recall = epoch_spike_recall / num_batches
            else:
                avg_train_loss = float('inf')
                avg_train_spike_recall = 0.0
                
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                avg_val_spike_recall = val_spike_recall / val_batches
            else:
                avg_val_loss = float('inf')
                avg_val_spike_recall = 0.0
            
            # Step scheduler
            scheduler.step()
            
            # Report to Ray Tune - only if metrics are valid
            if not (math.isnan(avg_train_loss) or math.isinf(avg_train_loss) or 
                    math.isnan(avg_val_loss) or math.isinf(avg_val_loss)):
                session.report({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_spike_recall": avg_train_spike_recall,
                    "val_spike_recall": avg_val_spike_recall,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "lr": scheduler.get_last_lr()[0]
                })
            else:
                # Report failure metrics for invalid results
                session.report({
                    "train_loss": float('inf'),
                    "val_loss": float('inf'),
                    "train_spike_recall": 0.0,
                    "val_spike_recall": 0.0,
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "lr": scheduler.get_last_lr()[0]
                })
            
            model.train()
        
        print(f"Training completed successfully")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Report failure metrics
        session.report({
            "train_loss": float('inf'),
            "val_loss": float('inf'),
            "train_spike_recall": 0.0,
            "val_spike_recall": 0.0,
            "epoch": 0,
            "global_step": 0
        })
        raise e

# ───────────────────────────── MAIN EXECUTION ──────────────────────

def main():
    """
    Main function to run hyperparameter optimization with Ray Tune.
    """
    
    # Suppress Docker CPU detection warnings and deprecation warnings
    import os
    os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
    os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
    
    # Initialize Ray (disable dashboard only on Windows to avoid handle errors)
    import platform
    is_windows = platform.system().lower() == 'windows'
    
    if not ray.is_initialized():
        ray.init(
            include_dashboard=not is_windows,  # Enable dashboard on Linux/Mac
            ignore_reinit_error=True,
            log_to_driver=False,  # Reduce logging overhead
            num_cpus=None,  # Auto-detect
            num_gpus=None   # Auto-detect
        )
    
    # Define refined search space for large GPU memory (48-90GB)
    search_space = {
        # Model architecture - optimized for large memory
        "hidden_size": tune.choice([512, 768, 1024, 1536]),  # Larger sizes for big GPUs
        "layers": tune.choice([2, 3, 4]),  # Include layers for capacity exploration
        "dropout": tune.uniform(0.05, 0.3),
        
        # Training hyperparameters - more conservative ranges for stability
        "lr": tune.qloguniform(1e-5, 5e-3, q=1e-6),  # More conservative max LR
        "batch_size": tune.choice([256, 512, 768, 1024]),  # Larger batches for big GPUs
        "epochs": tune.choice([8, 12, 16, 20])  # Reasonable range for HPO
    }
    
    # Setup Bayesian optimization with OptunaSearch
    search_algorithm = OptunaSearch(
        metric=["val_loss", "val_spike_recall"],
        mode=["min", "max"],  # Multi-objective: minimize loss, maximize spike recall
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="epss_lstm_hpo"
    )
    
    # Setup ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        time_attr="global_step",
        metric="val_loss",
        mode="min",
        max_t=20000,  # Maximum global steps
        grace_period=2000,  # Minimum steps before stopping
        reduction_factor=2
    )
    
    # Setup reporter
    reporter = CLIReporter(
        parameter_columns={
            "hidden_size": "hidden",
            "layers": "layers",
            "dropout": "dropout",
            "lr": "lr",
            "batch_size": "batch",
            "epochs": "epochs"
        },
        metric_columns=[
            "val_loss", 
            "val_spike_recall", 
            "train_loss", 
            "train_spike_recall", 
            "epoch",
            "global_step"
        ],
        max_progress_rows=20,
        max_error_rows=5
    )
    
    # Create and run tuner with explicit GPU allocation
    tuner = tune.Tuner(
        tune.with_resources(
            train_lstm_with_tune,
            resources={"cpu": 1, "gpu": 1}  # Explicitly request 1 GPU per trial
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            search_alg=search_algorithm,
            scheduler=scheduler,
            num_samples=50,  # Number of trials
            max_concurrent_trials=2,  # Limit to available GPUs (2 GPUs = max 2 concurrent)
        ),
        run_config=tune.RunConfig(
            name="epss_lstm_comprehensive_hpo",
            progress_reporter=reporter,
            stop={"global_step": 20000},  # Stop condition
            failure_config=tune.FailureConfig(max_failures=3),
            storage_path=str(Path("./ray_results").absolute()),  # Use absolute path
            log_to_file=True
        )
    )
    
    print("Starting comprehensive hyperparameter optimization...")
    print("Search space:")
    for key, value in search_space.items():
        print(f"  {key}: {value}")
    print(f"Number of trials: 50")
    print(f"Max concurrent trials: 2")
    print("Optimization objectives: minimize val_loss, maximize val_spike_recall")
    print("=" * 80)
    
    # Run optimization
    results = tuner.fit()
    
    # Analyze results
    print("\n" + "=" * 80)
    print("HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("=" * 80)
    
    # Get best results for each objective
    best_loss_result = results.get_best_result(metric="val_loss", mode="min")
    best_spike_result = results.get_best_result(metric="val_spike_recall", mode="max")
    
    print("\nBest configuration for VALIDATION LOSS:")
    print(f"  Config: {best_loss_result.config}")
    print(f"  Val Loss: {best_loss_result.metrics['val_loss']:.6f}")
    print(f"  Val Spike Recall: {best_loss_result.metrics['val_spike_recall']:.4f}")
    
    print("\nBest configuration for SPIKE RECALL:")
    print(f"  Config: {best_spike_result.config}")
    print(f"  Val Loss: {best_spike_result.metrics['val_loss']:.6f}")
    print(f"  Val Spike Recall: {best_spike_result.metrics['val_spike_recall']:.4f}")
    
    # Save best configurations
    results_dir = Path("ml_pipeline/results")
    results_dir.mkdir(exist_ok=True)
    
    best_configs = {
        "best_for_loss": {
            "config": best_loss_result.config,
            "metrics": {
                "val_loss": best_loss_result.metrics["val_loss"],
                "val_spike_recall": best_loss_result.metrics["val_spike_recall"]
            }
        },
        "best_for_spike_recall": {
            "config": best_spike_result.config,
            "metrics": {
                "val_loss": best_spike_result.metrics["val_loss"],
                "val_spike_recall": best_spike_result.metrics["val_spike_recall"]
            }
        }
    }
    
    with open(results_dir / "best_configs.json", "w") as f:
        json.dump(best_configs, f, indent=2)
    
    print(f"\nBest configurations saved to: {results_dir / 'best_configs.json'}")
    print("Hyperparameter optimization completed successfully!")
    
    # Cleanup Ray
    try:
        ray.shutdown()
    except Exception as e:
        print(f"Warning: Ray shutdown error (this is normal): {e}")
    
    return results

if __name__ == "__main__":
    main() 