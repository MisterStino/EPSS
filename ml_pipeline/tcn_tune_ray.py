#!/usr/bin/env python
"""
Ray Tune-enabled EPSS TCN Forecaster with Advanced HPO
Comprehensive hyperparameter optimization for the streaming TCN model using Ray Tune
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
from ml_pipeline.training.dataset_iterable_sus import CVEIterableDatasetSUS, collate_sus_train

# NEW: dilated Temporal-Convolutional Network
from pytorch_tcn import TCN

# ───────────────────────────── GLOBAL SUS CONFIGURATION ──────────────────────

# SUS Configuration - Global constants
USE_SUS = True  # Enable significance-based undersampling
SUS_BETA = 0.5  # Default SUS significance threshold (0.1-0.9)
SUS_LOOK_AHEAD = 10

# ───────────────────────────── HELPER FUNCTIONS ──────────────────────

def get_device():
    """Get the best available device without setting global state."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        return device
    else:
        print("WARNING: CUDA not available, falling back to CPU")
        return torch.device("cpu")



def find_project_root():
    """Find project root directory containing ml_pipeline."""
    current = Path.cwd()
    if (current / "ml_pipeline").exists():
        return current
    for parent in current.parents:
        if (parent / "ml_pipeline").exists():
            return parent
    for path in ["/notebooks/EPSS", "/notebooks", Path.home() / "EPSS"]:
        path = Path(path)
        if path.exists() and (path / "ml_pipeline").exists():
            return path
    raise FileNotFoundError("Could not find project root")

def load_vocab_and_paths():
    """Load vocabulary and validate paths."""
    project_root = find_project_root()
    work_dir = project_root / "ml_pipeline" / "work"
    arrow_path = work_dir / "epss_stage1.arrow"
    vocab_path = work_dir / "vocab.json"
    
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    return vocab, arrow_path

def define_spike_threshold(epss_scores, percentile=95.0):
    """
    Define spike threshold based on EPSS score distribution.
    A spike is defined as an EPSS score above the specified percentile.
    Matches LSTM implementation with NaN handling improvement.
    """
    # Remove NaN values (improvement over LSTM version)
    valid_scores = epss_scores[~np.isnan(epss_scores)]
    if len(valid_scores) == 0:
        return 0.7  # Default threshold if no valid scores
    
    threshold = np.percentile(valid_scores, percentile)
    return threshold  # Use actual percentile (consistent with LSTM)

def calculate_spike_recall(y_true, y_pred, spike_threshold):
    """
    Calculate spike recall: proportion of actual spikes correctly predicted as spikes.
    Matches LSTM implementation for consistent EPSS forecasting evaluation.
    """
    # Remove NaN values (improvement over LSTM version)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return 0.0
        
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Identify actual spikes
    actual_spikes = y_true_clean >= spike_threshold
    predicted_spikes = y_pred_clean >= spike_threshold
    
    if np.sum(actual_spikes) == 0:
        return 0.0  # No actual spikes to recall (consistent with LSTM)
    
    # True positives: correctly predicted spikes
    true_positives = np.sum(actual_spikes & predicted_spikes)
    
    # Spike recall = TP / (TP + FN)
    spike_recall = true_positives / np.sum(actual_spikes)
    return float(spike_recall)

# ───────────────────────────── TCN MODEL DEFINITION ──────────────────────

class Seq2SeqTCN(nn.Module):
    """
    Sequence-to-sequence TCN for EPSS forecasting.
    Handles mixed input types: numeric, boolean, and categorical.
    """
    
    def __init__(self, vocab, num_features, bool_features, cat_features, 
                 nb_filters=128, levels=6, kernel_size=3, dropout=0.2, emb_dim=8, horizon=30):
        super().__init__()
        
        self.vocab = vocab
        self.num_features = num_features
        self.bool_features = bool_features
        self.cat_features = cat_features
        self.nb_filters = nb_filters
        self.levels = levels
        self.kernel_size = kernel_size
        self.horizon = horizon
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict()
        for col in cat_features:
            vocab_size = len(vocab[col])
            self.embeddings[col] = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        
        # Calculate input dimension
        input_dim = num_features + bool_features + len(cat_features) * emb_dim
        
        # TCN layers
        self.tcn = TCN(
            num_inputs=input_dim,
            num_channels=[nb_filters] * levels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_norm='weight_norm',
        )
        
        # Output projection
        self.output_proj = nn.Linear(nb_filters, horizon)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, num, boo, cat):
        batch_size, seq_len = num.shape[:2]
        
        # Process embeddings
        embedded_cats = []
        for i, col in enumerate(self.cat_features):
            # Clamp categorical values to valid vocabulary range
            cat_clamped = torch.clamp(cat[:, :, i], 0, len(self.vocab[col]) - 1)
            emb = self.embeddings[col](cat_clamped)
            embedded_cats.append(emb)
        
        # Concatenate all features
        if embedded_cats:
            cat_combined = torch.cat(embedded_cats, dim=-1)
            combined = torch.cat([num, boo.float(), cat_combined], dim=-1)
        else:
            combined = torch.cat([num, boo.float()], dim=-1)
        
        # TCN expects [B, F, L] format
        combined = combined.transpose(1, 2)  # [B, L, F] -> [B, F, L]
        
        # TCN forward pass
        tcn_out = self.tcn(combined)  # [B, nb_filters, L]
        tcn_out = tcn_out.transpose(1, 2)  # [B, F, L] -> [B, L, F]
        tcn_out = self.dropout(tcn_out)
        
        # Project to horizon dimension
        output = self.output_proj(tcn_out)
        
        return output

# ───────────────────────────── LOSS FUNCTION ──────────────────────

def focal_epss(pred, target, horizon, alpha0=2.0, gamma=2.5, lam=0.15):
    """
    Focal EPSS loss function for horizon-sensitive forecasting.
    
    Args:
        pred: predictions tensor of shape (B,) or (B, H)
        target: target tensor of shape (B,) or (B, H)
        horizon: days ahead forecast horizon (int or float per sample)
        alpha0: base multiplier
        gamma: focusing exponent
        lam: horizon-sensitivity factor
    """
    e = (pred - target).abs()
    alpha_h = alpha0 * (1 + lam * horizon)   # emphasise early jumps at longer lead time
    mod = (1 + alpha_h * e) ** gamma
    return (mod * e**2).mean()               # Focal-MSE core 

def focal_epss_masked(pred, target, m_t, m_eval, m_h, horizon):
    """
    Compute focal EPSS loss only where all masks permit.
    Combines the focal EPSS loss with masking functionality.
    """
    # Combine all masks
    combined_mask = m_t.unsqueeze(-1) * m_eval.unsqueeze(-1) * m_h.unsqueeze(0).unsqueeze(0)
    
    # Apply mask
    masked_pred = pred * combined_mask
    masked_target = target * combined_mask
    
    # Only compute loss on valid positions
    valid_positions = combined_mask.sum()
    if valid_positions > 0:
        # Flatten for focal_epss computation
        masked_pred_flat = masked_pred[combined_mask > 0]
        masked_target_flat = masked_target[combined_mask > 0]
        
        # Apply focal EPSS loss
        loss = focal_epss(masked_pred_flat, masked_target_flat, horizon)
    else:
        loss = torch.tensor(0.0, device=pred.device)
    
    return loss

# ───────────────────────────── MAIN TRAINING FUNCTION ──────────────────────

def train_tcn_with_tune(config):
    """
    Train TCN with hyperparameters from Ray Tune.
    Includes spike-aware metrics and early reporting.
    """
    
    # Extract hyperparameters - TCN specific
    nb_filters = config["nb_filters"]
    levels = config["levels"]
    kernel_size = config["kernel_size"]
    dropout = config["dropout"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    
    # SUS parameters (from config if optimizing, otherwise use defaults)
    trial_sus_beta = config.get("sus_beta", SUS_BETA)
    trial_sus_look_ahead = config.get("sus_look_ahead", SUS_LOOK_AHEAD)
    
    # Fixed max epochs - ASHA scheduler will handle early stopping
    max_epochs = 50  # High limit, scheduler will stop early if not improving
    
    # Constants
    HORIZON = 30
    REPORT_EVERY_N_BATCHES = 100  # Early metric reporting
    
    # Setup device and paths
    device = get_device()
    
    # Setup CUDA optimizations with local imports (avoids cloudpickle serialization issues)
    if device.type == "cuda":
        # Local imports - not visible as globals to cloudpickle
        import torch.backends.cudnn as _cudnn
        import torch.backends.cuda.matmul as _matmul
        _cudnn.benchmark = True
        _cudnn.allow_tf32 = True
        _matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    
    vocab, arrow_path = load_vocab_and_paths()
    
    # Load SUS configuration if enabled
    sus_z_norm = None
    use_sus_for_trial = USE_SUS
    
    if use_sus_for_trial:
        project_root = find_project_root()
        sus_config_path = project_root / "ml_pipeline" / "work" / f"sus_config_beta{trial_sus_beta}_d{trial_sus_look_ahead}_q0.995.json"
        
        if sus_config_path.exists():
            with open(sus_config_path, 'r') as f:
                sus_data = json.load(f)
                sus_z_norm = sus_data['Z']
                print(f"[Ray Worker] SUS enabled: β={trial_sus_beta}, Δ={trial_sus_look_ahead}, Z={sus_z_norm:.6f}")
        else:
            print(f"[Ray Worker] WARNING: SUS config not found: {sus_config_path}")
            print(f"[Ray Worker] Run: python -m ml_pipeline.tools.compute_weight_quantile --arrow {arrow_path} --beta {trial_sus_beta} --look-ahead {trial_sus_look_ahead}")
            use_sus_for_trial = False
            print(f"[Ray Worker] Falling back to standard dataset")
    else:
        print(f"[Ray Worker] Using standard dataset (SUS disabled)")
    
    # Debug GPU availability in Ray worker
    print(f"[Ray Worker] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Ray Worker] CUDA available: {torch.cuda.is_available()}")
        print(f"[Ray Worker] CUDA device count: {torch.cuda.device_count()}")
        print(f"[Ray Worker] Current CUDA device: {torch.cuda.current_device()}")
        print(f"[Ray Worker] CUDA device name: {torch.cuda.get_device_name()}")
        
        # Memory monitoring
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"[Ray Worker] GPU total memory: {total_memory:.2f} GB")
        torch.cuda.empty_cache()  # Clear cache before starting
    else:
        print("[Ray Worker] WARNING: CUDA not available - running on CPU only!")
    
    try:
        # Create datasets - use SUS if enabled, otherwise standard
        if use_sus_for_trial and sus_z_norm is not None:
            print(f"[Ray Worker] Creating SUS datasets...")
            train_dataset = CVEIterableDatasetSUS(arrow_path, horizon=HORIZON)
            val_dataset = CVEIterableDatasetSUS(arrow_path, horizon=HORIZON)
            test_dataset = CVEIterableDatasetSUS(arrow_path, horizon=HORIZON)
            
            # Create SUS dataloaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                collate_fn=partial(collate_sus_train, 
                                 flag_kind="train", 
                                 horizon=HORIZON,
                                 beta=trial_sus_beta,
                                 look_ahead=trial_sus_look_ahead,
                                 z_norm=sus_z_norm),
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                collate_fn=partial(collate_sus_train, 
                                 flag_kind="val", 
                                 horizon=HORIZON,
                                 beta=trial_sus_beta,
                                 look_ahead=trial_sus_look_ahead,
                                 z_norm=sus_z_norm),
                num_workers=0
            )
        else:
            print(f"[Ray Worker] Creating standard datasets...")
            train_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
            val_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
            test_dataset = CVEIterableDatasetFixed(arrow_path, horizon=HORIZON)
            
            # Create standard dataloaders
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
        print(f"[Ray Worker] Numeric tensor range: [{num_tensor.min():.6f}, {num_tensor.max():.6f}]")
        print(f"[Ray Worker] Target tensor range: [{target_tensor.min():.6f}, {target_tensor.max():.6f}]")
        print(f"[Ray Worker] Boolean tensor range: [{boo_tensor.min():.6f}, {boo_tensor.max():.6f}]")
        print(f"[Ray Worker] Categorical tensor range: [{cat_tensor.min()}, {cat_tensor.max()}]")
        
        # Check for any initial NaN/Inf in the data
        if torch.isnan(num_tensor).any():
            print("[Ray Worker] WARNING: NaN found in numeric features!")
        if torch.isnan(target_tensor).any():
            print("[Ray Worker] WARNING: NaN found in targets!")
        if torch.isinf(num_tensor).any():
            print("[Ray Worker] WARNING: Inf found in numeric features!")
        if torch.isinf(target_tensor).any():
            print("[Ray Worker] WARNING: Inf found in targets!")
        
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
        
        # Initialize model - TCN specific parameters
        model = Seq2SeqTCN(
            vocab=vocab,
            num_features=num_features,
            bool_features=bool_features,
            cat_features=cat_features_list,
            nb_filters=nb_filters,
            levels=levels,
            kernel_size=kernel_size,
            dropout=dropout,
            horizon=HORIZON
        ).to(device)
        
        # Initialize weights properly to prevent NaN/Inf
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
        
        model.apply(init_weights)
        
        # Optimizer and scheduler with improved stability
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=0.01,
            eps=1e-8,  # Numerical stability
            betas=(0.9, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Training loop
        model.train()
        global_step = 0
        
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            epoch_spike_recall = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = batch
                
                # Move to device
                num_tensor = num_tensor.to(device, non_blocking=True)
                boo_tensor = boo_tensor.to(device, non_blocking=True)
                cat_tensor = cat_tensor.to(device, non_blocking=False)  # Synchronous for index tensor
                target_tensor = target_tensor.to(device, non_blocking=True)
                m_t = m_t.to(device, non_blocking=True)
                m_eval = m_eval.to(device, non_blocking=True)
                m_h = m_h.to(device, non_blocking=True)
                
                # Input validation - check for NaN/Inf
                if torch.isnan(num_tensor).any() or torch.isinf(num_tensor).any():
                    print(f"[Ray Worker] NaN or Inf found in num_tensor")
                    continue
                if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
                    print(f"[Ray Worker] NaN or Inf found in target_tensor")
                    continue
                
                # Forward pass with mixed precision and OOM handling
                optimizer.zero_grad()
                try:
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            pred = model(num_tensor, boo_tensor, cat_tensor)
                            loss = focal_epss_masked(pred, target_tensor, m_t, m_eval, m_h, HORIZON)
                    else:
                        pred = model(num_tensor, boo_tensor, cat_tensor)
                        loss = focal_epss_masked(pred, target_tensor, m_t, m_eval, m_h, HORIZON)
                        
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[Ray Worker] [OOM] Forward pass failed: {e}")
                    torch.cuda.empty_cache()
                    continue
                
                # Check predictions and loss for NaN/Inf
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"[Ray Worker] NaN or Inf found in predictions")
                    continue
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Ray Worker] NaN or Inf found in loss: {loss.item()}")
                    continue
                
                # Calculate spike recall for this batch
                with torch.no_grad():
                    pred_flat = pred.cpu().numpy().flatten()
                    target_flat = target_tensor.cpu().numpy().flatten()
                    batch_spike_recall = calculate_spike_recall(target_flat, pred_flat, spike_threshold)
                
                # Backward pass with mixed precision and OOM handling
                try:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        # Check gradients for NaN/Inf before clipping
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"[Ray Worker] NaN or Inf found in gradients, skipping update")
                            continue
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            print(f"[Ray Worker] NaN or Inf found in gradients, skipping update")
                            continue
                        optimizer.step()
                        
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[Ray Worker] [OOM] Backward pass failed: {e}")
                    torch.cuda.empty_cache()
                    continue
                
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
                    num_tensor = num_tensor.to(device, non_blocking=True)
                    boo_tensor = boo_tensor.to(device, non_blocking=True)
                    cat_tensor = cat_tensor.to(device, non_blocking=False)
                    target_tensor = target_tensor.to(device, non_blocking=True)
                    m_t = m_t.to(device, non_blocking=True)
                    m_eval = m_eval.to(device, non_blocking=True)
                    m_h = m_h.to(device, non_blocking=True)
                    
                    # Input validation
                    if torch.isnan(num_tensor).any() or torch.isinf(num_tensor).any():
                        continue
                    if torch.isnan(target_tensor).any() or torch.isinf(target_tensor).any():
                        continue
                    
                    # Forward pass
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            pred = model(num_tensor, boo_tensor, cat_tensor)
                            loss = focal_epss_masked(pred, target_tensor, m_t, m_eval, m_h, HORIZON)
                    else:
                        pred = model(num_tensor, boo_tensor, cat_tensor)
                        loss = focal_epss_masked(pred, target_tensor, m_t, m_eval, m_h, HORIZON)
                    
                    # Check predictions and loss
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        continue
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
        
        print(f"[Ray Worker] Training completed successfully")
        
    except Exception as e:
        print(f"[Ray Worker] Training failed with error: {e}")
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
    
    # Define search space optimized for TCN architecture
    search_space = {
        # TCN architecture - memory-tested configurations
        "nb_filters": tune.choice([64, 128, 256]),  # Number of filters per TCN level
        "levels": tune.choice([4, 6, 8, 10]),  # Number of dilated convolutional levels
        "kernel_size": tune.choice([3, 5, 7]),  # Kernel size for convolutions
        "dropout": tune.uniform(0.05, 0.3),
        
        # Training hyperparameters - memory-safe batch sizes
        "lr": tune.qloguniform(1e-5, 5e-3, q=1e-6),  # More conservative max LR
        "batch_size": tune.choice([128, 256, 512]),  # Memory-tested batch sizes
        
        # SUS hyperparameters (if SUS is enabled)
        "sus_beta": tune.uniform(0.3, 0.7) if USE_SUS else SUS_BETA,  # SUS significance threshold
        "sus_look_ahead": tune.choice([5, 10, 15, 20]) if USE_SUS else SUS_LOOK_AHEAD,  # SUS window
    }
    
    # Setup Bayesian optimization with OptunaSearch
    search_algorithm = OptunaSearch(
        metric=["val_loss", "val_spike_recall"],
        mode=["min", "max"],  # Multi-objective: minimize loss, maximize spike recall
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="epss_tcn_hpo"
    )
    
    # Setup ASHA scheduler for aggressive early stopping
    scheduler = ASHAScheduler(
        time_attr="training_iteration",  # Use epochs instead of global steps
        metric="val_loss",
        mode="min",
        max_t=50,  # Maximum epochs (matches max_epochs in training)
        grace_period=3,  # Minimum epochs before stopping (allow 3 epochs to see progress)
        reduction_factor=2  # Stop bottom 50% of trials at each rung
    )
    
    # Setup reporter
    reporter = CLIReporter(
        parameter_columns={
            "nb_filters": "filters",
            "levels": "levels",
            "kernel_size": "kernel", 
            "dropout": "dropout",
            "lr": "lr",
            "batch_size": "batch"
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
            train_tcn_with_tune,
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
            name="epss_tcn_comprehensive_hpo",
            progress_reporter=reporter,
            stop={"training_iteration": 50},  # Stop condition (matches max_epochs)
            failure_config=tune.FailureConfig(max_failures=3),
            storage_path=str(Path("./ray_results").absolute()),  # Use absolute path
            log_to_file=True
        )
    )
    
    print("Starting comprehensive TCN hyperparameter optimization...")
    print("Search space:")
    for key, value in search_space.items():
        print(f"  {key}: {value}")
    print(f"Number of trials: 50")
    print(f"Max concurrent trials: 2")
    print(f"Max epochs per trial: 50 (ASHA scheduler will stop early)")
    print("Optimization objectives: minimize val_loss, maximize val_spike_recall")
    print("Early stopping: ASHA scheduler (grace period: 3 epochs)")
    print(f"Training strategy: {'SUS (Significance-based Undersampling)' if USE_SUS else 'Standard'}")
    if USE_SUS:
        print(f"SUS optimization: β ∈ [0.3, 0.7], Δ ∈ [5, 10, 15, 20]")
    print("=" * 80)
    
    # Run optimization
    results = tuner.fit()
    
    # Analyze results
    print("\n" + "=" * 80)
    print("TCN HYPERPARAMETER OPTIMIZATION COMPLETED")
    print("=" * 80)
    
    # Get best results for each objective
    best_loss_result = results.get_best_result(metric="val_loss", mode="min")
    best_recall_result = results.get_best_result(metric="val_spike_recall", mode="max")
    
    print(f"\nBest Validation Loss: {best_loss_result.metrics['val_loss']:.4f}")
    print("Best loss configuration:")
    for key, value in best_loss_result.config.items():
        print(f"  {key}: {value}")
    
    print(f"\nBest Spike Recall: {best_recall_result.metrics['val_spike_recall']:.4f}")
    print("Best recall configuration:")
    for key, value in best_recall_result.config.items():
        print(f"  {key}: {value}")
    
    # Save best configurations
    best_configs = {
        "best_loss": {
            "config": best_loss_result.config,
            "metrics": best_loss_result.metrics
        },
        "best_recall": {
            "config": best_recall_result.config,
            "metrics": best_recall_result.metrics
        }
    }
    
    # Ensure results directory exists
    results_dir = Path("ml_pipeline/results")
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "best_tcn_configs.json", 'w') as f:
        json.dump(best_configs, f, indent=2)
    
    print(f"\nBest configurations saved to: ml_pipeline/results/best_tcn_configs.json")
    print("=" * 80)

if __name__ == "__main__":
    main() 