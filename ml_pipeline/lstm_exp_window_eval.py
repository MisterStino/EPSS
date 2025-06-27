#!/usr/bin/env python
# lstm_epss_fullsequence_cuda_v3.py
"""
Feature-rich, leakage-proof EPSS forecaster - STREAMING MEMORY-EFFICIENT VERSION

• streams from preprocessed Arrow file (work/epss_stage1.arrow)
• uses per-batch padding instead of global padding (10× memory reduction)
• maintains identical mathematical behavior with original approach
• trains a mixed-type Seq-to-Seq LSTM that predicts the next 30-day EPSS path
"""

# ───────────────────────────── CELL 1: IMPORTS & SETUP ──────────────────────
# Configure CuDNN workspace limits BEFORE importing torch for safety
import os
os.environ["CUDNN_WORKSPACE_LIMIT_IN_MB"] = "4096"  # Cap scratch at 4 GB
# Configure PyTorch memory allocator to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"

import json, numpy as np, torch, torch.nn as nn
torch.backends.cudnn.benchmark = False              # Obey the workspace cap

# Enable TF32 for better performance on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader  # Removed: Dataset (old approach)
from tqdm import tqdm
from functools import partial
from pathlib import Path
import pytorch_lightning as pl
from torch import amp
import sys

# Detect execution context and adjust paths accordingly
is_notebook_execution = os.path.basename(os.getcwd()) != "ml-pipeline"
if is_notebook_execution:
    # Running as notebook - add relative path to training module
    sys.path.append('./training')
else:
    # Running as module - direct path to training
    sys.path.append('training')

from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed

# Define if local or paperspace:
local_execution  = False

# Hardware-specific configurations
LOCAL_CONFIG = {
    'batch_size': 512,      # 4GB GPU limit
    'hidden_size': 256,    # Reduced model capacity
    'lstm_layers': 2,      # Keep same depth
    'emb_dim': 8,          # Keep same embedding size
    'num_workers': 0,      # Windows multiprocessing fix
    'prefetch_factor': 1,  # Reduced queue depth for memory efficiency
}

CLOUD_CONFIG = {
    'batch_size': 512,     # 90GB GPU capacity
    'hidden_size': 896,    # Adjusted model capacity for optimized performance
    'lstm_layers': 3,      # Same depth
    'emb_dim': 8,          # Same embedding size
    'num_workers': 16,     # Increased for better GPU saturation (was 10)
    'prefetch_factor': 4,  # Increased for better pipeline efficiency (was 2)
}

# Select configuration based on execution environment
CONFIG = LOCAL_CONFIG if local_execution else CLOUD_CONFIG
print(f"[INFO] Using {'LOCAL' if local_execution else 'CLOUD'} configuration:")
print(f"[INFO] Batch: {CONFIG['batch_size']}, Hidden: {CONFIG['hidden_size']}")

# === put right below your CONFIG block ===
MAX_BATCH   = 768           # Optimal for persistent RNN kernels (H×B ≤ 8192×1536)
HIDDEN_SIZE = 768           # Optimal for persistent RNN kernels
LR          = 1.6e-3 * (MAX_BATCH/3072)  # = 0.0004 (scaled for smaller batch)
CONFIG['batch_size']  = MAX_BATCH
CONFIG['hidden_size'] = HIDDEN_SIZE
BATCH = MAX_BATCH

# ──────────────────────────── helpers ───────────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        # Note: benchmark=False is set globally for workspace cap compliance
        
        # Tensor-Core optimization for better performance on modern NVIDIA GPUs
        torch.set_float32_matmul_precision("high")
        
        print("[INFO] GPU:", torch.cuda.get_device_name(0))
        print("[INFO] Tensor-Core optimization enabled")
        print("[INFO] CuDNN workspace capped at 4GB for OOM prevention")
        return torch.device("cuda")
    print("[WARN] CUDA unavailable → CPU")
    return torch.device("cpu")



WORK_DIR = Path("work") if not is_notebook_execution else Path("./work")
ARROW_PATH = WORK_DIR / "epss_stage1.arrow"
VOCAB_PATH = WORK_DIR / "vocab.json"

print(f"[INFO] Work directory: {WORK_DIR.resolve()}")
print(f"[INFO] Looking for artifacts in: {ARROW_PATH.parent.resolve()}") 

if not ARROW_PATH.exists():
    raise FileNotFoundError(f"Arrow file not found: {ARROW_PATH}. Run: python -m models.models.data_prep.00_build_arrow")
if not VOCAB_PATH.exists():
    raise FileNotFoundError(f"Vocab file not found: {VOCAB_PATH}. Run: python -m models.models.data_prep.00_build_arrow")

# Load vocabulary for model initialization (built from training data only)
with open(VOCAB_PATH, 'r') as f:
    VOCAB = json.load(f)

print(f"[INFO] Streaming from: {ARROW_PATH}")
print(f"[INFO] Vocabulary loaded: {len(VOCAB)} categorical columns")
print(f"[INFO] Memory-efficient streaming approach - no DataFrame loading!")


# ───────────────────────────── 6  model ──────────────────────────────────────
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

    def forward(self, num, boo, cat, lens=None):
        # Handle case where no categorical embeddings exist
        if len(self.emb):                           # tiny micro-perf
            cat = cat.long()                        # ⚠ NEW: ensure correct dtype
            # Safety check: clamp any out-of-bounds indices to UNK (0)
            cat_sizes = [emb.num_embeddings for emb in self.emb]
            for i, vocab_size in enumerate(cat_sizes):
                # Clamp out-of-bounds indices to 0 (UNK)
                out_of_bounds = cat[..., i] >= vocab_size
                if out_of_bounds.any():
                    print(f"WARNING: Found {out_of_bounds.sum()} out-of-bounds indices in column {i}, clamping to UNK")
                    cat[..., i] = torch.clamp(cat[..., i], 0, vocab_size - 1)
            e = torch.cat([emb(cat[..., i]) for i, emb in enumerate(self.emb)], dim=-1)
            x = torch.cat([num, boo.float(), e], dim=-1)
        else:
            x = torch.cat([num, boo.float()], dim=-1)
        
        # Use packed sequences for better performance if lengths provided
        if lens is not None:
            # Pack sequences to skip computation on padding tokens
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lens.cpu(), batch_first=True, enforce_sorted=False
            )
            h_packed, _ = self.lstm(packed)
            # Unpack back to original padded format
            h, _ = nn.utils.rnn.pad_packed_sequence(
                h_packed, batch_first=True, total_length=lens.max()
            )
        else:
            # Fallback to standard LSTM (for backward compatibility)
            h, _ = self.lstm(x)
            
        return self.head(h)

# Lightning wrapper that can recreate its own DataLoader for batch size tuning
class LightningWrapper(pl.LightningModule):
    def __init__(self, model, dataset, collate_fn, num_workers, batch_size: int = 1024):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self._collate = collate_fn
        self._num_workers = num_workers
        self.save_hyperparameters({"batch_size": batch_size})
    
    def forward(self, num, boo, cat):
        return self.model(num, boo, cat)
    
    def training_step(self, batch, batch_idx):
        num, boo, cat, Y, mt, mh, me, date_pad = batch
        loss = masked_mse(self.model(num, boo, cat), Y, mt, mh, me)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    # ← CRUCIAL: Lightning calls this to get fresh dataloaders during batch size probing
    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,  # Lightning modifies this during tuning
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self._num_workers,
            pin_memory=self._num_workers > 0,  # Only use pin_memory with multiprocessing
            persistent_workers=self._num_workers > 0,
        )

# ───────────────────────────── 7  masked loss ───────────────────────────────
def masked_mse(pred, true, m_t, m_h, m_eval):
    mask = m_t * m_eval                        # [B,L]
    err  = (pred - true) ** 2                  # [B,L,H]
    return (err * mask.unsqueeze(-1) * m_h).sum() / m_h.sum()

# %%
# ───────────────────────────── CELL 6: MODEL TRAINING & EVALUATION ──────────
# Main training routine - optimized for notebook execution
import time

print("🚀 STARTING MODEL TRAINING PIPELINE")
print("=" * 50)

torch.manual_seed(0)
dev = get_device()

HORIZON, EPOCHS = 30, 12

# ──────────────────────── STEP 1: Streaming Dataset Creation ─────────────────────────
print(f"\n[STEP 1/6] Creating streaming datasets (memory-efficient)...")
start_time = time.time()

# NEW: Create streaming datasets - no global padding, no DataFrame in memory
print("  → Training dataset (streaming)...")
tr_ds = CVEIterableDatasetFixed(ARROW_PATH, horizon=HORIZON)

print("  → Validation dataset (streaming)...")  
va_ds = CVEIterableDatasetFixed(ARROW_PATH, horizon=HORIZON)

print("  → Test dataset (streaming)...")
te_ds = CVEIterableDatasetFixed(ARROW_PATH, horizon=HORIZON)

elapsed = time.time() - start_time
print(f"✓ All streaming datasets created in {elapsed:.1f}s")

# ──────────────────────── STEP 2: DataLoader Creation (Per-Batch Padding) ─────────────────
print(f"\n[STEP 2/6] Creating data loaders with per-batch padding...")
start_time = time.time()

# NEW: Use per-batch padding collate function instead of global padding
# Note: IterableDataset doesn't support shuffle - randomness handled by worker sharding
tr_ld = DataLoader(tr_ds, BATCH, shuffle=False,
                   collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=HORIZON),
                   num_workers=CONFIG['num_workers'], 
                   pin_memory=CONFIG['num_workers'] > 0,  # Only use pin_memory with multiprocessing
                   persistent_workers=CONFIG['num_workers'] > 0,
                   prefetch_factor=CONFIG['prefetch_factor'])

va_ld = DataLoader(va_ds, BATCH, shuffle=False,
                   collate_fn=partial(pad_and_mask_fixed, flag_kind="val", horizon=HORIZON),
                   num_workers=CONFIG['num_workers'], 
                   pin_memory=CONFIG['num_workers'] > 0,  # Only use pin_memory with multiprocessing
                   persistent_workers=CONFIG['num_workers'] > 0,
                   prefetch_factor=CONFIG['prefetch_factor'])

te_ld = DataLoader(te_ds, BATCH, shuffle=False,
                   collate_fn=partial(pad_and_mask_fixed, flag_kind="test", horizon=HORIZON),
                   num_workers=CONFIG['num_workers'], 
                   pin_memory=CONFIG['num_workers'] > 0,  # Only use pin_memory with multiprocessing
                   persistent_workers=CONFIG['num_workers'] > 0,
                   prefetch_factor=CONFIG['prefetch_factor'])

elapsed = time.time() - start_time
print(f"✓ Memory-efficient data loaders ready in {elapsed:.1f}s")
print(f"✓ Per-batch padding (not global) - massive memory savings!")

# ──────────────────────── STEP 3: Dynamic Model Dimension Detection ─────────────────
print(f"\n[STEP 3/6] Detecting model dimensions from streaming data...")
start_time = time.time()

# NEW: Get dimensions from actual streaming data sample (not DataFrame)
print("  → Sampling batch to determine feature dimensions...")
sample_batch = next(iter(tr_ld))
n_num = sample_batch[0].shape[-1]   # Numeric features dimension 
n_bool = sample_batch[1].shape[-1]  # Boolean features dimension
n_cat = sample_batch[2].shape[-1]   # Categorical features dimension
# Note: sample_batch now has 9 elements (added lengths), but we only need first 3 for dimensions

# Get categorical vocabulary sizes for embeddings
cat_sizes = [len(VOCAB[col]) for col in VOCAB.keys()]

print(f"  → Detected dimensions: {n_num} numeric, {n_bool} boolean, {n_cat} categorical")
print(f"  → Categorical vocab sizes: {cat_sizes}")

elapsed = time.time() - start_time
print(f"✓ Dynamic dimensions detected in {elapsed:.1f}s")

# ──────────────────────── STEP 4: Model Setup ───────────────────────────────
print(f"\n[STEP 4/6] Setting up model with dynamic dimensions...")
start_time = time.time()

model = Seq2SeqLSTM(
            n_num     = n_num,        # From streaming data sample
            n_bool    = n_bool,       # From streaming data sample  
            cat_sizes = cat_sizes,    # From loaded vocabulary
            horizon   = HORIZON,
            hidden    = CONFIG['hidden_size'],
            layers    = CONFIG['lstm_layers']).to(dev)

# ──────────────────────── STEP 5: Batch Size Tuning ──────────────────
print(f"\n[STEP 5/6] Setting up optimized batch size and mixed precision...")
tuning_start = time.time()

# COMMENTED OUT: Lightning batch size tuning - using fixed optimized batch size instead
# ❶ Create Lightning wrapper that can rebuild its own dataloader
# lightning_model = LightningWrapper(
#     model,
#     dataset=tr_ds,  # Pass dataset, not pre-built dataloader
#     collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=HORIZON),
#     num_workers=CONFIG['num_workers'],
#     batch_size=CONFIG['batch_size']
# )

# ❂ Create plain trainer (no batch-size flag for Lightning 2.5+)
# trainer = pl.Trainer(
#     max_epochs=EPOCHS,  # Explicitly set to silence warning (real training is manual)
#     limit_train_batches=1,
#     logger=False,  # Disable logging for tuning
#     enable_checkpointing=False,  # Disable checkpointing for tuning
#     enable_progress_bar=False  # Disable progress bar for cleaner output
# )

# ❃ Create tuner and scale batch size with controlled limits
# import math

# MAX_BATCH = 1024          # Upper limit for batch size
# INIT_VAL  = 512           # First value the tuner will try

# tuner = pl.tuner.Tuner(trainer)
# tuner.scale_batch_size(
#     lightning_model,
#     mode="power",              # Doubling strategy (512 → 1024)
#     init_val=INIT_VAL,
#     # Stop after log2(MAX_BATCH / INIT_VAL) successful doublings
#     max_trials=int(math.log2(MAX_BATCH // INIT_VAL)),
# )

# Just in case: never let the tuned value exceed the cap
# lightning_model.hparams.batch_size = min(lightning_model.hparams.batch_size, MAX_BATCH)

# optimal_batch_size = lightning_model.hparams.batch_size
# print(f"✓ Optimal batch size found: {optimal_batch_size} (was {CONFIG['batch_size']})")
# tuning_elapsed = time.time() - tuning_start
# print(f"✓ Batch size tuning completed in {tuning_elapsed:.1f}s")

# CRITICAL: Lightning moves model back to CPU after tuning - move it back to GPU
# model = lightning_model.model  # Extract the tuned model
# model.to(dev)                  # Move back to GPU for manual training
# print(f"✓ Model moved back to GPU after Lightning tuning")

# NEW: Set optimized batch size directly for mixed precision training
optimal_batch_size = MAX_BATCH
print(f"✓ Using optimized batch size: {optimal_batch_size} (was originally {512})")
print("✓ Mixed precision training enabled - will use FP16 for better performance")

# Configuration already updated in CONFIG block above

# Skip torch.compile() to avoid kernel cache OOM with variable-length sequences
# Eager mode is more stable for streaming, per-batch-padded RNNs
if hasattr(torch, "compile") and dev.type == "cuda" and not local_execution:
    print("  → Skipping model compilation (avoiding kernel cache OOM with variable lengths)")
    # model = torch.compile(model)  # DISABLED: causes OOM with variable sequence lengths
elif local_execution:
    print("  → Running in eager mode (local execution)")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
elapsed = time.time() - start_time
print(f"✓ Model ready: {total_params:,} total params, {trainable_params:,} trainable ({elapsed:.1f}s)")

# Recreate data loaders with optimal batch size
print(f"\n[RECREATING DATALOADERS] Using optimal batch size: {BATCH}")
recreate_start = time.time()

tr_ld = DataLoader(tr_ds, BATCH, shuffle=False,
                   collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=HORIZON),
                   num_workers=CONFIG['num_workers'], 
                   pin_memory=CONFIG['num_workers'] > 0,  # Only use pin_memory with multiprocessing
                   persistent_workers=CONFIG['num_workers'] > 0,
                   prefetch_factor=CONFIG['prefetch_factor'])

va_ld = DataLoader(va_ds, BATCH, shuffle=False,
                   collate_fn=partial(pad_and_mask_fixed, flag_kind="val", horizon=HORIZON),
                   num_workers=CONFIG['num_workers'], 
                   pin_memory=CONFIG['num_workers'] > 0,  # Only use pin_memory with multiprocessing
                   persistent_workers=CONFIG['num_workers'] > 0,
                   prefetch_factor=CONFIG['prefetch_factor'])

te_ld = DataLoader(te_ds, BATCH, shuffle=False,
                   collate_fn=partial(pad_and_mask_fixed, flag_kind="test", horizon=HORIZON),
                   num_workers=CONFIG['num_workers'], 
                   pin_memory=CONFIG['num_workers'] > 0,  # Only use pin_memory with multiprocessing
                   persistent_workers=CONFIG['num_workers'] > 0,
                   prefetch_factor=CONFIG['prefetch_factor'])

recreate_elapsed = time.time() - recreate_start
print(f"✓ Data loaders recreated with optimal batch size in {recreate_elapsed:.1f}s")

opt = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)

# ──────────────────────── MIXED PRECISION SETUP ─────────────────────────
# Initialize gradient scaler for mixed precision training
scaler = torch.cuda.amp.GradScaler()
print("✓ Mixed precision gradient scaler initialized")

# ──────────────────────── STEP 6: Training ──────────────────────────────────
print(f"\n[STEP 6/6] Training for {EPOCHS} epochs with optimized performance...")
print("=" * 50)

# Print optimization summary
print("🚀 PERFORMANCE OPTIMIZATIONS ENABLED:")
print("  ✓ Packed sequences (skip padding computation)")
print("  ✓ Persistent RNN kernels (optimal batch×hidden size)")
print("  ✓ Fused AdamW optimizer")
print("  ✓ TF32 + Mixed precision (FP16)")
print("  ✓ Memory fragmentation prevention")
print("  ✓ Increased DataLoader workers/prefetch")
print("=" * 50)

# Initialize training history tracking
history = {"epoch": [], "tr_loss": [], "va_loss": []}
for ep in range(1, EPOCHS + 1):
    model.train(); tr_loss = 0.0; tr_batches = 0
    for num, boo, cat, Y, mt, mh, me, date_pad, lens in tqdm(tr_ld, desc=f"train {ep}/{EPOCHS}"):
        # Move tensors to device - cat synchronously to avoid embedding race conditions
        num = num.to(dev, non_blocking=True)
        boo = boo.to(dev, non_blocking=True)
        cat = cat.to(dev, non_blocking=False)  # Synchronous for index tensor
        Y = Y.to(dev, non_blocking=True)
        mt = mt.to(dev, non_blocking=True)
        mh = mh.to(dev, non_blocking=True)
        me = me.to(dev, non_blocking=True)
        lens = lens.to(dev, non_blocking=True)  # Sequence lengths for packed sequences
        # date_pad stays on CPU - not needed for training
        opt.zero_grad()
        with amp.autocast(device_type="cuda"):
            loss = masked_mse(model(num, boo, cat, lens), Y, mt, mh, me)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)  # Unscale gradients before clipping
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        scaler.step(opt)
        scaler.update()
        tr_loss += loss.item()
        tr_batches += 1
    tr_loss /= tr_batches

    model.eval(); va_loss = 0.0; va_batches = 0
    with torch.no_grad():
        for num, boo, cat, Y, mt, mh, me, date_pad, lens in va_ld:
            # Move tensors to device - cat synchronously to avoid embedding race conditions
            num = num.to(dev, non_blocking=True)
            boo = boo.to(dev, non_blocking=True)
            cat = cat.to(dev, non_blocking=False)  # Synchronous for index tensor
            Y = Y.to(dev, non_blocking=True)
            mt = mt.to(dev, non_blocking=True)
            mh = mh.to(dev, non_blocking=True)
            me = me.to(dev, non_blocking=True)
            lens = lens.to(dev, non_blocking=True)  # Sequence lengths for packed sequences
            # date_pad stays on CPU - not needed for validation
            with amp.autocast(device_type="cuda"):
                va_loss += masked_mse(model(num, boo, cat, lens), Y, mt, mh, me).item()
            va_batches += 1
    va_loss /= va_batches
    print(f"epoch {ep:02d}  train {tr_loss:.4f}  val {va_loss:.4f}")
    
    # Track training history
    history["epoch"].append(ep)
    history["tr_loss"].append(tr_loss)
    history["va_loss"].append(va_loss)

print("\n" + "=" * 50)
print("🎯 FINAL EVALUATION")
print("=" * 50)
model.eval(); tot_mse = tot_mae = tot_n = 0.0
with torch.no_grad():
    for num, boo, cat, Y, mt, mh, me, date_pad, lens in te_ld:
        # Move tensors to device - cat synchronously to avoid embedding race conditions
        num = num.to(dev, non_blocking=True)
        boo = boo.to(dev, non_blocking=True)
        cat = cat.to(dev, non_blocking=False)  # Synchronous for index tensor
        Y = Y.to(dev, non_blocking=True)
        mt = mt.to(dev, non_blocking=True)
        mh = mh.to(dev, non_blocking=True)
        me = me.to(dev, non_blocking=True)
        lens = lens.to(dev, non_blocking=True)  # Sequence lengths for packed sequences
        # date_pad stays on CPU - not needed for evaluation metrics
        with amp.autocast(device_type="cuda"):
            P = model(num, boo, cat, lens)
        m = mh * me.unsqueeze(-1)
        err = P - Y
        tot_mse += (err.pow(2) * m).sum().item()
        tot_mae += (err.abs() * m).sum().item()
        tot_n   += m.sum().item()

print(f"[RESULT] test MSE {tot_mse / tot_n:.4f} | MAE {tot_mae / tot_n:.4f}")

# Save training history and model checkpoint
print("\n" + "=" * 50)
print("💾 SAVING RESULTS")
print("=" * 50)

# CRITICAL FIX: Ensure directories exist before saving
from pathlib import Path
Path("ml_pipeline/results/predictions").mkdir(parents=True, exist_ok=True)
print("✓ Results directories created/verified")

# Save training history to CSV
import pandas as pd
history_df = pd.DataFrame(history)
history_path = "ml_pipeline/results/loss_history.csv"
history_df.to_csv(history_path, index=False)
print(f"✓ Training history saved: {history_path}")

# Save model checkpoint
checkpoint = {
    "cfg": CONFIG,
    "state_dict": model.state_dict(),
    "final_test_mse": tot_mse / tot_n,
    "final_test_mae": tot_mae / tot_n
}
checkpoint_path = "ml_pipeline/results/checkpoint.pt"
torch.save(checkpoint, checkpoint_path)
print(f"✓ Model checkpoint saved: {checkpoint_path}")
print(f"✓ Checkpoint includes: config, weights, final test metrics")

# Save detailed predictions to NetCDF for analysis
print("\n" + "=" * 50)
print("📊 SAVING DETAILED PREDICTIONS")
print("=" * 50)

# Create single-batch test loader for prediction collection (deterministic order)
# Reset CVE list to avoid duplication from previous iterations
te_ds.collected_cve_ids.clear()

test_pred_loader = DataLoader(
    te_ds, batch_size=1, shuffle=False,
    collate_fn=partial(pad_and_mask_fixed, flag_kind="test", horizon=HORIZON),
    num_workers=0,  # Single worker for deterministic CVE order
    prefetch_factor=1  # Consistent with optimized configuration
)

# Collect predictions, ground truth, masks, and metadata
pred_list, true_list, mh_list, me_list, date_list = [], [], [], [], []

print("  → Collecting predictions from test set...")
model.eval()
with torch.no_grad():
    for batch_data in tqdm(test_pred_loader, desc="collect preds"):
        if len(batch_data) == 9:  # New format with lengths
            num, boo, cat, Y, mt, mh, me, date_pad, lens = batch_data
            # Forward pass with packed sequences
            with amp.autocast(device_type="cuda"):
                P = model(num.to(dev), boo.to(dev), cat.to(dev), lens.to(dev)).cpu()
        else:  # Backward compatibility - old format without lengths
            num, boo, cat, Y, mt, mh, me, date_pad = batch_data
            # Forward pass without packed sequences
            with amp.autocast(device_type="cuda"):
                P = model(num.to(dev), boo.to(dev), cat.to(dev)).cpu()
        
        # Store results (remove batch dimension since batch_size=1)
        pred_list.append(P[0])          # [L, H]
        true_list.append(Y[0])          # [L, H] 
        mh_list.append(mh[0])           # [L, H]
        me_list.append(me[0])           # [L]
        date_list.append(date_pad[0])   # [L]

print(f"  → Collected {len(pred_list)} CVE sequences")

# Pad all sequences to same length for rectangular array
L_max = max(t.shape[0] for t in pred_list)
print(f"  → Maximum sequence length: {L_max}")

def right_pad_tensor(tensor, target_length, pad_value):
    """Pad tensor to target length on the first dimension."""
    if tensor.ndim == 1:
        # 1D tensor (eval_mask, dates)
        pad_width = (0, target_length - tensor.shape[0])
    else:
        # 2D tensor (pred, true, mask_h)
        pad_width = (0, 0, 0, target_length - tensor.shape[0])
    return torch.nn.functional.pad(tensor, pad_width, value=pad_value)

# Pad and stack all tensors
P_padded = torch.stack([right_pad_tensor(t, L_max, 0.0) for t in pred_list])    # [N, L_max, H]
T_padded = torch.stack([right_pad_tensor(t, L_max, 0.0) for t in true_list])    # [N, L_max, H]
MH_padded = torch.stack([right_pad_tensor(t, L_max, 0) for t in mh_list])       # [N, L_max, H]
ME_padded = torch.stack([right_pad_tensor(t, L_max, 0) for t in me_list])       # [N, L_max]
DT_padded = torch.stack([right_pad_tensor(t, L_max, 0) for t in date_list])     # [N, L_max]

# Convert date tensors to numpy datetime64
DT_numpy = DT_padded.numpy()
DT_numpy[DT_numpy == 0] = np.datetime64("NaT").view("int64")  # Replace padding with NaT
dates_2d = DT_numpy.view("datetime64[ns]")  # [N, L_max]

# Get CVE IDs (collected during iteration)
cve_ids = te_ds.collected_cve_ids
print(f"  → CVE IDs collected: {len(cve_ids)}")

# Create xarray Dataset
import xarray as xr
import numpy as np

ds = xr.Dataset(
    data_vars={
        "pred": (["cve", "time", "horizon"], P_padded.numpy()),
        "true": (["cve", "time", "horizon"], T_padded.numpy()),
        "mask_h": (["cve", "time", "horizon"], MH_padded.numpy().astype("uint8")),
        "eval_mask": (["cve", "time"], ME_padded.numpy().astype("uint8")),
    },
    coords={
        "cve": ("cve", np.array(cve_ids, dtype=object)),
        "time": (["cve", "time"], dates_2d),
        "horizon": ("horizon", np.arange(HORIZON))
    }
)

# Save to NetCDF with compression
netcdf_path = "ml_pipeline/results/predictions/predictions_stream.nc"
encoding = {var: {"zlib": True, "complevel": 3} for var in ds.data_vars}
ds.to_netcdf(netcdf_path, encoding=encoding)

print(f"✓ Detailed predictions saved: {netcdf_path}")
print(f"✓ Dataset shape: {len(cve_ids)} CVEs × {L_max} timesteps × {HORIZON} horizons")
print(f"✓ Variables: predictions, ground truth, horizon mask, eval mask")
print(f"✓ Coordinates: CVE IDs, calendar dates, forecast horizons")
