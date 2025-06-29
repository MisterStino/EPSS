#!/usr/bin/env python
"""
Replay Test Prediction Generator

Reconstructs detailed predictions from saved model checkpoint.
Use this if model training completed but prediction saving failed.

Usage:
    python -m ml_pipeline.results.generate.replay_test_prediction
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from pathlib import Path
import xarray as xr
import sys
import os

# Import dataset classes using full module path
from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed

# ────────────────────────── CONFIGURATION ──────────────────────────
WORK_DIR = Path("ml_pipeline/work")
ARROW_PATH = WORK_DIR / "epss_stage1.arrow"
VOCAB_PATH = WORK_DIR / "vocab.json"
CHECKPOINT_PATH = Path("ml_pipeline/results/checkpoint.pt")
OUTPUT_PATH = Path("ml_pipeline/results/predictions/predictions_stream_replay.nc")

HORIZON = 30  # Fixed horizon for EPSS forecasting

print("🔄 EPSS PREDICTION REPLAY SYSTEM")
print("=" * 50)

# ────────────────────────── VALIDATION ──────────────────────────
print("\n[STEP 1/7] Validating required files...")

required_files = [
    (ARROW_PATH, "Arrow data file"),
    (CHECKPOINT_PATH, "Model checkpoint")
]

for file_path, description in required_files:
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    print(f"✓ Found {description}: {file_path}")

# Ensure output directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"✓ Output directory ready: {OUTPUT_PATH.parent}")

# ────────────────────────── MODEL ARCHITECTURE ──────────────────────────
class Seq2SeqLSTM(nn.Module):
    """Identical model architecture to training script"""
    def __init__(self, n_num: int, n_bool: int, cat_sizes,
                 horizon: int = 30, hidden: int = 512,
                 layers: int = 3, emb_dim: int = 8,
                 dropout: float = 0.3):
        super().__init__()
        # Build embeddings for categorical features
        self.emb = nn.ModuleList([nn.Embedding(s, emb_dim, padding_idx=0) for s in cat_sizes])
        in_dim = n_num + n_bool + emb_dim * len(cat_sizes)

        self.lstm = nn.LSTM(in_dim, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, num, boo, cat):
        if len(self.emb):
            cat = cat.long() 
            # Clamp out-of-bounds indices to UNK (0)
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

# ────────────────────────── CHECKPOINT LOADING ──────────────────────────
print("\n[STEP 2/7] Loading model checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
config = checkpoint['cfg']
state_dict = checkpoint['state_dict']

print(f"✓ Checkpoint loaded from: {CHECKPOINT_PATH}")
print(f"✓ Original config: batch={config.get('batch_size')}, hidden={config.get('hidden_size')}")
print(f"✓ Test metrics: MSE={checkpoint.get('final_test_mse', 'N/A'):.4f}, MAE={checkpoint.get('final_test_mae', 'N/A'):.4f}")

# ────────────────────────── VOCABULARY & DIMENSIONS ──────────────────────────
print("\n[STEP 3/7] Extracting model dimensions from checkpoint...")

# Extract vocabulary sizes directly from checkpoint embedding layers
# This ensures we use the exact same vocab sizes that were used during training
cat_sizes = []
embedding_keys = [k for k in state_dict.keys() if k.startswith('emb.') and k.endswith('.weight')]
embedding_keys.sort(key=lambda x: int(x.split('.')[1]))  # Sort by embedding index

for key in embedding_keys:
    vocab_size = state_dict[key].shape[0]  # First dimension is vocabulary size
    cat_sizes.append(vocab_size)

print(f"✓ Extracted {len(cat_sizes)} categorical vocabularies from checkpoint")
print(f"✓ Categorical vocab sizes from checkpoint: {cat_sizes}")

# Create test dataset to detect numeric/boolean dimensions
test_dataset = CVEIterableDatasetFixed(ARROW_PATH, horizon=HORIZON)
sample_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    collate_fn=partial(pad_and_mask_fixed, flag_kind="test", horizon=HORIZON),
    num_workers=0
)

# Get sample batch for dimension detection
sample_batch = next(iter(sample_loader))
n_num = sample_batch[0].shape[-1]   # Numeric features
n_bool = sample_batch[1].shape[-1]  # Boolean features  
n_cat = sample_batch[2].shape[-1]   # Categorical features

print(f"✓ Detected dimensions: {n_num} numeric, {n_bool} boolean, {n_cat} categorical")

# Validate categorical dimensions match
if n_cat != len(cat_sizes):
    print(f"WARNING: Current data has {n_cat} categorical features, but checkpoint expects {len(cat_sizes)}")
    print("This may cause issues during prediction collection.")

# ────────────────────────── MODEL RECONSTRUCTION ──────────────────────────
print("\n[STEP 4/7] Reconstructing model architecture...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Using device: {device}")

model = Seq2SeqLSTM(
    n_num=n_num,
    n_bool=n_bool, 
    cat_sizes=cat_sizes,
    horizon=HORIZON,
    hidden=config.get('hidden_size', 500),
    layers=config.get('lstm_layers', 3),
    emb_dim=config.get('emb_dim', 8)
).to(device)

# Load trained weights
model.load_state_dict(state_dict)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model reconstructed: {total_params:,} parameters")
print(f"✓ Weights loaded from checkpoint")

# ────────────────────────── TEST DATASET CREATION ──────────────────────────
print("\n[STEP 5/7] Creating test dataset for prediction replay...")

# Reset any previous CVE collection
test_dataset.collected_cve_ids.clear()

# Create deterministic test loader (identical to original)
test_pred_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    collate_fn=partial(pad_and_mask_fixed, flag_kind="test", horizon=HORIZON),
    num_workers=0,  # Single worker for deterministic CVE order
)

print("✓ Test dataset created with deterministic ordering")
print("✓ Using batch_size=1 for individual CVE processing")

# ────────────────────────── PREDICTION COLLECTION ──────────────────────────
print("\n[STEP 6/7] Collecting detailed predictions...")

pred_list, true_list, mh_list, me_list, date_list = [], [], [], [], []

print("  → Processing test sequences...")
with torch.no_grad():
    for num, boo, cat, Y, mt, mh, me, date_pad, lengths in tqdm(test_pred_loader, desc="collect preds"):
        # Forward pass (identical to original)
        with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
            P = model(num.to(device), boo.to(device), cat.to(device)).cpu()
        
        # Store results (remove batch dimension since batch_size=1)
        pred_list.append(P[0])          # [L, H]
        true_list.append(Y[0])          # [L, H] 
        mh_list.append(mh[0])           # [L, H]
        me_list.append(me[0])           # [L]
        date_list.append(date_pad[0])   # [L]

print(f"✓ Collected {len(pred_list)} CVE sequences")

# ────────────────────────── NETCDF CREATION ──────────────────────────
print("\n[STEP 7/7] Creating NetCDF prediction file...")

# Pad all sequences to same length (identical to original)
L_max = max(t.shape[0] for t in pred_list)
print(f"  → Maximum sequence length: {L_max}")

def right_pad_tensor(tensor, target_length, pad_value):
    """Pad tensor to target length on the first dimension."""
    if tensor.ndim == 1:
        pad_width = (0, target_length - tensor.shape[0])
    else:
        pad_width = (0, 0, 0, target_length - tensor.shape[0])
    return torch.nn.functional.pad(tensor, pad_width, value=pad_value)

# Pad and stack all tensors (identical to original)
P_padded = torch.stack([right_pad_tensor(t, L_max, 0.0) for t in pred_list])    # [N, L_max, H]
T_padded = torch.stack([right_pad_tensor(t, L_max, 0.0) for t in true_list])    # [N, L_max, H]
MH_padded = torch.stack([right_pad_tensor(t, L_max, 0) for t in mh_list])       # [N, L_max, H]
ME_padded = torch.stack([right_pad_tensor(t, L_max, 0) for t in me_list])       # [N, L_max]
DT_padded = torch.stack([right_pad_tensor(t, L_max, 0) for t in date_list])     # [N, L_max]

# Convert dates (identical to original)
DT_numpy = DT_padded.numpy()
DT_numpy[DT_numpy == 0] = np.datetime64("NaT").view("int64")
dates_2d = DT_numpy.view("datetime64[ns]")  # [N, L_max]

# Ensure float32 for NetCDF compatibility
P_padded = P_padded.float()
T_padded = T_padded.float()

# Get CVE IDs
cve_ids = test_dataset.collected_cve_ids
print(f"  → CVE IDs collected: {len(cve_ids)}")

# Create xarray Dataset (identical structure to original)
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

# Save with compression if available (identical to original)
try:
    import netCDF4
    encoding = {var: {"zlib": True, "complevel": 3} for var in ds.data_vars}
    print("  → Using netCDF4 backend with compression")
    ds.to_netcdf(OUTPUT_PATH, encoding=encoding, engine='netcdf4')
except ImportError:
    print("  → Using scipy backend (no compression)")
    ds.to_netcdf(OUTPUT_PATH, engine='scipy')

print("\n" + "=" * 50)
print("🎯 REPLAY COMPLETE")
print("=" * 50)
print(f"✓ Replayed predictions saved: {OUTPUT_PATH}")
print(f"✓ Dataset shape: {len(cve_ids)} CVEs × {L_max} timesteps × {HORIZON} horizons")
print(f"✓ Variables: predictions, ground truth, horizon mask, eval mask")
print(f"✓ Coordinates: CVE IDs, calendar dates, forecast horizons")
print(f"✓ File size: {OUTPUT_PATH.stat().st_size / (1024**2):.1f} MB")

# Verification message
print(f"\n💡 VERIFICATION:")
print(f"   Original: ml_pipeline/results/predictions/predictions_stream.nc")
print(f"   Replay:   {OUTPUT_PATH}")
print(f"   Use xarray to compare: ds1 = xr.open_dataset('original'), ds2 = xr.open_dataset('replay')") 