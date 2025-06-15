#!/usr/bin/env python
"""
Debug script to investigate DataLoader batching issues
"""

import json, numpy as np, pandas as pd, duckdb, torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

# Load the same data as the main script
local_execution = True
if local_execution:
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")

parquet_glob = str(PARQUET_DIR / "*.parquet") if PARQUET_DIR.is_dir() else str(PARQUET_DIR)

DROP_COLS = [
    "cve_date_key", "original_date",
    "reconstruction_timestamp", "reconstruction_timestamp_raw",
    "details_combined", "details_longest", "event_data_merged",
    "description_all", "description_en",
    "primary_cvss_vec", "cve_tags", "reference_count",
]

# Load with exclusions
schema_df = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}') LIMIT 0").df()
probe_cols = schema_df.columns.tolist()
drop_real = [c for c in DROP_COLS if c in probe_cols]
quote_cols = ", ".join(f'"{c}"' for c in drop_real)
exclude = f" EXCLUDE ({quote_cols})" if drop_real else ""

BIG = duckdb.sql(f"SELECT *{exclude} FROM parquet_scan('{parquet_glob}')").df()
BIG["date"] = pd.to_datetime(BIG["date"])

print(f"[INFO] loaded {len(BIG):,} rows × {BIG.shape[1]} columns")

# Define columns
CAT_COLS = ['primary_cvss_sev', 'dominant_event_type']
BOOL_COLS = [c for c in BIG.columns if c.startswith(("has_", "is_")) or c in ("same_day_multi_source", "has_v2","has_v30","has_v31","has_v40")]
NUM_COLS = [c for c, t in BIG.dtypes.items() 
            if (pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t)) 
            and c not in BOOL_COLS + ["flag_train", "flag_val", "flag_test"]]

# Create train/val/test splits
days = np.sort(BIG["date"].unique())
VAL_CUT = pd.to_datetime(days[int(0.64 * len(days))])
TEST_CUT = pd.to_datetime(days[int(0.80 * len(days))])

BIG["flag_train"] = (BIG["date"] < VAL_CUT).astype("uint8")
BIG["flag_val"] = ((BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT)).astype("uint8")
BIG["flag_test"] = (BIG["date"] >= TEST_CUT).astype("uint8")

# Create vocab
train_mask = BIG["flag_train"] == 1
VOCAB = {}

for col in CAT_COLS:
    train_values = BIG.loc[train_mask, col].dropna().unique()
    train_values_clean = [c for c in train_values if c != "UNK"]
    mapping = {c: i + 1 for i, c in enumerate(sorted(train_values_clean))}
    mapping["UNK"] = 0
    VOCAB[col] = mapping

# Apply mappings
for col in CAT_COLS:
    BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")

print(f"Vocabulary sizes: {[len(VOCAB[c]) for c in CAT_COLS]}")

# Transform EPSS
def transform_epss(arr, mode="logit", eps=1e-6):
    p = np.clip(arr.astype("float64"), eps, 1.0 - eps)
    if mode == "logit":
        out = np.log(p / (1.0 - p))
    return out.astype("float32")

BIG["epss"] = transform_epss(BIG["epss"].values, mode="logit", eps=1e-6)

# Handle booleans
for col in BOOL_COLS:
    if pd.api.types.is_bool_dtype(BIG[col]):
        BIG[col] = BIG[col].fillna(False).astype("uint8")
    else:
        BIG[col] = BIG[col].fillna(0).astype("uint8")

# Handle numerics (simplified - no scaling for debugging)
for col in NUM_COLS:
    BIG[col] = BIG[col].fillna(0).astype("float32")

# Recreate the CVEDataset class
class CVEDataset(Dataset):
    def __init__(self, df: pd.DataFrame, L_max: int, horizon: int, flag_col: str, cat_cols: list):
        self.num, self.boo, self.cat = [], [], []
        self.Y, self.m_t, self.m_h, self.m_eval = [], [], [], []
        self.cat_cols = cat_cols

        num = df[NUM_COLS].to_numpy("float32")
        boo = df[BOOL_COLS].to_numpy("uint8")
        cat = df[cat_cols].to_numpy("int64").astype("int64") if cat_cols else np.empty((len(df), 0), dtype="int64")
        epss = df["epss"].to_numpy("float32").reshape(-1, 1)
        flag = df[flag_col].to_numpy("float32")

        print(f"Creating dataset with cat dtype: {cat.dtype}")
        print(f"Cat values range: [{cat.min()}, {cat.max()}]")

        for cve_name, idx in df.groupby("cve", sort=False).groups.items():
            idx = np.asarray(idx)
            T = len(idx)
            pad = L_max - T

            # Debug specific CVE
            cat_sequence = cat[idx]
            print(f"CVE {cve_name}: T={T}, pad={pad}, cat_seq_range=[{cat_sequence.min()}, {cat_sequence.max()}]")

            self.num.append(torch.from_numpy(np.pad(num[idx], ((0, pad), (0, 0)))))
            self.boo.append(torch.from_numpy(np.pad(boo[idx], ((0, pad), (0, 0)))))
            
            if cat_cols:
                padded_cat = np.pad(cat[idx], ((0, pad), (0, 0)))
                cat_tensor = torch.from_numpy(padded_cat)
                print(f"  Padded cat range: [{cat_tensor.min().item()}, {cat_tensor.max().item()}]")
                self.cat.append(cat_tensor)
            else:
                self.cat.append(torch.zeros((L_max, 0), dtype=torch.int64))

            self.m_t.append(torch.from_numpy(np.r_[np.ones(T), np.zeros(pad)]))
            self.m_eval.append(torch.from_numpy(np.r_[flag[idx], np.zeros(pad)]))

            y = np.zeros((L_max, horizon), "float32")
            mh = np.zeros_like(y)
            for t in range(T):
                k = min(horizon, T - t - 1)
                if k:
                    y[t, :k] = epss[idx][t + 1:t + 1 + k, 0]
                    mh[t, :k] = 1
            self.Y.append(torch.from_numpy(y))
            self.m_h.append(torch.from_numpy(mh))

    def __len__(self):
        return len(self.num)

    def __getitem__(self, i):
        return (self.num[i], self.boo[i], self.cat[i], self.Y[i], self.m_t[i], self.m_h[i], self.m_eval[i])

def collate(batch):
    return tuple(torch.stack(x, 0) for x in zip(*batch))

# Create a small dataset and dataloader
L_max = BIG.groupby("cve", observed=True).size().max()
print(f"Max sequence length: {L_max}")

# Create a very small training dataset for debugging
small_df = BIG[BIG["flag_train"] == 1].groupby("cve").head(5).copy()  # First 5 rows per CVE
small_dataset = CVEDataset(small_df, L_max, 30, "flag_train", CAT_COLS)
small_loader = DataLoader(small_dataset, batch_size=2, shuffle=False, collate_fn=collate)

print(f"\nCreated small dataset with {len(small_dataset)} sequences")

# Debug first batch
print(f"\nDebugging first batch...")
for i, (num, boo, cat, Y, mt, mh, me) in enumerate(small_loader):
    print(f"Batch {i}:")
    print(f"  cat.shape: {cat.shape}")
    print(f"  cat.dtype: {cat.dtype}")
    for j in range(len(CAT_COLS)):
        cat_col = cat[:, :, j]  # [batch, seq, col]
        col_min = cat_col.min().item()
        col_max = cat_col.max().item()
        vocab_size = len(VOCAB[CAT_COLS[j]])
        print(f"  Column {j} ({CAT_COLS[j]}): vocab_size={vocab_size}, range=[{col_min}, {col_max}]")
        
        if col_max >= vocab_size:
            print(f"    ⚠️ ERROR: Found index {col_max} >= vocab_size {vocab_size}")
            # Find where the problematic values are
            problematic_mask = cat_col >= vocab_size
            problematic_positions = torch.where(problematic_mask)
            print(f"    Problematic positions (batch, seq): {list(zip(problematic_positions[0].tolist(), problematic_positions[1].tolist()))}")
            problematic_values = cat_col[problematic_mask]
            print(f"    Problematic values: {problematic_values.tolist()}")
    
    if i >= 2:  # Only check first few batches
        break

print("Debug completed!") 