#!/usr/bin/env python3
"""
Trace CUDA assertion under exact training conditions.
This replicates the exact setup from lstm_exp_window_eval.py to find the bug.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import duckdb
from pathlib import Path
from tqdm import tqdm

def trace_training_conditions():
    print("=" * 80)
    print("TRACING CUDA ASSERTION UNDER EXACT TRAINING CONDITIONS")
    print("=" * 80)
    
    # Replicate exact setup from lstm_exp_window_eval.py
    local_execution = True
    
    LOCAL_CONFIG = {
        'batch_size': 64,
        'hidden_size': 256,
        'lstm_layers': 3,
        'emb_dim': 8,
    }
    
    CONFIG = LOCAL_CONFIG
    print(f"Using LOCAL config: batch={CONFIG['batch_size']}, hidden={CONFIG['hidden_size']}")
    
    # 1. Load data exactly as main script
    print("\n[STEP 1] Loading data exactly as main script...")
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    if PARQUET_DIR.is_dir():
        parquet_glob = str(PARQUET_DIR / "*.parquet")
    else:
        parquet_glob = str(PARQUET_DIR)
    
    BIG = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}')").df()
    BIG["date"] = pd.to_datetime(BIG["date"])
    print(f"✓ Loaded {len(BIG):,} rows")
    
    # 2. Apply exact preprocessing from main script
    print("\n[STEP 2] Applying exact preprocessing...")
    
    # Drop columns
    DROP_COLS = [
        "cve_date_key", "original_date",
        "reconstruction_timestamp", "reconstruction_timestamp_raw",
        "details_combined", "details_longest", "event_data_merged",
        "description_all", "description_en",
        "primary_cvss_vec", "cve_tags", "reference_count",
    ]
    
    for col in DROP_COLS:
        if col in BIG.columns:
            BIG = BIG.drop(columns=[col])
    
    # Transform EPSS
    def transform_epss(arr, mode="logit", eps=1e-6):
        p = np.clip(arr.astype("float64"), eps, 1.0 - eps)
        if mode == "logit":
            out = np.log(p / (1.0 - p))
        return out.astype("float32")
    
    BIG["epss"] = transform_epss(BIG["epss"].values, mode="logit", eps=1e-6)
    
    # Create temporal splits
    days = np.sort(BIG["date"].unique())
    VAL_CUT = pd.to_datetime(days[int(0.64 * len(days))])
    TEST_CUT = pd.to_datetime(days[int(0.80 * len(days))])
    
    BIG["flag_train"] = (BIG["date"] < VAL_CUT).astype("uint8")
    BIG["flag_val"] = ((BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT)).astype("uint8")
    BIG["flag_test"] = (BIG["date"] >= TEST_CUT).astype("uint8")
    
    train_mask = BIG["flag_train"] == 1
    print(f"✓ Train split: {train_mask.sum():,} rows")
    
    # 3. Create vocabularies and apply mapping exactly as main script
    print("\n[STEP 3] Creating vocabularies and mapping...")
    CAT_COLS = ['primary_cvss_sev', 'dominant_event_type']
    VOCAB = {}
    
    for col in CAT_COLS:
        cats = BIG.loc[train_mask, col].dropna().unique()
        VOCAB[col] = {"UNK": 0, **{c: i + 1 for i, c in enumerate(sorted(cats))}}
        BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")
        print(f"  {col}: {len(VOCAB[col])} categories, max_mapped={BIG[col].max()}")
    
    # 4. Create exact CVEDataset class from main script
    print("\n[STEP 4] Creating exact CVEDataset...")
    
    # Get column definitions exactly as main script
    BOOL_COLS = lambda df: [c for c in df.columns
                           if c.startswith(("has_", "is_"))
                           or c in ("same_day_multi_source",
                                   "has_v2","has_v30","has_v31","has_v40")]
    
    NUM_COLS = [c for c in BIG.columns 
                if c not in CAT_COLS + BOOL_COLS(BIG) + 
                ["cve", "date", "epss", "flag_train", "flag_val", "flag_test"]]
    
    print(f"  Numeric columns: {len(NUM_COLS)}")
    print(f"  Boolean columns: {len(BOOL_COLS(BIG))}")
    print(f"  Categorical columns: {len(CAT_COLS)}")
    
    class CVEDataset(Dataset):
        def __init__(self, df: pd.DataFrame, L_max: int,
                     horizon: int, flag_col: str, cat_cols: list):
            self.num, self.boo, self.cat = [], [], []
            self.Y, self.m_t, self.m_h, self.m_eval = [], [], [], []
            self.cat_cols = cat_cols

            num = df[NUM_COLS].to_numpy("float32")
            boo = df[BOOL_COLS(df)].to_numpy("uint8")
            cat = (
                df[cat_cols].to_numpy("int64").astype("int64")
                if cat_cols else np.empty((len(df), 0), dtype="int64")
            )
            epss = df["epss"].to_numpy("float32").reshape(-1, 1)
            flag = df[flag_col].to_numpy("float32")

            print(f"    Processing {len(df.groupby('cve'))} CVE sequences...")
            
            for cve_idx, (cve, idx) in enumerate(df.groupby("cve", sort=False).groups.items()):
                idx = np.asarray(idx)
                T = len(idx)
                pad = L_max - T

                self.num.append(torch.from_numpy(np.pad(num[idx], ((0, pad), (0, 0)))))
                self.boo.append(torch.from_numpy(np.pad(boo[idx], ((0, pad), (0, 0)))))
                
                if cat_cols:
                    cat_sequence = cat[idx]
                    
                    # Check this specific CVE sequence for out-of-bounds
                    for col_idx, col in enumerate(cat_cols):
                        col_data = cat_sequence[:, col_idx]
                        max_val = col_data.max()
                        vocab_size = len(VOCAB[col])
                        
                        if max_val >= vocab_size:
                            print(f"      ❌ FOUND BUG in CVE {cve} (sequence {cve_idx})!")
                            print(f"        Column: {col}")
                            print(f"        Max value: {max_val}, Vocab size: {vocab_size}")
                            print(f"        Sequence length: {T}")
                            print(f"        Bad positions: {np.where(col_data >= vocab_size)[0]}")
                            print(f"        Bad values: {col_data[col_data >= vocab_size]}")
                            
                            # Get original dataframe info
                            original_rows = idx[col_data >= vocab_size]
                            print(f"        Original DF rows: {original_rows[:5]}...")
                            print(f"        Dates: {df.loc[original_rows[:5], 'date'].values}")
                            print(f"        Split flags: train={df.loc[original_rows[0], 'flag_train']}, val={df.loc[original_rows[0], 'flag_val']}, test={df.loc[original_rows[0], 'flag_test']}")
                            
                            return  # Stop at first bug
                    
                    padded_cat = np.pad(cat_sequence, ((0, pad), (0, 0)))
                    self.cat.append(torch.from_numpy(padded_cat))
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
            return (self.num[i], self.boo[i], self.cat[i],
                    self.Y[i], self.m_t[i], self.m_h[i], self.m_eval[i])
    
    def collate(batch):
        return tuple(torch.stack(x, 0) for x in zip(*batch))
    
    # 5. Create datasets exactly as main script
    print("\n[STEP 5] Creating datasets...")
    L_max = BIG.groupby("cve", observed=True).size().max()
    print(f"  Max sequence length: {L_max}")
    
    HORIZON = 30
    
    print("  Creating training dataset...")
    tr_ds = CVEDataset(BIG, L_max, HORIZON, "flag_train", CAT_COLS)
    print(f"  ✓ Training dataset: {len(tr_ds)} sequences")
    
    # 6. Create dataloader and test batching
    print("\n[STEP 6] Testing dataloader with exact training conditions...")
    BATCH = CONFIG['batch_size']
    
    tr_ld = DataLoader(tr_ds, BATCH, True, collate_fn=collate, pin_memory=True)
    print(f"  ✓ DataLoader created: {len(tr_ld)} batches")
    
    # 7. Test first few batches
    print("\n[STEP 7] Testing first few training batches...")
    
    for batch_idx, batch in enumerate(tr_ld):
        if batch_idx >= 3:  # Test first 3 batches
            break
            
        num, boo, cat, Y, mt, mh, me = batch
        print(f"\n  Batch {batch_idx + 1}:")
        print(f"    Shapes: num={num.shape}, boo={boo.shape}, cat={cat.shape}")
        
        # Check categorical bounds in this batch
        for col_idx, col in enumerate(CAT_COLS):
            col_data = cat[:, :, col_idx]
            max_val = col_data.max().item()
            vocab_size = len(VOCAB[col])
            
            print(f"    {col}: max={max_val}, vocab_size={vocab_size}")
            
            if max_val >= vocab_size:
                print(f"      ❌ BATCH BUG FOUND!")
                print(f"        Column: {col} (index {col_idx})")
                print(f"        Max value: {max_val}, Vocab size: {vocab_size}")
                
                # Find exact positions
                bad_mask = col_data >= vocab_size
                bad_positions = torch.where(bad_mask)
                bad_values = col_data[bad_mask]
                
                print(f"        Bad positions (batch, seq): {list(zip(bad_positions[0].tolist()[:5], bad_positions[1].tolist()[:5]))}")
                print(f"        Bad values: {bad_values[:10].tolist()}")
                
                # This is the exact assertion that would fail
                print(f"\n        🔥 EXACT ASSERTION FAILURE:")
                print(f"        assert (cat[..., {col_idx}] < {vocab_size}).all()")
                print(f"        Failed because max value is {max_val}")
                
                return  # Stop at first failure
        
        print(f"    ✅ Batch {batch_idx + 1} OK")
    
    # 8. Test model creation and forward pass
    print("\n[STEP 8] Testing model creation and forward pass...")
    
    class TestSeq2SeqLSTM(nn.Module):
        def __init__(self, n_num: int, n_bool: int, cat_sizes,
                     horizon: int = 30, hidden: int = 256,
                     layers: int = 3, emb_dim: int = 8):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(s, emb_dim, padding_idx=0) for s in cat_sizes])
            in_dim = n_num + n_bool + emb_dim * len(cat_sizes)
            self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, horizon)

        def forward(self, num, boo, cat):
            if len(self.emb):
                cat = cat.long()
                # This is the exact assertion from the main script
                cat_sizes = [emb.num_embeddings for emb in self.emb]
                for i, vocab_size in enumerate(cat_sizes):
                    if not (cat[..., i] < vocab_size).all():
                        print(f"        ❌ ASSERTION FAILURE in forward pass!")
                        print(f"        Column {i}: max={cat[..., i].max().item()}, vocab_size={vocab_size}")
                        bad_mask = cat[..., i] >= vocab_size
                        bad_positions = torch.where(bad_mask)
                        print(f"        Bad positions: {list(zip(bad_positions[0].tolist()[:5], bad_positions[1].tolist()[:5]))}")
                        raise AssertionError(f"Found id ≥ vocab_size in column {i} (vocab_size={vocab_size})!")
                
                e = torch.cat([emb(cat[..., i]) for i, emb in enumerate(self.emb)], dim=-1)
                x = torch.cat([num, boo.float(), e], dim=-1)
            else:
                x = torch.cat([num, boo.float()], dim=-1)
            h, _ = self.lstm(x)
            return self.head(h)
    
    model = TestSeq2SeqLSTM(
        n_num=len(NUM_COLS),
        n_bool=len(BOOL_COLS(BIG)),
        cat_sizes=[len(VOCAB[c]) for c in CAT_COLS],
        horizon=HORIZON,
        hidden=CONFIG['hidden_size'],
        layers=CONFIG['lstm_layers']
    )
    
    print(f"  ✓ Model created with cat_sizes: {[len(VOCAB[c]) for c in CAT_COLS]}")
    
    # Test forward pass on first batch
    print("\n[STEP 9] Testing forward pass on first batch...")
    first_batch = next(iter(tr_ld))
    num, boo, cat, Y, mt, mh, me = first_batch
    
    try:
        with torch.no_grad():
            pred = model(num, boo, cat)
        print(f"  ✓ Forward pass successful: {pred.shape}")
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        print("  This is the exact error from the main script!")
    
    print("\n" + "=" * 80)
    print("TRAINING CONDITIONS TRACE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    trace_training_conditions() 