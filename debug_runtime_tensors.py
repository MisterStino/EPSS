#!/usr/bin/env python3
"""
Debug the actual tensor values created during CVEDataset processing.
"""

import pandas as pd
import numpy as np
import torch
import duckdb
from pathlib import Path

def debug_runtime_tensors():
    """Debug the actual tensor creation process."""
    
    print("=" * 80)
    print("DEBUGGING RUNTIME TENSOR CREATION")
    print("=" * 80)
    
    # Load and preprocess data exactly as main script
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    if PARQUET_DIR.is_dir():
        parquet_glob = str(PARQUET_DIR / "*.parquet")
    else:
        parquet_glob = str(PARQUET_DIR)
    
    BIG = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}')").df()
    BIG["date"] = pd.to_datetime(BIG["date"])
    
    CAT_COLS = ['primary_cvss_sev', 'dominant_event_type']
    
    # Create temporal splits exactly as main script
    days = np.sort(BIG["date"].unique())
    VAL_CUT = pd.to_datetime(days[int(0.64 * len(days))])
    TEST_CUT = pd.to_datetime(days[int(0.80 * len(days))])
    
    BIG["flag_train"] = (BIG["date"] < VAL_CUT).astype("uint8")
    BIG["flag_val"] = ((BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT)).astype("uint8")
    BIG["flag_test"] = (BIG["date"] >= TEST_CUT).astype("uint8")
    
    train_mask = BIG["flag_train"] == 1
    
    # Create vocabularies exactly as main script
    VOCAB = {}
    for col in CAT_COLS:
        cats = BIG.loc[train_mask, col].dropna().unique()
        VOCAB[col] = {"UNK": 0, **{c: i + 1 for i, c in enumerate(sorted(cats))}}
        BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")
    
    print("Vocabularies created:")
    for col in CAT_COLS:
        print(f"  {col}: {VOCAB[col]} (size: {len(VOCAB[col])})")
    print()
    
    # Check the preprocessed data
    print("Checking preprocessed categorical data:")
    for col in CAT_COLS:
        max_val = BIG[col].max()
        min_val = BIG[col].min()
        vocab_size = len(VOCAB[col])
        print(f"  {col}: min={min_val}, max={max_val}, vocab_size={vocab_size}")
        if max_val >= vocab_size:
            print(f"    ❌ PREPROCESSING BUG: max_val {max_val} >= vocab_size {vocab_size}")
        else:
            print(f"    ✅ Preprocessing OK")
    print()
    
    # Simulate CVEDataset tensor creation for first few CVEs
    print("Simulating CVEDataset tensor creation:")
    
    # Get categorical data as numpy array (as CVEDataset does)
    cat_data = BIG[CAT_COLS].to_numpy("int64").astype("int64")
    print(f"Categorical array shape: {cat_data.shape}")
    print(f"Categorical array dtype: {cat_data.dtype}")
    print()
    
    # Process first 3 CVEs to debug
    cve_groups = list(BIG.groupby("cve", sort=False).groups.items())[:3]
    L_max = BIG.groupby("cve", observed=True).size().max()
    
    for cve_idx, (cve, idx) in enumerate(cve_groups):
        print(f"=" * 60)
        print(f"DEBUGGING CVE {cve_idx + 1}: {cve}")
        print(f"=" * 60)
        
        idx = np.asarray(idx)
        T = len(idx)
        pad = L_max - T
        
        print(f"CVE sequence length: {T}")
        print(f"Padding needed: {pad}")
        print(f"Row indices: {idx[:5]}...{idx[-5:] if len(idx) > 5 else idx}")
        print()
        
        # Extract categorical data for this CVE
        cve_cat_data = cat_data[idx]  # Shape: [T, 2]
        print(f"CVE categorical data shape: {cve_cat_data.shape}")
        print(f"CVE categorical data dtype: {cve_cat_data.dtype}")
        
        # Check bounds before padding
        for col_idx, col in enumerate(CAT_COLS):
            col_data = cve_cat_data[:, col_idx]
            max_val = col_data.max()
            min_val = col_data.min()
            vocab_size = len(VOCAB[col])
            print(f"  {col} (col {col_idx}): min={min_val}, max={max_val}, vocab_size={vocab_size}")
            if max_val >= vocab_size:
                print(f"    ❌ PRE-PADDING BUG: max_val {max_val} >= vocab_size {vocab_size}")
                # Find problematic rows
                bad_mask = col_data >= vocab_size
                bad_indices = np.where(bad_mask)[0]
                print(f"    Bad row indices in sequence: {bad_indices}")
                print(f"    Bad values: {col_data[bad_mask]}")
                # Map back to original dataframe
                original_rows = idx[bad_indices]
                print(f"    Original dataframe rows: {original_rows}")
                print(f"    Original values: {BIG.loc[original_rows, col].values}")
            else:
                print(f"    ✅ Pre-padding OK")
        print()
        
        # Apply padding (as CVEDataset does)
        padded_cat_data = np.pad(cve_cat_data, ((0, pad), (0, 0)))
        print(f"Padded categorical data shape: {padded_cat_data.shape}")
        
        # Check bounds after padding
        for col_idx, col in enumerate(CAT_COLS):
            col_data = padded_cat_data[:, col_idx]
            max_val = col_data.max()
            min_val = col_data.min()
            vocab_size = len(VOCAB[col])
            print(f"  {col} (col {col_idx}) after padding: min={min_val}, max={max_val}, vocab_size={vocab_size}")
            if max_val >= vocab_size:
                print(f"    ❌ POST-PADDING BUG: max_val {max_val} >= vocab_size {vocab_size}")
                # Find where the bad values are
                bad_mask = col_data >= vocab_size
                bad_positions = np.where(bad_mask)[0]
                print(f"    Bad positions: {bad_positions}")
                print(f"    Bad values: {col_data[bad_mask]}")
                # Check if these are in the padding region
                if any(pos >= T for pos in bad_positions):
                    print(f"    ❌ PADDING INTRODUCED BAD VALUES!")
                else:
                    print(f"    Bad values are in original data, not padding")
            else:
                print(f"    ✅ Post-padding OK")
        print()
        
        # Convert to tensor (as CVEDataset does)
        tensor_cat_data = torch.from_numpy(padded_cat_data)
        print(f"Tensor shape: {tensor_cat_data.shape}")
        print(f"Tensor dtype: {tensor_cat_data.dtype}")
        
        # Final bounds check on tensor
        for col_idx, col in enumerate(CAT_COLS):
            col_tensor = tensor_cat_data[:, col_idx]
            max_val = col_tensor.max().item()
            min_val = col_tensor.min().item()
            vocab_size = len(VOCAB[col])
            print(f"  {col} (col {col_idx}) tensor: min={min_val}, max={max_val}, vocab_size={vocab_size}")
            if max_val >= vocab_size:
                print(f"    ❌ TENSOR BUG: max_val {max_val} >= vocab_size {vocab_size}")
            else:
                print(f"    ✅ Tensor OK")
        print()

if __name__ == "__main__":
    debug_runtime_tensors() 