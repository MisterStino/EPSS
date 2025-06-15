#!/usr/bin/env python
"""
Debug script to investigate vocabulary size mismatches
"""

import json, numpy as np, pandas as pd, duckdb
from pathlib import Path

# Load the same data as the main script
local_execution = True
if local_execution:
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
else:
    PARQUET_DIR = Path("data/minimal_v1_timeseries.parquet")

# Load data
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

# Define categorical columns
CAT_COLS = [
    "primary_cvss_ver", "primary_cvss_sev", "dominant_event_type",
    "prev_event_type", "primary_source", "cwe_id",
    "vuln_status", "source_identifier",
]

# Filter to existing columns
CAT_COLS = [c for c in CAT_COLS if c in BIG.columns]
print(f"[INFO] Available CAT_COLS: {CAT_COLS}")

# Create train/val/test splits
days = np.sort(BIG["date"].unique())
VAL_CUT = pd.to_datetime(days[int(0.64 * len(days))])
TEST_CUT = pd.to_datetime(days[int(0.80 * len(days))])

BIG["flag_train"] = (BIG["date"] < VAL_CUT).astype("uint8")
BIG["flag_val"] = ((BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT)).astype("uint8")
BIG["flag_test"] = (BIG["date"] >= TEST_CUT).astype("uint8")

# Create vocab using training data only
train_mask = BIG["flag_train"] == 1
VOCAB = {}

print("\n" + "="*50)
print("VOCABULARY ANALYSIS")
print("="*50)

for col in CAT_COLS:
    print(f"\n[{col}]")
    
    # Get all unique values in the dataset
    all_values = BIG[col].dropna().unique()
    print(f"  All unique values in dataset: {len(all_values)}")
    print(f"  Sample values: {sorted(all_values)[:10]}")
    
    # Check for "UNK" in the data
    has_unk = "UNK" in all_values
    print(f"  Contains 'UNK' string: {has_unk}")
    
    # Get training values only
    train_values = BIG.loc[train_mask, col].dropna().unique()
    print(f"  Training unique values: {len(train_values)}")
    
    # Remove UNK from training values if present
    train_values_clean = [c for c in train_values if c != "UNK"]
    print(f"  Training values after removing UNK: {len(train_values_clean)}")
    
    # Build vocabulary (same as main script)
    mapping = {c: i + 1 for i, c in enumerate(sorted(train_values_clean))}
    mapping["UNK"] = 0
    
    VOCAB[col] = mapping
    vocab_size = len(mapping)
    print(f"  Final vocab size: {vocab_size}")
    print(f"  Vocab mapping: {mapping}")
    
    # Apply mapping to entire dataset
    mapped_values = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")
    
    # Check for out-of-bounds values
    max_mapped = mapped_values.max()
    min_mapped = mapped_values.min()
    print(f"  Mapped values range: [{min_mapped}, {max_mapped}]")
    
    if max_mapped >= vocab_size:
        print(f"  ⚠️  ERROR: Found mapped values >= vocab_size ({vocab_size})!")
        problematic = mapped_values >= vocab_size
        print(f"  Problematic values count: {problematic.sum()}")
        
        # Show some examples
        prob_indices = np.where(problematic)[0][:5]
        for idx in prob_indices:
            orig_val = BIG.iloc[idx][col]
            mapped_val = mapped_values.iloc[idx]
            print(f"    Row {idx}: '{orig_val}' -> {mapped_val}")
    else:
        print(f"  ✓ All mapped values are within bounds")

print(f"\nFinal vocabulary saved to vocab.json")
with open("vocab_debug.json", "w") as f:
    json.dump(VOCAB, f, indent=2) 