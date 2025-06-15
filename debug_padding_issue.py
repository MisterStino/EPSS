#!/usr/bin/env python
"""
Debug script to investigate padding issues in dataset creation
"""

import json, numpy as np, pandas as pd, duckdb, torch
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

for col in CAT_COLS:
    train_values = BIG.loc[train_mask, col].dropna().unique()
    train_values_clean = [c for c in train_values if c != "UNK"]
    mapping = {c: i + 1 for i, c in enumerate(sorted(train_values_clean))}
    mapping["UNK"] = 0
    VOCAB[col] = mapping

# Apply mappings
for col in CAT_COLS:
    BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")

print(f"\nVocabulary sizes: {[len(VOCAB[c]) for c in CAT_COLS]}")

# Check a specific CVE sequence
L_max = BIG.groupby("cve", observed=True).size().max()
print(f"Max sequence length: {L_max}")

# Test a few CVEs with different lengths to find padding issues
print(f"\nTesting multiple CVEs for padding issues...")

cve_lengths = BIG.groupby("cve", observed=True).size().sort_values()
print(f"CVE lengths: min={cve_lengths.min()}, max={cve_lengths.max()}")

# Test the 10 shortest CVEs (most padding needed)
test_cves = cve_lengths.head(10)
found_issue = False

for cve, length in test_cves.items():
    cve_data = BIG[BIG["cve"] == cve].copy()
    cat = cve_data[CAT_COLS].to_numpy("int64").astype("int64")
    T = len(cat)
    pad = L_max - T
    
    print(f"\nCVE {cve}: length={T}, pad={pad}")
    print(f"Original values: {cat}")
    
    if pad > 0:
        padded_cat = np.pad(cat, ((0, pad), (0, 0)))
        torch_cat = torch.from_numpy(padded_cat)
        
        # Check padding region
        padding_region = padded_cat[T:]
        print(f"Padding region shape: {padding_region.shape}")
        print(f"Padding region unique values: {np.unique(padding_region)}")
        
        vocab_sizes = [len(VOCAB[c]) for c in CAT_COLS]
        for i, vocab_size in enumerate(vocab_sizes):
            col_max = torch_cat[:, i].max().item()
            col_min = torch_cat[:, i].min().item()
            print(f"Column {i} ({CAT_COLS[i]}): vocab_size={vocab_size}, range=[{col_min}, {col_max}]")
            if col_max >= vocab_size:
                print(f"  ⚠️ ERROR: Column {i}: max={col_max} >= vocab_size={vocab_size}")
                problematic_vals = torch_cat[:, i][torch_cat[:, i] >= vocab_size]
                print(f"  Problematic values: {problematic_vals}")
                found_issue = True
                break
        
        if found_issue:
            break
    else:
        print("No padding needed")

if not found_issue:
    print("No padding issues found in the tested CVEs!") 