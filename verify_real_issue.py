#!/usr/bin/env python
"""
Verify if the categorical index issue is real or fabricated
"""

import json, numpy as np, pandas as pd, duckdb, torch
from pathlib import Path

def main():
    print("🔍 VERIFYING THE ACTUAL CATEGORICAL INDEX ISSUE")
    print("=" * 60)
    
    # Load the exact same data as the training script
    local_execution = True
    if local_execution:
        PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    else:
        PARQUET_DIR = Path("data/minimal_v1_timeseries.parquet")

    if PARQUET_DIR.is_dir():
        parquet_glob = str(PARQUET_DIR / "*.parquet")
    else:
        parquet_glob = str(PARQUET_DIR)
    
    # Fix Windows path separators for DuckDB
    parquet_glob = parquet_glob.replace("\\", "/")

    DROP_COLS = [
        "cve_date_key", "original_date",
        "reconstruction_timestamp", "reconstruction_timestamp_raw",
        "details_combined", "details_longest", "event_data_merged",
        "description_all", "description_en",
        "primary_cvss_vec", "cve_tags", "reference_count",
    ]

    # Load data exactly as the training script does
    print("📊 Loading data...")
    BIG = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}')").df()
    print(f"Raw data shape: {BIG.shape}")
    
    # Apply the same filtering as training script
    BIG = BIG.drop(columns=DROP_COLS)
    print(f"After dropping columns: {BIG.shape}")
    
    # Check the categorical columns
    CAT_COLS = ["cve_id", "dominant_event_type"]
    print(f"\n🏷️ EXAMINING CATEGORICAL COLUMNS: {CAT_COLS}")
    
    for col in CAT_COLS:
        print(f"\n--- {col} ---")
        unique_vals = BIG[col].unique()
        print(f"Unique values count: {len(unique_vals)}")
        print(f"Data type: {BIG[col].dtype}")
        print(f"Sample values: {list(unique_vals[:10])}")
        print(f"Min/Max: [{BIG[col].min()}, {BIG[col].max()}]")
        
        # Check for any weird values
        if BIG[col].dtype == 'object':
            print(f"String lengths: {[len(str(v)) for v in unique_vals[:5]]}")
        
        # Check for nulls
        null_count = BIG[col].isnull().sum()
        print(f"Null values: {null_count}")
    
    print(f"\n🔢 EXAMINING THE VOCABULARY CREATION PROCESS")
    print("-" * 40)
    
    # Load vocabulary file
    vocab_file = Path("vocab.json")
    if vocab_file.exists():
        with open(vocab_file, "r") as f:
            vocab = json.load(f)
        print(f"Vocabulary loaded from {vocab_file}")
    else:
        print("Creating vocabulary from scratch...")
        vocab = {}
        for col in CAT_COLS:
            unique_vals = BIG[col].unique()
            vocab[col] = {"UNK": 0}
            for i, val in enumerate(sorted(unique_vals), 1):
                vocab[col][str(val)] = i
    
    # Print vocabulary details
    for col in CAT_COLS:
        print(f"\n--- {col} vocabulary ---")
        print(f"Vocabulary size: {len(vocab[col])}")
        print(f"Sample mappings: {dict(list(vocab[col].items())[:5])}")
    
    print(f"\n🎯 TESTING CATEGORICAL ENCODING")
    print("-" * 40)
    
    # Test the actual encoding process
    for col in CAT_COLS:
        print(f"\n--- Testing {col} ---")
        
        # Map values exactly as the training script does
        mapped_values = BIG[col].map(lambda x: vocab[col].get(str(x), 0))
        
        print(f"Original values range: [{BIG[col].min()}, {BIG[col].max()}]")
        print(f"Mapped values range: [{mapped_values.min()}, {mapped_values.max()}]")
        print(f"Vocab size: {len(vocab[col])}")
        print(f"Max allowed index: {len(vocab[col]) - 1}")
        
        # Check if any mapped values are out of bounds
        max_valid_idx = len(vocab[col]) - 1
        out_of_bounds = mapped_values > max_valid_idx
        out_of_bounds_count = out_of_bounds.sum()
        
        if out_of_bounds_count > 0:
            print(f"❌ FOUND {out_of_bounds_count} OUT-OF-BOUNDS VALUES!")
            print(f"Out-of-bounds values: {mapped_values[out_of_bounds].unique()}")
            # Show some examples
            problem_rows = BIG[out_of_bounds]
            print(f"Sample problematic rows:")
            print(problem_rows[[col]].head())
        else:
            print(f"✅ All mapped values are within bounds [0, {max_valid_idx}]")
    
    print(f"\n🏗️ TESTING DATASET CREATION PROCESS")
    print("-" * 40)
    
    # Test a small sample of the dataset creation
    sample_cve = BIG['cve'].iloc[0]
    sample_data = BIG[BIG['cve'] == sample_cve].copy()
    print(f"Testing with sample CVE: {sample_cve} ({len(sample_data)} rows)")
    
    # Apply categorical mapping
    for col in CAT_COLS:
        sample_data[col] = sample_data[col].map(lambda x: vocab[col].get(str(x), 0))
    
    # Extract categorical array
    cat_array = sample_data[CAT_COLS].to_numpy(dtype=np.int64)
    print(f"Categorical array shape: {cat_array.shape}")
    print(f"Categorical array values:")
    print(cat_array)
    print(f"Min/Max values: [{cat_array.min()}, {cat_array.max()}]")
    
    # Check bounds
    for i, col in enumerate(CAT_COLS):
        col_values = cat_array[:, i]
        vocab_size = len(vocab[col])
        max_val = col_values.max()
        print(f"{col}: max_value={max_val}, vocab_size={vocab_size}, valid={max_val < vocab_size}")

if __name__ == "__main__":
    main() 