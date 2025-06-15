#!/usr/bin/env python3
"""
Check if validation/test data contains indices that exceed training vocabulary.
"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path

def check_val_test_indices():
    """Check validation/test data for out-of-bounds indices."""
    
    print("=" * 80)
    print("CHECKING VALIDATION/TEST DATA FOR OUT-OF-BOUNDS INDICES")
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
    val_mask = BIG["flag_val"] == 1
    test_mask = BIG["flag_test"] == 1
    
    print(f"Data splits:")
    print(f"  Train: {train_mask.sum():,} rows")
    print(f"  Val:   {val_mask.sum():,} rows")
    print(f"  Test:  {test_mask.sum():,} rows")
    print()
    
    # Create vocabularies from training data only
    VOCAB = {}
    for col in CAT_COLS:
        cats = BIG.loc[train_mask, col].dropna().unique()
        VOCAB[col] = {"UNK": 0, **{c: i + 1 for i, c in enumerate(sorted(cats))}}
        BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")
    
    print("Training vocabularies:")
    for col in CAT_COLS:
        print(f"  {col}: {VOCAB[col]} (size: {len(VOCAB[col])})")
    print()
    
    # Check each split for out-of-bounds indices
    for split_name, split_mask in [("TRAIN", train_mask), ("VAL", val_mask), ("TEST", test_mask)]:
        print(f"=" * 60)
        print(f"CHECKING {split_name} DATA")
        print(f"=" * 60)
        
        split_data = BIG[split_mask]
        print(f"{split_name} rows: {len(split_data):,}")
        
        for col in CAT_COLS:
            vocab_size = len(VOCAB[col])
            col_data = split_data[col]
            
            max_val = col_data.max()
            min_val = col_data.min()
            
            print(f"  {col}: min={min_val}, max={max_val}, vocab_size={vocab_size}")
            
            if max_val >= vocab_size:
                print(f"    ❌ OUT-OF-BOUNDS: max_val {max_val} >= vocab_size {vocab_size}")
                
                # Find the problematic values
                bad_mask = col_data >= vocab_size
                bad_count = bad_mask.sum()
                bad_values = col_data[bad_mask].unique()
                
                print(f"    Bad value count: {bad_count:,}")
                print(f"    Bad values: {sorted(bad_values)}")
                
                # Sample some bad rows
                bad_rows = split_data[bad_mask].head(5)
                print(f"    Sample bad rows:")
                for _, row in bad_rows.iterrows():
                    print(f"      CVE: {row['cve']}, Date: {row['date']}, {col}: {row[col]}")
                
            else:
                print(f"    ✅ All values within bounds")
        
        print()

if __name__ == "__main__":
    check_val_test_indices() 