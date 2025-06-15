#!/usr/bin/env python3
"""
Test the UNK string collision hypothesis.
"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path

def test_unk_collision():
    """Test if literal 'UNK' strings exist in categorical columns."""
    
    print("=" * 80)
    print("TESTING UNK STRING COLLISION HYPOTHESIS")
    print("=" * 80)
    
    # Load data exactly as main script
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    if PARQUET_DIR.is_dir():
        parquet_glob = str(PARQUET_DIR / "*.parquet")
    else:
        parquet_glob = str(PARQUET_DIR)
    
    BIG = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}')").df()
    BIG["date"] = pd.to_datetime(BIG["date"])
    
    CAT_COLS = ['primary_cvss_sev', 'dominant_event_type']
    
    # Create temporal splits
    days = np.sort(BIG["date"].unique())
    VAL_CUT = pd.to_datetime(days[int(0.64 * len(days))])
    BIG["flag_train"] = (BIG["date"] < VAL_CUT).astype("uint8")
    train_mask = BIG["flag_train"] == 1
    
    print(f"Dataset shape: {BIG.shape}")
    print(f"Training rows: {train_mask.sum():,}")
    print()
    
    # Test each categorical column for literal "UNK" strings
    for col in CAT_COLS:
        print(f"=" * 60)
        print(f"TESTING COLUMN: {col}")
        print(f"=" * 60)
        
        # Check for literal "UNK" strings in the data
        unk_mask = BIG[col] == "UNK"
        unk_count = unk_mask.sum()
        
        print(f"Rows with literal 'UNK' string: {unk_count:,}")
        
        if unk_count > 0:
            print(f"❌ FOUND LITERAL 'UNK' STRINGS!")
            print(f"Sample rows with 'UNK':")
            unk_samples = BIG[unk_mask][["cve", "date", col]].head(5)
            print(unk_samples)
            print()
            
            # Check if these are in training data
            unk_in_train = (unk_mask & train_mask).sum()
            print(f"'UNK' strings in training data: {unk_in_train:,}")
            print()
        else:
            print(f"✅ No literal 'UNK' strings found")
            print()
        
        # Simulate the vocabulary creation process
        print("Simulating vocabulary creation:")
        cats = BIG.loc[train_mask, col].dropna().unique()
        print(f"Unique values in training: {sorted(cats)}")
        
        # Check if "UNK" is in the training categories
        if "UNK" in cats:
            print(f"❌ 'UNK' found in training categories!")
            
            # Simulate the buggy vocabulary creation
            print("\nBuggy vocabulary creation:")
            VOCAB_BUGGY = {"UNK": 0}
            for i, c in enumerate(sorted(cats)):
                VOCAB_BUGGY[c] = i + 1
                print(f"  Step {i+1}: {c} → {i+1}")
            
            print(f"Final buggy vocabulary: {VOCAB_BUGGY}")
            print(f"Vocabulary size: {len(VOCAB_BUGGY)}")
            print(f"Max index: {max(VOCAB_BUGGY.values())}")
            print(f"Index bounds violation: {max(VOCAB_BUGGY.values()) >= len(VOCAB_BUGGY)}")
            print()
            
            # Simulate the fixed vocabulary creation
            print("Fixed vocabulary creation:")
            cats_filtered = [c for c in cats if c != "UNK"]
            VOCAB_FIXED = {c: i + 1 for i, c in enumerate(sorted(cats_filtered))}
            VOCAB_FIXED["UNK"] = 0
            
            print(f"Filtered categories: {sorted(cats_filtered)}")
            print(f"Fixed vocabulary: {VOCAB_FIXED}")
            print(f"Vocabulary size: {len(VOCAB_FIXED)}")
            print(f"Max index: {max(VOCAB_FIXED.values())}")
            print(f"Index bounds check: {max(VOCAB_FIXED.values()) < len(VOCAB_FIXED)}")
            print()
            
        else:
            print(f"✅ No 'UNK' in training categories")
            
            # Standard vocabulary creation
            VOCAB = {"UNK": 0, **{c: i + 1 for i, c in enumerate(sorted(cats))}}
            print(f"Standard vocabulary: {VOCAB}")
            print(f"Vocabulary size: {len(VOCAB)}")
            print(f"Max index: {max(VOCAB.values())}")
            print()
        
        print()

if __name__ == "__main__":
    test_unk_collision() 