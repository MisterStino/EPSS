#!/usr/bin/env python3
"""
Verify the UNK string collision theory
"""

import pandas as pd
import duckdb
from pathlib import Path
import json

def verify_unk_collision():
    print("=" * 80)
    print("VERIFYING UNK STRING COLLISION THEORY")
    print("=" * 80)
    
    # Load data
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    parquet_glob = str(PARQUET_DIR / "*.parquet") if PARQUET_DIR.is_dir() else str(PARQUET_DIR)
    
    print(f"Loading data from: {parquet_glob}")
    BIG = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}')").df()
    print(f"Loaded {len(BIG):,} rows")
    
    # Check the categorical columns from the LSTM script
    CAT_COLS = [
        "primary_cvss_ver", "primary_cvss_sev", "dominant_event_type",
        "prev_event_type", "primary_source", "cwe_id",
        "vuln_status", "source_identifier",
    ]
    
    # Filter to existing columns
    CAT_COLS = [c for c in CAT_COLS if c in BIG.columns]
    print(f"Available categorical columns: {CAT_COLS}")
    
    # Check each column for literal "UNK" strings
    for col in CAT_COLS:
        print(f"\n--- Analyzing column: {col} ---")
        
        # Get all unique values
        unique_vals = BIG[col].dropna().unique()
        print(f"Total unique values: {len(unique_vals)}")
        print(f"Sample values: {list(unique_vals)[:10]}")
        
        # Check for literal "UNK" string
        has_unk = "UNK" in unique_vals
        print(f"Contains literal 'UNK' string: {has_unk}")
        
        if has_unk:
            unk_count = (BIG[col] == "UNK").sum()
            print(f"🚨 FOUND LITERAL 'UNK'! Count: {unk_count}")
            print(f"This will cause dictionary key collision!")
            
            # Simulate the vocabulary creation bug
            print(f"\nSimulating vocabulary creation:")
            cats = unique_vals
            
            # Step 1: Start with UNK: 0
            vocab_step1 = {"UNK": 0}
            print(f"Step 1 - Initial: {vocab_step1}")
            
            # Step 2: Add enumerated values (including UNK from data)
            enumerated = {c: i + 1 for i, c in enumerate(sorted(cats))}
            print(f"Step 2 - Enumerated: {enumerated}")
            
            # Step 3: Merge (UNK gets overwritten!)
            vocab_final = {"UNK": 0, **enumerated}
            print(f"Step 3 - Final (COLLISION!): {vocab_final}")
            
            # Show the problem
            vocab_size = len(vocab_final)
            max_index = max(vocab_final.values())
            print(f"\nThe Problem:")
            print(f"  Vocabulary size (embedding rows): {vocab_size}")
            print(f"  Maximum index in data: {max_index}")
            print(f"  Index {max_index} >= {vocab_size}? {max_index >= vocab_size}")
            
            if max_index >= vocab_size:
                print(f"🔥 CONFIRMED: This causes CUDA assertion failure!")
                print(f"   Embedding has rows [0, {vocab_size-1}] but data contains index {max_index}")
        else:
            print(f"✅ No literal 'UNK' found in this column")
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    verify_unk_collision() 