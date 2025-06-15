#!/usr/bin/env python3
"""
Find the exact CVE sequence that contains out-of-bounds categorical indices.
This uses the exact preprocessing from the main LSTM script.
"""

import numpy as np
import pandas as pd
import torch
import duckdb
from pathlib import Path
import json

def find_problematic_cve():
    print("=" * 80)
    print("FINDING PROBLEMATIC CVE WITH OUT-OF-BOUNDS INDICES")
    print("=" * 80)
    
    # 1. Load data exactly as main script
    print("\n[STEP 1] Loading data...")
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    if PARQUET_DIR.is_dir():
        parquet_glob = str(PARQUET_DIR / "*.parquet")
    else:
        parquet_glob = str(PARQUET_DIR)
    
    # Replicate exact loading from main script
    DROP_COLS = [
        "cve_date_key", "original_date",
        "reconstruction_timestamp", "reconstruction_timestamp_raw",
        "details_combined", "details_longest", "event_data_merged",
        "description_all", "description_en",
        "primary_cvss_vec", "cve_tags", "reference_count",
    ]
    
    schema_df = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}') LIMIT 0").df()
    probe_cols = schema_df.columns.tolist()
    
    drop_real = [c for c in DROP_COLS if c in probe_cols]
    quote_cols = ", ".join(f'"{c}"' for c in drop_real)
    exclude = f" EXCLUDE ({quote_cols})" if drop_real else ""
    
    BIG = duckdb.sql(f"SELECT *{exclude} FROM parquet_scan('{parquet_glob}')").df()
    BIG["date"] = pd.to_datetime(BIG["date"])
    print(f"✓ Loaded {len(BIG):,} rows × {BIG.shape[1]} columns")
    
    # 2. Apply exact preprocessing from main script
    print("\n[STEP 2] Applying exact preprocessing...")
    
    # Transform EPSS
    def transform_epss(arr, mode="logit", eps=1e-6):
        p = np.clip(arr.astype("float64"), eps, 1.0 - eps)
        if mode == "logit":
            out = np.log(p / (1.0 - p))
        return out.astype("float32")
    
    BIG["epss"] = transform_epss(BIG["epss"].values, mode="logit", eps=1e-6)
    
    # Create temporal splits exactly as main script
    days = np.sort(BIG["date"].unique())
    VAL_CUT = pd.to_datetime(days[int(0.64 * len(days))])
    TEST_CUT = pd.to_datetime(days[int(0.80 * len(days))])
    
    BIG["flag_train"] = (BIG["date"] < VAL_CUT).astype("uint8")
    BIG["flag_val"] = ((BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT)).astype("uint8")
    BIG["flag_test"] = (BIG["date"] >= TEST_CUT).astype("uint8")
    
    train_mask = BIG["flag_train"] == 1
    print(f"✓ Train split: {train_mask.sum():,} rows")
    
    # Filter categorical columns to only existing ones
    CAT_COLS_ALL = [
        "primary_cvss_ver", "primary_cvss_sev", "dominant_event_type",
        "prev_event_type", "primary_source", "cwe_id",
        "vuln_status", "source_identifier",
    ]
    CAT_COLS = [c for c in CAT_COLS_ALL if c in BIG.columns]
    print(f"✓ Available categorical columns: {CAT_COLS}")
    
    # Handle boolean columns exactly as main script
    BOOL_COLS = lambda df: [c for c in df.columns
                           if c.startswith(("has_", "is_"))
                           or c in ("same_day_multi_source",
                                   "has_v2","has_v30","has_v31","has_v40")]
    
    for col in BOOL_COLS(BIG):
        if pd.api.types.is_bool_dtype(BIG[col]):
            BIG[col] = BIG[col].fillna(False).astype("uint8")
        else:
            BIG[col] = BIG[col].fillna(0).astype("uint8")
    
    # Create vocabularies from training data only
    print("\n[STEP 3] Creating vocabularies...")
    VOCAB = {}
    for col in CAT_COLS:
        cats = BIG.loc[train_mask, col].dropna().unique()
        VOCAB[col] = {"UNK": 0, **{c: i + 1 for i, c in enumerate(sorted(cats))}}
        print(f"  {col}: {len(VOCAB[col])} categories")
        print(f"    Vocab: {VOCAB[col]}")
    
    # Apply vocabulary mapping
    print("\n[STEP 4] Applying vocabulary mapping...")
    for col in CAT_COLS:
        original_col = BIG[col].copy()
        BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")
        
        max_val = BIG[col].max()
        vocab_size = len(VOCAB[col])
        print(f"  {col}: max_mapped={max_val}, vocab_size={vocab_size}")
        
        if max_val >= vocab_size:
            print(f"    ❌ GLOBAL MAPPING ERROR!")
            bad_mask = BIG[col] >= vocab_size
            bad_count = bad_mask.sum()
            print(f"    Bad values count: {bad_count}")
            print(f"    Bad values: {sorted(BIG[col][bad_mask].unique())}")
            
            # Find which original values caused this
            bad_rows = BIG[bad_mask]
            print(f"    Sample bad rows:")
            for i, (_, row) in enumerate(bad_rows.head(5).iterrows()):
                orig_val = original_col.loc[row.name]
                mapped_val = row[col]
                print(f"      Row {row.name}: '{orig_val}' -> {mapped_val} (vocab_size={vocab_size})")
            
            return  # Stop here if global mapping is wrong
    
    # 5. Scan through CVE sequences to find the problematic one
    print("\n[STEP 5] Scanning CVE sequences for out-of-bounds indices...")
    
    cve_groups = list(BIG.groupby("cve", sort=False).groups.items())
    print(f"Total CVE sequences to check: {len(cve_groups)}")
    
    problematic_cves = []
    
    for cve_idx, (cve, idx) in enumerate(cve_groups):
        if cve_idx % 1000 == 0:
            print(f"  Checked {cve_idx:,} CVEs...")
        
        idx = np.asarray(idx)
        
        # Extract categorical data for this CVE
        if CAT_COLS:
            cve_cat_data = BIG.loc[idx, CAT_COLS].to_numpy("int64")
            
            # Check each categorical column
            for col_idx, col in enumerate(CAT_COLS):
                col_data = cve_cat_data[:, col_idx]
                max_val = col_data.max()
                vocab_size = len(VOCAB[col])
                
                if max_val >= vocab_size:
                    print(f"\n    ❌ FOUND PROBLEMATIC CVE: {cve}")
                    print(f"      CVE index: {cve_idx}")
                    print(f"      Sequence length: {len(idx)}")
                    print(f"      Column: {col} (index {col_idx})")
                    print(f"      Max value: {max_val}, Vocab size: {vocab_size}")
                    
                    # Find bad positions
                    bad_positions = np.where(col_data >= vocab_size)[0]
                    bad_values = col_data[bad_positions]
                    print(f"      Bad positions in sequence: {bad_positions[:10]}")
                    print(f"      Bad values: {bad_values[:10]}")
                    
                    # Map back to original dataframe
                    original_rows = idx[bad_positions]
                    print(f"      Original DF rows: {original_rows[:5]}")
                    
                    # Get temporal split info
                    sample_row = original_rows[0]
                    train_flag = BIG.loc[sample_row, 'flag_train']
                    val_flag = BIG.loc[sample_row, 'flag_val']
                    test_flag = BIG.loc[sample_row, 'flag_test']
                    date = BIG.loc[sample_row, 'date']
                    
                    print(f"      Sample row {sample_row}:")
                    print(f"        Date: {date}")
                    print(f"        Flags: train={train_flag}, val={val_flag}, test={test_flag}")
                    
                    # Check if this is in validation/test data
                    if val_flag == 1:
                        print(f"        🔍 This is VALIDATION data!")
                    elif test_flag == 1:
                        print(f"        🔍 This is TEST data!")
                    else:
                        print(f"        🔍 This is TRAINING data!")
                    
                    # Get original values that caused the problem
                    print(f"      Original values analysis:")
                    for i, bad_row in enumerate(original_rows[:5]):
                        original_val = BIG.loc[bad_row, col]  # This is already mapped
                        print(f"        Row {bad_row}: mapped_value={original_val}")
                    
                    problematic_cves.append({
                        'cve': cve,
                        'cve_idx': cve_idx,
                        'column': col,
                        'max_val': max_val,
                        'vocab_size': vocab_size,
                        'bad_positions': bad_positions.tolist(),
                        'bad_values': bad_values.tolist(),
                        'sample_date': str(date),
                        'is_train': train_flag == 1,
                        'is_val': val_flag == 1,
                        'is_test': test_flag == 1
                    })
                    
                    # Stop after finding first few problematic CVEs
                    if len(problematic_cves) >= 3:
                        break
        
        if len(problematic_cves) >= 3:
            break
    
    print(f"\n[STEP 6] Summary of findings...")
    if problematic_cves:
        print(f"Found {len(problematic_cves)} problematic CVE sequences:")
        for i, prob in enumerate(problematic_cves):
            print(f"\n  Problem {i+1}:")
            print(f"    CVE: {prob['cve']}")
            print(f"    Column: {prob['column']}")
            print(f"    Max value: {prob['max_val']}, Vocab size: {prob['vocab_size']}")
            print(f"    Split: {'TRAIN' if prob['is_train'] else 'VAL' if prob['is_val'] else 'TEST'}")
            print(f"    Date: {prob['sample_date']}")
        
        # Save results
        with open('problematic_cves.json', 'w') as f:
            json.dump(problematic_cves, f, indent=2)
        print(f"\n✓ Saved detailed results to 'problematic_cves.json'")
        
    else:
        print("✅ No problematic CVE sequences found!")
        print("This suggests the issue might be in:")
        print("  1. Batch collation process")
        print("  2. Device transfer (CPU -> GPU)")
        print("  3. Model forward pass tensor operations")
        print("  4. Different data loading in actual training")
    
    print("\n" + "=" * 80)
    print("CVE SCAN COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    find_problematic_cve() 