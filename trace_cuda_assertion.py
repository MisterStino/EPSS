#!/usr/bin/env python3
"""
Comprehensive trace of CUDA assertion failure in LSTM training.
This script will follow the exact data flow to identify where out-of-bounds indices appear.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import duckdb
from pathlib import Path
from tqdm import tqdm

# Replicate exact preprocessing from main script
def trace_cuda_assertion():
    print("=" * 80)
    print("COMPREHENSIVE CUDA ASSERTION TRACE")
    print("=" * 80)
    
    # 1. Load data exactly as main script
    print("\n[STEP 1] Loading data...")
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    if PARQUET_DIR.is_dir():
        parquet_glob = str(PARQUET_DIR / "*.parquet")
    else:
        parquet_glob = str(PARQUET_DIR)
    
    BIG = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}')").df()
    BIG["date"] = pd.to_datetime(BIG["date"])
    print(f"✓ Loaded {len(BIG):,} rows")
    
    # 2. Create temporal splits exactly as main script
    print("\n[STEP 2] Creating temporal splits...")
    days = np.sort(BIG["date"].unique())
    VAL_CUT = pd.to_datetime(days[int(0.64 * len(days))])
    TEST_CUT = pd.to_datetime(days[int(0.80 * len(days))])
    
    BIG["flag_train"] = (BIG["date"] < VAL_CUT).astype("uint8")
    BIG["flag_val"] = ((BIG["date"] >= VAL_CUT) & (BIG["date"] < TEST_CUT)).astype("uint8")
    BIG["flag_test"] = (BIG["date"] >= TEST_CUT).astype("uint8")
    
    train_mask = BIG["flag_train"] == 1
    val_mask = BIG["flag_val"] == 1
    test_mask = BIG["flag_test"] == 1
    
    print(f"✓ Train: {train_mask.sum():,}, Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")
    
    # 3. Create vocabularies from training data only
    print("\n[STEP 3] Creating vocabularies...")
    CAT_COLS = ['primary_cvss_sev', 'dominant_event_type']
    VOCAB = {}
    
    for col in CAT_COLS:
        cats = BIG.loc[train_mask, col].dropna().unique()
        VOCAB[col] = {"UNK": 0, **{c: i + 1 for i, c in enumerate(sorted(cats))}}
        print(f"  {col}: {len(VOCAB[col])} categories -> {VOCAB[col]}")
    
    # 4. Apply vocabulary mapping
    print("\n[STEP 4] Applying vocabulary mapping...")
    for col in CAT_COLS:
        original_values = BIG[col].copy()
        BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")
        
        # Check for any mapping issues
        mapped_values = BIG[col]
        max_val = mapped_values.max()
        vocab_size = len(VOCAB[col])
        
        print(f"  {col}: max_mapped={max_val}, vocab_size={vocab_size}")
        if max_val >= vocab_size:
            print(f"    ❌ MAPPING ERROR: max_val {max_val} >= vocab_size {vocab_size}")
            bad_mask = mapped_values >= vocab_size
            print(f"    Bad count: {bad_mask.sum()}")
            print(f"    Bad values: {sorted(mapped_values[bad_mask].unique())}")
        else:
            print(f"    ✅ Mapping OK")
    
    # 5. Create simplified dataset class for tracing
    print("\n[STEP 5] Creating traced dataset...")
    
    class TracedCVEDataset:
        def __init__(self, df, L_max, horizon, flag_col, cat_cols):
            print(f"    Initializing dataset with {len(df):,} rows")
            self.cat_data_list = []
            self.cve_names = []
            
            # Extract categorical data
            if cat_cols:
                cat_data = df[cat_cols].to_numpy("int64").astype("int64")
            else:
                cat_data = np.empty((len(df), 0), dtype="int64")
            
            print(f"    Categorical data shape: {cat_data.shape}")
            print(f"    Categorical data dtype: {cat_data.dtype}")
            
            # Process each CVE sequence
            cve_groups = list(df.groupby("cve", sort=False).groups.items())
            print(f"    Processing {len(cve_groups)} CVE sequences...")
            
            for cve_idx, (cve, idx) in enumerate(cve_groups[:5]):  # Only first 5 for detailed trace
                idx = np.asarray(idx)
                T = len(idx)
                pad = L_max - T
                
                print(f"\n    CVE {cve_idx + 1}: {cve}")
                print(f"      Sequence length: {T}, Padding: {pad}")
                
                # Extract categorical data for this CVE
                cve_cat_data = cat_data[idx]
                print(f"      Pre-padding shape: {cve_cat_data.shape}")
                
                # Check bounds before padding
                for col_idx, col in enumerate(cat_cols):
                    col_data = cve_cat_data[:, col_idx]
                    max_val = col_data.max()
                    min_val = col_data.min()
                    vocab_size = len(VOCAB[col])
                    print(f"        {col}: min={min_val}, max={max_val}, vocab_size={vocab_size}")
                    
                    if max_val >= vocab_size:
                        print(f"          ❌ PRE-PADDING BUG!")
                        bad_indices = np.where(col_data >= vocab_size)[0]
                        print(f"          Bad sequence positions: {bad_indices}")
                        print(f"          Bad values: {col_data[bad_indices]}")
                        
                        # Map back to original dataframe
                        original_rows = idx[bad_indices]
                        print(f"          Original DF rows: {original_rows}")
                        original_dates = df.loc[original_rows, 'date'].values
                        print(f"          Dates: {original_dates}")
                
                # Apply padding
                padded_cat_data = np.pad(cve_cat_data, ((0, pad), (0, 0)))
                print(f"      Post-padding shape: {padded_cat_data.shape}")
                
                # Check bounds after padding
                for col_idx, col in enumerate(cat_cols):
                    col_data = padded_cat_data[:, col_idx]
                    max_val = col_data.max()
                    vocab_size = len(VOCAB[col])
                    
                    if max_val >= vocab_size:
                        print(f"        ❌ POST-PADDING BUG in {col}!")
                        bad_positions = np.where(col_data >= vocab_size)[0]
                        print(f"          Bad positions: {bad_positions}")
                        print(f"          Values at bad positions: {col_data[bad_positions]}")
                        
                        # Check if bad positions are in padding region
                        padding_positions = bad_positions[bad_positions >= T]
                        original_positions = bad_positions[bad_positions < T]
                        
                        if len(padding_positions) > 0:
                            print(f"          ❌ PADDING INTRODUCED BAD VALUES at positions: {padding_positions}")
                        if len(original_positions) > 0:
                            print(f"          ❌ ORIGINAL DATA HAD BAD VALUES at positions: {original_positions}")
                
                # Convert to tensor
                tensor_cat_data = torch.from_numpy(padded_cat_data)
                self.cat_data_list.append(tensor_cat_data)
                self.cve_names.append(cve)
                
                print(f"      Final tensor shape: {tensor_cat_data.shape}")
                print(f"      Final tensor dtype: {tensor_cat_data.dtype}")
                
                # Final tensor bounds check
                for col_idx, col in enumerate(cat_cols):
                    col_tensor = tensor_cat_data[:, col_idx]
                    max_val = col_tensor.max().item()
                    vocab_size = len(VOCAB[col])
                    
                    if max_val >= vocab_size:
                        print(f"        ❌ FINAL TENSOR BUG in {col}!")
                        print(f"          Max value: {max_val}, Vocab size: {vocab_size}")
        
        def __len__(self):
            return len(self.cat_data_list)
        
        def __getitem__(self, i):
            return self.cat_data_list[i], self.cve_names[i]
    
    # 6. Create traced dataset
    L_max = BIG.groupby("cve", observed=True).size().max()
    print(f"    Max sequence length: {L_max}")
    
    traced_dataset = TracedCVEDataset(BIG, L_max, 30, "flag_train", CAT_COLS)
    
    # 7. Test batch collation
    print("\n[STEP 6] Testing batch collation...")
    
    def traced_collate(batch):
        print(f"    Collating batch of {len(batch)} items")
        cat_tensors = [item[0] for item in batch]
        cve_names = [item[1] for item in batch]
        
        print(f"    Individual tensor shapes: {[t.shape for t in cat_tensors]}")
        
        # Check each tensor before stacking
        for i, (cat_tensor, cve_name) in enumerate(zip(cat_tensors, cve_names)):
            print(f"      Item {i} ({cve_name}):")
            for col_idx, col in enumerate(CAT_COLS):
                col_data = cat_tensor[:, col_idx]
                max_val = col_data.max().item()
                vocab_size = len(VOCAB[col])
                print(f"        {col}: max={max_val}, vocab_size={vocab_size}")
                
                if max_val >= vocab_size:
                    print(f"          ❌ PRE-STACK BUG!")
        
        # Stack tensors
        try:
            stacked = torch.stack(cat_tensors, 0)
            print(f"    Stacked tensor shape: {stacked.shape}")
            
            # Check stacked tensor
            for col_idx, col in enumerate(CAT_COLS):
                col_data = stacked[:, :, col_idx]
                max_val = col_data.max().item()
                vocab_size = len(VOCAB[col])
                print(f"      Stacked {col}: max={max_val}, vocab_size={vocab_size}")
                
                if max_val >= vocab_size:
                    print(f"        ❌ POST-STACK BUG!")
                    
                    # Find which batch item and position has the problem
                    bad_mask = col_data >= vocab_size
                    bad_positions = torch.where(bad_mask)
                    print(f"        Bad positions (batch, seq): {list(zip(bad_positions[0].tolist(), bad_positions[1].tolist()))}")
                    print(f"        Bad values: {col_data[bad_mask].tolist()}")
            
            return stacked
            
        except Exception as e:
            print(f"    ❌ STACKING ERROR: {e}")
            return None
    
    # Create dataloader with traced collation
    dataloader = DataLoader(traced_dataset, batch_size=2, shuffle=False, collate_fn=traced_collate)
    
    # 8. Test one batch
    print("\n[STEP 7] Testing dataloader batch...")
    try:
        batch = next(iter(dataloader))
        if batch is not None:
            print(f"    ✓ Batch created successfully: {batch.shape}")
            
            # 9. Test model forward pass simulation
            print("\n[STEP 8] Testing model forward pass simulation...")
            
            # Create minimal embedding layers
            cat_sizes = [len(VOCAB[col]) for col in CAT_COLS]
            print(f"    Embedding sizes: {cat_sizes}")
            
            embeddings = nn.ModuleList([nn.Embedding(s, 8, padding_idx=0) for s in cat_sizes])
            
            # Test embedding lookup
            cat_tensor = batch.long()
            print(f"    Input tensor shape: {cat_tensor.shape}")
            print(f"    Input tensor dtype: {cat_tensor.dtype}")
            
            # Check bounds one more time before embedding
            for col_idx, col in enumerate(CAT_COLS):
                col_data = cat_tensor[:, :, col_idx]
                max_val = col_data.max().item()
                vocab_size = len(VOCAB[col])
                print(f"      Pre-embedding {col}: max={max_val}, vocab_size={vocab_size}")
                
                if max_val >= vocab_size:
                    print(f"        ❌ PRE-EMBEDDING BUG!")
                    
                    # This is where the assertion would fail
                    bad_mask = col_data >= vocab_size
                    bad_positions = torch.where(bad_mask)
                    bad_values = col_data[bad_mask]
                    
                    print(f"        CRITICAL: Found {len(bad_values)} out-of-bounds indices!")
                    print(f"        Bad positions (batch, seq): {list(zip(bad_positions[0].tolist(), bad_positions[1].tolist()))}")
                    print(f"        Bad values: {bad_values.tolist()}")
                    print(f"        Expected range: [0, {vocab_size-1}]")
                    
                    # This would be the exact assertion that fails
                    print(f"\n        🔥 THIS IS THE ASSERTION THAT FAILS:")
                    print(f"        assert (cat[..., {col_idx}] < {vocab_size}).all()")
                    print(f"        Actual max value: {max_val}")
                    
                    return  # Stop here - we found the bug!
            
            # If we get here, try the actual embedding
            try:
                embedded = []
                for col_idx, emb in enumerate(embeddings):
                    col_data = cat_tensor[:, :, col_idx]
                    embedded_col = emb(col_data)
                    embedded.append(embedded_col)
                    print(f"      ✓ Embedded {CAT_COLS[col_idx]}: {embedded_col.shape}")
                
                print(f"    ✓ All embeddings successful!")
                
            except Exception as e:
                print(f"    ❌ EMBEDDING ERROR: {e}")
                print(f"    This is likely the CUDA assertion failure!")
        
    except Exception as e:
        print(f"    ❌ DATALOADER ERROR: {e}")
    
    print("\n" + "=" * 80)
    print("TRACE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    trace_cuda_assertion() 