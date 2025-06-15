#!/usr/bin/env python3
"""
Final debugging script using exact LSTM vocabulary and full dataset
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import duckdb
from pathlib import Path
import json

def final_debug():
    print("=" * 80)
    print("FINAL DEBUG - USING EXACT LSTM VOCABULARY")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the exact vocabulary from the LSTM script
    with open("vocab.json", "r") as f:
        VOCAB = json.load(f)
    
    print("Loaded vocabulary:")
    for col, vocab in VOCAB.items():
        print(f"  {col}: {vocab} (size: {len(vocab)})")
    
    # Load full dataset
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    parquet_glob = str(PARQUET_DIR / "*.parquet") if PARQUET_DIR.is_dir() else str(PARQUET_DIR)
    
    print(f"\nLoading FULL dataset from: {parquet_glob}")
    BIG = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}')").df()
    print(f"Loaded {len(BIG):,} rows")
    
    CAT_COLS = list(VOCAB.keys())
    print(f"Categorical columns: {CAT_COLS}")
    
    # Apply exact same mapping as LSTM script
    print("\nApplying vocabulary mapping...")
    for col in CAT_COLS:
        print(f"\nProcessing {col}:")
        
        # Check original data
        original_unique = BIG[col].dropna().unique()
        print(f"  Original unique values: {len(original_unique)}")
        print(f"  Sample values: {list(original_unique)[:10]}")
        
        # Apply mapping exactly as LSTM script
        original_values = BIG[col].copy()
        BIG[col] = BIG[col].map(VOCAB[col]).fillna(0).astype("int32")
        
        # Check for unmapped values (these become 0 = UNK)
        unmapped_mask = original_values.notna() & BIG[col].eq(0)
        unmapped_values = original_values[unmapped_mask].unique()
        if len(unmapped_values) > 0:
            print(f"  ⚠️  Unmapped values (became UNK): {list(unmapped_values)}")
        
        # Check final mapping
        mapped_min = BIG[col].min()
        mapped_max = BIG[col].max()
        vocab_size = len(VOCAB[col])
        
        print(f"  After mapping - Min: {mapped_min}, Max: {mapped_max}")
        print(f"  Vocabulary size: {vocab_size}")
        
        # Critical check: out-of-bounds values
        oob_mask = BIG[col] >= vocab_size
        oob_count = oob_mask.sum()
        
        if oob_count > 0:
            print(f"  🚨 CRITICAL: {oob_count} OUT-OF-BOUNDS VALUES!")
            oob_values = BIG[col][oob_mask].unique()
            print(f"  OOB indices: {oob_values}")
            
            # Find original values that caused this
            oob_original = original_values[oob_mask].unique()
            print(f"  Original values that caused OOB: {oob_original}")
            
            # This is the smoking gun!
            print(f"  🔥 FOUND THE BUG! These values will cause CUDA assertion failure!")
        else:
            print(f"  ✅ All values within bounds [0, {vocab_size-1}]")
    
    # Test with embeddings using exact vocabulary sizes
    print("\n" + "=" * 40)
    print("TESTING EMBEDDINGS WITH EXACT VOCAB SIZES")
    print("=" * 40)
    
    cat_sizes = [len(VOCAB[col]) for col in CAT_COLS]
    print(f"Categorical sizes: {cat_sizes}")
    
    embeddings = nn.ModuleList([
        nn.Embedding(size, 16, padding_idx=0) for size in cat_sizes
    ]).to(device)
    
    # Test with problematic data if found
    print(f"\nTesting with sample data...")
    
    # Take a larger sample to increase chance of hitting problematic values
    sample_size = min(10000, len(BIG))
    cat_data = BIG[CAT_COLS].iloc[:sample_size].values
    cat_tensor = torch.tensor(cat_data, dtype=torch.long).to(device)
    
    print(f"Testing tensor shape: {cat_tensor.shape}")
    
    for i, (col, emb) in enumerate(zip(CAT_COLS, embeddings)):
        col_data = cat_tensor[:, i]
        vocab_size = cat_sizes[i]
        
        print(f"\nTesting embedding {i} ({col}):")
        print(f"  Input range: [{col_data.min().item()}, {col_data.max().item()}]")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding num_embeddings: {emb.num_embeddings}")
        
        # The critical check that fails in CUDA
        oob_mask = col_data >= vocab_size
        oob_count = oob_mask.sum().item()
        
        if oob_count > 0:
            print(f"  🚨 ASSERTION FAILURE: {oob_count} values >= {vocab_size}")
            oob_values = col_data[oob_mask].unique()
            print(f"  Failing values: {oob_values}")
            print(f"  This is exactly what causes: assert (cat[..., i] < vocab_size).all()")
            
            # Show the exact assertion that would fail
            print(f"  Assertion check: (cat[..., {i}] < {vocab_size}).all() = {(col_data < vocab_size).all()}")
            
            return  # Found the issue!
        else:
            print(f"  ✅ All values within bounds")
        
        # Try the embedding
        try:
            result = emb(col_data)
            print(f"  ✅ Embedding successful: {col_data.shape} -> {result.shape}")
        except Exception as e:
            print(f"  🚨 EMBEDDING FAILED: {e}")
            print(f"  This is the CUDA assertion error!")
            return
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("If no out-of-bounds values were found, the issue might be:")
    print("1. In a different part of the dataset not sampled")
    print("2. During batch collation or data loading")
    print("3. In the actual training loop with different data splits")
    print("=" * 80)

if __name__ == "__main__":
    final_debug() 