#!/usr/bin/env python3
"""
Reproduce the exact CUDA assertion issue
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import duckdb
from pathlib import Path

def reproduce_exact_issue():
    print("=" * 80)
    print("REPRODUCING EXACT CUDA ASSERTION ISSUE")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    parquet_glob = str(PARQUET_DIR / "*.parquet") if PARQUET_DIR.is_dir() else str(PARQUET_DIR)
    
    print(f"\nLoading data from: {parquet_glob}")
    BIG = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}') LIMIT 5000").df()
    print(f"Loaded {len(BIG)} rows")
    
    # Check available categorical columns
    CAT_COLS = ["primary_cvss_sev", "dominant_event_type"]
    CAT_COLS = [c for c in CAT_COLS if c in BIG.columns]
    print(f"Available categorical columns: {CAT_COLS}")
    
    if not CAT_COLS:
        print("No categorical columns found!")
        return
    
    # Create vocabularies
    VOCAB = {}
    for col in CAT_COLS:
        print(f"\nProcessing {col}:")
        unique_vals = BIG[col].dropna().unique()
        print(f"  Unique values: {list(unique_vals)}")
        
        vocab = {"UNK": 0}
        vocab.update({val: i+1 for i, val in enumerate(sorted(unique_vals))})
        VOCAB[col] = vocab
        
        print(f"  Vocabulary: {vocab}")
        print(f"  Vocab size: {len(vocab)}")
        
        # Map to indices
        BIG[col] = BIG[col].map(vocab).fillna(0).astype("int32")
        print(f"  After mapping - range: [{BIG[col].min()}, {BIG[col].max()}]")
    
    # Create embeddings
    cat_sizes = [len(VOCAB[col]) for col in CAT_COLS]
    print(f"\nCategorical sizes: {cat_sizes}")
    
    embeddings = nn.ModuleList([
        nn.Embedding(size, 16, padding_idx=0) for size in cat_sizes
    ]).to(device)
    
    # Test with data
    print(f"\nTesting embeddings...")
    cat_data = BIG[CAT_COLS].iloc[:100].values
    cat_tensor = torch.tensor(cat_data, dtype=torch.long).to(device)
    
    print(f"Categorical tensor shape: {cat_tensor.shape}")
    
    for i, (col, emb) in enumerate(zip(CAT_COLS, embeddings)):
        col_data = cat_tensor[:, i]
        vocab_size = cat_sizes[i]
        
        print(f"\nColumn {i} ({col}):")
        print(f"  Range: [{col_data.min().item()}, {col_data.max().item()}]")
        print(f"  Vocab size: {vocab_size}")
        
        # Check bounds
        oob_mask = col_data >= vocab_size
        if oob_mask.any():
            print(f"  🚨 OUT OF BOUNDS: {oob_mask.sum()} values")
            print(f"  OOB values: {col_data[oob_mask].unique()}")
        else:
            print(f"  ✅ All values in bounds")
        
        # Try embedding
        try:
            result = emb(col_data)
            print(f"  ✅ Embedding successful: {result.shape}")
        except Exception as e:
            print(f"  🚨 EMBEDDING FAILED: {e}")

if __name__ == "__main__":
    reproduce_exact_issue() 