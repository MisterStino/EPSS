#!/usr/bin/env python3
"""
Inspect the actual data structure to understand column types and content
"""

import pandas as pd
import duckdb
from pathlib import Path

def inspect_data_structure():
    print("=" * 80)
    print("DATA STRUCTURE INSPECTION")
    print("=" * 80)
    
    # Load data
    PARQUET_DIR = Path("data/full_db/v1/data/minimal_v1_timeseries_sample.parquet")
    if PARQUET_DIR.is_dir():
        parquet_glob = str(PARQUET_DIR / "*.parquet")
    else:
        parquet_glob = str(PARQUET_DIR)
    
    print(f"Loading from: {parquet_glob}")
    
    try:
        df = duckdb.execute(f"""
            SELECT * FROM read_parquet('{parquet_glob}')
            LIMIT 1000
        """).df()
        print(f"Loaded sample of {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print("\n[COLUMN ANALYSIS]")
    print(f"Total columns: {len(df.columns)}")
    print(f"Column names: {list(df.columns)}")
    
    print("\n[DATA TYPES]")
    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].notna().sum()
        unique_vals = df[col].nunique()
        print(f"  {col:25} | {str(dtype):15} | Non-null: {non_null:4} | Unique: {unique_vals:4}")
        
        # Show sample values for categorical-looking columns
        if unique_vals <= 10 or col in ['dominant_event_type', 'primary_cwe_category']:
            sample_vals = df[col].dropna().unique()[:5]
            print(f"    Sample values: {list(sample_vals)}")
    
    print("\n[CATEGORICAL COLUMN ANALYSIS]")
    categorical_candidates = ['dominant_event_type', 'primary_cwe_category']
    
    for col in categorical_candidates:
        if col in df.columns:
            print(f"\n{col}:")
            value_counts = df[col].value_counts()
            print(f"  Unique values: {len(value_counts)}")
            print(f"  Value counts:")
            for val, count in value_counts.head(10).items():
                print(f"    {val}: {count}")
            
            # Check for missing values
            missing = df[col].isna().sum()
            print(f"  Missing values: {missing}")
    
    print("\n[NUMERICAL COLUMN ANALYSIS]")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"Numerical columns: {list(numerical_cols)}")
    
    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  Min: {df[col].min()}")
        print(f"  Max: {df[col].max()}")
        print(f"  Mean: {df[col].mean():.3f}")
        print(f"  Missing: {df[col].isna().sum()}")
    
    print("\n[STRING/OBJECT COLUMN ANALYSIS]")
    string_cols = df.select_dtypes(include=['object']).columns
    print(f"String/Object columns: {list(string_cols)}")
    
    for col in string_cols:
        print(f"\n{col}:")
        unique_count = df[col].nunique()
        print(f"  Unique values: {unique_count}")
        if unique_count <= 20:
            print(f"  All values: {list(df[col].dropna().unique())}")
        else:
            print(f"  Sample values: {list(df[col].dropna().unique()[:10])}")
    
    print("\n[SAMPLE ROWS]")
    print(df.head(3).to_string())

if __name__ == "__main__":
    inspect_data_structure() 