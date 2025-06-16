#!/usr/bin/env python3
"""
COMPREHENSIVE CSAF DATA QUALITY INVESTIGATION
==============================================

This script traces CSAF data quality through the entire pipeline:
1. Raw CSV analysis (csaf_temporal_events_latest.csv)
2. Aggregated CSV analysis (csaf_temporal_daily_aggregated.csv)
3. Processed Parquet analysis (csaf_processed.parquet)
4. Final database integration analysis

Goal: Identify where data quality degrades or information is lost
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pyarrow.parquet as pq

def analyze_raw_csaf_data():
    """Step 1: Analyze the raw CSAF temporal events data"""
    print("🔍 STEP 1: RAW CSAF DATA ANALYSIS")
    print("=" * 60)
    
    file_path = "data/csaf/raw/csaf_temporal_events_latest.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"📂 Loading: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"📊 Basic Statistics:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   - Columns: {list(df.columns)}")
    
    # Data quality checks
    print(f"\n🔍 Data Quality Analysis:")
    
    # Missing values
    print(f"   Missing Values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"     - {col}: {count:,} ({pct:.1f}%)")
    
    # CVE analysis
    print(f"\n📋 CVE Analysis:")
    print(f"   - Total CVEs: {df['cve'].nunique():,}")
    print(f"   - Total events: {len(df):,}")
    print(f"   - Events per CVE (avg): {len(df) / df['cve'].nunique():.1f}")
    
    # Date analysis
    if 'date_parsed' in df.columns:
        df['date_parsed'] = pd.to_datetime(df['date_parsed'], errors='coerce')
        valid_dates = df['date_parsed'].dropna()
        print(f"\n📅 Date Analysis:")
        print(f"   - Valid dates: {len(valid_dates):,} / {len(df):,} ({len(valid_dates)/len(df)*100:.1f}%)")
        if len(valid_dates) > 0:
            print(f"   - Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
            print(f"   - Date span: {(valid_dates.max() - valid_dates.min()).days} days")
    
    # Event type analysis
    if 'event_type' in df.columns:
        print(f"\n🎯 Event Type Distribution:")
        event_counts = df['event_type'].value_counts()
        for event_type, count in event_counts.head(10).items():
            pct = (count / len(df)) * 100
            print(f"   - {event_type}: {count:,} ({pct:.1f}%)")
    
    # Source analysis
    if 'source' in df.columns:
        print(f"\n📡 Source Distribution:")
        source_counts = df['source'].value_counts()
        for source, count in source_counts.head(10).items():
            pct = (count / len(df)) * 100
            print(f"   - {source}: {count:,} ({pct:.1f}%)")
    
    # Duplicates analysis
    print(f"\n🔄 Duplicate Analysis:")
    if 'cve' in df.columns and 'date_parsed' in df.columns:
        df['date_only'] = df['date_parsed'].dt.date
        duplicates = df.groupby(['cve', 'date_only']).size()
        multi_event_days = (duplicates > 1).sum()
        max_events_per_day = duplicates.max()
        print(f"   - CVE-date pairs with multiple events: {multi_event_days:,}")
        print(f"   - Max events per CVE per day: {max_events_per_day}")
    
    # Sample data
    print(f"\n📋 Sample Data (first 3 rows):")
    sample_cols = ['cve', 'date_parsed', 'event_type', 'source', 'details']
    available_cols = [col for col in sample_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].head(3).to_string(index=False, max_colwidth=50))
    
    return df

def analyze_aggregated_csaf_data():
    """Step 2: Analyze the aggregated CSAF data"""
    print("\n🔍 STEP 2: AGGREGATED CSAF DATA ANALYSIS")
    print("=" * 60)
    
    file_path = "data/csaf/raw/csaf_temporal_daily_aggregated.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"📂 Loading: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"📊 Basic Statistics:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"   - Columns: {list(df.columns)}")
    
    # Data quality checks
    print(f"\n🔍 Data Quality Analysis:")
    
    # Missing values
    print(f"   Missing Values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"     - {col}: {count:,} ({pct:.1f}%)")
    
    # CVE analysis
    print(f"\n📋 CVE Analysis:")
    print(f"   - Total CVEs: {df['cve'].nunique():,}")
    print(f"   - Total daily records: {len(df):,}")
    print(f"   - Records per CVE (avg): {len(df) / df['cve'].nunique():.1f}")
    
    # Date analysis
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        valid_dates = df['date'].dropna()
        print(f"\n📅 Date Analysis:")
        print(f"   - Valid dates: {len(valid_dates):,} / {len(df):,} ({len(valid_dates)/len(df)*100:.1f}%)")
        if len(valid_dates) > 0:
            print(f"   - Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
            print(f"   - Date span: {(valid_dates.max() - valid_dates.min()).days} days")
    
    # Check for perfect uniqueness
    if 'cve' in df.columns and 'date' in df.columns:
        total_rows = len(df)
        unique_pairs = df[['cve', 'date']].drop_duplicates().shape[0]
        print(f"\n🔑 Uniqueness Check:")
        print(f"   - Total rows: {total_rows:,}")
        print(f"   - Unique (CVE, date) pairs: {unique_pairs:,}")
        print(f"   - Perfect uniqueness: {'✅' if total_rows == unique_pairs else '❌'}")
    
    # Feature analysis
    feature_cols = [col for col in df.columns if col not in ['cve', 'date', 'original_date', 'date_parsed', 'cve_date_key']]
    print(f"\n🎯 Feature Analysis ({len(feature_cols)} features):")
    for col in feature_cols[:10]:  # Show first 10 features
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"   - {col}: {non_null:,} non-null ({pct:.1f}%)")
    
    return df

def analyze_processed_csaf_data():
    """Step 3: Analyze the processed CSAF parquet data"""
    print("\n🔍 STEP 3: PROCESSED CSAF DATA ANALYSIS")
    print("=" * 60)
    
    file_path = "data/csaf/processed/csaf_processed.parquet"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"📂 Loading sample from: {file_path}")
    
    try:
        # Get file size first
        if os.path.isdir(file_path):
            total_size = sum(os.path.getsize(os.path.join(file_path, f)) 
                           for f in os.listdir(file_path) 
                           if f.endswith('.parquet'))
        else:
            total_size = os.path.getsize(file_path)
        
        print(f"📊 File Size: {total_size / 1024**3:.1f} GB")
        
        # Use pyarrow to read metadata first
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        print(f"📊 File Metadata:")
        print(f"   - File size: {total_size / 1024**3:.1f} GB")
        print(f"   - Total rows: {total_rows:,}")
        print(f"   - Schema: {len(parquet_file.schema)} columns")
        
        # Read just first few row groups
        table = parquet_file.read_row_group(0)
        df = table.to_pandas()
        
        print(f"📊 Sample Statistics (first row group):")
        print(f"   - Sample shape: {df.shape}")
        print(f"   - Columns: {len(df.columns)}")
        print(f"   - Column names: {list(df.columns)}")
        
        # Data quality checks on sample
        print(f"\n🔍 Data Quality Analysis (sample):")
        
        # Missing values
        print(f"   Missing Values in Sample:")
        missing = df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"     - {col}: {count:,} ({pct:.1f}%)")
        
        # CVE analysis
        print(f"\n📋 CVE Analysis (sample):")
        print(f"   - CVEs in sample: {df['cve'].nunique():,}")
        print(f"   - Sample records: {len(df):,}")
        
        # Date analysis
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            valid_dates = df['date'].dropna()
            print(f"\n📅 Date Analysis (sample):")
            print(f"   - Valid dates: {len(valid_dates):,} / {len(df):,} ({len(valid_dates)/len(df)*100:.1f}%)")
            if len(valid_dates) > 0:
                print(f"   - Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
        
        # Check for actual CSAF events vs gaps
        csaf_event_cols = [col for col in df.columns if 'event' in col.lower() or 'source' in col.lower()]
        print(f"\n🎯 CSAF Event Columns ({len(csaf_event_cols)} found):")
        for col in csaf_event_cols[:10]:
            if col in df.columns:
                non_null = df[col].notna().sum()
                pct = (non_null / len(df)) * 100
                print(f"   - {col}: {non_null:,} non-null ({pct:.1f}%)")
        
        # Check if this is mostly gap-filled data
        if 'dominant_event_type' in df.columns:
            actual_events = df['dominant_event_type'].notna().sum()
            gap_days = df['dominant_event_type'].isna().sum()
            print(f"\n📊 Event vs Gap Analysis (sample):")
            print(f"   - Actual events: {actual_events:,} ({actual_events/len(df)*100:.1f}%)")
            print(f"   - Gap days: {gap_days:,} ({gap_days/len(df)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"❌ Error reading processed file: {e}")
        print(f"   This suggests the file is very large (estimated 34GB with 253M rows)")
        print(f"   This matches the master EPSS timeline size - CSAF was expanded to full timeline!")
        return None

def analyze_final_database_integration():
    """Step 4: Analyze CSAF integration in final database"""
    print("\n🔍 STEP 4: FINAL DATABASE INTEGRATION ANALYSIS")
    print("=" * 60)
    
    # Check if CSAF is in the final database
    final_db_path = "data/full_db/processed/final_full_data.parquet"
    
    if not os.path.exists(final_db_path):
        print(f"❌ Final database not found: {final_db_path}")
        return None
    
    print(f"📂 Loading final database: {final_db_path}")
    
    # Use chunked reading for large file
    try:
        # Read just the first few rows to check columns
        import pyarrow.parquet as pq
        
        # Use pyarrow to get metadata first
        parquet_file = pq.ParquetFile(final_db_path)
        total_rows = parquet_file.metadata.num_rows
        
        # Get file size
        file_size_gb = os.path.getsize(final_db_path) / 1024**3
        
        print(f"📊 Final Database Metadata:")
        print(f"   - File size: {file_size_gb:.1f} GB")
        print(f"   - Total rows: {total_rows:,}")
        print(f"   - Schema: {len(parquet_file.schema)} columns")
        
        # Read just first row group for analysis
        table = parquet_file.read_row_group(0)
        sample_df = table.to_pandas()
        
        print(f"📊 Final Database Sample:")
        print(f"   - Sample shape: {sample_df.shape}")
        print(f"   - Total columns: {len(sample_df.columns)}")
        
        # Identify CSAF columns
        csaf_cols = [col for col in sample_df.columns if any(keyword in col.lower() 
                    for keyword in ['event', 'source', 'discovery', 'release', 'threat', 
                                   'remediation', 'stage', 'detail', 'doc_id', 'csaf', 
                                   'temporal', 'advisory'])]
        
        print(f"\n🎯 CSAF Columns in Final Database ({len(csaf_cols)} found):")
        for col in csaf_cols:
            non_null = sample_df[col].notna().sum()
            pct = (non_null / len(sample_df)) * 100
            print(f"   - {col}: {non_null:,} non-null ({pct:.1f}%)")
        
        if len(csaf_cols) == 0:
            print("   ❌ No CSAF columns found in final database!")
        
        return sample_df
        
    except Exception as e:
        print(f"❌ Error reading final database: {e}")
        return None

def compare_data_flow():
    """Compare data quality across all stages"""
    print("\n🔍 STEP 5: DATA FLOW COMPARISON")
    print("=" * 60)
    
    stages = {}
    
    # Raw data
    raw_path = "data/csaf/raw/csaf_temporal_events_latest.csv"
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path, nrows=1000)  # Sample for comparison
        stages['raw'] = {
            'rows': len(raw_df),
            'cves': raw_df['cve'].nunique() if 'cve' in raw_df.columns else 0,
            'columns': len(raw_df.columns),
            'file_size_mb': os.path.getsize(raw_path) / 1024**2
        }
    
    # Aggregated data
    agg_path = "data/csaf/raw/csaf_temporal_daily_aggregated.csv"
    if os.path.exists(agg_path):
        agg_df = pd.read_csv(agg_path, nrows=1000)  # Sample for comparison
        stages['aggregated'] = {
            'rows': len(agg_df),
            'cves': agg_df['cve'].nunique() if 'cve' in agg_df.columns else 0,
            'columns': len(agg_df.columns),
            'file_size_mb': os.path.getsize(agg_path) / 1024**2
        }
    
    # Processed data
    proc_path = "data/csaf/processed/csaf_processed.parquet"
    if os.path.exists(proc_path):
        try:
            proc_df = pd.read_parquet(proc_path, nrows=1000)  # Sample for comparison
            stages['processed'] = {
                'rows': len(proc_df),
                'cves': proc_df['cve'].nunique() if 'cve' in proc_df.columns else 0,
                'columns': len(proc_df.columns),
                'file_size_mb': sum(os.path.getsize(os.path.join(proc_path, f)) 
                                  for f in os.listdir(proc_path) 
                                  if f.endswith('.parquet')) / 1024**2 if os.path.isdir(proc_path) else 0
            }
        except:
            stages['processed'] = {'error': 'Could not read processed file'}
    
    # Print comparison
    print("📊 Data Flow Comparison (sample data):")
    print(f"{'Stage':<12} {'Rows':<10} {'CVEs':<8} {'Cols':<6} {'Size(MB)':<10}")
    print("-" * 50)
    
    for stage, stats in stages.items():
        if 'error' in stats:
            print(f"{stage:<12} {stats['error']}")
        else:
            print(f"{stage:<12} {stats['rows']:<10} {stats['cves']:<8} {stats['columns']:<6} {stats['file_size_mb']:<10.1f}")

def main():
    """Run comprehensive CSAF data quality investigation"""
    print("🚀 COMPREHENSIVE CSAF DATA QUALITY INVESTIGATION")
    print("=" * 80)
    print("Tracing data quality from raw CSV through final database integration")
    print("=" * 80)
    
    # Step 1: Raw data analysis
    raw_df = analyze_raw_csaf_data()
    
    # Step 2: Aggregated data analysis
    agg_df = analyze_aggregated_csaf_data()
    
    # Step 3: Processed data analysis
    proc_df = analyze_processed_csaf_data()
    
    # Step 4: Final database integration
    final_df = analyze_final_database_integration()
    
    # Step 5: Compare data flow
    compare_data_flow()
    
    print("\n🎯 INVESTIGATION COMPLETE")
    print("=" * 40)
    print("Check the output above for data quality issues and information loss points.")

if __name__ == "__main__":
    main() 