#!/usr/bin/env python3
"""
Investigation of Primary Key Duplicates and JSON Issues
"""

import pandas as pd
import json
from collections import Counter

def investigate_primary_key_duplicates():
    """Investigate the primary key duplicate issue"""
    print("🔍 INVESTIGATING PRIMARY KEY DUPLICATES")
    print("="*50)
    
    df = pd.read_csv("catalogs_processed/master_cve_timeseries_20250610.csv")
    
    # Find duplicate timestamps
    duplicate_mask = df.duplicated(subset=['cve_id', 'reconstruction_timestamp'], keep=False)
    duplicates = df[duplicate_mask]
    
    print(f"Total duplicate rows: {len(duplicates):,}")
    
    # Analyze timestamp patterns
    timestamp_counts = duplicates['reconstruction_timestamp'].value_counts()
    print(f"\nMost common duplicate timestamps:")
    for ts, count in timestamp_counts.head(10).items():
        print(f"  {ts}: {count:,} CVEs")
    
    # Check if this is the snapshot date issue
    snapshot_pattern = "2025-06-10T00:00:00.000"
    snapshot_duplicates = duplicates[duplicates['reconstruction_timestamp'] == snapshot_pattern]
    print(f"\nSnapshot date duplicates ({snapshot_pattern}): {len(snapshot_duplicates):,}")
    
    # Look at specific CVE with duplicates
    sample_cve = duplicates['cve_id'].iloc[0]
    cve_rows = df[df['cve_id'] == sample_cve].copy()
    cve_rows = cve_rows.sort_values('reconstruction_timestamp')
    
    print(f"\nExample CVE with duplicates: {sample_cve}")
    print(f"Total states: {len(cve_rows)}")
    print("Timestamps:")
    for i, row in cve_rows.iterrows():
        print(f"  {row['reconstruction_timestamp']} | {row['reconstruction_timestamp_raw']}")

def investigate_json_errors():
    """Investigate JSON parsing errors"""
    print("\n🔍 INVESTIGATING JSON PARSING ERRORS")
    print("="*50)
    
    df = pd.read_csv("catalogs_processed/master_cve_timeseries_20250610.csv")
    
    json_columns = ['descriptions_json', 'metrics_json', 'weaknesses_json', 
                   'configurations_json', 'references_json']
    
    for col in json_columns:
        print(f"\n📦 Analyzing {col}:")
        
        # Find empty strings (likely cause of parse errors)
        empty_strings = (df[col] == '').sum()
        null_values = df[col].isnull().sum()
        
        print(f"  Empty strings: {empty_strings:,}")
        print(f"  Null values: {null_values:,}")
        
        # Sample problematic values
        if empty_strings > 0:
            empty_indices = df[df[col] == ''].index[:5]
            print(f"  Sample empty string rows: {list(empty_indices)}")
            
        # Check for other non-JSON values
        non_json_patterns = []
        sample_values = df[col].dropna().head(100)
        
        for val in sample_values:
            if val and not val.startswith(('{', '[')):
                non_json_patterns.append(val[:50])
        
        if non_json_patterns:
            pattern_counts = Counter(non_json_patterns)
            print(f"  Non-JSON patterns found: {len(pattern_counts)}")
            for pattern, count in pattern_counts.most_common(3):
                print(f"    '{pattern}...': {count} times")

if __name__ == "__main__":
    investigate_primary_key_duplicates()
    investigate_json_errors() 