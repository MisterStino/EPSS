#!/usr/bin/env python3
"""
Rigorous Validation of Composite Key Duplicates

Step-by-step analysis to determine if we actually have problematic duplicates
and what's causing them.
"""

import pandas as pd
import json
from datetime import datetime

def step1_load_and_basic_check():
    """Step 1: Load data and understand what we're working with"""
    print("🔍 STEP 1: LOADING DATA AND BASIC UNDERSTANDING")
    print("="*60)
    
    df = pd.read_csv("catalogs_processed/master_cve_timeseries_20250610.csv")
    
    print(f"Total rows in dataset: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    
    # What should be unique?
    print(f"\nWhat SHOULD be unique: (cve_id, reconstruction_timestamp) combinations")
    print(f"This means: Each CVE can have multiple timestamps, but each CVE+timestamp combo should appear only once")
    
    return df

def step2_check_composite_key_uniqueness(df):
    """Step 2: Properly check for composite key duplicates"""
    print(f"\n🔍 STEP 2: CHECKING COMPOSITE KEY UNIQUENESS")
    print("="*60)
    
    print("Thinking process:")
    print("- If (CVE-A, timestamp-1) appears twice → PROBLEM")
    print("- If (CVE-A, timestamp-1) and (CVE-B, timestamp-1) both exist → NO PROBLEM")
    
    # Create composite key
    df['composite_key'] = df['cve_id'] + '|' + df['reconstruction_timestamp']
    
    total_rows = len(df)
    unique_composite_keys = df['composite_key'].nunique()
    
    print(f"\nTotal rows: {total_rows:,}")
    print(f"Unique composite keys: {unique_composite_keys:,}")
    print(f"Difference: {total_rows - unique_composite_keys:,}")
    
    if total_rows == unique_composite_keys:
        print("✅ NO COMPOSITE KEY DUPLICATES - Each (CVE, timestamp) appears exactly once")
        return df, False
    else:
        print("❌ COMPOSITE KEY DUPLICATES FOUND - Same (CVE, timestamp) appears multiple times")
        return df, True

def step3_analyze_actual_duplicates(df, has_duplicates):
    """Step 3: If duplicates exist, analyze them in detail"""
    if not has_duplicates:
        print(f"\n✅ STEP 3: SKIPPED - No duplicates to analyze")
        return
    
    print(f"\n🔍 STEP 3: ANALYZING ACTUAL DUPLICATE COMPOSITE KEYS")
    print("="*60)
    
    # Find rows that share the same composite key
    duplicate_mask = df.duplicated(subset=['composite_key'], keep=False)
    duplicate_rows = df[duplicate_mask].copy()
    
    print(f"Total rows involved in duplicates: {len(duplicate_rows):,}")
    
    # Group by composite key to see duplicate sets
    duplicate_groups = duplicate_rows.groupby('composite_key')
    print(f"Number of duplicate composite keys: {len(duplicate_groups):,}")
    
    # Analyze sizes of duplicate groups
    group_sizes = duplicate_groups.size()
    print(f"\nDuplicate group sizes:")
    for size, count in group_sizes.value_counts().sort_index().items():
        print(f"  {size} identical rows per key: {count:,} composite keys")
    
    return duplicate_groups

def step4_deep_dive_duplicate_content(duplicate_groups):
    """Step 4: Deep dive into what's different between duplicate rows"""
    print(f"\n🔍 STEP 4: DEEP DIVE INTO DUPLICATE CONTENT")
    print("="*60)
    
    print("Questions to answer:")
    print("1. Are duplicate rows identical in ALL columns?")
    print("2. If different, which columns differ?")
    print("3. What could explain these differences?")
    
    # Take first few duplicate groups for analysis
    sample_groups = list(duplicate_groups)[:5]
    
    for i, (composite_key, group_df) in enumerate(sample_groups):
        cve_id, timestamp = composite_key.split('|', 1)
        print(f"\n--- DUPLICATE GROUP {i+1}: {cve_id} at {timestamp} ---")
        print(f"Number of rows: {len(group_df)}")
        
        # Check if all rows are identical
        first_row = group_df.iloc[0]
        all_identical = True
        differing_columns = []
        
        for col in group_df.columns:
            if col == 'composite_key':  # Skip our helper column
                continue
                
            values = group_df[col].tolist()
            if len(set(str(v) for v in values)) > 1:  # More than one unique value
                all_identical = False
                differing_columns.append(col)
        
        if all_identical:
            print("✅ All rows are IDENTICAL - Perfect duplicates")
            print("   → This suggests a bug in data generation/processing")
        else:
            print(f"❌ Rows DIFFER in {len(differing_columns)} columns: {differing_columns}")
            
            # Show the differences
            for col in differing_columns[:3]:  # Show first 3 differing columns
                print(f"   {col}:")
                for idx, (_, row) in enumerate(group_df.iterrows()):
                    value = str(row[col])[:100]  # Truncate for display
                    print(f"     Row {idx+1}: {value}")

def step5_timeline_analysis(df, duplicate_groups):
    """Step 5: Analyze if there are patterns in when duplicates occur"""
    print(f"\n🔍 STEP 5: TIMELINE ANALYSIS OF DUPLICATES")
    print("="*60)
    
    if duplicate_groups is None:
        print("No duplicates to analyze")
        return
    
    print("Looking for patterns in when duplicates occur...")
    
    # Get all composite keys that have duplicates
    duplicate_keys = list(duplicate_groups.groups.keys())
    
    # Extract timestamps and analyze patterns
    duplicate_timestamps = []
    for key in duplicate_keys:
        cve_id, timestamp = key.split('|', 1)
        duplicate_timestamps.append(timestamp)
    
    # Convert to datetime for analysis
    try:
        dt_timestamps = pd.to_datetime(duplicate_timestamps)
        
        print(f"\nTemporal patterns in duplicates:")
        print(f"  Earliest duplicate: {dt_timestamps.min()}")
        print(f"  Latest duplicate: {dt_timestamps.max()}")
        
        # Year distribution
        years = dt_timestamps.dt.year.value_counts().sort_index()
        print(f"\nDuplicates by year:")
        for year, count in years.head(10).items():
            print(f"  {year}: {count:,} duplicate composite keys")
            
    except Exception as e:
        print(f"Could not parse timestamps for temporal analysis: {e}")

def step6_investigate_specific_cve_timeline(df):
    """Step 6: Look at a specific CVE's complete timeline to understand what happened"""
    print(f"\n🔍 STEP 6: INVESTIGATING SPECIFIC CVE TIMELINE")
    print("="*60)
    
    # Find a CVE with duplicates
    duplicate_mask = df.duplicated(subset=['cve_id', 'reconstruction_timestamp'], keep=False)
    if not duplicate_mask.any():
        print("No duplicates found for detailed timeline analysis")
        return
    
    sample_cve = df[duplicate_mask]['cve_id'].iloc[0]
    cve_timeline = df[df['cve_id'] == sample_cve].copy()
    cve_timeline = cve_timeline.sort_values('reconstruction_timestamp')
    
    print(f"Analyzing timeline for: {sample_cve}")
    print(f"Total states: {len(cve_timeline)}")
    
    print(f"\nComplete timeline:")
    for i, (_, row) in enumerate(cve_timeline.iterrows()):
        ts = row['reconstruction_timestamp']
        ts_raw = row['reconstruction_timestamp_raw']
        vuln_status = row['vuln_status']
        cvss_ver = row['primary_cvss_ver']
        cvss_score = row['primary_cvss_score']
        
        duplicate_marker = "🔄" if cve_timeline['reconstruction_timestamp'].tolist().count(ts) > 1 else "  "
        
        print(f"  {duplicate_marker} {i+1:2d}. {ts} | Status: {vuln_status} | CVSS: {cvss_ver} ({cvss_score})")
        if ts != ts_raw:
            print(f"       Raw timestamp: {ts_raw}")

def main():
    """Execute rigorous step-by-step validation"""
    print("🚀 RIGOROUS COMPOSITE KEY VALIDATION")
    print("="*80)
    print("Goal: Determine if we have actual composite key duplicates")
    print("Composite key = (cve_id, reconstruction_timestamp)")
    print("="*80)
    
    # Step by step analysis
    df = step1_load_and_basic_check()
    df, has_duplicates = step2_check_composite_key_uniqueness(df)
    duplicate_groups = step3_analyze_actual_duplicates(df, has_duplicates)
    
    if has_duplicates:
        step4_deep_dive_duplicate_content(duplicate_groups)
        step5_timeline_analysis(df, duplicate_groups)
        step6_investigate_specific_cve_timeline(df)
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print("="*40)
        print("❌ CONFIRMED: We have composite key duplicates")
        print("   This means the same CVE has multiple rows with identical timestamps")
        print("   This violates the intended unique temporal state requirement")
    else:
        print(f"\n🎯 FINAL ASSESSMENT:")
        print("="*40)
        print("✅ NO ISSUES: All composite keys are unique")
        print("   Each (CVE, timestamp) combination appears exactly once")
        print("   Multiple CVEs can share timestamps - this is expected and fine")

if __name__ == "__main__":
    main() 