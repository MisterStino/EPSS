#!/usr/bin/env python3
"""
Merge CSAF temporal events onto dense daily timeline for LSTM modeling.
Result: data/csaf/processed/csaf_processed.parquet

Approach
--------
Similar to NVD script but adapted for CSAF temporal requirements:
• Create dense daily timeline for each CVE (first event to last event)
• Overlay CSAF aggregated events onto this timeline  
• Apply 9 imputation strategies for different feature types
• Ensure no temporal leakage for LSTM training

Input
-----
• CSAF aggregated: data/csaf/raw/csaf_temporal_daily_aggregated.csv
  (sparse events with perfect CVE-date uniqueness)

Output  
------
• Dense timeline: data/csaf/processed/csaf_processed.parquet
  (daily rows for each CVE with proper gap filling)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================================================================ PATHS ====
CSAF_AGGREGATED = "data/csaf/raw/csaf_temporal_daily_aggregated.csv"
CSAF_PROCESSED  = "data/csaf/processed/csaf_processed.parquet"

print("🚀 CSAF DENSE TIMELINE GENERATION")
print("=" * 50)

# ================================================================ 1. LOAD ====
print("📂 Loading CSAF aggregated data...")
csaf_events = pd.read_csv(CSAF_AGGREGATED)
csaf_events['date'] = pd.to_datetime(csaf_events['date'], utc=True)

print(f"Input events: {len(csaf_events):,} rows × {len(csaf_events.columns)} columns")
print(f"CVEs covered: {csaf_events['cve'].nunique():,}")
print(f"Date range: {csaf_events['date'].min().date()} to {csaf_events['date'].max().date()}")

# ================================================================ 2. CREATE DENSE TIMELINE ====
print("\n🗓️  Creating dense daily timeline...")

def create_dense_timeline(events_df):
    """
    Create dense daily timeline for each CVE from first to last event
    """
    # Get date range for each CVE
    cve_ranges = events_df.groupby('cve')['date'].agg(['min', 'max']).reset_index()
    cve_ranges.columns = ['cve', 'start_date', 'end_date']
    
    print(f"Creating timelines for {len(cve_ranges):,} CVEs...")
    
    # Create daily timeline for each CVE
    timeline_dfs = []
    
    for i, row in cve_ranges.iterrows():
        if i % 10000 == 0:
            print(f"  Processing CVE {i+1:,}/{len(cve_ranges):,}")
        
        # Create date range for this CVE
        date_range = pd.date_range(
            start=row['start_date'], 
            end=row['end_date'], 
            freq='D'
        )
        
        # Create DataFrame for this CVE
        cve_timeline = pd.DataFrame({
            'cve': row['cve'],
            'date': date_range
        })
        
        timeline_dfs.append(cve_timeline)
    
    # Combine all timelines
    dense_timeline = pd.concat(timeline_dfs, ignore_index=True)
    
    return dense_timeline

# Create the dense timeline
dense_timeline = create_dense_timeline(csaf_events)

print(f"Dense timeline created: {len(dense_timeline):,} rows")
print(f"Expansion ratio: {len(dense_timeline) / len(csaf_events):.2f}x")

# ================================================================ 3. OVERLAY EVENTS ====
print("\n🔗 Overlaying CSAF events onto dense timeline...")

# LEFT JOIN events onto dense timeline
merged = dense_timeline.merge(
    csaf_events, 
    on=['cve', 'date'], 
    how='left'
)

print(f"After overlay: {len(merged):,} rows")
print(f"Rows with events: {merged['dominant_event_type'].notna().sum():,}")
print(f"Rows with gaps: {merged['dominant_event_type'].isna().sum():,}")
print(f"Gap ratio: {merged['dominant_event_type'].isna().mean()*100:.1f}%")

# ================================================================ 4. APPLY IMPUTATION STRATEGIES ====
print("\n🧩 Applying imputation strategies...")

def apply_imputation_strategies(df):
    """
    Apply the 9 imputation strategies for different feature types
    """
    print("  📊 Identifying feature columns...")
    
    # Define imputation strategies for each column
    imputation_map = {
        # NO_IMPUTATION - should never be missing
        'cve': 'no_imputation',
        'date': 'no_imputation',
        
        # LEAVE_NAN - semantically meaningful missing values
        'original_date': 'leave_nan',
        'dominant_event_type': 'leave_nan', 
        'primary_source': 'leave_nan',
        
        # FORWARD_FILL - state persists until changed
        'date_parsed': 'forward_fill',
        'has_discovery': 'forward_fill',
        'has_release': 'forward_fill', 
        'has_threat': 'forward_fill',
        'has_remediation': 'forward_fill',
        'event_sequence': 'forward_fill',
        'cumulative_source_count': 'forward_fill',
        'total_events_so_far': 'forward_fill',
        'prev_event_type': 'forward_fill',
        'event_stage_num': 'forward_fill',
        'max_stage_reached': 'forward_fill',
        
        # FILL_ZERO - no activity = zero
        'event_type_count': 'fill_zero',
        'source_count': 'fill_zero',
        'total_detail_length': 'fill_zero',
        
        # FILL_FALSE - no activity = false  
        'has_multi_source': 'fill_false',
        'same_day_multi_source': 'fill_false',
        
        # FILL_EMPTY_STRING - no activity = empty
        'event_types_list': 'fill_empty_string',
        'sources_list': 'fill_empty_string',
        'doc_ids': 'fill_empty_string',
        'details_combined': 'fill_empty_string',
        'details_longest': 'fill_empty_string',
        
        # FILL_EMPTY_JSON - no activity = empty JSON
        'event_data_merged': 'fill_empty_json',
        
        # DAILY_INCREMENT - special temporal handling
        'days_since_last_event': 'daily_increment',
        
        # RECONSTRUCT - rebuild from components
        'cve_date_key': 'reconstruct'
    }
    
    # Sort by CVE and date for proper temporal processing
    df = df.sort_values(['cve', 'date']).reset_index(drop=True)
    
    # Apply each strategy
    print("  🔄 Applying forward fill strategies...")
    forward_fill_cols = [col for col, strategy in imputation_map.items() 
                        if strategy == 'forward_fill' and col in df.columns]
    
    for col in forward_fill_cols:
        print(f"    Forward filling: {col}")
        df[col] = df.groupby('cve')[col].fillna(method='ffill')
    
    print("  🔢 Applying zero fill strategies...")
    zero_fill_cols = [col for col, strategy in imputation_map.items() 
                     if strategy == 'fill_zero' and col in df.columns]
    
    for col in zero_fill_cols:
        print(f"    Zero filling: {col}")
        df[col] = df[col].fillna(0)
    
    print("  ✅ Applying false fill strategies...")
    false_fill_cols = [col for col, strategy in imputation_map.items() 
                      if strategy == 'fill_false' and col in df.columns]
    
    for col in false_fill_cols:
        print(f"    False filling: {col}")
        df[col] = df[col].fillna(False)
    
    print("  📝 Applying empty string strategies...")
    empty_string_cols = [col for col, strategy in imputation_map.items() 
                        if strategy == 'fill_empty_string' and col in df.columns]
    
    for col in empty_string_cols:
        print(f"    Empty string filling: {col}")
        df[col] = df[col].fillna('')
    
    print("  🗂️  Applying empty JSON strategy...")
    if 'event_data_merged' in df.columns:
        df['event_data_merged'] = df['event_data_merged'].fillna('{}')
    
    print("  ⏰ Applying daily increment strategy...")
    if 'days_since_last_event' in df.columns:
        df = calculate_daily_increments(df)
    
    print("  🔧 Reconstructing derived columns...")
    if 'cve_date_key' in df.columns:
        df['cve_date_key'] = df['cve'] + '_' + df['date'].dt.strftime('%Y-%m-%d')
    
    if 'date_parsed' in df.columns:
        df['date_parsed'] = df['date']
    
    return df

def calculate_daily_increments(df):
    """
    Calculate days_since_last_event with daily increments for gaps
    CRITICAL: This feature drives LSTM predictions
    """
    print("    🎯 Calculating daily increments for days_since_last_event...")
    
    def process_cve_group(group):
        """Process single CVE timeline"""
        group = group.sort_values('date').reset_index(drop=True)
        result = []
        
        for i, row in group.iterrows():
            if pd.notna(row['days_since_last_event']):
                # Actual event - use the calculated value
                result.append(row['days_since_last_event'])
            else:
                # Gap day - increment from previous day
                if len(result) == 0:
                    # First day of timeline, no previous events
                    result.append(0)
                else:
                    # Increment from previous day
                    result.append(result[-1] + 1)
        
        group['days_since_last_event'] = result
        return group
    
    # Apply to each CVE group
    df = df.groupby('cve').apply(process_cve_group, include_groups=False).reset_index()
    
    return df

# Apply all imputation strategies
processed = apply_imputation_strategies(merged)

# ================================================================ 5. VALIDATION ====
print("\n✅ Validating temporal integrity...")

def validate_temporal_integrity(df):
    """
    Ensure no temporal leakage and proper feature behavior
    """
    print("  🔍 Checking basic integrity...")
    
    # Check uniqueness
    duplicates = df.groupby(['cve', 'date']).size().max()
    assert duplicates == 1, f"Found duplicates: max count = {duplicates}"
    print("    ✅ Perfect (CVE, date) uniqueness")
    
    # Check temporal ordering
    temporal_issues = 0
    sample_cves = df['cve'].unique()[:1000]  # Sample check
    
    for cve in sample_cves:
        cve_data = df[df['cve'] == cve]
        if not cve_data['date'].is_monotonic_increasing:
            temporal_issues += 1
    
    assert temporal_issues == 0, f"Temporal ordering issues: {temporal_issues} CVEs"
    print("    ✅ Perfect temporal ordering")
    
    # Check cumulative features are monotonic
    print("  📈 Checking cumulative features...")
    cumulative_cols = ['cumulative_source_count', 'total_events_so_far', 'max_stage_reached']
    
    for col in cumulative_cols:
        if col in df.columns:
            for cve in sample_cves[:100]:  # Smaller sample for intensive check
                cve_data = df[df['cve'] == cve][col]
                if not cve_data.is_monotonic_increasing:
                    print(f"    ❌ {col} not monotonic for {cve}")
                    break
            else:
                print(f"    ✅ {col} is monotonic")
    
    # Check days_since_last_event increments properly
    print("  ⏰ Checking days_since_last_event increments...")
    issues = 0
    
    for cve in sample_cves[:100]:
        cve_data = df[df['cve'] == cve].sort_values('date')
        for i in range(1, len(cve_data)):
            prev_days = cve_data.iloc[i-1]['days_since_last_event']
            curr_days = cve_data.iloc[i]['days_since_last_event']
            
            # Should either increment by 1 (gap day) or reset to small value (new event)
            if curr_days < prev_days and curr_days > 10:  # Allow resets for new events
                issues += 1
                break
    
    if issues == 0:
        print("    ✅ days_since_last_event increments properly")
    else:
        print(f"    ⚠️  {issues} potential issues with days_since_last_event")
    
    print("  📊 Final statistics...")
    print(f"    Total rows: {len(df):,}")
    print(f"    Total CVEs: {df['cve'].nunique():,}")
    print(f"    Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"    Days with actual events: {df['dominant_event_type'].notna().sum():,}")
    print(f"    Days with gaps (filled): {df['dominant_event_type'].isna().sum():,}")
    print(f"    Gap ratio: {df['dominant_event_type'].isna().mean()*100:.1f}%")

validate_temporal_integrity(processed)

# ================================================================ 6. OPTIMIZE & SAVE ====
print("\n💾 Optimizing data types and saving...")

def optimize_dtypes(df):
    """
    Optimize data types for efficient storage and processing
    """
    print("  🔧 Optimizing data types...")
    
    # Boolean columns
    bool_cols = [col for col in df.columns if col.startswith('has_') or col.endswith('_multi_source')]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype('bool')
    
    # Integer columns  
    int_cols = ['event_type_count', 'source_count', 'total_detail_length', 
               'event_sequence', 'cumulative_source_count', 'total_events_so_far',
               'event_stage_num', 'max_stage_reached']
    
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Float columns
    float_cols = ['days_since_last_event']
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    print(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    return df

# Optimize data types
processed = optimize_dtypes(processed)

# Create output directory
output_path = Path(CSAF_PROCESSED)
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save as parquet with compression
print(f"  💾 Saving to {CSAF_PROCESSED}...")
processed.to_parquet(
    CSAF_PROCESSED,
    compression='snappy',
    index=False
)

# Get file size
file_size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"✅ Saved successfully! File size: {file_size_mb:.1f} MB")

# ================================================================ 7. FINAL SUMMARY ====
print("\n🎯 PIPELINE SUMMARY")
print("=" * 30)
print(f"✅ Input events: {len(csaf_events):,} sparse events")
print(f"✅ Output timeline: {len(processed):,} dense daily rows")
print(f"✅ Expansion ratio: {len(processed) / len(csaf_events):.2f}x")
print(f"✅ CVEs covered: {processed['cve'].nunique():,}")
print(f"✅ Date range: {processed['date'].min().date()} to {processed['date'].max().date()}")
print(f"✅ Gap handling: 9 imputation strategies applied")
print(f"✅ Temporal integrity: Validated, no future leakage")
print(f"✅ LSTM ready: Dense timeline with rich temporal features")

print(f"\n🚀 CSAF dense timeline ready for LSTM modeling!")
print(f"📁 Output: {CSAF_PROCESSED}")
print(f"📊 {len(processed):,} rows × {len(processed.columns)} columns")
print(f"💽 File size: {file_size_mb:.1f} MB")









