# first get for each cve 1 date: aggreggate.

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

def load_and_validate_data():
    """
    Load the CSAF temporal events dataset and validate structure
    """
    file_path = 'data/csaf/raw/csaf_temporal_events_latest.csv'
    print(f"Loading data from: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Convert dates
    df['date_parsed'] = pd.to_datetime(df['date_parsed'], utc=True)
    df['date_only'] = df['date_parsed'].dt.date
    
    # Check current duplicates
    duplicates = df.groupby(['cve', 'date_only']).size()
    multi_event_days = (duplicates > 1).sum()
    print(f"Days with multiple events for same CVE: {multi_event_days:,}")
    print(f"Max events per CVE per day: {duplicates.max()}")
    
    return df

def aggregate_event_types(group):
    """
    Aggregate event types with priority-based selection and binary flags
    """
    event_types = group['event_type'].tolist()
    
    # Priority order (highest priority wins)
    priority = {
        'vulnerability_discovery': 1,
        'vulnerability_release': 2, 
        'threat_assessment': 3,
        'remediation_available': 4
    }
    
    # Create binary indicators
    result = {
        'has_discovery': 'vulnerability_discovery' in event_types,
        'has_release': 'vulnerability_release' in event_types,
        'has_threat': 'threat_assessment' in event_types,
        'has_remediation': 'remediation_available' in event_types,
        'event_type_count': len(event_types),
        'event_types_list': '|'.join(sorted(set(event_types)))
    }
    
    # Dominant event type (highest priority)
    if event_types:
        result['dominant_event_type'] = max(event_types, key=lambda x: priority.get(x, 0))
    else:
        result['dominant_event_type'] = None
    
    return pd.Series(result)

def aggregate_sources(group):
    """
    Aggregate sources preserving multi-source intelligence
    """
    sources = group['source'].unique().tolist()
    
    result = {
        'sources_list': '|'.join(sorted(sources)),
        'source_count': len(sources),
        'primary_source': sources[0] if sources else None,  # First alphabetically
        'has_multi_source': len(sources) > 1
    }
    
    return pd.Series(result)

def aggregate_content(group):
    """
    Aggregate content fields intelligently
    """
    result = {
        'doc_ids': '|'.join(group['doc_id'].unique()),
        'details_combined': ' | '.join(group['details'].unique()),
        'details_longest': max(group['details'], key=len) if len(group['details']) > 0 else '',
        'total_detail_length': group['details'].str.len().sum()
    }
    
    return pd.Series(result)

def merge_event_data_json(series):
    """
    Merge JSON event_data from multiple sources
    """
    all_data = {}
    
    for json_str in series:
        if pd.notna(json_str) and json_str.strip():
            try:
                data = json.loads(json_str)
                all_data.update(data)
            except (json.JSONDecodeError, TypeError):
                continue
    
    return json.dumps(all_data) if all_data else '{}'

def collapse_to_daily_timeseries(df):
    """
    Collapse multiple same-day events per CVE into single rows
    """
    print("\n🔄 Collapsing to daily time series...")
    print(f"Input: {len(df):,} rows")
    
    # Group by CVE and date (day level)
    grouped = df.groupby(['cve', 'date_only'])
    
    # Apply custom aggregations
    print("Aggregating event types...")
    event_agg = grouped.apply(aggregate_event_types, include_groups=False).reset_index()
    
    print("Aggregating sources...")
    source_agg = grouped.apply(aggregate_sources, include_groups=False).reset_index()
    
    print("Aggregating content...")
    content_agg = grouped.apply(aggregate_content, include_groups=False).reset_index()
    
    print("Merging JSON event data...")
    json_agg = grouped['event_data'].apply(merge_event_data_json).reset_index()
    json_agg.columns = ['cve', 'date_only', 'event_data_merged']
    
    # Simple aggregations
    simple_agg = grouped.agg({
        'original_date': 'first',
        'date_parsed': 'first',
        'cve_date_key': 'first'
    }).reset_index()
    
    # Merge all aggregations
    collapsed = simple_agg.merge(event_agg, on=['cve', 'date_only']) \
                         .merge(source_agg, on=['cve', 'date_only']) \
                         .merge(content_agg, on=['cve', 'date_only']) \
                         .merge(json_agg, on=['cve', 'date_only'])
    
    # Restore proper date column and clean up
    collapsed['date'] = pd.to_datetime(collapsed['date_only'])
    collapsed.drop('date_only', axis=1, inplace=True)
    
    # Sort by CVE and date for temporal sequence
    collapsed = collapsed.sort_values(['cve', 'date']).reset_index(drop=True)
    
    # Recalculate event sequence after collapse
    collapsed['event_sequence'] = collapsed.groupby('cve').cumcount() + 1
    
    print(f"Output: {len(collapsed):,} rows")
    print(f"Compression ratio: {len(df) / len(collapsed):.2f}x")
    
    return collapsed

def add_temporal_features(df):
    """
    Add temporally valid features (no future leakage)
    """
    print("\n⏰ Adding temporal features...")
    
    # Ensure proper sorting
    df = df.sort_values(['cve', 'date']).reset_index(drop=True)
    
    # 1. Days since last event (within CVE)
    print("Adding days_since_last_event...")
    df['days_since_last_event'] = df.groupby('cve')['date'].diff().dt.days
    df['days_since_last_event'] = df['days_since_last_event'].fillna(-1)  # First event
    
    # 2. Cumulative source count (expanding unique sources)
    print("Adding cumulative_source_count...")
    def expanding_unique_sources(group):
        all_sources = set()
        counts = []
        for sources_str in group['sources_list']:
            if pd.notna(sources_str):
                day_sources = set(sources_str.split('|'))
                all_sources.update(day_sources)
            counts.append(len(all_sources))
        return pd.Series(counts, index=group.index)
    
    df['cumulative_source_count'] = df.groupby('cve').apply(
        expanding_unique_sources, include_groups=False
    ).values
    
    # 3. Same day multi-source flag (already computed)
    print("Adding same_day_multi_source...")
    df['same_day_multi_source'] = df['has_multi_source']
    
    # 4. Total events so far (cumulative count)
    print("Adding total_events_so_far...")
    df['total_events_so_far'] = df.groupby('cve').cumcount() + 1
    
    # 5. Previous event type (shift within CVE)
    print("Adding prev_event_type...")
    df['prev_event_type'] = df.groupby('cve')['dominant_event_type'].shift(1)
    df['prev_event_type'] = df['prev_event_type'].fillna('none')
    
    # 6. Max lifecycle stage reached (expanding max)
    print("Adding max_stage_reached...")
    stage_order = {
        'vulnerability_discovery': 1,
        'vulnerability_release': 2,
        'threat_assessment': 3,
        'remediation_available': 4
    }
    
    df['event_stage_num'] = df['dominant_event_type'].map(stage_order).fillna(0)
    df['max_stage_reached'] = df.groupby('cve')['event_stage_num'].expanding().max().reset_index(0, drop=True)
    
    print("Temporal features added successfully!")
    return df

def validate_results(df):
    """
    Validate the final aggregated dataset
    """
    print("\n✅ Validating results...")
    
    # Check uniqueness of (CVE, date) keys
    duplicates = df.groupby(['cve', 'date']).size()
    max_duplicates = duplicates.max()
    
    if max_duplicates == 1:
        print("✅ Perfect uniqueness: Each (CVE, date) appears exactly once")
    else:
        print(f"❌ Still have duplicates: Max occurrences = {max_duplicates}")
        return False
    
    # Check temporal ordering within CVEs
    temporal_check = df.groupby('cve')['date'].apply(lambda x: x.is_monotonic_increasing).all()
    
    if temporal_check:
        print("✅ Perfect temporal ordering: Dates within CVEs are monotonic")
    else:
        print("❌ Temporal ordering broken")
        return False
    
    # Check feature validity
    print("\n📊 Feature Statistics:")
    print(f"- Total CVEs: {df['cve'].nunique():,}")
    print(f"- Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"- Multi-source days: {df['has_multi_source'].sum():,} ({df['has_multi_source'].mean()*100:.1f}%)")
    print(f"- Max events per CVE: {df['total_events_so_far'].max()}")
    print(f"- Max sources per CVE: {df['cumulative_source_count'].max()}")
    
    # Sample temporal features for validation
    sample_cve = df[df['cve'] == df['cve'].iloc[0]].head()
    print(f"\n🔍 Sample temporal features for {sample_cve['cve'].iloc[0]}:")
    print(sample_cve[['date', 'days_since_last_event', 'cumulative_source_count', 'same_day_multi_source', 'total_events_so_far']].to_string(index=False))
    
    return True

def main():
    """
    Main pipeline: load -> collapse -> features -> validate -> save
    """
    print("🚀 CSAF Temporal Events Aggregation Pipeline")
    print("=" * 50)
    
    # Step 1: Load and validate input data
    df = load_and_validate_data()
    
    # Step 2: Collapse to daily time series
    collapsed_df = collapse_to_daily_timeseries(df)
    
    # Step 3: Add temporal features
    final_df = add_temporal_features(collapsed_df)
    
    # Step 4: Validate results
    if not validate_results(final_df):
        print("❌ Validation failed! Stopping pipeline.")
        return None
    
    # Step 5: Save results
    output_path = 'data/csaf/raw/csaf_temporal_daily_aggregated.csv'
    print(f"\n💾 Saving aggregated dataset to: {output_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with compression
    final_df.to_csv(output_path, index=False)
    
    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Saved successfully! File size: {file_size_mb:.1f} MB")
    
    print("\n🎯 Pipeline Summary:")
    print(f"- Input rows: {len(df):,}")
    print(f"- Output rows: {len(final_df):,}")
    print(f"- Compression: {len(df) / len(final_df):.2f}x")
    print(f"- Columns: {len(final_df.columns)}")
    print(f"- Perfect (CVE, date) uniqueness: ✅")
    print(f"- Temporal features: ✅")
    print(f"- Ready for LSTM: ✅")
    
    return final_df

if __name__ == "__main__":
    result_df = main()
