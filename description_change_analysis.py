#!/usr/bin/env python3
"""
Description Change Analysis for CVE Temporal Snapshots

Analyzes how CVE textual descriptions evolve over time in the temporal reconstruction dataset.
Examines frequency, patterns, and types of description changes.
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import difflib
import re
from datetime import datetime
import json

def load_temporal_data():
    """Load the temporal snapshots dataset"""
    print("Loading temporal snapshots dataset...")
    df = pd.read_parquet('catalogs_processed/cve_snapshots_irregular.parquet')
    print(f"Loaded {len(df):,} snapshots")
    return df

def analyze_description_changes(df):
    """Analyze how descriptions change over time for each CVE"""
    print("\n=== DESCRIPTION CHANGE ANALYSIS ===")
    
    # Focus on CVEs with multiple snapshots
    multi_snapshot_cves = df.groupby('cve_id').size()
    dynamic_cves = multi_snapshot_cves[multi_snapshot_cves > 1].index
    dynamic_df = df[df['cve_id'].isin(dynamic_cves)].copy()
    
    print(f"Analyzing {len(dynamic_cves):,} CVEs with multiple snapshots")
    print(f"Total snapshots for dynamic CVEs: {len(dynamic_df):,}")
    
    # Sort by CVE ID and reconstruction timestamp for proper temporal ordering
    dynamic_df = dynamic_df.sort_values(['cve_id', 'reconstruction_timestamp'])
    
    # Analyze description changes
    description_changes = []
    cves_with_desc_changes = set()
    
    for cve_id in dynamic_cves:
        cve_snapshots = dynamic_df[dynamic_df['cve_id'] == cve_id].copy()
        
        # Check for description changes (using English descriptions)
        descriptions = cve_snapshots['description_en'].tolist()
        timestamps = cve_snapshots['reconstruction_timestamp'].tolist()
        
        for i in range(1, len(descriptions)):
            prev_desc = descriptions[i-1] if pd.notna(descriptions[i-1]) else ""
            curr_desc = descriptions[i] if pd.notna(descriptions[i]) else ""
            
            if prev_desc != curr_desc:
                cves_with_desc_changes.add(cve_id)
                
                # Calculate change metrics
                change_info = {
                    'cve_id': cve_id,
                    'snapshot_idx': i,
                    'reconstruction_timestamp': timestamps[i],
                    'prev_timestamp': timestamps[i-1],
                    'prev_desc': prev_desc,
                    'curr_desc': curr_desc,
                    'prev_length': len(prev_desc),
                    'curr_length': len(curr_desc),
                    'length_change': len(curr_desc) - len(prev_desc),
                    'change_type': classify_change_type(prev_desc, curr_desc)
                }
                description_changes.append(change_info)
    
    print(f"\nCVEs with description changes: {len(cves_with_desc_changes):,}")
    print(f"Total description changes: {len(description_changes):,}")
    print(f"Percentage of dynamic CVEs with desc changes: {len(cves_with_desc_changes)/len(dynamic_cves)*100:.1f}%")
    
    return description_changes, cves_with_desc_changes

def classify_change_type(prev_desc, curr_desc):
    """Classify the type of description change"""
    if not prev_desc and curr_desc:
        return "addition"
    elif prev_desc and not curr_desc:
        return "removal"
    elif not prev_desc and not curr_desc:
        return "both_empty"
    
    # Calculate similarity
    similarity = difflib.SequenceMatcher(None, prev_desc, curr_desc).ratio()
    
    if similarity > 0.9:
        return "minor_edit"
    elif similarity > 0.7:
        return "moderate_edit"
    elif similarity > 0.3:
        return "major_rewrite"
    else:
        return "complete_rewrite"

def analyze_change_patterns(description_changes):
    """Analyze patterns in description changes"""
    print("\n=== DESCRIPTION CHANGE PATTERNS ===")
    
    if not description_changes:
        print("No description changes found")
        return
    
    changes_df = pd.DataFrame(description_changes)
    
    # Change type distribution
    change_types = changes_df['change_type'].value_counts()
    print(f"\nChange type distribution:")
    for change_type, count in change_types.items():
        print(f"  {change_type}: {count:,} ({count/len(changes_df)*100:.1f}%)")
    
    # Length change analysis
    print(f"\nLength change statistics:")
    print(f"  Mean length change: {changes_df['length_change'].mean():.1f} characters")
    print(f"  Median length change: {changes_df['length_change'].median():.1f} characters")
    print(f"  Max length increase: {changes_df['length_change'].max():,} characters")
    print(f"  Max length decrease: {changes_df['length_change'].min():,} characters")
    
    # Temporal analysis
    changes_df['reconstruction_timestamp'] = pd.to_datetime(changes_df['reconstruction_timestamp'])
    changes_df['year'] = changes_df['reconstruction_timestamp'].dt.year
    
    yearly_changes = changes_df['year'].value_counts().sort_index()
    print(f"\nDescription changes by year (recent years):")
    for year in sorted(yearly_changes.index)[-10:]:
        print(f"  {year}: {yearly_changes[year]:,}")
    
    return changes_df

def show_example_changes(description_changes, n_examples=5):
    """Show examples of different types of description changes"""
    print(f"\n=== EXAMPLE DESCRIPTION CHANGES ===")
    
    if not description_changes:
        print("No description changes to show")
        return
    
    changes_df = pd.DataFrame(description_changes)
    
    # Show examples for each change type
    for change_type in ['minor_edit', 'moderate_edit', 'major_rewrite', 'complete_rewrite', 'addition']:
        type_changes = changes_df[changes_df['change_type'] == change_type]
        if len(type_changes) == 0:
            continue
            
        print(f"\n--- {change_type.upper()} EXAMPLES ---")
        
        # Show a few examples
        for i, (_, change) in enumerate(type_changes.head(min(n_examples, 3)).iterrows()):
            print(f"\nExample {i+1}: {change['cve_id']}")
            print(f"Timestamp: {change['reconstruction_timestamp']}")
            print(f"Length change: {change['length_change']:+d} characters")
            
            prev_desc = change['prev_desc'][:200] + "..." if len(change['prev_desc']) > 200 else change['prev_desc']
            curr_desc = change['curr_desc'][:200] + "..." if len(change['curr_desc']) > 200 else change['curr_desc']
            
            print(f"BEFORE: {prev_desc}")
            print(f"AFTER:  {curr_desc}")
            
            if change_type in ['minor_edit', 'moderate_edit']:
                # Show diff for minor/moderate changes
                diff = list(difflib.unified_diff(
                    change['prev_desc'].splitlines(keepends=True),
                    change['curr_desc'].splitlines(keepends=True),
                    lineterm='',
                    n=1
                ))
                if len(diff) > 0:
                    print("DIFF:")
                    for line in diff[2:8]:  # Skip headers, show first few lines
                        print(f"  {line.rstrip()}")

def analyze_description_evolution_correlation(df, description_changes):
    """Analyze correlation between description changes and other field changes"""
    print(f"\n=== DESCRIPTION CHANGE CORRELATIONS ===")
    
    if not description_changes:
        print("No description changes to correlate")
        return
    
    changes_df = pd.DataFrame(description_changes)
    
    # For each CVE with description changes, check what other fields changed
    correlations = defaultdict(int)
    
    for cve_id in changes_df['cve_id'].unique():
        cve_snapshots = df[df['cve_id'] == cve_id].sort_values('reconstruction_timestamp')
        
        if len(cve_snapshots) < 2:
            continue
        
        # Check which fields changed for this CVE
        changed_fields = set()
        
        for col in ['primary_cvss_score', 'primary_cvss_ver', 'primary_cvss_sev', 
                   'cwe_id', 'n_cpes', 'n_vendors', 'vuln_status']:
            if col in cve_snapshots.columns:
                values = cve_snapshots[col].dropna().unique()
                if len(values) > 1:
                    changed_fields.add(col)
        
        # Count correlations
        for field in changed_fields:
            correlations[field] += 1
    
    total_desc_change_cves = len(changes_df['cve_id'].unique())
    
    print(f"Field changes correlated with description changes:")
    for field, count in sorted(correlations.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_desc_change_cves * 100
        print(f"  {field}: {count:,} CVEs ({percentage:.1f}%)")

def main():
    """Main analysis function"""
    print("CVE Description Change Analysis")
    print("=" * 50)
    
    # Load data
    df = load_temporal_data()
    
    # Analyze description changes
    description_changes, cves_with_desc_changes = analyze_description_changes(df)
    
    # Analyze patterns
    changes_df = analyze_change_patterns(description_changes)
    
    # Show examples
    show_example_changes(description_changes)
    
    # Analyze correlations
    analyze_description_evolution_correlation(df, description_changes)
    
    # Summary statistics
    print(f"\n=== SUMMARY ===")
    total_cves = df['cve_id'].nunique()
    multi_snapshot_cves = df.groupby('cve_id').size()
    dynamic_cves = len(multi_snapshot_cves[multi_snapshot_cves > 1])
    
    print(f"Total CVEs: {total_cves:,}")
    print(f"CVEs with multiple snapshots: {dynamic_cves:,}")
    print(f"CVEs with description changes: {len(cves_with_desc_changes):,}")
    print(f"Percentage of all CVEs with desc changes: {len(cves_with_desc_changes)/total_cves*100:.2f}%")
    print(f"Percentage of dynamic CVEs with desc changes: {len(cves_with_desc_changes)/dynamic_cves*100:.1f}%")
    
    if description_changes:
        print(f"Total description change events: {len(description_changes):,}")
        print(f"Average changes per CVE (among changing CVEs): {len(description_changes)/len(cves_with_desc_changes):.1f}")

if __name__ == "__main__":
    main() 