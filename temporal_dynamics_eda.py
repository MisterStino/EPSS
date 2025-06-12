#!/usr/bin/env python3
"""
Temporal Dynamics EDA - CVE Dataset Evolution Analysis
Deep analysis of what actually changes over time and temporal patterns for LSTM modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TEMPORAL DYNAMICS EDA - CVE EVOLUTION ANALYSIS")
print("="*80)

# Load dataset
df = pd.read_parquet('catalogs_processed/cve_snapshots_irregular.parquet')
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Unique CVEs: {df['cve_id'].nunique():,}")
print(f"Date range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}")

# Define feature categories for temporal analysis
temporal_features = {
    'cvss_features': ['canon_base', 'canon_severity', 'has_v2', 'has_v30', 'has_v31', 'has_v40'],
    'platform_features': ['is_windows', 'is_linux', 'is_android', 'is_ios', 'is_macos', 
                          'is_hardware', 'is_application', 'is_os'],
    'content_features': ['n_cpes', 'n_vendors', 'n_refs', 'desc_len_en', 'desc_len_all'],
    'status_features': ['vuln_status', 'cwe_id']
}

print("\n" + "="*80)
print("1. SNAPSHOT DISTRIBUTION ANALYSIS")
print("="*80)

# Analyze snapshot counts per CVE
snapshots_per_cve = df.groupby('cve_id').size()
print(f"\nSnapshot distribution per CVE:")
print(f"  Mean: {snapshots_per_cve.mean():.2f}")
print(f"  Median: {snapshots_per_cve.median():.0f}")
print(f"  Min: {snapshots_per_cve.min()}")
print(f"  Max: {snapshots_per_cve.max()}")

# Distribution breakdown
distribution = snapshots_per_cve.value_counts().sort_index()
print(f"\nSnapshot count distribution:")
for count, freq in distribution.head(20).items():
    pct = (freq / len(snapshots_per_cve)) * 100
    print(f"  {count} snapshots: {freq:,} CVEs ({pct:.1f}%)")

# Identify static vs dynamic CVEs
static_cves = snapshots_per_cve[snapshots_per_cve == 1].index
dynamic_cves = snapshots_per_cve[snapshots_per_cve > 1].index

print(f"\nCVE Temporal Categories:")
print(f"  Static (1 snapshot): {len(static_cves):,} CVEs ({len(static_cves)/len(snapshots_per_cve)*100:.1f}%)")
print(f"  Dynamic (2+ snapshots): {len(dynamic_cves):,} CVEs ({len(dynamic_cves)/len(snapshots_per_cve)*100:.1f}%)")

# Focus on dynamic CVEs for temporal analysis
df_dynamic = df[df['cve_id'].isin(dynamic_cves)].copy()
print(f"\nDynamic CVE dataset: {len(df_dynamic):,} snapshots from {len(dynamic_cves):,} CVEs")

print("\n" + "="*80)
print("2. FEATURE CHANGE ANALYSIS")
print("="*80)

def analyze_feature_changes(df, feature_dict):
    """Analyze which features actually change over time within CVEs"""
    
    change_analysis = {}
    
    for category, features in feature_dict.items():
        print(f"\n📊 {category.upper()} ANALYSIS:")
        print("-" * 50)
        
        category_changes = {}
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            # Count CVEs where this feature changes
            changing_cves = 0
            total_changes = 0
            change_patterns = []
            
            for cve_id in dynamic_cves[:5000]:  # Sample for performance
                cve_data = df[df['cve_id'] == cve_id][feature].dropna()
                
                if len(cve_data) > 1:
                    if cve_data.nunique() > 1:  # Feature has multiple values
                        changing_cves += 1
                        total_changes += cve_data.nunique() - 1
                        
                        # Analyze change pattern
                        if feature in ['canon_base']:
                            first_val = cve_data.iloc[0]
                            last_val = cve_data.iloc[-1]
                            if pd.notna(first_val) and pd.notna(last_val):
                                change_patterns.append(last_val - first_val)
            
            change_rate = (changing_cves / min(len(dynamic_cves), 5000)) * 100
            
            print(f"  {feature}:")
            print(f"    CVEs with changes: {changing_cves:,} ({change_rate:.1f}%)")
            print(f"    Total changes: {total_changes:,}")
            
            if change_patterns and feature == 'canon_base':
                increases = sum(1 for x in change_patterns if x > 0)
                decreases = sum(1 for x in change_patterns if x < 0)
                print(f"    CVSS increases: {increases}, decreases: {decreases}")
            
            category_changes[feature] = {
                'changing_cves': changing_cves,
                'change_rate': change_rate,
                'total_changes': total_changes
            }
        
        change_analysis[category] = category_changes
    
    return change_analysis

# Analyze changes in dynamic CVEs
change_analysis = analyze_feature_changes(df_dynamic, temporal_features)

print("\n" + "="*80)
print("3. TEMPORAL PATTERN CLASSIFICATION")
print("="*80)

def classify_temporal_patterns(df, min_snapshots=3):
    """Classify CVEs by their temporal evolution patterns"""
    
    patterns = {
        'highly_dynamic': [],      # 10+ snapshots, multiple feature changes
        'moderately_dynamic': [],  # 5-9 snapshots, some changes
        'low_dynamic': [],         # 2-4 snapshots, few changes
        'static_multi': []         # Multiple snapshots but no changes
    }
    
    pattern_details = {}
    
    for cve_id in dynamic_cves[:3000]:  # Sample for detailed analysis
        cve_data = df[df['cve_id'] == cve_id].sort_values('snapshot_date')
        n_snapshots = len(cve_data)
        
        if n_snapshots < 2:
            continue
        
        # Count changing features
        changing_features = 0
        feature_changes = {}
        
        for category, features in temporal_features.items():
            for feature in features:
                if feature in cve_data.columns:
                    if cve_data[feature].nunique() > 1:
                        changing_features += 1
                        feature_changes[feature] = cve_data[feature].nunique()
        
        # Classify pattern
        if n_snapshots >= 10 and changing_features >= 3:
            patterns['highly_dynamic'].append(cve_id)
        elif n_snapshots >= 5 and changing_features >= 2:
            patterns['moderately_dynamic'].append(cve_id)
        elif n_snapshots >= 2 and changing_features >= 1:
            patterns['low_dynamic'].append(cve_id)
        else:
            patterns['static_multi'].append(cve_id)
        
        pattern_details[cve_id] = {
            'snapshots': n_snapshots,
            'changing_features': changing_features,
            'feature_changes': feature_changes
        }
    
    return patterns, pattern_details

patterns, pattern_details = classify_temporal_patterns(df_dynamic)

print("CVE Temporal Pattern Classification:")
for pattern_type, cve_list in patterns.items():
    pct = (len(cve_list) / 3000) * 100  # Based on sample size
    print(f"  {pattern_type}: {len(cve_list):,} CVEs ({pct:.1f}%)")

print("\n" + "="*80)
print("4. DETAILED CASE STUDIES")
print("="*80)

# Analyze specific examples from each pattern
def analyze_cve_evolution(cve_id, df):
    """Detailed analysis of a specific CVE's evolution"""
    cve_data = df[df['cve_id'] == cve_id].sort_values('snapshot_date').copy()
    
    print(f"\n🔍 {cve_id} Evolution Analysis:")
    print(f"   Snapshots: {len(cve_data)}")
    print(f"   Time span: {cve_data['snapshot_date'].min()} → {cve_data['snapshot_date'].max()}")
    
    # CVSS evolution
    cvss_scores = cve_data['canon_base'].dropna()
    if len(cvss_scores) > 0:
        if cvss_scores.nunique() > 1:
            print(f"   CVSS evolution: {cvss_scores.tolist()}")
        else:
            print(f"   CVSS stable: {cvss_scores.iloc[0]}")
    
    # Platform changes
    platform_cols = ['is_windows', 'is_linux', 'is_android', 'is_ios', 'is_macos']
    platform_changes = []
    for col in platform_cols:
        if cve_data[col].nunique() > 1:
            platform_changes.append(f"{col}:{cve_data[col].tolist()}")
    
    if platform_changes:
        print(f"   Platform changes: {', '.join(platform_changes)}")
    
    # Content changes
    if cve_data['n_cpes'].nunique() > 1:
        print(f"   CPE count evolution: {cve_data['n_cpes'].tolist()}")
    
    if cve_data['desc_len_en'].nunique() > 1:
        print(f"   Description length changes: {cve_data['desc_len_en'].tolist()}")
    
    return cve_data

# Analyze examples from each pattern
print("Case Studies by Pattern Type:")

for pattern_type, cve_list in patterns.items():
    if cve_list:
        print(f"\n--- {pattern_type.upper()} EXAMPLES ---")
        # Analyze 2-3 examples from each category
        for cve_id in cve_list[:3]:
            analyze_cve_evolution(cve_id, df)

print("\n" + "="*80)
print("5. LSTM MODELING INSIGHTS")
print("="*80)

def analyze_lstm_potential(df, patterns, pattern_details):
    """Analyze what features would be useful for LSTM modeling"""
    
    print("🤖 LSTM Modeling Assessment:")
    print("-" * 40)
    
    # Identify most informative features for temporal modeling
    feature_informativeness = {}
    
    for category, features in temporal_features.items():
        for feature in features:
            if feature in df.columns:
                # Calculate temporal variance
                changing_rate = 0
                total_variance = 0
                
                for cve_id in list(patterns['highly_dynamic'])[:100]:
                    cve_data = df[df['cve_id'] == cve_id][feature]
                    if len(cve_data) > 1 and cve_data.nunique() > 1:
                        changing_rate += 1
                        if feature == 'canon_base':
                            total_variance += cve_data.var()
                
                feature_informativeness[feature] = changing_rate
    
    # Rank features by temporal informativeness
    sorted_features = sorted(feature_informativeness.items(), 
                           key=lambda x: x[1], reverse=True)
    
    print("Most temporally informative features for LSTM:")
    for feature, score in sorted_features[:10]:
        print(f"  {feature}: {score}/100 highly dynamic CVEs show changes")
    
    # Sequence length analysis
    sequence_lengths = []
    for pattern_type in ['highly_dynamic', 'moderately_dynamic']:
        for cve_id in patterns[pattern_type]:
            if cve_id in pattern_details:
                sequence_lengths.append(pattern_details[cve_id]['snapshots'])
    
    if sequence_lengths:
        print(f"\nSequence length analysis for dynamic CVEs:")
        print(f"  Mean sequence length: {np.mean(sequence_lengths):.1f}")
        print(f"  Median sequence length: {np.median(sequence_lengths):.0f}")
        print(f"  95th percentile: {np.percentile(sequence_lengths, 95):.0f}")
    
    # Temporal density analysis
    temporal_gaps = []
    for cve_id in patterns['highly_dynamic'][:100]:
        cve_data = df[df['cve_id'] == cve_id].sort_values('snapshot_date')
        if len(cve_data) > 1:
            dates = pd.to_datetime(cve_data['snapshot_date'])
            gaps = [(dates.iloc[i+1] - dates.iloc[i]).days for i in range(len(dates)-1)]
            temporal_gaps.extend(gaps)
    
    if temporal_gaps:
        print(f"\nTemporal gap analysis:")
        print(f"  Mean gap: {np.mean(temporal_gaps):.0f} days")
        print(f"  Median gap: {np.median(temporal_gaps):.0f} days")
        print(f"  95th percentile: {np.percentile(temporal_gaps, 95):.0f} days")

analyze_lstm_potential(df_dynamic, patterns, pattern_details)

print("\n" + "="*80)
print("6. SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("📋 KEY FINDINGS:")
print("-" * 20)

# Calculate overall temporal richness
total_dynamic_cves = len(dynamic_cves)
highly_dynamic_pct = (len(patterns['highly_dynamic']) / 3000) * 100
moderately_dynamic_pct = (len(patterns['moderately_dynamic']) / 3000) * 100

print(f"1. TEMPORAL RICHNESS:")
print(f"   • {len(dynamic_cves):,} CVEs have multiple snapshots ({len(dynamic_cves)/df['cve_id'].nunique()*100:.1f}%)")
print(f"   • {highly_dynamic_pct:.1f}% are highly dynamic (10+ snapshots, 3+ changing features)")
print(f"   • {moderately_dynamic_pct:.1f}% are moderately dynamic (5+ snapshots, 2+ changing features)")

print(f"\n2. MOST CHANGING FEATURES:")
most_changing = []
for category, features in change_analysis.items():
    for feature, stats in features.items():
        if stats['change_rate'] > 5:  # More than 5% of CVEs show changes
            most_changing.append((feature, stats['change_rate']))

most_changing.sort(key=lambda x: x[1], reverse=True)
for feature, rate in most_changing[:5]:
    print(f"   • {feature}: {rate:.1f}% of dynamic CVEs show changes")

print(f"\n3. LSTM MODELING POTENTIAL:")
if highly_dynamic_pct > 5:
    print(f"   • ✅ HIGH: {highly_dynamic_pct:.1f}% of CVEs show rich temporal evolution")
elif moderately_dynamic_pct > 15:
    print(f"   • ⚠️  MODERATE: {moderately_dynamic_pct:.1f}% show some temporal patterns")
else:
    print(f"   • ❌ LOW: Limited temporal dynamics for sequence modeling")

print(f"\n4. RECOMMENDED APPROACH:")
if highly_dynamic_pct > 5:
    print(f"   • Focus on highly dynamic CVEs ({len(patterns['highly_dynamic']):,} CVEs)")
    print(f"   • Use variable-length sequences (mean: ~{np.mean([pattern_details[cve]['snapshots'] for cve in patterns['highly_dynamic'][:100] if cve in pattern_details]):.1f} snapshots)")
    print(f"   • Primary features: CVSS scores, CPE counts, platform flags")
else:
    print(f"   • Consider static modeling or simple time-series features")
    print(f"   • Temporal patterns may be too sparse for LSTM effectiveness")

print(f"\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80) 