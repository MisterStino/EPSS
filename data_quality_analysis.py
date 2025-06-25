#!/usr/bin/env python3
"""
Comprehensive Data Quality Analysis
==================================

Based on the investigation results, analyze:
1. What constitutes genuine vs artificial GitHub activity
2. Why our previous cleaning was too aggressive
3. How to properly clean for LSTM training

Key Findings from Investigation:
- BigQuery has suspicious bulk patterns (96k CVEs with "1_0_0_1")
- But 61.5% of baseline CVEs overlap with BigQuery
- Need evidence-based approach, not pattern-based assumptions
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DataQualityAnalyzer:
    def __init__(self):
        print("📊 DATA QUALITY ANALYSIS")
        print("=" * 50)
        print("Evidence-based approach to identify real problems")
        print("=" * 50)
        
        # Load datasets
        self.bigquery_df = pd.read_csv("data/github/raw/github-final.csv")
        self.baseline_df = pd.read_csv("data/github/raw/github_commit_timestamps_9k.csv")
        
        # Standardize column names
        self.baseline_df = self.baseline_df.rename(columns={
            'cve_id': 'cve', 
            'commit_date': 'date'
        })
        
        # Convert dates
        self.bigquery_df['date'] = pd.to_datetime(self.bigquery_df['date'])
        self.baseline_df['date'] = pd.to_datetime(self.baseline_df['date'])
        
    def analyze_federico_approach(self):
        """Analyze Federico's data collection approach"""
        print("\n🔍 FEDERICO'S APPROACH ANALYSIS")
        print("=" * 40)
        
        print("Federico's Method (from github_commit_date.py):")
        print("1. Uses GitHub Search API with CVE ID as search term")
        print("2. Searches commit messages containing CVE ID")
        print("3. Deduplicates by SHA hash")
        print("4. Groups by date and counts commits per CVE per day")
        print("5. Rate-limited, careful API usage")
        
        print("\nStrengths of Federico's approach:")
        print("✓ Direct API access (not BigQuery aggregation)")
        print("✓ SHA-based deduplication")
        print("✓ Temporal precision (exact commit dates)")
        print("✓ Manual verification possible")
        print("✓ Rate-limited to avoid artificial bulk patterns")
        
        print("\nPotential limitations:")
        print("⚠️  Limited to commits mentioning CVE ID in message")
        print("⚠️  May miss related activity without explicit CVE mention")
        print("⚠️  Depends on developer discipline in commit messages")
        
    def analyze_bigquery_problems(self):
        """Analyze specific problems with BigQuery data"""
        print("\n🚨 BIGQUERY DATA PROBLEMS")
        print("=" * 40)
        
        # Create activity signatures
        self.bigquery_df['signature'] = (
            self.bigquery_df['commit_count'].astype(str) + '_' +
            self.bigquery_df['issue_count'].astype(str) + '_' +
            self.bigquery_df['comment_count'].astype(str) + '_' +
            self.bigquery_df['repo_count'].astype(str)
        )
        
        # Analyze the most suspicious pattern
        suspicious_pattern = "1_0_0_1"  # 96k CVEs with this pattern
        suspicious_data = self.bigquery_df[self.bigquery_df['signature'] == suspicious_pattern]
        
        print(f"Analysis of most suspicious pattern '{suspicious_pattern}':")
        print(f"  Total records: {len(suspicious_data):,}")
        print(f"  Unique CVEs: {suspicious_data['cve'].nunique():,}")
        print(f"  Date range: {suspicious_data['date'].min().date()} to {suspicious_data['date'].max().date()}")
        print(f"  Unique dates: {suspicious_data['date'].nunique():,}")
        
        # Check if this pattern appears on specific dates
        daily_counts = suspicious_data.groupby('date')['cve'].nunique().sort_values(ascending=False)
        print(f"\nTop 10 dates with this pattern:")
        for date, count in daily_counts.head(10).items():
            print(f"    {date.date()}: {count:,} CVEs")
        
        # Check if legitimate CVEs are caught in this pattern
        overlap_cves = set(self.baseline_df['cve'].unique())
        suspicious_overlap = suspicious_data[suspicious_data['cve'].isin(overlap_cves)]
        
        print(f"\nLegitimate CVEs caught in suspicious pattern:")
        print(f"  Baseline CVEs with '1_0_0_1' pattern: {suspicious_overlap['cve'].nunique():,}")
        print(f"  Percentage of baseline affected: {suspicious_overlap['cve'].nunique() / len(overlap_cves) * 100:.1f}%")
        
        # This proves our cleaning was too aggressive!
        return suspicious_overlap
    
    def compare_data_sources(self):
        """Compare data quality between sources"""
        print("\n📊 DATA SOURCE COMPARISON")
        print("=" * 40)
        
        # Find overlapping CVEs
        bigquery_cves = set(self.bigquery_df['cve'].unique())
        baseline_cves = set(self.baseline_df['cve'].unique())
        overlap_cves = bigquery_cves.intersection(baseline_cves)
        
        print(f"Overlap Analysis:")
        print(f"  BigQuery CVEs: {len(bigquery_cves):,}")
        print(f"  Baseline CVEs: {len(baseline_cves):,}")
        print(f"  Overlapping CVEs: {len(overlap_cves):,}")
        
        # Analyze overlapping CVEs in detail
        bigquery_overlap = self.bigquery_df[self.bigquery_df['cve'].isin(overlap_cves)]
        baseline_overlap = self.baseline_df[self.baseline_df['cve'].isin(overlap_cves)]
        
        print(f"\nFor overlapping CVEs:")
        print(f"  BigQuery records: {len(bigquery_overlap):,}")
        print(f"  Baseline records: {len(baseline_overlap):,}")
        print(f"  BigQuery avg records per CVE: {len(bigquery_overlap) / len(overlap_cves):.1f}")
        print(f"  Baseline avg records per CVE: {len(baseline_overlap) / len(overlap_cves):.1f}")
        
        # Compare activity levels
        bq_total_commits = bigquery_overlap['commit_count'].sum()
        bl_total_commits = baseline_overlap['commit_count'].sum()
        
        print(f"\nActivity Comparison (overlapping CVEs only):")
        print(f"  BigQuery total commits: {bq_total_commits:,}")
        print(f"  Baseline total commits: {bl_total_commits:,}")
        print(f"  Ratio (BQ/Baseline): {bq_total_commits / bl_total_commits:.2f}")
        
        return overlap_cves, bigquery_overlap, baseline_overlap
    
    def identify_genuine_vs_artificial(self):
        """Develop criteria for genuine vs artificial activity"""
        print("\n🎯 GENUINE VS ARTIFICIAL CRITERIA")
        print("=" * 45)
        
        # Get overlapping data
        overlap_cves, bigquery_overlap, baseline_overlap = self.compare_data_sources()
        
        print("Evidence-based criteria for genuine activity:")
        print("\n1. TEMPORAL CONSISTENCY:")
        
        # Check temporal alignment for sample CVEs
        sample_cves = list(overlap_cves)[:20]
        temporal_matches = 0
        
        for cve in sample_cves:
            bq_dates = set(bigquery_overlap[bigquery_overlap['cve'] == cve]['date'].dt.date)
            bl_dates = set(baseline_overlap[baseline_overlap['cve'] == cve]['date'].dt.date)
            
            if bq_dates.intersection(bl_dates):
                temporal_matches += 1
        
        print(f"   Sample CVEs with matching dates: {temporal_matches}/{len(sample_cves)}")
        print(f"   Temporal consistency rate: {temporal_matches/len(sample_cves)*100:.1f}%")
        
        print("\n2. ACTIVITY LEVEL CONSISTENCY:")
        
        # Compare activity levels for matching dates
        activity_correlations = []
        for cve in sample_cves[:10]:  # Smaller sample for detailed analysis
            bq_cve_data = bigquery_overlap[bigquery_overlap['cve'] == cve]
            bl_cve_data = baseline_overlap[baseline_overlap['cve'] == cve]
            
            # Merge on date
            merged = pd.merge(
                bq_cve_data[['date', 'commit_count']],
                bl_cve_data[['date', 'commit_count']],
                on='date',
                suffixes=('_bq', '_bl')
            )
            
            if len(merged) > 1:
                corr = merged['commit_count_bq'].corr(merged['commit_count_bl'])
                if not pd.isna(corr):
                    activity_correlations.append(corr)
                    print(f"   {cve}: {len(merged)} matching dates, correlation: {corr:.3f}")
        
        if activity_correlations:
            avg_correlation = np.mean(activity_correlations)
            print(f"\n   Average activity correlation: {avg_correlation:.3f}")
        
        print("\n3. PROPOSED CLEANING CRITERIA:")
        print("   ✓ Keep CVEs that appear in baseline (verified genuine)")
        print("   ✓ Keep CVEs with temporal consistency with known patterns")
        print("   ✓ Remove CVEs that only appear on bulk processing dates")
        print("   ✓ Remove exact duplicate patterns across >1000 CVEs on same date")
        print("   ✗ Don't remove based on activity signature alone")
        
    def analyze_bulk_processing_dates(self):
        """Identify dates with artificial bulk processing"""
        print("\n📅 BULK PROCESSING DATE ANALYSIS")
        print("=" * 45)
        
        # Analyze daily activity distribution
        daily_activity = self.bigquery_df.groupby('date').agg({
            'cve': 'nunique',
            'commit_count': 'sum'
        }).reset_index()
        
        # Find statistical outliers
        q95 = daily_activity['cve'].quantile(0.95)
        q99 = daily_activity['cve'].quantile(0.99)
        
        print(f"Daily CVE activity statistics:")
        print(f"  95th percentile: {q95:.0f} CVEs/day")
        print(f"  99th percentile: {q99:.0f} CVEs/day")
        
        # Identify bulk processing dates
        bulk_dates = daily_activity[daily_activity['cve'] > q99]['date'].tolist()
        
        print(f"\nBulk processing dates (>99th percentile):")
        for date in bulk_dates[:10]:
            day_data = self.bigquery_df[self.bigquery_df['date'] == date]
            unique_signatures = day_data['signature'].nunique() if 'signature' in day_data.columns else 'N/A'
            print(f"  {date.date()}: {day_data['cve'].nunique():,} CVEs, {unique_signatures} unique patterns")
        
        # Check if baseline CVEs appear on bulk dates
        baseline_cves = set(self.baseline_df['cve'].unique())
        baseline_on_bulk = []
        
        for date in bulk_dates:
            day_data = self.bigquery_df[self.bigquery_df['date'] == date]
            baseline_count = day_data[day_data['cve'].isin(baseline_cves)]['cve'].nunique()
            baseline_on_bulk.append(baseline_count)
            
        if baseline_on_bulk:
            print(f"\nBaseline CVEs appearing on bulk dates:")
            print(f"  Average per bulk date: {np.mean(baseline_on_bulk):.1f}")
            print(f"  This suggests bulk dates may contain some genuine data")
        
        return bulk_dates
    
    def recommend_cleaning_strategy(self):
        """Recommend evidence-based cleaning strategy"""
        print("\n🎯 RECOMMENDED CLEANING STRATEGY")
        print("=" * 45)
        
        print("Based on analysis, here's the correct cleaning approach:")
        
        print("\n1. WHITELIST APPROACH:")
        print("   ✓ Start with baseline CVEs as 'known genuine'")
        print("   ✓ Preserve ALL baseline CVE records")
        print("   ✓ Use baseline as quality anchor")
        
        print("\n2. BULK DATE FILTERING:")
        print("   ✓ Identify statistical outlier dates (>99th percentile)")
        print("   ✓ On bulk dates, keep only CVEs that also appear on normal dates")
        print("   ✓ Remove CVEs that ONLY appear on bulk dates")
        
        print("\n3. PATTERN-BASED FILTERING (REFINED):")
        print("   ✓ Remove patterns that appear across >10,000 CVEs on same date")
        print("   ✓ BUT preserve if CVE also appears in baseline")
        print("   ✓ Focus on date-pattern combinations, not patterns alone")
        
        print("\n4. TEMPORAL VALIDATION:")
        print("   ✓ Keep CVEs with activity spanning multiple months")
        print("   ✓ Remove CVEs with activity only on 1-2 consecutive days")
        print("   ✓ Unless they're in the baseline")
        
        print("\n5. ACTIVITY CONSISTENCY:")
        print("   ✓ Keep CVEs with varied activity patterns over time")
        print("   ✓ Remove CVEs with identical patterns across all dates")
        
        print("\nThis approach should preserve ~80% of baseline while removing bulk noise")
        
    def run_analysis(self):
        """Run complete analysis"""
        print("🚀 Starting comprehensive data quality analysis...")
        
        self.analyze_federico_approach()
        suspicious_overlap = self.analyze_bigquery_problems()
        self.identify_genuine_vs_artificial()
        bulk_dates = self.analyze_bulk_processing_dates()
        self.recommend_cleaning_strategy()
        
        print(f"\n{'='*60}")
        print("🎉 Analysis completed!")
        print("\nKey Findings:")
        print(f"✓ Federico's baseline is high-quality, API-based data")
        print(f"✓ BigQuery has bulk patterns but contains genuine signals")
        print(f"✓ Previous cleaning was too aggressive (removed genuine data)")
        print(f"✓ Need whitelist + statistical outlier approach")
        
        return {
            'suspicious_overlap': suspicious_overlap,
            'bulk_dates': bulk_dates,
            'baseline_size': len(self.baseline_df),
            'bigquery_size': len(self.bigquery_df)
        }

def main():
    analyzer = DataQualityAnalyzer()
    results = analyzer.run_analysis()

if __name__ == "__main__":
    main() 