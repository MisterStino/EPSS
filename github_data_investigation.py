#!/usr/bin/env python3
"""
Comprehensive GitHub Data Investigation
======================================

Step-by-step analysis to understand:
1. What data we actually have
2. What constitutes genuine vs artificial activity
3. How to properly clean for LSTM training

Focus: Evidence-based analysis, not assumptions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GitHubDataInvestigator:
    def __init__(self):
        print("🔬 GITHUB DATA INVESTIGATION")
        print("=" * 50)
        print("Empirical analysis to understand data quality")
        print("=" * 50)
        
        # Data paths
        self.bigquery_path = "data/github/raw/github-final.csv"
        self.baseline_path = "data/github/raw/github_commit_timestamps_9k.csv"
        
    def load_datasets(self):
        """Load and examine both datasets"""
        print("\n📊 STEP 1: LOADING DATASETS")
        print("=" * 40)
        
        # Load BigQuery data
        print("Loading BigQuery dataset...")
        try:
            self.bigquery_df = pd.read_csv(self.bigquery_path)
            print(f"✓ BigQuery data loaded: {len(self.bigquery_df):,} records")
            print(f"  Columns: {list(self.bigquery_df.columns)}")
            print(f"  Date range: {self.bigquery_df['date'].min()} to {self.bigquery_df['date'].max()}")
            print(f"  Unique CVEs: {self.bigquery_df['cve'].nunique():,}")
        except Exception as e:
            print(f"❌ Failed to load BigQuery data: {e}")
            return False
            
        # Load baseline data
        print("\nLoading baseline (9k) dataset...")
        try:
            self.baseline_df = pd.read_csv(self.baseline_path)
            print(f"✓ Baseline data loaded: {len(self.baseline_df):,} records")
            print(f"  Columns: {list(self.baseline_df.columns)}")
            print(f"  Date range: {self.baseline_df['commit_date'].min()} to {self.baseline_df['commit_date'].max()}")
            print(f"  Unique CVEs: {self.baseline_df['cve_id'].nunique():,}")
        except Exception as e:
            print(f"❌ Failed to load baseline data: {e}")
            return False
            
        return True
    
    def analyze_data_structure(self):
        """Analyze the structure and patterns in both datasets"""
        print("\n🔍 STEP 2: DATA STRUCTURE ANALYSIS")
        print("=" * 45)
        
        # Standardize column names for comparison
        bigquery_std = self.bigquery_df.copy()
        baseline_std = self.baseline_df.copy()
        
        # Rename columns to match
        if 'cve_id' in baseline_std.columns:
            baseline_std = baseline_std.rename(columns={'cve_id': 'cve'})
        if 'commit_date' in baseline_std.columns:
            baseline_std = baseline_std.rename(columns={'commit_date': 'date'})
        if 'commit_count' in baseline_std.columns:
            baseline_std = baseline_std.rename(columns={'commit_count': 'commit_count'})
        
        print("BigQuery Dataset Structure:")
        print(bigquery_std.info())
        print(f"\nFirst 5 rows:")
        print(bigquery_std.head())
        
        print("\n" + "="*50)
        print("Baseline Dataset Structure:")
        print(baseline_std.info())
        print(f"\nFirst 5 rows:")
        print(baseline_std.head())
        
        # Store standardized versions
        self.bigquery_std = bigquery_std
        self.baseline_std = baseline_std
        
        return True
    
    def analyze_overlap(self):
        """Analyze overlap between datasets"""
        print("\n🔗 STEP 3: DATASET OVERLAP ANALYSIS")
        print("=" * 45)
        
        # CVE overlap
        bigquery_cves = set(self.bigquery_std['cve'].unique())
        baseline_cves = set(self.baseline_std['cve'].unique())
        
        overlap_cves = bigquery_cves.intersection(baseline_cves)
        
        print(f"CVE Overlap Analysis:")
        print(f"  BigQuery CVEs: {len(bigquery_cves):,}")
        print(f"  Baseline CVEs: {len(baseline_cves):,}")
        print(f"  Overlapping CVEs: {len(overlap_cves):,}")
        print(f"  Baseline coverage: {len(overlap_cves)/len(baseline_cves):.1%}")
        print(f"  BigQuery coverage: {len(overlap_cves)/len(bigquery_cves):.1%}")
        
        # Analyze records for overlapping CVEs
        bigquery_overlap = self.bigquery_std[self.bigquery_std['cve'].isin(overlap_cves)]
        baseline_overlap = self.baseline_std[self.baseline_std['cve'].isin(overlap_cves)]
        
        print(f"\nRecord Overlap for Common CVEs:")
        print(f"  BigQuery records: {len(bigquery_overlap):,}")
        print(f"  Baseline records: {len(baseline_overlap):,}")
        
        # Store overlap data
        self.overlap_cves = overlap_cves
        self.bigquery_overlap = bigquery_overlap
        self.baseline_overlap = baseline_overlap
        
        return True
    
    def analyze_activity_patterns(self):
        """Analyze activity patterns to identify genuine vs artificial signals"""
        print("\n📈 STEP 4: ACTIVITY PATTERN ANALYSIS")
        print("=" * 45)
        
        # Analyze BigQuery activity patterns
        print("BigQuery Activity Patterns:")
        
        # Check for feature columns in BigQuery
        feature_cols = [col for col in self.bigquery_std.columns 
                       if col not in ['cve', 'date'] and 'count' in col.lower()]
        
        if feature_cols:
            print(f"  Feature columns: {feature_cols}")
            
            # Analyze distributions
            for col in feature_cols:
                values = self.bigquery_std[col].dropna()
                print(f"  {col}:")
                print(f"    Range: {values.min()} - {values.max()}")
                print(f"    Mean: {values.mean():.2f}")
                print(f"    Median: {values.median():.2f}")
                print(f"    Zero rate: {(values == 0).mean():.1%}")
                
                # Most common values
                top_values = values.value_counts().head(5)
                print(f"    Top values: {dict(top_values)}")
        
        # Analyze baseline patterns
        print(f"\nBaseline Activity Patterns:")
        if 'commit_count' in self.baseline_std.columns:
            commit_counts = self.baseline_std['commit_count'].dropna()
            print(f"  Commit counts:")
            print(f"    Range: {commit_counts.min()} - {commit_counts.max()}")
            print(f"    Mean: {commit_counts.mean():.2f}")
            print(f"    Median: {commit_counts.median():.2f}")
            print(f"    Top values: {dict(commit_counts.value_counts().head(5))}")
        
        return True
    
    def analyze_temporal_patterns(self):
        """Analyze temporal distribution patterns"""
        print("\n📅 STEP 5: TEMPORAL PATTERN ANALYSIS")
        print("=" * 45)
        
        # Convert dates
        self.bigquery_std['date'] = pd.to_datetime(self.bigquery_std['date'])
        self.baseline_std['date'] = pd.to_datetime(self.baseline_std['date'])
        
        # Analyze BigQuery temporal patterns
        print("BigQuery Temporal Patterns:")
        bigquery_daily = self.bigquery_std.groupby('date').agg({
            'cve': 'nunique',
            'commit_count': 'sum' if 'commit_count' in self.bigquery_std.columns else 'count'
        }).reset_index()
        
        print(f"  Date range: {self.bigquery_std['date'].min()} to {self.bigquery_std['date'].max()}")
        print(f"  Total days: {len(bigquery_daily)}")
        print(f"  Avg CVEs per day: {bigquery_daily['cve'].mean():.1f}")
        print(f"  Max CVEs in one day: {bigquery_daily['cve'].max()}")
        
        # Find days with suspiciously high activity
        high_activity_threshold = bigquery_daily['cve'].quantile(0.95)
        high_activity_days = bigquery_daily[bigquery_daily['cve'] > high_activity_threshold]
        print(f"  Days with >95th percentile activity: {len(high_activity_days)}")
        print(f"  Top 5 highest activity days:")
        top_days = bigquery_daily.nlargest(5, 'cve')
        for _, row in top_days.iterrows():
            print(f"    {row['date'].date()}: {row['cve']} CVEs")
        
        # Analyze baseline temporal patterns
        print(f"\nBaseline Temporal Patterns:")
        baseline_daily = self.baseline_std.groupby('date').agg({
            'cve': 'nunique',
            'commit_count': 'sum' if 'commit_count' in self.baseline_std.columns else 'count'
        }).reset_index()
        
        print(f"  Date range: {self.baseline_std['date'].min()} to {self.baseline_std['date'].max()}")
        print(f"  Total days: {len(baseline_daily)}")
        print(f"  Avg CVEs per day: {baseline_daily['cve'].mean():.1f}")
        print(f"  Max CVEs in one day: {baseline_daily['cve'].max()}")
        
        # Store temporal data
        self.bigquery_daily = bigquery_daily
        self.baseline_daily = baseline_daily
        self.high_activity_days = high_activity_days
        
        return True
    
    def compare_overlapping_records(self):
        """Compare records for the same CVEs between datasets"""
        print("\n🔍 STEP 6: OVERLAPPING RECORD COMPARISON")
        print("=" * 50)
        
        # Sample some overlapping CVEs for detailed comparison
        sample_cves = list(self.overlap_cves)[:10]
        
        print(f"Detailed comparison for sample CVEs:")
        
        for cve in sample_cves:
            print(f"\n--- {cve} ---")
            
            # Get records from both datasets
            bq_records = self.bigquery_overlap[self.bigquery_overlap['cve'] == cve]
            bl_records = self.baseline_overlap[self.baseline_overlap['cve'] == cve]
            
            print(f"  BigQuery records: {len(bq_records)}")
            print(f"  Baseline records: {len(bl_records)}")
            
            if len(bq_records) > 0 and len(bl_records) > 0:
                # Compare date ranges
                bq_dates = pd.to_datetime(bq_records['date'])
                bl_dates = pd.to_datetime(bl_records['date'])
                
                print(f"  BigQuery date range: {bq_dates.min().date()} to {bq_dates.max().date()}")
                print(f"  Baseline date range: {bl_dates.min().date()} to {bl_dates.max().date()}")
                
                # Compare activity levels if possible
                if 'commit_count' in bq_records.columns and 'commit_count' in bl_records.columns:
                    bq_total = bq_records['commit_count'].sum()
                    bl_total = bl_records['commit_count'].sum()
                    print(f"  BigQuery total commits: {bq_total}")
                    print(f"  Baseline total commits: {bl_total}")
        
        return True
    
    def identify_suspicious_patterns(self):
        """Identify potentially artificial patterns in the data"""
        print("\n🚨 STEP 7: SUSPICIOUS PATTERN DETECTION")
        print("=" * 50)
        
        # Look for exact duplicate patterns in BigQuery data
        print("Analyzing BigQuery for suspicious patterns...")
        
        # Create activity signatures
        if all(col in self.bigquery_std.columns for col in ['commit_count', 'issue_count', 'comment_count', 'repo_count']):
            self.bigquery_std['signature'] = (
                self.bigquery_std['commit_count'].astype(str) + '_' +
                self.bigquery_std['issue_count'].astype(str) + '_' +
                self.bigquery_std['comment_count'].astype(str) + '_' +
                self.bigquery_std['repo_count'].astype(str)
            )
            
            # Analyze signature frequency
            signature_counts = self.bigquery_std['signature'].value_counts()
            
            print(f"  Total unique signatures: {len(signature_counts)}")
            print(f"  Most common signatures:")
            for sig, count in signature_counts.head(10).items():
                print(f"    {sig}: {count:,} occurrences")
                
                # Check how many CVEs have this signature
                cves_with_sig = self.bigquery_std[self.bigquery_std['signature'] == sig]['cve'].nunique()
                print(f"      → {cves_with_sig:,} unique CVEs")
            
            # Find signatures that appear across many CVEs (potentially artificial)
            signature_cve_counts = self.bigquery_std.groupby('signature')['cve'].nunique().sort_values(ascending=False)
            
            print(f"\n  Signatures appearing in most CVEs:")
            for sig, cve_count in signature_cve_counts.head(10).items():
                total_records = signature_counts[sig]
                print(f"    {sig}: {cve_count:,} CVEs, {total_records:,} total records")
                
                # This could indicate artificial bulk patterns
                if cve_count > 1000:  # Threshold for investigation
                    print(f"      ⚠️  SUSPICIOUS: Same pattern across {cve_count:,} CVEs")
        
        return True
    
    def run_investigation(self):
        """Run the complete investigation"""
        print("🚀 Starting comprehensive GitHub data investigation...")
        
        steps = [
            self.load_datasets,
            self.analyze_data_structure,
            self.analyze_overlap,
            self.analyze_activity_patterns,
            self.analyze_temporal_patterns,
            self.compare_overlapping_records,
            self.identify_suspicious_patterns
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"\n{'='*60}")
            try:
                if not step():
                    print(f"❌ Step {i} failed")
                    return False
            except Exception as e:
                print(f"❌ Step {i} failed with error: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n{'='*60}")
        print("🎉 Investigation completed successfully!")
        
        return True

def main():
    investigator = GitHubDataInvestigator()
    investigator.run_investigation()

if __name__ == "__main__":
    main() 