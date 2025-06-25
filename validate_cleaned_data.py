#!/usr/bin/env python3
"""
Cleaned Data Validation
======================

Validate the quality of our evidence-based cleaned GitHub data:
1. Verify baseline preservation
2. Check data quality metrics
3. Compare with original datasets
4. Ensure LSTM training readiness
"""

import pandas as pd
import numpy as np
from datetime import datetime

class CleanedDataValidator:
    def __init__(self):
        print("🔍 CLEANED DATA VALIDATION")
        print("=" * 40)
        print("Verifying evidence-based cleaning results")
        print("=" * 40)
        
        # Load datasets
        print("Loading datasets...")
        self.cleaned_df = pd.read_csv("github_cleaned_evidence_based.csv")
        self.baseline_df = pd.read_csv("data/github/raw/github_commit_timestamps_9k.csv")
        self.original_df = pd.read_csv("data/github/raw/github-final.csv")
        
        # Standardize column names
        self.baseline_df = self.baseline_df.rename(columns={
            'cve_id': 'cve', 
            'commit_date': 'date'
        })
        
        # Convert dates
        for df in [self.cleaned_df, self.baseline_df, self.original_df]:
            df['date'] = pd.to_datetime(df['date'])
        
        print(f"✓ Cleaned data: {len(self.cleaned_df):,} records, {self.cleaned_df['cve'].nunique():,} CVEs")
        print(f"✓ Baseline data: {len(self.baseline_df):,} records, {self.baseline_df['cve'].nunique():,} CVEs")
        print(f"✓ Original data: {len(self.original_df):,} records, {self.original_df['cve'].nunique():,} CVEs")
        
    def validate_baseline_preservation(self):
        """Validate that all baseline CVEs are preserved"""
        print("\\n🔒 BASELINE PRESERVATION VALIDATION")
        print("=" * 45)
        
        baseline_cves = set(self.baseline_df['cve'].unique())
        cleaned_cves = set(self.cleaned_df['cve'].unique())
        
        preserved_baseline = baseline_cves.intersection(cleaned_cves)
        missing_baseline = baseline_cves - cleaned_cves
        
        print(f"Baseline preservation check:")
        print(f"  Original baseline CVEs: {len(baseline_cves):,}")
        print(f"  Preserved in cleaned data: {len(preserved_baseline):,}")
        print(f"  Missing from cleaned data: {len(missing_baseline):,}")
        print(f"  Preservation rate: {len(preserved_baseline) / len(baseline_cves) * 100:.1f}%")
        
        if len(missing_baseline) > 0:
            print(f"  ⚠️  Missing CVEs: {list(missing_baseline)[:10]}...")
        else:
            print(f"  ✅ Perfect baseline preservation!")
        
        # Check record counts for preserved CVEs
        baseline_overlap = self.baseline_df[self.baseline_df['cve'].isin(preserved_baseline)]
        cleaned_overlap = self.cleaned_df[self.cleaned_df['cve'].isin(preserved_baseline)]
        
        print(f"\\nRecord preservation for baseline CVEs:")
        print(f"  Baseline records: {len(baseline_overlap):,}")
        print(f"  Cleaned records: {len(cleaned_overlap):,}")
        print(f"  Record ratio: {len(cleaned_overlap) / len(baseline_overlap):.2f}")
        
        return len(preserved_baseline) / len(baseline_cves)
    
    def validate_data_quality(self):
        """Validate overall data quality metrics"""
        print("\\n📊 DATA QUALITY VALIDATION")
        print("=" * 35)
        
        # Temporal distribution
        print("Temporal distribution:")
        print(f"  Date range: {self.cleaned_df['date'].min().date()} to {self.cleaned_df['date'].max().date()}")
        print(f"  Unique dates: {self.cleaned_df['date'].nunique():,}")
        print(f"  Avg records per day: {len(self.cleaned_df) / self.cleaned_df['date'].nunique():.1f}")
        
        # Activity metrics
        print(f"\\nActivity metrics:")
        for col in ['commit_count', 'issue_count', 'comment_count', 'repo_count']:
            if col in self.cleaned_df.columns:
                values = self.cleaned_df[col]
                print(f"  {col}:")
                print(f"    Range: {values.min()} - {values.max()}")
                print(f"    Mean: {values.mean():.2f}")
                print(f"    Zero rate: {(values == 0).mean():.1%}")
        
        # CVE activity spans
        cve_spans = self.cleaned_df.groupby('cve')['date'].agg(['min', 'max', 'count'])
        cve_spans['span_days'] = (cve_spans['max'] - cve_spans['min']).dt.days
        
        print(f"\\nCVE activity spans:")
        print(f"  Mean span: {cve_spans['span_days'].mean():.1f} days")
        print(f"  Median span: {cve_spans['span_days'].median():.1f} days")
        print(f"  CVEs with 0-day span: {(cve_spans['span_days'] == 0).sum():,}")
        print(f"  CVEs with >30-day span: {(cve_spans['span_days'] > 30).sum():,}")
        
        # Check for remaining bulk patterns
        daily_activity = self.cleaned_df.groupby('date')['cve'].nunique()
        q99 = daily_activity.quantile(0.99)
        bulk_days = daily_activity[daily_activity > q99]
        
        print(f"\\nBulk pattern check:")
        print(f"  99th percentile daily activity: {q99:.0f} CVEs")
        print(f"  Days above 99th percentile: {len(bulk_days)}")
        if len(bulk_days) > 0:
            print(f"  Top bulk days remaining:")
            for date, count in bulk_days.nlargest(5).items():
                print(f"    {date.date()}: {count:,} CVEs")
        
    def validate_lstm_readiness(self):
        """Validate readiness for LSTM training"""
        print("\\n🤖 LSTM TRAINING READINESS")
        print("=" * 35)
        
        # Check data completeness
        print("Data completeness:")
        for col in self.cleaned_df.columns:
            null_count = self.cleaned_df[col].isnull().sum()
            null_rate = null_count / len(self.cleaned_df) * 100
            print(f"  {col}: {null_rate:.1f}% null")
        
        # Check feature distributions
        print(f"\\nFeature suitability:")
        numeric_cols = ['commit_count', 'issue_count', 'comment_count', 'repo_count']
        for col in numeric_cols:
            if col in self.cleaned_df.columns:
                values = self.cleaned_df[col]
                non_zero_rate = (values > 0).mean()
                print(f"  {col}: {non_zero_rate:.1%} non-zero (good signal)")
        
        # Check temporal consistency
        print(f"\\nTemporal consistency:")
        date_gaps = self.cleaned_df['date'].sort_values().diff().dt.days.dropna()
        print(f"  Date gaps (days): min={date_gaps.min()}, max={date_gaps.max()}, mean={date_gaps.mean():.1f}")
        
        # Check CVE representation
        cve_record_counts = self.cleaned_df['cve'].value_counts()
        print(f"\\nCVE representation:")
        print(f"  CVEs with 1 record: {(cve_record_counts == 1).sum():,}")
        print(f"  CVEs with 2-5 records: {((cve_record_counts >= 2) & (cve_record_counts <= 5)).sum():,}")
        print(f"  CVEs with 6+ records: {(cve_record_counts >= 6).sum():,}")
        print(f"  Max records per CVE: {cve_record_counts.max()}")
        
        # Overall readiness score
        readiness_score = 0
        readiness_checks = []
        
        # Check 1: No null values in key columns
        key_cols = ['date', 'cve', 'commit_count']
        if all(self.cleaned_df[col].isnull().sum() == 0 for col in key_cols):
            readiness_score += 25
            readiness_checks.append("✅ No null values in key columns")
        else:
            readiness_checks.append("❌ Null values found in key columns")
        
        # Check 2: Reasonable temporal span
        if self.cleaned_df['date'].nunique() > 365:  # More than 1 year of data
            readiness_score += 25
            readiness_checks.append("✅ Sufficient temporal coverage")
        else:
            readiness_checks.append("❌ Limited temporal coverage")
        
        # Check 3: Sufficient CVE diversity
        if self.cleaned_df['cve'].nunique() > 10000:  # More than 10k CVEs
            readiness_score += 25
            readiness_checks.append("✅ Sufficient CVE diversity")
        else:
            readiness_checks.append("❌ Limited CVE diversity")
        
        # Check 4: Balanced activity distribution
        non_zero_activity = (self.cleaned_df['commit_count'] > 0).mean()
        if 0.3 <= non_zero_activity <= 0.8:  # 30-80% non-zero activity
            readiness_score += 25
            readiness_checks.append("✅ Balanced activity distribution")
        else:
            readiness_checks.append("❌ Imbalanced activity distribution")
        
        print(f"\\nLSTM Readiness Assessment:")
        for check in readiness_checks:
            print(f"  {check}")
        print(f"\\nOverall Readiness Score: {readiness_score}/100")
        
        if readiness_score >= 75:
            print("🎉 EXCELLENT: Dataset is ready for LSTM training!")
        elif readiness_score >= 50:
            print("⚠️  GOOD: Dataset is suitable with minor considerations")
        else:
            print("❌ POOR: Dataset needs further cleaning")
        
        return readiness_score
    
    def compare_with_original(self):
        """Compare cleaned data with original BigQuery data"""
        print("\\n📈 COMPARISON WITH ORIGINAL DATA")
        print("=" * 40)
        
        print("Size comparison:")
        print(f"  Original records: {len(self.original_df):,}")
        print(f"  Cleaned records: {len(self.cleaned_df):,}")
        print(f"  Retention rate: {len(self.cleaned_df) / len(self.original_df) * 100:.1f}%")
        
        print(f"\\nCVE comparison:")
        print(f"  Original CVEs: {self.original_df['cve'].nunique():,}")
        print(f"  Cleaned CVEs: {self.cleaned_df['cve'].nunique():,}")
        print(f"  CVE retention rate: {self.cleaned_df['cve'].nunique() / self.original_df['cve'].nunique() * 100:.1f}%")
        
                          # Activity comparison
         print(f"\\nActivity comparison:")
         for col in ['commit_count', 'issue_count', 'comment_count', 'repo_count']:
             if col in self.cleaned_df.columns and col in self.original_df.columns:
                 orig_total = self.original_df[col].sum()
                 clean_total = self.cleaned_df[col].sum()
                 print(f"  {col}: {clean_total:,} / {orig_total:,} ({clean_total/orig_total*100:.1f}%)")
        
        # Quality improvement metrics
        orig_zero_commits = (self.original_df['commit_count'] == 0).mean()
        clean_zero_commits = (self.cleaned_df['commit_count'] == 0).mean()
        
        print(f"\\nQuality improvements:")
        print(f"  Zero commit rate: {orig_zero_commits:.1%} → {clean_zero_commits:.1%}")
        print(f"  Signal improvement: {(orig_zero_commits - clean_zero_commits)*100:.1f} percentage points")
        
    def run_validation(self):
        """Run complete validation"""
        print("🚀 Starting cleaned data validation...")
        
        preservation_rate = self.validate_baseline_preservation()
        self.validate_data_quality()
        readiness_score = self.validate_lstm_readiness()
        self.compare_with_original()
        
        print(f"\\n{'='*60}")
        print("🏁 VALIDATION SUMMARY")
        print("=" * 20)
        print(f"✓ Baseline preservation: {preservation_rate*100:.1f}%")
        print(f"✓ LSTM readiness score: {readiness_score}/100")
        print(f"✓ Data retention: {len(self.cleaned_df) / len(self.original_df) * 100:.1f}%")
        
        if preservation_rate >= 0.95 and readiness_score >= 75:
            print("\\n🎉 VALIDATION PASSED: Dataset is high-quality and LSTM-ready!")
        elif preservation_rate >= 0.90 and readiness_score >= 50:
            print("\\n✅ VALIDATION GOOD: Dataset is suitable for training")
        else:
            print("\\n⚠️  VALIDATION CONCERNS: Review results before training")
        
        return {
            'preservation_rate': preservation_rate,
            'readiness_score': readiness_score,
            'retention_rate': len(self.cleaned_df) / len(self.original_df)
        }

def main():
    validator = CleanedDataValidator()
    results = validator.run_validation()

if __name__ == "__main__":
    main() 