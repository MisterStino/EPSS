#!/usr/bin/env python3
"""
Cleaned Data Validation
======================

Validate the quality of our evidence-based cleaned GitHub data
"""

import pandas as pd
import numpy as np
from datetime import datetime

class CleanedDataValidator:
    def __init__(self):
        print("🔍 CLEANED DATA VALIDATION")
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
        print("\n🔒 BASELINE PRESERVATION VALIDATION")
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
            print(f"  ⚠️  Missing CVEs: {list(missing_baseline)[:5]}...")
        else:
            print(f"  ✅ Perfect baseline preservation!")
        
        return len(preserved_baseline) / len(baseline_cves)
    
    def validate_data_quality(self):
        """Validate overall data quality metrics"""
        print("\n📊 DATA QUALITY VALIDATION")
        print("=" * 35)
        
        # Temporal distribution
        print("Temporal distribution:")
        print(f"  Date range: {self.cleaned_df['date'].min().date()} to {self.cleaned_df['date'].max().date()}")
        print(f"  Unique dates: {self.cleaned_df['date'].nunique():,}")
        print(f"  Avg records per day: {len(self.cleaned_df) / self.cleaned_df['date'].nunique():.1f}")
        
        # Activity metrics
        print(f"\nActivity metrics:")
        for col in ['commit_count', 'issue_count', 'comment_count', 'repo_count']:
            if col in self.cleaned_df.columns:
                values = self.cleaned_df[col]
                print(f"  {col}:")
                print(f"    Range: {values.min()} - {values.max()}")
                print(f"    Mean: {values.mean():.2f}")
                print(f"    Zero rate: {(values == 0).mean():.1%}")
        
    def validate_lstm_readiness(self):
        """Validate readiness for LSTM training"""
        print("\n🤖 LSTM TRAINING READINESS")
        print("=" * 35)
        
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
        
        print(f"LSTM Readiness Assessment:")
        for check in readiness_checks:
            print(f"  {check}")
        print(f"\nOverall Readiness Score: {readiness_score}/100")
        
        if readiness_score >= 75:
            print("🎉 EXCELLENT: Dataset is ready for LSTM training!")
        elif readiness_score >= 50:
            print("⚠️  GOOD: Dataset is suitable with minor considerations")
        else:
            print("❌ POOR: Dataset needs further cleaning")
        
        return readiness_score
    
    def compare_with_original(self):
        """Compare cleaned data with original BigQuery data"""
        print("\n📈 COMPARISON WITH ORIGINAL DATA")
        print("=" * 40)
        
        print("Size comparison:")
        print(f"  Original records: {len(self.original_df):,}")
        print(f"  Cleaned records: {len(self.cleaned_df):,}")
        print(f"  Retention rate: {len(self.cleaned_df) / len(self.original_df) * 100:.1f}%")
        
        print(f"\nCVE comparison:")
        print(f"  Original CVEs: {self.original_df['cve'].nunique():,}")
        print(f"  Cleaned CVEs: {self.cleaned_df['cve'].nunique():,}")
        print(f"  CVE retention rate: {self.cleaned_df['cve'].nunique() / self.original_df['cve'].nunique() * 100:.1f}%")
        
        # Quality improvement metrics
        orig_zero_commits = (self.original_df['commit_count'] == 0).mean()
        clean_zero_commits = (self.cleaned_df['commit_count'] == 0).mean()
        
        print(f"\nQuality improvements:")
        print(f"  Zero commit rate: {orig_zero_commits:.1%} → {clean_zero_commits:.1%}")
        print(f"  Signal improvement: {(orig_zero_commits - clean_zero_commits)*100:.1f} percentage points")
        
    def run_validation(self):
        """Run complete validation"""
        print("🚀 Starting cleaned data validation...")
        
        preservation_rate = self.validate_baseline_preservation()
        self.validate_data_quality()
        readiness_score = self.validate_lstm_readiness()
        self.compare_with_original()
        
        print(f"\n{'='*60}")
        print("🏁 VALIDATION SUMMARY")
        print("=" * 20)
        print(f"✓ Baseline preservation: {preservation_rate*100:.1f}%")
        print(f"✓ LSTM readiness score: {readiness_score}/100")
        print(f"✓ Data retention: {len(self.cleaned_df) / len(self.original_df) * 100:.1f}%")
        
        if preservation_rate >= 0.95 and readiness_score >= 75:
            print("\n🎉 VALIDATION PASSED: Dataset is high-quality and LSTM-ready!")
        elif preservation_rate >= 0.90 and readiness_score >= 50:
            print("\n✅ VALIDATION GOOD: Dataset is suitable for training")
        else:
            print("\n⚠️  VALIDATION CONCERNS: Review results before training")
        
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