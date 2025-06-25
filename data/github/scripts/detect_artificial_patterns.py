#!/usr/bin/env python
"""
Artificial Pattern Detection in GitHub Data
==========================================

Systematically detects:
1. Bot Activity: Identical patterns across multiple CVEs
2. Database Migrations: Massive spikes on specific dates (batch processing)
3. Scraping Artifacts: Data collection errors creating false signals

Usage:
    python -m data.github.scripts.detect_artificial_patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Use project's battle-tested Spark session
from t3_spark.session import get_spark_session

class ArtificalPatternDetector:
    """Detect artificial patterns in GitHub-CVE activity data."""
    
    def __init__(self):
        self.spark = get_spark_session()
        self.github_path = "data/github/raw/github-final.csv"
        self.output_dir = Path("data/github/scripts/pattern_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔍 Artificial Pattern Detector Initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_github_data(self):
        """Load GitHub data using Spark."""
        print("\n📊 LOADING GITHUB DATA WITH SPARK")
        print("=" * 50)
        
        # Load GitHub data
        self.github_spark = self.spark.read.option("header", "true").option("inferSchema", "true").csv(self.github_path)
        self.github_spark.cache()  # Cache for repeated analysis
        
        print(f"✓ GitHub records: {self.github_spark.count():,}")
        print(f"✓ GitHub columns: {self.github_spark.columns}")
        
        # Basic data quality check
        from pyspark.sql import functions as F
        
        print("\n📈 BASIC DATA STATISTICS:")
        stats = self.github_spark.select([
            F.count("*").alias("total_records"),
            F.countDistinct("cve").alias("unique_cves"),
            F.countDistinct("date").alias("unique_dates"),
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date")
        ]).collect()[0]
        
        print(f"   Total records: {stats.total_records:,}")
        print(f"   Unique CVEs: {stats.unique_cves:,}")
        print(f"   Unique dates: {stats.unique_dates:,}")
        print(f"   Date range: {stats.min_date} to {stats.max_date}")
        
        return self.github_spark
    
    def detect_identical_patterns(self):
        """Detect CVEs with identical activity patterns using Spark."""
        print("\n🤖 DETECTING IDENTICAL PATTERNS (BOT ACTIVITY)")
        print("=" * 60)
        
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
        
        # Create activity signatures for each CVE
        print("Creating activity signatures for each CVE...")
        
        # For each CVE, create a concatenated string of all daily activities
        cve_signatures = self.github_spark.groupBy("cve").agg(
            F.collect_list(
                F.concat_ws("_", 
                    F.col("date").cast("string"),
                    F.col("commit_count").cast("string"),
                    F.col("issue_count").cast("string"), 
                    F.col("comment_count").cast("string"),
                    F.col("repo_count").cast("string")
                )
            ).alias("activity_signature")
        )
        
        # Convert list to string for comparison
        cve_signatures = cve_signatures.withColumn(
            "signature_string", 
            F.concat_ws("|", F.col("activity_signature"))
        )
        
        # Find duplicate signatures
        print("Finding duplicate activity signatures...")
        signature_counts = cve_signatures.groupBy("signature_string").agg(
            F.count("*").alias("signature_count"),
            F.collect_list("cve").alias("cves_with_signature")
        ).filter(F.col("signature_count") > 1).orderBy(F.col("signature_count").desc())
        
        # Collect results for analysis
        duplicate_signatures = signature_counts.collect()
        
        print(f"✓ Found {len(duplicate_signatures)} duplicate signature groups")
        
        # Analyze the results
        total_duplicate_cves = 0
        signature_analysis = []
        
        for row in duplicate_signatures:
            count = row.signature_count
            cves = row.cves_with_signature
            total_duplicate_cves += count
            
            signature_analysis.append({
                'signature_hash': hash(row.signature_string),
                'cve_count': count,
                'sample_cves': cves[:5],  # First 5 CVEs as examples
                'all_cves': cves
            })
        
        print(f"✓ Total CVEs with duplicate patterns: {total_duplicate_cves:,}")
        print(f"✓ Largest duplicate group: {max([s['cve_count'] for s in signature_analysis]) if signature_analysis else 0} CVEs")
        
        # Show top duplicate groups
        print(f"\n📊 TOP 10 DUPLICATE SIGNATURE GROUPS:")
        for i, sig in enumerate(signature_analysis[:10]):
            print(f"   {i+1}. {sig['cve_count']:,} CVEs with identical patterns")
            print(f"      Sample CVEs: {', '.join(sig['sample_cves'][:3])}")
        
        return signature_analysis
    
    def detect_batch_processing_dates(self):
        """Detect dates with suspicious mass activity across multiple CVEs."""
        print("\n🗄️ DETECTING BATCH PROCESSING DATES")
        print("=" * 50)
        
        from pyspark.sql import functions as F
        
        # Calculate daily statistics across all CVEs
        print("Calculating daily activity statistics...")
        
        daily_stats = self.github_spark.groupBy("date").agg(
            F.sum("commit_count").alias("total_commits"),
            F.sum("issue_count").alias("total_issues"),
            F.sum("comment_count").alias("total_comments"),
            F.sum("repo_count").alias("total_repos"),
            F.countDistinct("cve").alias("active_cves"),
            F.count("*").alias("total_records"),
            F.avg("commit_count").alias("avg_commits_per_cve"),
            F.avg("issue_count").alias("avg_issues_per_cve"),
            F.avg("comment_count").alias("avg_comments_per_cve"),
            F.avg("repo_count").alias("avg_repos_per_cve")
        ).orderBy("date")
        
        # Convert to Pandas for statistical analysis
        daily_stats_pd = daily_stats.toPandas()
        daily_stats_pd['date'] = pd.to_datetime(daily_stats_pd['date'])
        
        print(f"✓ Analyzed {len(daily_stats_pd)} unique dates")
        
        # Calculate z-scores for spike detection
        numeric_cols = ['total_commits', 'total_issues', 'total_comments', 'total_repos', 'active_cves']
        
        spike_analysis = {}
        for col in numeric_cols:
            mean_val = daily_stats_pd[col].mean()
            std_val = daily_stats_pd[col].std()
            daily_stats_pd[f'{col}_zscore'] = (daily_stats_pd[col] - mean_val) / std_val
            
            # Find extreme spikes (>3 standard deviations)
            spikes = daily_stats_pd[daily_stats_pd[f'{col}_zscore'] > 3].copy()
            spike_analysis[col] = {
                'spike_dates': spikes['date'].tolist(),
                'spike_values': spikes[col].tolist(),
                'spike_zscores': spikes[f'{col}_zscore'].tolist()
            }
        
        # Find dates that are spikes across multiple features
        all_spike_dates = set()
        for col_data in spike_analysis.values():
            all_spike_dates.update(col_data['spike_dates'])
        
        # Analyze multi-feature spikes
        multi_feature_spikes = []
        for date in all_spike_dates:
            date_row = daily_stats_pd[daily_stats_pd['date'] == date].iloc[0]
            spike_features = []
            
            for col in numeric_cols:
                if date_row[f'{col}_zscore'] > 3:
                    spike_features.append(col)
            
            if len(spike_features) >= 2:  # Spike in multiple features
                multi_feature_spikes.append({
                    'date': date,
                    'spike_features': spike_features,
                    'active_cves': date_row['active_cves'],
                    'total_commits': date_row['total_commits'],
                    'total_issues': date_row['total_issues'],
                    'total_comments': date_row['total_comments']
                })
        
        print(f"\n📊 SPIKE DETECTION RESULTS:")
        print(f"   Dates with extreme spikes: {len(all_spike_dates)}")
        print(f"   Dates with multi-feature spikes: {len(multi_feature_spikes)}")
        
        # Show top suspicious dates
        multi_feature_spikes.sort(key=lambda x: x['active_cves'], reverse=True)
        
        print(f"\n🚨 TOP 10 MOST SUSPICIOUS BATCH PROCESSING DATES:")
        for i, spike in enumerate(multi_feature_spikes[:10]):
            print(f"   {i+1}. {spike['date'].date()}")
            print(f"      CVEs affected: {spike['active_cves']:,}")
            print(f"      Spike features: {', '.join(spike['spike_features'])}")
            print(f"      Total commits: {spike['total_commits']:,}")
        
        return spike_analysis, multi_feature_spikes, daily_stats_pd
    
    def analyze_cve_cooccurrence_on_spikes(self, multi_feature_spikes):
        """Analyze which CVEs appear together on suspicious spike dates."""
        print("\n🔗 ANALYZING CVE CO-OCCURRENCE ON SPIKE DATES")
        print("=" * 55)
        
        from pyspark.sql import functions as F
        
        # Get top 5 most suspicious dates
        top_spike_dates = [spike['date'].strftime('%Y-%m-%d') for spike in multi_feature_spikes[:5]]
        
        cooccurrence_analysis = {}
        
        for date_str in top_spike_dates:
            print(f"\nAnalyzing date: {date_str}")
            
            # Get all CVEs active on this date
            date_cves = self.github_spark.filter(F.col("date") == date_str).select("cve").distinct()
            active_cves_list = [row.cve for row in date_cves.collect()]
            
            print(f"   CVEs active on {date_str}: {len(active_cves_list):,}")
            
            # Check how many of these CVEs are ONLY active on spike dates
            suspicious_cves = []
            
            for cve in active_cves_list[:100]:  # Sample first 100 for performance
                # Get all dates when this CVE was active
                cve_dates = self.github_spark.filter(F.col("cve") == cve).select("date").distinct()
                cve_dates_list = [row.date for row in cve_dates.collect()]
                
                # Check if all dates are spike dates
                if all(str(d) in [s['date'].strftime('%Y-%m-%d') for s in multi_feature_spikes] for d in cve_dates_list):
                    suspicious_cves.append(cve)
            
            cooccurrence_analysis[date_str] = {
                'total_active_cves': len(active_cves_list),
                'sampled_cves': min(100, len(active_cves_list)),
                'suspicious_cves': suspicious_cves,
                'suspicion_rate': len(suspicious_cves) / min(100, len(active_cves_list)) if active_cves_list else 0
            }
            
            print(f"   Suspicious CVEs (only active on spike dates): {len(suspicious_cves)}")
            print(f"   Suspicion rate: {cooccurrence_analysis[date_str]['suspicion_rate']:.1%}")
        
        return cooccurrence_analysis
    
    def detect_scraping_artifacts(self):
        """Detect logical inconsistencies and scraping errors."""
        print("\n🕷️ DETECTING SCRAPING ARTIFACTS")
        print("=" * 40)
        
        from pyspark.sql import functions as F
        
        artifacts = {}
        
        # 1. Logical inconsistencies
        print("Checking logical inconsistencies...")
        
        # Impossible repo counts (more repos than total activity)
        impossible_repos = self.github_spark.filter(
            F.col("repo_count") > (
                F.col("commit_count") + F.col("issue_count") + F.col("comment_count")
            )
        )
        impossible_count = impossible_repos.count()
        artifacts['impossible_repo_counts'] = impossible_count
        
        # Negative values
        negative_commits = self.github_spark.filter(F.col("commit_count") < 0).count()
        negative_issues = self.github_spark.filter(F.col("issue_count") < 0).count()
        negative_comments = self.github_spark.filter(F.col("comment_count") < 0).count()
        negative_repos = self.github_spark.filter(F.col("repo_count") < 0).count()
        
        artifacts['negative_values'] = {
            'commits': negative_commits,
            'issues': negative_issues,
            'comments': negative_comments,
            'repos': negative_repos
        }
        
        # 2. Extreme outliers (beyond physical possibility)
        print("Checking extreme outliers...")
        
        limits = {
            'commit_count': 100,
            'issue_count': 50,
            'comment_count': 200,
            'repo_count': 20
        }
        
        outliers = {}
        for col, limit in limits.items():
            extreme_count = self.github_spark.filter(F.col(col) > limit).count()
            outliers[col] = {
                'count': extreme_count,
                'limit': limit
            }
            
            # Get some examples
            if extreme_count > 0:
                examples = self.github_spark.filter(F.col(col) > limit).select("cve", "date", col).limit(5).collect()
                outliers[col]['examples'] = [(row.cve, str(row.date), getattr(row, col)) for row in examples]
        
        artifacts['extreme_outliers'] = outliers
        
        # 3. Duplicate detection
        print("Checking for duplicates...")
        
        total_records = self.github_spark.count()
        unique_records = self.github_spark.distinct().count()
        exact_duplicates = total_records - unique_records
        
        # Partial duplicates (same CVE + date, different values)
        unique_cve_date = self.github_spark.select("cve", "date").distinct().count()
        total_cve_date = self.github_spark.select("cve", "date").count()
        partial_duplicates = total_cve_date - unique_cve_date
        
        artifacts['duplicates'] = {
            'exact_duplicates': exact_duplicates,
            'partial_duplicates': partial_duplicates,
            'total_records': total_records
        }
        
        print(f"\n📊 SCRAPING ARTIFACTS SUMMARY:")
        print(f"   Impossible repo counts: {impossible_count:,}")
        print(f"   Negative values: {sum(artifacts['negative_values'].values()):,}")
        print(f"   Extreme outliers: {sum(o['count'] for o in outliers.values()):,}")
        print(f"   Exact duplicates: {exact_duplicates:,}")
        print(f"   Partial duplicates: {partial_duplicates:,}")
        
        return artifacts
    
    def create_summary_report(self, identical_patterns, spike_analysis, cooccurrence_analysis, scraping_artifacts):
        """Create comprehensive analysis report."""
        print("\n📋 CREATING COMPREHENSIVE ANALYSIS REPORT")
        print("=" * 50)
        
        report = f"""
Artificial Pattern Detection Report
==================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
================

1. BOT ACTIVITY (Identical Patterns):
   - Duplicate signature groups found: {len(identical_patterns)}
   - Total CVEs with duplicate patterns: {sum(p['cve_count'] for p in identical_patterns):,}
   - Largest duplicate group: {max([p['cve_count'] for p in identical_patterns]) if identical_patterns else 0} CVEs
   - Assessment: {'HIGH CONCERN' if len(identical_patterns) > 100 else 'MODERATE CONCERN' if len(identical_patterns) > 10 else 'LOW CONCERN'}

2. BATCH PROCESSING (Mass Spikes):
   - Suspicious spike dates identified: {len([d for dates in spike_analysis.values() for d in dates['spike_dates']])}
   - Multi-feature spike dates: {len([d for d in cooccurrence_analysis.keys()])}
   - CVEs affected by batch processing: {sum(c['total_active_cves'] for c in cooccurrence_analysis.values()):,}
   - Assessment: {'HIGH CONCERN' if len(cooccurrence_analysis) > 5 else 'MODERATE CONCERN' if len(cooccurrence_analysis) > 2 else 'LOW CONCERN'}

3. SCRAPING ARTIFACTS:
   - Impossible repo counts: {scraping_artifacts['impossible_repo_counts']:,}
   - Negative values: {sum(scraping_artifacts['negative_values'].values()):,}
   - Extreme outliers: {sum(o['count'] for o in scraping_artifacts['extreme_outliers'].values()):,}
   - Exact duplicates: {scraping_artifacts['duplicates']['exact_duplicates']:,}
   - Partial duplicates: {scraping_artifacts['duplicates']['partial_duplicates']:,}
   - Assessment: {'HIGH CONCERN' if scraping_artifacts['duplicates']['exact_duplicates'] > 1000 else 'MODERATE CONCERN' if scraping_artifacts['duplicates']['exact_duplicates'] > 100 else 'LOW CONCERN'}

DETAILED FINDINGS:
=================

Bot Activity Analysis:
---------------------
"""
        
        # Add detailed bot analysis
        if identical_patterns:
            report += f"\nTop 5 Duplicate Pattern Groups:\n"
            for i, pattern in enumerate(identical_patterns[:5]):
                report += f"{i+1}. {pattern['cve_count']:,} CVEs with identical patterns\n"
                report += f"   Sample CVEs: {', '.join(pattern['sample_cves'][:3])}\n"
        
        # Add batch processing details
        report += f"\nBatch Processing Analysis:\n"
        report += f"-------------------------\n"
        for date, analysis in cooccurrence_analysis.items():
            report += f"Date: {date}\n"
            report += f"  Active CVEs: {analysis['total_active_cves']:,}\n"
            report += f"  Suspicious CVEs: {len(analysis['suspicious_cves'])}\n"
            report += f"  Suspicion Rate: {analysis['suspicion_rate']:.1%}\n\n"
        
        # Add scraping artifacts details
        report += f"Scraping Artifacts Analysis:\n"
        report += f"---------------------------\n"
        for col, data in scraping_artifacts['extreme_outliers'].items():
            if data['count'] > 0:
                report += f"{col}: {data['count']:,} records exceed limit of {data['limit']}\n"
                if 'examples' in data:
                    report += f"  Examples: {data['examples'][:3]}\n"
        
        report += f"""

RECOMMENDATIONS:
===============

1. Data Cleaning Priority:
   - Remove CVEs with identical patterns (bot activity)
   - Flag dates with batch processing artifacts
   - Clean logical inconsistencies and extreme outliers

2. Model Training Impact:
   - Clean data expected to improve EPSS forecasting accuracy
   - Artificial patterns would create false correlations
   - Recommend using only high-quality CVEs for LSTM training

3. Next Steps:
   - Implement data cleaning based on these findings
   - Validate cleaned dataset with sample plots
   - Re-run analysis after cleaning to verify improvements

Analysis completed with Spark for efficient big data processing.
"""
        
        # Save report
        report_path = self.output_dir / "artificial_pattern_analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Comprehensive report saved: {report_path}")
        print("\n🎯 KEY FINDINGS:")
        print(f"   Bot patterns: {len(identical_patterns)} duplicate groups")
        print(f"   Batch processing dates: {len(cooccurrence_analysis)}")
        print(f"   Scraping artifacts: {scraping_artifacts['duplicates']['exact_duplicates']:,} exact duplicates")
        
        return report_path
    
    def run_full_analysis(self):
        """Run complete artificial pattern detection analysis."""
        print("🚀 STARTING COMPREHENSIVE ARTIFICIAL PATTERN ANALYSIS")
        print("=" * 80)
        
        try:
            # Step 1: Load data
            self.load_github_data()
            
            # Step 2: Detect identical patterns (bot activity)
            identical_patterns = self.detect_identical_patterns()
            
            # Step 3: Detect batch processing dates
            spike_analysis, multi_feature_spikes, daily_stats = self.detect_batch_processing_dates()
            
            # Step 4: Analyze CVE co-occurrence on spike dates
            cooccurrence_analysis = self.analyze_cve_cooccurrence_on_spikes(multi_feature_spikes)
            
            # Step 5: Detect scraping artifacts
            scraping_artifacts = self.detect_scraping_artifacts()
            
            # Step 6: Create comprehensive report
            report_path = self.create_summary_report(
                identical_patterns, spike_analysis, cooccurrence_analysis, scraping_artifacts
            )
            
            print(f"\n🎉 ANALYSIS COMPLETE!")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📋 Full report: {report_path}")
            
            return {
                'identical_patterns': identical_patterns,
                'spike_analysis': spike_analysis,
                'cooccurrence_analysis': cooccurrence_analysis,
                'scraping_artifacts': scraping_artifacts,
                'report_path': report_path
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            raise
        
        finally:
            # Clean up Spark session
            self.spark.stop()

def main():
    """Main entry point."""
    detector = ArtificalPatternDetector()
    results = detector.run_full_analysis()
    return results

if __name__ == "__main__":
    main() 