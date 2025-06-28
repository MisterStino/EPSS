#!/usr/bin/env python3
"""
Comprehensive EPSS Research Analysis
====================================

Rigorous investigation of ~280,000 CVE time series to identify optimal subset 
for EPSS forecasting research. Addresses temporal dynamics, cross-sectional 
patterns, forecast difficulty, and operational value.

Research Questions:
- RQ1: How accurately can sequence models forecast EPSS probabilities?
- RQ2: Can models provide actionable early warnings for high-risk states (>0.7)?
- RQ3: What additional reaction time do models provide vs naive strategies?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
import warnings
warnings.filterwarnings('ignore')

# Import Spark session
import sys
sys.path.append('../../..')
from t3_spark.session import get_spark_session

# Analysis configuration
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

class EPSSResearchAnalyzer:
    """Comprehensive EPSS time series research analyzer"""
    
    def __init__(self):
        self.spark = get_spark_session()
        self.epss_df = None
        self.full_df = None
        self.results = {}
        
    def load_data(self):
        """Load EPSS time series and full feature datasets"""
        print("🔄 Loading EPSS datasets...")
        
        # Load EPSS time series (long format: cve, date, epss)
        try:
            self.epss_df = self.spark.read.parquet("../processed/epss_processed.parquet")
            epss_count = self.epss_df.count()
            epss_cves = self.epss_df.select("cve").distinct().count()
            print(f"✓ EPSS time series: {epss_count:,} observations, {epss_cves:,} unique CVEs")
        except Exception as e:
            print(f"❌ Error loading EPSS data: {e}")
            return False
            
        # Load full feature dataset  
        try:
            self.full_df = self.spark.read.parquet("../../full_db/processed/final_full_data.parquet")
            full_count = self.full_df.count()
            full_cves = self.full_df.select("cve").distinct().count()
            print(f"✓ Full dataset: {full_count:,} observations, {full_cves:,} unique CVEs")
        except Exception as e:
            print(f"⚠️  Could not load full dataset: {e}")
            self.full_df = None
            
        return True
        
    def analyze_temporal_dynamics(self):
        """Comprehensive temporal behavior analysis"""
        print("\n" + "="*60)
        print("📊 TEMPORAL DYNAMICS ANALYSIS")
        print("="*60)
        
        # Per-CVE summary statistics
        print("Computing per-CVE temporal statistics...")
        
        cve_stats = (self.epss_df
                    .groupBy("cve")
                    .agg(
                        F.count("*").alias("sequence_length"),
                        F.min("date").alias("first_date"),
                        F.max("date").alias("last_date"),
                        F.mean("epss").alias("mean_epss"),
                        F.stddev("epss").alias("std_epss"),
                        F.min("epss").alias("min_epss"), 
                        F.max("epss").alias("max_epss"),
                        F.sum(F.when(F.col("epss") > 0.001, 1).otherwise(0)).alias("nonzero_days"),
                        F.sum(F.when(F.col("epss") >= 0.1, 1).otherwise(0)).alias("days_above_01"),
                        F.sum(F.when(F.col("epss") >= 0.3, 1).otherwise(0)).alias("days_above_03"),
                        F.sum(F.when(F.col("epss") >= 0.7, 1).otherwise(0)).alias("days_above_07"),
                        F.sum(F.when(F.col("epss") >= 0.9, 1).otherwise(0)).alias("days_above_09"),
                        F.max(F.when(F.col("epss") >= 0.7, F.col("date"))).alias("first_above_07")
                    ))
        
        # Collect stats for analysis
        stats_pd = cve_stats.toPandas()
        self.results['cve_stats'] = stats_pd
        
        print(f"✓ Analyzed {len(stats_pd):,} CVE time series")
        print(f"✓ Sequence length: mean={stats_pd['sequence_length'].mean():.1f}, median={stats_pd['sequence_length'].median():.1f}")
        print(f"✓ EPSS statistics: mean={stats_pd['mean_epss'].mean():.4f}, std={stats_pd['std_epss'].mean():.4f}")
        
        # High-risk CVE analysis
        high_risk_cves = stats_pd[stats_pd['days_above_07'] > 0]
        print(f"✓ CVEs reaching ≥0.7 threshold: {len(high_risk_cves):,} ({len(high_risk_cves)/len(stats_pd)*100:.2f}%)")
        
        return stats_pd
        
    def analyze_jump_patterns(self):
        """Detect and analyze EPSS jump patterns"""
        print("\n📈 Analyzing EPSS jump patterns...")
        
        # Calculate day-to-day changes using window functions
        window_spec = Window.partitionBy("cve").orderBy("date")
        
        jumps_df = (self.epss_df
                   .withColumn("prev_epss", F.lag("epss").over(window_spec))
                   .withColumn("epss_change", F.col("epss") - F.col("prev_epss"))
                   .withColumn("abs_change", F.abs(F.col("epss_change")))
                   .filter(F.col("prev_epss").isNotNull()))
        
        # Define jump thresholds
        jump_thresholds = [0.1, 0.2, 0.3, 0.5]
        
        jump_stats = {}
        for threshold in jump_thresholds:
            jumps = (jumps_df
                    .filter(F.col("abs_change") >= threshold)
                    .groupBy("cve")
                    .agg(F.count("*").alias(f"jumps_{str(threshold).replace('.', '')}"),
                         F.max("abs_change").alias(f"max_jump_{str(threshold).replace('.', '')}"))
                    .toPandas())
            jump_stats[threshold] = jumps
            
        self.results['jump_patterns'] = jump_stats
        
        # Overall jump statistics
        total_changes = jumps_df.select("abs_change").toPandas()
        print(f"✓ Total day-to-day changes analyzed: {len(total_changes):,}")
        print(f"✓ Changes ≥0.1: {(total_changes['abs_change'] >= 0.1).sum():,}")
        print(f"✓ Changes ≥0.2: {(total_changes['abs_change'] >= 0.2).sum():,}")
        print(f"✓ Changes ≥0.5: {(total_changes['abs_change'] >= 0.5).sum():,}")
        
        return jump_stats
        
    def analyze_epss_distribution(self):
        """Analyze EPSS score distribution and percentile buckets"""
        print("\n📊 EPSS Distribution Analysis...")
        
        # Current EPSS percentiles
        percentiles = [0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]
        epss_percentiles = (self.epss_df
                          .select("epss")
                          .summary(*[str(p) for p in percentiles])
                          .toPandas())
        
        print("EPSS Percentile Distribution:")
        for i, p in enumerate(percentiles):
            value = float(epss_percentiles.iloc[i+1, 1])  # Skip count row
            print(f"  {p*100:5.1f}%: {value:.4f}")
            
        # Create EPSS buckets for stratification
        bucket_conditions = [
            (F.col("epss") < 0.1, "very_low"),
            ((F.col("epss") >= 0.1) & (F.col("epss") < 0.3), "low"), 
            ((F.col("epss") >= 0.3) & (F.col("epss") < 0.7), "medium"),
            ((F.col("epss") >= 0.7) & (F.col("epss") < 0.9), "high"),
            (F.col("epss") >= 0.9, "very_high")
        ]
        
        bucket_df = self.epss_df.select("cve", "epss")
        for condition, label in bucket_conditions:
            bucket_df = bucket_df.withColumn("bucket", 
                                           F.when(condition, label)
                                           .otherwise(F.col("bucket") if "bucket" in bucket_df.columns else None))
        
        bucket_counts = bucket_df.groupBy("bucket").count().toPandas()
        print("\nEPSS Bucket Distribution:")
        for _, row in bucket_counts.iterrows():
            pct = row['count'] / bucket_counts['count'].sum() * 100
            print(f"  {row['bucket']:>10}: {row['count']:>8,} ({pct:5.2f}%)")
            
        self.results['epss_distribution'] = {
            'percentiles': epss_percentiles,
            'buckets': bucket_counts
        }
        
        return bucket_counts
        
    def analyze_behavioral_archetypes(self):
        """Identify behavioral archetypes using statistical patterns"""
        print("\n🎭 Behavioral Archetype Analysis...")
        
        # Calculate volatility and pattern metrics per CVE
        archetype_metrics = (self.epss_df
                           .groupBy("cve")
                           .agg(
                               F.mean("epss").alias("mean_epss"),
                               F.stddev("epss").alias("volatility"),
                               F.max("epss").alias("max_epss"),
                               F.min("epss").alias("min_epss"),
                               (F.max("epss") - F.min("epss")).alias("range_epss"),
                               F.count("*").alias("length"),
                               F.sum(F.when(F.col("epss") == 0, 1).otherwise(0)).alias("zero_days"),
                               F.sum(F.when(F.col("epss") >= 0.7, 1).otherwise(0)).alias("high_risk_days")
                           )
                           .withColumn("zero_ratio", F.col("zero_days") / F.col("length"))
                           .withColumn("high_risk_ratio", F.col("high_risk_days") / F.col("length"))
                           .toPandas())
        
        # Define archetypes based on statistical patterns
        def classify_archetype(row):
            if row['zero_ratio'] > 0.95:
                return "always_zero"
            elif row['volatility'] < 0.01 and row['mean_epss'] < 0.1:
                return "low_stable" 
            elif row['volatility'] < 0.01 and row['mean_epss'] >= 0.7:
                return "high_stable"
            elif row['volatility'] > 0.2:
                return "highly_volatile"
            elif row['high_risk_ratio'] > 0.1 and row['max_epss'] >= 0.7:
                return "high_risk_spiker"
            elif row['range_epss'] > 0.5:
                return "wide_range_mover"
            else:
                return "moderate_dynamic"
                
        archetype_metrics['archetype'] = archetype_metrics.apply(classify_archetype, axis=1)
        
        archetype_counts = archetype_metrics['archetype'].value_counts()
        print("Behavioral Archetype Distribution:")
        for archetype, count in archetype_counts.items():
            pct = count / len(archetype_metrics) * 100
            print(f"  {archetype:>18}: {count:>8,} ({pct:5.2f}%)")
            
        self.results['behavioral_archetypes'] = {
            'metrics': archetype_metrics,
            'counts': archetype_counts
        }
        
        return archetype_metrics
        
    def analyze_forecast_difficulty(self):
        """Assess forecasting difficulty using predictability metrics"""
        print("\n🎯 Forecast Difficulty Analysis...")
        
        # Calculate autocorrelation and predictability metrics
        difficulty_metrics = []
        
        # Sample CVEs for detailed analysis (computationally intensive)
        sample_cves = (self.epss_df
                      .select("cve")
                      .distinct()
                      .sample(0.01, seed=42)  # 1% sample
                      .collect())
        
        print(f"Analyzing forecast difficulty for {len(sample_cves):,} sampled CVEs...")
        
        for row in sample_cves[:1000]:  # Limit for performance
            cve_id = row['cve']
            cve_data = (self.epss_df
                       .filter(F.col("cve") == cve_id)
                       .orderBy("date")
                       .select("epss")
                       .toPandas())
            
            if len(cve_data) < 30:  # Need minimum length for analysis
                continue
                
            epss_series = cve_data['epss'].values
            
            # Calculate metrics
            metrics = {
                'cve': cve_id,
                'length': len(epss_series),
                'mean': np.mean(epss_series),
                'std': np.std(epss_series),
                'autocorr_lag1': np.corrcoef(epss_series[:-1], epss_series[1:])[0,1] if len(epss_series) > 1 else np.nan,
                'trend_slope': np.polyfit(range(len(epss_series)), epss_series, 1)[0] if len(epss_series) > 1 else 0,
                'zero_crossings': np.sum(np.diff(epss_series > 0.7)),
                'max_consecutive_stable': self._max_consecutive_stable(epss_series),
            }
            
            difficulty_metrics.append(metrics)
            
        difficulty_df = pd.DataFrame(difficulty_metrics)
        
        if len(difficulty_df) > 0:
            print(f"✓ Forecast difficulty metrics computed for {len(difficulty_df):,} CVEs")
            print(f"✓ Mean autocorrelation (lag-1): {difficulty_df['autocorr_lag1'].mean():.3f}")
            print(f"✓ High autocorr (>0.9): {(difficulty_df['autocorr_lag1'] > 0.9).sum():,} CVEs")
            print(f"✓ Low autocorr (<0.5): {(difficulty_df['autocorr_lag1'] < 0.5).sum():,} CVEs")
            
        self.results['forecast_difficulty'] = difficulty_df
        return difficulty_df
        
    def _max_consecutive_stable(self, series, threshold=0.05):
        """Calculate maximum consecutive stable periods"""
        if len(series) < 2:
            return 0
        changes = np.abs(np.diff(series))
        stable = changes < threshold
        if not stable.any():
            return 0
        groups = np.split(np.arange(len(stable)), np.where(~stable)[0] + 1)
        return max((len(group) for group in groups if len(group) > 0 and group[0] < len(stable) and stable[group[0]]), default=0)
        
    def analyze_early_warning_potential(self):
        """Analyze potential for early warning systems (RQ2, RQ3)"""
        print("\n⚠️  Early Warning Potential Analysis...")
        
        # Identify CVEs that cross 0.7 threshold
        threshold_crossers = (self.epss_df
                            .filter(F.col("epss") >= 0.7)
                            .select("cve")
                            .distinct()
                            .toPandas())
        
        print(f"✓ CVEs that reach ≥0.7 threshold: {len(threshold_crossers):,}")
        
        # For each CVE that crosses threshold, analyze the buildup pattern
        warning_analysis = []
        
        for cve_row in threshold_crossers.head(1000).itertuples():  # Sample for performance
            cve_id = cve_row.cve
            
            # Get time series for this CVE
            cve_series = (self.epss_df
                         .filter(F.col("cve") == cve_id)
                         .orderBy("date")
                         .toPandas())
            
            # Find first crossing of 0.7
            high_risk_mask = cve_series['epss'] >= 0.7
            if not high_risk_mask.any():
                continue
                
            first_high_risk_idx = high_risk_mask.idxmax()
            
            # Analyze buildup period
            buildup_window = 30  # 30 days before crossing
            start_idx = max(0, first_high_risk_idx - buildup_window)
            
            if start_idx < first_high_risk_idx:
                buildup_series = cve_series.iloc[start_idx:first_high_risk_idx]['epss']
                
                analysis = {
                    'cve': cve_id,
                    'first_high_risk_date': cve_series.iloc[first_high_risk_idx]['date'],
                    'buildup_days': len(buildup_series),
                    'buildup_start_epss': buildup_series.iloc[0] if len(buildup_series) > 0 else 0,
                    'buildup_max_epss': buildup_series.max() if len(buildup_series) > 0 else 0,
                    'buildup_slope': np.polyfit(range(len(buildup_series)), buildup_series, 1)[0] if len(buildup_series) > 1 else 0,
                    'days_above_01_in_buildup': (buildup_series >= 0.1).sum(),
                    'days_above_03_in_buildup': (buildup_series >= 0.3).sum(),
                    'potential_warning_days': (buildup_series >= 0.3).sum()  # Days above 0.3 before 0.7
                }
                
                warning_analysis.append(analysis)
        
        warning_df = pd.DataFrame(warning_analysis)
        
        if len(warning_df) > 0:
            print(f"✓ Early warning analysis for {len(warning_df):,} high-risk CVEs")
            print(f"✓ Mean potential warning days: {warning_df['potential_warning_days'].mean():.1f}")
            print(f"✓ CVEs with >7 days warning: {(warning_df['potential_warning_days'] >= 7).sum():,}")
            print(f"✓ CVEs with >14 days warning: {(warning_df['potential_warning_days'] >= 14).sum():,}")
            
        self.results['early_warning'] = warning_df
        return warning_df
        
    def generate_sampling_recommendations(self):
        """Generate evidence-based sampling strategy recommendations"""
        print("\n" + "="*60)
        print("🎯 SAMPLING STRATEGY RECOMMENDATIONS")
        print("="*60)
        
        # Get results from previous analyses
        cve_stats = self.results.get('cve_stats')
        archetypes = self.results.get('behavioral_archetypes', {}).get('metrics')
        early_warning = self.results.get('early_warning')
        
        recommendations = []
        
        if cve_stats is not None:
            total_cves = len(cve_stats)
            
            # Strategy 1: Balanced Archetype Sampling
            if archetypes is not None:
                archetype_counts = archetypes['archetype'].value_counts()
                print("\n📊 Strategy 1: Balanced Archetype Sampling")
                print("Recommended samples per archetype (for 10K total):")
                
                target_total = 10000
                min_per_archetype = 100
                
                for archetype, count in archetype_counts.items():
                    if count < min_per_archetype:
                        recommended = count  # Take all if very rare
                    else:
                        # Balanced sampling with minimum guarantees
                        base_allocation = target_total // len(archetype_counts)
                        recommended = max(min_per_archetype, min(base_allocation, count))
                    
                    print(f"  {archetype:>18}: {recommended:>5,} / {count:>6,} available ({recommended/count*100:5.1f}%)")
                
                recommendations.append({
                    'strategy': 'balanced_archetype',
                    'total_cves': target_total,
                    'rationale': 'Ensures representation of all behavioral patterns'
                })
            
            # Strategy 2: Risk-Stratified Sampling  
            high_risk_cves = len(cve_stats[cve_stats['days_above_07'] > 0])
            medium_risk_cves = len(cve_stats[(cve_stats['days_above_03'] > 0) & (cve_stats['days_above_07'] == 0)])
            low_risk_cves = total_cves - high_risk_cves - medium_risk_cves
            
            print(f"\n📈 Strategy 2: Risk-Stratified Sampling")
            print(f"Risk distribution in full dataset:")
            print(f"  High risk (≥0.7):     {high_risk_cves:>8,} ({high_risk_cves/total_cves*100:5.2f}%)")
            print(f"  Medium risk (0.3-0.7): {medium_risk_cves:>8,} ({medium_risk_cves/total_cves*100:5.2f}%)")
            print(f"  Low risk (<0.3):       {low_risk_cves:>8,} ({low_risk_cves/total_cves*100:5.2f}%)")
            
            # Oversampling high-risk for better model training
            print(f"\nRecommended stratified sampling (10K total):")
            high_risk_sample = min(3000, high_risk_cves)  # Oversample high-risk
            medium_risk_sample = min(3000, medium_risk_cves)
            low_risk_sample = 4000
            
            print(f"  High risk:     {high_risk_sample:>5,} (30% of sample)")
            print(f"  Medium risk:   {medium_risk_sample:>5,} (30% of sample)")  
            print(f"  Low risk:      {low_risk_sample:>5,} (40% of sample)")
            
            recommendations.append({
                'strategy': 'risk_stratified',
                'total_cves': high_risk_sample + medium_risk_sample + low_risk_sample,
                'rationale': 'Oversamples rare high-risk cases for better early warning training'
            })
            
            # Strategy 3: Temporal Diversity Sampling
            print(f"\n📅 Strategy 3: Temporal Diversity Sampling")
            print("Sample CVEs with diverse temporal characteristics:")
            
            sequence_length_buckets = pd.qcut(cve_stats['sequence_length'], q=5, labels=['very_short', 'short', 'medium', 'long', 'very_long'])
            volatility_buckets = pd.qcut(cve_stats['std_epss'].fillna(0), q=5, labels=['stable', 'low_vol', 'med_vol', 'high_vol', 'very_volatile'])
            
            temporal_grid = pd.crosstab(sequence_length_buckets, volatility_buckets)
            print("Length × Volatility distribution:")
            print(temporal_grid)
            
            recommendations.append({
                'strategy': 'temporal_diversity', 
                'total_cves': 10000,
                'rationale': 'Ensures representation across sequence lengths and volatility regimes'
            })
            
        print(f"\n🎯 FINAL RECOMMENDATIONS:")
        print(f"For computationally tractable research addressing RQ1-RQ3:")
        print(f"")
        print(f"1. 🎭 BALANCED ARCHETYPE (Recommended): 10,000 CVEs")
        print(f"   - Ensures all behavioral patterns represented")
        print(f"   - Maintains statistical diversity for generalization")
        print(f"   - Computationally feasible for LSTM training")
        print(f"")
        print(f"2. 📈 RISK-STRATIFIED: 10,000 CVEs") 
        print(f"   - Oversamples high-risk CVEs (30% vs 2.5% natural)")
        print(f"   - Optimal for early warning research (RQ2, RQ3)")
        print(f"   - May sacrifice some low-risk pattern diversity")
        print(f"")
        print(f"3. 🔬 HYBRID: 15,000 CVEs")
        print(f"   - Combines archetype balance + risk oversampling")
        print(f"   - Best for comprehensive research across all RQs")
        print(f"   - Higher computational cost but maximum insight")
        
        return recommendations
        
    def run_comprehensive_analysis(self):
        """Execute the complete research analysis pipeline"""
        print("🚀 Starting Comprehensive EPSS Research Analysis")
        print("="*60)
        
        # Load data
        if not self.load_data():
            return False
            
        # Run all analyses
        try:
            self.analyze_temporal_dynamics()
            self.analyze_jump_patterns() 
            self.analyze_epss_distribution()
            self.analyze_behavioral_archetypes()
            self.analyze_forecast_difficulty()
            self.analyze_early_warning_potential()
            self.generate_sampling_recommendations()
            
            print(f"\n" + "="*60)
            print("✅ COMPREHENSIVE ANALYSIS COMPLETED")
            print("="*60)
            print(f"📊 Results stored in analyzer.results dictionary")
            print(f"🎯 Sampling recommendations generated")
            print(f"📈 Ready for subset selection and model training")
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def save_results(self, output_path="epss_research_results.pkl"):
        """Save analysis results for further use"""
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"💾 Results saved to {output_path}")

if __name__ == "__main__":
    # Execute comprehensive analysis
    analyzer = EPSSResearchAnalyzer()
    success = analyzer.run_comprehensive_analysis()
    
    if success:
        analyzer.save_results()
        print("\n🎯 Analysis complete! Use results to guide CVE subset selection.")
    else:
        print("\n❌ Analysis failed. Check data paths and Spark configuration.") 