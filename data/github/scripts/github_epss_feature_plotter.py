#!/usr/bin/env python
"""
GitHub-EPSS Feature Plotter
===========================

Creates visual validation plots for GitHub features vs EPSS scores to:
1. Detect temporal patterns and correlations
2. Spot artificial patterns in GitHub data
3. Validate feature quality for LSTM training
4. Quality control feature engineering pipeline

Usage:
    python -m data.github.scripts.github_epss_feature_plotter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Use project's battle-tested Spark session for big data
from t3_spark.session import get_spark_session

class GitHubEPSSPlotter:
    """Plot GitHub features alongside EPSS scores for pattern analysis."""
    
    def __init__(self):
        self.spark = get_spark_session()
        self.output_dir = Path("data/github/scripts/figures_cleaned")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data paths
        self.github_path = "data/github/raw/github_cleaned_evidence_based.csv"
        self.epss_path = "data/epss/processed/epss_processed.parquet"
        
        print("🔍 GitHub-EPSS Feature Plotter Initialized")
        print(f"📁 Output directory: {self.output_dir}")
    
    def load_data(self):
        """Load and prepare EPSS and GitHub data using Spark (keep in Spark format)."""
        print("\n📊 LOADING DATA WITH SPARK")
        print("=" * 50)
        
        # Load EPSS data (large parquet file) - KEEP IN SPARK
        print("Loading EPSS data...")
        self.epss_spark = self.spark.read.parquet(str(self.epss_path))
        print(f"✓ EPSS records: {self.epss_spark.count():,}")
        print(f"✓ EPSS columns: {self.epss_spark.columns}")
        
        # Load GitHub data - KEEP IN SPARK
        print("Loading GitHub data...")
        self.github_spark = self.spark.read.option("header", "true").option("inferSchema", "true").csv(self.github_path)
        print(f"✓ GitHub records: {self.github_spark.count():,}")
        print(f"✓ GitHub columns: {self.github_spark.columns}")
        
        # Cache the datasets for repeated access
        self.epss_spark.cache()
        self.github_spark.cache()
        
        print("✅ Data loaded and cached in Spark - NO Pandas conversion")
        
        return self.epss_spark, self.github_spark
    
    def select_cves_for_plotting(self, n_cves=1000):
        """Select CVEs that have both EPSS and GitHub data using Spark operations."""
        print(f"\n🎯 SELECTING {n_cves} CVEs FOR PLOTTING")
        print("=" * 50)
        
        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType
        
        # Find CVEs that exist in both datasets using Spark
        print("Finding common CVEs...")
        epss_cves = self.epss_spark.select("cve").distinct()
        github_cves = self.github_spark.select("cve").distinct()
        
        # Inner join to find common CVEs
        common_cves_spark = epss_cves.join(github_cves, on="cve", how="inner")
        common_cves_count = common_cves_spark.count()
        
        print(f"✓ EPSS CVEs: {epss_cves.count():,}")
        print(f"✓ GitHub CVEs: {github_cves.count():,}")
        print(f"✓ Common CVEs: {common_cves_count:,}")
        
        if common_cves_count == 0:
            raise ValueError("No CVEs found in both datasets!")
        
        # Calculate data quality metrics for each CVE using Spark aggregations
        print("Calculating CVE quality metrics...")
        
        # EPSS metrics per CVE
        epss_metrics = self.epss_spark.groupBy("cve").agg(
            F.count("*").alias("epss_count"),
            F.min("date").alias("epss_min_date"),
            F.max("date").alias("epss_max_date")
        ).withColumn("epss_span_days", 
                    F.datediff(F.col("epss_max_date"), F.col("epss_min_date")))
        
        # GitHub metrics per CVE
        github_metrics = self.github_spark.groupBy("cve").agg(
            F.count("*").alias("github_count"),
            F.min("date").alias("github_min_date"),
            F.max("date").alias("github_max_date")
        ).withColumn("github_span_days", 
                    F.datediff(F.col("github_max_date"), F.col("github_min_date")))
        
        # Join metrics and calculate quality score
        cve_quality = epss_metrics.join(github_metrics, on="cve", how="inner")
        
        # Calculate quality score: (record_count + time_span) weighted
        cve_quality = cve_quality.withColumn(
            "quality_score",
            ((F.col("epss_count") + F.col("github_count")) * 
             (F.least(F.col("epss_span_days"), F.col("github_span_days")) + F.lit(1))) / F.lit(1000)
        )
        
        # Sort by quality score and take top N
        top_cves = cve_quality.orderBy(F.col("quality_score").desc()).limit(n_cves)
        
        # Collect results to driver (small dataset now)
        selected_cves_data = top_cves.collect()
        selected_cves = [row.cve for row in selected_cves_data]
        
        print(f"✓ Selected {len(selected_cves)} CVEs for plotting")
        
        # Show some statistics
        if selected_cves_data:
            quality_scores = [row.quality_score for row in selected_cves_data]
            epss_counts = [row.epss_count for row in selected_cves_data]
            github_counts = [row.github_count for row in selected_cves_data]
            epss_spans = [row.epss_span_days for row in selected_cves_data]
            github_spans = [row.github_span_days for row in selected_cves_data]
            
            print(f"✓ Quality score range: {min(quality_scores):.1f} - {max(quality_scores):.1f}")
            print(f"\n📊 SELECTED CVE STATISTICS:")
            print(f"   Avg EPSS records per CVE: {sum(epss_counts)/len(epss_counts):.1f}")
            print(f"   Avg GitHub records per CVE: {sum(github_counts)/len(github_counts):.1f}")
            print(f"   Avg EPSS time span: {sum(epss_spans)/len(epss_spans):.0f} days")
            print(f"   Avg GitHub time span: {sum(github_spans)/len(github_spans):.0f} days")
        
        return selected_cves
    
    def prepare_features(self):
        """Identify and prepare GitHub features for plotting using Spark."""
        print("\n🔧 PREPARING GITHUB FEATURES")
        print("=" * 50)
        
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType, IntegerType, LongType
        
        # Get schema and identify numeric columns
        schema_fields = self.github_spark.schema.fields
        numeric_cols = []
        
        for field in schema_fields:
            if isinstance(field.dataType, (DoubleType, IntegerType, LongType)):
                numeric_cols.append(field.name)
        
        # Remove non-feature columns
        exclude_cols = ['date', 'cve']  # Add other non-feature columns if needed
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"✓ Available GitHub features: {feature_cols}")
        
        # Validate features have some variation using Spark
        valid_features = []
        for col in feature_cols:
            try:
                # Calculate standard deviation using Spark
                stats = self.github_spark.select(F.stddev(col).alias('std')).collect()
                std_val = stats[0]['std']
                
                if std_val is not None and std_val > 0:  # Has variation
                    valid_features.append(col)
                else:
                    print(f"   ⚠️ Skipping {col}: no variation (std={std_val})")
            except Exception as e:
                print(f"   ⚠️ Skipping {col}: error calculating stats ({str(e)})")
        
        print(f"✓ Valid features for plotting: {valid_features}")
        
        self.feature_columns = valid_features
        return valid_features
    
    def create_cve_plot(self, cve, save_plot=True):
        """Create a single plot for one CVE showing EPSS + GitHub features using Spark."""
        
        # Extract data for this specific CVE using Spark (efficient filtering)
        epss_spark_filtered = self.epss_spark.filter(self.epss_spark.cve == cve).orderBy("date")
        github_spark_filtered = self.github_spark.filter(self.github_spark.cve == cve).orderBy("date")
        
        # Convert only this CVE's data to Pandas (small subset)
        cve_epss = epss_spark_filtered.toPandas()
        cve_github = github_spark_filtered.toPandas()
        
        if len(cve_epss) == 0 or len(cve_github) == 0:
            print(f"   ⚠️ Skipping {cve}: insufficient data")
            return False
        
        # Ensure proper date formatting
        cve_epss['date'] = pd.to_datetime(cve_epss['date'])
        cve_github['date'] = pd.to_datetime(cve_github['date'])
        
        # Determine full time range
        min_date = min(cve_epss['date'].min(), cve_github['date'].min())
        max_date = max(cve_epss['date'].max(), cve_github['date'].max())
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot EPSS score (reference line)
        ax.plot(cve_epss['date'], cve_epss['epss'], 
                color='gold', linewidth=3, label='EPSS Score', alpha=0.9, zorder=10)
        
        # Standardize and plot GitHub features
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.feature_columns)))
        
        for i, feature in enumerate(self.feature_columns):
            if feature in cve_github.columns:
                # Standardize feature to 0-1 scale for comparison with EPSS
                feature_values = cve_github[feature].values
                
                if len(feature_values) > 0 and feature_values.std() > 0:
                    # Min-max scaling to [0, 1] range
                    feature_scaled = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())
                    
                    ax.plot(cve_github['date'], feature_scaled,
                           color=colors[i], linewidth=2, label=f'{feature} (scaled)',
                           alpha=0.7, linestyle='--')
        
        # Formatting
        ax.set_title(f'{cve} | EPSS Score vs GitHub Features\n'
                    f'Time Range: {min_date.date()} to {max_date.date()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Score (0-1 scale)', fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add some statistics as text
        epss_stats = f"EPSS: μ={cve_epss['epss'].mean():.3f}, σ={cve_epss['epss'].std():.3f}"
        ax.text(0.02, 0.98, epss_stats, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plot:
            # Save plot
            plot_path = self.output_dir / f"{cve}_epss_github_features.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            return plot_path
        else:
            plt.show()
            return None
    
    def create_all_plots(self, selected_cves):
        """Create plots for all selected CVEs."""
        print(f"\n📈 CREATING PLOTS FOR {len(selected_cves)} CVEs")
        print("=" * 50)
        
        successful_plots = 0
        failed_plots = 0
        
        for i, cve in enumerate(selected_cves, 1):
            try:
                plot_path = self.create_cve_plot(cve, save_plot=True)
                if plot_path:
                    successful_plots += 1
                    if i % 50 == 0:  # Progress update every 50 plots
                        print(f"   ✓ Progress: {i}/{len(selected_cves)} plots created")
                else:
                    failed_plots += 1
                    
            except Exception as e:
                print(f"   ❌ Error plotting {cve}: {str(e)}")
                failed_plots += 1
        
        print(f"\n✅ PLOTTING COMPLETE")
        print(f"   Successful plots: {successful_plots}")
        print(f"   Failed plots: {failed_plots}")
        print(f"   Output directory: {self.output_dir}")
        
        return successful_plots, failed_plots
    
    def create_summary_report(self, selected_cves, successful_plots):
        """Create a summary report of the plotting process using Spark."""
        print(f"\n📋 CREATING SUMMARY REPORT")
        print("=" * 50)
        
        from pyspark.sql import functions as F
        
        # Calculate some overall statistics using Spark
        selected_cves_broadcast = self.spark.sparkContext.broadcast(selected_cves)
        
        # Count records for selected CVEs using Spark
        epss_filtered = self.epss_spark.filter(self.epss_spark.cve.isin(selected_cves))
        github_filtered = self.github_spark.filter(self.github_spark.cve.isin(selected_cves))
        
        total_epss_records = epss_filtered.count()
        total_github_records = github_filtered.count()
        
        # Date ranges using Spark
        epss_date_stats = self.epss_spark.select(
            F.min("date").alias("min_date"), 
            F.max("date").alias("max_date")
        ).collect()[0]
        
        github_date_stats = self.github_spark.select(
            F.min("date").alias("min_date"), 
            F.max("date").alias("max_date")
        ).collect()[0]
        
        epss_date_range = (epss_date_stats.min_date, epss_date_stats.max_date)
        github_date_range = (github_date_stats.min_date, github_date_stats.max_date)
        
        report = f"""
GitHub-EPSS Feature Analysis Report
==================================

Data Summary:
- Total CVEs plotted: {successful_plots}
- EPSS records analyzed: {total_epss_records:,}
- GitHub records analyzed: {total_github_records:,}
- GitHub features plotted: {len(self.feature_columns)}

Date Ranges:
- EPSS data: {epss_date_range[0]} to {epss_date_range[1]}
- GitHub data: {github_date_range[0]} to {github_date_range[1]}

Features Analyzed:
{chr(10).join(f"- {feature}" for feature in self.feature_columns)}

Output:
- Plot directory: {self.output_dir}
- Individual CVE plots: {successful_plots} files
- Plot format: PNG (1500x800 pixels)

Analysis Purpose:
- Detect temporal patterns in GitHub features vs EPSS evolution
- Identify artificial patterns in GitHub data
- Validate feature quality for LSTM training
- Quality control feature engineering pipeline

Next Steps:
1. Visual inspection of plots for correlation patterns
2. Identify CVEs with clean vs artificial GitHub patterns
3. Select high-quality features for LSTM training
4. Filter out CVEs with obvious data quality issues
"""
        
        # Save report
        report_path = self.output_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Report saved: {report_path}")
        print(report)
    
    def run_analysis(self, n_cves=1000):
        """Run the complete analysis pipeline."""
        print("🚀 STARTING GITHUB-EPSS FEATURE ANALYSIS")
        print("=" * 80)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Select CVEs
            selected_cves = self.select_cves_for_plotting(n_cves)
            
            # Step 3: Prepare features
            self.prepare_features()
            
            # Step 4: Create plots
            successful_plots, failed_plots = self.create_all_plots(selected_cves)
            
            # Step 5: Create summary report
            self.create_summary_report(selected_cves, successful_plots)
            
            print(f"\n🎉 ANALYSIS COMPLETE!")
            print(f"📁 Check output directory: {self.output_dir}")
            print(f"🔍 Review {successful_plots} plots for pattern analysis")
            
        except Exception as e:
            print(f"❌ Analysis failed: {str(e)}")
            raise
        
        finally:
            # Clean up Spark session
            self.spark.stop()

def main():
    """Main entry point."""
    plotter = GitHubEPSSPlotter()
    plotter.run_analysis(n_cves=1000)

if __name__ == "__main__":
    main() 