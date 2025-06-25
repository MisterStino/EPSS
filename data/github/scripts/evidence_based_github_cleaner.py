#!/usr/bin/env python3
"""
Evidence-Based GitHub Data Cleaner (Spark-Optimized)
===================================================

Uses Spark for fast processing of 600k+ records while preserving genuine signals.
"""

import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Use project's battle-tested Spark session
from t3_spark.session import get_spark_session
from pyspark.sql import functions as F
from pyspark.sql.types import *

class SparkEvidenceBasedCleaner:
    def __init__(self):
        print("🧹 SPARK-BASED EVIDENCE-BASED GITHUB DATA CLEANER")
        print("=" * 60)
        print("Preserving genuine signals, removing bulk noise - FAST!")
        print("=" * 60)
        
        # Initialize Spark
        print("Initializing Spark session...")
        self.spark = get_spark_session()
        print("✓ Spark session ready")
        
        # Load datasets with Spark
        print("Loading datasets with Spark...")
        
        # Load BigQuery data
        self.bigquery_spark = self.spark.read.csv(
            "data/github/raw/github-final.csv",
            header=True,
            inferSchema=True
        )
        
        # Load baseline data
        self.baseline_spark = self.spark.read.csv(
            "data/github/raw/github_commit_timestamps_9k.csv",
            header=True,
            inferSchema=True
        )
        
        # Standardize column names
        self.baseline_spark = self.baseline_spark.withColumnRenamed("cve_id", "cve") \
                                               .withColumnRenamed("commit_date", "date")
        
        # Convert date columns to proper date type
        self.bigquery_spark = self.bigquery_spark.withColumn("date", F.to_date("date"))
        self.baseline_spark = self.baseline_spark.withColumn("date", F.to_date("date"))
        
        # Cache for performance
        self.bigquery_spark.cache()
        self.baseline_spark.cache()
        
        # Get counts
        bigquery_count = self.bigquery_spark.count()
        baseline_count = self.baseline_spark.count()
        bigquery_cves = self.bigquery_spark.select("cve").distinct().count()
        baseline_cves = self.baseline_spark.select("cve").distinct().count()
        
        print(f"✓ BigQuery loaded: {bigquery_count:,} records, {bigquery_cves:,} CVEs")
        print(f"✓ Baseline loaded: {baseline_count:,} records, {baseline_cves:,} CVEs")
        
        # Initialize cleaning statistics
        self.stats = {
            'original_records': bigquery_count,
            'original_cves': bigquery_cves,
            'final_records': 0,
            'final_cves': 0
        }
        
    def step1_preserve_baseline(self):
        """Step 1: Preserve ALL baseline CVEs as ground truth"""
        print("\n🔒 STEP 1: PRESERVE BASELINE CVEs")
        print("=" * 40)
        
        # Get baseline CVEs as broadcast variable for efficiency
        baseline_cves_list = [row.cve for row in self.baseline_spark.select("cve").distinct().collect()]
        baseline_cves_broadcast = self.spark.sparkContext.broadcast(set(baseline_cves_list))
        
        # Mark baseline CVEs in BigQuery data
        is_baseline_udf = F.udf(lambda cve: cve in baseline_cves_broadcast.value, BooleanType())
        self.bigquery_spark = self.bigquery_spark.withColumn("is_baseline", is_baseline_udf("cve"))
        
        # Get statistics
        baseline_records = self.bigquery_spark.filter(F.col("is_baseline") == True)
        baseline_record_count = baseline_records.count()
        baseline_cve_count = baseline_records.select("cve").distinct().count()
        
        print(f"Baseline CVEs in BigQuery: {baseline_cve_count:,}")
        print(f"Baseline records in BigQuery: {baseline_record_count:,}")
        print(f"Baseline coverage: {baseline_cve_count / len(baseline_cves_list) * 100:.1f}%")
        
        print("✓ All baseline CVEs marked for preservation")
        
    def step2_identify_bulk_dates(self):
        """Step 2: Identify bulk processing dates (statistical outliers)"""
        print("\n📅 STEP 2: IDENTIFY BULK PROCESSING DATES")
        print("=" * 45)
        
        # Analyze daily activity distribution with Spark
        daily_activity = self.bigquery_spark.groupBy("date") \
                                          .agg(F.countDistinct("cve").alias("cve_count")) \
                                          .orderBy("cve_count", ascending=False)
        
        # Get statistics
        daily_stats = daily_activity.agg(
            F.mean("cve_count").alias("mean"),
            F.expr("percentile_approx(cve_count, 0.99)").alias("q99")
        ).collect()[0]
        
        print(f"Daily activity statistics:")
        print(f"  Mean: {daily_stats['mean']:.1f} CVEs/day")
        print(f"  99th percentile: {daily_stats['q99']:.0f} CVEs/day")
        
        # Identify bulk dates (>99th percentile)
        q99_threshold = daily_stats['q99']
        bulk_dates_df = daily_activity.filter(F.col("cve_count") > q99_threshold)
        
        # Collect bulk dates for later use
        self.bulk_dates_list = [row.date for row in bulk_dates_df.select("date").collect()]
        bulk_dates_broadcast = self.spark.sparkContext.broadcast(set(self.bulk_dates_list))
        
        # Add bulk date flag to main dataset
        is_bulk_date_udf = F.udf(lambda date: date in bulk_dates_broadcast.value, BooleanType())
        self.bigquery_spark = self.bigquery_spark.withColumn("is_bulk_date", is_bulk_date_udf("date"))
        
        print(f"\nBulk processing dates identified: {len(self.bulk_dates_list)}")
        print("✓ Bulk dates identified (but baseline CVEs protected)")
        
    def step3_apply_cleaning(self):
        """Step 3: Apply evidence-based cleaning filters"""
        print("\n🧹 STEP 3: APPLY CLEANING FILTERS")
        print("=" * 40)
        
        # Strategy 1: Remove CVEs that ONLY appear on bulk dates (unless baseline)
        cve_date_analysis = self.bigquery_spark.groupBy("cve").agg(
            F.countDistinct("date").alias("total_dates"),
            F.sum(F.when(F.col("is_bulk_date") == True, 1).otherwise(0)).alias("bulk_dates"),
            F.first("is_baseline").alias("is_baseline")
        ).withColumn(
            "only_bulk_dates",
            F.col("total_dates") == F.col("bulk_dates")
        )
        
        # Get CVEs to remove (only on bulk dates AND not baseline)
        cves_to_remove = cve_date_analysis.filter(
            (F.col("only_bulk_dates") == True) & 
            (F.col("is_baseline") == False)
        ).select("cve")
        
        cves_to_remove_list = [row.cve for row in cves_to_remove.collect()]
        cves_to_remove_broadcast = self.spark.sparkContext.broadcast(set(cves_to_remove_list))
        
        # Apply cleaning: remove bulk-only non-baseline CVEs
        should_remove_udf = F.udf(lambda cve: cve in cves_to_remove_broadcast.value, BooleanType())
        
        self.cleaned_spark = self.bigquery_spark.filter(
            ~should_remove_udf("cve")  # Keep everything except CVEs to remove
        )
        
        # Remove temporary columns
        columns_to_remove = ['is_baseline', 'is_bulk_date']
        for col in columns_to_remove:
            if col in self.cleaned_spark.columns:
                self.cleaned_spark = self.cleaned_spark.drop(col)
        
        # Get final statistics
        final_records = self.cleaned_spark.count()
        final_cves = self.cleaned_spark.select("cve").distinct().count()
        
        self.stats['final_records'] = final_records
        self.stats['final_cves'] = final_cves
        
        # Calculate removed records
        removed_records = self.stats['original_records'] - final_records
        
        print(f"Cleaning results:")
        print(f"  Original: {self.stats['original_records']:,} records, {self.stats['original_cves']:,} CVEs")
        print(f"  Removed: {removed_records:,} records ({len(cves_to_remove_list):,} CVEs)")
        print(f"  Final: {final_records:,} records, {final_cves:,} CVEs")
        print(f"  Retention rate: {final_records / self.stats['original_records'] * 100:.1f}%")
        
    def step4_save_results(self):
        """Step 4: Save cleaned dataset"""
        print("\n💾 STEP 4: SAVE RESULTS")
        print("=" * 25)
        
        # Save cleaned CSV using Spark
        output_file = "github_cleaned_evidence_based.csv"
        
        # Convert to single CSV file
        self.cleaned_spark.coalesce(1).write.mode("overwrite").option("header", "true").csv("temp_cleaned_output")
        
        # Move the part file to final location
        import glob
        import shutil
        part_files = glob.glob("temp_cleaned_output/part-*.csv")
        if part_files:
            shutil.move(part_files[0], output_file)
            shutil.rmtree("temp_cleaned_output")
        
        print(f"✓ Cleaned data saved to: {output_file}")
        print(f"  Records: {self.stats['final_records']:,}")
        print(f"  CVEs: {self.stats['final_cves']:,}")
        
        return output_file
    
    def run_cleaning(self):
        """Run the complete evidence-based cleaning process"""
        print("🚀 Starting Spark-based evidence-based cleaning process...")
        
        steps = [
            self.step1_preserve_baseline,
            self.step2_identify_bulk_dates,
            self.step3_apply_cleaning,
            self.step4_save_results
        ]
        
        for i, step in enumerate(steps, 1):
            print(f"\n{'='*60}")
            try:
                result = step()
                if i == len(steps):  # Last step returns filename
                    output_file = result
            except Exception as e:
                print(f"❌ Step {i} failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        print(f"\n{'='*60}")
        print("🎉 SPARK-BASED EVIDENCE-BASED CLEANING COMPLETED!")
        print("\nKey achievements:")
        print("✓ Preserved all legitimate baseline CVEs")
        print("✓ Removed bulk processing artifacts")
        print("✓ Used Spark for fast, efficient processing")
        print("✓ Created high-quality dataset for LSTM training")
        
        # Stop Spark session
        self.spark.stop()
        
        return output_file

def main():
    cleaner = SparkEvidenceBasedCleaner()
    output_file = cleaner.run_cleaning()
    
    if output_file:
        print(f"\n🎯 SUCCESS: Clean dataset ready at {output_file}")
        print("This dataset is now suitable for LSTM training!")
    else:
        print("\n❌ FAILED: Cleaning process encountered errors")

if __name__ == "__main__":
    main() 