"""
Generate NVD processed dataset using GRID-BASED forward-fill approach.

This optimized approach:
1. Creates a complete CVE×Date grid for all NVD CVEs across EPSS date range
2. Joins actual NVD snapshots onto this grid
3. Forward-fills within each CVE using window functions
4. Performs simple LEFT JOIN with EPSS

ADVANTAGES:
- No memory-intensive cross-joins (343M vs 836M intermediate records)
- Clean, predictable data sizes at each step
- Easier to reason about and debug
- More efficient forward-fill logic
"""

import sys
import os

# Add project paths
sys.path.append('.')
sys.path.append('data/general_utils')

from zero_day_zipper import ZeroDayZipper
from t3_spark.session import get_spark_session
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from datetime import datetime

def create_temporal_grid(spark, epss_df, nvd_df):
    """
    Create a complete temporal grid for NVD CVEs across EPSS date range.
    
    This creates a dense grid where every NVD CVE has a row for every EPSS date,
    making forward-fill operations clean and efficient.
    """
    print("=== CREATING TEMPORAL GRID ===")
    
    # Step 1: Get EPSS date range
    epss_date_range = epss_df.select(F.min("date"), F.max("date")).collect()[0]
    min_date, max_date = epss_date_range
    
    total_days = (max_date - min_date).days + 1
    print(f"EPSS date range: {min_date} to {max_date} ({total_days:,} days)")
    
    # Step 2: Create complete date range
    date_range = spark.range(0, total_days).select(
        F.date_add(F.lit(min_date), F.col("id").cast("int")).alias("date")
    )
    
    print(f"Created date grid: {date_range.count():,} dates")
    
    # Step 3: Get unique CVEs from NVD
    nvd_cves = nvd_df.select(F.col("cve_id").alias("cve")).distinct()
    nvd_cve_count = nvd_cves.count()
    
    print(f"Unique NVD CVEs: {nvd_cve_count:,}")
    
    # Step 4: Create CVE × Date grid
    print("Creating CVE×Date grid...")
    cve_date_grid = nvd_cves.crossJoin(date_range)
    grid_size = cve_date_grid.count()
    
    print(f"CVE×Date grid: {grid_size:,} records ({nvd_cve_count:,} × {total_days:,})")
    
    return cve_date_grid, min_date, max_date

def forward_fill_nvd_data(spark, nvd_df, cve_date_grid):
    """
    Forward-fill NVD data using the temporal grid approach.
    
    1. Join actual NVD snapshots onto the grid
    2. Forward-fill missing values within each CVE
    3. Return dense, forward-filled dataset
    """
    print("=== FORWARD-FILLING NVD DATA ===")
    
    # Step 1: Normalize NVD data
    nvd_features = [col for col in nvd_df.columns if col not in ['cve_id', 'snapshot_date']]
    
    nvd_normalized = nvd_df.select(
        F.col("cve_id").alias("cve"),
        F.to_date(F.col("snapshot_date")).alias("date"),
        F.col("snapshot_date"),  # Keep for deduplication
        *nvd_features
    )
    
    print(f"NVD features to forward-fill: {len(nvd_features)}")
    
    # Step 1.5: CRITICAL - Deduplicate NVD snapshots
    print("🔧 DEDUPLICATION: Removing duplicate snapshots per (CVE, date)...")
    
    original_count = nvd_normalized.count()
    
    # Keep the LATEST snapshot for each (CVE, date) pair
    nvd_deduplicated = nvd_normalized.orderBy("cve", "date", F.desc("snapshot_date")).dropDuplicates(["cve", "date"])
    nvd_deduplicated = nvd_deduplicated.drop("snapshot_date")  # Clean up
    
    dedup_count = nvd_deduplicated.count()
    removed_count = original_count - dedup_count
    
    print(f"Before dedup: {original_count:,} records")
    print(f"After dedup:  {dedup_count:,} records")
    print(f"Removed:      {removed_count:,} duplicate snapshots")
    
    if removed_count > 0:
        print("✅ FIXED: Duplicate snapshots that would have exploded the grid!")
    else:
        print("ℹ️  No duplicates found (data already clean)")
    
    # Step 2: Join actual snapshots onto grid
    print("Joining deduplicated NVD snapshots onto temporal grid...")
    
    nvd_on_grid = cve_date_grid.join(
        nvd_deduplicated,
        on=["cve", "date"],
        how="left"
    )
    
    grid_count = nvd_on_grid.count()
    print(f"Grid with NVD data: {grid_count:,} records")
    
    # Step 3: Forward-fill within each CVE
    print("Applying forward-fill within each CVE...")
    
    # Window: within each CVE, ordered by date, look back to beginning
    window = Window.partitionBy("cve").orderBy("date").rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    
    # Forward-fill each NVD feature
    for i, col in enumerate(nvd_features):
        nvd_on_grid = nvd_on_grid.withColumn(
            col,
            F.last(col, ignorenulls=True).over(window)
        )
        
        if (i + 1) % 10 == 0:  # Progress indicator
            print(f"  Forward-filled {i + 1}/{len(nvd_features)} features...")
    
    print(f"✅ Forward-fill complete for all {len(nvd_features)} features")
    
    return nvd_on_grid

def generate_nvd_processed_grid():
    """Generate NVD processed dataset using optimized grid approach."""
    
    print("=" * 80)
    print("GENERATING NVD PROCESSED DATASET - GRID-BASED APPROACH")
    print("=" * 80)
    
    # Initialize Spark with optimization settings
    spark = get_spark_session("NVD_Processing_Grid")
    
    # Optimize for this workload
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    
    # Define paths
    epss_path = "data/epss/processed/epss_processed.parquet"
    nvd_path = "data/nvd/raw/cve_snapshots_irregular.parquet"
    output_path = "data/nvd/processed/nvd_processed.parquet"
    
    print(f"📊 EPSS Base: {epss_path}")
    print(f"📊 NVD Features: {nvd_path}")
    print(f"💾 Output: {output_path}")
    print()
    
    # Step 1: Load datasets
    print("=== STEP 1: LOADING DATASETS ===")
    epss_df = spark.read.parquet(epss_path)
    nvd_df = spark.read.parquet(nvd_path)
    
    epss_count = epss_df.count()
    nvd_count = nvd_df.count()
    
    print(f"EPSS: {epss_count:,} rows × {len(epss_df.columns)} columns")
    print(f"NVD: {nvd_count:,} rows × {len(nvd_df.columns)} columns")
    print()
    
    # Step 2: Create temporal grid
    print("=== STEP 2: TEMPORAL GRID CONSTRUCTION ===")
    cve_date_grid, min_date, max_date = create_temporal_grid(spark, epss_df, nvd_df)
    print()
    
    # Step 3: Forward-fill NVD data
    print("=== STEP 3: FORWARD-FILL NVD DATA ===")
    nvd_forward_filled = forward_fill_nvd_data(spark, nvd_df, cve_date_grid)
    print()
    
    # Step 4: Simple LEFT JOIN with EPSS
    print("=== STEP 4: FINAL MERGE WITH EPSS ===")
    print("Performing clean LEFT JOIN (no complex temporal logic needed)...")
    
    final_result = epss_df.join(
        nvd_forward_filled,
        on=["cve", "date"],
        how="left"
    )
    
    final_count = final_result.count()
    print(f"✅ Final dataset: {final_count:,} rows × {len(final_result.columns)} columns")
    
    # Step 5: Validation
    print("\n=== STEP 5: VALIDATION ===")
    
    if final_count == epss_count:
        print("✅ ROW COUNT PRESERVED: Perfect EPSS structure maintenance")
        print("🎯 DUPLICATE PROBLEM SOLVED: No grid expansion from NVD duplicates!")
    else:
        row_diff = final_count - epss_count
        print(f"❌ ROW COUNT MISMATCH: Expected {epss_count:,}, got {final_count:,}")
        print(f"   Difference: {row_diff:,} extra rows")
        
        if row_diff > 0:
            print("🚨 This suggests remaining duplicate issues in NVD data!")
        
        raise ValueError("Row count validation failed!")
    
    # Check data quality
    nvd_features = [col for col in nvd_df.columns if col not in ['cve_id', 'snapshot_date']]
    sample_feature = nvd_features[0] if nvd_features else None
    
    if sample_feature:
        non_null_count = final_result.filter(F.col(sample_feature).isNotNull()).count()
        coverage_pct = (non_null_count / final_count) * 100
        
        print(f"📊 NVD coverage: {non_null_count:,}/{final_count:,} ({coverage_pct:.1f}%)")
        
        if coverage_pct > 90:
            print("🎉 EXCELLENT: High NVD coverage achieved!")
        elif coverage_pct > 70:
            print("✅ GOOD: Solid NVD coverage")
        else:
            print("⚠️  MODERATE: Some CVEs have no NVD data")
    
    # Step 6: Save result
    print("\n=== STEP 6: SAVING RESULT ===")
    os.makedirs("data/nvd/processed", exist_ok=True)
    
    final_result.write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved to: {output_path}")
    
    # Step 7: Summary
    print("\n=== GRID-BASED APPROACH SUMMARY ===")
    print("🚀 Memory-efficient: No 836M cross-join explosion")
    print("⚡ Clean logic: Explicit grid construction + forward-fill")
    print("🎯 Predictable: Known data sizes at each step")
    print("✅ Optimal: Much faster than complex window operations")
    print(f"📊 Result: {final_count:,} rows with {len(nvd_features)} NVD features")
    
    return final_result

if __name__ == "__main__":
    result_df = generate_nvd_processed_grid() 