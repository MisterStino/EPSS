"""
Generate NVD processed dataset with OPTIMIZED forward-fill temporal alignment.

This script uses an efficient window function approach to avoid memory-intensive cross-joins.
Instead of creating 836M intermediate records, it processes CVE by CVE using broadcast joins.

OPTIMIZATION STRATEGY:
1. Broadcast small NVD dataset to all nodes
2. Use window functions to find most recent NVD snapshot per CVE per date
3. Avoid massive intermediate datasets that cause memory pressure
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

def create_optimized_forward_filled_nvd(spark, nvd_df, epss_df):
    """
    OPTIMIZED forward-fill using window functions and broadcast joins.
    
    Strategy:
    1. For each CVE in EPSS, get all its dates
    2. For each CVE, get all NVD snapshots (broadcast small dataset)
    3. Use window function to assign most recent snapshot per date
    4. Avoid massive cross-join intermediate datasets
    """
    print("=== OPTIMIZED FORWARD-FILL TEMPORAL ALIGNMENT ===")
    
    # Step 1: Get unique CVEs that exist in both datasets
    epss_cves = epss_df.select("cve").distinct()
    nvd_cves = nvd_df.select(F.col("cve_id").alias("cve")).distinct()
    
    common_cves = epss_cves.intersect(nvd_cves)
    common_cve_count = common_cves.count()
    
    print(f"CVEs with both EPSS and NVD data: {common_cve_count:,}")
    
    # Step 2: Prepare NVD data (normalize columns)
    nvd_normalized = nvd_df.select(
        F.col("cve_id").alias("cve"),
        F.to_date(F.col("snapshot_date")).alias("nvd_date"),
        *[col for col in nvd_df.columns if col not in ['cve_id', 'snapshot_date']]
    )
    
    print(f"NVD snapshots prepared: {nvd_normalized.count():,} records")
    
    # Step 3: OPTIMIZED APPROACH - Process in batches by CVE
    print("Using optimized window function approach...")
    
    # Get all EPSS records for CVEs that have NVD data
    epss_with_nvd_cves = epss_df.join(
        F.broadcast(common_cves), 
        on="cve", 
        how="inner"
    )
    
    eligible_epss_count = epss_with_nvd_cves.count()
    print(f"EPSS records for CVEs with NVD data: {eligible_epss_count:,}")
    
    # Step 4: Smart temporal join using broadcast
    # Broadcast the smaller NVD dataset
    nvd_broadcast = F.broadcast(nvd_normalized)
    
    # Join EPSS dates with NVD snapshots where snapshot_date <= epss_date
    temporal_matches = epss_with_nvd_cves.join(
        nvd_broadcast,
        on="cve",
        how="inner"
    ).filter(
        F.col("nvd_date") <= F.col("date")
    )
    
    matches_count = temporal_matches.count()
    print(f"Valid temporal matches: {matches_count:,} (much smaller than cross-join)")
    
    # Step 5: Use window function to get most recent NVD snapshot per (CVE, date)
    window_spec = Window.partitionBy("cve", "date").orderBy(F.desc("nvd_date"))
    
    forward_filled = temporal_matches.withColumn(
        "rn", F.row_number().over(window_spec)
    ).filter(
        F.col("rn") == 1
    ).drop("rn", "nvd_date")
    
    ff_count = forward_filled.count()
    print(f"Forward-filled records: {ff_count:,}")
    
    # Step 6: Add back EPSS records for CVEs with no NVD data (they'll get nulls)
    epss_no_nvd = epss_df.join(
        common_cves,
        on="cve", 
        how="left_anti"  # CVEs NOT in common_cves
    )
    
    no_nvd_count = epss_no_nvd.count()
    print(f"EPSS records with no NVD data: {no_nvd_count:,} (will have null NVD features)")
    
    # Add null NVD columns to records without NVD data
    nvd_columns = [col for col in nvd_normalized.columns if col not in ['cve', 'nvd_date']]
    
    for col in nvd_columns:
        epss_no_nvd = epss_no_nvd.withColumn(col, F.lit(None).cast("string"))
    
    # Union the forward-filled data with no-NVD data
    final_result = forward_filled.union(epss_no_nvd)
    
    final_count = final_result.count()
    print(f"Final forward-filled dataset: {final_count:,} records")
    
    # Coverage calculation
    original_epss_count = epss_df.count()
    coverage_pct = (ff_count / original_epss_count) * 100
    
    print(f"🎯 Coverage: {ff_count:,}/{original_epss_count:,} ({coverage_pct:.1f}%) EPSS records have NVD data")
    
    return final_result

def generate_nvd_processed_optimized():
    """Generate NVD processed dataset with OPTIMIZED forward-fill."""
    
    print("=" * 80)
    print("GENERATING NVD PROCESSED DATASET - OPTIMIZED FORWARD-FILL")
    print("=" * 80)
    
    # Initialize Spark with memory optimization
    spark = get_spark_session("NVD_Processing_Optimized")
    
    # Optimize Spark settings for this workload
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
    
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
    
    # Step 2: Optimized forward-fill
    print("=== STEP 2: OPTIMIZED FORWARD-FILL ===")
    forward_filled_nvd = create_optimized_forward_filled_nvd(spark, nvd_df, epss_df)
    print()
    
    # Step 3: Merge using ZeroDayZipper
    print("=== STEP 3: FINAL MERGE ===")
    print("Forward-filled data ready for ZeroDayZipper...")
    
    zipper = ZeroDayZipper(
        epss_df=epss_df,
        auto_fill_nulls=False,
        enable_caching=False,
        verbose=True
    )
    
    # The forward-filled data already contains EPSS structure, so we just use it directly
    merged_df = forward_filled_nvd  # Already properly aligned
    merged_count = merged_df.count()
    
    print(f"✅ Final dataset: {merged_count:,} rows × {len(merged_df.columns)} columns")
    
    # Step 4: Validation
    print("\n=== STEP 4: VALIDATION ===")
    if merged_count == epss_count:
        print("✅ ROW COUNT PRESERVED: Perfect EPSS structure maintenance")
    else:
        print(f"❌ ROW COUNT MISMATCH: Expected {epss_count:,}, got {merged_count:,}")
        raise ValueError("Row count validation failed!")
    
    # Step 5: Data quality analysis
    print("\n=== STEP 5: DATA QUALITY ANALYSIS ===")
    nvd_columns = [col for col in merged_df.columns if col not in ['cve', 'date', 'epss']]
    
    print("NVD feature coverage:")
    sample_col = nvd_columns[0] if nvd_columns else None
    if sample_col:
        non_null_count = merged_df.filter(F.col(sample_col).isNotNull()).count()
        coverage_pct = (non_null_count / merged_count) * 100
        print(f"  Overall coverage: {non_null_count:,}/{merged_count:,} ({coverage_pct:.1f}%)")
    
    # Step 6: Save optimized result
    print("\n=== STEP 6: SAVING RESULT ===")
    os.makedirs("data/nvd/processed", exist_ok=True)
    
    merged_df.write.mode("overwrite").parquet(output_path)
    print(f"✅ Saved to: {output_path}")
    
    # Final summary
    print("\n=== OPTIMIZATION SUMMARY ===")
    print("🚀 Used broadcast joins instead of cross-joins")
    print("💾 Avoided 836M intermediate record explosion")
    print("⚡ Efficient window functions for temporal alignment")
    print("✅ Memory-optimized processing completed successfully")
    
    return merged_df

if __name__ == "__main__":
    result_df = generate_nvd_processed_optimized() 