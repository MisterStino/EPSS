"""
Deep analysis of temporal alignment between EPSS and NVD data.
"""

import sys
import os
sys.path.append('.')

from t3_spark.session import get_spark_session
import pyspark.sql.functions as F
from pyspark.sql.window import Window

def analyze_temporal_alignment():
    """Analyze why we have so many missing values."""
    
    print("=" * 70)
    print("TEMPORAL ALIGNMENT ANALYSIS")
    print("=" * 70)
    
    spark = get_spark_session("Temporal_Analysis")
    
    # Load both datasets
    print("Loading datasets...")
    epss_df = spark.read.parquet("data/epss/processed/epss_processed.parquet")
    nvd_df = spark.read.parquet("data/nvd/raw/cve_snapshots_irregular.parquet")
    
    print(f"EPSS: {epss_df.count():,} rows")
    print(f"NVD: {nvd_df.count():,} rows")
    print()
    
    # Analyze temporal patterns
    print("=== TEMPORAL PATTERNS ===")
    
    # EPSS date range
    epss_dates = epss_df.select(F.min("date"), F.max("date")).collect()[0]
    print(f"EPSS dates: {epss_dates[0]} to {epss_dates[1]}")
    
    # NVD date range
    nvd_dates = nvd_df.select(F.min("snapshot_date"), F.max("snapshot_date")).collect()[0]
    print(f"NVD dates: {nvd_dates[0]} to {nvd_dates[1]}")
    print()
    
    # Check specific CVE patterns
    print("=== CVE-SPECIFIC TEMPORAL ANALYSIS ===")
    
    # Pick CVE-2021-32829 from our inspection
    test_cve = "CVE-2021-32829"
    print(f"Analyzing {test_cve}...")
    
    # EPSS records for this CVE
    epss_cve = epss_df.filter(F.col("cve") == test_cve).orderBy("date")
    epss_count = epss_cve.count()
    epss_range = epss_cve.select(F.min("date"), F.max("date")).collect()[0]
    
    print(f"EPSS records: {epss_count:,} from {epss_range[0]} to {epss_range[1]}")
    
    # NVD records for this CVE
    nvd_cve = nvd_df.filter(F.col("cve_id") == test_cve).orderBy("snapshot_date")
    nvd_count = nvd_cve.count()
    
    if nvd_count > 0:
        nvd_range = nvd_cve.select(F.min("snapshot_date"), F.max("snapshot_date")).collect()[0]
        print(f"NVD records: {nvd_count:,} from {nvd_range[0]} to {nvd_range[1]}")
        
        print("\nNVD snapshot dates for this CVE:")
        for row in nvd_cve.select("snapshot_date").collect():
            print(f"  {row['snapshot_date']}")
    else:
        print(f"NVD records: 0 (No NVD data for {test_cve})")
    
    print()
    
    # Analyze the fundamental issue
    print("=== ROOT CAUSE ANALYSIS ===")
    
    # NVD snapshots are IRREGULAR - not daily
    nvd_daily_counts = nvd_df.groupBy(F.to_date("snapshot_date").alias("date")).count().orderBy("date")
    
    print("NVD snapshot frequency (first 10 dates):")
    for row in nvd_daily_counts.limit(10).collect():
        print(f"  {row['date']}: {row['count']:,} snapshots")
    
    print()
    
    # The core issue: EPSS is daily, NVD is irregular
    print("=== THE PROBLEM ===")
    print("1. EPSS data: Daily observations for every CVE")
    print("2. NVD data: Irregular snapshots when CVEs are updated")
    print("3. Current join: Exact date matching only")
    print("4. Result: Most EPSS dates have no matching NVD snapshot")
    print()
    
    # Solution analysis
    print("=== SOLUTION ANALYSIS ===")
    print("Current approach: LEFT JOIN on exact date match")
    print("  → Only ~0.1% of EPSS records get NVD data")
    print()
    print("Better approach: Forward-fill (last-known-value)")
    print("  → Use most recent NVD snapshot before each EPSS date")
    print("  → Represents 'state of knowledge' at that time")
    print("  → No future data leakage")
    print("  → Semantically correct for ML")
    print()
    
    # Demonstrate forward-fill potential
    print("=== FORWARD-FILL SIMULATION ===")
    
    # For our test CVE, simulate forward-fill
    if nvd_count > 0:
        print(f"Simulating forward-fill for {test_cve}...")
        
        # Get NVD snapshots as (date, data) pairs
        nvd_dates_only = nvd_cve.select(
            F.to_date("snapshot_date").alias("nvd_date"),
            F.lit(1).alias("has_nvd_data")  # Simplified - just mark presence
        ).distinct()
        
        # Get EPSS dates
        epss_dates_only = epss_cve.select("date").distinct()
        
        # Create all combinations and find the latest NVD date <= EPSS date
        from pyspark.sql.window import Window
        
        # Cross join and filter
        cross_join = epss_dates_only.crossJoin(nvd_dates_only).filter(
            F.col("nvd_date") <= F.col("date")
        )
        
        # For each EPSS date, get the latest NVD date
        windowSpec = Window.partitionBy("date").orderBy(F.desc("nvd_date"))
        
        forward_filled = cross_join.withColumn(
            "rn", F.row_number().over(windowSpec)
        ).filter(F.col("rn") == 1).select("date", "nvd_date", "has_nvd_data")
        
        coverage_with_ff = forward_filled.count()
        total_epss_dates = epss_dates_only.count()
        
        print(f"  Current coverage: {nvd_count}/{epss_count} = {(nvd_count/epss_count)*100:.1f}%")
        print(f"  With forward-fill: {coverage_with_ff}/{total_epss_dates} = {(coverage_with_ff/total_epss_dates)*100:.1f}%")
    
    print()
    print("=== CONCLUSION ===")
    print("✅ The high missing value rate is EXPECTED with exact date matching")
    print("✅ Forward-fill is the correct solution for this temporal ML problem")
    print("✅ It preserves temporal validity while maximizing data utilization")
    
    return epss_df, nvd_df

if __name__ == "__main__":
    epss_df, nvd_df = analyze_temporal_alignment() 