"""
Investigate NVD duplicate snapshots on the same date.
This will help us understand why the grid expansion is happening.
"""

import sys
import os
sys.path.append('.')

from t3_spark.session import get_spark_session
import pyspark.sql.functions as F

def investigate_nvd_duplicates():
    """Investigate NVD duplicates causing grid expansion."""
    
    print("=" * 70)
    print("INVESTIGATING NVD DUPLICATE SNAPSHOTS")
    print("=" * 70)
    
    spark = get_spark_session("NVD_Duplicate_Investigation")
    
    # Load NVD data
    nvd_df = spark.read.parquet("data/nvd/raw/cve_snapshots_irregular.parquet")
    
    print(f"Original NVD data: {nvd_df.count():,} rows")
    
    # Normalize to date (this is where duplicates are created)
    nvd_normalized = nvd_df.select(
        F.col("cve_id").alias("cve"),
        F.to_date(F.col("snapshot_date")).alias("date"),
        F.col("snapshot_date").alias("original_timestamp"),  # Keep for analysis
        *[col for col in nvd_df.columns if col not in ['cve_id', 'snapshot_date']]
    )
    
    print(f"Normalized NVD data: {nvd_normalized.count():,} rows")
    
    # Check for duplicates on (cve, date) after normalization
    print("\n=== DUPLICATE ANALYSIS ===")
    
    duplicates = nvd_normalized.groupBy("cve", "date").count().filter(F.col("count") > 1)
    duplicate_pairs = duplicates.count()
    
    print(f"(CVE, date) pairs with duplicates: {duplicate_pairs:,}")
    
    if duplicate_pairs > 0:
        print(f"Total duplicate records: {duplicates.agg(F.sum('count')).collect()[0][0]:,}")
        
        # Show examples of duplicates
        print("\n=== EXAMPLE DUPLICATES ===")
        sample_duplicates = duplicates.orderBy(F.desc("count")).limit(5)
        
        for row in sample_duplicates.collect():
            cve, date, count = row['cve'], row['date'], row['count']
            print(f"\nCVE {cve} on {date}: {count} snapshots")
            
            # Show the actual timestamps for this CVE+date
            examples = nvd_normalized.filter(
                (F.col("cve") == cve) & (F.col("date") == date)
            ).select("original_timestamp").orderBy("original_timestamp")
            
            for ts_row in examples.collect():
                print(f"  Timestamp: {ts_row['original_timestamp']}")
    
    # Analyze the impact on grid size
    print("\n=== GRID SIZE IMPACT ===")
    
    unique_cve_dates = nvd_normalized.select("cve", "date").distinct().count()
    total_records = nvd_normalized.count()
    
    print(f"Unique (CVE, date) pairs: {unique_cve_dates:,}")
    print(f"Total NVD records: {total_records:,}")
    print(f"Expansion factor: {total_records / unique_cve_dates:.2f}x")
    
    # Calculate expected grid sizes
    nvd_cves = nvd_df.select(F.col("cve_id").alias("cve")).distinct().count()
    epss_dates = 1160  # From terminal output
    
    expected_grid_size = nvd_cves * epss_dates
    actual_grid_after_join = expected_grid_size + (total_records - unique_cve_dates)
    
    print(f"\nGrid size calculation:")
    print(f"  Expected grid: {nvd_cves:,} CVEs × {epss_dates} dates = {expected_grid_size:,}")
    print(f"  After joining duplicates: {actual_grid_after_join:,}")
    print(f"  Observed in terminal: 343,427,751")
    
    # Solution: Show how to deduplicate
    print("\n=== SOLUTION ===")
    print("Need to deduplicate NVD data before joining to grid:")
    print("1. Keep only the LATEST snapshot per (CVE, date)")
    print("2. Or aggregate/merge multiple snapshots per day")
    
    # Demonstrate the fix
    nvd_deduplicated = nvd_normalized.orderBy("cve", "date", F.desc("original_timestamp")).dropDuplicates(["cve", "date"])
    
    dedup_count = nvd_deduplicated.count()
    print(f"\nAfter keeping latest snapshot per (CVE, date): {dedup_count:,} records")
    print(f"Reduction: {total_records - dedup_count:,} duplicate records removed")
    
    return nvd_df, nvd_normalized, nvd_deduplicated

if __name__ == "__main__":
    nvd_df, nvd_normalized, nvd_deduplicated = investigate_nvd_duplicates() 