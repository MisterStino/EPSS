"""
Comprehensive inspection of NVD processed dataset to understand missing value patterns.
"""

import sys
import os
sys.path.append('.')

from t3_spark.session import get_spark_session
import pyspark.sql.functions as F
from pyspark.sql.types import *

def inspect_nvd_processed():
    """Inspect the NVD processed dataset to understand missing values."""
    
    print("=" * 70)
    print("NVD PROCESSED DATASET INSPECTION")
    print("=" * 70)
    
    spark = get_spark_session("NVD_Inspection")
    
    # Load the processed dataset
    print("Loading NVD processed dataset...")
    df = spark.read.parquet("data/nvd/processed/nvd_processed.parquet")
    
    print(f"✓ Dataset loaded: {df.count():,} rows × {len(df.columns)} columns")
    print()
    
    # Basic structure analysis
    print("=== DATASET STRUCTURE ===")
    print(f"Columns: {df.columns}")
    print()
    
    # Date range analysis
    print("=== TEMPORAL COVERAGE ===")
    date_range = df.select(F.min("date"), F.max("date")).collect()[0]
    print(f"Date range: {date_range[0]} to {date_range[1]}")
    
    cve_count = df.select("cve").distinct().count()
    print(f"Unique CVEs: {cve_count:,}")
    print()
    
    # Missing value analysis for NVD columns
    nvd_columns = [col for col in df.columns if col not in ['cve', 'date', 'epss']]
    print(f"=== MISSING VALUES ANALYSIS ({len(nvd_columns)} NVD columns) ===")
    
    total_rows = df.count()
    
    for col in nvd_columns[:10]:  # First 10 columns
        null_count = df.filter(F.col(col).isNull()).count()
        non_null_count = total_rows - null_count
        null_pct = (null_count / total_rows) * 100
        
        print(f"{col:25} | {non_null_count:>12,} non-null | {null_count:>12,} null ({null_pct:5.1f}%)")
    
    print(f"... and {len(nvd_columns)-10} more columns")
    print()
    
    # Sample data inspection
    print("=== SAMPLE DATA (First 5 rows) ===")
    sample_df = df.limit(5)
    for row in sample_df.collect():
        print(f"CVE: {row['cve']}, Date: {row['date']}, EPSS: {row['epss']}")
        print(f"  NVD fields: {dict((col, row[col]) for col in nvd_columns[:3])}")
        print()
    
    # CVE-specific analysis: Check if ANY CVE has NVD data
    print("=== CVE-LEVEL DATA AVAILABILITY ===")
    
    # CVEs with at least one non-null NVD value
    cves_with_nvd_data = df.filter(
        F.col(nvd_columns[0]).isNotNull() |  # At least one NVD column is not null
        F.col(nvd_columns[1]).isNotNull() if len(nvd_columns) > 1 else F.lit(False)
    ).select("cve").distinct().count()
    
    print(f"CVEs with ANY NVD data: {cves_with_nvd_data:,} out of {cve_count:,}")
    print(f"CVEs with NO NVD data: {cve_count - cves_with_nvd_data:,}")
    print()
    
    # Temporal pattern analysis
    print("=== TEMPORAL PATTERN ANALYSIS ===")
    
    # Group by date and count non-null values
    daily_coverage = df.groupBy("date").agg(
        F.count("*").alias("total_cves"),
        F.sum(F.when(F.col(nvd_columns[0]).isNotNull(), 1).otherwise(0)).alias("cves_with_nvd")
    ).orderBy("date")
    
    print("Sample daily coverage (first 10 dates):")
    for row in daily_coverage.limit(10).collect():
        coverage_pct = (row['cves_with_nvd'] / row['total_cves']) * 100 if row['total_cves'] > 0 else 0
        print(f"  {row['date']}: {row['cves_with_nvd']:,}/{row['total_cves']:,} CVEs ({coverage_pct:.1f}%) have NVD data")
    
    print()
    
    # Check for temporal gaps in NVD data
    print("=== NVD SNAPSHOT TEMPORAL ANALYSIS ===")
    
    # Load original NVD data to understand snapshot patterns
    print("Loading original NVD data for comparison...")
    nvd_raw = spark.read.parquet("data/nvd/raw/cve_snapshots_irregular.parquet")
    
    print(f"Original NVD snapshots: {nvd_raw.count():,} rows")
    
    nvd_date_range = nvd_raw.select(F.min("snapshot_date"), F.max("snapshot_date")).collect()[0]
    print(f"NVD date range: {nvd_date_range[0]} to {nvd_date_range[1]}")
    
    # Check overlap between EPSS and NVD dates
    epss_dates = df.select("date").distinct()
    nvd_dates = nvd_raw.select(F.to_date("snapshot_date").alias("date")).distinct()
    
    common_dates = epss_dates.intersect(nvd_dates).count()
    total_epss_dates = epss_dates.count()
    
    print(f"Common dates between EPSS and NVD: {common_dates:,} out of {total_epss_dates:,} EPSS dates")
    print(f"Date overlap: {(common_dates/total_epss_dates)*100:.1f}%")
    print()
    
    # Specific CVE trace analysis
    print("=== SPECIFIC CVE ANALYSIS ===")
    
    # Pick a few CVEs and trace their NVD data availability
    sample_cves = df.select("cve").distinct().limit(3).collect()
    
    for cve_row in sample_cves:
        cve_id = cve_row['cve']
        print(f"\nCVE: {cve_id}")
        
        # EPSS records for this CVE
        cve_epss = df.filter(F.col("cve") == cve_id).orderBy("date")
        epss_count = cve_epss.count()
        
        # NVD records for this CVE (with data)
        cve_nvd_count = cve_epss.filter(F.col(nvd_columns[0]).isNotNull()).count()
        
        print(f"  EPSS records: {epss_count:,}")
        print(f"  Records with NVD data: {cve_nvd_count:,}")
        print(f"  Coverage: {(cve_nvd_count/epss_count)*100:.1f}%")
        
        # Show first few records
        print("  Sample records:")
        for row in cve_epss.limit(3).collect():
            nvd_status = "Has NVD data" if row[nvd_columns[0]] is not None else "No NVD data"
            print(f"    {row['date']}: EPSS={row['epss']:.3f}, {nvd_status}")
    
    print()
    
    return df

if __name__ == "__main__":
    df = inspect_nvd_processed() 