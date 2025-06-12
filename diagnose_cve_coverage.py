"""
Diagnostic script to investigate CVE coverage discrepancy.
Why do we only have 613,738 records for CVEs with no NVD data?
"""

import sys
import os
sys.path.append('.')

from t3_spark.session import get_spark_session
import pyspark.sql.functions as F

def diagnose_cve_coverage():
    """Diagnose the CVE coverage issue."""
    
    print("=" * 70)
    print("DIAGNOSING CVE COVERAGE DISCREPANCY")
    print("=" * 70)
    
    spark = get_spark_session("CVE_Coverage_Diagnosis")
    
    # Load datasets
    print("Loading datasets...")
    epss_df = spark.read.parquet("data/epss/processed/epss_processed.parquet")
    nvd_df = spark.read.parquet("data/nvd/raw/cve_snapshots_irregular.parquet")
    
    print(f"EPSS: {epss_df.count():,} total records")
    print(f"NVD: {nvd_df.count():,} total records")
    print()
    
    # Step 1: Analyze unique CVEs
    print("=== STEP 1: UNIQUE CVE ANALYSIS ===")
    
    epss_cves = epss_df.select("cve").distinct()
    nvd_cves = nvd_df.select(F.col("cve_id").alias("cve")).distinct()
    
    epss_cve_count = epss_cves.count()
    nvd_cve_count = nvd_cves.count()
    
    print(f"Unique CVEs in EPSS: {epss_cve_count:,}")
    print(f"Unique CVEs in NVD: {nvd_cve_count:,}")
    
    # Find overlap
    common_cves = epss_cves.intersect(nvd_cves)
    common_cve_count = common_cves.count()
    
    epss_only_cves = epss_cves.subtract(nvd_cves)
    epss_only_count = epss_only_cves.count()
    
    print(f"CVEs in BOTH datasets: {common_cve_count:,}")
    print(f"CVEs ONLY in EPSS: {epss_only_count:,}")
    print(f"Expected: {epss_cve_count:,} = {common_cve_count:,} + {epss_only_count:,}")
    print(f"Actual: {epss_cve_count:,} = {common_cve_count + epss_only_count:,} ✓")
    print()
    
    # Step 2: Analyze records per CVE
    print("=== STEP 2: RECORDS PER CVE ANALYSIS ===")
    
    # Get records count per CVE for EPSS-only CVEs
    epss_only_records = epss_df.join(epss_only_cves, on="cve", how="inner")
    epss_only_record_count = epss_only_records.count()
    
    print(f"EPSS records for EPSS-only CVEs: {epss_only_record_count:,}")
    print(f"This should match our 613,738 if logic is correct...")
    
    # Check if this matches our suspicious number
    if epss_only_record_count == 613738:
        print("✅ MATCH: Our logic is working correctly")
    else:
        print(f"❌ MISMATCH: Expected ~613,738, got {epss_only_record_count:,}")
    
    # Analyze temporal distribution
    records_per_cve = epss_only_records.groupBy("cve").count().orderBy(F.desc("count"))
    
    print("\nRecords per CVE distribution (EPSS-only CVEs):")
    stats = records_per_cve.select(
        F.min("count").alias("min_records"),
        F.max("count").alias("max_records"), 
        F.avg("count").alias("avg_records"),
        F.expr("percentile_approx(count, 0.5)").alias("median_records")
    ).collect()[0]
    
    print(f"  Min records per CVE: {stats['min_records']}")
    print(f"  Max records per CVE: {stats['max_records']}")
    print(f"  Avg records per CVE: {stats['avg_records']:.1f}")
    print(f"  Median records per CVE: {stats['median_records']}")
    
    print("\nTop 10 EPSS-only CVEs by record count:")
    for row in records_per_cve.limit(10).collect():
        print(f"  {row['cve']}: {row['count']:,} records")
    
    print()
    
    # Step 3: Temporal analysis
    print("=== STEP 3: TEMPORAL ANALYSIS ===")
    
    # Check date ranges for EPSS-only CVEs
    epss_date_range = epss_df.select(F.min("date"), F.max("date")).collect()[0]
    epss_only_date_range = epss_only_records.select(F.min("date"), F.max("date")).collect()[0]
    
    print(f"Full EPSS date range: {epss_date_range[0]} to {epss_date_range[1]}")
    print(f"EPSS-only date range: {epss_only_date_range[0]} to {epss_only_date_range[1]}")
    
    # Check if EPSS-only CVEs have shorter tracking periods
    total_epss_dates = epss_df.select("date").distinct().count()
    epss_only_dates = epss_only_records.select("date").distinct().count()
    
    print(f"Total unique dates in EPSS: {total_epss_dates:,}")
    print(f"Unique dates for EPSS-only CVEs: {epss_only_dates:,}")
    
    if epss_only_dates < total_epss_dates:
        print("🔍 FINDING: EPSS-only CVEs have shorter temporal coverage!")
        missing_dates = total_epss_dates - epss_only_dates
        print(f"   Missing {missing_dates:,} dates for EPSS-only CVEs")
    
    print()
    
    # Step 4: Sample investigation
    print("=== STEP 4: SAMPLE CVE INVESTIGATION ===")
    
    # Pick a few EPSS-only CVEs and investigate their temporal patterns
    sample_cves = epss_only_cves.limit(3).collect()
    
    for cve_row in sample_cves:
        cve_id = cve_row['cve']
        cve_records = epss_df.filter(F.col("cve") == cve_id).orderBy("date")
        cve_count = cve_records.count()
        
        if cve_count > 0:
            date_range = cve_records.select(F.min("date"), F.max("date")).collect()[0]
            print(f"CVE {cve_id}:")
            print(f"  Records: {cve_count:,}")
            print(f"  Date range: {date_range[0]} to {date_range[1]}")
            
            # Check if this CVE has full temporal coverage
            from datetime import datetime, timedelta
            start_date = date_range[0]
            end_date = date_range[1]
            expected_days = (end_date - start_date).days + 1
            
            print(f"  Expected days: {expected_days:,}")
            print(f"  Actual records: {cve_count:,}")
            
            if cve_count < expected_days:
                print(f"  ⚠️  SPARSE: Missing {expected_days - cve_count:,} days")
            else:
                print(f"  ✅ COMPLETE: Full temporal coverage")
        
        print()
    
    # Step 5: Conclusion
    print("=== CONCLUSION ===")
    
    expected_records = epss_only_count * total_epss_dates
    actual_records = epss_only_record_count
    
    print(f"If all EPSS-only CVEs had full coverage:")
    print(f"  {epss_only_count:,} CVEs × {total_epss_dates:,} dates = {expected_records:,} records")
    print(f"Actually found: {actual_records:,} records")
    
    if actual_records < expected_records:
        coverage_pct = (actual_records / expected_records) * 100
        print(f"  Coverage: {coverage_pct:.1f}%")
        print(f"  📊 FINDING: EPSS-only CVEs have incomplete temporal coverage")
        print(f"     This explains why we see fewer records than expected")
    
    return epss_df, nvd_df

if __name__ == "__main__":
    epss_df, nvd_df = diagnose_cve_coverage() 