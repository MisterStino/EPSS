"""
Quick inspection of the fixed NVD processed result.
Check if the deduplication resolved the row count mismatch.
"""

import sys
import os
sys.path.append('.')

from t3_spark.session import get_spark_session
import pyspark.sql.functions as F

def inspect_fixed_result():
    """Inspect the fixed NVD processed result."""
    
    print("=" * 70)
    print("INSPECTING FIXED NVD PROCESSED RESULT")
    print("=" * 70)
    
    spark = get_spark_session("Inspect_Fixed_Result")
    
    # Load the datasets
    epss_df = spark.read.parquet("data/epss/processed/epss_processed.parquet")
    nvd_processed_df = spark.read.parquet("data/nvd/processed/nvd_processed.parquet")
    
    epss_count = epss_df.count()
    nvd_processed_count = nvd_processed_df.count()
    
    print(f"Original EPSS rows: {epss_count:,}")
    print(f"NVD processed rows: {nvd_processed_count:,}")
    
    if epss_count == nvd_processed_count:
        print("✅ SUCCESS: Row counts match perfectly!")
        print("🎯 DUPLICATE PROBLEM SOLVED!")
    else:
        difference = nvd_processed_count - epss_count
        print(f"❌ MISMATCH: {difference:,} row difference")
        
    # Check column count
    epss_cols = len(epss_df.columns)
    nvd_cols = len(nvd_processed_df.columns)
    
    print(f"\nColumn counts:")
    print(f"EPSS columns: {epss_cols}")
    print(f"NVD processed columns: {nvd_cols}")
    print(f"Added NVD features: {nvd_cols - epss_cols}")
    
    # Check NVD feature coverage
    nvd_features = [col for col in nvd_processed_df.columns if col not in ['cve', 'date', 'epss']]
    if nvd_features:
        sample_feature = nvd_features[0]
        non_null_count = nvd_processed_df.filter(F.col(sample_feature).isNotNull()).count()
        coverage_pct = (non_null_count / nvd_processed_count) * 100
        
        print(f"\nNVD coverage check (using '{sample_feature}'):")
        print(f"Non-null records: {non_null_count:,}")
        print(f"Coverage: {coverage_pct:.1f}%")
    
    return epss_count == nvd_processed_count

if __name__ == "__main__":
    success = inspect_fixed_result()
    if success:
        print("\n🎉 VALIDATION PASSED: Grid approach with deduplication worked!")
    else:
        print("\n❌ VALIDATION FAILED: Still have issues to resolve") 