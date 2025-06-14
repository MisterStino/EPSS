#!/usr/bin/env python3
"""
Memory-Efficient CSAF Merge Analysis
Focuses on CSAF data integrity in the final merged dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def get_parquet_info(file_path):
    """Get basic info about parquet file without loading it"""
    try:
        # Try reading as parquet dataset (handles directories)
        dataset = pq.ParquetDataset(file_path)
        schema = dataset.schema
        
        # Get total size of all parquet files
        total_size = 0
        parquet_files = list(Path(file_path).glob('*.parquet'))
        for pf in parquet_files:
            total_size += pf.stat().st_size
        
        # Get column names
        columns = [field.name for field in schema]
        
        # Get approximate row count by reading metadata from first file
        if parquet_files:
            first_file = pq.ParquetFile(parquet_files[0])
            approx_rows_per_file = first_file.metadata.num_rows
            estimated_total_rows = approx_rows_per_file * len(parquet_files)
        else:
            estimated_total_rows = 0
        
        return {
            'num_rows': estimated_total_rows,
            'num_columns': len(columns),
            'columns': columns,
            'file_size_mb': total_size / (1024*1024),
            'num_files': len(parquet_files)
        }
    except Exception as e:
        return {'error': str(e)}

def identify_csaf_columns(columns):
    """Identify likely CSAF columns from column names"""
    csaf_keywords = [
        'event', 'source', 'discovery', 'release', 'threat', 'remediation',
        'stage', 'detail', 'doc_id', 'csaf', 'temporal', 'advisory',
        'vendor', 'product', 'vulnerability', 'impact', 'exploit'
    ]
    
    csaf_cols = []
    for col in columns:
        if any(keyword in col.lower() for keyword in csaf_keywords):
            csaf_cols.append(col)
    
    return csaf_cols

def analyze_csaf_coverage_chunked(file_path, csaf_cols, chunk_size=50000):
    """Analyze CSAF coverage by processing data in chunks"""
    print(f"📊 Analyzing CSAF coverage in chunks of {chunk_size:,} rows...")
    
    # Use pandas to read parquet directory in chunks
    try:
        # Get total rows estimate
        info = get_parquet_info(file_path)
        total_rows = info['num_rows']
    except:
        total_rows = 0
    
    # Initialize counters
    total_processed = 0
    csaf_coverage_counts = {col: 0 for col in csaf_cols}
    rows_with_any_csaf = 0
    sample_cves_with_csaf = []
    
    # Process in chunks using pandas
    try:
        for df_chunk in pd.read_parquet(file_path, chunksize=chunk_size):
            total_processed += len(df_chunk)
            
            if total_rows > 0:
                print(f"  Processing chunk: {total_processed:,}/{total_rows:,} rows ({total_processed/total_rows*100:.1f}%)")
            else:
                print(f"  Processing chunk: {total_processed:,} rows")
            
            # Count non-null values in CSAF columns
            for col in csaf_cols:
                if col in df_chunk.columns:
                    csaf_coverage_counts[col] += df_chunk[col].notna().sum()
            
            # Count rows with any CSAF data
            if csaf_cols:
                chunk_csaf_mask = df_chunk[csaf_cols].notna().any(axis=1)
                rows_with_any_csaf += chunk_csaf_mask.sum()
                
                # Collect sample CVEs with CSAF data
                cves_with_csaf = df_chunk[chunk_csaf_mask]['cve'].unique()
                sample_cves_with_csaf.extend(cves_with_csaf[:5])  # Take first 5 from each chunk
                
            # Break after processing a reasonable amount for analysis
            if total_processed >= 500000:  # Process max 500k rows for analysis
                print(f"  Stopping analysis after {total_processed:,} rows (sufficient for coverage analysis)")
                break
                
    except Exception as e:
        print(f"  ❌ Error processing chunks: {e}")
        # Fallback: try to read a small sample
        try:
            print("  Trying to read a small sample instead...")
            df_sample = pd.read_parquet(file_path).head(50000)
            total_processed = len(df_sample)
            
            # Count non-null values in CSAF columns
            for col in csaf_cols:
                if col in df_sample.columns:
                    csaf_coverage_counts[col] += df_sample[col].notna().sum()
            
            # Count rows with any CSAF data
            if csaf_cols:
                chunk_csaf_mask = df_sample[csaf_cols].notna().any(axis=1)
                rows_with_any_csaf += chunk_csaf_mask.sum()
                
                # Collect sample CVEs with CSAF data
                cves_with_csaf = df_sample[chunk_csaf_mask]['cve'].unique()
                sample_cves_with_csaf.extend(cves_with_csaf[:10])
                
        except Exception as e2:
            print(f"  ❌ Fallback also failed: {e2}")
            return {}, 0, []
    
    # Calculate coverage percentages
    coverage_results = {}
    for col, count in csaf_coverage_counts.items():
        if total_processed > 0:
            coverage_pct = (count / total_processed) * 100
        else:
            coverage_pct = 0
        coverage_results[col] = {
            'count': count,
            'percentage': coverage_pct
        }
    
    if total_processed > 0:
        overall_csaf_coverage = (rows_with_any_csaf / total_processed) * 100
    else:
        overall_csaf_coverage = 0
    
    return coverage_results, overall_csaf_coverage, sample_cves_with_csaf[:20]

def compare_with_source_csaf(sample_cves):
    """Compare sample CVEs with source CSAF data"""
    print(f"\n🔬 Comparing {len(sample_cves)} sample CVEs with source CSAF data...")
    
    try:
        # Load CSAF source data
        csaf_source = pd.read_parquet('data/csaf/processed/csaf_processed.parquet')
        print(f"CSAF source dataset: {len(csaf_source):,} rows")
        
        # Check which sample CVEs exist in CSAF source
        csaf_cves = set(csaf_source['cve'].unique())
        sample_cves_set = set(sample_cves)
        
        common_cves = csaf_cves.intersection(sample_cves_set)
        print(f"Sample CVEs in CSAF source: {len(common_cves)}/{len(sample_cves)}")
        
        if len(common_cves) > 0:
            print(f"✅ Found {len(common_cves)} sample CVEs in CSAF source data")
            print(f"Sample CVEs with CSAF data: {list(common_cves)[:5]}")
        else:
            print("❌ No sample CVEs found in CSAF source data!")
            
        return len(common_cves) > 0
        
    except Exception as e:
        print(f"⚠️  Could not load CSAF source data: {e}")
        return False

def analyze_specific_cve_detailed(cve_id, file_path, csaf_cols):
    """Analyze a specific CVE in detail"""
    print(f"\n🔍 Detailed analysis of {cve_id}:")
    
    # Load only data for this specific CVE
    try:
        # Use filter to load only relevant rows
        df_filtered = pd.read_parquet(file_path, filters=[('cve', '==', cve_id)])
        
        if len(df_filtered) == 0:
            print(f"  ❌ CVE {cve_id} not found in final dataset")
            return
        
        print(f"  📊 Found {len(df_filtered):,} rows for {cve_id}")
        print(f"  📅 Date range: {df_filtered['date'].min()} to {df_filtered['date'].max()}")
        
        # Check CSAF data presence
        csaf_data_present = {}
        for col in csaf_cols:
            if col in df_filtered.columns:
                non_null_count = df_filtered[col].notna().sum()
                csaf_data_present[col] = non_null_count
                if non_null_count > 0:
                    print(f"  ✅ {col}: {non_null_count} non-null values")
                    # Show sample values
                    sample_vals = df_filtered[col].dropna().head(3).tolist()
                    print(f"      Sample values: {sample_vals}")
        
        total_csaf_fields = sum(csaf_data_present.values())
        print(f"  📈 Total CSAF data points: {total_csaf_fields}")
        
        return total_csaf_fields > 0
        
    except Exception as e:
        print(f"  ❌ Error analyzing {cve_id}: {e}")
        return False

def main():
    """Main analysis function"""
    print("🚀 MEMORY-EFFICIENT CSAF MERGE ANALYSIS")
    print("=" * 50)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    file_path = 'data/full_db/processed/final_full_data.parquet/'
    
    # 1. Get basic file info
    print("\n📂 DATASET OVERVIEW")
    print("-" * 30)
    info = get_parquet_info(file_path)
    if 'error' in info:
        print(f"❌ Error reading file: {info['error']}")
        return
    
    print(f"File size: {info['file_size_mb']:.1f} MB")
    print(f"Total rows (estimated): {info['num_rows']:,}")
    print(f"Total columns: {info['num_columns']}")
    print(f"Number of parquet files: {info.get('num_files', 'Unknown')}")
    
    # 2. Identify CSAF columns
    print(f"\n🔍 CSAF COLUMN IDENTIFICATION")
    print("-" * 35)
    csaf_cols = identify_csaf_columns(info['columns'])
    print(f"Total columns: {len(info['columns'])}")
    print(f"CSAF columns identified: {len(csaf_cols)}")
    
    if len(csaf_cols) == 0:
        print("❌ No CSAF columns found! This suggests the merge may have failed.")
        print("\nAll columns:")
        for i, col in enumerate(info['columns']):
            print(f"  {i+1:2d}. {col}")
        return
    
    print("CSAF columns found:")
    for i, col in enumerate(csaf_cols):
        print(f"  {i+1:2d}. {col}")
    
    # 3. Analyze CSAF coverage
    print(f"\n📊 CSAF COVERAGE ANALYSIS")
    print("-" * 30)
    coverage_results, overall_coverage, sample_cves = analyze_csaf_coverage_chunked(
        file_path, csaf_cols, chunk_size=50000
    )
    
    print(f"\nCSAF Column Coverage:")
    for col, data in coverage_results.items():
        print(f"  {col}: {data['count']:,} ({data['percentage']:.2f}%)")
    
    print(f"\nOverall CSAF Coverage:")
    print(f"  Rows with ANY CSAF data: {overall_coverage:.2f}%")
    print(f"  Rows with NO CSAF data: {100-overall_coverage:.2f}%")
    
    if overall_coverage < 1.0:
        print("⚠️  WARNING: Very low CSAF coverage detected!")
    
    # 4. Compare with source CSAF data
    print(f"\n🔬 SOURCE DATA COMPARISON")
    print("-" * 30)
    source_comparison_success = compare_with_source_csaf(sample_cves)
    
    # 5. Detailed analysis of specific CVEs
    print(f"\n🎯 DETAILED CVE ANALYSIS")
    print("-" * 25)
    
    # Analyze a few specific CVEs in detail
    test_cves = sample_cves[:5] if sample_cves else []
    if not test_cves:
        print("❌ No sample CVEs with CSAF data found for detailed analysis")
    else:
        successful_analyses = 0
        for cve in test_cves:
            success = analyze_specific_cve_detailed(cve, file_path, csaf_cols)
            if success:
                successful_analyses += 1
        
        print(f"\n📊 Detailed Analysis Summary:")
        print(f"  CVEs analyzed: {len(test_cves)}")
        print(f"  CVEs with CSAF data: {successful_analyses}")
        print(f"  Success rate: {successful_analyses/len(test_cves)*100:.1f}%")
    
    # 6. Final assessment
    print(f"\n🎯 FINAL ASSESSMENT")
    print("=" * 20)
    
    if len(csaf_cols) == 0:
        print("❌ CRITICAL: No CSAF columns found - merge likely failed completely")
    elif overall_coverage < 0.1:
        print("❌ CRITICAL: CSAF coverage < 0.1% - merge likely failed")
    elif overall_coverage < 1.0:
        print("⚠️  WARNING: CSAF coverage < 1% - merge may have issues")
    elif overall_coverage < 10.0:
        print("⚠️  CAUTION: CSAF coverage < 10% - check if this is expected")
    else:
        print("✅ GOOD: CSAF coverage looks reasonable")
    
    print(f"\n📊 Summary Statistics:")
    print(f"  CSAF columns: {len(csaf_cols)}")
    print(f"  Overall coverage: {overall_coverage:.2f}%")
    print(f"  Sample CVEs found: {len(sample_cves)}")
    print(f"  Source comparison: {'✅ Success' if source_comparison_success else '❌ Failed'}")
    
    print(f"\n✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 