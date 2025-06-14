#!/usr/bin/env python3
"""
Rigorous EDA on Final Merged Dataset
Focuses on CSAF merge integrity and data quality validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load all relevant datasets for comparison"""
    print("📂 Loading datasets...")
    
    # Load final merged dataset
    final_df = pd.read_parquet('data/full_db/processed/final_full_data.parquet')
    print(f"Final dataset: {len(final_df):,} rows × {len(final_df.columns)} columns")
    
    # Load individual source datasets for comparison
    datasets = {}
    
    # EPSS base
    try:
        datasets['epss'] = pd.read_parquet('data/epss/processed/epss_processed.parquet')
        print(f"EPSS dataset: {len(datasets['epss']):,} rows")
    except:
        print("⚠️  EPSS dataset not found")
    
    # CSAF
    try:
        datasets['csaf'] = pd.read_parquet('data/csaf/processed/csaf_processed.parquet')
        print(f"CSAF dataset: {len(datasets['csaf']):,} rows")
    except:
        print("⚠️  CSAF dataset not found")
    
    # NVD
    try:
        datasets['nvd'] = pd.read_parquet('data/nvd/processed/nvd_processed.parquet')
        print(f"NVD dataset: {len(datasets['nvd']):,} rows")
    except:
        print("⚠️  NVD dataset not found")
    
    # EPSS Features
    try:
        datasets['epss_features'] = pd.read_parquet('data/epss_features/processed/epss_features_processed.parquet')
        print(f"EPSS Features dataset: {len(datasets['epss_features']):,} rows")
    except:
        print("⚠️  EPSS Features dataset not found")
    
    return final_df, datasets

def analyze_column_structure(final_df, datasets):
    """Analyze column structure and identify source modules"""
    print("\n🔍 COLUMN STRUCTURE ANALYSIS")
    print("=" * 50)
    
    print(f"Final dataset columns ({len(final_df.columns)}):")
    for i, col in enumerate(final_df.columns):
        print(f"  {i+1:2d}. {col}")
    
    # Identify likely CSAF columns
    csaf_keywords = ['event', 'source', 'discovery', 'release', 'threat', 'remediation', 
                     'stage', 'detail', 'doc_id', 'csaf', 'temporal']
    
    likely_csaf_cols = [col for col in final_df.columns 
                       if any(keyword in col.lower() for keyword in csaf_keywords)]
    
    print(f"\nLikely CSAF columns ({len(likely_csaf_cols)}):")
    for col in likely_csaf_cols:
        print(f"  - {col}")
    
    return likely_csaf_cols

def analyze_csaf_coverage(final_df, datasets, csaf_cols):
    """Analyze CSAF data coverage in the final dataset"""
    print("\n📊 CSAF COVERAGE ANALYSIS")
    print("=" * 50)
    
    if not csaf_cols:
        print("❌ No CSAF columns identified in final dataset")
        return
    
    # Check for non-null values in CSAF columns
    csaf_coverage = {}
    for col in csaf_cols:
        if col in final_df.columns:
            non_null_count = final_df[col].notna().sum()
            coverage_pct = (non_null_count / len(final_df)) * 100
            csaf_coverage[col] = {
                'non_null_count': non_null_count,
                'coverage_pct': coverage_pct
            }
            print(f"  {col}: {non_null_count:,} non-null ({coverage_pct:.2f}%)")
    
    # Overall CSAF coverage (any CSAF column has data)
    csaf_mask = final_df[csaf_cols].notna().any(axis=1)
    rows_with_csaf = csaf_mask.sum()
    overall_coverage = (rows_with_csaf / len(final_df)) * 100
    
    print(f"\nOverall CSAF coverage:")
    print(f"  Rows with ANY CSAF data: {rows_with_csaf:,} ({overall_coverage:.2f}%)")
    print(f"  Rows with NO CSAF data: {len(final_df) - rows_with_csaf:,} ({100-overall_coverage:.2f}%)")
    
    return csaf_coverage, csaf_mask

def compare_specific_cves(final_df, datasets, sample_size=10):
    """Compare specific CVEs across datasets to validate merge"""
    print("\n🔬 SPECIFIC CVE COMPARISON")
    print("=" * 50)
    
    if 'csaf' not in datasets:
        print("❌ CSAF dataset not available for comparison")
        return
    
    # Get CVEs that exist in both CSAF and final dataset
    csaf_cves = set(datasets['csaf']['cve'].unique())
    final_cves = set(final_df['cve'].unique())
    common_cves = csaf_cves.intersection(final_cves)
    
    print(f"CVEs in CSAF dataset: {len(csaf_cves):,}")
    print(f"CVEs in final dataset: {len(final_cves):,}")
    print(f"Common CVEs: {len(common_cves):,}")
    
    if len(common_cves) == 0:
        print("❌ No common CVEs found between CSAF and final dataset!")
        return
    
    # Sample CVEs for detailed comparison
    sample_cves = list(common_cves)[:sample_size]
    print(f"\nAnalyzing {len(sample_cves)} sample CVEs...")
    
    comparison_results = []
    
    for cve in sample_cves:
        print(f"\n🔍 Analyzing {cve}:")
        
        # Get data from CSAF dataset
        csaf_data = datasets['csaf'][datasets['csaf']['cve'] == cve]
        csaf_dates = set(csaf_data['date'].dt.date if hasattr(csaf_data['date'], 'dt') else csaf_data['date'])
        
        # Get data from final dataset
        final_data = final_df[final_df['cve'] == cve]
        final_dates = set(final_data['date'].dt.date if hasattr(final_data['date'], 'dt') else final_data['date'])
        
        # Check for CSAF data in final dataset
        csaf_cols_in_final = [col for col in final_data.columns if 'event' in col.lower() or 'source' in col.lower()]
        csaf_data_in_final = final_data[csaf_cols_in_final].notna().any(axis=1).sum()
        
        result = {
            'cve': cve,
            'csaf_rows': len(csaf_data),
            'csaf_dates': len(csaf_dates),
            'final_rows': len(final_data),
            'final_dates': len(final_dates),
            'csaf_data_in_final': csaf_data_in_final,
            'merge_success': csaf_data_in_final > 0
        }
        
        comparison_results.append(result)
        
        print(f"  CSAF dataset: {len(csaf_data):,} rows, {len(csaf_dates)} unique dates")
        print(f"  Final dataset: {len(final_data):,} rows, {len(final_dates)} unique dates")
        print(f"  CSAF data in final: {csaf_data_in_final:,} rows")
        print(f"  Merge success: {'✅' if result['merge_success'] else '❌'}")
        
        # Show sample of actual data
        if len(final_data) > 0:
            sample_row = final_data.iloc[0]
            csaf_sample_cols = [col for col in csaf_cols_in_final[:3]]  # First 3 CSAF columns
            print(f"  Sample CSAF values: {dict(sample_row[csaf_sample_cols])}")
    
    # Summary of comparison results
    successful_merges = sum(1 for r in comparison_results if r['merge_success'])
    print(f"\n📊 Merge Success Summary:")
    print(f"  Successful merges: {successful_merges}/{len(comparison_results)} ({successful_merges/len(comparison_results)*100:.1f}%)")
    
    return comparison_results

def analyze_temporal_alignment(final_df, datasets):
    """Analyze temporal alignment between datasets"""
    print("\n⏰ TEMPORAL ALIGNMENT ANALYSIS")
    print("=" * 50)
    
    # Date range analysis
    final_date_range = (final_df['date'].min(), final_df['date'].max())
    print(f"Final dataset date range: {final_date_range[0]} to {final_date_range[1]}")
    
    for name, dataset in datasets.items():
        if 'date' in dataset.columns:
            date_range = (dataset['date'].min(), dataset['date'].max())
            print(f"{name.upper()} date range: {date_range[0]} to {date_range[1]}")
    
    # Check for date misalignment issues
    if 'csaf' in datasets:
        csaf_dates = set(datasets['csaf']['date'].dt.date if hasattr(datasets['csaf']['date'], 'dt') else datasets['csaf']['date'])
        final_dates = set(final_df['date'].dt.date if hasattr(final_df['date'], 'dt') else final_df['date'])
        
        common_dates = csaf_dates.intersection(final_dates)
        print(f"\nDate overlap with CSAF:")
        print(f"  CSAF unique dates: {len(csaf_dates):,}")
        print(f"  Final unique dates: {len(final_dates):,}")
        print(f"  Common dates: {len(common_dates):,}")
        print(f"  Date overlap: {len(common_dates)/len(csaf_dates)*100:.1f}%")

def analyze_data_quality(final_df, csaf_cols):
    """Analyze overall data quality and completeness"""
    print("\n🎯 DATA QUALITY ANALYSIS")
    print("=" * 50)
    
    # Missing data analysis
    print("Missing data by column:")
    missing_data = final_df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing_data / len(final_df) * 100).round(2)
    
    for col in missing_data.head(20).index:
        print(f"  {col}: {missing_data[col]:,} ({missing_pct[col]:.1f}%)")
    
    # CSAF-specific quality checks
    if csaf_cols:
        print(f"\nCSAF columns quality:")
        for col in csaf_cols:
            if col in final_df.columns:
                unique_vals = final_df[col].nunique()
                print(f"  {col}: {unique_vals:,} unique values")

def create_visualizations(final_df, csaf_coverage, csaf_mask):
    """Create visualizations for the analysis"""
    print("\n📈 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CSAF Merge Analysis', fontsize=16, fontweight='bold')
    
    # 1. CSAF Coverage by Column
    if csaf_coverage:
        coverage_data = {col: data['coverage_pct'] for col, data in csaf_coverage.items()}
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(coverage_data)), list(coverage_data.values()))
        ax1.set_title('CSAF Column Coverage (%)')
        ax1.set_xlabel('CSAF Columns')
        ax1.set_ylabel('Coverage Percentage')
        ax1.set_xticks(range(len(coverage_data)))
        ax1.set_xticklabels(list(coverage_data.keys()), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, coverage_data.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}%', ha='center', va='bottom')
    
    # 2. Overall CSAF vs Non-CSAF rows
    csaf_counts = [csaf_mask.sum(), len(final_df) - csaf_mask.sum()]
    ax2 = axes[0, 1]
    wedges, texts, autotexts = ax2.pie(csaf_counts, labels=['With CSAF', 'Without CSAF'], 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Rows with vs without CSAF Data')
    
    # 3. Data completeness heatmap (sample)
    ax3 = axes[1, 0]
    sample_data = final_df.sample(min(1000, len(final_df)))
    csaf_sample_cols = [col for col in sample_data.columns if any(keyword in col.lower() 
                       for keyword in ['event', 'source', 'discovery', 'release'])][:10]
    
    if csaf_sample_cols:
        completeness_matrix = sample_data[csaf_sample_cols].notna().astype(int)
        sns.heatmap(completeness_matrix.T, cmap='RdYlBu', cbar=True, ax=ax3)
        ax3.set_title('CSAF Data Completeness (Sample)')
        ax3.set_xlabel('Sample Rows')
        ax3.set_ylabel('CSAF Columns')
    
    # 4. Temporal distribution
    ax4 = axes[1, 1]
    if 'date' in final_df.columns:
        # Group by month and count rows with CSAF data
        final_df_copy = final_df.copy()
        final_df_copy['year_month'] = pd.to_datetime(final_df_copy['date']).dt.to_period('M')
        final_df_copy['has_csaf'] = csaf_mask
        
        temporal_summary = final_df_copy.groupby('year_month')['has_csaf'].agg(['count', 'sum']).reset_index()
        temporal_summary['csaf_pct'] = (temporal_summary['sum'] / temporal_summary['count'] * 100)
        
        # Plot last 24 months
        recent_data = temporal_summary.tail(24)
        ax4.plot(range(len(recent_data)), recent_data['csaf_pct'], marker='o')
        ax4.set_title('CSAF Coverage Over Time (Last 24 Months)')
        ax4.set_xlabel('Time Period')
        ax4.set_ylabel('CSAF Coverage (%)')
        ax4.set_xticks(range(0, len(recent_data), 6))
        ax4.set_xticklabels([str(recent_data.iloc[i]['year_month']) for i in range(0, len(recent_data), 6)], 
                           rotation=45)
    
    plt.tight_layout()
    plt.savefig('csaf_merge_analysis.png', dpi=300, bbox_inches='tight')
    print("📊 Visualizations saved to: csaf_merge_analysis.png")
    plt.show()

def main():
    """Main analysis function"""
    print("🚀 RIGOROUS EDA ON FINAL MERGED DATASET")
    print("=" * 60)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load datasets
    final_df, datasets = load_datasets()
    
    # Analyze column structure
    csaf_cols = analyze_column_structure(final_df, datasets)
    
    # Analyze CSAF coverage
    csaf_coverage, csaf_mask = analyze_csaf_coverage(final_df, datasets, csaf_cols)
    
    # Compare specific CVEs
    comparison_results = compare_specific_cves(final_df, datasets, sample_size=15)
    
    # Analyze temporal alignment
    analyze_temporal_alignment(final_df, datasets)
    
    # Analyze data quality
    analyze_data_quality(final_df, csaf_cols)
    
    # Create visualizations
    create_visualizations(final_df, csaf_coverage, csaf_mask)
    
    # Final summary
    print("\n🎯 FINAL SUMMARY")
    print("=" * 30)
    print(f"✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Final dataset: {len(final_df):,} rows × {len(final_df.columns)} columns")
    print(f"🔍 CSAF columns identified: {len(csaf_cols)}")
    
    if csaf_coverage:
        avg_coverage = np.mean([data['coverage_pct'] for data in csaf_coverage.values()])
        print(f"📈 Average CSAF coverage: {avg_coverage:.1f}%")
    
    if comparison_results:
        successful_merges = sum(1 for r in comparison_results if r['merge_success'])
        print(f"✅ Successful CVE merges: {successful_merges}/{len(comparison_results)}")
    
    print("\n💡 Check the generated visualization for detailed insights!")

if __name__ == "__main__":
    main() 