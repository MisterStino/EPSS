#!/usr/bin/env python
"""
Comprehensive analysis of production EPSS dataset
Analyzes all columns for feature engineering decisions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from t3_spark.session import get_spark_session
import warnings
warnings.filterwarnings('ignore')

def analyze_column(df, col_name, sample_size=10000):
    """Comprehensive analysis of a single column"""
    
    # Basic info
    col_data = df[col_name]
    total_count = len(col_data)
    
    # Sample for detailed analysis if dataset is large
    if total_count > sample_size:
        sample_data = col_data.sample(sample_size, random_state=42)
    else:
        sample_data = col_data
    
    analysis = {
        'column': col_name,
        'dtype': str(col_data.dtype),
        'total_count': total_count,
        'non_null_count': col_data.notna().sum(),
        'null_count': col_data.isna().sum(),
        'null_percentage': (col_data.isna().sum() / total_count * 100),
        'unique_count': col_data.nunique(),
        'memory_usage_mb': col_data.memory_usage(deep=True) / 1024 / 1024,
    }
    
    # Type-specific analysis
    if col_data.dtype == 'object' or col_data.dtype.name == 'string':
        # String/Object analysis
        non_null_data = sample_data.dropna()
        if len(non_null_data) > 0:
            analysis.update({
                'sample_values': non_null_data.head(5).tolist(),
                'value_lengths': [len(str(x)) for x in non_null_data.head(10)],
                'avg_length': np.mean([len(str(x)) for x in non_null_data]),
                'max_length': max([len(str(x)) for x in non_null_data]),
                'contains_json': any(str(x).strip().startswith(('{', '[')) for x in non_null_data.head(100)),
                'top_values': non_null_data.value_counts().head(5).to_dict()
            })
    
    elif np.issubdtype(col_data.dtype, np.number):
        # Numeric analysis
        non_null_data = sample_data.dropna()
        if len(non_null_data) > 0:
            analysis.update({
                'min_value': non_null_data.min(),
                'max_value': non_null_data.max(),
                'mean_value': non_null_data.mean(),
                'median_value': non_null_data.median(),
                'std_value': non_null_data.std(),
                'zero_count': (non_null_data == 0).sum(),
                'negative_count': (non_null_data < 0).sum(),
                'sample_values': non_null_data.head(10).tolist(),
                'value_range': non_null_data.max() - non_null_data.min()
            })
    
    elif col_data.dtype == 'bool':
        # Boolean analysis
        analysis.update({
            'true_count': col_data.sum(),
            'false_count': (col_data == False).sum(),
            'true_percentage': (col_data.sum() / total_count * 100)
        })
    
    # Date/time detection
    if 'date' in col_name.lower() or 'time' in col_name.lower():
        try:
            date_col = pd.to_datetime(sample_data, errors='coerce')
            analysis.update({
                'is_datetime': True,
                'datetime_null_count': date_col.isna().sum(),
                'date_range': [date_col.min(), date_col.max()] if date_col.notna().any() else None
            })
        except:
            analysis['is_datetime'] = False
    
    return analysis

def categorize_columns(analyses):
    """Categorize columns based on analysis"""
    
    categories = {
        'large_text': [],           # Large text fields to drop
        'metadata': [],             # Metadata/provenance to drop
        'timestamps_safe': [],      # Timestamps that are safe (never future)
        'timestamps_leaky': [],     # Timestamps that could leak future info
        'boolean_flags': [],        # Boolean indicator columns
        'categorical': [],          # Categorical variables for embedding
        'numeric_features': [],     # Numeric features
        'id_columns': [],          # ID/key columns
        'json_fields': [],         # JSON/structured data
        'low_variance': [],        # Low variance columns
        'high_null': []            # High missing value columns
    }
    
    for analysis in analyses:
        col = analysis['column']
        
        # Large text fields (>1000 avg characters or contains detailed descriptions)
        if (analysis.get('avg_length', 0) > 1000 or 
            any(keyword in col.lower() for keyword in ['description', 'details', 'combined', 'merged', 'all'])):
            categories['large_text'].append(col)
        
        # Metadata/provenance fields
        elif any(keyword in col.lower() for keyword in ['timestamp', 'reconstruction', 'key', 'original']):
            categories['metadata'].append(col)
        
        # JSON fields
        elif analysis.get('contains_json', False):
            categories['json_fields'].append(col)
        
        # High null percentage (>80%)
        elif analysis['null_percentage'] > 80:
            categories['high_null'].append(col)
        
        # Low variance (single value >95% of time)
        elif analysis['unique_count'] == 1 or (analysis['unique_count'] < 5 and analysis['total_count'] > 1000):
            categories['low_variance'].append(col)
        
        # Timestamps
        elif 'date' in col.lower() or 'time' in col.lower() or analysis.get('is_datetime', False):
            # Safe timestamps (published, created, etc. - never in future)
            if any(keyword in col.lower() for keyword in ['publish', 'create', 'discover', 'report']):
                categories['timestamps_safe'].append(col)
            # Potentially leaky timestamps (modified, updated, snapshot, etc.)
            else:
                categories['timestamps_leaky'].append(col)
        
        # Boolean flags
        elif (analysis['dtype'] == 'bool' or 
              col.startswith(('has_', 'is_')) or
              analysis['unique_count'] == 2):
            categories['boolean_flags'].append(col)
        
        # Categorical (string/object with reasonable cardinality)
        elif (analysis['dtype'] in ['object', 'string'] and 
              2 < analysis['unique_count'] < 1000 and
              analysis['null_percentage'] < 50):
            categories['categorical'].append(col)
        
        # Numeric features
        elif analysis['dtype'] in ['int32', 'int64', 'float32', 'float64'] or 'int' in analysis['dtype'] or 'float' in analysis['dtype']:
            categories['numeric_features'].append(col)
        
        # ID columns
        elif ('id' in col.lower() or 'cve' in col.lower()) and analysis['unique_count'] > analysis['total_count'] * 0.8:
            categories['id_columns'].append(col)
    
    return categories

def main():
    # Initialize Spark
    spark = get_spark_session()
    
    # Load dataset
    dataset_path = "data/full_db/processed/final_full_data.parquet"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    print(f"🔍 ANALYZING PRODUCTION DATASET: {dataset_path}")
    print("=" * 80)
    
    # Load with Spark for efficiency
    df_spark = spark.read.parquet(dataset_path)
    total_rows = df_spark.count()
    total_cols = len(df_spark.columns)
    
    print(f"📊 Dataset shape: {total_rows:,} rows × {total_cols} columns")
    
    # Convert to pandas for detailed analysis (sample if too large)
    sample_size = min(100000, total_rows)
    if total_rows > sample_size:
        df = df_spark.sample(sample_size / total_rows, seed=42).toPandas()
        print(f"📝 Analyzing sample: {len(df):,} rows for detailed statistics")
    else:
        df = df_spark.toPandas()
        print(f"📝 Analyzing full dataset: {len(df):,} rows")
    
    print("\n" + "="*80)
    print("DETAILED COLUMN ANALYSIS")
    print("="*80)
    
    # Analyze each column
    analyses = []
    for col in df.columns:
        print(f"\n🔍 Analyzing: {col}")
        analysis = analyze_column(df, col, sample_size=10000)
        analyses.append(analysis)
        
        # Print key info
        print(f"  Type: {analysis['dtype']}")
        print(f"  Missing: {analysis['null_percentage']:.1f}%")
        print(f"  Unique values: {analysis['unique_count']:,}")
        print(f"  Memory: {analysis['memory_usage_mb']:.2f} MB")
        
        if 'sample_values' in analysis:
            print(f"  Sample values: {analysis['sample_values']}")
        
        if 'avg_length' in analysis:
            print(f"  Avg length: {analysis['avg_length']:.1f} chars")
            print(f"  Max length: {analysis['max_length']} chars")
        
        if 'min_value' in analysis:
            print(f"  Range: [{analysis['min_value']:.3f}, {analysis['max_value']:.3f}]")
        
        if 'true_percentage' in analysis:
            print(f"  True values: {analysis['true_percentage']:.1f}%")
    
    # Categorize columns
    print("\n" + "="*80)
    print("COLUMN CATEGORIZATION")
    print("="*80)
    
    categories = categorize_columns(analyses)
    
    for category, columns in categories.items():
        if columns:
            print(f"\n📋 {category.upper().replace('_', ' ')}: ({len(columns)} columns)")
            for col in sorted(columns):
                col_analysis = next(a for a in analyses if a['column'] == col)
                print(f"  • {col} - {col_analysis['null_percentage']:.1f}% null, {col_analysis['unique_count']} unique")
    
    # Generate recommendations
    print("\n" + "="*80)
    print("FEATURE ENGINEERING RECOMMENDATIONS")
    print("="*80)
    
    # Columns to drop
    drop_candidates = (categories['large_text'] + 
                      categories['metadata'] + 
                      categories['json_fields'] + 
                      categories['high_null'] + 
                      categories['low_variance'])
    
    print(f"\n🗑️  RECOMMENDED DROP_COLS ({len(drop_candidates)} columns):")
    print("These columns should be dropped due to:")
    print("- Large text content (>1000 chars avg)")
    print("- Metadata/provenance information")
    print("- JSON/structured data requiring special parsing")
    print("- High missing values (>80%)")
    print("- Low variance (essentially constant)")
    print()
    
    for col in sorted(drop_candidates):
        col_analysis = next(a for a in analyses if a['column'] == col)
        reason = []
        if col in categories['large_text']:
            reason.append("large_text")
        if col in categories['metadata']:
            reason.append("metadata")
        if col in categories['json_fields']:
            reason.append("json")
        if col in categories['high_null']:
            reason.append(f"{col_analysis['null_percentage']:.0f}%_null")
        if col in categories['low_variance']:
            reason.append("low_variance")
        
        print(f"  '{col}',  # {', '.join(reason)}")
    
    print(f"\n📅 TS_SAFE ({len(categories['timestamps_safe'])} columns):")
    print("Safe timestamps (never in future):")
    for col in sorted(categories['timestamps_safe']):
        print(f"  '{col}',")
    
    print(f"\n⚠️  TS_LEAKY ({len(categories['timestamps_leaky'])} columns):")
    print("Potentially leaky timestamps (could be in future):")
    for col in sorted(categories['timestamps_leaky']):
        print(f"  '{col}',")
    
    print(f"\n🔵 BOOL_COLS ({len(categories['boolean_flags'])} columns):")
    print("Boolean flag columns:")
    bool_expr = "lambda df: ["
    for i, col in enumerate(sorted(categories['boolean_flags'])):
        if i == 0:
            bool_expr += f"'{col}'"
        else:
            bool_expr += f", '{col}'"
    
    # Also add pattern-based detection
    print("# Pattern-based detection + explicit columns")
    print("BOOL_COLS = lambda df: [c for c in df.columns")
    print("                       if c.startswith(('has_', 'is_'))")
    print("                       or c in (")
    for col in sorted(categories['boolean_flags']):
        print(f"                           '{col}',")
    print("                       )]")
    
    print(f"\n🏷️  CAT_COLS ({len(categories['categorical'])} columns):")
    print("Categorical columns for embedding:")
    for col in sorted(categories['categorical']):
        col_analysis = next(a for a in analyses if a['column'] == col)
        print(f"  '{col}',  # {col_analysis['unique_count']} categories")
    
    print(f"\n🔢 NUMERIC_FEATURES ({len(categories['numeric_features'])} columns):")
    print("Numeric features (will be auto-detected):")
    for col in sorted(categories['numeric_features'][:10]):  # Show first 10
        col_analysis = next(a for a in analyses if a['column'] == col)
        if 'min_value' in col_analysis:
            print(f"  {col}: [{col_analysis['min_value']:.3f}, {col_analysis['max_value']:.3f}]")
    if len(categories['numeric_features']) > 10:
        print(f"  ... and {len(categories['numeric_features']) - 10} more")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_features = (len(categories['timestamps_safe']) + 
                     len(categories['timestamps_leaky']) + 
                     len(categories['boolean_flags']) + 
                     len(categories['categorical']) + 
                     len(categories['numeric_features']))
    
    print(f"📊 Total columns in dataset: {total_cols}")
    print(f"🗑️  Recommended to drop: {len(drop_candidates)} ({len(drop_candidates)/total_cols*100:.1f}%)")
    print(f"✅ Useful features: {total_features} ({total_features/total_cols*100:.1f}%)")
    print(f"🔍 Remaining columns: {total_cols - len(drop_candidates) - total_features}")
    
    # Save detailed analysis
    import json
    with open("column_analysis_detailed.json", "w") as f:
        json.dump(analyses, f, indent=2, default=str)
    
    print(f"\n💾 Detailed analysis saved to: column_analysis_detailed.json")
    print("🎉 Analysis complete!")

if __name__ == "__main__":
    main() 