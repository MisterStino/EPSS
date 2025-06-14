# 1. To get a mininmal case working I want to cut the columns down to a minimal set. + cut time (range)

# ==== CONFIGURABLE TIME RANGE ====
START_DATE = "2023-03-20"  # Start date for time filtering (YYYY-MM-DD)
END_DATE = "2024-04-08"    # End date for time filtering (YYYY-MM-DD)
TIME_COLUMN = "date"       # Specific time column to use for filtering

from t3_spark.session import get_spark_session
from pyspark.sql import functions as F
from pyspark.sql.types import *
import os

def create_minimal_dataset():
    """
    Create minimal v1 time series dataset by selecting key features while preserving
    composite key (CVE, date) and target variable for EPSS forecasting.
    """
    # Initialize Spark session using battle-tested method
    spark = get_spark_session()
    
    # Define composite key columns (MUST be preserved for time series)
    composite_key_columns = [
        'cve',                   # CVE identifier - required for time series entity
        'date'                   # Date - required for temporal ordering
    ]
    
    # Define target variable for forecasting
    target_columns = [
        'epss'                   # Target variable for EPSS forecasting
    ]
    
    # Define the 7 key feature columns for minimal case
    feature_columns = [
        'age_epss_pub',          # numeric - EPSS-features
        'primary_cvss_score',    # numeric - NVD
        'days_since_last_event', # numeric - CSAF
        'has_threat',            # boolean - CSAF
        'is_windows',            # boolean - NVD
        'primary_cvss_sev',      # categorical - NVD
        'dominant_event_type'    # categorical - CSAF
    ]
    
    # Combine all required columns (keys + target + features)
    all_required_columns = composite_key_columns + target_columns + feature_columns
    
    print("Loading full dataset...")
    # Load the big dataset
    input_path = "data/full_db/processed/final_full_data.parquet"
    df = spark.read.parquet(input_path)
    
    print(f"Original dataset shape: {df.count()} rows, {len(df.columns)} columns")
    print("Original columns:", df.columns)
    
    # Check which time columns are available for filtering
    time_columns = [col for col in df.columns if any(time_word in col.lower() 
                   for time_word in ['date', 'time', 'published', 'created', 'modified'])]
    print(f"Available time columns: {time_columns}")
    
    # Validate that all required columns exist
    print(f"Checking for required columns...")
    missing_cols = [col for col in all_required_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {df.columns}")
        return None
    
    print(f"✅ All required columns found")
    print(f"Composite keys: {composite_key_columns}")
    print(f"Target variable: {target_columns}")
    print(f"Feature columns: {feature_columns}")
    
    # Select all required columns
    df_selected = df.select(*all_required_columns)
    
    print(f"After column selection: {df_selected.count()} rows, {len(df_selected.columns)} columns")
    
    # Apply time filtering using the date column
    print(f"Applying time filtering using column: {TIME_COLUMN}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    
    # Get date range info from actual data
    date_stats = df_selected.select(
        F.min(TIME_COLUMN).alias('min_date'),
        F.max(TIME_COLUMN).alias('max_date')
    ).collect()[0]
    
    print(f"Actual data date range: {date_stats['min_date']} to {date_stats['max_date']}")
    
    # Apply the specific date range filter
    df_filtered = df_selected.filter(
        (F.col(TIME_COLUMN) >= F.lit(START_DATE)) & 
        (F.col(TIME_COLUMN) <= F.lit(END_DATE))
    )
    
    filtered_count = df_filtered.count()
    print(f"After time filtering ({START_DATE} to {END_DATE}): {filtered_count} rows")
    
    # Validate composite key uniqueness (critical for time series)
    print("Validating composite key uniqueness...")
    unique_combinations = df_filtered.select('cve', 'date').distinct().count()
    if unique_combinations == filtered_count:
        print(f"✅ Composite key (CVE, date) is unique: {unique_combinations} unique combinations")
    else:
        print(f"⚠️  WARNING: Composite key not unique! {filtered_count} rows vs {unique_combinations} unique combinations")
        print("This indicates duplicate (CVE, date) pairs which breaks time series structure")
    
    # Show sample of the data with proper column ordering
    print("\nSample of the minimal time series dataset:")
    # Reorder columns: keys first, then target, then features
    column_order = composite_key_columns + target_columns + feature_columns
    df_final = df_filtered.select(*column_order)
    df_final.show(5, truncate=False)
    
    # Show time series structure for one CVE
    print("\nExample time series for one CVE:")
    sample_cve = df_final.select('cve').distinct().limit(1).collect()[0]['cve']
    df_final.filter(F.col('cve') == sample_cve).orderBy('date').show(10, truncate=False)
    
    # Create output directory if it doesn't exist
    output_dir = "data/full_db/v1/data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the minimal dataset
    output_path = f"{output_dir}/minimal_v1_timeseries.parquet"
    print(f"Saving minimal time series dataset to: {output_path}")
    
    df_final.write.mode("overwrite").parquet(output_path)
    
    print(f"✅ Minimal v1 time series dataset created successfully!")
    print(f"Final dataset: {filtered_count} rows, {len(df_final.columns)} columns")
    print(f"Columns: {df_final.columns}")
    print(f"Saved to: {output_path}")
    
    # Show data types and basic stats
    print("\nData types:")
    df_final.printSchema()
    
    # Show basic statistics for numeric columns
    print("\nBasic statistics:")
    df_final.describe().show()
    
    return df_final

if __name__ == "__main__":
    create_minimal_dataset()


