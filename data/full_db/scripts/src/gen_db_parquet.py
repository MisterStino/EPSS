import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from t3_spark.session import get_spark_session
from data.epss.scripts.src.gen_epss_ts_parq import create_big_parquet
from data.epss.scripts.utils import decompress_all_files_concurrently
from data.epss.scripts.src.get_epss_data import get_all_epss_data
# from data.full_db.scripts.src.gen_db import generate_full_database
from data.full_db.scripts.src.gen_db_parquet import generate_full_database_parquet
from data.epss.scripts.src.handle_missing import fill_missing_dates_and_interpolate
import os
import logging
def cast_common_columns(df):
    """
    Ensures that cve is a string, date is a Spark DateType,
    and epss is a DoubleType. Only applies casting if the column exists.
    Returns a new DataFrame with updated dtypes.
    """
    if 'cve' in df.columns:
        df = df.withColumn('cve', F.col('cve').cast(T.StringType()))
    if 'date' in df.columns:
        # Use DateType() if you only have YYYY-MM-DD. 
        # If you have timestamps, consider TimestampType().
        df = df.withColumn('date', F.col('date').cast(T.DateType()))
    if 'epss' in df.columns:
        df = df.withColumn('epss', F.col('epss').cast(T.DoubleType()))
    return df

def generate_full_database_parquet(modules=['epss'], download_epss=True):
    """
    Generates the final full dataset for training using Spark and Parquet.
    
    - Reads a base module's Parquet file (e.g., epss_processed.parquet).
    - Casts cve/date/epss to correct data types, if those columns exist.
    - Iterates over additional modules, merging them (left join) on (cve, date).
    - Sorts by (cve, date).
    - Writes out the final dataset as a Parquet dataset to:
          data/full_db/processed/final_full_data_parquet
    """

    # Setup logging with INFO level.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Starting the full DB pipeline...")
    # Initialize Spark using your helper
    spark = get_spark_session()

    if not download_epss:
        # Step 1: Download/fetch raw EPS data.
        raw_eps_folder = os.path.join('data', 'epss', 'raw')
        error_file = "temp_error.json"
        logging.info("Fetching raw EPS data...")
        # This function should download/fetch EPS data and store it in raw_eps_folder.
        get_all_epss_data(raw_folder=raw_eps_folder, error_file=error_file)
        logging.info("Raw EPS data fetched successfully.")

    # Step 2: Process raw EPS data to create a time series.

    logging.info("decompressing all files...")
    decompress_all_files_concurrently()
    
    logging.info("EPS time series created successfully.")
    
    logging.info("Creating big parquet epss file...")
    # this is for epss specifically, sorry bad naming
    create_big_parquet()

    # logging.info("handling missing epss dates...")

    fill_missing_dates_and_interpolate()
    # Step 3: Generate the final full database by merging features.
    logging.info("Generating the final full dataset by merging features...")


    # List all modules here; 'epss' is the base module
    modules = ['epss', 'mock']  # Add more modules like 'reddit', 'twitter', etc. as needed
    base_module = 'epss'
    
    # Build the path to the base module's Parquet file
    base_parquet_path = os.path.join('data', base_module, 'processed', f'{base_module}_processed.parquet')
    if not os.path.exists(base_parquet_path):
        print(f"Error: Base parquet file not found: {base_parquet_path}")
        spark.stop()
        return
    
    # Read the base DataFrame and cast important columns
    base_df = spark.read.parquet(base_parquet_path)
    base_df = cast_common_columns(base_df)
    
    row_count = base_df.count()
    print(f"Loaded base data from {base_parquet_path} with {row_count} rows.")
    
    # Merge additional modules onto the base DataFrame
    for module in modules:
        # Skip the base module since it's already loaded
        if module == base_module:
            continue
        
        # Build the path to the module's Parquet file
        module_parquet_path = os.path.join('data', module, 'processed', f'{module}_processed.parquet')
        if not os.path.exists(module_parquet_path):
            print(f"Warning: Processed parquet for module '{module}' not found at {module_parquet_path}. Skipping.")
            continue
        
        # Read and cast the module DataFrame
        module_df = spark.read.parquet(module_parquet_path)
        module_df = cast_common_columns(module_df)
        
        module_row_count = module_df.count()
        print(f"Loaded {module} data from {module_parquet_path} with {module_row_count} rows.")
        
        # Left join on (cve, date)
        base_df = base_df.join(module_df, on=['cve', 'date', 'epss'], how='left')
        
        # Check the shape after merge
        row_count = base_df.count()
        col_count = len(base_df.columns)
        print(f"After merging {module}, dataset now has {row_count} rows and {col_count} columns.")
    
    # Sort the final DataFrame by (cve, date)
    base_df = base_df.orderBy(['cve', 'date'])
    
    # Define the output directory for the final Parquet dataset
    output_dir = os.path.join('data', 'full_db', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    final_parquet_dir = os.path.join(output_dir, 'final_full_data.parquet')
    
    # Write the final dataset as Parquet (overwriting any existing data)
    base_df.write.mode('overwrite').parquet(final_parquet_dir)
    print(f"Final full dataset (Parquet) has been saved to: {final_parquet_dir}")
    
    # Stop the Spark session when done
    spark.stop()

if __name__ == '__main__':
    generate_full_database_parquet()
