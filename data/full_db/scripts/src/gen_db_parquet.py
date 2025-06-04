import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from t3_spark.session import get_spark_session
from data.epss.scripts.src.gen_epss_ts_parq import create_epss_long_table
from data.epss.scripts.utils import decompress_all_files_concurrently, standardize_epss_files_concurrently
from data.epss.scripts.src.get_epss_data import get_all_epss_data
from data.epss.scripts.src.handle_missing import fill_missing_dates_and_forward_fill
import os
import logging
from data.full_db.scripts.utils import clean_directory_concurrent, filter_epss_dates
from data.full_db.quick_inspection.check_merge import quick_viz
from data.epss_features.scripts.age_epss_pub import create_epss_pub 

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

def generate_full_database_parquet(modules=['epss'], download_epss=False):
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

    # if download_epss:
    #     # Step 1: Download/fetch raw EPS data.
    #     raw_eps_folder = os.path.join('data', 'epss', 'raw')
    #     error_file = "temp_error.json"
    #     logging.info("Fetching raw EPS data...")
    #     # This function should download/fetch EPS data and store it in raw_eps_folder.
    #     get_all_epss_data(raw_folder=raw_eps_folder, error_file=error_file)
    #     logging.info("Raw EPS data fetched successfully.")

    # # Step 2: Process raw EPS data to create a time series.

    # logging.info("decompressing all files...")
    # # make sure the directory is cleaned before decompressing
    # clean_directory_concurrent('data/epss/uncompressed')
    # # actually decompress the files
    # decompress_all_files_concurrently(raw_folder = 'data/epss/raw', output_folder = 'data/epss/uncompressed')

    # # standardize the csv files in the uncompressed folder
    # clean_directory_concurrent('data/epss/standardized')
    # standardize_epss_files_concurrently('data/epss/uncompressed', 'data/epss/standardized')
        
    # logging.info("Creating from csv files a long format parquet epss file...")
    # # # remove all files from data/epss/epss_parquet before creating the new parquet file
    # clean_directory_concurrent('data/epss/epss_parquet')
    # # convert all daily csvs to a single parquet file in long format
    # create_epss_long_table(input_folder = 'data/epss/standardized', output_folder = 'data/epss/epss_parquet', output_parquet = 'epss_all.parquet')

    # logging.info("handling missing epss dates...")
    #delelte all files in data/epss/processed before creating the new parquet file
    # clean_directory_concurrent('data/epss/epss_parquet/epss_interpolated.parquet')
    # # fill in / add missing dates: fill epss with null and interpolate epss values
    # logging.info("Filling missing dates with null (from files not downloaded) and forward filling epss values...")
    # fill_missing_dates_and_forward_fill(input_parquet  = "data/epss/epss_parquet/epss_all.parquet", output_parquet = "data/epss/epss_parquet/epss_interpolated.parquet")

    # # Step 3: create epss age since epss pub release features
    # logging.info("Creating age since epss publication feature...")
    # create_epss_pub(input_parquet = "data/epss/epss_parquet/epss_interpolated.parquet", output_parquet = "data/epss_features/raw/epss_pub_features.parquet")

    
    # # step 4: Delete epss information before release of epss v2 for both the epss and the features parquet files
    # logging.info("Deleting epss information before release of epss v2...")

    # filter_epss_dates(
    #     input_parquet="data/epss/epss_parquet/epss_interpolated.parquet",
    #     output_parquet="data/epss/processed/epss_processed.parquet",
    #     cutoff_date="2022-02-04"
    # )
    # filter_epss_dates(
    #     input_parquet="data/epss_features/raw/epss_pub_features.parquet",
    #     output_parquet="data/epss_features/processed/epss_features_processed.parquet",
    #     cutoff_date="2022-02-04"
    # )
    # # Step 5: Generate the final full database by merging features.
    # logging.info("Generating the final full dataset by merging features...")

    # #### Here we start merging all the features together. ####
    # logging.warning("Starting to merge features...")
    
    # # just making sure we have spark session after long run..
    # spark = get_spark_session()
    # List all modules here; 'epss' is the base module
    modules = ['epss', 'epss_features', 'github']  # Add more modules like 'reddit', 'github', etc. 
    base_module = 'epss'
    
    # Build the path to the base module's Parquet file
    base_parquet_path = os.path.join('data', base_module, 'processed', f'{base_module}_processed.parquet')
    if not os.path.exists(base_parquet_path):
        print(f"Error: Base parquet file not found: {base_parquet_path}")
        spark.stop()
        return
    
    # Read the base DataFrame and cast important columns
    logging.info(f"Loading base data from {base_parquet_path}...")
    base_df = spark.read.parquet(base_parquet_path)
    base_df = cast_common_columns(base_df)
    logging.info(f"Base data loaded with {base_df.count()} rows. And head:")
    logging.info(base_df.show(10))

    row_count = base_df.count()

    # Merge additional modules onto the base DataFrame
    for module in modules:
        # Skip the base module since it's already loaded  
        if module == base_module:
            continue
        logging.info(f"Merging module: {module}...")
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
        # Drop 'epss' column from the module DataFrame so we only join on 'cve' and 'date'
        if 'epss' in module_df.columns:
            module_df = module_df.drop('epss')
        # Left join on (cve, date)
        base_df = base_df.join(module_df, on=['cve', 'date'], how='left')
        
        # Check the shape after merge
        row_count = base_df.count()
        col_count = len(base_df.columns)
        print(f"After merging {module}, dataset now has {row_count} rows and {col_count} columns.")
    



    logging.info(f"Saving the final merged dataset...")
    # Sort the final DataFrame by (cve, date)
    base_df = base_df.orderBy(['cve', 'date'])
    
    # Define the output directory for the final Parquet dataset
    output_dir = os.path.join('data', 'full_db', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    final_parquet_dir = os.path.join(output_dir, 'final_full_data.parquet')
    
    # Write the final dataset as Parquet (overwriting any existing data)
    base_df.write.mode('overwrite').parquet(final_parquet_dir)
    print(f"Final full dataset (Parquet) has been saved to: {final_parquet_dir}")
    
    # Sanity check for comforts
    modules_features = ['epss', 'age_epss_pub', 'github']
    # Columns (from the additional modules) which are numeric and should be normalized & plotted.
    numeric_features = ['age_epss_pub', 'commit_count']  # You can add more column names if needed.
    quick_viz(modules_features, numeric_features)
    # Stop the Spark session when done
    spark.stop()

if __name__ == '__main__':
    generate_full_database_parquet()
