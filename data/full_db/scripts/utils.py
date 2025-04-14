import os
import logging
import concurrent.futures
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
from t3_spark.session import get_spark_session

def cast_common_columns(df):
    """
    Ensure (cve, date, epss) have correct data types in Spark.
    """
    if 'cve' in df.columns:
        df = df.withColumn('cve', F.col('cve').cast(T.StringType()))
    if 'date' in df.columns:
        df = df.withColumn('date', F.col('date').cast(T.DateType()))
    if 'epss' in df.columns:
        df = df.withColumn('epss', F.col('epss').cast(T.DoubleType()))
    return df


# Setup logging with INFO level.
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def delete_file(file_path):
    """
    Deletes a single file and logs the operation.
    
    Parameters:
        file_path (str): The full path to the file to be deleted.
    """
    try:
        logging.info(f"Deleting file: {file_path}")
        os.remove(file_path)
    except Exception as e:
        logging.error(f"Failed to delete file: {file_path}. Error: {e}")

def clean_directory_concurrent(dir_path):
    """
    Deletes all files in the specified directory concurrently.
    Logs each step of the process.
    
    Parameters:
        dir_path (str): The path to the directory to be cleaned.
    
    Steps:
    - Confirm the provided path is a directory.
    - List all items in the directory.
    - Separate files from non-files (log non-file items as skipped).
    - Use a ThreadPoolExecutor to delete files concurrently.
    - Log any errors that occur during deletion.
    - Log when the directory cleaning is complete.
    """
    # Check if the provided path is a directory.
    if not os.path.isdir(dir_path):
        logging.info(f"Directory '{dir_path}' does not exist. No action taken.")
        return

    # Get the list of items in the directory.
    items = os.listdir(dir_path)
    
    # If the directory is empty, log and return.
    if not items:
        logging.info(f"Directory '{dir_path}' is already empty. Nothing to delete.")
        return

    # Prepare lists for files and non-file items.
    files_to_delete = []
    for item in items:
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            files_to_delete.append(item_path)
        else:
            logging.info(f"Skipped non-file item: {item_path}")

    # Determine the number of workers (threads) to use.
    max_workers = min(10, len(files_to_delete)) if files_to_delete else 1

    # Use ThreadPoolExecutor to delete files concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(delete_file, file_path) for file_path in files_to_delete]
        # Wait for all deletion tasks to complete.
        concurrent.futures.wait(futures)
    
    logging.info(f"Directory '{dir_path}' cleaning complete.")





def filter_epss_dates(input_parquet: str, output_parquet: str, cutoff_date: str = "2022-02-04"):
    """
    Reads the EPS data with the 'age_epss_pub' feature, removes all rows with dates before the cutoff date,
    and saves the cleaned DataFrame to the specified output location.
    
    Parameters:
        input_parquet (str): Path to the input Parquet file (e.g., 'data/epss/epss_parquet/epss_pub_features.parquet').
        output_parquet (str): Path where the output Parquet file will be saved (e.g., 'data/epss/processed/epss_processed.parquet').
        cutoff_date (str): The date (in 'YYYY-MM-DD' format) before which data should be removed. Default is '2022-02-04'.
    """
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Starting filter_epss_dates function.")

    # Initialize Spark session using the provided helper
    spark = get_spark_session()

    # Read the input Parquet file
    df = spark.read.parquet(input_parquet)
    logging.info(f"Loaded data from {input_parquet} with {df.count()} rows.")

    # Ensure that columns are cast correctly
    df = cast_common_columns(df)

    # Filter out rows with dates before the cutoff date
    df_filtered = df.filter(F.col("date") >= F.lit(cutoff_date))
    logging.info(f"Filtered data contains {df_filtered.count()} rows after removing dates before {cutoff_date}.")

    # Ensure that the output directory exists
    output_dir = os.path.dirname(output_parquet)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory {output_dir} is ready.")

    # Save the filtered DataFrame as a Parquet file in overwrite mode
    df_filtered.write.mode("overwrite").parquet(output_parquet)
    logging.info(f"Cleaned data saved to {output_parquet}")

    # Stop the Spark session
    spark.stop()

if __name__ == '__main__':
    filter_epss_dates(
        input_parquet="data/epss/epss_parquet/epss_pub_features.parquet",
        output_parquet="data/epss/processed/epss_processed.parquet",
        cutoff_date="2022-02-04"
    )

# Example usage:
if __name__ == "__main__":
    directory_path = "path/to/your/directory"
    clean_directory_concurrent(directory_path)
