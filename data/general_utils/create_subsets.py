import os
from pyspark.sql import functions as F
from pyspark.sql import types as T
import time
from pyspark.sql import Window
from t3_spark.session import get_spark_session

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
        df = df.withColumn('date', F.col('date').cast(T.DateType()))
    if 'epss' in df.columns:
        df = df.withColumn('epss', F.col('epss').cast(T.DoubleType()))
    return df

def create_high_score_newer_cves_subset():
    """
    Creates a subset of the full dataset that contains only those CVEs that:
      1. Have at least one row where epss > 0.9, and
      2. Have a CVE id year (extracted from the cve string) of 2022 or later.
    
    The resulting subset is saved as a Parquet file in:
         data/general_utils/files/high_score_newer_cves.parquet
    """
    # Initialize Spark session using your helper.
    spark = get_spark_session()
    
    # Define the path to the full final dataset (Parquet)
    full_data_path = os.path.join('data', 'full_db', 'processed', 'final_full_data_parquet')
    
    # Read the full dataset.
    full_df = spark.read.parquet(full_data_path)
    full_df = cast_common_columns(full_df)
    
    # Extract the year from the CVE id.
    # Assumes the CVE id is in the format: "CVE-YYYY-XXXX" (e.g., "CVE-2022-22536")
    full_df = full_df.withColumn("year", F.split(F.col("cve"), "-").getItem(1).cast(T.IntegerType()))
    
    # Identify CVEs that have at least one epss score > 0.9.
    high_score_cves = full_df.filter(F.col("epss") > 0.9).select("cve").distinct()
    
    # Join back to the full dataframe to filter only those CVEs that appear in high_score_cves.
    subset_df = full_df.join(high_score_cves, on="cve", how="inner")
    
    # Filter for CVEs with a year >= 2022.
    subset_df = subset_df.filter(F.col("year") >= 2022)
    
    # Optional: Remove the temporary "year" column if not needed.
    subset_df = subset_df.drop("year")
    
    # Define the output path.
    output_dir = os.path.join('data', 'general_utils', 'files')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "high_score_newer_cves.parquet")
    
    # Write the subset as a Parquet file.
    subset_df.write.mode("overwrite").parquet(output_path)
    print(f"Subset Parquet file saved to: {output_path}")
    
    # Stop Spark session when done.
    spark.stop()



import os
import time
from pyspark.sql import functions as F
from t3_spark.session import get_spark_session

def create_high_score_above_x_subset(x, date_range=None, full_data_path='data/full_db/processed/final_full_data.parquet'):
    """
    Creates a subset of the full dataset that contains only those CVEs that have at least
    one row where epss > x.
    
    If a date_range is provided, only rows where the 'date' column is between the given
    start and end dates (inclusive) will be included. If date_range is None, all dates are used.
    
    The resulting subset is saved as a Parquet file in:
         data/general_utils/files/high_score_above_x.parquet
         
    Parameters:
      x: the epss threshold (e.g., 0.98)
      date_range: A tuple of (start_date, end_date) in the format "YYYY-MM-DD", or None.
    """
    # Initialize the Spark session using your helper function.
    spark = get_spark_session()
    # Read the full dataset and cast columns to the proper types.
    full_df = spark.read.parquet(full_data_path)
    full_df = cast_common_columns(full_df)
    
    # If a date range is provided, filter the full dataset accordingly.
    if date_range is not None:
        start_date, end_date = date_range
        full_df = full_df.filter((F.col("date") >= start_date) & (F.col("date") <= end_date))
    
    # Identify CVEs with at least one row where epss > x.
    high_score_cves = full_df.filter(F.col("epss") > x).select("cve").distinct()
    
    # Join the high-score CVE list back to the full DataFrame to get all rows for these CVEs.
    subset_df = full_df.join(high_score_cves, on="cve", how="inner")
    
    # Define the output directory and file path.
    output_dir = os.path.join('data', 'general_utils', 'files')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"high_score_above_{x}.parquet")
    

    # print the number of rows in the subset DataFrame.
    print(f"Number of rows in the subset DataFrame: {subset_df.count()}")
    # Write the subset as a Parquet file (overwrite if it exists).
    subset_df.write.mode("overwrite").parquet(output_path)
    print(f"Subset Parquet file (epss > {x}) saved to: {output_path}")
    
    # Pause briefly before stopping Spark.
    time.sleep(3)
    
    # Stop the Spark session.
    spark.stop()


def create_subset_cve_initial_not_above_and_reaches_x(x, date_range=None):
    """
    Creates a subset of the full dataset for CVEs that satisfy:
      1. Their first (earliest) EPS score in the given date range is not above 0.4.
      2. At least one row in the date range has an EPS score greater than x.
    
    If a date_range is provided, only rows with a 'date' between the provided start and
    end dates (inclusive) are considered. If date_range is None, the entire dataset is used.
    
    The resulting subset (all rows for the selected CVEs) is saved as a Parquet file in:
         data/general_utils/files/high_score_above_{x}_with_initial_below_0.4.parquet
         
    Parameters:
      x: the EPS threshold that must be reached at least once (e.g., 0.98)
      date_range: A tuple (start_date, end_date) in "YYYY-MM-DD" format, or None.
    """
    # Initialize the Spark session using your helper.
    spark = get_spark_session()
    
    # Define the path to the full dataset (Parquet folder)
    full_data_path = os.path.join('data', 'full_db', 'processed', 'final_full_data_parquet')
    
    # Read the full dataset and cast columns to proper types.
    full_df = spark.read.parquet(full_data_path)
    full_df = cast_common_columns(full_df)
    
    # If a date range is provided, filter the dataset accordingly.
    if date_range is not None:
        start_date, end_date = date_range
        full_df = full_df.filter((F.col("date") >= start_date) & (F.col("date") <= end_date))
    
    # Create a window specification partitioned by CVE and ordered by date.
    w = Window.partitionBy("cve").orderBy("date")
    
    # Add a new column 'initial_epss' which is the first epss in the date range for each CVE.
    df_with_initial = full_df.withColumn("initial_epss", F.first("epss").over(w))
    
    # Identify CVEs that have at least one row where epss > x.
    high_score_cves = full_df.filter(F.col("epss") > x).select("cve").distinct()
    
    # Identify CVEs whose first (earliest) EPS score in the date range is not above 0.4.
    valid_initial_cves = df_with_initial.filter(F.col("initial_epss") <= 0.4).select("cve").distinct()
    
    # Intersect the two sets of CVEs: they must both have a low initial score and eventually reach above x.
    selected_cves = high_score_cves.join(valid_initial_cves, on="cve", how="inner")
    
    # Join back to the full dataset (filtered by date, if applicable) to get all rows for these CVEs.
    subset_df = full_df.join(selected_cves, on="cve", how="inner")
    
    # Define the output directory and file path.
    output_dir = os.path.join('data', 'general_utils', 'files')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"high_score_above_{x}_with_initial_below_0.4.parquet")
    
    # Write the subset as a Parquet file (overwrite if exists).
    subset_df.write.mode("overwrite").parquet(output_path)
    print(f"Subset Parquet file saved to: {output_path}")
    
    # Pause briefly to ensure all tasks complete.
    time.sleep(3)
    
    # Stop the Spark session.
    spark.stop()


if __name__ == '__main__':
    # create_subset_cve_initial_not_above_and_reaches_x(0.9, date_range=("2023-03-07", "2025-03-11"))
    create_subset_cve_initial_not_above_and_reaches_x(0.4)
    #create_high_score_newer_cves_subset()
