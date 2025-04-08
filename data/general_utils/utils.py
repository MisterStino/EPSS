import os
import time
import shutil
from pyspark.sql import functions as F
from pyspark.sql import types as T
from t3_spark.session import get_spark_session

def cast_common_columns(df):
    if 'cve' in df.columns:
        df = df.withColumn('cve', F.col('cve').cast(T.StringType()))
    if 'date' in df.columns:
        df = df.withColumn('date', F.col('date').cast(T.DateType()))
    if 'epss' in df.columns:
        df = df.withColumn('epss', F.col('epss').cast(T.DoubleType()))
    return df

def summarize_cve_time_ranges(
    interesting_cves_path='data/general_utils/files/high_score_above_0.9_with_initial_below_0.4.parquet',
    full_data_path='data/full_db/processed/final_full_data_parquet',
    output_file='data/general_utils/files/cve_time_ranges.csv'
):
    spark = get_spark_session()
    
    # 1) Read the 'interesting' CVEs
    cves_df = spark.read.parquet(interesting_cves_path).select("cve").distinct()
    
    # 2) Read the full dataset
    full_df = spark.read.parquet(full_data_path)
    full_df = cast_common_columns(full_df)
    
    # 3) Join and compute min/max dates
    joined_df = full_df.join(cves_df, on="cve", how="inner")
    date_range_df = (
        joined_df
        .groupBy("cve")
        .agg(
            F.min("date").alias("start_date"),
            F.max("date").alias("end_date")
        )
    )

    # 4) Save as single-part CSV and rename part file
    temp_dir = output_file + "_temp"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    date_range_df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)

    # Find the actual part file Spark wrote
    part_file = None
    for fname in os.listdir(temp_dir):
        if fname.startswith("part-") and fname.endswith(".csv"):
            part_file = fname
            break

    if part_file is None:
        raise FileNotFoundError("No part file found in Spark output.")

    # Move and rename the part file to the desired output_file
    shutil.move(os.path.join(temp_dir, part_file), output_file)

    # Remove the temporary Spark folder
    shutil.rmtree(temp_dir)

    print(f"CSV file successfully saved to: {output_file}")
    
    time.sleep(2)
    spark.stop()




def parquet_to_csv(input_parquet, output_csv):
    """
    Converts the entire content of a Parquet file to a single CSV file.
    
    Parameters:
      input_parquet (str): Path to the input Parquet file.
      output_csv (str): Desired path for the output CSV file.
    
    The function:
      - Reads the Parquet file using Spark.
      - Coalesces the DataFrame to one partition so that only one CSV part is written.
      - Writes the CSV to a temporary directory.
      - Locates the CSV part file, renames/moves it to output_csv.
      - Deletes the temporary directory.
    """
    
    # Initialize Spark using your helper function.
    spark = get_spark_session()
    
    # Read the entire Parquet file into a Spark DataFrame.
    df = spark.read.parquet(input_parquet)
    
    # Optionally, ensure the correct data types if needed.
    df = (df.withColumn("cve", F.col("cve").cast(T.StringType()))
            .withColumn("date", F.col("date").cast(T.DateType()))
            .withColumn("epss", F.col("epss").cast(T.DoubleType()))
         )
    
    # Define a temporary output directory for the CSV
    temp_dir = output_csv + "_temp"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Write the DataFrame as a CSV with a single partition.
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)
    
    # Find the CSV part file in the temporary directory.
    part_file = None
    for fname in os.listdir(temp_dir):
        if fname.startswith("part-") and fname.endswith(".csv"):
            part_file = fname
            break
    
    if part_file is None:
        raise FileNotFoundError("No CSV part file found in the temporary directory.")
    
    # Move and rename the part file to the final output CSV path.
    final_csv_path = os.path.join(os.path.dirname(output_csv), os.path.basename(output_csv))
    shutil.move(os.path.join(temp_dir, part_file), final_csv_path)
    
    # Remove the temporary directory.
    shutil.rmtree(temp_dir)
    
    print(f"CSV file successfully saved to: {final_csv_path}")
    
    time.sleep(2)
    spark.stop()
    
    return final_csv_path



def store_all_unique_cves_in_csv(
    input_parquet="data/full_db/processed/final_full_data_parquet",
    output_csv="data/general_utils/files/all_unique_cves.csv"
):
    """
    Reads the full dataset (Parquet) and extracts all unique CVE IDs.
    Writes them to a single CSV file (output_csv), 
    and prints the number of unique CVEs found.
    
    Steps:
      1) Read input_parquet with Spark.
      2) Select 'cve' column, distinct -> get unique CVEs.
      3) Count how many unique CVEs -> print.
      4) coalesce(1) => single CSV part file in a temp folder.
      5) rename the part file to output_csv.
      6) remove temp folder
    """

    # 1) Spark session
    spark = get_spark_session()
    
    # Read the dataset and cast columns if needed
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)

    # 2) Select only cve, distinct
    unique_cves_df = df.select("cve").distinct()

    # 3) Count how many unique CVEs
    count_unique = unique_cves_df.count()
    print(f"Number of unique CVEs in {input_parquet}: {count_unique}")

    # 4) Write to single CSV file
    temp_dir = output_csv + "_temp"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    (
        unique_cves_df
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(temp_dir)
    )
    
    # Find the part file
    part_file = None
    for fname in os.listdir(temp_dir):
        if fname.startswith("part-") and fname.endswith(".csv"):
            part_file = fname
            break
    
    if part_file is None:
        raise FileNotFoundError("No part file found in the temporary directory.")
    
    # Move and rename the part file to the final output_csv
    final_csv_path = os.path.join(os.path.dirname(output_csv), os.path.basename(output_csv))
    shutil.move(os.path.join(temp_dir, part_file), final_csv_path)
    
    # Clean up temporary folder
    shutil.rmtree(temp_dir)
    
    print(f"Unique CVEs CSV file successfully saved to: {final_csv_path}")
    
    time.sleep(2)
    spark.stop()




# Example usage:
if __name__ == "__main__":
    # input_parquet = "data/general_utils/files/high_score_above_0.9_with_initial_below_0.4.parquet"
    # output_csv = "data/general_utils/files/init_cve_feat_subset.csv"
    # parquet_to_csv(input_parquet, output_csv)
    # summarize_cve_time_ranges(interesting_cves_path=input_parquet,output_file=f"{output_csv}_time_ranges.csv")

    store_all_unique_cves_in_csv()


