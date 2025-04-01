import os
from pyspark.sql import functions as F
from pyspark.sql import types as T
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

if __name__ == '__main__':
    create_high_score_newer_cves_subset()
