import os
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
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

def add_pub_cohort_label(df, days_per_cohort=90):
    """
    Groups CVEs into cohorts based on their publication date.
    
    For each CVE, computes its publication date (i.e., the minimum 'date' for that CVE).
    Then, using the global minimum publication date across all CVEs, calculates the
    difference in days between each CVE's pub_date and the global minimum.
    The cohort index is then computed as:
        cohort_index = floor((pub_date - global_min_pub_date) / days_per_cohort) + 1,
    and the cohort label is formed as "C" concatenated with the cohort_index.
    
    Returns a Spark DataFrame with columns: (cve, pub_cohort, pub_date).
    """
    # Compute publication date per CVE
    w = Window.partitionBy("cve")
    df_pub = df.groupBy("cve").agg(F.min("date").alias("pub_date"))
    
    # Determine the global minimum publication date across all CVEs.
    global_min_pub_date = df_pub.agg(F.min("pub_date").alias("global_min")).collect()[0]["global_min"]
    # Convert the global minimum to a Spark date literal.
    global_min_pub_date_lit = F.to_date(F.lit(global_min_pub_date))
    
    # Compute the days from the global minimum to each CVE's publication date.
    df_pub = df_pub.withColumn("days_from_global_min", F.datediff(F.col("pub_date"), global_min_pub_date_lit))
    
    # Compute the cohort index and label.
    df_pub = df_pub.withColumn("cohort_index", F.floor(F.col("days_from_global_min") / days_per_cohort) + 1)
    df_pub = df_pub.withColumn("pub_cohort", F.concat(F.lit("C"), F.col("cohort_index").cast(T.StringType())))
    
    return df_pub.select("cve", "pub_cohort", "pub_date")

# Example usage:
if __name__ == "__main__":
    # Initialize Spark session (customize config as needed)
    spark = get_spark_session()
    
    # Read your dataset (Parquet file) and cast columns correctly.
    input_parquet = "data/full_db/processed/final_full_data_parquet"
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    # Create publication-based cohorts using a 90-day window.
    cohort_df = add_pub_cohort_label(df, days_per_cohort=90)
    
    # Save the mapping as a CSV for reference (if needed)
    output_csv = "data/general_utils/files/pub_cohorts.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    cohort_df.coalesce(1).write.mode("overwrite").option("header", True).csv(output_csv + "_temp")
    # (Add your code to rename/move the CSV from the temporary directory as needed.)
    
    print("Publication-based cohort assignment complete.")
    spark.stop()
