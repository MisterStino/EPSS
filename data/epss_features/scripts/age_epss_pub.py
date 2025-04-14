import os
import logging
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
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

def create_epss_pub(input_parquet: str, output_parquet: str):
    """
    Reads the EPS interpolated data, computes the 'age_epss_pub' feature for each CVE 
    (defined as the number of days elapsed since the first observed date for that CVE),
    and saves the resulting DataFrame as a Parquet file.

    Parameters:
        input_parquet (str): Path to the input Parquet file containing the columns (cve, date, epss).
        output_parquet (str): Path where the output Parquet file with the added feature will be saved.
    """
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Starting create_epss_pub function.")

    # Initialize Spark Session
    spark = get_spark_session()

    # Read the input Parquet file
    df = spark.read.parquet(input_parquet)
    logging.info(f"Input data loaded from {input_parquet} with {df.count()} rows.")

    # Ensure correct data types using the helper function
    df = cast_common_columns(df)

    # Define a window specification partitioned by cve
    windowSpec = Window.partitionBy("cve")
    
    # Compute the earliest date for each cve and add it as a temporary column 'first_date'
    df = df.withColumn("first_date", F.min("date").over(windowSpec))
    
    # Compute the age_epss_pub as the number of days difference between current date and the first_date
    df = df.withColumn("age_epss_pub", F.datediff(F.col("date"), F.col("first_date")))
    
    # Optionally, drop the temporary 'first_date' column as it is no longer needed
    df = df.drop("first_date")
    
    # Write the resulting DataFrame to the output Parquet directory in overwrite mode
    df.write.mode("overwrite").parquet(output_parquet)
    logging.info(f"Output data with the 'age_epss_pub' feature saved to {output_parquet}")

    # Stop the Spark session
    spark.stop()

if __name__ == '__main__':
    create_epss_pub(
        input_parquet="data/epss/epss_parquet/epss_interpolated.parquet",
        output_parquet="data/epss/epss_parquet/epss_pub_features.parquet"
    )
