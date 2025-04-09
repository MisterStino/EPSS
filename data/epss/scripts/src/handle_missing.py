import os
import sys
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window
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

def fill_missing_dates_and_interpolate(input_parquet, output_parquet):
    """
    Reads the Parquet file with EPS data (with composite key: cve and date),
    generates missing dates for each CVE from its min(date) to max(date), and
    fills in the missing 'epss' values using linear interpolation.
    
    The final DataFrame is sorted by cve and date and written to output_parquet.
    """
    spark = get_spark_session()   

    # 1. Read the dataset.
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    # 2. Get the date range for each CVE.
    df_range = df.groupBy("cve").agg(F.min("date").alias("min_date"), F.max("date").alias("max_date"))
    
    # 3. Generate a complete sequence of dates per CVE.
    # Use the sequence function to generate a list of dates from min_date to max_date with a 1 day step.
    df_range = df_range.withColumn("date_seq", F.expr("sequence(min_date, max_date, interval 1 day)"))
    
    # Explode the date sequence to get one row per date per CVE.
    df_full = df_range.withColumn("date", F.explode("date_seq")).select("cve", "date")
    
    # 4. Join the full date set with the original dataset to include missing dates.
    df_joined = df_full.join(df.select("cve", "date", "epss"), on=["cve", "date"], how="left")
    
    # 5. Interpolate missing EPS values.
    # Define a window partitioned by cve ordered by date.
    window_spec = Window.partitionBy("cve").orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    window_spec_rev = Window.partitionBy("cve").orderBy("date").rowsBetween(Window.currentRow, Window.unboundedFollowing)
    
    # Get the last non-null epss value and date in the past.
    df_joined = df_joined.withColumn("epss_prev", F.last("epss", ignorenulls=True).over(window_spec))
    df_joined = df_joined.withColumn("date_prev", F.last("date", ignorenulls=True).over(window_spec))
    
    # Get the first non-null epss value and date in the future.
    df_joined = df_joined.withColumn("epss_next", F.first("epss", ignorenulls=True).over(window_spec_rev))
    df_joined = df_joined.withColumn("date_next", F.first("date", ignorenulls=True).over(window_spec_rev))
    
    # Convert dates to timestamps (as seconds) to compute differences.
    df_joined = df_joined.withColumn("date_ts", F.unix_timestamp("date"))
    df_joined = df_joined.withColumn("date_prev_ts", F.unix_timestamp("date_prev"))
    df_joined = df_joined.withColumn("date_next_ts", F.unix_timestamp("date_next"))
    
    # Compute fraction: (date - date_prev) / (date_next - date_prev)
    df_joined = df_joined.withColumn(
        "fraction",
        (F.col("date_ts") - F.col("date_prev_ts")) / (F.col("date_next_ts") - F.col("date_prev_ts"))
    )
    
    # Compute interpolated epss: epss_prev + fraction * (epss_next - epss_prev)
    df_joined = df_joined.withColumn(
        "interp_epss",
        F.col("epss_prev") + F.col("fraction") * (F.col("epss_next") - F.col("epss_prev"))
    )
    
    # Use the original epss if present; otherwise use interp_epss.
    df_filled = df_joined.withColumn(
        "epss_filled",
        F.when(F.col("epss").isNull(), F.col("interp_epss")).otherwise(F.col("epss"))
    )
    
    # 6. Select the desired columns and sort by composite key.
    df_final = df_filled.select("cve", "date", F.col("epss_filled").alias("epss"))
    df_final = df_final.orderBy("cve", "date")
    
    # 7. Write the result to a Parquet file.
    df_final.write.mode("overwrite").parquet(output_parquet)
    
    print(f"Finished interpolation. Output written to {output_parquet}")
    spark.stop()
    
if __name__ == "__main__":
    input_parquet = "data/epss/epss_parquet/epss_all.parquet"
    output_parquet = "data/epss/processed/epss_processed.parquet"
    fill_missing_dates_and_interpolate(input_parquet, output_parquet)
