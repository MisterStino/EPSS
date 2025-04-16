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
        # Use DateType() if you only have YYYY-MM-DD.
        df = df.withColumn('date', F.col('date').cast(T.DateType()))
    if 'epss' in df.columns:
        df = df.withColumn('epss', F.col('epss').cast(T.DoubleType()))
    return df

def fill_missing_dates_and_forward_fill(
    input_parquet="data/epss/epss_parquet/epss_all.parquet",
    output_parquet="data/epss/processed/epss_processed.parquet"
):
    """
    Reads the Parquet file with EPS data (with composite key: cve and date),
    generates missing dates for each CVE from its min(date) to max(date), and
    fills in the missing 'epss' values using forward fill (last observation carried forward).
    
    **Before the forward fill,** for each CVE's time series, this function checks if more than 20%
    of the epss values are missing (where missing is defined as either null or NaN in the original
    epss column). If so, it collects that CVE's id into a Python list.
    
    Finally, the function:
      - Prints the number of CVEs with >20% missing epss,
      - Prints the first 30 such CVE ids,
      - Produces a final DataFrame with continuous dates where missing epss values are forward filled,
      - Writes the final DataFrame to output_parquet.
    """
    spark = get_spark_session()   
    
    # 1. Read the dataset and cast columns.
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    # 2. Get the date range for each CVE.
    df_range = df.groupBy("cve").agg(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date")
    )
    
    # 3. Generate a complete sequence of dates per CVE.
    df_range = df_range.withColumn("date_seq", F.expr("sequence(min_date, max_date, interval 1 day)"))
    df_full = df_range.withColumn("date", F.explode("date_seq")).select("cve", "date")
    
    # 4. Join the full date set with the original dataset to include missing dates.
    df_joined = df_full.join(df.select("cve", "date", "epss"), on=["cve", "date"], how="left")
    
    # 5. BEFORE forward fill: Check for missing epss ratios per CVE.
    #    Count total rows and missing epss (where epss is null or NaN).
    missing_summary = df_joined.groupBy("cve").agg(
        F.count("*").alias("total_rows"),
        F.sum(F.when(F.col("epss").isNull() | F.isnan(F.col("epss")), 1).otherwise(0)).alias("missing_count")
    ).withColumn("missing_ratio", F.col("missing_count") / F.col("total_rows"))
    
    cves_with_missing = missing_summary.filter(F.col("missing_ratio") > 0.2).select("cve")
    missing_cve_list = [row["cve"] for row in cves_with_missing.collect()]
    print("Number of CVEs with more than 20% missing epss values:", len(missing_cve_list))
    print("First 30 CVE ids with >20% missing epss values:", missing_cve_list[:30])
    
    # 6. Prepare for forward fill:
    #    Create epss_clean to treat both NaN and null as missing (set them to None).
    df_joined = df_joined.withColumn(
        "epss_clean", 
        F.when(F.isnan(F.col("epss")) | F.col("epss").isNull(), None)
         .otherwise(F.col("epss"))
    )
    
    # 7. Forward fill missing EPS values using the last non-null observation.
    window_spec = Window.partitionBy("cve").orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df_filled = df_joined.withColumn(
        "epss_filled", 
        F.last("epss_clean", ignorenulls=True).over(window_spec)
    )
    
    # 8. Prepare the final DataFrame: select desired columns and sort by cve and date.
    df_final = df_filled.select("cve", "date", F.col("epss_filled").alias("epss"))
    df_final = df_final.orderBy("cve", "date")
    
    # 9. Write the final result to a Parquet file.
    df_final.write.mode("overwrite").parquet(output_parquet)
    print(f"Finished forward fill. Output written to {output_parquet}")
    
if __name__ == "__main__":
    input_parquet = "data/epss/epss_parquet/epss_all.parquet"
    output_parquet = "data/epss/processed/epss_processed.parquet"
    fill_missing_dates_and_forward_fill(input_parquet, output_parquet)
