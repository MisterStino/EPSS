import os
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from t3_spark.session import get_spark_session

import random
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F
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



def quick_inspect_df(parquet_path="data/full_db/sampled/final_full_data_sampled.parquet"):
    """
    Quick-inspect the full dataset contained in a Parquet file.
    
    This function:
      1. Reads the Parquet file into a Spark DataFrame.
      2. Prints the head of the DataFrame.
      3. Prints the shape (number of rows and columns).
      4. Computes, for each CVE, the maximum EPS score.
      5. Randomly selects 2 CVEs that never have an EPS score above 0.7
         and 2 CVEs that have at least one EPS score above 0.7.
      6. For each of these 4 CVEs, extracts the full time series (date vs. epss)
         and plots them in a 2x2 grid.
      7. Plots a histogram of all EPS scores.
      8. Displays the combined figure.
    
    The input file is in long format (one row per (cve, date) observation), where:
      - 'cve' is the CVE identifier (string),
      - 'date' is the observation date (date type),
      - 'epss' is the EPS score (numeric).
    
    The composite key (cve, date) means that each row represents a unique time-stamped observation
    for that CVE, and the entire time series for a given CVE consists of all rows for that CVE.
    """
    
    # Initialize Spark session
    spark = get_spark_session()
    
    # 1. Read the Parquet file into a DataFrame.
    df = spark.read.parquet(parquet_path)
    
    # 2. Print the head of the DataFrame.
    print("DataFrame Head:")
    df.show(5)
    
    # 3. Print the shape: number of rows and columns.
    num_rows = df.count()
    num_cols = len(df.columns)
    print(f"DataFrame Shape: {num_rows} rows, {num_cols} columns")
    
    # 4. Compute the maximum epss score for each CVE.
    # This gives one row per CVE with the maximum epss observed.
    cve_max_df = df.groupBy("cve").agg(F.max("epss").alias("max_epss"))
    
    # 5. Pick randomly 2 CVEs that never get an epss score above 0.7.
    never_above_df = cve_max_df.filter(F.col("max_epss") <= 0.7).orderBy(F.rand()).limit(2)
    # And 2 CVEs that do get an epss score above 0.7 (at some point).
    above_df = cve_max_df.filter(F.col("max_epss") > 0.7).orderBy(F.rand()).limit(2)
    
    never_above_cves = [row["cve"] for row in never_above_df.collect()]
    above_cves = [row["cve"] for row in above_df.collect()]
    
    print("Selected CVEs that never exceed 0.7:")
    print(never_above_cves)
    print("Selected CVEs that have at least one EPS above 0.7:")
    print(above_cves)
    
    # Combine the selected CVEs (total 4).
    selected_cves = never_above_cves + above_cves
    
    # 6. For each selected CVE, extract its time series and convert to Pandas for plotting.
    ts_data = {}
    for cve in selected_cves:
        cve_df = df.filter(F.col("cve") == cve).orderBy("date")
        # Convert to Pandas DataFrame.
        ts_data[cve] = cve_df.toPandas()
        # Ensure that the 'date' column is parsed as a datetime type.
        ts_data[cve]["date"] = pd.to_datetime(ts_data[cve]["date"])
    
    # 7. For the EPS distribution, collect all epss values.
    # (Collecting to driver is acceptable if the dataset of epss values is not enormous.)
    epss_values = [row["epss"] for row in df.select("epss").collect() if row["epss"] is not None]
    
    # 8. Set up the plotting layout using Matplotlib's GridSpec.
    # We will create a figure where the top portion is a 2x2 grid for the 4 time series,
    # and the bottom portion is a larger subplot for the EPS distribution.
    fig = plt.figure(constrained_layout=True, figsize=(12, 10))
    
    # Create a grid with 3 rows and 2 columns.
    # The top two rows (rows 0 and 1) will contain 4 subplots.
    # The bottom row (row 2) will have one subplot spanning both columns.
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(nrows=3, ncols=2, figure=fig)
    
    # Plot the 4 time series in the top 2 rows (2x2 grid).
    for i, cve in enumerate(selected_cves):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        data = ts_data[cve]
        ax.plot(data["date"], data["epss"], marker="o", linestyle="-")
        ax.set_title(f"CVE: {cve}")
        ax.set_xlabel("Date")
        ax.set_ylabel("epss")
        ax.grid(True)
    
    # Plot the distribution of EPS values in the bottom row (spanning both columns).
    ax_dist = fig.add_subplot(gs[2, :])
    ax_dist.hist(epss_values, bins=50, color="skyblue", edgecolor="black")
    ax_dist.set_title("Distribution of all epss scores")
    ax_dist.set_xlabel("epss score")
    ax_dist.set_ylabel("Frequency")
    ax_dist.grid(True)
    
    # Display the plot.
    plt.tight_layout()
    plt.show()
    
    spark.stop()

if __name__ == "__main__":
    quick_inspect_df()
