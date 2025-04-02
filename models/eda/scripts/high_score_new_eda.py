from t3_spark.session import get_spark_session
from pyspark.sql import functions as F
import os
import matplotlib.pyplot as plt
from t3_spark.session import get_spark_session
from pyspark.sql import functions as F
import matplotlib.dates as mdates

def explore_high_score_cves():
    # Initialize the Spark session using your helper function.
    spark = get_spark_session()
    
    # Read the Parquet file into a Spark DataFrame.
    parquet_path = "data/general_utils/files/high_score_newer_cves.parquet"
    df = spark.read.parquet(parquet_path)
    
    # Show the first 10 rows of the DataFrame.
    # This gives an overview of the data (i.e., the "head" of the DataFrame).
    print("DataFrame Head:")
    df.show(10, truncate=False)
    
    # Get all unique values from the "cve" column.
    # distinct() returns a new DataFrame with unique "cve" values.
    unique_cves_df = df.select("cve").distinct()
    
    # Collect the unique CVE values to the driver as a list.
    unique_cves = unique_cves_df.rdd.map(lambda row: row["cve"]).collect()
    
    # Print out all unique CVE identifiers.
    print("\nUnique CVE IDs:")
    for cve in unique_cves:
        print(cve)
    
    # Stop the Spark session when done.
    spark.stop()





import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from t3_spark.session import get_spark_session
from pyspark.sql import functions as F

def plot_random_cve_time_series(n=3, input_path='data/general_utils/files/high_score_above_98.parquet'):
    """
    Plots the entire time series of EPS scores for n random unique CVE IDs.
    
    The x-axis is scaled to span from the earliest date to the latest date among the
    selected CVEs. The axis has a tick (minor tick) for every day, but text labels
    (major ticks) only appear every 10 days. Additionally, vertical gridlines are added
    at the minor tick positions to help trace data points to the axis.
    """
    # Initialize Spark session using your helper.
    spark = get_spark_session()
    
    # Read the Parquet file into a Spark DataFrame.
    df = spark.read.parquet(input_path)
    
    # Count and print the total number of distinct CVE IDs.
    distinct_cve_count = df.select("cve").distinct().count()
    print("Total distinct CVE IDs in the dataset:", distinct_cve_count)
    
    # Randomly select n distinct CVE IDs.
    distinct_cves = df.select("cve").distinct().orderBy(F.rand()).limit(n).collect()
    cve_list = [row["cve"] for row in distinct_cves]
    print("Selected CVE IDs:", cve_list)
    
    # List to store the Pandas DataFrames for each CVE.
    cve_pd_list = []
    
    # For each selected CVE, filter, sort by date, and convert to Pandas.
    for cve in cve_list:
        cve_df = df.filter(F.col("cve") == cve).orderBy("date")
        pandas_df = cve_df.toPandas()
        pandas_df.sort_values("date", inplace=True)
        cve_pd_list.append(pandas_df)
    
    # Determine global minimum and maximum dates across all selected CVEs.
    global_min_date = min(pd_df["date"].min() for pd_df in cve_pd_list)
    global_max_date = max(pd_df["date"].max() for pd_df in cve_pd_list)
    print("Global date range:", global_min_date, "to", global_max_date)
    
    # Create a matplotlib figure.
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot each CVE's time series.
    for cve, pd_df in zip(cve_list, cve_pd_list):
        ax.plot(pd_df["date"], pd_df["epss"], marker='o', label=cve)
    
    # Set the x-axis limits.
    ax.set_xlim(global_min_date, global_max_date)
    
    # Major ticks: show label every 10 days.
    major_locator = mdates.DayLocator(interval=10)
    major_formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_formatter)
    
    # Minor ticks: one tick for every day (without labels).
    minor_locator = mdates.DayLocator(interval=1)
    ax.xaxis.set_minor_locator(minor_locator)
    
    # Add vertical gridlines for minor ticks to help trace exact days.
    ax.grid(which='minor', axis='x', linestyle='--', color='gray', alpha=0.5)
    
    # Customize the plot.
    ax.set_xlabel("Date")
    ax.set_ylabel("EPS Score")
    ax.set_title("Time Series of EPS Score for Random CVEs")
    ax.legend(title="CVE ID")
    
    # Rotate the major tick labels for readability.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout so labels are not cut off.
    plt.tight_layout()
    plt.show()
    
    # Stop the Spark session.
    spark.stop()






if __name__ == "__main__":
    plot_random_cve_time_series(n=10,input_path="data/general_utils/files/high_score_above_0.9_with_initial_below_0.4.parquet")
    #explore_high_score_cves()
