import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from t3_spark.session import get_spark_session
from pyspark.sql import functions as F

def plot_single_cve_time_series(cve_id, input_path='data/full_db/processed/final_full_data.parquet'):
    """
    Plots the full time series of EPSS scores for a specific CVE ID.
    
    The x-axis:
      - Spans the full date range for this CVE.
      - Shows minor ticks (every day) and major labels (every 10 days).
      - Adds vertical grid lines at minor ticks to trace EPS values to exact days.
      
    Parameters:
      cve_id (str): The CVE ID to plot (e.g., 'CVE-2023-12345').
      input_path (str): Path to the Parquet dataset with full CVE time series.
    """
    # Step 1: Initialize Spark
    spark = get_spark_session()
    
    # Step 2: Load full dataset
    df = spark.read.parquet(input_path)
    distinct_cve_count = df.select("cve").distinct().count()
    print("Total distinct CVE IDs in the dataset:", distinct_cve_count)
    
    # Step 3: Filter only rows for the given CVE
    cve_df = df.filter(F.col("cve") == cve_id).orderBy("date")
    
    # Step 4: Convert to Pandas for plotting
    pandas_df = cve_df.toPandas()
    pandas_df.sort_values("date", inplace=True)
    
    if pandas_df.empty:
        print(f"No data found for CVE: {cve_id}")
        spark.stop()
        return
    
    # Step 5: Get min/max date for x-axis
    min_date = pandas_df["date"].min()
    max_date = pandas_df["date"].max()
    
    # Step 6: Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(pandas_df["date"], pandas_df["epss"], marker='o', label=cve_id)
    
    ax.set_xlim(min_date, max_date)
    
    # Major ticks: every 10 days, with labels
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Minor ticks: every day
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    
    # Add gridlines on minor ticks
    ax.grid(which='minor', axis='x', linestyle='--', color='gray', alpha=0.5)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("EPSS Score")
    ax.set_title(f"Time Series of EPSS Score for {cve_id}")
    ax.legend(title="CVE ID")
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Step 7: Stop Spark
    spark.stop()

# Example usage
if __name__ == "__main__":
    cve_list = [
        "CVE-2022-2888",
        "CVE-2022-32149",
        "CVE-2021-44576",
        "CVE-2017-2157",
        "CVE-2021-36185"
    ]
    
    # Call the plot function for each of the CVEs in the list.
    for cve in cve_list:
        print(f"Plotting time series for CVE: {cve}")
        plot_single_cve_time_series(cve)