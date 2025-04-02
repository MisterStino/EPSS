import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyspark.sql import functions as F
from pyspark.sql import types as T
from t3_spark.session import get_spark_session

def pooled_eda_plots(
    input_parquet='data/full_db/processed/final_full_data_parquet',
    output_folder='models/eda/figs/pooled'
):
    """
    Performs pooled EDA on a large multi-entity time-series dataset of CVEs and daily EPSS scores.

    Data context:
      - input_parquet: Path to a Parquet file with columns [cve, date, epss]
      - cve (string), date (date), epss (float in [0,1])
      - (cve, date) is the composite key for each time step of each CVE.

    Steps:
      1) Read the dataset via Spark.
      2) Per-CVE summary stats: mean, stddev, min, max, count(epss).
         -> Plot distributions of these stats across all CVEs.
      3) Global average over time (daily):
         -> group by date, compute avg(epss), plot line chart.
      4) Global monthly average:
         -> parse year-month from date, group, plot line chart.

    Saves each plot as a PNG to output_folder. 
    """

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # 1) Spark session & read data
    spark = get_spark_session()
    df = spark.read.parquet(input_parquet)

    # Cast columns to be sure they have correct types
    df = (df
          .withColumn("cve", F.col("cve").cast(T.StringType()))
          .withColumn("date", F.col("date").cast(T.DateType()))
          .withColumn("epss", F.col("epss").cast(T.DoubleType()))
         )

    # 2) Per-CVE summary stats
    # groupBy cve -> compute mean, stddev, min, max, count
    cve_stats = (df
                 .groupBy("cve")
                 .agg(
                     F.mean("epss").alias("mean_epss"),
                     F.stddev("epss").alias("std_epss"),
                     F.min("epss").alias("min_epss"),
                     F.max("epss").alias("max_epss"),
                     F.count("epss").alias("count_epss")
                 )
                )

    # Convert to Pandas for distribution plotting
    # ~280k CVEs should fit in memory, but be mindful if it grows bigger.
    cve_stats_pd = cve_stats.toPandas()

    # ---- Plot distribution of mean_epss ----
    plt.figure()
    cve_stats_pd["mean_epss"].hist(bins=50)
    plt.title("Distribution of Mean EPSS across CVEs")
    plt.xlabel("Mean EPSS")
    plt.ylabel("Count of CVEs")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "distribution_mean_epss.png"))
    plt.close()

    # ---- Plot distribution of std_epss ----
    # Some CVEs might have a single data point => std_epss could be null
    cve_stats_pd["std_epss"] = cve_stats_pd["std_epss"].fillna(0.0)
    plt.figure()
    cve_stats_pd["std_epss"].hist(bins=50)
    plt.title("Distribution of Std. Dev. EPSS across CVEs")
    plt.xlabel("Standard Deviation of EPSS")
    plt.ylabel("Count of CVEs")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "distribution_std_epss.png"))
    plt.close()

    # (Optionally) distribution of count_epss -> how many days each CVE has data for
    plt.figure()
    cve_stats_pd["count_epss"].hist(bins=50)
    plt.title("Distribution of Observations (count) per CVE")
    plt.xlabel("Number of time steps for each CVE")
    plt.ylabel("Count of CVEs")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "distribution_count_epss.png"))
    plt.close()

    # 3) Global average EPSS over time (daily)
    daily_avg = (df
                 .groupBy("date")
                 .agg(F.mean("epss").alias("avg_epss"))
                 .orderBy("date")
                )
    daily_pd = daily_avg.toPandas()
    # Plot daily average
    plt.figure(figsize=(12,6))
    plt.plot(daily_pd["date"], daily_pd["avg_epss"], marker='o', linestyle='-')
    plt.title("Global Daily Average EPSS over Time")
    plt.xlabel("Date")
    plt.ylabel("Average EPSS")
    # Format x-axis dates
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "global_daily_avg_epss.png"))
    plt.close()

    # 4) Global monthly average
    # create a year_month column e.g. 'YYYY-MM'
    monthly_df = (df
                  .withColumn("year_month", F.date_format(F.col("date"), "yyyy-MM"))
                  .groupBy("year_month")
                  .agg(F.mean("epss").alias("avg_epss"))
                  .orderBy("year_month")
                 )
    monthly_pd = monthly_df.toPandas()
    # Convert 'year_month' to a date (hack: append "-01")
    monthly_pd["month_date"] = monthly_pd["year_month"].apply(lambda x: x + "-01")
    monthly_pd["month_date"] = pd.to_datetime(monthly_pd["month_date"], format="%Y-%m-%d")

    # Plot monthly
    plt.figure(figsize=(12,6))
    plt.plot(monthly_pd["month_date"], monthly_pd["avg_epss"], marker='o', linestyle='-')
    plt.title("Global Monthly Average EPSS")
    plt.xlabel("Year-Month")
    plt.ylabel("Average EPSS")
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "global_monthly_avg_epss.png"))
    plt.close()

    print(f"EDA plots saved in folder: {output_folder}")

    time.sleep(2)
    spark.stop()

# Example usage
if __name__ == "__main__":
    pooled_eda_plots(
        input_parquet='data/full_db/processed/final_full_data_parquet',
        output_folder='models/eda/figs/pooled'
    )
