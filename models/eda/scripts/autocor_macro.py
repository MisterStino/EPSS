#!/usr/bin/env python
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  # for pacf

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
    The cohort index is computed as:
        cohort_index = floor((pub_date - global_min_pub_date) / days_per_cohort) + 1,
    and the cohort label is formed as "C" concatenated with the cohort_index.
    
    Returns a Spark DataFrame with columns: (cve, pub_cohort, pub_date).
    """
    w = Window.partitionBy("cve")
    df_pub = df.groupBy("cve").agg(F.min("date").alias("pub_date"))

    global_min_pub_date = df_pub.agg(F.min("pub_date").alias("global_min")).collect()[0]["global_min"]
    global_min_pub_date_lit = F.to_date(F.lit(global_min_pub_date))

    df_pub = df_pub.withColumn("days_from_global_min", F.datediff(F.col("pub_date"), global_min_pub_date_lit))
    df_pub = df_pub.withColumn("cohort_index", F.floor(F.col("days_from_global_min") / days_per_cohort) + F.lit(1))
    df_pub = df_pub.withColumn("pub_cohort", F.concat(F.lit("C"), F.col("cohort_index").cast(T.StringType())))

    return df_pub.select("cve", "pub_cohort", "pub_date")

def compute_pacf_for_cohorts(
    input_parquet="data/full_db/processed/final_full_data.parquet",
    output_folder="models/eda/figs/cohorts_pacf",
    days_per_cohort=90,
    maturity_threshold=30,
    user_max_lag=1100,      # this is the upper bound provided by the user
    do_difference=True
):
    """
    1) Read the dataset with Spark.
    2) Cast columns and compute publication-based cohorts using add_pub_cohort_label().
    3) Join back so that each row has its pub_cohort.
    4) Compute 'day_since_pub' for each row, then group by (pub_cohort, date)
       to compute the daily mean EPS for each cohort.
    5) For each cohort, form a time series (ordered by date) of the mean EPS.
       Optionally, difference the time series (if do_difference=True) to help stationarity.
    6) Dynamically calculate the max lag for the PACF based on the time series length:
         dynamic_max_lag = min(user_max_lag, int(T/2))
       where T is the number of time points.
    7) Compute the PACF for each cohort’s time series.
    8) Plot the PACF results for all cohorts in a stacked vertical layout.
    
    The figure is saved in high resolution.
    """
    os.makedirs(output_folder, exist_ok=True)
    spark = get_spark_session()
    
    # (1) Read dataset and cast columns.
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    # (2) Get publication-based cohorts.
    df_pub_cohorts = add_pub_cohort_label(df, days_per_cohort=days_per_cohort)
    
    # Compute unique CVE counts per cohort.
    cve_counts_df = df_pub_cohorts.groupBy("pub_cohort").agg(F.countDistinct("cve").alias("unique_cve_count"))
    cve_counts_pd = cve_counts_df.toPandas()
    unique_cve_dict = dict(zip(cve_counts_pd['pub_cohort'], cve_counts_pd['unique_cve_count']))
    
    # (3) Join to attach pub_cohort to each record.
    df_joined = df.join(df_pub_cohorts, on="cve", how="left")
    
    # (4) Compute day_since_pub per record (using each CVE's own pub_date).
    df_joined = df_joined.withColumn("pub_date", F.col("pub_date").cast(T.DateType()))
    df_joined = df_joined.withColumn("day_since_pub", F.datediff(F.col("date"), F.col("pub_date")))
    
    # (5) Filter out records for CVEs that are too new (if desired).
    if maturity_threshold is not None:
        df_joined = df_joined.filter(F.col("day_since_pub") >= maturity_threshold)
    
    # (6) Group by (pub_cohort, date) to compute daily mean EPS for each cohort.
    df_mean = df_joined.groupBy("pub_cohort", "date").agg(F.mean("epss").alias("mean_epss"))
    
    # Convert the aggregated daily time series to Pandas.
    pd_mean = df_mean.orderBy("pub_cohort", "date").toPandas()
    if pd_mean.empty:
        print("No data after filtering; check maturity_threshold or input data.")
        spark.stop()
        return
    
    # Dictionary to store the date range for each cohort.
    cohort_date_range = {}
    
    # (7) For each cohort, compute PACF with dynamic max_lag.
    cohorts = sorted(pd_mean["pub_cohort"].unique().tolist(), key=lambda x: int(x.replace("C","")))
    pacf_results = {}  # dictionary: key = cohort, value = (lags, pacf_values)
    
    for cohort in cohorts:
        subdf = pd_mean[pd_mean["pub_cohort"] == cohort].copy()
        subdf.sort_values("date", inplace=True)
        subdf["date"] = pd.to_datetime(subdf["date"])
        
        # Record the date range for this cohort.
        start_date = subdf["date"].min()
        end_date = subdf["date"].max()
        cohort_date_range[cohort] = (start_date, end_date)
        
        ts_values = subdf["mean_epss"].values
        T_length = len(ts_values)
        
        # Check for minimal length: require at least 10 points for PACF to be meaningful.
        if T_length < 10:
            print(f"Skipping cohort {cohort}: time series too short (length={T_length})")
            continue
        
        # If do_difference is True, we take the first difference to stabilize the series.
        if do_difference:
            ts_values = pd.Series(ts_values).diff().dropna().values
            T_length = len(ts_values)  # update T_length
        
        # Dynamically set max_lag to be at most half the series length.
        dynamic_max_lag = min(user_max_lag, max(1, int(T_length / 2)))
        
        # Compute PACF using statsmodels.
        pacf_vals = sm.tsa.stattools.pacf(ts_values, nlags=dynamic_max_lag, method='yw')
        pacf_results[cohort] = (list(range(dynamic_max_lag + 1)), pacf_vals)
    
    # (8) Plot PACF for each cohort in a vertically stacked layout.
    valid_cohorts = list(pacf_results.keys())
    if not valid_cohorts:
        print("No cohorts with sufficient data for PACF computation.")
        spark.stop()
        return
    
    num_plots = len(valid_cohorts)
    fig_height = 3 * num_plots  # approximately 3 inches per subplot
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(16, fig_height), dpi=240)
    
    # If there is only one subplot, ensure axes is a list.
    if num_plots == 1:
        axes = [axes]
    
    for idx, cohort in enumerate(valid_cohorts):
        ax = axes[idx]
        lags, pacf_vals = pacf_results[cohort]
        ax.bar(lags, pacf_vals, color='C0', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)
        
        # Retrieve unique CVE count for this cohort.
        unique_count = unique_cve_dict.get(cohort, "N/A")
        # Retrieve date range for this cohort.
        start_date, end_date = cohort_date_range.get(cohort, (None, None))
        if start_date is not None and end_date is not None:
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
        else:
            start_date_str, end_date_str = "N/A", "N/A"
        
        # Set the title with additional information.
        ax.set_title(f"PACF for Cohort {cohort} (diff={do_difference}) - {unique_count} CVEs, {start_date_str} to {end_date_str}", fontsize=12)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Partial Autocorrelation")
    
    plt.suptitle("Partial Autocorrelation (PACF) per Publication Cohort", fontsize=16, y=1.02)
    plt.tight_layout()
    
    outpng = os.path.join(output_folder, "pacf_per_cohort_dynamic.png")
    plt.savefig(outpng, dpi=240)
    plt.close()
    
    print(f"Saved multi-subplot PACF figure to {outpng}")
    
    spark.stop()
    print("Done. PACF analysis complete.")

if __name__ == "__main__":
    compute_pacf_for_cohorts(
        input_parquet="data/general_utils/files/high_score_above_0.4_with_initial_below_0.4.parquet",
        output_folder="models/eda/figs/cohorts_pacf",
        days_per_cohort=90,
        maturity_threshold=10,
        user_max_lag=1100,
        do_difference=True
    )
