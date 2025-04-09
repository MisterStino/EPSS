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
    
    For each CVE, compute its publication date (min(date)) and then, using the global 
    minimum publication date across all CVEs, calculate the difference in days between 
    each CVE's pub_date and the global minimum. The cohort index is computed as:
        cohort_index = floor((pub_date - global_min_pub_date) / days_per_cohort) + 1,
    and then the cohort label is formed as "C" concatenated with the cohort_index.
    
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

def compute_individual_pacf_avg(
    input_parquet="data/full_db/processed/final_full_data.parquet",
    output_folder="models/eda/figs/cohorts_pacf_indiv_avg",
    days_per_cohort=90,
    maturity_threshold=30,
    user_max_lag=1100,
    do_difference=True,
    min_series_length=10
):
    """
    For each individual CVE within each publication cohort, compute its own PACF and then 
    average (across CVEs) the PACF coefficients per lag within the cohort.
    
    Steps:
      1. Read the dataset, cast columns, and attach publication-based cohorts using add_pub_cohort_label().
      2. Compute each record's day_since_pub (relative to its CVE's pub_date).
      3. Filter out records with day_since_pub < maturity_threshold.
      4. Aggregate the data by CVE to produce an individual time series for EPS.
         Here we compute the daily mean EPS per CVE.
      5. For each CVE with sufficient data (length >= min_series_length), optionally difference 
         the series and compute its PACF up to a dynamically determined lag: dynamic_max_lag = min(user_max_lag, int(length/2)).
         Exclude lag 0.
      6. For each publication cohort, for each lag, average the individual CVE PACF values (only include those CVEs where the PACF was computed for that lag).
      7. Plot, for each cohort, a bar plot of the averaged PACF values versus lag (excluding lag 0) in a stacked vertical subplot layout.
         Save the high-resolution figure.
    """
    os.makedirs(output_folder, exist_ok=True)
    spark = get_spark_session()
    
    # (1) Read dataset and cast columns.
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    # (2) Compute publication-based cohorts.
    df_pub_cohorts = add_pub_cohort_label(df, days_per_cohort=days_per_cohort)
    
    # (3) Join the cohort info back to main dataset.
    df_joined = df.join(df_pub_cohorts, on="cve", how="left")
    
    # (4) Compute day_since_pub per record.
    df_joined = df_joined.withColumn("pub_date", F.col("pub_date").cast(T.DateType()))
    df_joined = df_joined.withColumn("day_since_pub", F.datediff(F.col("date"), F.col("pub_date")))
    
    # (5) Apply maturity filter.
    if maturity_threshold is not None:
        df_joined = df_joined.filter(F.col("day_since_pub") >= maturity_threshold)
    
    # (6) Aggregate by CVE and date to form an individual time series (use daily mean EPS per CVE).
    df_individual = df_joined.groupBy("cve", "pub_cohort", "date").agg(F.mean("epss").alias("mean_epss"))
    
    # Order by cve and date, and convert to Pandas.
    pd_individual = df_individual.orderBy("cve", "date").toPandas()
    if pd_individual.empty:
        print("No individual CVE time series after filtering.")
        spark.stop()
        return
    
    # Convert date to datetime.
    pd_individual["date"] = pd.to_datetime(pd_individual["date"])
    
    # Group by CVE to compute individual PACF.
    grouped = pd_individual.groupby(["cve", "pub_cohort"])
    
    # For each CVE, compute PACF (excluding lag 0) and record the PACF values per lag.
    cohort_pacf = {}  # Dictionary: key = pub_cohort, value = {lag: list of pacf values}
    for (cve, cohort), group in grouped:
        group = group.sort_values("date")
        ts = group["mean_epss"].values
        if len(ts) < min_series_length:
            continue
        if do_difference:
            ts = pd.Series(ts).diff().dropna().values
        T_length = len(ts)
        dynamic_max_lag = min(user_max_lag, max(1, int(T_length / 2)))
        # Compute PACF using Yule-Walker method
        pacf_values = sm.tsa.stattools.pacf(ts, nlags=dynamic_max_lag, method='yw')
        # Exclude lag 0 (always 1) – start from lag 1.
        for lag in range(1, dynamic_max_lag + 1):
            if cohort not in cohort_pacf:
                cohort_pacf[cohort] = {}
            if lag not in cohort_pacf[cohort]:
                cohort_pacf[cohort][lag] = []
            cohort_pacf[cohort][lag].append(pacf_values[lag])
    
    # (7) For each cohort, compute the average PACF per lag.
    averaged_pacf = {}  # key = cohort, value = (lags, avg_pacf_values)
    for cohort, lag_dict in cohort_pacf.items():
        lags = sorted(lag_dict.keys())
        avg_vals = []
        for lag in lags:
            avg_vals.append(sum(lag_dict[lag]) / len(lag_dict[lag]))
        averaged_pacf[cohort] = (lags, avg_vals)
    
    if not averaged_pacf:
        print("No valid PACF computed for any cohort.")
        spark.stop()
        return
    
    # (8) Plot the averaged PACF for each cohort in vertically stacked subplots.
    valid_cohorts = sorted(averaged_pacf.keys(), key=lambda x: int(x.replace("C", "")))
    num_plots = len(valid_cohorts)
    fig_height = 3 * num_plots  # 3 inches per subplot
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(16, fig_height), dpi=240)
    if num_plots == 1:
        axes = [axes]
    
    for i, cohort in enumerate(valid_cohorts):
        lags, avg_pacf_vals = averaged_pacf[cohort]
        ax = axes[i]
        # Plot as bar plot
        ax.bar(lags, avg_pacf_vals, color='C0', alpha=0.7)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(f"Average PACF for Cohort {cohort} (Individual CVEs)", fontsize=12)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Average Partial Autocorrelation")
        # Optionally set the x-ticks to only a subset if lags are many
    plt.suptitle("Averaged Partial Autocorrelation per Publication Cohort", fontsize=16, y=1.02)
    plt.tight_layout()
    outpath = os.path.join(output_folder, "averaged_pacf_per_cohort_micro.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"Saved averaged PACF plot to: {outpath}")
    
    spark.stop()
    print("Done. Individual PACF averaging analysis complete.")

if __name__ == "__main__":

    # Optionally, you can call the new function as:
    compute_individual_pacf_avg(
        input_parquet="data/general_utils/files/high_score_above_0.4_with_initial_below_0.4.parquet",
        output_folder="models/eda/figs/cohorts_pacf_indiv_avg",
        days_per_cohort=90,
        maturity_threshold=10,
        user_max_lag=1100,
        do_difference=True,
        min_series_length=10
    )
