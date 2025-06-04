#!/usr/bin/env python3
"""Spark-based analysis of relation between GitHub features and EPSS scores."""

import sys
import os
# Add project root to PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from t3_spark.session import get_spark_session
from pyspark.sql.functions import upper, col
import pyspark.sql.functions as F
import numpy as np


def main():
    # Initialize Spark session
    spark = get_spark_session("EPSS-GitHub Correlation")

    # Define paths
    tool_dir = os.path.dirname(__file__)
    gh_csv = os.path.join(tool_dir, '..', 'raw', 'github_bq.csv')
    epss_parquet = os.path.abspath(os.path.join(tool_dir, '..', '..', 'epss', 'epss_parquet', 'epss_all.parquet'))

    # Load GitHub event data
    print("Loading GitHub event data from", gh_csv)
    gh_df = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")
             .csv(gh_csv)
    )
    # Standardize cve_id and date
    gh_df = (
        gh_df
          .withColumn("cve_id", upper(col("cve_id")))
          .withColumn("date", col("date").cast("date"))
    )

    # Load EPSS data
    print("Loading EPSS data from", epss_parquet)
    epss_df = (
        spark.read
             .parquet(epss_parquet)
             .withColumnRenamed("cve", "cve_id")
    )
    epss_df = (
        epss_df
          .withColumn("cve_id", upper(col("cve_id")))
          .withColumn("date", col("date").cast("date"))
    )

    # Join datasets on cve_id and date
    joined = gh_df.join(epss_df, on=["cve_id", "date"], how="inner")
    joined_count = joined.count()
    print(f"Joined dataset has {joined_count} records")

    # Identify GitHub feature columns (exclude cve_id, date)
    feature_cols = [c for c in gh_df.columns if c not in ["cve_id", "date"]]
    print("GitHub feature columns:", feature_cols)

    # Compute Pearson correlation for each feature
    print("Computing Pearson correlations with epss:")
    for f in feature_cols:
        corr_value = joined.stat.corr(f, "epss")
        print(f"  {f} vs epss: {corr_value:.4f}")

    # Select CVEs that start with EPSS < 0.3 and eventually reach EPSS > 0.7
    print("Selecting CVEs with initial EPSS < 0.3 and eventual EPSS > 0.7...")
    stats_df = joined.groupBy("cve_id").agg(
        F.min("epss").alias("min_epss"),
        F.max("epss").alias("max_epss")
    )
    filtered = stats_df.filter((col("min_epss") < 0.3) & (col("max_epss") > 0.7))
    top_cve_list = [row.cve_id for row in filtered.collect()]
    print(f"Found {len(top_cve_list)} CVEs meeting EPSS thresholds")

    # Initialize storage for cross-correlations
    max_lag = 90  # days
    xcorrs = {f: {lag: [] for lag in range(-max_lag, max_lag+1)} for f in feature_cols}
    for cve in top_cve_list:
        print(f"Analyzing CVE {cve}...")
        sub = (
            joined.filter(col("cve_id") == cve)
                  .select("date", *feature_cols, "epss")
                  .orderBy("date")
        )
        pdf = sub.toPandas().set_index("date")
        for f in feature_cols:
            x = pdf[f].values
            y = pdf["epss"].values
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            x = (x - x.mean()) / x.std()
            y = (y - y.mean()) / y.std()
            for lag in range(-max_lag, max_lag+1):
                if lag < 0:
                    xi = x[:lag]
                    yi = y[-lag:]
                elif lag > 0:
                    xi = x[lag:]
                    yi = y[:-lag]
                else:
                    xi = x
                    yi = y
                # Only compute cross-correlation on windows with at least two points
                if len(xi) > 1 and len(yi) > 1:
                    # suppress runtime warnings for degenerate cases
                    with np.errstate(invalid='ignore', divide='ignore'):
                        corr = np.corrcoef(xi, yi)[0,1]
                    # record only valid correlations
                    if not np.isnan(corr):
                        xcorrs[f][lag].append(corr)

    # Summarize cross-correlations
    print("\nCross-correlation summary (lag vs mean corr):")
    for f in feature_cols:
        mean_corrs = {lag: np.nanmean(xcorrs[f][lag]) for lag in range(-max_lag, max_lag+1)}
        best_lag = max(mean_corrs, key=lambda lag: abs(mean_corrs[lag]))
        print(f"Feature {f}: best lag {best_lag} days, mean corr {mean_corrs[best_lag]:.4f}")

    # Stop Spark session
    spark.stop()


if __name__ == '__main__':
    main() 