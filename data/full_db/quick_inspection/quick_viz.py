#!/usr/bin/env python
import os
import sys
import logging
import datetime

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.functions import col, max as spark_max
import matplotlib.pyplot as plt
import pandas as pd

# If you have a helper for Spark session, you could import it here.
# from t3_spark.session import get_spark_session

def get_spark_session():
    """Initialize and return a Spark session."""
    return SparkSession.builder \
            .appName("Quick Viz") \
            .getOrCreate()

def print_dataset_summary(spark, final_path, base_path):
    """Load the final and base datasets and print summary information."""
    # Load the final merged dataset
    print(f"Loading final dataset from {final_path} ...")
    final_df = spark.read.parquet(final_path)
    final_count = final_df.count()
    final_cols = len(final_df.columns)
    print(f"Final dataset: {final_count} rows, {final_cols} columns.")

    # Load the base dataset (for epss)
    print(f"Loading base (EPS) dataset from {base_path} ...")
    base_df = spark.read.parquet(base_path)
    base_count = base_df.count()
    base_cols = len(base_df.columns)
    print(f"Base dataset: {base_count} rows, {base_cols} columns.")

    # Check that the number of rows in the final dataset and the base dataset are equal.
    if final_count != base_count:
        print("WARNING: The final dataset row count does not match the base dataset row count!")
    else:
        print("Row count check PASSED: final and base datasets have the same number of rows.")
    return final_df, base_df

def select_sample_cves(final_df, epss_threshold=0.7, num_per_group=5):
    """
    Select 5 CVEs that never get an epss value above the threshold and 5
    CVEs that do have an epss value above the threshold. Returns a list of 10 CVEs.
    """
    # Compute the maximum epss score per CVE.
    cve_max_df = final_df.groupBy("cve").agg(spark_max("epss").alias("max_epss"))
    
    # Select CVEs where the max epss is <= threshold (never above threshold)
    never_above_df = cve_max_df.filter(col("max_epss") <= epss_threshold).orderBy(F.rand()).limit(num_per_group)
    # Select CVEs where the max epss is > threshold (at some point above threshold)
    above_df = cve_max_df.filter(col("max_epss") > epss_threshold).orderBy(F.rand()).limit(num_per_group)
    
    # Collect CVE identifiers as lists.
    never_above_cves = [row["cve"] for row in never_above_df.collect()]
    above_cves = [row["cve"] for row in above_df.collect()]
    
    print("Selected CVEs for which epss never exceeds the threshold:")
    print(never_above_cves)
    print("Selected CVEs for which epss goes above the threshold:")
    print(above_cves)
    
    # Combine both lists (10 in total)
    sample_cves = never_above_cves + above_cves
    return sample_cves

def check_time_series_integrity(spark, sample_cves, final_df, base_df, modules):
    """
    For each selected CVE, extract the full time series from both final_df and base_df,
    and check that:
        - The number of rows is the same.
        - The composite key (cve, date) is the same.
        - The epss column is identical.
    Also print some information from the comparison.
    """
    integrity_ok = True
    for cve in sample_cves:
        print(f"\nChecking CVE: {cve}")
        final_cve_df = final_df.filter(col("cve") == cve).orderBy("date")
        base_cve_df = base_df.filter(col("cve") == cve).orderBy("date")
        
        # Convert to pandas for detailed comparison (each CVE has a relatively small number of rows)
        final_pd = final_cve_df.select("cve", "date", "epss").toPandas()
        base_pd = base_cve_df.select("cve", "date", "epss").toPandas()
        
        # Convert date columns to datetime (if not already)
        final_pd['date'] = pd.to_datetime(final_pd['date'])
        base_pd['date'] = pd.to_datetime(base_pd['date'])
        
        # Check that the lengths are the same
        if len(final_pd) != len(base_pd):
            integrity_ok = False
            print(f"  ERROR: Number of rows differ (final: {len(final_pd)}, base: {len(base_pd)})")
        else:
            print(f"  Row count check PASSED: {len(final_pd)} rows.")
        
        # Check that the composite key (cve, date) and epss values are identical
        comparison = final_pd[['cve', 'date', 'epss']].equals(base_pd[['cve', 'date', 'epss']])
        if not comparison:
            integrity_ok = False
            print("  ERROR: Mismatch found in composite key or epss values!")
        else:
            print("  Composite key and epss check PASSED for this CVE.")
            
        # Optionally, you can extend checks to additional module columns here if desired.
        # For example, if a module column is in final_pd, you might compare it with an expected value.
    
    if integrity_ok:
        print("\nAll selected CVE time series passed the integrity checks!")
    else:
        print("\nIntegrity checks encountered issues with some CVEs!")
    return integrity_ok

def normalize_series(series):
    """
    Normalize a pandas Series to range [0, 1].
    If the series is constant, return 0.5 for all values.
    """
    if series.max() - series.min() == 0:
        return series.apply(lambda x: 0.5)
    return (series - series.min()) / (series.max() - series.min())

def plot_time_series(sample_cves, final_df, numeric_modules):
    """
    For each of the selected CVEs, plot the time series of epss and additional numeric features.
    For additional modules, normalize the series to the range [0, 1] before plotting.
    All plots are displayed in a master plot with 10 subplots.
    """
    num_cves = len(sample_cves)
    
    # Create a figure with subplots; here we arrange them in a grid, e.g., 5 rows x 2 cols.
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20), sharex=False)
    axes = axes.flatten()
    
    for i, cve in enumerate(sample_cves):
        # Filter final_df for the CVE and convert to pandas (only a small time series per CVE)
        cve_df = final_df.filter(col("cve") == cve).orderBy("date")
        cve_pd = cve_df.toPandas()
        cve_pd['date'] = pd.to_datetime(cve_pd['date'])
        
        ax = axes[i]
        ax.plot(cve_pd['date'], cve_pd['epss'], label="epss", linewidth=2)
        
        # Plot additional numeric modules, normalizing each to [0,1]
        for module in numeric_modules:
            # Check if the column exists; note that depending on your naming convention
            # it might be prefixed (e.g., "mock" becomes "mock_<col>") but here we assume numeric_modules are the column names.
            if module in cve_pd.columns:
                norm_series = normalize_series(cve_pd[module])
                ax.plot(cve_pd['date'], norm_series, label=module, linestyle="--")
            else:
                print(f"Warning: Column '{module}' not found for CVE {cve} in the final dataset.")
        
        ax.set_title(f"CVE: {cve}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Value")
        ax.legend(fontsize='small')
    
    plt.tight_layout()
    plt.show()

def main(modules, numeric_modules):
    """
    Main function for quick visualization.
    modules: a list of module names used in generating the final dataset (e.g., ['epss', 'mock']).
    numeric_modules: a list of column names (besides epss) to be plotted (e.g., ['mock']).
    """
    spark = get_spark_session()
    
    # Define paths
    final_path = os.path.join('data', 'full_db', 'processed', 'final_full_data.parquet')
    base_path = os.path.join('data', 'epss', 'processed', 'epss_processed.parquet')
    
    # Print basic information about the datasets
    final_df, base_df = print_dataset_summary(spark, final_path, base_path)
    
    # Select 10 sample CVEs (5 that never exceed epss > 0.7, 5 that do)
    sample_cves = select_sample_cves(final_df, epss_threshold=0.7, num_per_group=5)
    
    # Check the integrity of the time series for the selected CVEs
    _ = check_time_series_integrity(spark, sample_cves, final_df, base_df, modules)
    
    # Plot the time series for these CVEs.
    # Note: The plotting uses data from final_df so it visualizes the merged feature columns as well.
    plot_time_series(sample_cves, final_df, numeric_modules)
    
    spark.stop()

if __name__ == '__main__':
    # Example of calling the main function:
    # modules used in the pipeline (base module included)
    modules = ['epss', 'mock']
    # Columns (from the additional modules) which are numeric and should be normalized & plotted.
    numeric_modules = ['mock']  # You can add more column names if needed.
    main(modules, numeric_modules)
