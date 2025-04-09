import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    For each CVE, compute the publication date (min(date)) and then, using that date,
    calculate the number of days from a fixed reference (here we use the minimum publication date 
    across all CVEs) to the CVE's pub_date. Then, assign a cohort label:
      Cohort = floor((pub_date - global_min_pub_date) / days_per_cohort) + 1,
      and label as "C1", "C2", etc.
    
    This function returns a DataFrame with columns (cve, pub_date, pub_cohort).
    """
    # For each CVE, compute its publication date.
    w = Window.partitionBy("cve")
    df_pub = df.groupBy("cve").agg(F.min("date").alias("pub_date"))
    
    # Determine the global minimum publication date across all CVEs.
    global_min_pub_date = df_pub.agg(F.min("pub_date").alias("global_min")).collect()[0]["global_min"]
    global_min_pub_date_lit = F.to_date(F.lit(global_min_pub_date))
    
    # Compute the days from the global minimum publication date.
    df_pub = df_pub.withColumn("days_from_global_min", F.datediff(F.col("pub_date"), global_min_pub_date_lit))
    
    # Compute the cohort index and label.
    df_pub = df_pub.withColumn("cohort_index", F.floor(F.col("days_from_global_min") / days_per_cohort) + 1)
    df_pub = df_pub.withColumn("pub_cohort", F.concat(F.lit("C"), F.col("cohort_index").cast(T.StringType())))
    
    return df_pub.select("cve", "pub_cohort", "pub_date")

def create_cohort_groupedbars_and_meanline_by_month_pubcohort(
    input_parquet="data/full_db/processed/final_full_data.parquet",
    output_folder="models/eda/figs/monthly_pubcohorts",
    days_per_cohort=90,             # Cohort window in days (3 months)
    maturity_threshold=30,          # Only consider records for CVEs older than this many days from their pub_date
    threshold_for_prop=0.7          # EPS threshold for computing the proportion above threshold
):
    """
    For each calendar month (by full month name), aggregates records by the publication cohort of the CVE.
    For each month and publication cohort, computes:
      - Mean EPS score (plotted as a line on the left y-axis).
      - Proportion of EPS > threshold (plotted as grouped bars on the right y-axis).
    The line graphs overlay the bars.
    
    The figure is rendered in high resolution (4K) with a smaller legend.
    
    This version defines cohorts based on each CVE's publication date.
    """
    os.makedirs(output_folder, exist_ok=True)
    spark = get_spark_session()
    
    # 1. Read dataset and cast columns.
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    # 2. For each CVE, compute the publication date.
    w = Window.partitionBy("cve")
    df = df.withColumn("pub_date", F.min("date").over(w))
    
    # 3. Filter out records for CVEs that are "too new" based on their age.
    # Compute each record's age relative to its CVE's publication date.
    df = df.withColumn("day_since_pub", F.datediff(F.col("date"), F.col("pub_date")))
    if maturity_threshold is not None:
        df = df.filter(F.col("day_since_pub") >= maturity_threshold)
    
    # 4. Compute publication cohorts based on each CVE's publication date.
    #   This groups CVEs that first appeared within a 3-month (or specified) window.
    df_pub_cohorts = add_pub_cohort_label(df, days_per_cohort=days_per_cohort)
    
    # 5. Join the publication cohort info back to the main dataset.
    df = df.join(df_pub_cohorts, on="cve", how="left")
    
    # 6. Extract month from the record's date.
    df = df.withColumn("month", F.date_format(F.col("date"), "MMMM"))
    
    # 7. Aggregation for the proportion of records above threshold, per month and publication cohort.
    df_prop = df.groupBy("month", "pub_cohort").agg(
        (F.sum(F.when(F.col("epss") > threshold_for_prop, 1).otherwise(0)) / F.count("*")).alias("prop_above")
    )
    
    # 8. Aggregation for the cohort-specific mean EPS score per month.
    df_mean = df.groupBy("month", "pub_cohort").agg(
        F.mean("epss").alias("mean_epss")
    )
    
    # Collect these aggregations into Pandas DataFrames.
    pd_prop = df_prop.toPandas()
    pd_mean = df_mean.toPandas()
    
    if pd_prop.empty or pd_mean.empty:
        print("No data to plot; check your filters or dataset.")
        spark.stop()
        return
    
    # 9. Define fixed month order.
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    
    # Pivot the proportions: rows = month, columns = pub_cohort, values = prop_above.
    pivot_prop = pd_prop.pivot(index="month", columns="pub_cohort", values="prop_above").fillna(0)
    pivot_prop = pivot_prop.reindex(month_order).fillna(0)
    
    # Pivot the mean EPS: rows = month, columns = pub_cohort, values = mean_epss.
    pivot_mean = pd_mean.pivot(index="month", columns="pub_cohort", values="mean_epss").fillna(0)
    pivot_mean = pivot_mean.reindex(month_order).fillna(0)
    
    # 10. Set up x-axis positions for months.
    x_positions = list(range(len(month_order)))  # indices 0 to 11
    num_cohorts = len(pivot_prop.columns)
    bar_width = 0.7 / max(1, num_cohorts)
    
    # 11. Use a color palette so that each cohort uses the same color for its bar and line.
    cohort_labels = sorted(list(pivot_prop.columns), key=lambda x: int(x.replace("C", "")))
    palette = sns.color_palette("Set2", n_colors=len(cohort_labels))
    color_map = {cohort: palette[i] for i, cohort in enumerate(cohort_labels)}
    
    # 12. Create dual-axis plot with high resolution (4K).
    # For 4K output (3840 x 2160), we set figsize=(16,9) with dpi=240.
    fig, ax1 = plt.subplots(figsize=(16,9), dpi=240)
    
    # Left y-axis: Plot a line for each cohort's mean EPS.
    ax1.set_ylabel("Mean EPS Score", color="C0", fontsize=14)
    ax1.set_xlabel("Month", fontsize=14)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(month_order, rotation=45, fontsize=12)
    
    for cohort in cohort_labels:
        y_vals = [pivot_mean.loc[month, cohort] for month in month_order]
        ax1.plot(
            x_positions,
            y_vals,
            marker="o",
            color=color_map[cohort],
            label=f"Mean EPS ({cohort})",
            zorder=5  # ensure lines overlay the bars
        )
    
    # Right y-axis: Plot grouped bars for the proportion above threshold.
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"Prop (EPS > {threshold_for_prop})", color="C1", fontsize=14)
    max_prop_val = pivot_prop.max().max()
    upper_bound = max_prop_val + (0.1 * max_prop_val if max_prop_val > 0 else 0.01)
    ax2.set_ylim(0, min(1.0, upper_bound))
    
    for i, cohort in enumerate(cohort_labels):
        # For each cohort, shift the x positions for grouped bars.
        x_shift = [p + i * bar_width - (bar_width * (len(cohort_labels) - 1) / 2) for p in x_positions]
        y_data = [pivot_prop.loc[month, cohort] for month in month_order]
        ax2.bar(
            x_shift,
            y_data,
            width=bar_width,
            color=color_map[cohort],
            alpha=0.7,  # set bar transparency so the overlaid lines are visible
            label=f"Prop ({cohort})",
            zorder=1
        )
    
    # Combine legends from both axes with a smaller font size.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", prop={'size': 8})
    
    plt.title("Month-of-Year: Publication Cohort-Specific Mean EPS & Proportion Above Threshold", fontsize=16)
    plt.tight_layout()
    
    outpath = os.path.join(output_folder, f"month_cohorts_pub_based_mean_line_and_grouped_bars.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    
    print(f"Saved final plot to: {outpath}")
    spark.stop()
    print("Done. Dual-axis high-resolution plot by month (publication-based cohorts) generated successfully.")

if __name__ == "__main__":
    create_cohort_groupedbars_and_meanline_by_month_pubcohort(
        input_parquet="data/full_db/processed/final_full_data_parquet",
        output_folder="models/eda/figs/age_cohort_stackedbar",
        days_per_cohort=90,          # half-year cohorts; change to 90 for quarterly
        maturity_threshold=10,        # only consider CVEs older than 30 days
        threshold_for_prop=0.7        # EPS threshold for computing the proportion
    )
