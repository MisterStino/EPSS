import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from t3_spark.session import get_spark_session
from models.eda.scripts.utils import add_pub_cohort_label

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
    
    For each CVE, computes the publication date (min(date)) and then, using the global minimum
    publication date across all CVEs, calculates the days from that global minimum to the CVE's
    pub_date. The cohort index is computed as:
        cohort_index = floor((pub_date - global_min_pub_date) / days_per_cohort) + 1,
    and then labeled as "C1", "C2", etc.
    
    Returns a DataFrame with columns: (cve, pub_cohort, pub_date).
    """
    # Compute publication date per CVE
    w = Window.partitionBy("cve")
    df_pub = df.groupBy("cve").agg(F.min("date").alias("pub_date"))
    # Determine global minimum publication date across all CVEs
    global_min_pub_date = df_pub.agg(F.min("pub_date").alias("global_min")).collect()[0]["global_min"]
    global_min_pub_date_lit = F.to_date(F.lit(global_min_pub_date))
    # Compute days from global minimum
    df_pub = df_pub.withColumn("days_from_global_min", F.datediff(F.col("pub_date"), global_min_pub_date_lit))
    # Compute the cohort index and label
    df_pub = df_pub.withColumn("cohort_index", F.floor(F.col("days_from_global_min") / days_per_cohort) + 1)
    df_pub = df_pub.withColumn("pub_cohort", F.concat(F.lit("C"), F.col("cohort_index").cast(T.StringType())))
    return df_pub.select("cve", "pub_cohort", "pub_date")

def create_cohort_groupedbars_and_meanline_by_day_pubcohort(
    input_parquet="data/full_db/processed/final_full_data.parquet",
    output_folder="models/eda/figs/dayofweek_pubcohorts",
    days_per_cohort=90,
    maturity_threshold=30,
    threshold_for_prop=0.7
):
    """
    For each day-of-week, aggregates data by publication-based cohort:
      - Computes the mean EPS score (plotted as a line on the left y-axis).
      - Computes the proportion of records with EPS > threshold (plotted as grouped bars on the right y-axis).
    The line graphs overlay the bars.
    
    The output is rendered in high resolution (4K) with a small legend.
    """
    os.makedirs(output_folder, exist_ok=True)
    spark = get_spark_session()
    
    # 1. Read dataset and cast columns.
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    # 2. Compute publication date per CVE (pub_date)
    w = Window.partitionBy("cve")
    df = df.withColumn("pub_date", F.min("date").over(w))
    
    # 3. Compute age as days since publication.
    df = df.withColumn("day_since_pub", F.datediff(F.col("date"), F.col("pub_date")))
    
    # 4. (Optional) Filter out records where CVEs are too new.
    if maturity_threshold is not None:
        df = df.filter(F.col("day_since_pub") >= maturity_threshold)
    
    # 5. Compute publication-based cohorts.
    # Use the function to assign cohorts based on the CVE's publication date.
    df_pub_cohorts = add_pub_cohort_label(df, days_per_cohort=days_per_cohort)
    df = df.join(df_pub_cohorts, on="cve", how="left")
    
    # 6. Extract the day-of-week from the record's date.
    df = df.withColumn("day_of_week", F.date_format(F.col("date"), "EEEE"))
    
    # 7. Aggregate for the proportion per day-of-week and publication cohort.
    df_prop = df.groupBy("day_of_week", "pub_cohort").agg(
        (F.sum(F.when(F.col("epss") > threshold_for_prop, 1).otherwise(0)) / F.count("*")).alias("prop_above")
    )
    
    # 8. Aggregate for the cohort-specific mean EPS score per day-of-week.
    df_mean = df.groupBy("day_of_week", "pub_cohort").agg(
        F.mean("epss").alias("mean_epss")
    )
    
    # 9. Collect the aggregations into Pandas DataFrames.
    pd_prop = df_prop.toPandas()
    pd_mean = df_mean.toPandas()
    
    if pd_prop.empty or pd_mean.empty:
        print("No data to plot; check your filters or dataset.")
        spark.stop()
        return
    
    # 10. Define a fixed day-of-week order.
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # 11. Pivot the proportions: rows = day_of_week, columns = pub_cohort.
    pivot_prop = pd_prop.pivot(index="day_of_week", columns="pub_cohort", values="prop_above").fillna(0)
    pivot_prop = pivot_prop.reindex(dow_order).fillna(0)
    
    # Pivot the mean EPS: rows = day_of_week, columns = pub_cohort.
    pivot_mean = pd_mean.pivot(index="day_of_week", columns="pub_cohort", values="mean_epss").fillna(0)
    pivot_mean = pivot_mean.reindex(dow_order).fillna(0)
    
    # 12. Set up x-axis positions for the 7 days.
    x_positions = list(range(len(dow_order)))  # 0 through 6
    num_cohorts = len(pivot_prop.columns)
    bar_width = 0.7 / max(1, num_cohorts)
    
    # 13. Use a color palette so that each cohort uses the same color for line and bar.
    cohort_labels = sorted(list(pivot_prop.columns), key=lambda x: int(x.replace("C", "")))
    palette = sns.color_palette("Set2", n_colors=len(cohort_labels))
    color_map = {cohort: palette[i] for i, cohort in enumerate(cohort_labels)}
    
    # 14. Create a dual-axis figure with high resolution (4K).
    # For 4K (3840 x 2160), use figsize=(16,9) at dpi=240.
    fig, ax1 = plt.subplots(figsize=(16,9), dpi=240)
    
    # Left y-axis: Plot a line for each cohort's mean EPS.
    ax1.set_ylabel("Mean EPS Score", color="C0", fontsize=14)
    ax1.set_xlabel("Day of Week", fontsize=14)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(dow_order, rotation=45, fontsize=12)
    for cohort in cohort_labels:
        y_vals = [pivot_mean.loc[dow, cohort] for dow in dow_order]
        ax1.plot(
            x_positions,
            y_vals,
            marker="o",
            color=color_map[cohort],
            label=f"Mean EPS ({cohort})",
            zorder=5  # Ensure lines appear on top of bars
        )
    
    # Right y-axis: Plot grouped bars for the proportion above threshold.
    ax2 = ax1.twinx()
    ax2.set_ylabel(f"Prop (EPS > {threshold_for_prop})", color="C1", fontsize=14)
    max_prop_val = pivot_prop.max().max()
    # Set a dynamic upper bound (with a margin) for the proportion axis.
    upper_bound = max_prop_val + (0.1 * max_prop_val if max_prop_val > 0 else 0.01)
    ax2.set_ylim(0, min(1.0, upper_bound))
    
    for i, cohort in enumerate(cohort_labels):
        # Shift x positions for grouped bars for this cohort.
        x_shift = [p + i * bar_width - (bar_width * (len(cohort_labels) - 1) / 2) for p in x_positions]
        y_data = [pivot_prop.loc[dow, cohort] for dow in dow_order]
        ax2.bar(
            x_shift,
            y_data,
            width=bar_width,
            color=color_map[cohort],
            alpha=0.7,  # Set bar transparency so lines are visible
            label=f"Prop ({cohort})",
            zorder=1
        )
    
    # Combine legends from both axes with a smaller font size.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", prop={'size': 8})
    
    plt.title("Day-of-Week: Publication Cohort-Specific Mean EPS & Proportion Above Threshold", fontsize=16)
    plt.tight_layout()
    
    outpath = os.path.join(output_folder, "dayofweek_pubcohort_mean_line_and_grouped_bars.png")
    plt.savefig(outpath, dpi=240)
    plt.close()
    
    print(f"Saved final plot to: {outpath}")
    spark.stop()
    print("Done. Dual-axis high-resolution plot by day-of-week (publication-based cohorts) generated successfully.")





def create_dual_axis_plot_dayofweek(pd_mean, pd_prop, dow_order, color_map, cohort_labels, outpath, title):
    """
    Creates the dual-axis plot (line for mean EPS and grouped bars for proportion)
    given pivoted Pandas dataframes for mean and proportion.
    """
    x_positions = list(range(len(dow_order)))
    num_cohorts = len(cohort_labels)
    bar_width = 0.7 / max(1, num_cohorts)
    
    fig, ax1 = plt.subplots(figsize=(16,9), dpi=240)
    
    # Left axis: mean EPS line(s)
    ax1.set_ylabel("Mean EPS Score", color="C0", fontsize=14)
    ax1.set_xlabel("Day of Week", fontsize=14)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(dow_order, rotation=45, fontsize=12)
    
    for cohort in cohort_labels:
        y_vals = [pd_mean.loc[dow, cohort] for dow in dow_order]
        ax1.plot(x_positions, y_vals, marker="o", color=color_map[cohort],
                 label=f"Mean EPS ({cohort})", zorder=5)
    
    # Right axis: grouped bars for proportion above threshold
    ax2 = ax1.twinx()
    ax2.set_ylabel("Proportion (EPS > threshold)", color="C1", fontsize=14)
    max_prop_val = pd_prop.max().max()
    upper_bound = max_prop_val + (0.1 * max_prop_val if max_prop_val > 0 else 0.01)
    ax2.set_ylim(0, min(1.0, upper_bound))
    
    for i, cohort in enumerate(cohort_labels):
        x_shift = [p + i * bar_width - (bar_width * (len(cohort_labels) - 1) / 2) for p in x_positions]
        y_data = [pd_prop.loc[dow, cohort] for dow in dow_order]
        ax2.bar(x_shift, y_data, width=bar_width, color=color_map[cohort],
                alpha=0.7, label=f"Prop ({cohort})", zorder=1)
    
    # Combine legends with small font size
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", prop={'size': 8})
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()
    print(f"Saved final plot to: {outpath}")

def create_cohort_plots_by_three_months_dayofweek(
    input_parquet="data/full_db/processed/final_full_data.parquet",
    output_folder="models/eda/figs/dayofweek_3month_windows",
    start_date_str="2022-02-04",
    window_days=90,           # Each window covers 90 days.
    maturity_threshold=30,
    threshold_for_prop=0.7
):
    """
    Creates dual-axis day-of-week plots (mean EPS line and proportion bar plots) for each 90-day window.
    
    The publication-based cohorts are computed using the global minimum pub_date.
    Then, for each 90-day window (based on the record's date), the data is filtered and aggregated 
    by day-of-week (Monday-Sunday) and by publication cohort.
    
    Only cohorts that have data in that window are shown.
    """
    os.makedirs(output_folder, exist_ok=True)
    spark = get_spark_session()
    
    # Read dataset and cast columns
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    # Compute publication date and day_since_pub for each CVE
    w = Window.partitionBy("cve")
    df = df.withColumn("pub_date", F.min("date").over(w))
    df = df.withColumn("day_since_pub", F.datediff(F.col("date"), F.col("pub_date")))
    
    # Filter out records with day_since_pub below maturity_threshold
    if maturity_threshold is not None:
        df = df.filter(F.col("day_since_pub") >= maturity_threshold)
    
    # Compute publication-based cohorts (this is global over the dataset)
    df_pub_cohorts = add_pub_cohort_label(df, days_per_cohort=90)  # Use 90 days for publication cohorts
    df = df.join(df_pub_cohorts, on="cve", how="left")
    
    # Extract day_of_week from record's date
    df = df.withColumn("day_of_week", F.date_format(F.col("date"), "EEEE"))
    
    # Determine overall min and max date in the dataset (in record 'date')
    overall_min = df.agg(F.min("date")).collect()[0][0]
    overall_max = df.agg(F.max("date")).collect()[0][0]
    
    # Convert overall_min and overall_max to Python datetime objects
    overall_min_dt = pd.to_datetime(overall_min.isoformat())
    overall_max_dt = pd.to_datetime(overall_max.isoformat())
    
    # Initialize current window start and end dates
    current_start = overall_min_dt
    current_end = current_start + timedelta(days=window_days)
    
    # Fixed day-of-week order for plots
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # For consistent color coding across plots, get the global publication cohorts (once)
    # (Collect distinct publication cohorts from the dataset)
    cohort_rows = df.select("pub_cohort").distinct().collect()
    all_cohorts = sorted([row["pub_cohort"] for row in cohort_rows if row["pub_cohort"] is not None],
                         key=lambda x: int(x.replace("C", "")))
    palette = sns.color_palette("Set2", n_colors=len(all_cohorts))
    color_map = {cohort: palette[i] for i, cohort in enumerate(all_cohorts)}
    
    # Loop over 90-day windows
    window_number = 1
    while current_start < overall_max_dt:
        # Define the window label (e.g., "2022-02-04_to_2022-05-05")
        window_label = f"{current_start.strftime('%Y-%m-%d')}_to_{(current_end - timedelta(days=1)).strftime('%Y-%m-%d')}"
        print(f"Processing window {window_number}: {window_label}")
        
        # Filter the dataframe for the current window (record's date falls within window)
        df_window = df.filter((F.col("date") >= F.lit(current_start.strftime("%Y-%m-%d"))) &
                              (F.col("date") < F.lit(current_end.strftime("%Y-%m-%d"))))
        
        # Only proceed if there is data in the window
        if df_window.rdd.isEmpty():
            print(f"No data in window {window_label}; skipping.")
        else:
            # Aggregate per day-of-week and publication cohort
            df_prop = df_window.groupBy("day_of_week", "pub_cohort").agg(
                (F.sum(F.when(F.col("epss") > threshold_for_prop, 1).otherwise(0)) / F.count("*")).alias("prop_above")
            )
            df_mean = df_window.groupBy("day_of_week", "pub_cohort").agg(
                F.mean("epss").alias("mean_epss")
            )
            
            pd_prop = df_prop.toPandas()
            pd_mean = df_mean.toPandas()
            
            if pd_prop.empty or pd_mean.empty:
                print(f"No aggregated data in window {window_label}; skipping.")
            else:
                # Pivot the aggregations to have rows = day_of_week, columns = pub_cohort.
                pivot_prop = pd_prop.pivot(index="day_of_week", columns="pub_cohort", values="prop_above").fillna(0)
                pivot_prop = pivot_prop.reindex(dow_order).fillna(0)
                pivot_mean = pd_mean.pivot(index="day_of_week", columns="pub_cohort", values="mean_epss").fillna(0)
                pivot_mean = pivot_mean.reindex(dow_order).fillna(0)
                
                # Get the cohort labels present in this window (sorted)
                cohorts_in_window = sorted(list(pivot_prop.columns), key=lambda x: int(x.replace("C", "")))
                
                # Create a dual-axis high-resolution plot (4K)
                fig, ax1 = plt.subplots(figsize=(16,9), dpi=240)
                ax1.set_ylabel("Mean EPS Score", color="C0", fontsize=14)
                ax1.set_xlabel("Day of Week", fontsize=14)
                ax1.set_xticks(list(range(len(dow_order))))
                ax1.set_xticklabels(dow_order, rotation=45, fontsize=12)
                
                # Plot line graphs for each cohort’s mean EPS
                for cohort in cohorts_in_window:
                    y_vals = [pivot_mean.loc[dow, cohort] for dow in dow_order]
                    ax1.plot(list(range(len(dow_order))), y_vals, marker="o",
                             color=color_map.get(cohort, "grey"),
                             label=f"Mean EPS ({cohort})", zorder=5)
                
                # Right y-axis: grouped bars for the proportion above threshold
                ax2 = ax1.twinx()
                ax2.set_ylabel(f"Prop (EPS > {threshold_for_prop})", color="C1", fontsize=14)
                max_prop_val = pivot_prop.max().max()
                upper_bound = max_prop_val + (0.1 * max_prop_val if max_prop_val > 0 else 0.01)
                ax2.set_ylim(0, min(1.0, upper_bound))
                
                for i, cohort in enumerate(cohorts_in_window):
                    # Shift x positions for grouped bars
                    x_shift = [p + i * (0.7 / max(1, len(cohorts_in_window))) - (0.7 * (len(cohorts_in_window) - 1) / (2 * len(cohorts_in_window)))
                               for p in range(len(dow_order))]
                    y_data = [pivot_prop.loc[dow, cohort] for dow in dow_order]
                    ax2.bar(x_shift, y_data, width=0.7 / max(1, len(cohorts_in_window)),
                            color=color_map.get(cohort, "grey"), alpha=0.7, label=f"Prop ({cohort})", zorder=1)
                
                # Combine legends (smaller font size)
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", prop={'size': 8})
                
                plt.title(f"Day-of-Week: Publication Cohort-Specific Mean EPS & Proportion (Window: {window_label})", fontsize=16)
                plt.tight_layout()
                outpath = os.path.join(output_folder, f"dayofweek_3monthwindow_{window_label}_pubcohort.png")
                plt.savefig(outpath, dpi=240)
                plt.close()
                print(f"Saved plot for window {window_label} to: {outpath}")
        
        # Move to next window
        current_start = current_end
        current_end = current_start + timedelta(days=window_days)
        window_number += 1
    
    spark.stop()
    print("Done. All day-of-week plots for 3-month windows have been generated.")



if __name__ == "__main__":
    # create_cohort_groupedbars_and_meanline_by_day_pubcohort(
    #     input_parquet="data/full_db/processed/final_full_data_parquet",
    #     output_folder="models/eda/figs/age_cohort_stackedbar",
    #     days_per_cohort=90,          # 90 days = 3-month cohorts
    #     maturity_threshold=10,        # Only consider CVEs older than 30 days
    #     threshold_for_prop=0.7        # EPS threshold for computing the proportion
    # )
    create_cohort_plots_by_three_months_dayofweek(
        input_parquet='data/full_db/processed/final_full_data_parquet',
        output_folder="models/eda/figs/age_cohort_stackedbar_by_3months",
        maturity_threshold=10,        # only consider CVEs older than 30 days
        threshold_for_prop=0.7        # EPS threshold for computing the proportion
    )
