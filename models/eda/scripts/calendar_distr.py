import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window
from t3_spark.session import get_spark_session



def cast_common_columns(df):
    if 'cve' in df.columns:
        df = df.withColumn('cve', F.col('cve').cast(T.StringType()))
    if 'date' in df.columns:
        df = df.withColumn('date', F.col('date').cast(T.DateType()))
    if 'epss' in df.columns:
        df = df.withColumn('epss', F.col('epss').cast(T.DoubleType()))
    return df

def add_cohort_label(df, start_date_str="2022-02-04", days_per_cohort=90):
    start_date_lit = F.to_date(F.lit(start_date_str))
    df = df.withColumn("days_since_start", F.datediff(F.col("date"), start_date_lit))
    df = df.withColumn("cohort_index", F.floor(F.col("days_since_start") / days_per_cohort) + 1)
    df = df.withColumn("cohort_label", F.concat(F.lit("C"), F.col("cohort_index").cast(T.StringType())))
    return df

def create_plots_dayofweek_mean_iqr_and_prop(
    input_parquet="data/full_db/processed/final_full_data_parquet",
    output_folder="models/eda/figs/dayofweek_cohorts",
    start_date_str="2022-02-04",
    days_per_cohort=180,
    maturity_threshold=30,
    threshold_for_prop=0.7
):
    """
    Creates day-of-week mean + IQR plots with a line for proportion > threshold, 
    with a custom or dynamic y-axis range for the proportion.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    spark = get_spark_session()
    df = spark.read.parquet(input_parquet)
    df = cast_common_columns(df)
    
    if maturity_threshold is not None:
        w = Window.partitionBy("cve")
        df = df.withColumn("pub_date", F.min("date").over(w))
        df = df.withColumn("age", F.datediff(F.col("date"), F.col("pub_date")))
        df = df.filter(F.col("age") >= maturity_threshold)
    
    # Add time-cohort label
    df = add_cohort_label(df, start_date_str=start_date_str, days_per_cohort=days_per_cohort)
    # Add day-of-week
    df = df.withColumn("day_of_week", F.date_format(F.col("date"), "EEEE"))
    
    # Distinct cohorts
    cohort_list = [r["cohort_label"] for r in df.select("cohort_label").distinct().collect()]
    cohort_list = sorted(cohort_list, key=lambda x: int(x.replace("C","")))
    
    # day-of-week order
    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    
    for cohort in cohort_list:
        df_cohort = df.filter(F.col("cohort_label") == cohort)
        date_minmax = df_cohort.agg(F.min("date").alias("dmin"), F.max("date").alias("dmax")).collect()[0]
        dmin = date_minmax["dmin"]
        dmax = date_minmax["dmax"]
        
        df_agg = (df_cohort
                  .groupBy("day_of_week")
                  .agg(
                      F.mean("epss").alias("mean_epss"),
                      F.expr("percentile_approx(epss, 0.25)").alias("q25"),
                      F.expr("percentile_approx(epss, 0.75)").alias("q75"),
                      (F.sum(F.when(F.col("epss") > threshold_for_prop, 1).otherwise(0)) /
                       F.count("*")).alias("prop_above")
                  ))
        
        pd_agg = df_agg.toPandas()
        if pd_agg.empty:
            print(f"No data for cohort {cohort}, skipping.")
            continue
        
        dow_map = {dw:i for i,dw in enumerate(dow_order)}
        pd_agg["dow_index"] = pd_agg["day_of_week"].map(dow_map)
        pd_agg.sort_values("dow_index", inplace=True)
        
        fig, ax1 = plt.subplots(figsize=(8,5))
        
        # Plot mean EPSS ± IQR on left axis
        ax1.plot(pd_agg["dow_index"], pd_agg["mean_epss"], marker='o', color='C0', label="Mean EPSS")
        ax1.fill_between(pd_agg["dow_index"], pd_agg["q25"], pd_agg["q75"], color='C0', alpha=0.2, label="IQR (Q25-Q75)")
        ax1.set_xlabel("Day of Week")
        ax1.set_ylabel("EPSS (Mean & IQR)", color='C0')
        ax1.set_xticks(pd_agg["dow_index"])
        ax1.set_xticklabels(pd_agg["day_of_week"], rotation=45)
        
        # Right axis for proportion
        ax2 = ax1.twinx()
        ax2.set_ylabel(f"Proportion > {threshold_for_prop}", color='C1')
        
        # Option A: Hard-coded range e.g. up to 0.05 if you know it never exceeds 0.03
        # ax2.set_ylim(0, 0.05)
        
        # Option B: dynamic range
        max_prop = pd_agg["prop_above"].max()
        # Add a margin
        upper_bound = max_prop + (0.1 * max_prop if max_prop>0 else 0.01)
        ax2.set_ylim(0, min(1.0, upper_bound))
        
        ax2.plot(pd_agg["dow_index"], pd_agg["prop_above"], marker='s', color='C1', label=f"Prop > {threshold_for_prop}")
        
        # Title
        dmin_str = str(dmin) if dmin else ""
        dmax_str = str(dmax) if dmax else ""
        date_range_text = f"({dmin_str} to {dmax_str})"
        plt.title(f"EPSS by Day-of-Week: {cohort} {date_range_text}")
        
        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1+lines_2, labels_1+labels_2, loc='upper left')
        
        plt.tight_layout()
        
        fname = f"dayofweek_mean_iqr_prop_{cohort}.png"
        outpath = os.path.join(output_folder, fname)
        plt.savefig(outpath)
        plt.close()
        
        print(f"Saved figure for cohort {cohort} => {outpath}")
    
    spark.stop()
    print("Done. All day-of-week mean+IQR + proportion>0.7 plots are saved.")

if __name__ == "__main__":
    create_plots_dayofweek_mean_iqr_and_prop(
        input_parquet="data/full_db/processed/final_full_data_parquet",
        output_folder="models/eda/figs/seasonality_quarterly",
        start_date_str="2022-02-04",
        days_per_cohort=180,  # e.g., for half-year
        maturity_threshold=30,
        threshold_for_prop=0.7
    )
