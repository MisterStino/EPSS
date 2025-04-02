import os
import time
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyspark.sql import functions as F
from pyspark.sql import types as T
from t3_spark.session import get_spark_session

def plot_age_since_publication(
    input_parquet='data/full_db/processed/final_full_data_parquet',
    output_folder='models/eda/figs/pooled_age',
    do_quartiles=True
):
    """
    Reads a large multi-entity time-series dataset of (cve, date, epss).
    Aligns them by 'day_since_pub': the days from the earliest date 
    that each CVE appears. Then aggregates across all CVEs for each day_since_pub
    to produce a line chart that shows how the typical EPSS evolves as CVEs age.
    
    Steps:
      1) For each cve => min(date) as pub_date
      2) Join => each row knows pub_date => define day_since_pub = datediff(date, pub_date)
      3) groupBy day_since_pub => aggregator (mean epss, optionally quartiles)
      4) Convert to Pandas => line chart x=day_since_pub, y= aggregator
      
    Produces an image 'eda_age_since_pub.png' in the output_folder.
    """

    # Make sure folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Start Spark
    spark = get_spark_session()
    
    # Read the dataset
    df = spark.read.parquet(input_parquet)
    
    # Cast columns properly
    df = (df
          .withColumn("cve", F.col("cve").cast(T.StringType()))
          .withColumn("date", F.col("date").cast(T.DateType()))
          .withColumn("epss", F.col("epss").cast(T.DoubleType()))
         )
    
    # 1) For each cve => min(date)
    cve_pub_df = (
        df.groupBy("cve")
          .agg(F.min("date").alias("pub_date"))
    )
    
    # 2) Join => define day_since_pub
    df_joined = df.join(cve_pub_df, on="cve", how="inner")
    df_joined = df_joined.withColumn(
        "day_since_pub",
        F.datediff(F.col("date"), F.col("pub_date"))
    )
    
    # 3) groupBy day_since_pub => aggregator
    if do_quartiles:
        # We'll do approximate percentiles plus a mean
        df_agg = (
            df_joined
            .groupBy("day_since_pub")
            .agg(
                F.mean("epss").alias("avg_epss"),
                F.expr("percentile_approx(epss, 0.25)").alias("q25"),
                F.expr("percentile_approx(epss, 0.5)").alias("median"),
                F.expr("percentile_approx(epss, 0.75)").alias("q75"),
                F.count("*").alias("count_rows")
            )
        )
    else:
        df_agg = (
            df_joined
            .groupBy("day_since_pub")
            .agg(F.mean("epss").alias("avg_epss"))
        )
    
    df_pd = df_agg.orderBy("day_since_pub").toPandas()
    
    # 4) Plot
    plt.figure(figsize=(10,6))
    plt.plot(df_pd["day_since_pub"], df_pd["avg_epss"], marker='o', label="Mean EPSS", linewidth=1)
    
    if do_quartiles:
        # optional: plot quartiles as fill or separate lines
        plt.plot(df_pd["day_since_pub"], df_pd["q25"], linestyle='--', label="Q25")
        plt.plot(df_pd["day_since_pub"], df_pd["median"], linestyle='--', label="Median")
        plt.plot(df_pd["day_since_pub"], df_pd["q75"], linestyle='--', label="Q75")
    
    plt.title("EPSS vs. Age Since Publication (All CVEs)")
    plt.xlabel("Days Since CVE Publication")
    plt.ylabel("EPSS")
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(output_folder, "eda_age_since_pub.png")
    plt.savefig(outpath)
    plt.close()
    
    print(f"Age-since-publication EDA plot saved: {outpath}")
    
    time.sleep(2)
    spark.stop()

if __name__ == "__main__":
    plot_age_since_publication(
        input_parquet='data/full_db/processed/final_full_data_parquet',
        output_folder='models/eda/figs/cohorts/pooled_age',
        do_quartiles=True
    )
