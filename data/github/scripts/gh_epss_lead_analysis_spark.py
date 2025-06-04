#!/usr/bin/env python3
"""Spark-based analysis: for CVEs with any GitHub activity, check if/when they cross EPSS > 0.7 and compute lead times."""
import sys
import os
from functools import reduce
from operator import add
# Add project root to PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from t3_spark.session import get_spark_session
from pyspark.sql.functions import upper, col, min as spark_min, datediff


def main():
    spark = get_spark_session("GH-to-EPSS Lead Analysis")
    # Paths
    tool_dir = os.path.dirname(__file__)
    gh_csv = os.path.join(tool_dir, '..', 'raw', 'github_bq.csv')
    epss_parquet = os.path.abspath(os.path.join(tool_dir, '..', '..', 'epss', 'epss_parquet', 'epss_all.parquet'))

    # Load GitHub data
    gh_df = (
        spark.read
            .option("header","true")
            .option("inferSchema","true")
            .csv(gh_csv)
            .withColumn("cve_id", upper(col("cve_id")))
            .withColumn("date", col("date").cast("date"))
    )
    # Identify event columns and compute total_events
    event_cols = [c for c in gh_df.columns if c not in ["cve_id","date"]]
    gh_df = gh_df.withColumn("total_events", reduce(add, [col(c) for c in event_cols]))

    # First GitHub activity date per CVE
    gh_signal = (
        gh_df.filter(col("total_events") > 0)
             .groupBy("cve_id")
             .agg(spark_min("date").alias("first_gh_date"))
    )

    # Load EPSS data
    epss_df = (
        spark.read
            .parquet(epss_parquet)
            .withColumnRenamed("cve","cve_id")
            .withColumn("cve_id", upper(col("cve_id")))
            .withColumn("date", col("date").cast("date"))
    )
    # First EPSS threshold cross per CVE
    epss_signal = (
        epss_df.filter(col("epss") > 0.7)
               .groupBy("cve_id")
               .agg(spark_min("date").alias("first_epss_date"))
    )

    # Join on CVE
    joined = gh_signal.join(epss_signal, on="cve_id", how="left")
    total_gh = joined.count()
    have_epss = joined.filter(col("first_epss_date").isNotNull()).count()
    no_epss = total_gh - have_epss
    print(f"Total CVEs with GitHub activity: {total_gh}")
    print(f"Of these, {have_epss} eventually cross EPSS>0.7, {no_epss} do not.")

    # Compute lead times for those that cross
    lead = (
        joined.filter(col("first_epss_date").isNotNull())
              .withColumn("lead_days", datediff(col("first_epss_date"), col("first_gh_date")))
    )
    # Summary stats
    lead.describe("lead_days").show()
    # Counts by sign
    plus = lead.filter(col("lead_days") > 0).count()
    zero = lead.filter(col("lead_days") == 0).count()
    minus = lead.filter(col("lead_days") < 0).count()
    print(f"Lead positive (>0): {plus}, zero: {zero}, negative: {minus}")

    spark.stop()

if __name__ == '__main__':
    main() 