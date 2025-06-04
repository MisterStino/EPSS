#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from t3_spark.session import get_spark_session
from pyspark.sql.functions import upper, col, min as spark_min, max as spark_max

def main():
    spark = get_spark_session("CountThresholdCVEs")
    gh_csv = os.path.join(os.path.dirname(__file__), '..', 'raw', 'github_bq.csv')
    epss_parquet = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'epss', 'epss_parquet', 'epss_all.parquet'))

    # Load GitHub and EPSS
    gh_df = (
        spark.read
             .option("header","true")
             .option("inferSchema","true")
             .csv(gh_csv)
             .withColumn("cve_id", upper(col("cve_id")))
             .withColumn("date", col("date").cast("date"))
    )
    epss_df = (
        spark.read
             .parquet(epss_parquet)
             .withColumnRenamed("cve","cve_id")
             .withColumn("cve_id", upper(col("cve_id")))
             .withColumn("date", col("date").cast("date"))
    )
    joined = gh_df.join(epss_df, on=["cve_id","date"], how="inner")

    # Compute min and max epss per CVE
    stats_df = joined.groupBy("cve_id").agg(
        spark_min("epss").alias("min_epss"),
        spark_max("epss").alias("max_epss")
    )
    # Filter CVEs with initial <0.3 and eventual >0.7
    filtered = stats_df.filter((col("min_epss") < 0.3) & (col("max_epss") > 0.7))
    count = filtered.count()
    print(f"{count} CVEs have min epss < 0.3 and max epss > 0.7")
    spark.stop()

if __name__ == '__main__':
    main() 