from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, trim, lower, first, count as spark_count
)

# ----------------------------------------------------------------------
# STEP 1: Spark session
# ----------------------------------------------------------------------
spark = SparkSession.builder \
    .appName("OptimizedEPSSPipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.broadcastTimeout", "3600") \
    .config("spark.local.dir", "D:/spark-temp") \
    .getOrCreate()

# ----------------------------------------------------------------------
# STEP 2: Load & clean Mastodon data
# ----------------------------------------------------------------------
print("=== STEP 2: Loading and cleaning Mastodon data ===")
raw_mastodon = (
    spark.read
         .option("header", True)
         .csv("parquet_preprocessing/merged_mastodon.csv")
         .filter(col("CVE_ID").isNotNull())
)

catalog_df = (
    raw_mastodon
      .withColumnRenamed("CVE_ID", "cve_id")
      .withColumn("cve_id", lower(trim(col("cve_id"))))
      .withColumnRenamed("date_mastodon", "orig_mastodon_date")
      .withColumn("mastodon_date", col("orig_mastodon_date").cast("timestamp"))
      .drop("orig_mastodon_date", "vendor")
)

cve_metadata = (
    catalog_df
      .groupBy("cve_id")
      .agg(
          first("date_published",    ignorenulls=True).alias("date_published"),
          first("date_updated",      ignorenulls=True).alias("date_updated"),
          first("cvss_score",        ignorenulls=True).alias("cvss_score"),
          first("cvss_version",      ignorenulls=True).alias("cvss_version"),
          first("cwes",              ignorenulls=True).alias("cwes"),
          first("exploitDB_type",    ignorenulls=True).alias("exploitDB_type"),
          first("exploitDB_platform",ignorenulls=True).alias("exploitDB_platform"),
          first("KEV_product",       ignorenulls=True).alias("KEV_product")
      )
)

mastodon_summary = (
    catalog_df
      .withColumn("mastodon_day", to_date("mastodon_date"))
      .groupBy("cve_id", "mastodon_day")
      .agg(
          first("mastodon_date", ignorenulls=True).alias("mastodon_date"),
          spark_count("*").alias("mastodon_day_count")
      )
      .withColumnRenamed("cve_id", "rs_cve_id")
)

print(f"Distinct CVEs in Mastodon: {catalog_df.select('cve_id').distinct().count()}")
print(f"Rows in cve_metadata:      {cve_metadata.count()}")
print(f"Rows in mastodon_summary:  {mastodon_summary.count()}")

# ----------------------------------------------------------------------
# STEP 3: Load & clean EPSS data
# ----------------------------------------------------------------------
print("\n=== STEP 3: Loading and cleaning EPSS data ===")
epss_df = (
    spark.read
         .parquet("minimal_v1_timeseries_sample_checked.parquet")
         .filter(col("cve").isNotNull())
         .withColumnRenamed("cve", "cve_id")
         .withColumn("cve_id", lower(trim(col("cve_id"))))
         .withColumn("date", to_date(col("date")))
)

print(f"EPSS ROW COUNT: {epss_df.count():,}")
print(f"EPSS COLUMNS   : {len(epss_df.columns)}")

# ----------------------------------------------------------------------
# STEP 4: Join EPSS with metadata and Mastodon summary
# ----------------------------------------------------------------------
print("\n=== STEP 4: Joining EPSS with metadata and Mastodon aggregates ===")
augmented = (
    epss_df
      .join(cve_metadata.hint("broadcast"), on="cve_id", how="left")
      .join(
          mastodon_summary,
          (epss_df["cve_id"] == mastodon_summary["rs_cve_id"]) &
          (epss_df["date"] == mastodon_summary["mastodon_day"]),
          how="left"
      )
      .drop("rs_cve_id", "mastodon_day")
)

print(f"FINAL ROW COUNT: {augmented.count():,}")  # should match EPSS row count

# ----------------------------------------------------------------------
# STEP 5: Save output
# ----------------------------------------------------------------------
print("\n=== STEP 5: Saving to Parquet ===")
output_path = "parquet_preprocessing/input_mastodon.parquet"
augmented.coalesce(1).write.mode("overwrite").parquet(output_path)
print(f"✅ Saved to {output_path}")

spark.catalog.clearCache()
print("✅ Spark cache cleared")
