from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, trim, lower, datediff, count, lag, avg, stddev, lead
)
from pyspark.sql.window import Window
from pyspark import StorageLevel

# Step 1: Start Spark session with tuned configs
from t3_spark.session import get_spark_session
spark = get_spark_session()

print("=== STEP 1: Loading and cleaning Reddit data ===")
catalog_df = spark.read.option("header", True).csv("parquet_preprocessing/merged_reddit.csv") \
    .filter(col("CVE_ID").isNotNull()) \
    .withColumn("CVE_ID", trim(lower(col("CVE_ID")))) \
    .withColumnRenamed("Date", "reddit_date") \
    .drop("vendor")

print("REDDIT COLUMNS:", catalog_df.columns)
print("REDDIT ROW COUNT:", catalog_df.count())

print("\n=== STEP 2: Loading and cleaning EPSS data ===")
epss_df = spark.read.parquet("data/epss/processed/epss_processed.parquet") \
    .filter(col("cve").isNotNull()) \
    .withColumn("cve", trim(lower(col("cve")))) \
    .withColumn("date", to_date(col("date")))

print("EPSS ROW COUNT:", epss_df.count())
print("EPSS COLUMNS:", len(epss_df.columns))

# Check data sizes before join
reddit_distinct_cves = catalog_df.select("CVE_ID").distinct().count()
epss_distinct_cves = epss_df.select("cve").distinct().count()
print(f"\nDistinct CVEs - Reddit: {reddit_distinct_cves}, EPSS: {epss_distinct_cves}")

catalog_df = catalog_df.alias("catalog")
epss_df = epss_df.alias("epss")

print("\n=== STEP 3: Performing join (without broadcast) ===")
# JOIN TYPE EXPLANATION:
# "inner" = Only CVEs that exist in BOTH Reddit AND EPSS data
# "left"  = ALL Reddit CVEs + matching EPSS data (NULLs where no EPSS match)
# "right" = ALL EPSS CVEs + matching Reddit data (NULLs where no Reddit match)
# "outer" = ALL CVEs from both sources (NULLs where no match on either side)

# Current: RIGHT JOIN - keeps ALL EPSS rows, adds Reddit data where available
# Use regular join instead of broadcast - EPSS data is too large to broadcast
merged_df = catalog_df.join(
    epss_df,
    catalog_df.CVE_ID == epss_df.cve,
    "right"  # RIGHT JOIN: All EPSS rows + Reddit data where available
).select(
    col("epss.cve").alias("cve_id"),  # Use EPSS CVE since it's always present
    col("catalog.date_published"),
    col("catalog.date_updated"),
    col("catalog.cvss_score"),
    col("catalog.cvss_version"),
    col("catalog.cwes"),
    col("catalog.exploitDB_type"),
    col("catalog.exploitDB_platform"),
    col("catalog.KEV_product"),
    col("catalog.Score"),
    col("catalog.reddit_date"), 
    col("catalog.Post ID"),
    *[col(f"epss.{c}") for c in epss_df.columns if c != "cve"]
).repartition(200, "cve_id")  # Increase partitions for better parallelism

print("MERGED ROW COUNT:", merged_df.count())

print("\n=== STEP 4: Saving to parquet ===")
output_path = "parquet_preprocessing/input_reddit.parquet"

merged_df.write.mode("overwrite").parquet(output_path)

print(f"✅ Successfully saved merged dataframe to: {output_path}")

# Clean up
spark.catalog.clearCache()
print("✅ Cleared Spark cache")