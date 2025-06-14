from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, trim, lower, datediff, count, lag, avg, stddev, lead, broadcast
)
from pyspark.sql.window import Window
from pyspark import StorageLevel


# Step 1: Start Spark session with tuned configs
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

catalog_df = spark.read.option("header", True).csv("features/merged_reddit.csv") \
    .filter(col("CVE_ID").isNotNull()) \
    .withColumn("CVE_ID", trim(lower(col("CVE_ID")))) \
    .withColumnRenamed("Date", "reddit_date") \
    .drop("vendor") \
    .cache()

print("COLUMNS:", catalog_df.columns)

epss_df = spark.read.parquet("epss_processed.parquet/*.parquet") \
    .filter(col("cve").isNotNull()) \
    .withColumn("cve", trim(lower(col("cve")))) \
    .withColumn("date", to_date(col("date"))) \
    .cache()

catalog_df = catalog_df.alias("catalog")
epss_df = epss_df.alias("epss")

# Step 3: Merge EPS + Catalog
merged_df = catalog_df.join(
    broadcast(epss_df),
    catalog_df.CVE_ID == epss_df.cve,
    "inner"
).select(
    col("catalog.CVE_ID").alias("cve_id"),
    col("catalog.date_published"),
    col("catalog.date_updated"),
    col("catalog.cvss_score"),
    col("catalog.cvss_version"),
    col("catalog.cwes"),
    col("catalog.exploitDB_type"),
    col("catalog.exploitDB_platform"),
    col("catalog.KEV_product"),
    #col("catalog.Vendor"),       # <--- alias here
    col("catalog.Score"),
    col("catalog.reddit_date"), 
    col("catalog.Post ID"),
    *[col(f"epss.{c}") for c in epss_df.columns if c != "cve"]
).repartition("cve_id").cache()


# Save as parquet files (you can specify directory path)
output_path = "features/input_reddit_parquet"

merged_df.write.mode("overwrite").parquet(output_path)

print(f"Saved merged dataframe to parquet files at: {output_path}")