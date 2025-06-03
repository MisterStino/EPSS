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

# Step 2: Load and clean datasets
catalog_df = spark.read.option("header", True).csv("EPSS/processed_data/enriched_catalog.csv") \
    .filter(col("CVE_ID").isNotNull()) \
    .withColumn("CVE_ID", trim(lower(col("CVE_ID")))) \
    .cache()

epss_df = spark.read.parquet("epss_processed.parquet/*.parquet") \
    .filter(col("cve").isNotNull()) \
    .withColumn("cve", trim(lower(col("cve")))) \
    .withColumn("date", to_date(col("date"))) \
    .cache()

mod_df = spark.read.option("header", True).csv("catalogs_processed/cve_modification_history.csv") \
    .filter(col("CVE_ID").isNotNull() & col("Modification_Date").isNotNull()) \
    .withColumn("CVE_ID", trim(lower(col("CVE_ID")))) \
    .withColumn("Modification_Date", to_date(col("Modification_Date"))) \
    .cache()

# Step 3: Merge EPS + Catalog
merged_df = catalog_df.join(broadcast(epss_df), catalog_df.CVE_ID == epss_df.cve, "inner") \
    .select(
        col("CVE_ID").alias("cve_id"),
        col("date_published"),
        col("cvss_score"),
        col("cvss_version"),
        col("cwes"),
        col("exploitDB_type"),
        col("exploitDB_platform"),
        col("KEV_product"),
        col("Vendor"),
        *[col for col in epss_df.columns if col != "cve"]
    ).repartition("cve_id").cache()

# Step 4: Join with Modification History
merged = merged_df.join(mod_df, on=["cve_id"], how="inner") \
    .filter(col("date") <= col("Modification_Date")) \
    .repartition(50) \
    .persist(StorageLevel.DISK_ONLY)

# Step 5: Define windows
cve_lag_window = Window.partitionBy("cve_id").orderBy("date")
cve_agg_window = Window.partitionBy("cve_id").orderBy("date").rowsBetween(Window.unboundedPreceding, -1)
vendor_window = Window.partitionBy("Vendor").orderBy("date").rowsBetween(Window.unboundedPreceding, -1)
product_window = Window.partitionBy("KEV_product").orderBy("date").rowsBetween(Window.unboundedPreceding, -1)

# Step 6: Feature Engineering
features = merged \
    .withColumn("days_since_publication", datediff(col("date"), col("date_published"))) \
    .withColumn("modification_count", count("Modification_Date").over(cve_agg_window)) \
    .withColumn("prev_mod_date", lag("Modification_Date").over(cve_lag_window)) \
    .withColumn("mod_interval", datediff(col("Modification_Date"), col("prev_mod_date"))) \
    .withColumn("mod_interval_avg", avg("mod_interval").over(cve_agg_window)) \
    .withColumn("mod_interval_stddev", stddev("mod_interval").over(cve_agg_window)) \
    .withColumn("epss_lag_1", lag("epss", 1).over(cve_lag_window)) \
    .withColumn("epss_lag_7", lag("epss", 7).over(cve_lag_window)) \
    .withColumn("epss_lag_30", lag("epss", 30).over(cve_lag_window)) \
    .withColumn("vendor_avg_epss", avg("epss").over(vendor_window)) \
    .withColumn("product_mod_count", count("Modification_Date").over(product_window)) \
    .withColumn("epss_plus_30d", lead("epss", 30).over(cve_lag_window)) \
    .filter(col("epss_plus_30d").isNotNull())

# Step 7: Optimize and write Parquet
features = features.repartition(50)

features.write \
    .option("compression", "snappy") \
    .option("mergeSchema", "false") \
    .mode("overwrite") \
    .parquet("features_no_leakage_v2.parquet")

features.unpersist()

# Step 8: Stop Spark session
spark.stop()
