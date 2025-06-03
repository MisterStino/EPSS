from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, trim, lower, datediff, count, lag, avg, stddev, min, max, abs as abs_diff
)
from pyspark.sql.window import Window

# Step 1: Start Spark session
spark = SparkSession.builder \
    .appName("FullEPSSPipeline") \
    .master("local[*]") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.local.dir", "D:/spark-temp") \
    .getOrCreate()

# Step 2: Load enriched CVE catalog
csv_df = spark.read.option("header", True).csv("EPSS\\processed_data\\enriched_catalog.csv") \
    .filter(col("CVE_ID").isNotNull()) \
    .withColumn("CVE_ID", trim(lower(col("CVE_ID"))))

# Step 3: Load EPSS Parquet data
parquet_df = spark.read.parquet("epss_processed.parquet/*.parquet") \
    .filter(col("cve").isNotNull()) \
    .withColumn("cve", trim(lower(col("cve"))))

# Step 4: Join on CVE ID
merged_df = csv_df.join(parquet_df, csv_df.CVE_ID == parquet_df.cve, how='inner') \
    .select("CVE_ID", "cve", *[col for col in csv_df.columns if col != "CVE_ID"], *[col for col in parquet_df.columns if col != "cve"])

# Step 5: Repartition and write merged parquet
merged_df.write.mode("overwrite").parquet("merged_output_2.parquet")

# Step 6: Load reddit catalog with semicolon separator
mod_df = spark.read.option("header", True).option("delimiter", ";") \
    .csv("EPSS/processed_data/reddit_catalog.csv") \
    .filter(col("CVE ID").isNotNull() & col("Date").isNotNull()) \
    .withColumnRenamed("CVE ID", "CVE_ID") \
    .withColumnRenamed("Date", "Modification_Date") \
    .withColumn("CVE_ID", trim(lower(col("CVE_ID")))) \
    .withColumn("Modification_Date", to_date(col("Modification_Date")))

# Step 7: Load merged parquet with proper date formatting
parquet_df = spark.read.parquet("merged_output_2.parquet/*.parquet") \
    .filter(col("CVE_ID").isNotNull() & col("date").isNotNull()) \
    .withColumn("CVE_ID", trim(lower(col("CVE_ID")))) \
    .withColumn("date", to_date(col("date"))) \
    .withColumn("date_published", to_date(col("date_published"))) \
    .withColumn("date_updated", to_date(col("date_updated")))

# Step 8: Fuzzy join on CVE_ID and date within ±1 day
temp_join = mod_df.alias("reddit").join(
    parquet_df.alias("cve"),
    col("reddit.CVE_ID") == col("cve.CVE_ID"),
    how="inner"
).withColumn(
    "date_diff", datediff(col("reddit.Modification_Date"), col("cve.date"))
).filter(
    abs_diff(col("date_diff")) <= 1
)

# Select relevant columns after fuzzy join
merged = temp_join.select(
    col("reddit.CVE_ID").alias("cve_id"),
    col("reddit.Modification_Date").alias("modification_date"),
    col("reddit.Score").cast("double"),
    col("reddit.`Post ID`").alias("post_id"),
    col("cve.date"),
    col("cve.date_published"),
    col("cve.date_updated"),
    col("cve.epss"),
    col("cve.cvss_score"),
    col("cve.cvss_version"),
    col("cve.cwes"),
    col("cve.references"),
    col("cve.exploitDB_type"),
    col("cve.exploitDB_platform"),
    col("cve.KEV_product"),
    col("cve.KEV_knownRansomwareCampaignUse"),
    col("cve.Vendor")
)

# Step 9: Temporal Feature Engineering
cve_window = Window.partitionBy("cve_id").orderBy("modification_date")
cve_group_window = Window.partitionBy("cve_id")
vendor_window = Window.partitionBy("Vendor").orderBy("modification_date")

merged = merged.withColumn("days_since_publication", datediff(col("modification_date"), col("date_published"))) \
    .withColumn("days_since_last_update", datediff(col("modification_date"), col("date_updated"))) \
    .withColumn("modification_count", count("modification_date").over(cve_group_window)) \
    .withColumn("prev_mod_date", lag("modification_date").over(cve_window)) \
    .withColumn("mod_interval", datediff(col("modification_date"), col("prev_mod_date"))) \
    .withColumn("modification_interval_avg", avg("mod_interval").over(cve_group_window)) \
    .withColumn("modification_interval_stddev", stddev("mod_interval").over(cve_group_window)) \
    .withColumn("is_first_modification", (col("modification_date") == min("modification_date").over(cve_group_window)).cast("int")) \
    .withColumn("is_last_modification", (col("modification_date") == max("modification_date").over(cve_group_window)).cast("int")) \
    .withColumn("first_mod_date", min("modification_date").over(cve_group_window)) \
    .withColumn("last_mod_date", max("modification_date").over(cve_group_window)) \
    .withColumn("days_to_first_modification", datediff(col("first_mod_date"), col("date_published"))) \
    .withColumn("days_to_last_modification", datediff(col("last_mod_date"), col("date_published"))) \
    .withColumn("epss_prev", lag("epss").over(vendor_window)) \
    .withColumn("epss_change", col("epss") - col("epss_prev")) \
    .withColumn("vendor_avg_epss_change", avg("epss_change").over(Window.partitionBy("Vendor")))

# Reddit engagement metrics
product_mod_counts = merged.groupBy("KEV_product").agg(
    count("modification_date").alias("product_modification_rate"),
    avg("Score").alias("avg_reddit_score")
)

merged = merged.join(product_mod_counts, on="KEV_product", how="left")

# Step 10: Save final output
merged.repartition(100).write.mode("overwrite").parquet("features_reddit.parquet")

# Step 11: Stop session
spark.stop()
