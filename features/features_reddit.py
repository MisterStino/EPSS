from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, trim, lower, datediff, count, lag, avg, stddev, lead, unix_date
)
from pyspark.sql.window import Window
from pyspark import StorageLevel

# # Step 1: Start Spark session with tuned configs
# from t3_spark.session import get_spark_session
# spark = get_spark_session()



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
    .getOrCreate()

# Load the reddit-EPSS merged dataset
df = spark.read.parquet("parquet_preprocessing/input_reddit.parquet")

# Ensure proper types
df = df.withColumn("reddit_date", to_date(col("reddit_date")))
df = df.withColumn("date_published", to_date(col("date_published")))
df = df.withColumn("date_updated", to_date(col("date_updated")))
df = df.withColumn("date", to_date(col("date")))  # EPSS date

# Filter out records with null CVE or reddit date
#df = df.filter(col("cve_id").isNotNull() & col("reddit_date").isNotNull())

# ========== STATIC FEATURES ==========
static_cols = [
    "cvss_score", "cvss_version", "cwes", "affected_products", "tags",
    "exploitDB_type", "exploitDB_platform", "KEV_product",
    "KEV_knownRansomwareCampaignUse", "Vendor", "assigner"
]

# # ========== TEMPORAL FEATURES ==========
# # 1. Time since publication
# df = df.withColumn("days_since_pub", datediff(col("date"), col("date_published")))

# # 2. Historical EPSS stats (7-day window BEFORE reddit event)
# epss_window_7d = Window.partitionBy("cve_id") \
#     .orderBy(unix_date(col("date"))) \
#     .rangeBetween(-7, -1)

# df = df.withColumn("epss_mean_past7", avg("epss").over(epss_window_7d))
# df = df.withColumn("epss_std_past7", stddev("epss").over(epss_window_7d))

# # 3. Reddit CVE mention frequency (before current mention)
# reddit_window = Window.partitionBy("cve_id").orderBy("reddit_date")
# df = df.withColumn("prev_reddit_date", lag("reddit_date", 1).over(reddit_window))
# df = df.withColumn("delta_days_prev_mention", datediff(col("reddit_date"), col("prev_reddit_date")))

# # 4. Daily Reddit CVE mentions (all CVEs)
# activity_window_1d = Window.partitionBy("reddit_date")
# df = df.withColumn("total_mentions_all_CVEs_past1", count("cve_id").over(activity_window_1d))

# # 5. Historical CVE mentions (per CVE)
# mention_count_window_7d = Window.partitionBy("cve_id") \
#     .orderBy(unix_date(col("reddit_date"))) \
#     .rangeBetween(-7, -1)
# df = df.withColumn("count_mentions_past7", count("cve_id").over(mention_count_window_7d))

# # 6. Global EPSS trend (average score of all CVEs on same day)
# global_epss_window = Window.partitionBy("date")
# df = df.withColumn("mean_epss_all_CVEs_past_day", avg("epss").over(global_epss_window))

# Select features for model training
selected = [
    "cve_id",
    "date",
    "epss",
    "age_epss_pub",
    #"original_date",
    #"date_parsed",
    #"cve_date_key",
    "has_discovery",
    "has_release",
    "has_threat",
    "has_remediation",
    "event_type_count",
    #"event_types_list",
    "dominant_event_type",
    #"sources_list",
    "source_count",
    "primary_source",
    "has_multi_source",
    #"doc_ids",
    #"details_combined",
    #"details_longest",
    "total_detail_length",
    #"event_data_merged",
    "event_sequence",
    "days_since_last_event",
    #"cumulative_source_count",
    #"same_day_multi_source",
    "total_events_so_far",
    #"prev_event_type",
    #"event_stage_num",
    #"max_stage_reached",
    #"reconstruction_timestamp",
    #"reconstruction_timestamp_raw",
    #"source_identifier",
    #"published_date",
    #"last_modified_date",
    #"vuln_status",
    #"cve_tags",
    "weakness_count",
    "reference_count",
    "configuration_count",
    "primary_cvss_ver",
    #"primary_cvss_vec",
    #"primary_cvss_score",
    #"primary_cvss_sev",
    #"snapshot_date",
    "canon_base",
    #"canon_severity",
    #"has_v2",
    #"has_v30",
    #"has_v31",
    #"has_v40",
    "n_cpes",
    "n_vendors",
    "is_windows",
    "is_linux",
    "is_android",
    "is_ios",
    "is_macos",
    "is_hardware",
    "is_application",
    "is_os",
    "cwe_id",
    "n_refs",
   # "description_all",
    "desc_len_all",
   # "description_en",
    #"desc_len_en",
    #"date_published",
    #"date_updated",
    #"cvss_score",
    #"cvss_version",
    #"cwes",
    #"exploitDB_type",
    #"exploitDB_platform",
    #"KEV_product",
    "reddit_date",
    "reddit_day_count",
]
features_df = df.select(*[col for col in selected if col in df.columns])

# Save feature set
features_df.coalesce(1).write.mode("overwrite").parquet("features/features_reddit.parquet")

print("✅ Feature extraction complete.")
