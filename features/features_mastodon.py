from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, trim, lower, datediff, count, lag, avg, stddev, lead, unix_date
)
from pyspark.sql.window import Window
from pyspark import StorageLevel

# Step 1: Start Spark session with tuned configs
from t3_spark.session import get_spark_session
spark = get_spark_session()

# Load the Mastodon-EPSS merged dataset
df = spark.read.parquet("parquet_preprocessing/input_mastodon.parquet")

# Ensure proper types
df = df.withColumn("date_mastodon", to_date(col("date_mastodon")))
df = df.withColumn("date_published", to_date(col("date_published")))
df = df.withColumn("date_updated", to_date(col("date_updated")))
df = df.withColumn("date", to_date(col("date")))  # EPSS date

# Filter out records with null CVE or mastodon date
df = df.filter(col("cve_id").isNotNull() & col("date_mastodon").isNotNull())

# ========== STATIC FEATURES ==========
static_cols = [
    "cvss_score", "cvss_version", "cwes", "affected_products", "tags",
    "exploitDB_type", "exploitDB_platform", "KEV_product",
    "KEV_knownRansomwareCampaignUse", "Vendor", "assigner"
]

# ========== TEMPORAL FEATURES ==========
# 1. Time since publication
df = df.withColumn("days_since_pub", datediff(col("date_mastodon"), col("date_published")))

# 2. Historical EPSS stats (7-day window BEFORE mastodon event)
epss_window_7d = Window.partitionBy("cve_id") \
    .orderBy(unix_date(col("date"))) \
    .rangeBetween(-7, -1)

df = df.withColumn("epss_mean_past7", avg("epss").over(epss_window_7d))
df = df.withColumn("epss_std_past7", stddev("epss").over(epss_window_7d))

# 3. Reddit CVE mention frequency (before current mention)
reddit_window = Window.partitionBy("cve_id").orderBy("date_mastodon")
df = df.withColumn("prev_date_mastodon", lag("date_mastodon", 1).over(reddit_window))
df = df.withColumn("delta_days_prev_mention", datediff(col("date_mastodon"), col("prev_date_mastodon")))

# 4. Daily Reddit CVE mentions (all CVEs)
activity_window_1d = Window.partitionBy("date_mastodon")
df = df.withColumn("total_mentions_all_CVEs_past1", count("cve_id").over(activity_window_1d))

# 5. Historical CVE mentions (per CVE)
mention_count_window_7d = Window.partitionBy("cve_id") \
    .orderBy(unix_date(col("date_mastodon"))) \
    .rangeBetween(-7, -1)
df = df.withColumn("count_mentions_past7", count("cve_id").over(mention_count_window_7d))

# 6. Global EPSS trend (average score of all CVEs on same day)
global_epss_window = Window.partitionBy("date")
df = df.withColumn("mean_epss_all_CVEs_past_day", avg("epss").over(global_epss_window))

# Select features for model training
selected_cols = [
    "cve_id", "date_mastodon", "epss",  # label
    "days_since_pub", "epss_mean_past7", "epss_std_past7",
    "delta_days_prev_mention", "count_mentions_past7", "total_mentions_all_CVEs_past1",
    "mean_epss_all_CVEs_past_day"
] + static_cols

features_df = df.select(*[col for col in selected_cols if col in df.columns])

# Save feature set
features_df.write.mode("overwrite").parquet("features/features_reddit.parquet")

print("✅ Feature extraction complete.")
