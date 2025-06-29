from pyspark.sql.functions import (
    col, isnan, isnull, count, when, to_date,
    min, max, year, month, avg, stddev, lag, abs
)
from pyspark.sql.functions import count as spark_count
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import os
from t3_spark.session import get_spark_session

# 1. Initialize Spark session using t3_spark
extra_configs = {
    "spark.sql.shuffle.partitions": "50",
    "spark.sql.adaptive.enabled": "true", 
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.sql.broadcastTimeout": "3600"
}
spark = get_spark_session("UnifiedEPSSPipeline", "local[*]", extra_configs)

# 2. Load data
folder_path = "data/full_db/sampled/final_full_data_v3_v4truncated.parquet"
df = spark.read.parquet(folder_path)
df = df.withColumn("date", to_date("date"))
df = df.withColumn("epss", col("epss").cast(DoubleType()))

# ------------------------------
# Compute Variability Metrics per CVE
# ------------------------------

# MISSING VALUE ANALYSIS
def get_null_condition(column_name):
    if column_name == 'epss':
        return isnull(col(column_name)) | isnan(col(column_name))
    else:
        return isnull(col(column_name))

df.select([
    count(when(get_null_condition(c), c)).alias(f"{c}_missing")
    for c in ['cve', 'date', 'epss']
]).show()

# DATE RANGE
df.select(min("date").alias("min_date"), max("date").alias("max_date")).show()

# EPSS BUCKETING
df = df.withColumn("epss_bucket", 
    when(col("epss") < 0.1, "<0.1")
    .when(col("epss") < 0.5, "0.1-0.5")
    .when(col("epss") < 0.7, "0.5-0.7")
    .otherwise(">=0.7")
)
df.groupBy("epss_bucket").count().orderBy("epss_bucket").show()

# DISTINCT CVEs
print("Distinct CVEs:", df.select("cve").distinct().count())

# Duplicate check
df.groupBy("cve").count().filter("count > 1").show()

# Avg EPSS per Year
df.groupBy(year("date").alias("year")).agg(
    avg("epss").alias("avg_epss")
).orderBy("year").show()

# High EPSS CVEs
df.filter(col("epss") > 0.7).select("cve", "date", "epss").show(10, truncate=False)

# Per-CVE base statistics
cve_stats_df = df.groupBy("cve").agg(
    min("date").alias("first_seen"),
    max("date").alias("last_seen"),
    spark_count("*").alias("observations"),
    avg("epss").alias("avg_epss"),
    min("epss").alias("min_epss"),
    max("epss").alias("max_epss")
)

# Add EPSS amplitude (range)
cve_stats_df = cve_stats_df.withColumn(
    "amplitude", col("max_epss") - col("min_epss")
)

# Standard deviation per CVE
epss_stddev_df = df.groupBy("cve").agg(
    stddev("epss").alias("epss_stddev")
)

# Max one-day jump
window_spec = Window.partitionBy("cve").orderBy("date")
df_with_lag = df.withColumn("prev_epss", lag("epss").over(window_spec))
df_with_lag = df_with_lag.withColumn("epss_jump", abs(col("epss") - col("prev_epss")))
max_jump_df = df_with_lag.groupBy("cve").agg(
    max("epss_jump").alias("max_one_day_jump")
)

# Combine all metrics
from functools import reduce
from pyspark.sql import DataFrame

dfs_to_join = [cve_stats_df, epss_stddev_df, max_jump_df]
final_cve_variability_df = reduce(lambda left, right: left.join(right, on="cve", how="left"), dfs_to_join)

# Show top CVEs by amplitude
#final_cve_variability_df.orderBy(col("amplitude").desc()).show(10, truncate=False)

from pyspark.sql.functions import when

# ------------------------------
# Stratify CVEs by Behavior
# ------------------------------

'''
Type A: amplitude ≥ 0.5 or max_epss ≥ 0.7  
Type B: 0.1 ≤ amplitude < 0.5 AND max_epss < 0.7 AND min_epss < 0.7  
Type C: amplitude < 0.1 AND max_epss < 0.7 AND min_epss < 0.7  
Type D: amplitude < 0.3 AND min_epss ≥ 0.7 AND not Type A
'''

# Step 1: Add disjoint boolean flags
final_cve_variability_df = final_cve_variability_df \
    .withColumn("is_type_a", (col("amplitude") >= 0.5) | (col("max_epss") >= 0.7)) \
    .withColumn("is_type_b", 
        (col("amplitude") >= 0.1) & (col("amplitude") < 0.5) &
        (col("max_epss") < 0.7) & (col("min_epss") < 0.7) &
        ~((col("amplitude") >= 0.5) | (col("max_epss") >= 0.7))  # exclude A
    ) \
    .withColumn("is_type_c",
        (col("amplitude") < 0.1) & (col("max_epss") < 0.7) & (col("min_epss") < 0.7) &
        ~((col("amplitude") >= 0.1) & (col("amplitude") < 0.5))  # exclude B
    ) \
    .withColumn("is_type_d",
        (col("amplitude") < 0.3) & (col("min_epss") >= 0.7) &
        ~((col("amplitude") >= 0.5) | (col("max_epss") >= 0.7))  # exclude A
    )

# Step 2: Assign type based on flags (non-overlapping, priority-based)
final_labeled_df = final_cve_variability_df.withColumn(
    "behavior_type",
    when(col("is_type_a"), "Type A")
    .when(col("is_type_b"), "Type B")
    .when(col("is_type_c"), "Type C")
    .when(col("is_type_d"), "Type D")
    .otherwise("Unclassified")
)

# Show breakdown by type
final_labeled_df.groupBy("behavior_type").count().orderBy("behavior_type").show()

# Save labeled results
final_labeled_df.write.mode("overwrite").parquet("cve_label_dataset.parquet")


labeled_cve_ids_df = final_labeled_df.select("cve", "behavior_type")

# Join to get full time series for each labeled CVE
labeled_time_series_df = df.join(labeled_cve_ids_df, on="cve", how="inner")

# Save to Parquet
labeled_time_series_df.write.mode("overwrite").parquet("cve_label_timeseries_dataset.parquet")

print("Saved:")
print("• Labeled CVE metrics summary → cve_label_dataset.parquet")
print("• Full CVE time series with behavior labels → cve_label_timeseries_dataset.parquet")


# ------------------------------
# Select CVEs for Training
# ------------------------------

from pyspark.sql.functions import col, rand

# Type A — keep all
type_a_df = final_labeled_df.filter(col("behavior_type") == "Type A")

# Type B — sample 30%
type_b_full = final_labeled_df.filter(col("behavior_type") == "Type B")
type_b_sampled = type_b_full.orderBy(rand()).limit(int(type_b_full.count() * 0.5))

# Type C — sample 5%
type_c_full = final_labeled_df.filter(col("behavior_type") == "Type C")
type_c_sampled = type_c_full.orderBy(rand()).limit(int(type_c_full.count() * 0.03))

# Combine selected CVEs
training_cve_ids_df = type_a_df.unionByName(type_b_sampled).unionByName(type_c_sampled).select("cve").distinct()

# Join with full time series
training_time_series_df = df.join(training_cve_ids_df, on="cve", how="inner")

# Save the result
training_time_series_df.write.mode("overwrite").parquet("data/full_db/sampled/training_dataset.parquet")

# Print summary
print("✅ Training dataset saved to 'training_dataset.parquet'")
print(f"- Type A included: {type_a_df.count()}")
print(f"- Type B sampled: {type_b_sampled.count()}")
print(f"- Type C sampled: {type_c_sampled.count()}")
print(f"- Total CVEs in training set: {training_cve_ids_df.count()}")