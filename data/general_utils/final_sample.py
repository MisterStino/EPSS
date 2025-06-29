from pyspark.sql.functions import (
    col, isnan, isnull, count, when, to_date,
    year, month, avg, stddev, lag, abs, rand
)
from pyspark.sql.functions import min as spark_min, max as spark_max
from pyspark.sql.functions import count as spark_count
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import os
from t3_spark.session import get_spark_session

# Use same global seed as sample.py for consistency
SEED = 42

# 1. Initialize Spark session
spark = get_spark_session()

# 2. Load data 
folder_path = "data/full_db/sampled/final_full_data_v3_v4truncated.parquet"
df = spark.read.parquet(folder_path)
df = df.withColumn("date", to_date("date"))
df = df.withColumn("epss", col("epss").cast(DoubleType()))

# ------------------------------
# Compute Variability Metrics per CVE
# ------------------------------

# MISSING VALUE ANALYSIS
missing_conditions = []
for c in ['cve', 'date', 'epss']:
    if c == 'epss':
        condition = isnull(col(c)) | isnan(col(c))
    else:
        condition = isnull(col(c))
    missing_conditions.append(count(when(condition, c)).alias(f"{c}_missing"))

df.select(missing_conditions).show()

# DATE RANGE
df.select(spark_min("date").alias("min_date"), spark_max("date").alias("max_date")).show()

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
    spark_min("date").alias("first_seen"),
    spark_max("date").alias("last_seen"),
    spark_count("*").alias("observations"),
    avg("epss").alias("avg_epss"),
    spark_min("epss").alias("min_epss"),
    spark_max("epss").alias("max_epss")
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
    spark_max("epss_jump").alias("max_one_day_jump")
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

# Label each CVE based on behavior
final_labeled_df = final_cve_variability_df.withColumn(
    "behavior_type",
    when((col("amplitude") < 0.3) & (col("min_epss") >= 0.7), "Type D")  # ← First
    .when((col("amplitude") >= 0.5) | (col("max_epss") >= 0.7), "Type A")
    .when((col("amplitude") >= 0.1) & (col("amplitude") < 0.5) & (col("max_epss") < 0.7), "Type B")
    .when((col("amplitude") < 0.1) & (col("max_epss") < 0.7), "Type C")
    .otherwise("Unclassified")  # fallback in case of data error
)

# Show breakdown by type
final_labeled_df.groupBy("behavior_type").count().orderBy("behavior_type").show()

# -----------------------------------------------------------------
# DATA INTEGRITY CHECK 1: One-row-per-CVE in labeled DataFrame
# -----------------------------------------------------------------
print("🔍 Running data integrity checks...")
assert final_labeled_df.count() == final_labeled_df.select("cve").distinct().count(), \
    "Duplicate CVE rows detected in final_labeled_df!"
print("✅ Check 1: One-row-per-CVE in labeled DataFrame - PASSED")

# ------------------------------
# Split Holdout Set (Deterministic)
# ------------------------------

# Define holdout fractions by behavior type
holdout_fractions = {
    "Type A": 0.10,  # 10% holdout, 90% for training
    "Type B": 0.10,  # 10% holdout, 90% for training  
    "Type C": 0.05,  # 5% holdout, 95% for training
    "Type D": 0.10   # 10% holdout, 90% for training
}

print("🔄 Creating stratified holdout split...")
print(f"Holdout fractions: {holdout_fractions}")

# Use sampleBy for deterministic stratified sampling
holdout_cve_df = final_labeled_df.sampleBy("behavior_type", holdout_fractions, seed=SEED)
holdout_cve_count = holdout_cve_df.count()

# Get remaining CVEs for training (subtract holdout from full set)
holdout_cve_ids = [row.cve for row in holdout_cve_df.select("cve").collect()]
remaining_cve_df = final_labeled_df.filter(~col("cve").isin(holdout_cve_ids))

# -----------------------------------------------------------------
# DATA INTEGRITY CHECK 2: No overlap between holdout and training CVE IDs
# -----------------------------------------------------------------
overlap = remaining_cve_df.select("cve").intersect(holdout_cve_df.select("cve")).count()
assert overlap == 0, "Hold-out CVEs leaked into training set!"
print("✅ Check 2: No overlap between holdout and training CVE IDs - PASSED")

print(f"📊 Holdout split summary:")
print(f"- Total CVEs: {final_labeled_df.count():,}")
print(f"- Holdout CVEs: {holdout_cve_count:,}")
print(f"- Remaining for training: {remaining_cve_df.count():,}")

# Show holdout distribution by behavior type
holdout_distribution = holdout_cve_df.groupBy("behavior_type").count().orderBy("behavior_type")
print("📈 Holdout set behavior type distribution:")
holdout_distribution.show()

# Create holdout time series dataset
holdout_cve_ids_df = holdout_cve_df.select("cve", "behavior_type")
holdout_time_series_df = df.join(holdout_cve_ids_df, on="cve", how="inner")

# -----------------------------------------------------------------
# DATA INTEGRITY CHECK 3: No duplicate (CVE, date) in holdout time series
# -----------------------------------------------------------------
dup_holdout = holdout_time_series_df.count() - holdout_time_series_df.dropDuplicates(["cve","date"]).count()
assert dup_holdout == 0, "Duplicate (cve, date) rows in hold-out time series!"
print("✅ Check 3: No duplicate (CVE, date) in holdout time series - PASSED")

# Save holdout set
import os
holdout_dir = "ml_pipeline/results/hold-out"
os.makedirs(holdout_dir, exist_ok=True)
holdout_path = f"{holdout_dir}/holdout_dataset.parquet"
holdout_time_series_df.write.mode("overwrite").parquet(holdout_path)

print(f"💾 Holdout set saved to: {holdout_path}")
print(f"📊 Holdout dataset: {holdout_time_series_df.count():,} rows with {holdout_cve_count:,} CVEs")

# Save labeled results (keep existing functionality)
final_labeled_df.write.mode("overwrite").parquet("cve_label_dataset_comparison.parquet")

labeled_cve_ids_df = final_labeled_df.select("cve", "behavior_type")

# Join to get full time series for each labeled CVE  
labeled_time_series_df = df.join(labeled_cve_ids_df, on="cve", how="inner")

# Save to Parquet
labeled_time_series_df.write.mode("overwrite").parquet("cve_label_timeseries_dataset_comparison.parquet")

print("Saved:")
print("• Labeled CVE metrics summary → cve_label_dataset_comparison.parquet")
print("• Full CVE time series with behavior labels → cve_label_timeseries_dataset_comparison.parquet")

# ------------------------------
# Select CVEs for Training (from remaining data after holdout split)
# ------------------------------

# Removed redundant import - rand already imported above

# Apply existing sampling logic to REMAINING data (after holdout split)
print("🎯 Applying balanced sampling to remaining CVEs...")

# Type A — keep all remaining
type_a_df = remaining_cve_df.filter(col("behavior_type") == "Type A")
type_a_count = type_a_df.count()

# Type B — sample up to Type A count (deterministic with fixed seed)
type_b_full = remaining_cve_df.filter(col("behavior_type") == "Type B")
type_b_limit = min(type_b_full.count(), type_a_count)
type_b_sampled = type_b_full.orderBy(rand(SEED)).limit(type_b_limit)

# Type C — sample up to Type A count (deterministic with fixed seed)
type_c_full = remaining_cve_df.filter(col("behavior_type") == "Type C")
type_c_limit = min(type_c_full.count(), type_a_count)
type_c_sampled = type_c_full.orderBy(rand(SEED)).limit(type_c_limit)

# Type D — keep all remaining (will be small)
type_d_df = remaining_cve_df.filter(col("behavior_type") == "Type D")

# Combine selected CVEs
training_cve_ids_df = type_a_df.unionByName(type_b_sampled).unionByName(type_c_sampled).unionByName(type_d_df).select("cve").distinct()

# Join with full time series
training_time_series_df = df.join(training_cve_ids_df, on="cve", how="inner")

# -----------------------------------------------------------------
# DATA INTEGRITY CHECK 4: No duplicate (CVE, date) in training time series
# -----------------------------------------------------------------
dup_train = training_time_series_df.count() - training_time_series_df.dropDuplicates(["cve","date"]).count()
assert dup_train == 0, "Duplicate (cve, date) rows in training time series!"
print("✅ Check 4: No duplicate (CVE, date) in training time series - PASSED")

# Save the result
training_time_series_df.write.mode("overwrite").parquet("training_dataset_prod.parquet")

# Print comprehensive summary
print("\n" + "="*60)
print("📋 FINAL DATASET SUMMARY")
print("="*60)
print(f"🏗️  Original dataset: {final_labeled_df.count():,} CVEs")
print(f"🔒 Holdout set: {holdout_cve_count:,} CVEs → ml_pipeline/results/hold-out/")
print(f"🎯 Training set: {training_cve_ids_df.count():,} CVEs → training_dataset_prod.parquet")
print()
print("📊 Training set breakdown (from remaining data after holdout):")
print(f"   - Type A included: {type_a_df.count():,}")
print(f"   - Type B sampled: {type_b_sampled.count():,} (capped at Type A count)")
print(f"   - Type C sampled: {type_c_sampled.count():,} (capped at Type A count)")
print(f"   - Type D included: {type_d_df.count():,}")
print()
print("🔍 DATA INTEGRITY SUMMARY:")
print("   ✅ All 4 data integrity checks PASSED")
print("   ✅ No duplicate CVEs in labeled dataset")
print("   ✅ No overlap between holdout and training sets")
print("   ✅ No duplicate time series entries in either dataset")
print()
print("✅ Both holdout and training datasets saved successfully!")
print("🔄 All sampling operations are deterministic (SEED = 42)")