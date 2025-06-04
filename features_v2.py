from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, count, min as spark_min, max as spark_max, avg, datediff,
    to_date, current_date, month, quarter, dayofweek, split, size,
    expr, sum as spark_sum, lit
)
from pyspark.sql.functions import col, count, min, max, avg, datediff, to_date, lit, when, size, split, dayofweek, month, quarter, year, lag
from pyspark.sql.window import Window


# Initialize Spark session
spark = SparkSession.builder.appName("CVE Feature Extraction").getOrCreate()

# File paths (adjust as needed)
mods_path = "catalogs_raw\cve_modification_history.csv"  # COLUMNS: CVE_ID, Modification_Date
catalog_path = "enriched_catalog.csv"    # COLUMNS: CVE_ID,state,assigner,date_published,...
output_path = "/path/to/output_features.csv"


# Read inputs
mods = spark.read.csv(mods_path, header=True, inferSchema=True)
catalog = spark.read.csv(catalog_path, header=True, inferSchema=True)

# Parse dates
mods = mods.withColumn(
    "Modification_Date", to_date(col("Modification_Date"), "yyyy-MM-dd")
)
catalog = (
    catalog
    .withColumn("date_published", to_date(col("date_published"), "yyyy-MM-dd"))
    .withColumn("date_updated", to_date(col("date_updated"), "yyyy-MM-dd"))
)

# Compute per-CVE modification aggregates
mod_stats = (
    mods.groupBy("CVE_ID")
    .agg(
        count("Modification_Date").alias("mod_count"),
        spark_min("Modification_Date").alias("first_mod_date"),
        spark_max("Modification_Date").alias("last_mod_date")
    )
)

# Join with catalog
df = mod_stats.join(catalog, on="CVE_ID", how="left")

# Temporal features: time to first/last mod, update lag
df = (
    df
    .withColumn("time_to_first_mod", datediff(col("first_mod_date"), col("date_published")))
    .withColumn("time_to_last_mod", datediff(col("last_mod_date"), col("date_published")))
    .withColumn("update_lag", datediff(col("date_updated"), col("date_published")))
)

# Inter-modification intervals
diff_window = Window.partitionBy("CVE_ID").orderBy("Modification_Date")
mods_with_lag = (
    mods.withColumn(
        "prev_date",
        lag(col("Modification_Date")).over(diff_window)
    )
    .withColumn("interval", datediff(col("Modification_Date"), col("prev_date")))
)
interval_stats = (
    mods_with_lag.groupBy("CVE_ID")
    .agg(
        avg("interval").alias("mean_inter_mod_interval"),
        spark_min("interval").alias("min_inter_mod_interval"),
        spark_max("interval").alias("max_inter_mod_interval")
    )
)
df = df.join(interval_stats, on="CVE_ID", how="left")

# Recency & frequency
cutoff_date = mods_with_lag.agg(spark_max("Modification_Date").alias("cutoff")).collect()[0][0]
df = (
    df
    .withColumn("days_since_last_mod", datediff(lit(cutoff_date), col("last_mod_date")))
    .withColumn(
        "mods_per_month",
        col("mod_count") / (datediff(col("last_mod_date"), col("first_mod_date")) / lit(30.0))
    )
)
recent_mods = (
    mods_with_lag
    .filter(datediff(lit(cutoff_date), col("Modification_Date")) <= 30)
    .groupBy("CVE_ID")
    .agg(count("Modification_Date").alias("mods_last_30d"))
)
df = df.join(recent_mods, on="CVE_ID", how="left").na.fill({"mods_last_30d": 0})

# Calendar features
first_mod = (
    mod_stats.select("CVE_ID", "first_mod_date")
    .withColumn("first_mod_dayofweek", dayofweek(col("first_mod_date")))
)
df = (
    df
    .withColumn("pub_month", month(col("date_published")))
    .withColumn("pub_quarter", quarter(col("date_published")))
    .join(first_mod.select("CVE_ID", "first_mod_dayofweek"), on="CVE_ID", how="left")
)

# Non-temporal features
# CWE count
# assume cwes is semicolon-separated string
from pyspark.sql.functions import explode
cwe_split = split(col("cwes"), ";")
df = df.withColumn("cwe_count", size(cwe_split))

# Convert KEV flags to integer
for flag in ["KEV_product", "KEV_knownRansomwareCampaignUse"]:
    df = df.withColumn(flag, col(flag).cast("int"))

# Vendor average mod rate
df = df.repartition(col("Vendor"))
vendor_rate = (
    df.groupBy("Vendor").agg(avg("mod_count").alias("vendor_mod_rate"))
)
df = df.join(vendor_rate, on="Vendor", how="left")

# CWE-specific CVSS mean
cwe_exploded = (
    df.select("CVE_ID", "cvss_score", cwe_split.alias("cwe_list"))
    .withColumn("cwe", explode(col("cwe_list")))
)
cwe_mean = (
    cwe_exploded.groupBy("cwe").agg(avg("cvss_score").alias("cwe_cvss_mean"))
)
cwe_per_cve = (
    cwe_exploded.join(cwe_mean, on="cwe")
    .groupBy("CVE_ID").agg(avg("cwe_cvss_mean").alias("cwe_cvss_mean"))
)
df = df.join(cwe_per_cve, on="CVE_ID", how="left")

# Cross features
# Avoid division by zero
df = (
    df
    .withColumn("cwe_per_mod", expr("cwe_count/NULLIF(mod_count,0)"))
    .withColumn("mods_per_cwe", expr("mod_count/NULLIF(cwe_count,0)"))
    .withColumn("cvss_x_KEV", col("cvss_score") * col("KEV_knownRansomwareCampaignUse"))
)

# Select and write
# selected_cols = [
#     "CVE_ID", "mod_count", "first_mod_date", "last_mod_date", "time_to_first_mod",
#     "time_to_last_mod", "update_lag", "mean_inter_mod_interval", "min_inter_mod_interval",
#     "max_inter_mod_interval", "days_since_last_mod", "mods_per_month", "mods_last_30d",
#     "pub_month", "pub_quarter", "first_mod_dayofweek", "cvss_score", "cvss_version",
#     "state", "assigner", "cwe_count", "exploitDB_type", "exploitDB_platform",
#     "KEV_product", "KEV_knownRansomwareCampaignUse", "Vendor", "vendor_mod_rate",
#     "cwe_cvss_mean", "cwe_per_mod", "mods_per_cwe", "cvss_x_KEV"
# ]

selected_cols = [
    "CVE_ID", "mod_count", "first_mod_date", "last_mod_date", "time_to_first_mod",
    "time_to_last_mod", "update_lag", "mean_inter_mod_interval", "min_inter_mod_interval",
    "max_inter_mod_interval", "days_since_last_mod", "mods_per_month", "mods_last_30d",
    "pub_month", "pub_quarter", "first_mod_dayofweek"
]

df.select(*selected_cols).coalesce(1).write.option("header", "true").mode("overwrite").csv("enriched_vulnerability_catalog2")


spark.stop()
