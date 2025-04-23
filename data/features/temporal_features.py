from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# Initialize Spark session
spark = (
    SparkSession.builder
    .appName("Temporal Feature Engineering for Vulnerabilities")
    .getOrCreate()
)

# File paths
dir_input = "data/final_dataset/enriched_catalog.csv"
dir_output = "data/features/temporal_features_spark"

# Date columns to parse
date_cols = [
    "Published_Date", "Last_Modified_Date",
    "exploitDB_date_published", "exploitDB_date_updated",
    "KEV_dateAdded", "metasploit_Disclosure Date",
    "ZDI_Published", "ZDI_Updated"
]

# Load CSV and parse date columns
df = (
    spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(dir_input)
)

# Convert string columns to timestamps (nullable)
for c in date_cols:
    df = df.withColumn(c, F.to_timestamp(F.col(c)))

# 1) Basic datetime features
for c in date_cols:
    df = df.withColumn(f"{c}_year", F.year(c))
    df = df.withColumn(f"{c}_month", F.month(c))
    df = df.withColumn(f"{c}_day", F.dayofmonth(c))
    df = df.withColumn(f"{c}_dayofweek", F.dayofweek(c) - 1)  # make Monday=0
    df = df.withColumn(f"{c}_is_weekend", (F.col(f"{c}_dayofweek").isin(5,6)).cast("int"))
    df = df.withColumn(f"{c}_quarter", F.quarter(c))
    df = df.withColumn(f"{c}_dayofyear", F.dayofyear(c))
    # cyclical encodings
    df = df.withColumn(f"{c}_month_sin", F.sin(2 * F.pi() * F.col(f"{c}_month") / F.lit(12)))
    df = df.withColumn(f"{c}_month_cos", F.cos(2 * F.pi() * F.col(f"{c}_month") / F.lit(12)))
    df = df.withColumn(f"{c}_dow_sin", F.sin(2 * F.pi() * F.col(f"{c}_dayofweek") / F.lit(7)))
    df = df.withColumn(f"{c}_dow_cos", F.cos(2 * F.pi() * F.col(f"{c}_dayofweek") / F.lit(7)))

# 2) Pairwise delta features (in days)
df = df.withColumn("time_to_last_mod", F.datediff(F.col("Last_Modified_Date"), F.col("Published_Date")))
df = df.withColumn("exploit_latency", F.datediff(F.col("exploitDB_date_published"), F.col("Published_Date")))
df = df.withColumn("exploit_update_delay", F.datediff(F.col("exploitDB_date_updated"), F.col("exploitDB_date_published")))
df = df.withColumn("kev_lag", F.datediff(F.col("KEV_dateAdded"), F.col("Published_Date")))
df = df.withColumn("metasploit_delay", F.datediff(F.col("metasploit_Disclosure Date"), F.col("Published_Date")))
df = df.withColumn("ZDI_publish_delay", F.datediff(F.col("ZDI_Published"), F.col("Published_Date")))
df = df.withColumn("ZDI_update_delay", F.datediff(F.col("ZDI_Updated"), F.col("ZDI_Published")))

# 3) Group-based "time since previous" using lag
for grp in ["Vendor", "CWE_ID", "KEV_product"]:
    w = Window.partitionBy(grp).orderBy("Published_Date")
    df = df.withColumn(f"{grp}_since_prev", F.datediff(F.col("Published_Date"), F.lag("Published_Date").over(w)))

# 4) Rolling counts per group in time windows
# Convert Published_Date to long seconds for rangeBetween
df = df.withColumn("ts", F.col("Published_Date").cast("long"))
for w in [7, 30, 90]:
    window_spec = Window.partitionBy("Vendor").orderBy("ts").rangeBetween(-w*86400, 0)
    df = df.withColumn(f"Vendor_count_last_{w}d", F.count("CVE_ID").over(window_spec))
    window_spec_cwe = Window.partitionBy("CWE_ID").orderBy("ts").rangeBetween(-w*86400, 0)
    df = df.withColumn(f"CWE_ID_count_last_{w}d", F.count("CVE_ID").over(window_spec_cwe))

# 5) Age feature: days since published until today
today_ts = F.unix_timestamp()
df = df.withColumn("age_days", (F.unix_timestamp() - F.unix_timestamp("Published_Date")) / 86400)

# Drop helper column
df = df.drop("ts")

# Write out as CSV
(
    df.write
      .mode("overwrite")
      .option("header", True)
      .csv(dir_output)
)

spark.stop()
