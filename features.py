from pyspark.sql import SparkSession
from pyspark.sql.functions import col, unix_timestamp, datediff, year, month, dayofmonth, when, lit, current_date, split, size
import pyspark.sql.functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("VulnerabilityFeatureEngineering").getOrCreate()

# Load the CSV file into a Spark DataFrame
df = spark.read.csv("enriched_catalog.csv", header=True, inferSchema=True)

# Convert date columns to proper date format
df = df.withColumn("date_published", unix_timestamp("date_published", "yyyy-MM-dd").cast("timestamp"))
df = df.withColumn("date_updated", unix_timestamp("date_updated", "yyyy-MM-dd").cast("timestamp"))

# Temporal Features
df = df.withColumn("days_between_published_updated", 
                   datediff(col("date_updated"), col("date_published")))

df = df.withColumn("published_year", year(col("date_published"))) \
       .withColumn("published_month", month(col("date_published"))) \
       .withColumn("published_day", dayofmonth(col("date_published")))

df = df.withColumn("updated_year", year(col("date_updated"))) \
       .withColumn("updated_month", month(col("date_updated"))) \
       .withColumn("updated_day", dayofmonth(col("date_updated")))

# Recent Update Indicator (within last 30 days)
recent_update_days = 30
df = df.withColumn("is_recently_updated", 
                   when(datediff(current_date(), col("date_updated")) <= recent_update_days, 1).otherwise(0))

# CVSS Score Categories
df = df.withColumn("cvss_score_category", 
                   when(col("cvss_score") < 4.0, "Low")
                   .when((col("cvss_score") >= 4.0) & (col("cvss_score") < 7.0), "Medium")
                   .otherwise("High"))

# Vendor-specific Features
df = df.withColumn("is_vendor_known", 
                   when(col("Vendor").isNotNull(), 1).otherwise(0))

# KEV_product-related Features
df = df.withColumn("kev_product_count", size(split(F.coalesce(col("KEV_product"), lit("")), ",")))
df = df.withColumn("kev_known_ransomware_campaign", 
                   when(col("KEV_knownRansomwareCampaignUse").isNotNull(), 1).otherwise(0))

# Write to a single CSV file
df.coalesce(1).write.option("header", "true").mode("overwrite").csv("enriched_vulnerability_catalog")

# Stop Spark session
spark.stop()
