# Import necessary functions from pyspark.
from pyspark.sql import functions as F
from pyspark.sql import Window

# Create Spark session using your utility function.
from t3_spark.session import get_spark_session
spark = get_spark_session()

# --------------------------------------------------------------------------
# 1. Load the Original Dataset
# --------------------------------------------------------------------------
# Replace 'path/to/input.parquet' with the actual path to your original parquet file.

input_path = "data/epss/processed/epss_processed.parquet"
df = spark.read.parquet(input_path)

# --------------------------------------------------------------------------
# 2. Create Static Features (Vendor and Remote)
# --------------------------------------------------------------------------
# Since these are static labels by CVE, we first aggregate by 'cve'
# and then use a deterministic hash to assign values.
df_static = df.select("cve").distinct() \
    .withColumn("hash_val", F.abs(F.hash(F.col("cve")))) \
    .withColumn("vendor_idx", (F.col("hash_val") % 5).cast("integer")) \
    .withColumn("remote", (F.col("hash_val") % 2).cast("integer")) \
    .withColumn(
        "vendor",
        F.when(F.col("vendor_idx") == 0, F.lit("Google"))
         .when(F.col("vendor_idx") == 1, F.lit("Microsoft"))
         .when(F.col("vendor_idx") == 2, F.lit("Apple"))
         .when(F.col("vendor_idx") == 3, F.lit("Amazon"))
         .otherwise(F.lit("Facebook"))
    ) \
    .select("cve", "vendor", "remote")

# Join the static features back to the original DataFrame by 'cve'
df = df.join(df_static, on="cve", how="left")

# --------------------------------------------------------------------------
# 3. Create Numeric Features
# --------------------------------------------------------------------------
# (a) Age of the CVE:
# Extract the year from the CVE string (e.g., from "CVE-2008-1098", extract "2008").
# We assume the publication date to be January 1 of that year.
df = df.withColumn("cve_year", F.substring(F.col("cve"), 5, 4)) \
       .withColumn("cve_year_date", F.to_date(F.concat(F.col("cve_year"), F.lit("-01-01")))) \
       .withColumn("age", F.datediff(F.col("date"), F.col("cve_year_date")))

# (b) Days since EPSs publication:
# For each CVE, determine the earliest (first) date and calculate the difference
# with the current row's date.
windowSpec = Window.partitionBy("cve")
df = df.withColumn("first_pub_date", F.min(F.col("date")).over(windowSpec)) \
       .withColumn("days_since_epss_pub", F.datediff(F.col("date"), F.col("first_pub_date")))

# (c) Reddit mentions:
# Create a random integer between 0 and 100 (using a fixed seed for reproducibility)
df = df.withColumn("reddit_mentions", F.floor(F.rand(42) * 100).cast("integer"))

# (d) Github commits:
# Create a random integer between 0 and 50 (using a different seed for variety)
df = df.withColumn("github_commits", F.floor(F.rand(24) * 50).cast("integer"))

# --------------------------------------------------------------------------
# 4. Clean Up Intermediate Columns
# --------------------------------------------------------------------------
# Drop temporary columns that were only used for computation.
df = df.drop("hash_val", "vendor_idx", "cve_year", "cve_year_date", "first_pub_date")

# --------------------------------------------------------------------------
# 5. Write the Final Dataset to Parquet
# --------------------------------------------------------------------------
# The dataset maintains the same row count but now contains the additional features.
output_path = "data/mock/processed/mock_processed.parquet"
df.write.mode("overwrite").parquet(output_path)
