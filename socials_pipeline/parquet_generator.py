from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, trim, lower, first, count as spark_count
)

from t3_spark.session import get_spark_session

# -----------------------------------------------------------------------------
# SET THIS TO EITHER "reddit" OR "mastodon"
# -----------------------------------------------------------------------------
platform = "reddit"  # or "mastodon"

# -----------------------------------------------------------------------------
# STEP 1: Spark session
# -----------------------------------------------------------------------------
spark = get_spark_session()

# -----------------------------------------------------------------------------
# STEP 2: Load & clean platform-specific data
# -----------------------------------------------------------------------------
print(f"=== STEP 2: Loading and cleaning {platform.capitalize()} data ===")

platform_file = f"socials_pipeline/parquet_inputs/merged_{platform}.csv"
raw_df = (
    spark.read
        .option("header", True)
        .csv(platform_file)
        .filter(col("CVE_ID").isNotNull())
)

date_col_in = "Date" if platform == "reddit" else "date_mastodon"
date_col_out = f"{platform}_date"
day_col = f"{platform}_day"
day_count_col = f"{platform}_day_count"

catalog_df = (
    raw_df
        .withColumnRenamed("CVE_ID", "cve_id")
        .withColumn("cve_id", lower(trim(col("cve_id"))))
        .withColumnRenamed(date_col_in, f"orig_{date_col_out}")
        .withColumn(date_col_out, col(f"orig_{date_col_out}").cast("timestamp"))
        .drop(f"orig_{date_col_out}", "vendor")
)

# 2.1 One-row-per-CVE metadata
cve_metadata = (
    catalog_df
        .groupBy("cve_id")
        .agg(
            first("date_published", ignorenulls=True).alias("date_published"),
            first("date_updated", ignorenulls=True).alias("date_updated"),
            first("cvss_score", ignorenulls=True).alias("cvss_score"),
            first("cvss_version", ignorenulls=True).alias("cvss_version"),
            first("cwes", ignorenulls=True).alias("cwes"),
            first("exploitDB_type", ignorenulls=True).alias("exploitDB_type"),
            first("exploitDB_platform", ignorenulls=True).alias("exploitDB_platform"),
            first("KEV_product", ignorenulls=True).alias("KEV_product"),
            first("Score", ignorenulls=True).alias("Score")
        )
)

# 2.2 Summarize platform posts per (CVE, day)
summary_df = (
    catalog_df
        .withColumn(day_col, to_date(col(date_col_out)))
        .groupBy("cve_id", day_col)
        .agg(
            first(date_col_out, ignorenulls=True).alias(date_col_out),
            spark_count("*").alias(day_count_col)
        )
        .withColumnRenamed("cve_id", "rs_cve_id")
)

print(f"Distinct CVEs in {platform.capitalize()}: {catalog_df.select('cve_id').distinct().count()}")
print(f"Rows in cve_metadata:   {cve_metadata.count()}")
print(f"Rows in {platform}_summary: {summary_df.count()}")

# -----------------------------------------------------------------------------
# STEP 3: Load & clean EPSS data
# -----------------------------------------------------------------------------
print("\n=== STEP 3: Loading and cleaning EPSS data ===")

epss_df = (
    spark.read
        .parquet("data/epss/processed/epss_processed.parquet")
        .filter(col("cve").isNotNull())
        .withColumnRenamed("cve", "cve_id")
        .withColumn("cve_id", lower(trim(col("cve_id"))))
        .withColumn("date", to_date(col("date")))
)

print(f"EPSS ROW COUNT: {epss_df.count():,}")
print(f"EPSS COLUMNS   : {len(epss_df.columns)}")

# -----------------------------------------------------------------------------
# STEP 4: Join EPSS with metadata and summary
# -----------------------------------------------------------------------------
print(f"\n=== STEP 4: Joining EPSS with metadata and {platform.capitalize()} aggregates ===")

augmented = (
    epss_df
        .join(cve_metadata.hint("broadcast"), on="cve_id", how="left")
        .join(
            summary_df,
            (epss_df["cve_id"] == summary_df["rs_cve_id"]) &
            (epss_df["date"] == summary_df[day_col]),
            how="left"
        )
        .drop("rs_cve_id", day_col)
)

print(f"FINAL ROW COUNT: {augmented.count():,}")

# -----------------------------------------------------------------------------
# STEP 5: Clean the data for model training
# -----------------------------------------------------------------------------
output_path = f"data/reddit/processed/reddit_processed.parquet"


# for the final parquet file we store i want to rename the cve_id to cve

selected = [
    "cve_id",
    "date",
    "epss",
#     "age_epss_pub",
#     #"original_date",
#     #"date_parsed",
#     #"cve_date_key",
#     "has_discovery",
#     "has_release",
#     "has_threat",
#     "has_remediation",
#     "event_type_count",
#     #"event_types_list",
#     "dominant_event_type",
#     #"sources_list",
#     "source_count",
#     "primary_source",
#     "has_multi_source",
#     #"doc_ids",
#     #"details_combined",
#     #"details_longest",
#     "total_detail_length",
#     #"event_data_merged",
#     "event_sequence",
#     "days_since_last_event",
#     #"cumulative_source_count",
#     #"same_day_multi_source",
#     "total_events_so_far",
#     #"prev_event_type",
#     #"event_stage_num",
#     #"max_stage_reached",
#     #"reconstruction_timestamp",
#     #"reconstruction_timestamp_raw",
#     #"source_identifier",
#     #"published_date",
#     #"last_modified_date",
#     #"vuln_status",
#     #"cve_tags",
#     "weakness_count",
#     "reference_count",
#     "configuration_count",
#     "primary_cvss_ver",
#     #"primary_cvss_vec",
#     #"primary_cvss_score",
#     #"primary_cvss_sev",
#     #"snapshot_date",
#     "canon_base",
#     #"canon_severity",
#     #"has_v2",
#     #"has_v30",
#     #"has_v31",
#     #"has_v40",
#     "n_cpes",
#     "n_vendors",
#     "is_windows",
#     "is_linux",
#     "is_android",
#     "is_ios",
#     "is_macos",
#     "is_hardware",
#     "is_application",
#     "is_os",
#     "cwe_id",
#     "n_refs",
#    # "description_all",
#     "desc_len_all",
#    # "description_en",
#     #"desc_len_en",
#     #"date_published",
#     #"date_updated",
#     #"cvss_score",
#     #"cvss_version",
#     #"cwes",
#     #"exploitDB_type",
#     #"exploitDB_platform",
#     #"KEV_product",
    # date_col_out, 
    day_count_col,
]

augmented = augmented.select(*[c for c in selected if c in augmented.columns])

# -----------------------------------------------------------------------------
# STEP 6: Feature Creation 
# -----------------------------------------------------------------------------
# If model is stable, add new features here

# -----------------------------------------------------------------------------
# STEP 7: Save output
# -----------------------------------------------------------------------------
print(f"\n=== STEP 5: Saving to Parquet ===")
augmented.withColumnRenamed("cve_id", "cve").write.mode("overwrite").parquet(output_path)
print(f"✅ Saved to {output_path}")

spark.catalog.clearCache()
print("✅ Spark cache cleared")
