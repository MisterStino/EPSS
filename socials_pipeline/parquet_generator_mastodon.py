from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, trim, lower, first, count as spark_count
)
from t3_spark.session import get_spark_session

# ----------------------------------------------------------------------
# STEP 1: Spark session
# ----------------------------------------------------------------------
spark = get_spark_session()


# ----------------------------------------------------------------------
# STEP 2: Load & clean Mastodon data
# ----------------------------------------------------------------------
print("=== STEP 2: Loading and cleaning Mastodon data ===")
raw_mastodon = (
    spark.read
         .option("header", True)
         .csv("socials_pipeline/parquet_inputs/merged_mastodon.csv")
         .filter(col("CVE_ID").isNotNull())
)

catalog_df = (
    raw_mastodon
      .withColumnRenamed("CVE_ID", "cve_id")
      .withColumn("cve_id", lower(trim(col("cve_id"))))
      .withColumnRenamed("date_mastodon", "orig_mastodon_date")
      .withColumn("mastodon_date", col("orig_mastodon_date").cast("timestamp"))
      .drop("orig_mastodon_date", "vendor")
)

cve_metadata = (
    catalog_df
      .groupBy("cve_id")
      .agg(
          first("date_published",    ignorenulls=True).alias("date_published"),
          first("date_updated",      ignorenulls=True).alias("date_updated"),
          first("cvss_score",        ignorenulls=True).alias("cvss_score"),
          first("cvss_version",      ignorenulls=True).alias("cvss_version"),
          first("cwes",              ignorenulls=True).alias("cwes"),
          first("exploitDB_type",    ignorenulls=True).alias("exploitDB_type"),
          first("exploitDB_platform",ignorenulls=True).alias("exploitDB_platform"),
          first("KEV_product",       ignorenulls=True).alias("KEV_product")
      )
)

mastodon_summary = (
    catalog_df
      .withColumn("mastodon_day", to_date("mastodon_date"))
      .groupBy("cve_id", "mastodon_day")
      .agg(
          first("mastodon_date", ignorenulls=True).alias("mastodon_date"),
          spark_count("*").alias("mastodon_day_count")
      )
      .withColumnRenamed("cve_id", "rs_cve_id")
)

print(f"Distinct CVEs in Mastodon: {catalog_df.select('cve_id').distinct().count()}")
print(f"Rows in cve_metadata:      {cve_metadata.count()}")
print(f"Rows in mastodon_summary:  {mastodon_summary.count()}")

# ----------------------------------------------------------------------
# STEP 3: Load & clean EPSS data
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# STEP 4: Join EPSS with metadata and Mastodon summary
# ----------------------------------------------------------------------
print("\n=== STEP 4: Joining EPSS with metadata and Mastodon aggregates ===")
augmented = (
    epss_df
      .join(cve_metadata.hint("broadcast"), on="cve_id", how="left")
      .join(
          mastodon_summary,
          (epss_df["cve_id"] == mastodon_summary["rs_cve_id"]) &
          (epss_df["date"] == mastodon_summary["mastodon_day"]),
          how="left"
      )
      .drop("rs_cve_id", "mastodon_day")
)

print(f"FINAL ROW COUNT: {augmented.count():,}")  # should match EPSS row count

# ----------------------------------------------------------------------
# STEP 5: Clean the data for model training
# ----------------------------------------------------------------------

output_path = "data/mastodon/processed/mastodon_processed.parquet"

# Select only the columns that exist in the augmented DataFrame

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
    #"mastodon_date",
    "mastodon_day_count",
]

augmented = augmented.select(*[c for c in selected if c in augmented.columns])



# ----------------------------------------------------------------------
# STEP 6: Feature Creation
# ----------------------------------------------------------------------


# -----------------------------------------------------------------------------
# STEP 7: Save output
# -----------------------------------------------------------------------------
print(f"\n=== STEP 5: Saving to Parquet ===")
augmented.withColumnRenamed("cve_id", "cve").write.mode("overwrite").parquet(output_path)
print(f"✅ Saved to {output_path}")

spark.catalog.clearCache()
print("✅ Spark cache cleared")