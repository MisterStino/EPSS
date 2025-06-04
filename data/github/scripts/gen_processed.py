"""
Merge feature data onto the full EPSS time-series table.

▪ Keeps every row that exists in `epss_processed.parquet` (LEFT join).
▪ Normalises CVE strings (trim + UPPER) to avoid case-mismatch.
▪ De-duplicates the feature data on (cve, date) before the join.
▪ Fills missing feature-derived columns with 0 (semantic = "no activity").
▪ Writes result to `data/<source>/processed/<source>_processed.parquet`.

Usage:
    python gen_processed.py --source github --input-file github_commit_timestamps_9k.csv
    python gen_processed.py --source reddit --input-file reddit_features.parquet
"""

import argparse
import pyspark.sql.functions as F
from t3_spark.session import get_spark_session


# --------------------------------------------------------------------------- #
#  helper utilities
# --------------------------------------------------------------------------- #
def log_head(df, name: str, n: int = 10) -> None:
    """Pretty-print a small sample without the extra 'None'."""
    print(f"\n{name} (showing {n} rows)")
    df.show(n, truncate=False)


def count_rows(df, name: str) -> None:
    print(f"{name} row-count = {df.count():,}")


def detect_cve_column(columns):
    """Detect CVE column name from available columns."""
    cve_candidates = [c for c in columns if c.lower() in {"cve", "cve_id"}]
    if not cve_candidates:
        raise ValueError(f"No CVE column found. Expected 'cve' or 'cve_id', got: {columns}")
    return cve_candidates[0]


def detect_date_column(columns):
    """Detect date column name from available columns."""
    date_candidates = [c for c in columns if c.lower() == "date"]
    if not date_candidates:
        raise ValueError(f"No date column found. Expected 'date', got: {columns}")
    return date_candidates[0]


# --------------------------------------------------------------------------- #
#  main ETL
# --------------------------------------------------------------------------- #
def main(source: str, input_file: str) -> None:
    spark = get_spark_session()

    # ---------- Load base EPSS table ------------------------------------------------
    epss_path = "data/epss/processed/epss_processed.parquet"
    epss_df = spark.read.parquet(epss_path)
    epss_df = epss_df.withColumn("cve", F.trim(F.upper(F.col("cve")))) \
                     .withColumn("date", F.to_date(F.col("date")))

    # ---------- Load feature data ---------------------------------------------------
    feature_path = f"data/{source}/raw/{input_file}"
    
    # Generic loader for CSV or Parquet
    if feature_path.lower().endswith(".csv"):
        feature_df = (
            spark.read
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .csv(feature_path)
        )
    elif feature_path.lower().endswith(".parquet"):
        feature_df = spark.read.parquet(feature_path)
    else:
        raise ValueError(f"Unsupported file format. Expected .csv or .parquet, got: {feature_path}")

    # ---------- Harmonize column names ----------------------------------------------
    cve_col = detect_cve_column(feature_df.columns)
    date_col = detect_date_column(feature_df.columns)
    
    feature_df = (
        feature_df
        .withColumnRenamed(cve_col, "cve")
        .withColumnRenamed(date_col, "date")
        .withColumn("cve", F.trim(F.upper(F.col("cve"))))
        .withColumn("date", F.to_date(F.col("date")))
    )

    # ---------- Detect & drop duplicate keys ----------------------------------------
    dupes = (
        feature_df.groupBy("cve", "date")
                  .count()
                  .filter("count > 1")
    )
    if dupes.count() > 0:
        print(f"\nWARNING: duplicate (cve,date) rows found in {source} data — keeping first:")
        dupes.show(truncate=False)
        feature_df = feature_df.dropDuplicates(["cve", "date"])

    # ---------- Sanity-check before the join ---------------------------------------
    count_rows(epss_df, "EPSS")
    count_rows(feature_df, source.title())

    # ---------- LEFT join on the composite key -------------------------------------
    joined_df = epss_df.join(feature_df, on=["cve", "date"], how="left")

    # ---------- Fill nulls in the new feature columns with 0 -----------------------
    feature_cols = [c for c in feature_df.columns if c not in ("cve", "date")]
    joined_df = joined_df.fillna(0, subset=feature_cols)

    # ---------- Post-join validation ----------------------------------------------
    count_rows(joined_df, "Joined")
    assert (
        joined_df.count() == epss_df.count()
    ), "Row-count mismatch! The LEFT join must not drop or add rows."

    # ---------- Persist result -----------------------------------------------------
    out_path = f"data/{source}/processed/{source}_processed.parquet"
    joined_df.write.mode("overwrite").parquet(out_path)
    print(f"\n✓  Successfully wrote: {out_path}")

    # ---------- Quick visual spot-checks -------------------------------------------
    log_head(epss_df, "EPSS head")
    log_head(feature_df, f"{source.title()} head")
    log_head(joined_df, "Joined head (after fillna)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge feature data onto EPSS time-series table")
    parser.add_argument("--source", required=True, help="Data source name (e.g., 'github', 'reddit')")
    parser.add_argument("--input-file", required=True, help="Input filename in data/<source>/raw/")
    
    args = parser.parse_args()
    main(args.source, args.input_file)
