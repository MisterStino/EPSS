"""
Merge GitHub activity features onto the full EPSS time-series table.

▪ Keeps every row that exists in `epss_processed.parquet` (LEFT join).
▪ Normalises CVE strings (trim + UPPER) to avoid case-mismatch.
▪ De-duplicates the GitHub CSV on (cve, date) before the join.
▪ Fills missing GitHub-derived columns with 0 (semantic = “no activity”).
▪ Writes result to `data/github/processed/github_processed.parquet`.
"""

import pyspark.sql.functions as F
from t3_spark.session import get_spark_session


# --------------------------------------------------------------------------- #
#  helper utilities
# --------------------------------------------------------------------------- #
def log_head(df, name: str, n: int = 10) -> None:
    """Pretty-print a small sample without the extra ‘None’."""
    print(f"\n{name} (showing {n} rows)")
    df.show(n, truncate=False)


def count_rows(df, name: str) -> None:
    print(f"{name} row-count = {df.count():,}")


# --------------------------------------------------------------------------- #
#  main ETL
# --------------------------------------------------------------------------- #
def main() -> None:
    spark = get_spark_session()

    # ---------- Load base EPSS table ------------------------------------------------
    epss_path = "data/epss/processed/epss_processed.parquet"
    epss_df = spark.read.parquet(epss_path)
    epss_df = epss_df.withColumn("cve", F.trim(F.upper(F.col("cve")))) \
                     .withColumn("date", F.to_date(F.col("date")))

    # ---------- Load GitHub feature export -----------------------------------------
    github_path = "data/github/raw/github_commit_timestamps_9k.csv"
    github_df = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")
             .csv(github_path)
             .withColumnRenamed("cve_id", "cve")          # unify column names
             .withColumn("cve",  F.trim(F.upper(F.col("cve"))))
             .withColumn("date", F.to_date(F.col("date")))
    )

    # ---------- Detect & drop duplicate keys in the CSV ----------------------------
    dupes = (
        github_df.groupBy("cve", "date")
                 .count()
                 .filter("count > 1")
    )
    if dupes.count() > 0:
        print("\nWARNING: duplicate (cve,date) rows found in GitHub CSV — keeping first:")
        dupes.show(truncate=False)
        github_df = github_df.dropDuplicates(["cve", "date"])

    # ---------- Sanity-check before the join ---------------------------------------
    count_rows(epss_df,   "EPSS")
    count_rows(github_df, "GitHub")

    # ---------- LEFT join on the composite key -------------------------------------
    joined_df = epss_df.join(github_df, on=["cve", "date"], how="left")

    # ---------- Fill nulls in the new feature columns with 0 -----------------------
    feature_cols = [c for c in github_df.columns if c not in ("cve", "date")]
    joined_df = joined_df.fillna(0, subset=feature_cols)

    # ---------- Post-join validation ----------------------------------------------
    count_rows(joined_df, "Joined")
    assert (
        joined_df.count() == epss_df.count()
    ), "Row-count mismatch! The LEFT join must not drop or add rows."

    # ---------- Persist result -----------------------------------------------------
    out_path = "data/github/processed/github_processed.parquet"
    joined_df.write.mode("overwrite").parquet(out_path)
    print(f"\n✓  Successfully wrote: {out_path}")

    # ---------- Quick visual spot-checks -------------------------------------------
    log_head(epss_df,   "EPSS head")
    log_head(github_df, "GitHub head")
    log_head(joined_df, "Joined head (after fillna)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
