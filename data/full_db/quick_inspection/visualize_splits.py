#!/usr/bin/env python
"""
check_cve_split_batch.py

1) Defines check_cve_split() to plot a single CVE’s split vs full EPSS series.
2) Adds sample_and_check() to randomly pick:
       - 5 CVEs whose max EPSS < 0.7
       - 5 CVEs whose max EPSS >= 0.7
   then invoke check_cve_split() on each.
3) All inputs (paths, thresholds) are hard‑coded below.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from pyspark.sql import functions as F, types as T
from t3_spark.session import get_spark_session


def check_cve_split(
    cve_id: str,
    full_db_path: str,
    split_base_path: str,
    spark=None
):
    """
    Plot the given CVE’s EPSS over time, top = train/val/test stitches,
    bottom = original full series.
    """
    if spark is None:
        spark = get_spark_session()

    # --- Load full series ---
    full_sdf = (
        spark.read
             .parquet(full_db_path)
             .filter(F.col("cve") == F.lit(cve_id))
             .select("date", "epss")
             .withColumn("date", F.col("date").cast(T.DateType()))
             .orderBy("date")
    )
    full_pd = full_sdf.toPandas()
    full_pd["date"] = pd.to_datetime(full_pd["date"])

    # --- Load each split ---
    def load_split(name: str):
        sdf = (
            spark.read
                 .parquet(str(Path(split_base_path) / name))
                 .filter(F.col("cve") == F.lit(cve_id))
                 .select("date", "epss")
                 .withColumn("date", F.col("date").cast(T.DateType()))
                 .orderBy("date")
        )
        pdf = sdf.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        return pdf

    train_pd = load_split("train")
    val_pd   = load_split("val")
    test_pd  = load_split("test")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 6),
        gridspec_kw={"height_ratios": [1, 1]}
    )

    for df, label, color in (
        (train_pd, "train", "C0"),
        (val_pd,   "val",   "C1"),
        (test_pd,  "test",  "C2")
    ):
        ax1.plot(df["date"], df["epss"],
                 marker="o", linestyle="-",
                 label=label, color=color)
    ax1.set_title(f"CVE {cve_id} — split segments")
    ax1.set_ylabel("EPSS")
    ax1.legend(loc="upper left")

    ax2.plot(full_pd["date"], full_pd["epss"],
             marker="o", linestyle="-", color="k")
    ax2.set_title(f"CVE {cve_id} — full original series")
    ax2.set_ylabel("EPSS")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def sample_and_check(
    full_db_path: str,
    split_base_path: str,
    low_threshold: float = 0.7,
    sample_size: int = 5,
    spark=None
):
    """
    1) Loads full_db_path into Spark.
    2) Computes max EPS S per CVE.
    3) Splits CVEs into low (< low_threshold) and high (>=).
    4) Samples sample_size from each.
    5) Calls check_cve_split() on each selected CVE.
    """
    if spark is None:
        spark = get_spark_session(app_name="EPSS_Sample_Check")

    sdf = spark.read.parquet(full_db_path).select("cve", "epss")

    # max epss per CVE
    max_sdf = sdf.groupBy("cve").agg(F.max("epss").alias("max_epss"))

    low_cves = (
        max_sdf.filter(F.col("max_epss") < low_threshold)
               .select("cve")
               .distinct()
               .orderBy(F.rand())
               .limit(sample_size)
               .toPandas()["cve"]
               .tolist()
    )
    high_cves = (
        max_sdf.filter(F.col("max_epss") >= low_threshold)
               .select("cve")
               .distinct()
               .orderBy(F.rand())
               .limit(sample_size)
               .toPandas()["cve"]
               .tolist()
    )

    print(f"[INFO] sampled LOW  ({len(low_cves)})  CVEs: {low_cves}")
    print(f"[INFO] sampled HIGH ({len(high_cves)}) CVEs: {high_cves}")

    for cve in low_cves + high_cves:
        print(f"\n=== Plotting CVE {cve} ===")
        check_cve_split(cve, full_db_path, split_base_path, spark=spark)


if __name__ == "__main__":
    # ── Hard‑coded parameters ───────────────────────────────
    FULL_DB_PARQUET = "data/full_db/sampled/final_full_data_sampled.parquet"
    SPLIT_BASE_DIR  = "data/full_db/ml-sets-sampled/raw"
    LOW_THRESHOLD   = 0.7
    SAMPLE_SIZE     = 5
    # ────────────────────────────────────────────────────────

    spark = get_spark_session(app_name="EPSS_Sample_Check")
    sample_and_check(
        FULL_DB_PARQUET,
        SPLIT_BASE_DIR,
        low_threshold=LOW_THRESHOLD,
        sample_size=SAMPLE_SIZE,
        spark=spark
    )
    spark.stop()
    print("[INFO] batch checking complete ✅")
