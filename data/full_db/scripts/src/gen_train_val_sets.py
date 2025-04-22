#!/usr/bin/env python
"""
split_epss_by_calendar.py

Reads the full EPSS long‑format parquet and writes
calendar‑consistent train / val / test splits,
using fixed input/output strings.
"""

from pathlib import Path
from pyspark.sql import functions as F, types as T
from t3_spark.session import get_spark_session


def load_and_cast(spark, path: str):
    df = spark.read.parquet(path)
    df = df.withColumn("date", F.col("date").cast(T.DateType()))
    return df.select("cve", "date", "epss", "age_epss_pub")


def compute_cutoffs(df_dates):
    days = [r["date"] for r in df_dates.collect()]
    n = len(days)
    test_idx = int(0.8 * n)
    val_idx  = int(0.8 * test_idx)
    return days[val_idx], days[test_idx]


def write_split(df, out_dir: Path, name: str, n_files: int):
    target = out_dir / name
    (
      df.repartition(n_files)
        .write
        .mode("overwrite")
        .parquet(str(target))
    )
    print(f"[INFO] ✅ wrote {name} → {target}")


def split_epss_by_calendar(
    spark,
    input_path: str,
    output_base: str,
    n_files: int = 64
):
    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load & cast
    df = load_and_cast(spark, input_path).cache()
    print(f"[INFO] full rows: {df.count():,}")

    # 2) get distinct days
    df_days = df.select("date").distinct().orderBy("date")
    print(f"[INFO] distinct calendar days: {df_days.count():,}")

    # 3) compute cutoffs
    val_start, test_start = compute_cutoffs(df_days)
    print(f"[INFO] cut‑offs → val from {val_start} | test from {test_start}")

    # 4) split by date
    df_train = df.filter(F.col("date") <  F.lit(val_start))
    df_val   = df.filter((F.col("date") >= F.lit(val_start)) &
                         (F.col("date") <  F.lit(test_start)))
    df_test  = df.filter(F.col("date") >= F.lit(test_start))

    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"[INFO] {name:<5} rows: {subset.count():,}")

    # 5) write
    write_split(df_train, out_dir, "train", n_files)
    write_split(df_val,   out_dir, "val",   n_files)
    write_split(df_test,  out_dir, "test",  n_files)


if __name__ == "__main__":
    INPUT   = "data/full_db/sampled/final_full_data_sampled.parquet"
    OUTPUT  = "data/full_db/ml-sets-sampled/raw"
    NFILES  = 64

    spark = get_spark_session(app_name="EPSS_calendar_split")
    split_epss_by_calendar(spark, INPUT, OUTPUT, NFILES)
    spark.stop()
    print("[INFO] all done ✅")
