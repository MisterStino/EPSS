import os
import random
from pyspark.sql import functions as F
from t3_spark.session import get_spark_session

def sample_1000_cve_timeseries(
    input_path="data/full_db/processed/final_full_data.parquet",
    output_path="data/full_db/sampled/final_full_data_sampled.parquet",
    num_cve=1000
):
    """
    Reads the full dataset in long format from input_path (with composite keys: cve and date),
    randomly samples `num_cve` distinct CVEs, and filters the dataset to include the complete time series
    for these CVEs. The resulting DataFrame (with the same columns) is written to output_path in Parquet format.
    
    Parameters:
    -----------
    input_path : str
        The path to the full dataset Parquet file.
    output_path : str
        The path to write the sampled dataset Parquet file.
    num_cve : int
        The number of distinct CVEs to sample.
    """
    spark = get_spark_session()
    
    # 1. Read the full dataset.
    df_full = spark.read.parquet(input_path)
    
    # 2. Extract distinct CVE ids.
    distinct_cves_df = df_full.select("cve").distinct()
    
    # Collect distinct CVE ids to driver (assuming the distinct set is reasonably small)
    distinct_cves = [row["cve"] for row in distinct_cves_df.collect()]
    
    # 3. Randomly sample the required number of CVEs.
    sampled_cves = random.sample(distinct_cves, num_cve)
    
    # 4. Filter the full DataFrame for rows corresponding to the sampled CVEs.
    df_sampled = df_full.filter(F.col("cve").isin(sampled_cves))
    
    # 5. Write the sampled DataFrame to the output path in Parquet format.
    df_sampled.write.mode("overwrite").parquet(output_path)
    print(f"Sampled dataset for {num_cve} CVEs written to: {output_path}")
    
    spark.stop()



import os, random
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from t3_spark.session import get_spark_session

def sample_9000_cve_timeseries(
    input_path:  str = "data/full_db/processed/final_full_data.parquet",
    output_path: str = "data/full_db/sampled/final_full_data_sampled.parquet",
    num_cve:     int = 9000
):
    """
    Samples exactly `num_cve` CVEs from the full time‐series dataset according to:
      • Group A: first_epss ≤ 0.3 AND max_epss ≥ 0.7 (take ALL of these)
      • Group B: remaining with max_epss ≥ 0.7 (sample half of the remainder)
      • Group C: remaining with max_epss <  0.7 (sample half of the remainder)
    Writes out the long‐format Parquet containing full series for the sampled CVEs.
    """
    spark = get_spark_session()

    # 1) load
    df = spark.read.parquet(input_path)

    # 2) find each CVE's start_date, first_epss, and max_epss
    df_start_date = (
        df.groupBy("cve")
          .agg(F.min("date").alias("start_date"))
    )
    df_first = (
        df_start_date
          .join(df, ["cve"])
          .filter(F.col("date") == F.col("start_date"))
          .select("cve", F.col("epss").alias("first_epss"))
    )
    df_max = (
        df.groupBy("cve")
          .agg(F.max("epss").alias("max_epss"))
    )
    per_cve = df_start_date\
        .join(df_first, "cve")\
        .join(df_max,   "cve")

    # 3) group membership
    condA = (F.col("first_epss") <= 0.3) & (F.col("max_epss") >= 0.7)
    groupA = per_cve.filter(condA).select("cve").distinct()
    listA  = [r.cve for r in groupA.collect()]
    nA     = len(listA)
    print(f"[INFO] |Group A| = {nA}  (first_epss≤0.3 ∧ max_epss≥0.7)")

    # 4) the remainder
    per_rem = per_cve.filter(~F.col("cve").isin(listA))
    groupB  = per_rem.filter(F.col("max_epss") >= 0.7).select("cve").distinct()
    groupC  = per_rem.filter(F.col("max_epss") <  0.7).select("cve").distinct()

    listB = [r.cve for r in groupB.collect()]
    listC = [r.cve for r in groupC.collect()]

    # 5) compute how many more to draw
    to_draw = num_cve - nA
    if to_draw < 0:
        raise ValueError(f"num_cve={num_cve} is smaller than |Group A|={nA}")
    nB = to_draw // 2
    nC = to_draw - nB

    if nB > len(listB) or nC > len(listC):
        raise ValueError(
            f"Not enough CVEs in B ({len(listB)}) or C ({len(listC)}) to fill "
            f"the remaining {to_draw} = {nB}+{nC} slots."
        )

    # 6) random sample (remove fixed seed if you want fresh draws each run)
    sampledB = random.sample(listB, nB)
    sampledC = random.sample(listC, nC)

    sampled = listA + sampledB + sampledC
    print(f"[INFO] sampling: {nA} from A + {nB} from B + {nC} from C = {len(sampled)} total")

    # 7) filter full df and write
    df_sampled = df.filter(F.col("cve").isin(sampled))
    df_sampled.write.mode("overwrite").parquet(output_path)
    print(f"[INFO] Written sampled time‐series ({len(sampled)} CVEs) to:\n  {output_path}")

    spark.stop()


from typing import Dict
from pathlib import Path
import pyspark.sql.functions as F
from t3_spark.session import get_spark_session


def truncate_time_series_by_date(
    input_path : str | Path = "data/full_db/processed/final_full_data.parquet",
    output_path: str | Path = "data/full_db/processed/final_full_data_truncated.parquet",
    start_date : str        = "2021-01-01",
    end_date   : str        = "2023-12-31",
    stop_session: bool      = True,          # set False if caller re-uses Spark
) -> Dict[str, str | int]:
    """
    Keep only rows whose *date* lies in the inclusive interval
    [start_date … end_date].  CVEs with zero rows in that window
    disappear – that is unavoidable with a strict slice.

    Returns a small dict with before/after statistics.
    """
    spark = get_spark_session()
    input_path  = str(input_path)
    output_path = str(output_path)

    print(f"[INFO] Truncating to {start_date} → {end_date}")

    df = spark.read.parquet(input_path)

    # make sure 'date' is of type DATE (avoids implicit cast surprises)
    if dict(df.dtypes)["date"] != "date":
        df = df.withColumn("date", F.to_date("date"))

    original_rows = df.count()
    original_cves = df.select("cve").distinct().count()
    print(f"[INFO] original : {original_rows:,} rows  | {original_cves:,} CVEs")

    # filter + cache so we scan the data only once afterwards
    df_filt = (
        df.filter(F.col("date").between(start_date, end_date))
          .cache()
    )

    filtered_rows = df_filt.count()
    filtered_cves = df_filt.select("cve").distinct().count()

    rng = df_filt.agg(
        F.min("date").alias("actual_start"),
        F.max("date").alias("actual_end")
    ).first()

    print(
        f"[INFO] filtered  : {filtered_rows:,} rows  | {filtered_cves:,} CVEs\n"
        f"[INFO] real range: {rng.actual_start} … {rng.actual_end}\n"
        f"[INFO] retention : {filtered_rows/original_rows:6.2%} rows  | "
        f"{filtered_cves/original_cves:6.2%} CVEs"
    )

    # write
    (df_filt
         .write
         .mode("overwrite")
         .parquet(output_path))
    print(f"[INFO] written   : {output_path}")

    if stop_session:
        spark.stop()

    return dict(
        original_rows  = original_rows,
        original_cves  = original_cves,
        filtered_rows  = filtered_rows,
        filtered_cves  = filtered_cves,
        actual_start   = str(rng.actual_start),
        actual_end     = str(rng.actual_end),
    )



if __name__ == "__main__":
    # sample_9000_cve_timeseries()
    # Sample 30k CVEs from your minimal dataset
    sample_9000_cve_timeseries(
        input_path="data/full_db/processed/final_full_data.parquet",
        output_path="data/full_db/v1/data/minimal_v1_timeseries_sample_checked.parquet",
        num_cve=9000
    )

    truncate_time_series_by_date(
        input_path="data/full_db/v1/data/minimal_v1_timeseries_sample_checked.parquet",
        output_path="data/prod/final_full_data_v3_v4truncated_plot.parquet",
        start_date="2023-03-08",
        end_date="2025-04-17"
    )

    # sample_1000_cve_timeseries(num_cve=80000)
