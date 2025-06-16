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
    random.seed(42)
    sampledB = random.sample(listB, nB)
    sampledC = random.sample(listC, nC)

    sampled = listA + sampledB + sampledC
    print(f"[INFO] sampling: {nA} from A + {nB} from B + {nC} from C = {len(sampled)} total")

    # 7) filter full df and write
    df_sampled = df.filter(F.col("cve").isin(sampled))
    df_sampled.write.mode("overwrite").parquet(output_path)
    print(f"[INFO] Written sampled time‐series ({len(sampled)} CVEs) to:\n  {output_path}")

    spark.stop()


def truncate_time_series_by_date(
    input_path: str = "data/full_db/processed/final_full_data.parquet",
    output_path: str = "data/full_db/processed/final_full_data_truncated.parquet", 
    start_date: str = "2021-01-01",
    end_date: str = "2023-12-31"
):
    """
    Truncates the time series for each CVE to only include dates within [start_date, end_date].
    Each CVE's time series will be clipped to the specified date range.
    
    Parameters:
    -----------
    input_path : str
        Path to the input parquet file containing CVE time series data
    output_path : str  
        Path to write the temporally truncated dataset
    start_date : str
        Start date in 'YYYY-MM-DD' format (inclusive)
    end_date : str
        End date in 'YYYY-MM-DD' format (inclusive)
    """
    spark = get_spark_session()
    
    print(f"[INFO] Truncating time series to date range: {start_date} to {end_date}")
    
    # 1. Load the full dataset
    df = spark.read.parquet(input_path)
    original_count = df.count()
    original_cves = df.select("cve").distinct().count()
    
    print(f"[INFO] Original dataset: {original_count:,} rows, {original_cves:,} CVEs")
    
    # 2. Apply date range filter
    df_filtered = df.filter(
        (F.col("date") >= F.lit(start_date)) & 
        (F.col("date") <= F.lit(end_date))
    )
    
    # 3. Get statistics on filtered data
    filtered_count = df_filtered.count()
    filtered_cves = df_filtered.select("cve").distinct().count()
    
    # 4. Get actual date range in filtered data
    date_stats = df_filtered.agg(
        F.min("date").alias("actual_start"),
        F.max("date").alias("actual_end")
    ).collect()[0]
    
    print(f"[INFO] Filtered dataset: {filtered_count:,} rows, {filtered_cves:,} CVEs")
    print(f"[INFO] Actual date range: {date_stats['actual_start']} to {date_stats['actual_end']}")
    print(f"[INFO] Retention: {filtered_count/original_count:.2%} rows, {filtered_cves/original_cves:.2%} CVEs")
    
    # 5. Write the truncated dataset
    df_filtered.write.mode("overwrite").parquet(output_path)
    print(f"[INFO] Truncated dataset written to: {output_path}")
    
    spark.stop()
    return {
        "original_rows": original_count,
        "original_cves": original_cves, 
        "filtered_rows": filtered_count,
        "filtered_cves": filtered_cves,
        "actual_start": str(date_stats['actual_start']),
        "actual_end": str(date_stats['actual_end'])
    }


if __name__ == "__main__":
    # sample_9000_cve_timeseries()
    # Sample 30k CVEs from your minimal dataset
    # sample_9000_cve_timeseries(
    #     input_path="data/full_db/v1/data/minimal_v1_timeseries.parquet",
    #     output_path="data/full_db/v1/data/minimal_v1_timeseries_sample.parquet",
    #     num_cve=10000
    # )

    truncate_time_series_by_date(
        input_path="data/full_db/sampled/final_full_data_sampled.parquet",
        output_path="data/full_db/sampled/final_full_data_sampled_truncated.parquet",
        start_date="2023-03-20",
        end_date="2024-03-20"
    )
    # sample_1000_cve_timeseries(num_cve=50000)
