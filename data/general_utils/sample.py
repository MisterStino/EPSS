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

    # filter data within date range (no caching to avoid memory pressure)
    df_filt = df.filter(F.col("date").between(start_date, end_date))

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


def sample_high_epss_and_jumps(
    input_path: str = "data/full_db/processed/final_full_data.parquet",
    output_path: str = "data/full_db/sampled/final_full_data_sampled.parquet",
    num_cve: int = 9000,
    high_epss_threshold: float = 0.7,
    jump_threshold: float = 0.3
):
    """
    Samples CVEs from the full time-series dataset according to:
      • Group A: CVEs with max EPSS ≥ high_epss_threshold (take ALL of these)
      • Group B: CVEs with max EPSS jump > jump_threshold (take ALL of these)  
      • Group C: Random CVEs to fill remaining slots up to num_cve
    
    Parameters:
    -----------
    input_path : str
        Path to the full dataset Parquet file
    output_path : str  
        Path to write the sampled dataset Parquet file
    num_cve : int
        Total number of CVEs to sample
    high_epss_threshold : float
        Threshold for high EPSS scores (default 0.7)
    jump_threshold : float
        Threshold for EPSS jumps (default 0.3)
    """
    spark = get_spark_session()
    
    print(f"[INFO] Sampling {num_cve} CVEs with high_epss≥{high_epss_threshold} and jumps>{jump_threshold}")
    
    # 1) Load the data
    df = spark.read.parquet(input_path)
    
    # 2) Group A: CVEs with max EPSS ≥ threshold
    df_max_epss = (
        df.groupBy("cve")
          .agg(F.max("epss").alias("max_epss"))
    )
    group_a = df_max_epss.filter(F.col("max_epss") >= high_epss_threshold).select("cve")
    list_a = [r.cve for r in group_a.collect()]
    n_a = len(list_a)
    print(f"[INFO] |Group A| = {n_a} (max EPSS ≥ {high_epss_threshold})")
    
    # 3) Group B: CVEs with max EPSS jump > threshold
    # Calculate consecutive EPSS differences for each CVE
    window_spec = Window.partitionBy("cve").orderBy("date")
    df_with_prev = df.withColumn("prev_epss", F.lag("epss").over(window_spec))
    df_with_jump = df_with_prev.withColumn("epss_jump", 
                                          F.col("epss") - F.col("prev_epss"))
    
    # Find max jump per CVE
    df_max_jump = (
        df_with_jump.groupBy("cve")
                    .agg(F.max("epss_jump").alias("max_jump"))
    )
    group_b = df_max_jump.filter(F.col("max_jump") > jump_threshold).select("cve")
    list_b = [r.cve for r in group_b.collect()]
    n_b = len(list_b)  
    print(f"[INFO] |Group B| = {n_b} (max EPSS jump > {jump_threshold})")
    
    # 4) Combine groups and remove duplicates
    combined_cves = list(set(list_a + list_b))
    n_combined = len(combined_cves)
    print(f"[INFO] |Combined A∪B| = {n_combined} (after removing duplicates)")
    
    # 5) Check if we need more CVEs
    if n_combined >= num_cve:
        # We have enough, just take the first num_cve from combined
        sampled_cves = combined_cves[:num_cve]
        print(f"[INFO] Taking first {num_cve} from combined groups")
    else:
        # Need to fill remaining slots with random CVEs
        remaining_needed = num_cve - n_combined
        
        # Get all CVEs not in combined groups
        all_cves = [r.cve for r in df.select("cve").distinct().collect()]
        remaining_cves = [cve for cve in all_cves if cve not in combined_cves]
        
        if len(remaining_cves) < remaining_needed:
            raise ValueError(
                f"Not enough CVEs to fill remaining {remaining_needed} slots. "
                f"Only {len(remaining_cves)} CVEs available after groups A and B."
            )
        
        # Random sample from remaining CVEs
        random_sample = random.sample(remaining_cves, remaining_needed)
        sampled_cves = combined_cves + random_sample
        
        print(f"[INFO] Sampling: {n_combined} from A∪B + {remaining_needed} random = {len(sampled_cves)} total")
    
    # 6) Filter full dataset and write
    df_sampled = df.filter(F.col("cve").isin(sampled_cves))
    df_sampled.write.mode("overwrite").parquet(output_path)
    print(f"[INFO] Written sampled time-series ({len(sampled_cves)} CVEs) to:\n  {output_path}")
    
    spark.stop()




from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def sample_by_temporal_behavior(
    input_path: str = "data/full_db/processed/final_full_data.parquet",
    output_path: str = "data/full_db/sampled/temporal_behavior_sampled.parquet",
    type_a_rate: float = 1.0,    # Keep 100% of high-dynamic CVEs
    type_b_rate: float = 0.3,    # Keep 30% of medium-dynamic CVEs  
    type_c_rate: float = 0.05,   # Keep 5% of low-dynamic CVEs
    stop_session: bool = True,
    verbose: bool = True
):
    """
    Sample CVEs based on temporal behavior patterns to improve LSTM training.
    
    This function addresses the core problem where LSTM models learn static patterns
    instead of temporal dynamics by strategically filtering the training data to
    emphasize CVEs with meaningful temporal variation.
    
    CVE Behavior Types:
    - Type A (High-Dynamic): amplitude >= 0.5 OR max_epss >= 0.7
    - Type B (Medium-Dynamic): 0.1 <= amplitude < 0.5 AND max_epss < 0.7  
    - Type C (Low-Dynamic): amplitude < 0.1 AND max_epss < 0.7
    - Type D (High-Risk Static): amplitude < 0.3 AND min_epss >= 0.7
    
    Output Dataset:
    - Same schema as input + 'behavior_type' column indicating CVE classification
    - Each row (cve, date) includes the behavior type for that CVE
    - Enables analysis of model performance by temporal behavior pattern
    
    Parameters:
    -----------
    input_path : str
        Path to the full dataset Parquet file in long format
    output_path : str
        Path to write the filtered dataset Parquet file
    type_a_rate : float
        Sampling rate for Type A CVEs (default 1.0 = keep all)
    type_b_rate : float
        Sampling rate for Type B CVEs (default 0.3 = keep 30%)
    type_c_rate : float
        Sampling rate for Type C CVEs (default 0.05 = keep 5%)
    stop_session : bool
        Whether to stop Spark session after completion
    verbose : bool
        Whether to print detailed progress information
        
    Returns:
    --------
    dict
        Summary statistics about the sampling process including behavior type distribution
    """
    
    # Initialize Spark session with optimized configuration
    spark = get_spark_session()
    
    if verbose:
        print(f"[INFO] 🚀 Starting temporal behavior-based sampling")
        print(f"[INFO] Input: {input_path}")
        print(f"[INFO] Output: {output_path}")
        print(f"[INFO] Sampling rates - Type A: {type_a_rate:.1%}, Type B: {type_b_rate:.1%}, Type C: {type_c_rate:.1%}")
    
    # ==========================================
    # STEP 1: Load and validate data
    # ==========================================
    
    if verbose:
        print(f"\n[STEP 1] 📁 Loading data from {input_path}")
    
    df = spark.read.parquet(input_path)
    
    # Ensure proper data types
    df = df.withColumn("date", F.to_date("date"))
    df = df.withColumn("epss", F.col("epss").cast("double"))
    
    # Basic validation
    original_rows = df.count()
    original_cves = df.select("cve").distinct().count()
    
    if verbose:
        print(f"[INFO] ✅ Loaded {original_rows:,} rows with {original_cves:,} unique CVEs")
    
    # Validate required columns exist
    required_cols = {"cve", "date", "epss"}
    actual_cols = set(df.columns)
    missing_cols = required_cols - actual_cols
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # ==========================================
    # STEP 2: Compute CVE variability metrics
    # ==========================================
    
    if verbose:
        print(f"\n[STEP 2] 📊 Computing temporal variability metrics per CVE")
    
    # 2a. Basic statistics per CVE
    cve_stats_df = df.groupBy("cve").agg(
        F.min("date").alias("first_seen"),
        F.max("date").alias("last_seen"),
        F.count("*").alias("observations"),
        F.avg("epss").alias("avg_epss"),
        F.min("epss").alias("min_epss"),
        F.max("epss").alias("max_epss")
    )
    
    # 2b. Add EPSS amplitude (total range)
    cve_stats_df = cve_stats_df.withColumn(
        "amplitude", F.col("max_epss") - F.col("min_epss")
    )
    
    # 2c. Standard deviation per CVE
    epss_stddev_df = df.groupBy("cve").agg(
        F.stddev("epss").alias("epss_stddev")
    )
    
    # 2d. Maximum one-day jump (temporal volatility)
    window_spec = Window.partitionBy("cve").orderBy("date")
    df_with_lag = df.withColumn("prev_epss", F.lag("epss").over(window_spec))
    df_with_lag = df_with_lag.withColumn("epss_jump", F.abs(F.col("epss") - F.col("prev_epss")))
    max_jump_df = df_with_lag.groupBy("cve").agg(
        F.max("epss_jump").alias("max_one_day_jump")
    )
    
    # 2e. Combine all variability metrics
    dfs_to_join = [cve_stats_df, epss_stddev_df, max_jump_df]
    final_cve_variability_df = reduce(
        lambda left, right: left.join(right, on="cve", how="left"), 
        dfs_to_join
    )
    
    if verbose:
        print(f"[INFO] ✅ Computed variability metrics for {final_cve_variability_df.count():,} CVEs")
    
    # ==========================================
    # STEP 3: Classify CVEs by behavior type
    # ==========================================
    
    if verbose:
        print(f"\n[STEP 3] 🏷️  Classifying CVEs by temporal behavior patterns")
    
    # Apply behavior classification logic
    final_labeled_df = final_cve_variability_df.withColumn(
        "behavior_type",
        F.when((F.col("amplitude") >= 0.5) | (F.col("max_epss") >= 0.7), "Type A")
        .when((F.col("amplitude") >= 0.1) & (F.col("amplitude") < 0.5) & (F.col("max_epss") < 0.7), "Type B")
        .when((F.col("amplitude") < 0.1) & (F.col("max_epss") < 0.7), "Type C")
        .when((F.col("amplitude") < 0.3) & (F.col("min_epss") >= 0.7), "Type D")
        .otherwise("Unclassified")
    )
    
    # Get behavior type counts
    behavior_counts = final_labeled_df.groupBy("behavior_type").count().collect()
    behavior_dict = {row.behavior_type: row["count"] for row in behavior_counts}
    
    if verbose:
        print(f"[INFO] 📈 CVE Behavior Classification:")
        for behavior_type in ["Type A", "Type B", "Type C", "Type D", "Unclassified"]:
            count = behavior_dict.get(behavior_type, 0)
            pct = (count / original_cves) * 100
            print(f"[INFO]   {behavior_type}: {count:,} CVEs ({pct:.1f}%)")
    
    # ==========================================  
    # STEP 4: Strategic sampling by behavior type
    # ==========================================
    
    if verbose:
        print(f"\n[STEP 4] 🎯 Strategic sampling by behavior type")
    
    # Sample each type according to specified rates
    type_a_df = final_labeled_df.filter(F.col("behavior_type") == "Type A")
    type_b_full = final_labeled_df.filter(F.col("behavior_type") == "Type B")
    type_c_full = final_labeled_df.filter(F.col("behavior_type") == "Type C")
    type_d_df = final_labeled_df.filter(F.col("behavior_type") == "Type D")
    
    # Apply sampling rates
    type_a_sampled = type_a_df.orderBy(F.rand()).limit(int(type_a_df.count() * type_a_rate))
    type_b_sampled = type_b_full.orderBy(F.rand()).limit(int(type_b_full.count() * type_b_rate))  
    type_c_sampled = type_c_full.orderBy(F.rand()).limit(int(type_c_full.count() * type_c_rate))
    type_d_sampled = type_d_df.orderBy(F.rand()).limit(int(type_d_df.count() * 1.0))  # Keep all Type D
    
    # Combine selected CVEs (preserving behavior type)
    training_cve_ids_df = (
        type_a_sampled.select("cve", "behavior_type")
        .unionByName(type_b_sampled.select("cve", "behavior_type"))
        .unionByName(type_c_sampled.select("cve", "behavior_type"))
        .unionByName(type_d_sampled.select("cve", "behavior_type"))
        .distinct()
    )
    
    selected_cves = training_cve_ids_df.count()
    
    if verbose:
        print(f"[INFO] 📊 Sampling Results:")
        print(f"[INFO]   Type A sampled: {type_a_sampled.count():,} / {type_a_df.count():,} ({type_a_rate:.1%})")
        print(f"[INFO]   Type B sampled: {type_b_sampled.count():,} / {type_b_full.count():,} ({type_b_rate:.1%})")
        print(f"[INFO]   Type C sampled: {type_c_sampled.count():,} / {type_c_full.count():,} ({type_c_rate:.1%})")
        print(f"[INFO]   Type D sampled: {type_d_sampled.count():,} / {type_d_df.count():,} (100.0%)")
        print(f"[INFO]   Total CVEs selected: {selected_cves:,}")
    
    # ==========================================
    # STEP 5: Filter full time series and save
    # ==========================================
    
    if verbose:
        print(f"\n[STEP 5] 💾 Filtering full time series and saving")
    
    # Join with full time series to get all rows for selected CVEs
    training_time_series_df = df.join(training_cve_ids_df, on="cve", how="inner")
    
    # Final validation
    final_rows = training_time_series_df.count()
    final_cves = training_time_series_df.select("cve").distinct().count()
    
    # Validate behavior type distribution in final dataset
    final_behavior_dist = training_time_series_df.select("cve", "behavior_type").distinct().groupBy("behavior_type").count().collect()
    final_behavior_dict = {row.behavior_type: row["count"] for row in final_behavior_dist}
    
    # Save the filtered dataset
    training_time_series_df.write.mode("overwrite").parquet(output_path)
    
    if verbose:
        print(f"[INFO] ✅ Saved filtered dataset to: {output_path}")
        print(f"[INFO] 📊 Final dataset: {final_rows:,} rows with {final_cves:,} CVEs")
        print(f"[INFO] 📉 Data reduction: {(1 - final_rows/original_rows):.1%} fewer rows, {(1 - final_cves/original_cves):.1%} fewer CVEs")
        print(f"[INFO] 🏷️  Final behavior type distribution:")
        for behavior_type in ["Type A", "Type B", "Type C", "Type D"]:
            count = final_behavior_dict.get(behavior_type, 0)
            pct = (count / final_cves) * 100 if final_cves > 0 else 0
            print(f"[INFO]   {behavior_type}: {count:,} CVEs ({pct:.1f}% of final dataset)")
    
    # ==========================================
    # STEP 6: Return summary statistics
    # ==========================================
    
    summary_stats = {
        "original_rows": original_rows,
        "original_cves": original_cves,
        "final_rows": final_rows,
        "final_cves": final_cves,
        "type_a_selected": type_a_sampled.count(),
        "type_b_selected": type_b_sampled.count(), 
        "type_c_selected": type_c_sampled.count(),
        "type_d_selected": type_d_sampled.count(),
        "total_selected": selected_cves,
        "row_reduction_pct": (1 - final_rows/original_rows) * 100,
        "cve_reduction_pct": (1 - final_cves/original_cves) * 100,
        "final_behavior_distribution": final_behavior_dict,
        "behavior_type_preserved": True  # Flag indicating behavior type is in final dataset
    }
    
    if stop_session:
        spark.stop()
    
    if verbose:
        print(f"\n[INFO] 🎉 Temporal behavior sampling completed successfully!")
        print(f"[INFO] Summary statistics: {summary_stats}")
    
    return summary_stats


if __name__ == "__main__":

    # Optional: Apply temporal truncation if needed
    truncate_time_series_by_date(
        input_path="data/full_db/processed/final_full_data.parquet",
        output_path="data/full_db/sampled/final_full_data_v3_v4truncated.parquet",
        start_date="2023-03-08",
        end_date="2025-04-15"
    )
    # Example: Use temporal behavior sampling to improve LSTM training
    # stats = sample_by_temporal_behavior(
    #     input_path="data/full_db/sampled/final_full_data_v3_v4truncated.parquet",
    #     output_path="data/full_db/sampled/temporal_behavior_training.parquet",
    #     type_a_rate=1.0,   # Keep all high-dynamic CVEs
    #     type_b_rate=0.3,   # Keep 30% of medium-dynamic CVEs
    #     type_c_rate=0.05,  # Keep 5% of low-dynamic CVEs
    #     verbose=True
    # )
    
    # Optional: Apply temporal truncation if needed
    # truncate_time_series_by_date(
    #     input_path="data/full_db/sampled/temporal_behavior_training.parquet",
    #     output_path="data/prod/final_full_data_v3_v4truncated_csaf_social.parquet",
    #     start_date="2023-03-08",
    #     end_date="2025-04-15"
    # )
    
    # Legacy sampling methods (commented out)
    # sample_9000_cve_timeseries()
    # sample_high_epss_and_jumps(
    #     input_path="data/full_db/processed/final_full_data.parquet",
    #     output_path="data/full_db/prod/prod.parquet",
    #     num_cve=2000
    # )
    # sample_1000_cve_timeseries(num_cve=80000)
