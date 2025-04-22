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

if __name__ == "__main__":
    sample_1000_cve_timeseries()
