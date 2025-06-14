from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, mean, stddev, min, max, 
    collect_list, array_distinct, size,
    datediff, split, cast,
    percentile_approx, when, lit, desc, asc,
    window, expr, lag, lead, first, last,
    year, month, dayofmonth, dayofweek,
    countDistinct, sum, avg, variance,
    regexp_replace, concat_ws, array, struct
)
from pyspark.sql.types import DoubleType, IntegerType, StringType, BooleanType, ArrayType
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
from typing import List, Dict, Tuple, Any
def create_spark_session():
    """Create and configure Spark session using t3_spark."""
    from t3_spark.session import get_spark_session
    return get_spark_session("CVE Time Series Analysis")

def analyze_column_efficient(df, column_name, group_name="Full Dataset"):
    """Memory-efficient analysis of a single column."""
    try:
        field = df.schema[column_name]
        
        # Use sampling for very large datasets to get approximate statistics
        sample_df = df.sample(0.5, seed=42)  # 1% sample for efficiency
        total_rows = df.count()
        
        base_stats = {
            "column": column_name,
            "group": group_name,
            "type": str(field.dataType),
            "total_rows": total_rows,
        }
        
        # Calculate null counts efficiently
        null_count = df.filter(col(column_name).isNull()).count()
        base_stats["null_count"] = null_count
        base_stats["null_percentage"] = (null_count / total_rows) * 100
        base_stats["non_null_count"] = total_rows - null_count
        base_stats["non_null_percentage"] = ((total_rows - null_count) / total_rows) * 100
        
        if isinstance(field.dataType, (DoubleType, IntegerType)):
            # Numeric column analysis using sample
            stats = sample_df.select(
                mean(col(column_name)).alias("mean"),
                stddev(col(column_name)).alias("stddev"),
                min(col(column_name)).alias("min"),
                max(col(column_name)).alias("max"),
                percentile_approx(col(column_name), 0.5).alias("median"),
                countDistinct(col(column_name)).alias("unique_values")
            ).collect()[0]
            
            base_stats.update({
                "mean": stats["mean"],
                "stddev": stats["stddev"],
                "min": stats["min"],
                "max": stats["max"],
                "median": stats["median"],
                "unique_values": stats["unique_values"],
                "range": stats["max"] - stats["min"] if stats["max"] and stats["min"] else None
            })
            
            # Check for zeros (using sample)
            zero_count = sample_df.filter(col(column_name) == 0).count()
            base_stats["zero_count_sample"] = zero_count
            base_stats["zero_percentage_sample"] = (zero_count / sample_df.count()) * 100
            
        elif isinstance(field.dataType, BooleanType):
            # Boolean column analysis
            true_count = df.filter(col(column_name) == True).count()
            false_count = df.filter(col(column_name) == False).count()
            
            base_stats.update({
                "true_count": true_count,
                "false_count": false_count,
                "true_percentage": (true_count / total_rows) * 100,
                "false_percentage": (false_count / total_rows) * 100,
                "unique_values": 2 if true_count > 0 and false_count > 0 else 1
            })
            
        elif isinstance(field.dataType, StringType):
            # String column analysis using sample
            unique_count = sample_df.select(countDistinct(col(column_name))).collect()[0][0]
            
            # Get top 5 most frequent values from sample
            top_values = sample_df.groupBy(column_name).count().orderBy(desc("count")).limit(5).collect()
            for i, row in enumerate(top_values):
                base_stats[f"top_{i+1}_value"] = row[column_name]
                base_stats[f"top_{i+1}_count"] = row["count"]
            
            # Check for empty strings using sample
            empty_count = sample_df.filter(col(column_name) == "").count()
            
            base_stats.update({
                "unique_values_sample": unique_count,
                "empty_string_count_sample": empty_count,
                "empty_string_percentage_sample": (empty_count / sample_df.count()) * 100
            })
            
        elif isinstance(field.dataType, ArrayType):
            # Array column analysis using sample
            stats = sample_df.select(
                mean(size(col(column_name))).alias("avg_array_length"),
                min(size(col(column_name))).alias("min_array_length"),
                max(size(col(column_name))).alias("max_array_length"),
                stddev(size(col(column_name))).alias("stddev_array_length")
            ).collect()[0]
            
            empty_array_count = sample_df.filter(size(col(column_name)) == 0).count()
            
            base_stats.update({
                "avg_array_length": stats["avg_array_length"],
                "min_array_length": stats["min_array_length"],
                "max_array_length": stats["max_array_length"],
                "stddev_array_length": stats["stddev_array_length"],
                "empty_array_count_sample": empty_array_count,
                "empty_array_percentage_sample": (empty_array_count / sample_df.count()) * 100
            })
            
        return base_stats
        
    except Exception as e:
        return {
            "column": column_name,
            "group": group_name,
            "analysis_error": str(e)
        }

def analyze_cve_groups_efficient(df):
    """Efficiently compute CVE groups without loading all data into memory."""
    print("Computing CVE groups efficiently...")
    
    # Use aggregation to compute group membership efficiently - NO CACHING
    from pyspark.sql.functions import min as spark_min, max as spark_max
    cve_epss_stats = df.groupBy("cve").agg(
        spark_min("epss").alias("min_epss"),
        spark_max("epss").alias("max_epss"),
        first("epss").alias("initial_epss"),
        count("*").alias("record_count")
    )  # Removed .cache()
    
    print("Computing group sizes...")
    
    # Count group sizes efficiently
    group_a_count = cve_epss_stats.filter(col("max_epss") <= 0.7).count()
    group_b_count = cve_epss_stats.filter(col("initial_epss") > 0.7).count()
    group_c_count = cve_epss_stats.filter(
        (col("initial_epss") < 0.3) & (col("max_epss") > 0.7)
    ).count()
    
    print(f"Group A (EPSS never > 0.7): {group_a_count:,} CVEs")
    print(f"Group B (EPSS > 0.7 at start): {group_b_count:,} CVEs")
    print(f"Group C (EPSS starts < 0.3, later > 0.7): {group_c_count:,} CVEs")
    
    # Sample CVEs from each group for detailed analysis
    # Calculate sample size, ensuring at least 1
    calculated_sample = group_a_count // 200
    sample_size = 500 if calculated_sample > 500 else (1 if calculated_sample < 1 else calculated_sample)
    
    print(f"Sampling {sample_size} CVEs from each group for detailed analysis...")
    
    try:
        group_a_sample = (cve_epss_stats
                         .filter(col("max_epss") <= 0.7)
                         .sample(0.05, seed=42)  # Smaller sample
                         .limit(sample_size)
                         .select("cve")
                         .rdd.map(lambda x: x[0]).collect())
        print(f"Successfully sampled {len(group_a_sample)} CVEs from Group A")
    except Exception as e:
        print(f"Error sampling Group A: {str(e)}")
        group_a_sample = []
    
    try:
        group_b_limit = sample_size if sample_size < group_b_count else group_b_count
        group_b_sample = (cve_epss_stats
                         .filter(col("initial_epss") > 0.7)
                         .limit(group_b_limit)
                         .select("cve")
                         .rdd.map(lambda x: x[0]).collect())
        print(f"Successfully sampled {len(group_b_sample)} CVEs from Group B")
    except Exception as e:
        print(f"Error sampling Group B: {str(e)}")
        group_b_sample = []
    
    try:
        group_c_limit = sample_size if sample_size < group_c_count else group_c_count
        group_c_sample = (cve_epss_stats
                         .filter((col("initial_epss") < 0.3) & (col("max_epss") > 0.7))
                         .limit(group_c_limit)
                         .select("cve")
                         .rdd.map(lambda x: x[0]).collect())
        print(f"Successfully sampled {len(group_c_sample)} CVEs from Group C")
    except Exception as e:
        print(f"Error sampling Group C: {str(e)}")
        group_c_sample = []
    
    # No unpersist needed since we didn't cache
    
    return {
        "Group_A": (group_a_sample, group_a_count, "EPSS never > 0.7"),
        "Group_B": (group_b_sample, group_b_count, "EPSS > 0.7 at start"),
        "Group_C": (group_c_sample, group_c_count, "EPSS starts < 0.3, later > 0.7")
    }

def analyze_cve_time_series(spark, parquet_path):
    """Main function to analyze CVE time series data efficiently."""
    print("Loading data...")
    try:
        df = spark.read.parquet(parquet_path)
        total_rows = df.count()
        print(f"Successfully loaded data with {total_rows:,} rows")
        
        # Print schema for verification
        print("\nDataFrame Schema:")
        df.printSchema()
        
        # NO CACHING - removed df.cache()
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(parquet_path), "eda_results")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")
    
    # Get all column names
    all_columns = [f.name for f in df.schema.fields]
    print(f"Total columns to analyze: {len(all_columns)}")
    
    # 1. FULL DATAFRAME EDA - Analyze every single column efficiently
    print("\n" + "="*50)
    print("PERFORMING FULL DATAFRAME EDA")
    print("="*50)
    
    full_eda_results = []
    for i, column in enumerate(all_columns, 1):
        print(f"Analyzing column {i}/{len(all_columns)}: {column}")
        try:
            stats = analyze_column_efficient(df, column, "Full Dataset")
            full_eda_results.append(stats)
        except Exception as e:
            print(f"Warning: Could not analyze column {column}: {str(e)}")
            full_eda_results.append({
                "column": column,
                "group": "Full Dataset",
                "analysis_error": str(e)
            })
    
    # Save full dataframe EDA
    full_eda_df = pd.DataFrame(full_eda_results)
    full_eda_path = os.path.join(output_dir, "full_dataframe_eda.csv")
    full_eda_df.to_csv(full_eda_path, index=False)
    print(f"\nSaved full dataframe EDA to: {full_eda_path}")
    
    # 2. GROUP-BASED ANALYSIS
    print("\n" + "="*50)
    print("COMPUTING CVE GROUPS")
    print("="*50)
    
    try:
        groups = analyze_cve_groups_efficient(df)
        
        # 3. ANALYZE EACH GROUP
        for group_code, (cve_sample, total_count, description) in groups.items():
            if len(cve_sample) == 0:
                print(f"\nSkipping {group_code} - no CVEs in sample")
                continue
                
            print(f"\n" + "="*50)
            print(f"ANALYZING {group_code}: {description}")
            print(f"Total CVEs in group: {total_count:,}")
            print(f"Sample size for analysis: {len(cve_sample):,}")
            print("="*50)
            
            # Filter dataframe for this group sample - NO CACHING
            group_df = df.filter(col("cve").isin(cve_sample))
            
            # Column-wise EDA for this group
            print(f"\nPerforming column-wise EDA for {group_code}...")
            group_eda_results = []
            for i, column in enumerate(all_columns, 1):
                print(f"  Analyzing column {i}/{len(all_columns)}: {column}")
                try:
                    stats = analyze_column_efficient(group_df, column, group_code)
                    group_eda_results.append(stats)
                except Exception as e:
                    print(f"    Warning: Could not analyze column {column}: {str(e)}")
                    group_eda_results.append({
                        "column": column,
                        "group": group_code,
                        "analysis_error": str(e)
                    })
            
            # Save group EDA
            group_eda_df = pd.DataFrame(group_eda_results)
            group_eda_path = os.path.join(output_dir, f"{group_code}_column_eda.csv")
            group_eda_df.to_csv(group_eda_path, index=False)
            print(f"Saved {group_code} column EDA to: {group_eda_path}")
            
            # Basic time series summary for this group
            print(f"\nComputing time series summary for {group_code}...")
            try:
                from pyspark.sql.functions import min as spark_min, max as spark_max
                ts_summary = group_df.groupBy("cve").agg(
                    count("*").alias("series_length"),
                    spark_min("date").alias("first_date"),
                    spark_max("date").alias("last_date"),
                    mean("epss").alias("mean_epss"),
                    stddev("epss").alias("stddev_epss"),
                    spark_min("epss").alias("min_epss"),
                    spark_max("epss").alias("max_epss")
                ).toPandas()
                
                ts_path = os.path.join(output_dir, f"{group_code}_time_series_summary.csv")
                ts_summary.to_csv(ts_path, index=False)
                print(f"Saved {group_code} time series summary to: {ts_path}")
                
            except Exception as e:
                print(f"Error in time series summary for {group_code}: {str(e)}")
    
    except Exception as e:
        print(f"Error in group analysis: {str(e)}")
    
    # NO df.unpersist() since we didn't cache
    
    print(f"\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"All results saved to: {output_dir}")

def main():
    """Main function to run the analysis."""
    spark = create_spark_session()
    try:
        # Get the absolute path to the parquet file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parquet_path = os.path.join(current_dir, "../processed/final_full_data.parquet")
        
        if not os.path.exists(parquet_path):
            print(f"Error: Parquet file not found at {parquet_path}")
            return
            
        analyze_cve_time_series(spark, parquet_path)
    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 