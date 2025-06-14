from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, mean, stddev, min, max, 
    approx_count_distinct, skewness, kurtosis,
    year, month, dayofmonth, dayofweek,
    desc, asc
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from t3_spark.session import get_spark_session

def analyze_dataframe(spark, parquet_path):
    """
    Perform comprehensive EDA on the final dataframe.
    """
    print("\n=== Loading Data ===")
    df = spark.read.parquet(parquet_path)
    
    print("\n=== Basic Information ===")
    print(f"Number of rows: {df.count():,}")
    print("\nSchema:")
    df.printSchema()
    
    print("\n=== Column Statistics ===")
    # Get all numeric columns
    numeric_cols = [field.name for field in df.schema.fields 
                   if isinstance(field.dataType, (IntegerType, DoubleType))]
    
    # Get all string columns
    string_cols = [field.name for field in df.schema.fields 
                  if isinstance(field.dataType, StringType)]
    
    # Get all timestamp columns
    timestamp_cols = [field.name for field in df.schema.fields 
                     if isinstance(field.dataType, TimestampType)]
    
    print("\nNumeric Columns Statistics:")
    if numeric_cols:
        # Calculate statistics for each numeric column
        for col_name in numeric_cols:
            print(f"\nStatistics for {col_name}:")
            stats = df.select(
                mean(col_name).alias("mean"),
                stddev(col_name).alias("stddev"),
                min(col_name).alias("min"),
                max(col_name).alias("max"),
                skewness(col_name).alias("skewness"),
                kurtosis(col_name).alias("kurtosis")
            )
            stats.show(truncate=False)
    
    print("\nString Columns Cardinality:")
    if string_cols:
        for col_name in string_cols:
            distinct_count = df.select(approx_count_distinct(col_name)).collect()[0][0]
            print(f"{col_name}: {distinct_count:,} distinct values")
            
            # Show top 5 most frequent values
            print(f"\nTop 5 most frequent values in {col_name}:")
            df.groupBy(col_name).count().orderBy(desc("count")).show(5, truncate=False)
    
    print("\n=== Temporal Analysis ===")
    if timestamp_cols:
        for col_name in timestamp_cols:
            print(f"\nTemporal patterns for {col_name}:")
            temporal_stats = df.select(
                year(col_name).alias("year"),
                month(col_name).alias("month"),
                dayofmonth(col_name).alias("day"),
                dayofweek(col_name).alias("day_of_week")
            )
            
            print("\nDistribution by year:")
            temporal_stats.groupBy("year").count().orderBy("year").show()
            
            print("\nDistribution by month:")
            temporal_stats.groupBy("month").count().orderBy("month").show()
            
            print("\nDistribution by day of week:")
            temporal_stats.groupBy("day_of_week").count().orderBy("day_of_week").show()
    
    print("\n=== Missing Values Analysis ===")
    for field in df.schema.fields:
        null_count = df.filter(col(field.name).isNull()).count()
        if null_count > 0:
            print(f"{field.name}: {null_count:,} null values ({(null_count/df.count())*100:.2f}%)")
    
    print("\n=== Correlation Analysis ===")
    if len(numeric_cols) > 1:
        corr_matrix = df.select(numeric_cols).stat.corr()
        print("\nCorrelation Matrix:")
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Only show upper triangle
                    print(f"{col1} vs {col2}: {corr_matrix[i][j]:.3f}")

def main():
    # Initialize Spark session
    spark = get_spark_session("EDA_Final_Dataframe")
    
    try:
        # Path to the final parquet file
        parquet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../processed/final_full_data.parquet'))
        
        # Perform analysis
        analyze_dataframe(spark, parquet_path)
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main() 