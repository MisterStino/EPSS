# spark_session.py

import os
import sys
from pyspark.sql import SparkSession



def get_spark_session(app_name="MyApp", master="local[*]", extra_configs=None):
    """
    Creates and returns a SparkSession with the necessary configuration.

    Parameters:
      app_name (str): Name for the Spark application.
      master (str): Spark master URL (default: local[*]).
      extra_configs (dict): Additional configuration as key-value pairs.

    Returns:
      SparkSession: A configured Spark session.
    """
    # Ensure the correct Python interpreter is used for driver and workers.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # Start building the Spark session.
    builder = SparkSession.builder.appName(app_name).master(master)
    
    # Add any additional configuration options if provided.
    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, value)
    
    spark = (
        builder
            .appName("MyApp")
            .master("local[*]")
            .config("spark.driver.memory", "14g")
            .config("spark.executor.memory", "14g")
            .getOrCreate()
        )
    return spark

if __name__ == "__main__":
    # Quick test to verify the session creation.
    spark = get_spark_session("TestApp")
    print("Spark session created successfully!")
    spark.stop()
