from t3_spark.session import get_spark_session
from pyspark.sql import functions as F
import os
import matplotlib.pyplot as plt
from t3_spark.session import get_spark_session
from pyspark.sql import functions as F

def explore_high_score_cves():
    # Initialize the Spark session using your helper function.
    spark = get_spark_session()
    
    # Read the Parquet file into a Spark DataFrame.
    parquet_path = "data/general_utils/files/high_score_newer_cves.parquet"
    df = spark.read.parquet(parquet_path)
    
    # Show the first 10 rows of the DataFrame.
    # This gives an overview of the data (i.e., the "head" of the DataFrame).
    print("DataFrame Head:")
    df.show(10, truncate=False)
    
    # Get all unique values from the "cve" column.
    # distinct() returns a new DataFrame with unique "cve" values.
    unique_cves_df = df.select("cve").distinct()
    
    # Collect the unique CVE values to the driver as a list.
    unique_cves = unique_cves_df.rdd.map(lambda row: row["cve"]).collect()
    
    # Print out all unique CVE identifiers.
    print("\nUnique CVE IDs:")
    for cve in unique_cves:
        print(cve)
    
    # Stop the Spark session when done.
    spark.stop()





def plot_random_cve_time_series(n=3):
    """
    Plots the entire time series of EPS scores for n random unique CVE IDs.
    
    The data is assumed to be in a Parquet file at:
        data/general_utils/files/high_score_newer_cves.parquet
    
    For each selected CVE:
      - We filter the data by the CVE.
      - Sort by the date column (composite key: cve, date).
      - Convert the filtered Spark DataFrame to a Pandas DataFrame.
      - Plot the time series using matplotlib.
    """
    # Initialize Spark session using your helper.
    spark = get_spark_session()
    
    # Define the path to the Parquet file.
    parquet_path = os.path.join('data', 'general_utils', 'files', 'high_score_newer_cves.parquet')
    
    # Read the Parquet file into a Spark DataFrame.
    df = spark.read.parquet(parquet_path)
    
    # Get n random unique CVE IDs.
    # We select the "cve" column, remove duplicates, order by a random value, and limit to n rows.
    distinct_cves = df.select("cve").distinct().orderBy(F.rand()).limit(n).collect()
    cve_list = [row["cve"] for row in distinct_cves]
    print("Selected CVE IDs:", cve_list)
    
    # Create a new matplotlib figure.
    plt.figure(figsize=(12, 8))
    
    # For each selected CVE, filter its time series, convert to pandas, and plot.
    for cve in cve_list:
        # Filter rows for the current CVE and order by date.
        cve_df = df.filter(F.col("cve") == cve).orderBy("date")
        # Convert the Spark DataFrame to a Pandas DataFrame.
        pandas_df = cve_df.toPandas()
        # (Optional) Sort the pandas DataFrame by date.
        pandas_df.sort_values("date", inplace=True)
        
        # Plot the time series: dates on the x-axis, epss on the y-axis.
        plt.plot(pandas_df["date"], pandas_df["epss"], marker='o', label=cve)
    
    # Customize the plot.
    plt.xlabel("Date")
    plt.ylabel("EPS Score")
    plt.title("Time Series of EPS Score for Random CVEs")
    plt.legend(title="CVE ID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Show the plot.
    plt.show()
    
    # Stop the Spark session.
    spark.stop()

if __name__ == "__main__":
    plot_random_cve_time_series(n=10)
    #explore_high_score_cves()
