import os
import re
import glob
from functools import reduce

from pyspark.sql.functions import lit
from pyspark.sql import DataFrame

# We'll assume you have a helper that returns a SparkSession:
from t3_spark.session import get_spark_session

def create_big_parquet(
    input_folder='data/epss/uncompressed',
    output_folder='data/epss/epss_parquet',
    output_parquet='epss_all.parquet'
):
    spark = get_spark_session()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Get list of CSV files in input_folder
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))

    df_list = []

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")

        # Extract date from filename, e.g. 'epss_scores-2023-01-23.csv'
        match = re.match(r'epss_scores-(\d{4}-\d{2}-\d{2})\.csv', filename)
        if not match:
            print(f"WARNING: Could not parse date from filename: {filename}")
            continue
        file_date = match.group(1)

        # Read CSV directly. No need to skip metadata row anymore!
        df_temp = (
            spark.read
                 .option("header", "true")
                 .option("inferSchema", "true")
                 .option("mode", "FAILFAST")
                 .csv(file_path)
        )

        # Keep only the columns of interest and add the 'date' column.
        # In case there are other columns, we ignore them.
        df_temp = df_temp.select("cve", "epss").withColumn("date", lit(file_date))

        # Collect in a list to union all data later.
        df_list.append(df_temp)

    # If we found no valid CSV files, exit gracefully.
    if not df_list:
        print("No valid data found in the input folder.")
        return

    # Union all DataFrames into one final DataFrame
    final_df = reduce(DataFrame.union, df_list)

    # Sort the final DataFrame by cve and date.
    final_df = final_df.orderBy(["cve", "date"])

    # Write out to Parquet (overwrite mode).
    output_path = os.path.join(output_folder, output_parquet)
    final_df.write.mode("overwrite").parquet(output_path)

    print(f"Done. Consolidated Parquet written to {output_path}")

if __name__ == "__main__":
    create_big_parquet()
