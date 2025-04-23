from pyspark.sql import SparkSession
import os
import glob
import shutil

# ---------- CONFIGURATION ----------
folder_path = "data/catalogs"         # Input folder with CSVs
output_dir = "data/temp_output"     # Temp output folder (created by Spark)
final_csv_path = "data/final_dataset/enriched_catalog.csv"  # Final CSV file location

# ---------- START SPARK ----------
spark = SparkSession.builder \
    .appName("Join Multiple CSVs on CVE_ID") \
    .getOrCreate()

# ---------- READ & JOIN CSVs ----------
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Start with the first file
main_df = spark.read.csv(csv_files[0], header=True, inferSchema=True)

# Join all other CSVs on 'CVE_ID'
for file in csv_files[1:]:
    temp_df = spark.read.csv(file, header=True, inferSchema=True)
    main_df = main_df.join(temp_df, on='CVE_ID', how='left')  # Use 'outer' or 'left' if needed

# ---------- SAVE TO A SINGLE CSV FILE ----------
main_df.coalesce(1).write.csv(output_dir, header=True, mode="overwrite")

# ---------- RENAME TO SINGLE CSV ----------
# Find the actual CSV file Spark created
csv_file = glob.glob(f"{output_dir}/part-*.csv")[0]

# Move it to final destination
shutil.move(csv_file, final_csv_path)

# Clean up Spark output directory
shutil.rmtree(output_dir)

print(f"✅ CSVs joined and saved to: {final_csv_path}")
