import pandas as pd

# Load the Parquet dataset (folder, but represents a single logical dataset)
df_parquet = pd.read_parquet('data/epss/processed/epss_processed.parquet')

# Load the CSV file
df_csv = pd.read_parquet('data/full_db/processed/final_full_data_parquet')

# Print the first few rows of each DataFrame
print("Parquet DataFrame head:")
print(df_parquet.head())

print("\nCSV DataFrame head:")
print(df_csv.head())

# Define the canonical column order that you want to compare.
canonical_order = ["cve", "date", "epss"]

# Reorder both DataFrames using the canonical order and reset the index.
df_parquet_reordered = df_parquet[canonical_order].reset_index(drop=True)
df_csv_reordered = df_csv[canonical_order].reset_index(drop=True)

# --- Convert columns to standard types ---
df_parquet_reordered["cve"] = df_parquet_reordered["cve"].astype(str)
df_csv_reordered["cve"] = df_csv_reordered["cve"].astype(str)

df_parquet_reordered["date"] = pd.to_datetime(df_parquet_reordered["date"])
df_csv_reordered["date"] = pd.to_datetime(df_csv_reordered["date"])

df_parquet_reordered["epss"] = df_parquet_reordered["epss"].astype(float)
df_csv_reordered["epss"] = df_csv_reordered["epss"].astype(float)

# --- Round the 'epss' column to 5 decimal places ---
df_parquet_reordered["epss"] = df_parquet_reordered["epss"].round(5)
df_csv_reordered["epss"] = df_csv_reordered["epss"].round(5)

# --- Shape and type inspection ---
print("\nParquet DataFrame shape:", df_parquet_reordered.shape)
print("CSV DataFrame shape:", df_csv_reordered.shape)

print("\nParquet DataFrame dtypes:")
print(df_parquet_reordered.dtypes)
print("\nCSV DataFrame dtypes:")
print(df_csv_reordered.dtypes)

# --- Comparison ---
if df_parquet_reordered.equals(df_csv_reordered):
    print("\nThe DataFrames are exactly the same after aligning column order, types, index, and rounding.")
else:
    print("\nThe DataFrames are different even after aligning column order, types, index, and rounding.")
    
    # Show the actual differences
    diff = df_parquet_reordered.compare(df_csv_reordered)
    print("\nDifferences between DataFrames:")
    print(diff)

from pandas.testing import assert_frame_equal

try:
    # Using check_exact=False and a tolerance value
    assert_frame_equal(df_parquet_reordered, df_csv_reordered, check_exact=False, atol=1e-8)
    print("The DataFrames are effectively equal within the specified tolerance.")
except AssertionError as e:
    print("Differences found:", e)
