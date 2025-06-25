import pandas as pd
import os

# Define input and output file paths
csv1_input = 'catalogs_raw/comments_reddit.csv'
csv2_input = 'catalogs_raw/submissions_reddit.csv'
csv1_output = 'data/reddit/scripts/comments_reddit.csv'
csv2_output = 'data/reddit/scripts/submissions_reddit.csv'
merged_output = 'processed_data/reddit_catalog.csv'  # Make sure to define this!

# Columns to remove from both CSVs
remove_cols = ['Title/Body', 'Author', 'Permalink', 'Content', 'URL']

# Initialize cleaned DataFrames
df1_cleaned = pd.DataFrame()
df2_cleaned = pd.DataFrame()

try:
    # Load and clean CSV 1
    df1 = pd.read_csv(csv1_input, delimiter=';', encoding='cp1252')
    df1_cleaned = df1.drop(columns=remove_cols, errors='ignore')
    df1_cleaned = df1_cleaned.rename(columns={'CVE ID': 'CVE_ID'})
    df1_cleaned.to_csv(csv1_output, index=False, sep=';', encoding='utf-8')
    print("CSV 1 cleaned and saved.")
except FileNotFoundError:
    print(f"CSV 1 not found at: {csv1_input}")
except Exception as e:
    print(f"Error processing CSV 1: {e}")

try:
    # Load and clean CSV 2
    df2 = pd.read_csv(csv2_input, delimiter=';', encoding='cp1252')
    df2_cleaned = df2.drop(columns=remove_cols, errors='ignore')
    df2_cleaned = df2_cleaned.rename(columns={'CVE ID': 'CVE_ID'})
    df2_cleaned.to_csv(csv2_output, index=False, sep=';', encoding='utf-8')
    print("CSV 2 cleaned and saved.")
except FileNotFoundError:
    print(f"CSV 2 not found at: {csv2_input}")
except Exception as e:
    print(f"Error processing CSV 2: {e}")

try:
    # Merge both cleaned DataFrames
    merged = pd.concat([df1_cleaned, df2_cleaned], ignore_index=True)

    # Convert 'Date' to datetime with dayfirst=True
    merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce', dayfirst=True)

    # Localize to UTC (assume dates are UTC if no tz info)
    merged['Date'] = merged['Date'].dt.tz_localize('UTC')

    # Format date as ISO 8601 with milliseconds and Z timezone
    merged['Date'] = merged['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str[:-3] + 'Z'

    # Save merged DataFrame
    merged.to_csv(merged_output, index=False, sep=',', encoding='utf-8')
    print("Merged data cleaned and saved.")
except Exception as e:
    print(f"Error merging and formatting date: {e}")
