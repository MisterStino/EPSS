import pandas as pd
import os

# Define input and output file paths
csv1_input = 'catalogs_raw/comments_reddit.csv'
csv2_input = 'catalogs_raw/submissions_reddit.csv'
csv1_output = 'reddit_data/comments_reddit.csv'
csv2_output = 'reddit_data/submissions_reddit.csv'

# Columns to remove from both CSVs
remove_cols = ['Title/Body', 'Author', 'Permalink', 'Content', 'URL']

try:
    # Load and clean CSV 1
    df1 = pd.read_csv(csv1_input, delimiter=';', encoding='cp1252')
    df1_cleaned = df1.drop(columns=remove_cols, errors='ignore')
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
    df2_cleaned.to_csv(csv2_output, index=False, sep=';', encoding='utf-8')
    print("CSV 2 cleaned and saved.")
except FileNotFoundError:
    print(f"CSV 2 not found at: {csv2_input}")
except Exception as e:
    print(f"Error processing CSV 2: {e}")
