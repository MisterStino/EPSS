import gzip
import pandas as pd

file_path = "data/raw/epss_scores-2021-04-15.csv.gz"

# Open the .gz file and read the first few lines
with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    for i, line in enumerate(f):
        print(f"Line {i+1}: {line.strip()}")


# Read full content as a Pandas DataFrame
df = pd.read_csv(file_path, compression='gzip')
print("\n[INFO] DataFrame Loaded:")
print(df.head(100))  # Show first few rows
