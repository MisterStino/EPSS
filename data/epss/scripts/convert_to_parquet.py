import os
import re
import gzip
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def convert_csv_gz_to_parquet(raw_folder, processed_folder):
    """
    Convert each .csv.gz file in `raw_folder` to a daily Parquet file in `processed_folder`.
    """
    # List all files in raw_folder
    files = os.listdir(raw_folder)

    # Regex to find files of the form epss_scores-YYYY-MM-DD.csv.gz
    pattern = r"epss_scores-(\d{4}-\d{2}-\d{2})\.csv\.gz"

    for f in files:
        match = re.match(pattern, f)
        if match:
            date_str = match.group(1)  # e.g. 2024-10-26

            # Full paths
            gz_path = os.path.join(raw_folder, f)
            daily_folder = os.path.join(processed_folder, date_str)
            os.makedirs(daily_folder, exist_ok=True)
            parquet_path = os.path.join(daily_folder, "epss.parquet")

            print(f"[INFO] Processing file {f} for date {date_str}")

            # Read CSV from gzip
            df = pd.read_csv(gz_path, compression="gzip", low_memory=False)
            
            # The original columns might look something like:
            # "#model_version:v2023.03.01 score_date:2024-10-26T00:00:00+0000", "cve", "epss", "percentile"
            # Let's rename them if needed.

            # Identify existing columns
            original_cols = list(df.columns)
            # Example typical columns might be:
            # 0: '#model_version:v2023.03.01 score_date:2024-10-26T00:00:00+0000'
            # 1: 'cve'
            # 2: 'epss'
            # 3: 'percentile'

            # If the first column is something weird, you can rename it:
            # For demonstration, let's rename them in a standardized way:
            rename_dict = {}
            for col in original_cols:
                # If we see 'epss' exactly, rename to 'epss_score'
                if col.strip().lower() == 'epss':
                    rename_dict[col] = 'epss_score'
                # If we see 'percentile', keep it or rename to 'epss_percentile'
                elif col.strip().lower() == 'percentile':
                    rename_dict[col] = 'epss_percentile'
                # If 'cve' is fine, we leave it
                # If anything else is suspicious, just keep it or rename:
                # e.g. rename the big model_version column
                elif '#model_version:' in col or 'score_date:' in col:
                    rename_dict[col] = 'model_info'
            
            df.rename(columns=rename_dict, inplace=True)

            # Optionally, add a 'date' column if you want it in your final dataset
            df['date'] = date_str  # so you have an explicit date field

            # Convert to Arrow Table
            table = pa.Table.from_pandas(df)

            # Write to Parquet
            pq.write_table(table, parquet_path, compression='snappy')
            
            print(f"[INFO] Wrote Parquet file to {parquet_path}")


if __name__ == "__main__":
    raw_folder = "data/epss/raw"
    processed_folder = "data/epss/processed/daily_parquet"

    convert_csv_gz_to_parquet(raw_folder, processed_folder)

    print("[INFO] Conversion to Parquet completed.")
