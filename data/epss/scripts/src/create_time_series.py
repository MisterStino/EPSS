import os
import glob
import re
import pandas as pd

import os
import glob
import re
import pandas as pd

def create_single_time_series_csv(
    raw_folder='data/epss/raw',
    output_folder='data/epss/cve-time-series',
    output_filename='all_cves_time_series.csv'
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    files = glob.glob(os.path.join(raw_folder, '*.csv.gz'))

    # List of dataframes
    df_list = []

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing file: {filename}")

        match = re.match(r'epss_scores-(\d{4}-\d{2}-\d{2})\.csv\.gz', filename)
        if not match:
            print(f"WARNING: Could not parse date from filename: {filename}")
            continue
        file_date = match.group(1)

        # Read CSV once; skip meta row
        df = pd.read_csv(file_path, skiprows=1, compression='gzip')

        # Add a 'date' column
        df["date"] = file_date

        # Keep only relevant columns (cve, date, epss)
        df_list.append(df[["cve", "date", "epss"]])

    # Concatenate all the smaller dataframes
    if not df_list:
        print("No data found!")
        return

    big_df = pd.concat(df_list, ignore_index=True)

    # Now sort by cve, date
    big_df.sort_values(by=["cve", "date"], inplace=True)

    # Write to output
    output_path = os.path.join(output_folder, output_filename)
    big_df.to_csv(output_path, index=False)
    print(f"Consolidated file written to: {output_path}")


# Example usage:
if __name__ == "__main__":
    create_single_time_series_csv()
