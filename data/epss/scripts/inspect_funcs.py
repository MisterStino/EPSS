import gzip
import pandas as pd
import os
import re
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def inspect_csv_gz(file_path, n_lines=100):
    """
    Reads a compressed CSV (.csv.gz) file into a Pandas DataFrame
    and prints the first n_lines rows.

    Parameters:
      file_path (str): Path to the .csv.gz file.
      n_lines (int): Number of rows to display.
    """
    try:
        df = pd.read_csv(file_path, compression='gzip', low_memory=False)
        print("\n[INFO] DataFrame loaded from CSV.GZ:")
        print(df.head(n_lines))
    except Exception as e:
        print(f"[ERROR] Could not load CSV.GZ file: {e}")

def inspect_parquet(file_path, n_lines=5):
    """
    Reads a Parquet file into a Pandas DataFrame
    and prints the first n_lines rows.

    Parameters:
      file_path (str): Path to the Parquet file.
      n_lines (int): Number of rows to display.
    """
    try:
        df = pd.read_parquet(file_path)
        print("\n[INFO] DataFrame loaded from Parquet:")
        print(df.head(n_lines))
    except Exception as e:
        print(f"[ERROR] Could not load Parquet file: {e}")


import os
import re
import pandas as pd
from datetime import datetime, timedelta

def analyze_cve_lifetimes(raw_folder="data/epss/raw"):
    """
    Analyze the lifetime of each CVE across the dataset.

    For each file (which represents a day), the function extracts the list of CVE IDs.
    Then, for each CVE, it determines:
      - The first and last day it appears.
      - Whether there are any missing days (gaps) in the continuous period from its first to last appearance.
    
    This analysis helps verify if a CVE, once it appears, is continuously tracked or if it disappears temporarily 
    or permanently.
    
    :param raw_folder: Directory containing the raw EPS scores files.
    :return: A dictionary with detailed lifetime information for further analysis.
    """
    # Regex pattern to match filenames: epss_scores-YYYY-MM-DD.csv.gz
    pattern = re.compile(r"^epss_scores-(\d{4}-\d{2}-\d{2})\.csv\.gz$")
    
    # Dictionary to store each CVE's appearance dates: { cve_id: set of dates }
    cve_dates = {}
    
    # Set to collect all dates (for overall range analysis)
    all_dates = set()
    
    # File tracking counters
    files = sorted(os.listdir(raw_folder))  # sort files for orderly processing
    total_files = len(files)
    matched_files = 0
    omitted_files = 0
    
    # Process each file in the folder
    for filename in files:
        match = pattern.match(filename)
        if match:
            matched_files += 1
            date_str = match.group(1)  # extract date as string, e.g., "2025-03-12"
            try:
                current_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                all_dates.add(current_date)
                file_path = os.path.join(raw_folder, filename)
                # Read only the 'cve' column; assume each file has at least this column.
                df = pd.read_csv(file_path, compression="gzip", usecols=["cve"], comment="#")
                # Get unique CVE IDs from the file (to avoid duplicates in the same day)
                cve_ids = df["cve"].unique()
                for cve in cve_ids:
                    if pd.isna(cve):
                        continue
                    if cve not in cve_dates:
                        cve_dates[cve] = set()
                    cve_dates[cve].add(current_date)
            except Exception as e:
                print(f"[ERROR] Could not process '{filename}' due to error: {e}")
        else:
            omitted_files += 1

    # Print file-level statistics
    print("\n--- FILE STATISTICS ---")
    print(f"Total files found: {total_files}")
    print(f"Files matched pattern: {matched_files}")
    print(f"Files omitted: {omitted_files}")

    if not all_dates:
        print("No valid dates found in the dataset. Exiting analysis.")
        return

    # Compute overall date range in the dataset
    all_dates = sorted(all_dates)
    overall_start = all_dates[0]
    overall_end = all_dates[-1]
    print("\n--- OVERALL DATE RANGE ---")
    print(f"Earliest date in dataset: {overall_start}")
    print(f"Latest date in dataset:   {overall_end}")

    # Helper function to generate a continuous set of dates between start and end (inclusive)
    def generate_date_range(start, end):
        date_range = set()
        current = start
        while current <= end:
            date_range.add(current)
            current += timedelta(days=1)
        return date_range

    # Analyze the lifetime of each CVE
    continuous_cve_count = 0  # CVEs with no gaps between first and last appearance
    gap_cve_count = 0         # CVEs with one or more missing days (gaps)
    # We'll store detailed info for CVEs with gaps for inspection
    cve_gap_details = {}

    for cve, dates in cve_dates.items():
        sorted_dates = sorted(dates)
        first_date = sorted_dates[0]
        last_date = sorted_dates[-1]
        # Generate the full set of dates that should be present if the CVE were tracked continuously
        expected_dates = generate_date_range(first_date, last_date)
        # Identify missing dates for this CVE
        missing_dates = sorted(expected_dates - dates)
        if missing_dates:
            gap_cve_count += 1
            cve_gap_details[cve] = {
                "first_date": first_date,
                "last_date": last_date,
                "missing_dates": missing_dates
            }
        else:
            continuous_cve_count += 1

    total_cves = len(cve_dates)
    print("\n--- CVE LIFETIME STATISTICS ---")
    print(f"Total CVEs processed: {total_cves}")
    print(f"CVEs with continuous appearance (no gaps): {continuous_cve_count}")
    print(f"CVEs with gaps in appearance: {gap_cve_count}")
    if gap_cve_count > 0:
        print("\nExample CVEs with gaps (showing up to 10 examples):")
        count = 0
        for cve, info in cve_gap_details.items():
            if count >= 10:
                break
            print(f"  CVE: {cve}")
            print(f"    First appearance: {info['first_date']}")
            print(f"    Last appearance:  {info['last_date']}")
            print(f"    Missing dates:    {info['missing_dates']}")
            count += 1

    # Return detailed analysis (in case further processing is needed)
    return {
        "cve_dates": cve_dates,
        "total_cves": total_cves,
        "continuous_cve_count": continuous_cve_count,
        "gap_cve_count": gap_cve_count,
        "cve_gap_details": cve_gap_details,
        "overall_date_range": (overall_start, overall_end)
    }



# ---------------------------
# Helper: Generate the overall date range list
# ---------------------------
def generate_overall_date_range(start_date, end_date):
    """
    Generate a list of dates (datetime.date objects) from start_date to end_date (inclusive).
    """
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates

# ---------------------------


# ---------------------------
# Plotting function for the CVE timelines
# ---------------------------
def plot_cve_timelines(raw_folder="data/epss/raw", sample_size=30, analysis_result=None):
    """
    Plot the presence timeline for a random sample of CVEs.
    
    For each day in the overall EPSS date range:
      - If a file exists and the CVE is present with a valid epss score: mark as 1 (blue).
      - If a file exists and the CVE is present but epss is missing (NaN): mark as 2 (red).
      - If the file does not exist or the CVE is not in the file: mark as 0 (black).
    
    The x-axis shows the full date range, and the y-axis shows sample CVE IDs.
    """
    # Get analysis_result if not provided
    if analysis_result is None:
        analysis_result = analyze_cve_lifetimes(raw_folder)
    
    overall_start, overall_end = analysis_result["overall_date_range"]
    all_dates = generate_overall_date_range(overall_start, overall_end)
    
    # Get union of all CVE IDs from analysis_result
    all_cves = list(analysis_result["cve_dates"].keys())
    if len(all_cves) < sample_size:
        sample_size = len(all_cves)
    sample_cves = random.sample(all_cves, sample_size)
    
    # Initialize a matrix to store codes:
    # 0: Not present (black), 1: Present with valid epss (blue), 2: Present but epss missing (red)
    timeline_matrix = np.zeros((sample_size, len(all_dates)), dtype=int)
    
    # Loop over all dates and fill in the matrix for each CVE
    for j, current_date in enumerate(all_dates):
        # Build file name for current_date
        file_name = f"epss_scores-{current_date.strftime('%Y-%m-%d')}.csv.gz"
        file_path = os.path.join(raw_folder, file_name)
        if os.path.exists(file_path):
            try:
                # Read file for current_date
                df = pd.read_csv(file_path, compression="gzip", usecols=["cve", "epss"], comment="#")
                # Create a mapping: CVE -> epss value (first occurrence if duplicate)
                cve_to_epss = df.drop_duplicates(subset="cve").set_index("cve")["epss"].to_dict()
            except Exception as e:
                print(f"[ERROR] Could not process '{file_name}': {e}")
                cve_to_epss = {}
        else:
            # File does not exist
            cve_to_epss = {}
        
        # For each sampled CVE, set the code
        for i, cve in enumerate(sample_cves):
            if cve in cve_to_epss:
                epss_value = cve_to_epss[cve]
                if pd.isna(epss_value):
                    timeline_matrix[i, j] = 2  # present but epss missing -> red
                else:
                    timeline_matrix[i, j] = 1  # present with valid epss -> blue
            else:
                timeline_matrix[i, j] = 0  # not present -> black
    
    # Create a discrete colormap
    cmap = ListedColormap(["black", "blue", "red"])
    
    # Plot the matrix using imshow
    plt.figure(figsize=(15, 8))
    im = plt.imshow(timeline_matrix, aspect="auto", cmap=cmap, origin="lower")
    
    # Set x-axis ticks: we will only show a subset of date labels for clarity.
    num_dates = len(all_dates)
    tick_interval = max(1, num_dates // 10)  # show about 10 ticks
    x_ticks = np.arange(0, num_dates, tick_interval)
    x_labels = [all_dates[i].strftime("%Y-%m-%d") for i in x_ticks]
    plt.xticks(x_ticks, x_labels, rotation=45, ha="right")
    
    # Set y-axis ticks: label with the sample CVE IDs
    plt.yticks(np.arange(sample_size), sample_cves)
    
    plt.xlabel("Date")
    plt.ylabel("CVE ID")
    plt.title("Presence Timeline for Sample CVEs\nBlue: Present with EPS, Red: Present but EPS missing, Black: Not present")
    
    # Create a custom legend
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color="blue", label="Present with EPS"),
                      mpatches.Patch(color="red", label="Present but EPS missing"),
                      mpatches.Patch(color="black", label="Not present")]
    plt.legend(handles=legend_patches, loc="upper right")
    
    plt.tight_layout()
    plt.show()


## checking vectorized long table result: all_cve_tiome_series.csv
import pandas as pd

def manual_check_sample_rows():
    # Define the path to the consolidated CSV file
    csv_file = 'data/epss/cve-time-series/all_cves_time_series.csv'

    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file)

    # Randomly sample 50 rows from the DataFrame (using a fixed random state for reproducibility)
    sample_df = df.sample(n=50, random_state=42)

    # Print the sample rows
    print(sample_df)





if __name__ == "__main__":
    # Define file paths for inspection.
    # csv_file_path = "data/epss/raw/epss_scores-2022-02-04.csv.gz"
    # parquet_file_path = "data/epss/processed/daily_parquet/2021-04-14/epss.parquet"
    
    # Inspect the CSV.GZ file.
    # inspect_csv_gz(csv_file_path, n_lines=100)


    # csv_file_path = "data/epss/raw/epss_scores-2025-02-27.csv.gz"
    # inspect_csv_gz(csv_file_path, n_lines=100)

    # Inspect the Parquet file.
    # inspect_parquet(parquet_file_path, n_lines=5)
    # analysis_result = analyze_cve_lifetimes(raw_folder="data/epss/raw")
    # plot_cve_timelines(raw_folder="data/epss/raw", sample_size=30, analysis_result=analysis_result)
    manual_check_sample_rows()