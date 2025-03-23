import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def load_all_epss_scores(raw_folder="data/epss/raw"):
    """
    Load all EPSS scores from daily CSV.GZ files in the given folder.
    Assumes that each file is named in the format 'epss_scores-YYYY-MM-DD.csv.gz'
    and that the file contains at least a column named 'epss'.
    
    Additionally, this function:
    1. Tracks and prints how many files were found in total.
    2. Tracks and prints how many files matched the expected filename pattern.
    3. Tracks and prints how many files were omitted because they did not match the filename pattern.
    4. Collects all valid dates from the filenames and checks if any daily file is missing
       between the earliest and latest date.
    5. Prints any missing dates or reports that there are none missing.

    :param raw_folder: Directory containing the raw EPSS CSV.GZ files.
    :return: A Pandas Series containing all EPSS scores.
    """
    # Prepare counters and lists
    total_files = 0
    matched_files = 0
    omitted_files = 0
    epss_list = []
    date_list = []

    # Regex to match filenames: epss_scores-YYYY-MM-DD.csv.gz
    pattern = re.compile(r"^epss_scores-(\d{4}-\d{2}-\d{2})\.csv\.gz$")

    # List all files in the directory
    all_files = os.listdir(raw_folder)
    total_files = len(all_files)

    for filename in all_files:
        match = pattern.match(filename)
        if match:
            matched_files += 1
            # Extract date from filename
            date_str = match.group(1)  # e.g., "2025-03-01"
            try:
                file_path = os.path.join(raw_folder, filename)

                # Read only the 'epss' column, ignore comment lines
                df = pd.read_csv(file_path, compression="gzip", usecols=["epss"], comment="#")
                epss_list.append(df["epss"])

                # Parse the date string into a datetime.date object
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                date_list.append(date_obj)

            except Exception as e:
                print(f"[ERROR] Could not load '{filename}' due to error: {e}")
        else:
            # File does not match the pattern
            omitted_files += 1

    # Print basic file-level statistics
    print(f"\n--- FILE STATISTICS ---")
    print(f"Total files found in folder: {total_files}")
    print(f"Files matched the pattern:  {matched_files}")
    print(f"Files omitted:              {omitted_files}")

    # If no valid files loaded, return an empty Series
    if not epss_list:
        print("[INFO] No valid files were loaded. Check the folder path or file naming conventions.")
        return pd.Series([], dtype=float)

    # Concatenate all EPSS data into a single Series
    all_epss = pd.concat(epss_list, ignore_index=True)

    # Now check for missing dates
    unique_dates = sorted(set(date_list))  # remove duplicates, sort them

    if len(unique_dates) < 2:
        # If we have fewer than 2 dates, there's no "range" to check
        print("\n--- DATE CHECK ---")
        print("Cannot check continuity because there's only one (or zero) valid date.")
        return all_epss

    # Determine the full range of dates
    min_date = unique_dates[0]
    max_date = unique_dates[-1]
    print(f"\n--- DATE CHECK ---")
    print(f"Earliest date found: {min_date}")
    print(f"Latest date found:   {max_date}")

    # Build a continuous range from min_date to max_date
    full_range = []
    current_date = min_date
    while current_date <= max_date:
        full_range.append(current_date)
        current_date += timedelta(days=1)

    # Compare to see which dates are missing
    full_range_set = set(full_range)
    found_dates_set = set(unique_dates)
    missing_dates = sorted(list(full_range_set - found_dates_set))

    if missing_dates:
        print("Missing dates in the daily time series:")
        for missing_date in missing_dates:
            print(f"  - {missing_date}")
    else:
        print("No missing dates. The dataset is continuous from min to max date.")

    return all_epss


def visualize_distribution(epss_series):
    """
    Visualize the distribution of EPSS scores using descriptive statistics,
    a histogram, a box plot, and a density plot.
    
    :param epss_series: A Pandas Series of EPSS scores.
    """
    # ---------- Descriptive Statistics ----------
    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(epss_series.describe())
    
    # ---------- Histogram ----------
    plt.figure(figsize=(10, 6))
    plt.hist(epss_series, bins=50, range=(0, 1), edgecolor='black')
    plt.title("Histogram of EPSS Scores")
    plt.xlabel("EPSS Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    
    # ---------- Box Plot ----------
    plt.figure(figsize=(6, 8))
    plt.boxplot(epss_series, vert=True)
    plt.title("Box Plot of EPSS Scores")
    plt.ylabel("EPSS Score")
    plt.grid(True)
    plt.show()
    
    # ---------- Density Plot ---------- 
    # this is waaaay to heavy. Either leave out or sample from the data: prob still good to show the distribution
    # plt.figure(figsize=(10, 6))
    # epss_series.plot(kind='density')
    # plt.title("Density Plot of EPSS Scores")
    # plt.xlabel("EPSS Score")
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    # Load all EPSS scores from the raw data folder.
    all_epss_scores = load_all_epss_scores(raw_folder="data/epss/raw")
    print(f"\nLoaded {len(all_epss_scores)} EPSS scores in total.")
    
    # Visualize the distribution of the scores.
    if not all_epss_scores.empty:
        visualize_distribution(all_epss_scores)
