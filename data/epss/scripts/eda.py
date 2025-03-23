import os
import glob
import pandas as pd

# Define the directory containing your raw files
raw_folder = 'data/epss/raw'

# List all CSV.GZ files in the raw folder (assuming they are already in order)
files = glob.glob(os.path.join(raw_folder, '*.csv.gz'))

# Create a list to accumulate the summary for each file
summary_list = []

prev_cves = None  # To store the previous day's set of CVEs
prev_file = None  # To store the previous file's name

# Process each file in the folder
for file in files:
    filename = os.path.basename(file)
    # Optional: Extract a date part from the filename if it follows a known pattern.
    # For example: 'epss_scores-2022-02-04.csv.gz' -> '2022-02-04'
    try:
        date_part = filename.split('-')[1].split('.')[0]
    except IndexError:
        date_part = None

    print("=" * 80)
    print(f"Processing file: {filename}")
    
    # Load the data while skipping the first meta row so that the next row becomes the header
    df = pd.read_csv(file, compression='gzip', skiprows=1)
    
    # Compute the missing values per column (all columns)
    missing_values = df.isnull().sum()
    # Filter to only include columns that have missing values (if any)
    missing = missing_values[missing_values > 0].to_dict()
    
    # Specifically count missing values in the 'epss' column (if it exists)
    epss_missing = int(df['epss'].isnull().sum()) if 'epss' in df.columns else None
    
    # Extract the set of CVEs using the 'cve' column
    if 'cve' in df.columns:
        current_cves = set(df['cve'])
    else:
        current_cves = set()
    
    # Initialize subset check variables
    subset_result = None
    missing_cves = None
    if prev_cves is not None:
        # Check if all CVEs from the previous file are present in the current file
        subset_result = prev_cves.issubset(current_cves)
        if not subset_result:
            # Determine which CVEs from the previous day are missing in the current day
            missing_cves = list(prev_cves - current_cves)
        print(f"\nAre all CVEs from {os.path.basename(prev_file)} present in {filename}? {subset_result}")
        if not subset_result:
            print("Missing CVEs from previous day:")
            for cve in missing_cves:
                print(cve)
    else:
        print("No previous file to compare.")
    
    # Log the results for this file into a dictionary
    summary_list.append({
        'file': filename,
        'date': date_part,
        'missing_values': missing,       # Dictionary of columns with missing values
        'epss_missing': epss_missing,      # Count of missing entries in the epss column
        'subset_check': subset_result,     # True/False if previous day's CVEs are all present (None for the first file)
        'missing_cves': missing_cves       # List of missing CVEs (if any) or None
    })
    
    # Update previous day's CVEs and file name for the next iteration
    prev_cves = current_cves
    prev_file = file
    
    print("=" * 80 + "\n")

# Create a summary DataFrame for a final overview
summary_df = pd.DataFrame(summary_list)

print("Summary Overview:")
print(summary_df)

# Save the summary to a CSV file for later inspection
output_csv = 'epss_summary_overview.csv'
summary_df.to_csv(output_csv, index=False)
print(f"Summary saved to {output_csv}")
