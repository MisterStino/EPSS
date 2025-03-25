import os
import pandas as pd

def analyze_final_dataset():
    """
    This function reads the final full dataset CSV, computes key overall metrics
    and per-CVE metrics for feature development, and saves a summary file that has
    two sections:
    
      1. Overall Dataset Metrics:
         - Total number of rows and columns.
         - Total unique CVEs.
         - Maximum, minimum, and mean time series lengths (i.e., number of rows per CVE).
      
      2. Per-CVE Summary Table:
         For each CVE, metrics include:
         - Time series length.
         - Minimum and maximum date.
         - Expected length (if the series were continuous).
         - Number and proportion of missing dates.
         - A flag indicating if the time series is continuous.
    
    The summary is saved as:
         data/full_db/processed/full_data_summary.txt
    """
    
    # Define the path to the final full dataset CSV.
    final_csv = os.path.join('data', 'full_db', 'processed', 'final_full_data.csv')
    if not os.path.exists(final_csv):
        print(f"Error: Final dataset not found at {final_csv}")
        return
    
    # Load the final dataset.
    df = pd.read_csv(final_csv)
    
    # Convert the 'date' column to datetime format.
    df['date'] = pd.to_datetime(df['date'])
    
    # Overall dataset metrics.
    total_rows, total_cols = df.shape
    unique_cves = df['cve'].unique()
    total_cves = len(unique_cves)
    
    print(f"Final dataset shape: {total_rows} rows, {total_cols} columns")
    print(f"Total unique CVEs: {total_cves}")
    
    # Prepare a list to accumulate per-CVE summary metrics.
    summary_list = []
    
    # Group by CVE and compute metrics for each time series.
    for cve, group in df.groupby('cve'):
        # Sort the group by date.
        group = group.sort_values('date')
        ts_length = len(group)
        min_date = group['date'].min()
        max_date = group['date'].max()
        # Expected number of days if the time series were continuous.
        expected_length = (max_date - min_date).days + 1
        missing_days = expected_length - ts_length
        missing_ratio = missing_days / expected_length if expected_length > 0 else 0
        is_continuous = (missing_days == 0)
        
        summary_list.append({
            'cve': cve,
            'ts_length': ts_length,
            'min_date': min_date.strftime('%Y-%m-%d'),
            'max_date': max_date.strftime('%Y-%m-%d'),
            'expected_length': expected_length,
            'missing_days': missing_days,
            'missing_ratio': missing_ratio,
            'is_continuous': is_continuous
        })
    
    # Create a DataFrame from the per-CVE summary.
    summary_df = pd.DataFrame(summary_list)
    
    # Compute aggregate metrics from the per-CVE summaries.
    max_ts_length = summary_df['ts_length'].max()
    min_ts_length = summary_df['ts_length'].min()
    mean_ts_length = summary_df['ts_length'].mean()
    
    overall_summary = (
        "Overall Dataset Metrics:\n"
        "-------------------------\n"
        f"Total rows (long table): {total_rows}\n"
        f"Total columns: {total_cols}\n"
        f"Total unique CVEs: {total_cves}\n"
        f"Maximum time series length: {max_ts_length}\n"
        f"Minimum time series length: {min_ts_length}\n"
        f"Mean time series length: {mean_ts_length:.2f}\n"
    )
    
    # Prepare the final summary file with two sections.
    output_dir = os.path.join('data', 'full_db', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, 'full_data_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write(overall_summary)
        f.write("\n\n")
        f.write("Per-CVE Summary Metrics:\n")
        f.write("-------------------------\n")
        # Write the per-CVE summary DataFrame as CSV text.
        f.write(summary_df.to_csv(index=False))
    
    print(f"\nFull data summary has been saved to: {summary_file}")


import os
import pandas as pd

def generate_base_keys():
    """
    Generates the base keys for merging final data by extracting the composite primary key 
    (cve and date) from the processed EPSS data.

    It assumes the EPSS processed data is stored at:
        data/epss/processed/epss_processed.csv

    The function then:
      1. Loads the EPSS processed CSV.
      2. Converts the 'date' column to datetime format (for consistency).
      3. Extracts the columns 'cve' and 'date' (dropping any duplicates).
      4. Sorts the resulting DataFrame by 'cve' and 'date'.
      5. Saves the final base keys CSV to:
         data/full_db/processed/base_keys.csv
    """
    # Define the source file path for the processed EPSS data.
    source_file = os.path.join('data', 'epss', 'processed', 'epss_processed.csv')
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return

    # Load the processed EPSS data.
    df = pd.read_csv(source_file)
    
    # Ensure that the 'date' column is in datetime format.
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract the composite primary key columns: 'cve' and 'date'.
    # Dropping duplicates ensures each (cve, date) pair is unique.
    base_keys_df = df[['cve', 'date']].drop_duplicates()
    
    # Sort by cve and date to have a consistent order.
    base_keys_df = base_keys_df.sort_values(by=['cve', 'date'])
    
    # Define the output directory and file path.
    output_dir = os.path.join('data', 'full_db', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'base_keys.csv')
    
    # Save the resulting DataFrame to CSV.
    base_keys_df.to_csv(output_file, index=False)
    print(f"Base keys file has been generated and saved to: {output_file}")

if __name__ == '__main__':
    generate_base_keys()
    #analyze_final_dataset()




