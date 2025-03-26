import os
import pandas as pd
import numpy as np
import math



def sample_from_pre_filtered_files(num_cves=5, num_timesteps=5, epss_lower=0.7, epss_upper=0.85,
                                   input_dir='data/general_utils/files',
                                   output_dir='data/general_utils/files'):
    """
    Loads pre-filtered CVE data from in_range_cves.csv and out_range_cves.csv, samples a specified number of CVEs,
    and for each sampled CVE selects num_timesteps rows as follows:
    
      - For in-range CVEs:
          * Sort the CVE's time series by date.
          * Find the first row (the pivot) where the epss score is between epss_lower and epss_upper.
          * Take up to 2 rows immediately preceding that pivot (if available) and then as many rows after the pivot 
            as needed so that the total number of rows equals num_timesteps.
      
      - For out-of-range CVEs:
          * Simply take the first num_timesteps rows (sorted by date).
    
    The sampled data is then saved to three CSV files:
      - small_in_range_sample.csv (sampled in-range CVEs)
      - small_out_range_sample.csv (sampled out-of-range CVEs)
      - small_sampled_data.csv (combined sample)
    
    Parameters:
      - num_cves: int
            Total number of CVEs to sample (roughly half from each group).
      - num_timesteps: int
            Number of time steps (rows) to sample per CVE.
      - epss_lower: float
            Lower bound for epss score (used to locate the pivot for in-range CVEs).
      - epss_upper: float
            Upper bound for epss score (used to locate the pivot for in-range CVEs).
      - input_dir: str
            Directory where the pre-filtered CSV files are located.
      - output_dir: str
            Directory where the output CSV files will be saved.
    """
    # Define input file paths.
    in_range_file = os.path.join(input_dir, 'in_range_cves.csv')
    out_range_file = os.path.join(input_dir, 'out_range_cves.csv')
    
    # Verify that the pre-filtered files exist.
    if not os.path.exists(in_range_file):
        raise FileNotFoundError(f"{in_range_file} does not exist.")
    if not os.path.exists(out_range_file):
        raise FileNotFoundError(f"{out_range_file} does not exist.")
    
    # Load the pre-filtered data (parsing the date column).
    in_range_df = pd.read_csv(in_range_file, parse_dates=['date'])
    out_range_df = pd.read_csv(out_range_file, parse_dates=['date'])
    
    # Get unique CVEs from each pre-filtered dataset.
    in_range_unique = pd.Series(in_range_df['cve'].unique())
    out_range_unique = pd.Series(out_range_df['cve'].unique())
    
    # Determine target numbers (roughly half from each).
    in_range_target = math.ceil(num_cves / 2)
    out_range_target = num_cves - in_range_target
    
    # Sample CVEs from each group (using a fixed random state for reproducibility).
    sampled_in_range = in_range_unique.sample(n=min(in_range_target, len(in_range_unique)), random_state=42).tolist()
    sampled_out_range = out_range_unique.sample(n=min(out_range_target, len(out_range_unique)), random_state=42).tolist()
    
    sampled_groups = []
    
    # Process in-range CVEs.
    for cve in sampled_in_range:
        group = in_range_df[in_range_df['cve'] == cve].sort_values('date').reset_index(drop=True)
        # Find the pivot: first row where epss is between epss_lower and epss_upper.
        condition = (group['epss'] >= epss_lower) & (group['epss'] <= epss_upper)
        # Since these CVEs are pre-filtered, this condition will always be True for at least one row.
        pivot_idx = condition.idxmax()  # first occurrence of True
        # Determine number of pre-rows available (up to 2).
        pre_count = min(2, pivot_idx)
        # Determine number of post rows needed to reach num_timesteps.
        post_count = num_timesteps - pre_count
        # Select rows: pre-rows (if any) and then post-rows (starting at pivot).
        selected_rows = group.iloc[max(0, pivot_idx - pre_count): pivot_idx].copy()
        selected_rows = pd.concat([selected_rows, group.iloc[pivot_idx: pivot_idx + post_count]])
        sampled_groups.append(selected_rows)
    
    # Process out-of-range CVEs.
    for cve in sampled_out_range:
        group = out_range_df[out_range_df['cve'] == cve].sort_values('date').reset_index(drop=True)
        selected_rows = group.head(num_timesteps)
        sampled_groups.append(selected_rows)
    
    # Combine all sampled groups.
    sampled_data = pd.concat(sampled_groups)
    
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file paths.
    small_in_range_file = os.path.join(output_dir, 'small_in_range_sample.csv')
    small_out_range_file = os.path.join(output_dir, 'small_out_range_sample.csv')
    combined_file = os.path.join(output_dir, 'small_sampled_data.csv')
    
    # Split the combined sample back into in-range and out-of-range subsets.
    in_range_sample_df = sampled_data[sampled_data['cve'].isin(sampled_in_range)]
    out_range_sample_df = sampled_data[sampled_data['cve'].isin(sampled_out_range)]
    
    # Save the sampled data to CSV files.
    in_range_sample_df.to_csv(small_in_range_file, index=False)
    out_range_sample_df.to_csv(small_out_range_file, index=False)
    sampled_data.to_csv(combined_file, index=False)
    
    print(f"Small in-range sample saved to: {small_in_range_file}")
    print(f"Small out-of-range sample saved to: {small_out_range_file}")
    print(f"Combined small sample saved to: {combined_file}")







def save_cves_by_epss_range(df, epss_lower, epss_upper, output_dir='data/general_utils/files'):
    """
    Separates and saves CVEs based on whether they have at least one epss score 
    within the specified range [epss_lower, epss_upper]. The entire time series
    for each CVE is saved. CVEs with at least one epss score in range are saved 
    to one file and the rest to another file.
    
    Parameters:
      - df: pandas.DataFrame
            Input DataFrame with at least 'cve' and 'epss' columns.
      - epss_lower: float
            Lower bound for epss score.
      - epss_upper: float
            Upper bound for epss score.
      - output_dir: str
            Directory where the CSV files will be saved.
            
    The files will be named:
      - in_range_cves.csv
      - out_range_cves.csv
    """
    # Ensure the necessary columns exist
    required_cols = {'cve', 'epss'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")
    
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Efficiently filter using groupby.filter for in-range CVEs
    in_range_df = df.groupby('cve', group_keys=False).filter(
        lambda group: group['epss'].between(epss_lower, epss_upper, inclusive='both').any()
    )
    
    # Get the unique CVEs that are in-range
    in_range_cves = set(in_range_df['cve'].unique())
    
    # Filter the remaining (out-of-range) CVEs
    out_range_df = df[~df['cve'].isin(in_range_cves)]
    
    # Save the results to CSV files
    in_range_path = os.path.join(output_dir, 'in_range_cves.csv')
    out_range_path = os.path.join(output_dir, 'out_range_cves.csv')
    
    in_range_df.to_csv(in_range_path, index=False)
    out_range_df.to_csv(out_range_path, index=False)
    
    print(f"In-range CVEs saved to: {in_range_path}")
    print(f"Out-of-range CVEs saved to: {out_range_path}")
    return in_range_df, out_range_df


    



if __name__ == '__main__':
    final_full_data_path = os.path.join('data', 'full_db', 'processed', 'final_full_data.csv')
    save_cves_by_epss_range(pd.read_csv(final_full_data_path), 0.85, 0.92)
    #sample_from_pre_filtered_files(num_cves=5, num_timesteps=5, epss_lower=0.85, epss_upper=0.92)