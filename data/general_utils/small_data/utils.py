import os
import pandas as pd
import numpy as np
import math

def generate_small_subset(num_cves=5, num_timesteps=5, epss_lower=0.7, epss_upper=0.85):
    """
    Generates a small subset of the final full dataset and the base keys for feature development,
    with the following additional requirements:
    
      - Roughly half of the sampled CVEs (rounding upward) must have at least one epss score 
        between epss_lower and epss_upper (inclusive). For these CVEs, the sampling of time steps 
        is done so that 2 time steps occur *before* the first occurrence of an epss score in that range,
        and the remaining rows are taken from after that time step.
      
      - CVEs that do not meet the epss range condition are sampled by simply taking the first 
        num_timesteps rows (sorted by date).
    
    Steps:
      1. Load the final full dataset from:
         data/full_db/processed/final_full_data.csv
      2. Load the base keys from:
         data/full_db/processed/base_keys.csv
      3. Identify unique CVEs and split them into two groups:
         - In-range: CVEs with at least one epss value in [epss_lower, epss_upper].
         - Out-of-range: The remaining CVEs.
      4. Determine target counts:
         - in_range_target = ceil(num_cves / 2)
         - out_range_target = num_cves - in_range_target
         Adjust if one group has fewer than needed.
      5. Sample the CVEs from each group using a fixed random state.
      6. For each sampled CVE:
         - If in the in-range group, find the first time step where epss is in the specified range.
           Then, take up to 2 rows immediately preceding that time step (if available) and fill the 
           remaining rows from the time steps starting at that occurrence, so that total rows equals num_timesteps.
         - If in the out-of-range group, simply take the first num_timesteps rows.
      7. Save the resulting small subset of the full dataset and the base keys to:
         data/general_utils/small_data/small_full_data.csv and small_base_keys.csv respectively.
    """
    # Define input file paths.
    final_full_data_path = os.path.join('data', 'full_db', 'processed', 'final_full_data.csv')
    base_keys_path = os.path.join('data', 'full_db', 'processed', 'base_keys.csv')
    
    # Verify that the input files exist.
    if not os.path.exists(final_full_data_path):
        raise FileNotFoundError(f"Final full data not found at: {final_full_data_path}")
    if not os.path.exists(base_keys_path):
        raise FileNotFoundError(f"Base keys not found at: {base_keys_path}")
    
    # Load the datasets.
    full_df = pd.read_csv(final_full_data_path)
    base_keys_df = pd.read_csv(base_keys_path)
    
    # Ensure the 'date' columns are in datetime format.
    full_df['date'] = pd.to_datetime(full_df['date'])
    base_keys_df['date'] = pd.to_datetime(base_keys_df['date'])
    
    # Get the list of unique CVEs from the base keys.
    unique_cves = base_keys_df['cve'].unique()
    
    # Group the CVEs into two groups based on epss condition.
    in_range_cves = []
    out_range_cves = []
    
    # Use the full dataset for checking epss scores.
    for cve in unique_cves:
        group = full_df[full_df['cve'] == cve].sort_values('date')
        # Check if any epss score in this CVE's time series is in the desired range.
        if group['epss'].between(epss_lower, epss_upper, inclusive='both').any():
            in_range_cves.append(cve)
        else:
            out_range_cves.append(cve)
    
    # Determine target counts.
    in_range_target = math.ceil(num_cves / 2)
    out_range_target = num_cves - in_range_target
    
    # Sample CVEs from each group using a fixed random state.
    sampled_in_range = pd.Series(in_range_cves).sample(
        n=min(in_range_target, len(in_range_cves)), random_state=42
    ).tolist()
    
    remaining_needed = num_cves - len(sampled_in_range)
    sampled_out_range = pd.Series(out_range_cves).sample(
        n=min(remaining_needed, len(out_range_cves)), random_state=42
    ).tolist()
    
    sampled_cves = sampled_in_range + sampled_out_range
    
    if len(sampled_cves) < num_cves:
        print(f"Warning: Only {len(sampled_cves)} CVEs could be sampled out of the requested {num_cves}.")
    
    # For each sampled CVE, select the appropriate time steps.
    sampled_groups = []
    for cve in sampled_cves:
        group = full_df[full_df['cve'] == cve].sort_values('date').reset_index(drop=True)
        if cve in sampled_in_range:
            # For CVEs with epss in the range: find the first time step where epss is within [epss_lower, epss_upper].
            condition = group['epss'].between(epss_lower, epss_upper, inclusive='both')
            if condition.any():
                first_in_range_pos = condition.idxmax()  # index of the first occurrence (since condition is boolean, idxmax() returns first True)
            else:
                # Fallback, should not happen because we already filtered these CVEs.
                first_in_range_pos = 0
            
            # Determine number of pre-rows available (up to 2).
            pre_count = min(2, first_in_range_pos)
            # Calculate number of rows needed after (including the first in-range row).
            post_count = num_timesteps - pre_count
            
            # Select pre-rows (if any) and post-rows.
            pre_rows = group.iloc[max(0, first_in_range_pos - pre_count): first_in_range_pos] if pre_count > 0 else pd.DataFrame()
            post_rows = group.iloc[first_in_range_pos: first_in_range_pos + post_count]
            
            sampled_group = pd.concat([pre_rows, post_rows])
        else:
            # For CVEs without epss in the range, simply take the first num_timesteps rows.
            sampled_group = group.head(num_timesteps)
        sampled_groups.append(sampled_group)
    
    # Concatenate the sampled groups to form the small full dataset.
    small_full_df = pd.concat(sampled_groups)
    
    # For the small base keys, extract only the 'cve' and 'date' columns.
    small_base_keys_df = small_full_df[['cve', 'date']]
    
    # Define the output directory and file paths.
    output_dir = os.path.join('data', 'general_utils', 'small_data')
    os.makedirs(output_dir, exist_ok=True)
    small_base_keys_path = os.path.join(output_dir, 'small_base_keys.csv')
    small_full_data_path = os.path.join(output_dir, 'small_full_data.csv')
    
    # Save the small subsets to CSV.
    small_base_keys_df.to_csv(small_base_keys_path, index=False)
    small_full_df.to_csv(small_full_data_path, index=False)
    
    print(f"Small base keys saved to: {small_base_keys_path}")
    print(f"Small full data saved to: {small_full_data_path}")

if __name__ == '__main__':
    generate_small_subset()
