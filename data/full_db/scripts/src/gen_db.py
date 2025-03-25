# create script that gets the full y data
# merges features
# create mock data
# create utils first for feature development: so get shapes for each cve time series: which dataes as well. 
# for for each cve all the dates that we need. 
# and list of all cves
# maybe more? 
# also check if for cve time series is continuous.: missing dates?





import os
import pandas as pd

def generate_full_database():
    """
    This function generates the final full dataset for training.
    
    It assumes that for each module (e.g., 'epss', 'twitter', etc.) there is a 
    processed CSV file in the folder:
    
        data/<module>/processed/<module>_processed.csv
        
    The epss_processed.csv file serves as the base dataset (it contains the target 'y').
    The function then dynamically loops through a list of module names (other than epss)
    and merges their data onto the epss data based on the composite key (cve, date).
    
    The final merged data is sorted by cve and date, then saved to:
    
        data/full_db/final_full_data.csv
    """
    # Define the list of module names. These should match the folder names.
    modules = ['epss']  # Add more module names as needed.
    
    # The base module is epss because its processed file contains the target variable.
    base_module = 'epss'
    
    # Construct the file path for the base module.
    base_file = os.path.join('data', base_module, 'processed', f'{base_module}_processed.csv')
    if not os.path.exists(base_file):
        print(f"Error: Base file not found: {base_file}")
        return
    
    # Load the base DataFrame (this will be our y and our starting point).
    full_df = pd.read_csv(base_file)
    print(f"Loaded base data from {base_file} with {len(full_df)} rows.")
    
    # Loop over the modules; merge additional features onto the base data.
    for module in modules:
        # Skip the base module since it is already loaded.
        if module == base_module:
            continue
        
        # Construct the file path for the current module's processed CSV.
        module_file = os.path.join('data', module, 'processed', f'{module}_processed.csv')
        if not os.path.exists(module_file):
            print(f"Warning: Processed file for module '{module}' not found at {module_file}. Skipping.")
            continue
        
        # Load the module's DataFrame.
        module_df = pd.read_csv(module_file)
        print(f"Loaded {module} data from {module_file} with {len(module_df)} rows.")
        
        # Merge this module's data with the base DataFrame on 'cve' and 'date'.
        # We use a left join to keep all epss data (y) and merge in available features.
        full_df = full_df.merge(module_df, on=['cve', 'date'], how='left')
        print(f"After merging {module}, dataset now has {len(full_df)} rows and {full_df.shape[1]} columns.")
    
    # Optionally, sort the final DataFrame by 'cve' and 'date' for consistency.
    full_df.sort_values(by=['cve', 'date'], inplace=True)
    
    # Define the output directory and file path for the final dataset.
    output_dir = os.path.join('data', 'full_db','processed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'final_full_data.csv')
    
    # Save the final merged DataFrame to CSV.
    full_df.to_csv(output_file, index=False)
    print(f"\nFinal full dataset has been generated and saved to: {output_file}")

if __name__ == '__main__':
    generate_full_database()
