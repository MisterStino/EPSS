import os
import pandas as pd
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
      2. Verifies that the required columns 'cve' and 'date' exist.
      3. Converts the 'date' column to datetime format for consistency.
      4. Extracts the columns 'cve' and 'date' (without dropping duplicates).
      5. Checks that no duplicate (cve, date) pairs exist and, if any are found, raises an error
         with a clear message and sample duplicates.
      6. Sorts the resulting DataFrame by 'cve' and 'date'.
      7. Saves the final base keys CSV to:
         data/full_db/processed/base_keys.csv
    """
    # Define the source file path for the processed EPSS data.
    source_file = os.path.join('data', 'epss', 'processed', 'epss_processed.csv')
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Error: Source file not found: {source_file}")

    # Load the processed EPSS data.
    df = pd.read_csv(source_file)
    
    # Check that the required columns are present.
    required_columns = ['cve', 'date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in the source data: {missing_columns}")
    
    # Convert the 'date' column to datetime format.
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract the composite primary key columns: 'cve' and 'date'.
    keys_df = df[['cve', 'date']]
    
    # Check for duplicate composite keys.
    duplicates = keys_df.duplicated(keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        sample_duplicates = keys_df[duplicates].drop_duplicates().head(5)
        raise ValueError(
            f"Duplicate composite keys found: {dup_count} duplicates. "
            f"Sample duplicate keys:\n{sample_duplicates}"
        )
    
    # At this point, keys_df is an exact mirror of the original keys.
    base_keys_df = keys_df.copy()
    
    # Sort by cve and date to have a consistent order.
    base_keys_df = base_keys_df.sort_values(by=['cve', 'date'])
    
    # Final check: Assert that the number of rows in base_keys_df equals the original df.
    if len(base_keys_df) != len(df):
        raise ValueError(
            f"Base keys length mismatch: expected {len(df)} rows, got {len(base_keys_df)} rows."
        )

    # Define the output directory and file path.
    output_dir = os.path.join('data', 'full_db', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'base_keys.csv')
    
    # Save the resulting DataFrame to CSV.
    base_keys_df.to_csv(output_file, index=False)
    print(f"Base keys file has been generated and saved to: {output_file}")

import os
import pandas as pd

def load_base_keys(return_format="dataframe", small=False):
    """
    Loads the base keys file and returns it in the desired format.
    
    When small is True, loads the small base keys file from:
        data/general_utils/small_data/small_base_keys.csv
    Otherwise, loads the full base keys file from:
        data/full_db/processed/base_keys.csv

    The file should contain at least the following columns:
        - 'cve'
        - 'date'
        
    Parameters:
        return_format (str): Determines the return type.
            - "dataframe": Load the CSV into a pandas DataFrame (default).
            - "csv": Return the file path of the CSV.
        small (bool): If True, load the small version of the base keys file.
    
    Returns:
        pandas.DataFrame or str: The base keys as a DataFrame (if return_format is "dataframe")
                                 or the file path (if return_format is "csv").
    
    Raises:
        FileNotFoundError: If the base keys file does not exist.
        ValueError: If the loaded file does not contain the required columns or an invalid 
                    return_format is specified.
    
    Example usage:
        # Load as DataFrame (full dataset):
        base_keys_df = load_base_keys(return_format="dataframe")
        print("Base keys as DataFrame:")
        print(base_keys_df.head())
        
        # Load as DataFrame (small subset):
        small_base_keys_df = load_base_keys(return_format="dataframe", small=True)
        print("Small base keys as DataFrame:")
        print(small_base_keys_df.head())
        
        # Or get the CSV file path:
        base_keys_csv = load_base_keys(return_format="csv")
        print(f"Base keys CSV file path: {base_keys_csv}")
    """
    # Determine the file path based on the 'small' flag.
    if small:
        base_keys_path = os.path.join('data', 'general_utils', 'small_data', 'small_base_keys.csv')
    else:
        base_keys_path = os.path.join('data', 'full_db', 'processed', 'base_keys.csv')
    
    # Check if the file exists.
    if not os.path.exists(base_keys_path):
        raise FileNotFoundError(f"Base keys file not found at {base_keys_path}")
    
    # If user wants the CSV file path, return it.
    if return_format.lower() == "csv":
        return base_keys_path
    elif return_format.lower() == "dataframe":
        # Load the CSV file into a pandas DataFrame.
        df = pd.read_csv(base_keys_path)
        
        # Ensure the required headers are present.
        required_columns = {'cve', 'date'}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(f"The base keys file must contain columns {required_columns}, but found {set(df.columns)}")
        
        # Convert the 'date' column to datetime for consistency.
        df['date'] = pd.to_datetime(df['date'])
        
        return df
    else:
        raise ValueError("Invalid return_format specified. Use 'dataframe' or 'csv'.")


if __name__ == '__main__':
    generate_base_keys()
