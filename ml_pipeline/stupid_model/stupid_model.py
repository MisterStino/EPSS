import pandas as pd
import xarray as xr
import numpy as np
import glob
import os

def read_parquet_files(directory):
    """
    Read Parquet files, extracting 'cve', 'date', 'epss', and optional 'primary_cvss_score', 'has_remediation'.
    Returns a dictionary mapping CVE to a list of (date, epss, cvss, remediation) tuples.
    """
    parquet_files = glob.glob(os.path.join(directory, '*.parquet'))
    if not parquet_files:
        raise ValueError(f"No Parquet files found in {directory}")

    cve_data = {}
    for file_path in parquet_files:
        try:
            # Read relevant columns, handling optional ones
            columns = ['cve', 'date', 'epss']
            if 'primary_cvss_score' in pd.read_parquet(file_path, columns=['primary_cvss_score']).columns:
                columns.append('primary_cvss_score')
            if 'has_remediation' in pd.read_parquet(file_path, columns=['has_remediation']).columns:
                columns.append('has_remediation')
            df = pd.read_parquet(file_path, columns=columns)
            if not {'cve', 'date', 'epss'}.issubset(df.columns):
                raise ValueError(f"Parquet file {file_path} must contain 'cve', 'date', 'epss'")

            for cve_id, group in df.groupby('cve'):
                try:
                    # Convert dates to datetime, sort, and get unique (date, epss, cvss, remediation) tuples
                    group['date'] = pd.to_datetime(group['date'])
                    group = group.sort_values('date').drop_duplicates('date')
                    data = [
                        (
                            row['date'].strftime('%Y-%m-%d'),
                            row['epss'] if pd.notna(row['epss']) else None,
                            row.get('primary_cvss_score', None),
                            row.get('has_remediation', None)
                        )
                        for _, row in group.iterrows()
                    ]
                    if cve_id in cve_data:
                        cve_data[cve_id].extend(data)
                    else:
                        cve_data[cve_id] = data
                except ValueError as e:
                    print(f"Error parsing data for CVE {cve_id} in {file_path}: {e}")
                    continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Remove duplicates and sort by date
    for cve_id in cve_data:
        cve_data[cve_id] = sorted(list(set(cve_data[cve_id])), key=lambda x: x[0])
    
    if not cve_data:
        raise ValueError("No valid CVE data found in Parquet files")
    
    print(f"Found {len(cve_data)} CVEs with timestamps.")
    return cve_data

def generate_predictions(cve_data, h=30, use_jump=False, use_remediation=False, epss_threshold=0.1, cvss_threshold=7.0):
    """
    Generate 30 AR(1) predictions per timestamp, with optional jump and remediation adjustments.
    Returns a 3D array [N, L_max, H] and metadata.
    """
    cve_ids = list(cve_data.keys())
    n = len(cve_ids)
    l_max = max(len(data) for data in cve_data.values())
    
    pred = np.full((n, l_max, h), np.nan)  # NaN for padding
    
    for i, cve_id in enumerate(cve_ids):
        timestamps = cve_data[cve_id]
        for j, (date, epss, cvss, remediation) in enumerate(timestamps):
            # Use 0 if epss is missing or below threshold
            base_pred = epss if (epss is not None and epss >= epss_threshold) else 0.0
            
            for k in range(h):
                pred_value = base_pred
                # Optional jump for high-severity CVEs
                if use_jump and cvss is not None and cvss > cvss_threshold and k >= 10:
                    pred_value = min(1.0, base_pred + 0.2)  # Step increase after k=10
                # Optional decay for remediated CVEs
                if use_remediation and remediation is True:
                    pred_value = base_pred * np.exp(-0.05 * k)  # Exponential decay
                pred[i, j, k] = pred_value
    
    print(f"Generated predictions with shape: {pred.shape} (N={n}, L_max={l_max}, H={h})")
    return pred, cve_ids, l_max

def save_to_netcdf(pred, cve_ids, l_max, output_file='predictions.nc', netcdf_format='NETCDF4'):
    """
    Save predictions to a NetCDF file.
    """
    try:
        ds = xr.Dataset(
            {
                'pred': (['cve_id', 'timestamp', 'horizon'], pred)
            },
            coords={
                'cve_id': cve_ids,
                'timestamp': np.arange(l_max),
                'horizon': np.arange(30)
            }
        )
        ds.to_netcdf(output_file, engine='netcdf4', format=netcdf_format)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving NetCDF file: {e}")
        raise

def main(parquet_directory, output_file='predictions.nc', netcdf_format='NETCDF4', use_jump=False, use_remediation=False):
    """
    Main function to process Parquet files and generate NetCDF output using AR(1).
    """
    cve_data = read_parquet_files(parquet_directory)
    pred, cve_ids, l_max = generate_predictions(cve_data, use_jump=use_jump, use_remediation=use_remediation)
    save_to_netcdf(pred, cve_ids, l_max, output_file, netcdf_format)

if __name__ == "__main__":
    parquet_directory = 'data/full_db/processed/final_full_data.parquet'  
    try:
        main(
            parquet_directory=parquet_directory,
            output_file='ml_pipeline\stupid_model\predictions.nc',
            # Use this to make the stupid model more complex
            use_jump=False,  # Set to True to enable jump sensitivity
            use_remediation=False  # Set to True to enable remediation decay
        )
    except Exception as e:
        print(f"Error in main: {e}")