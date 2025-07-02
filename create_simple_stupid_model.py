#!/usr/bin/env python3
"""
Simplified stupid model that creates true persistence forecasting.
This version avoids the memory error by processing data more efficiently.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def create_simple_stupid_predictions():
    """Create a simple stupid model that just repeats current values."""
    
    print("🔄 CREATING SIMPLE STUPID MODEL")
    print("=" * 50)
    
    # Load template
    template_path = Path("ml_pipeline/stupid_model/predictions_stream_copy.nc")
    ds = xr.open_dataset(template_path)
    
    print(f"Template shape: {ds.pred.shape}")
    
    # Create stupid predictions: just copy the true values to all horizons
    print("Creating persistence predictions...")
    
    stupid_preds = np.full_like(ds.pred.values, np.nan, dtype=np.float32)
    
    # For each CVE and timestamp, set all horizons to the current true value
    for cve_idx in range(ds.sizes['cve']):
        if cve_idx % 5000 == 0:
            print(f"  Processing CVE {cve_idx}/{ds.sizes['cve']}")
        
        eval_mask = ds.eval_mask.values[cve_idx, :].astype(bool)
        mask_h = ds.mask_h.values[cve_idx, :, :].astype(bool)
        true_values = ds.true.values[cve_idx, :, :]
        
        for time_idx in np.where(eval_mask)[0]:
            # Get the true value at horizon 0 (current day)
            current_true = true_values[time_idx, 0]
            
            if not np.isnan(current_true):
                # Set all valid horizons to this same value
                for h in range(ds.sizes['horizon']):
                    if mask_h[time_idx, h]:
                        stupid_preds[cve_idx, time_idx, h] = current_true
    
    # Create output dataset
    output_ds = ds.copy(deep=True)
    output_ds = output_ds.assign(pred=(['cve', 'time', 'horizon'], stupid_preds))
    
    # Add metadata
    output_ds.attrs['title'] = 'Simple Stupid Model Predictions'
    output_ds.attrs['description'] = 'True persistence: repeat current day value for all horizons'
    output_ds.attrs['model_type'] = 'simple_persistence'
    
    # Save
    output_path = Path("ml_pipeline/stupid_model/simple_stupid_predictions.nc")
    encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in output_ds.data_vars}
    output_ds.to_netcdf(output_path, encoding=encoding)
    
    print(f"✓ Saved simple stupid predictions to: {output_path}")
    
    # Validate one CVE
    print("\n🔍 VALIDATION:")
    target_cve = "CVE-2024-3094"
    if target_cve in ds.cve.values:
        cve_idx = list(ds.cve.values).index(target_cve)
        cve_data = output_ds.isel(cve=cve_idx)
        
        eval_mask = cve_data.eval_mask.values.astype(bool)
        eval_times = np.where(eval_mask)[0]
        
        if len(eval_times) > 0:
            t_idx = eval_times[0]
            pred_row = cve_data.pred.values[t_idx, :]
            mask_row = cve_data.mask_h.values[t_idx, :].astype(bool)
            
            valid_preds = pred_row[mask_row]
            unique_count = len(np.unique(valid_preds[~np.isnan(valid_preds)]))
            
            print(f"CVE {target_cve} at timestamp {t_idx}:")
            print(f"  Predictions: {pred_row[:10]}")
            print(f"  Unique values: {unique_count} (should be 1 for persistence)")
            print(f"  All identical? {unique_count <= 1}")
    
    return output_ds

if __name__ == "__main__":
    create_simple_stupid_predictions() 