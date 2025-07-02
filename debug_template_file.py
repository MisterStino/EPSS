#!/usr/bin/env python3
"""
Analyze the template file used by stupid model to understand why predictions aren't identical.
"""

import numpy as np
import xarray as xr
from pathlib import Path

def analyze_template_file():
    print("🔍 ANALYZING TEMPLATE FILE: predictions_stream_copy.nc")
    print("=" * 60)
    
    nc_path = Path("ml_pipeline/stupid_model/predictions_stream_copy.nc")
    ds = xr.open_dataset(nc_path)
    
    # Check CVE-2024-3094 specifically
    target_cve = "CVE-2024-3094"
    cve_idx = list(ds.cve.values).index(target_cve)
    cve_data = ds.isel(cve=cve_idx)
    
    print(f"CVE: {target_cve}")
    print(f"Prediction data shape: {cve_data.pred.shape}")
    
    # Check first evaluation timestamp
    eval_mask = cve_data.eval_mask.values.astype(bool)
    eval_times = np.where(eval_mask)[0]
    
    if len(eval_times) > 0:
        t_idx = eval_times[0]
        print(f"\nAt first evaluation timestamp {t_idx}:")
        
        pred_row = cve_data.pred.values[t_idx, :]
        true_row = cve_data.true.values[t_idx, :]
        mask_row = cve_data.mask_h.values[t_idx, :]
        
        print(f"Predictions (first 10): {pred_row[:10]}")
        print(f"True values (first 10): {true_row[:10]}")
        print(f"Mask (first 10): {mask_row[:10]}")
        
        # Check if this is already from a real model (not a stupid baseline)
        unique_preds = np.unique(pred_row[mask_row.astype(bool)])
        print(f"Number of unique prediction values: {len(unique_preds)}")
        print(f"Prediction range: {pred_row[mask_row.astype(bool)].min():.6f} to {pred_row[mask_row.astype(bool)].max():.6f}")
        
        if len(unique_preds) > 5:
            print("⚠️  This appears to be from a REAL MODEL, not a baseline!")
            print("   The stupid model is using sophisticated predictions as template")
            print("   and then trying to overwrite them.")
    
    # Check dataset attributes
    print(f"\nDataset attributes:")
    for key, value in ds.attrs.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    analyze_template_file() 