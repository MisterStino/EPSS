#!/usr/bin/env python3
"""
Examine the mask_h logic to understand why different horizons have different valid times.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def examine_mask_logic():
    """Examine the mask_h logic to understand the temporal constraints."""
    
    print("🔍 EXAMINING MASK_H LOGIC")
    print("=" * 60)
    
    # Load data
    nc_path = Path("ml_plot/eval_plots/models/stupid_predictions.nc")
    ds = xr.open_dataset(nc_path)
    
    target_cve = "CVE-2024-3094"
    cve_idx = list(ds.cve.values).index(target_cve)
    
    # Get the masks
    mask_h_all = ds['mask_h'].values[cve_idx, :, :]  # [time, horizon]
    eval_mask_all = ds['eval_mask'].values[cve_idx, :]  # [time]
    
    print(f"Data shape: {mask_h_all.shape} (time, horizon)")
    print(f"Total time steps: {len(eval_mask_all)}")
    
    # Find evaluation time range
    eval_times = np.where(eval_mask_all == 1)[0]
    print(f"Evaluation times: {eval_times[0]} to {eval_times[-1]} (indices)")
    print(f"That's {len(eval_times)} evaluation days")
    
    print("\n" + "="*60)
    print("UNDERSTANDING MASK_H LOGIC:")
    print("="*60)
    print("mask_h[t, h] = 1 means: at time t, we can make a valid prediction for horizon h")
    print("This should mean: t + h is within the valid data range")
    print()
    
    # Let's check this logic
    max_time = len(eval_mask_all) - 1
    print(f"Maximum time index: {max_time}")
    
    print("\nFor each horizon, what's the latest time we can make a prediction?")
    for h in [1, 5, 10, 15, 20, 25, 30]:
        h_idx = h - 1  # Convert to 0-based index
        
        # Find latest time where this horizon is valid
        valid_times_for_h = np.where(mask_h_all[:, h_idx] == 1)[0]
        if len(valid_times_for_h) > 0:
            latest_time = valid_times_for_h[-1]
            target_time = latest_time + h  # Where this prediction points to
            
            print(f"Horizon {h:2d}: Latest prediction time = {latest_time}, "
                  f"predicts time {target_time} (max={max_time})")
            
            # Check if this makes sense
            if target_time <= max_time:
                print(f"           ✓ Valid: {latest_time} + {h} = {target_time} ≤ {max_time}")
            else:
                print(f"           ✗ Invalid: {latest_time} + {h} = {target_time} > {max_time}")
    
    print("\n" + "="*60)
    print("EXAMINING SPECIFIC CASES:")
    print("="*60)
    
    # Let's look at the boundary cases
    for eval_time in [374, 373, 372, 370, 365, 360, 350, 345]:
        if eval_time < len(eval_mask_all):
            is_eval = eval_mask_all[eval_time] == 1
            print(f"\nTime {eval_time} (eval={is_eval}):")
            
            for h in [1, 10, 20, 30]:
                h_idx = h - 1
                is_valid_h = mask_h_all[eval_time, h_idx] == 1
                target_time = eval_time + h
                
                print(f"  Horizon {h:2d}: mask_h={is_valid_h}, "
                      f"would predict time {target_time} (max={max_time})")
    
    print("\n" + "="*60)
    print("THE REAL EXPLANATION:")
    print("="*60)
    print("Now I understand! The mask_h constraint is:")
    print("mask_h[t, h] = 1 only if t + h ≤ max_valid_time")
    print()
    print("This means:")
    print("- Horizon 1: Can predict up to time 373 (because 373 + 1 = 374 ≤ max)")
    print("- Horizon 30: Can only predict up to time 344 (because 344 + 30 = 374 ≤ max)")
    print()
    print("So different horizons have different LATEST evaluation times!")
    print("This creates the averaging bias we observed.")

if __name__ == "__main__":
    examine_mask_logic() 