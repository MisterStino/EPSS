#!/usr/bin/env python3
"""
Examine exactly which time periods are being used for each horizon.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)

def examine_time_periods():
    """Examine exactly which time periods are used for each horizon."""
    
    print("🔍 EXAMINING TIME PERIODS PER HORIZON")
    print("=" * 60)
    
    # Load data
    nc_path = Path("ml_plot/eval_plots/models/stupid_predictions.nc")
    ds = xr.open_dataset(nc_path)
    
    target_cve = "CVE-2024-3094"
    cve_idx = list(ds.cve.values).index(target_cve)
    
    # Get the data for this CVE
    y_true_all = ds['true'].values[cve_idx, :, :]  # [time, horizon]
    y_pred_all = ds['pred'].values[cve_idx, :, :]  # [time, horizon] 
    mask_h_all = ds['mask_h'].values[cve_idx, :, :]  # [time, horizon]
    eval_mask_all = ds['eval_mask'].values[cve_idx, :]  # [time]
    
    print(f"Total time steps: {len(eval_mask_all)}")
    print(f"Evaluation time steps: {np.sum(eval_mask_all)}")
    
    # Find evaluation time range
    eval_times = np.where(eval_mask_all == 1)[0]
    print(f"Evaluation times: {eval_times[0]} to {eval_times[-1]} (indices)")
    print(f"That's {len(eval_times)} evaluation days")
    
    print("\n" + "="*60)
    print("ANALYZING WHICH TIMES ARE USED FOR EACH HORIZON:")
    print("="*60)
    
    # For each horizon, find which times are actually used
    for h_idx in [0, 1, 4, 9, 19, 29]:  # Sample horizons
        mask_h = mask_h_all[:, h_idx] == 1
        mask_eval = eval_mask_all == 1
        valid = mask_h & mask_eval
        
        valid_times = np.where(valid)[0]
        
        print(f"\nHorizon {h_idx+1:2d}:")
        print(f"  Valid times: {valid_times[0]} to {valid_times[-1]} ({len(valid_times)} days)")
        print(f"  Missing from eval range: {len(eval_times) - len(valid_times)} days")
        
        # Show which specific times are missing
        missing_times = set(eval_times) - set(valid_times)
        if missing_times:
            missing_sorted = sorted(list(missing_times))
            print(f"  Missing times: {missing_sorted[:5]}..." if len(missing_sorted) > 5 else f"  Missing times: {missing_sorted}")
    
    print("\n" + "="*60)
    print("EXAMINING ACTUAL PREDICTION VALUES BY TIME PERIOD:")
    print("="*60)
    
    # Let's examine the actual prediction values in different time periods
    eval_times = np.where(eval_mask_all == 1)[0]
    
    # Early evaluation period (first 30 days)
    early_times = eval_times[:30]
    # Late evaluation period (last 30 days)  
    late_times = eval_times[-30:]
    
    print(f"\nEarly evaluation period: times {early_times[0]} to {early_times[-1]}")
    print(f"Late evaluation period: times {late_times[0]} to {late_times[-1]}")
    
    # Check predictions in early vs late periods
    for h_idx in [0, 9, 19, 29]:  # Sample horizons
        print(f"\nHorizon {h_idx+1}:")
        
        # Early period predictions
        early_valid = mask_h_all[early_times, h_idx] == 1
        if np.any(early_valid):
            early_preds = y_pred_all[early_times[early_valid], h_idx]
            early_prob = _log_to_prob(early_preds)
            print(f"  Early period mean prediction: {np.mean(early_prob):.6f}")
        
        # Late period predictions
        late_valid = mask_h_all[late_times, h_idx] == 1
        if np.any(late_valid):
            late_preds = y_pred_all[late_times[late_valid], h_idx]
            late_prob = _log_to_prob(late_preds)
            print(f"  Late period mean prediction:  {np.mean(late_prob):.6f}")
    
    print("\n" + "="*60)
    print("EXAMINING THE PERSISTENCE MODEL LOGIC:")
    print("="*60)
    
    # Let's trace through what the persistence model actually does
    print("Remember: Persistence model uses EPSS[t] to predict ALL horizons t+1, t+2, ..., t+30")
    print("So predictions should be identical across horizons for each time t")
    
    # Check a few specific times
    for i, t in enumerate(eval_times[:5]):
        if mask_h_all[t, 0] == 1:  # If any horizon is valid
            pred_t = y_pred_all[t, :]  # All horizons for this time
            pred_prob = _log_to_prob(pred_t)
            
            print(f"\nTime {t}: All predictions = {pred_prob[0]:.6f}")
            print(f"  Identical across horizons? {np.allclose(pred_prob, pred_prob[0], rtol=1e-6)}")
    
    print("\n" + "="*60)
    print("THE MYSTERY: WHY DIFFERENT AVERAGES?")
    print("="*60)
    print("If all horizons have identical predictions at each time,")
    print("how can the averages be different across horizons?")
    print("The only explanation is that different horizons use different time sets!")

if __name__ == "__main__":
    examine_time_periods() 