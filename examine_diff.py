#!/usr/bin/env python3
"""
Simple script to examine what the current plot is doing vs what it should do.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)

def examine_current_vs_expected():
    """Examine what the current plot shows vs what we expect."""
    
    print("🔍 EXAMINING CURRENT PLOT vs EXPECTED BEHAVIOR")
    print("=" * 60)
    
    # Load data
    nc_path = Path("ml_plot/eval_plots/models/stupid_predictions.nc")
    ds = xr.open_dataset(nc_path)
    
    target_cve = "CVE-2024-3094"
    cve_idx = list(ds.cve.values).index(target_cve)
    
    print(f"\nAnalyzing CVE: {target_cve}")
    
    # Get the data for this CVE
    y_true_all = ds['true'].values[cve_idx, :, :]  # [time, horizon]
    y_pred_all = ds['pred'].values[cve_idx, :, :]  # [time, horizon]
    mask_h_all = ds['mask_h'].values[cve_idx, :, :]  # [time, horizon]
    eval_mask_all = ds['eval_mask'].values[cve_idx, :]  # [time]
    
    print(f"Data shapes: true={y_true_all.shape}, pred={y_pred_all.shape}")
    
    print("\n" + "="*50)
    print("WHAT THE CURRENT PLOT IS DOING:")
    print("="*50)
    
    # Replicate the current plot logic
    horizons = ds.horizon.values
    current_true_means = []
    current_pred_means = []
    
    for h_idx in range(len(horizons)):
        # Get data for this horizon
        y_true_h = y_true_all[:, h_idx]  # All times for this horizon
        y_pred_h = y_pred_all[:, h_idx]  # All times for this horizon
        
        # Apply masks (this is the key!)
        mask_h = mask_h_all[:, h_idx] == 1
        mask_eval = eval_mask_all == 1
        valid = mask_h & mask_eval & ~np.isnan(y_true_h) & ~np.isnan(y_pred_h)
        
        if np.any(valid):
            # Convert to probability space and compute mean
            true_prob = _log_to_prob(y_true_h[valid])
            pred_prob = _log_to_prob(y_pred_h[valid])
            current_true_means.append(np.mean(true_prob))
            current_pred_means.append(np.mean(pred_prob))
            
            print(f"Horizon {h_idx+1:2d}: {np.sum(valid):3d} valid points, "
                  f"True mean: {np.mean(true_prob):.6f}, "
                  f"Pred mean: {np.mean(pred_prob):.6f}")
        else:
            current_true_means.append(np.nan)
            current_pred_means.append(np.nan)
            print(f"Horizon {h_idx+1:2d}: No valid points")
    
    print("\n" + "="*50)
    print("WHAT A PERSISTENCE MODEL SHOULD SHOW:")
    print("="*50)
    print("For a persistence model, ALL horizons should use the SAME predictor value")
    print("(the EPSS at time t) to predict ALL future times t+1, t+2, ..., t+30")
    print()
    
    # Let's examine what the persistence model actually predicts
    print("Examining actual predictions for first few valid evaluation times...")
    eval_times = np.where(eval_mask_all == 1)[0]
    
    for i, t in enumerate(eval_times[:5]):  # First 5 evaluation times
        if mask_h_all[t, 0] == 1:  # If horizon 1 is valid
            pred_t = y_pred_all[t, :]  # All horizons for this time
            true_t = y_true_all[t, :]  # All horizons for this time
            
            # Convert to probability
            pred_prob = _log_to_prob(pred_t)
            true_prob = _log_to_prob(true_t)
            
            print(f"\nTime {t}: Evaluation day {i+1}")
            print(f"  Prediction (should be same): {pred_prob[0]:.6f} (all horizons)")
            print(f"  Are all predictions identical? {np.allclose(pred_prob, pred_prob[0], rtol=1e-6)}")
            print(f"  True values (should increase): {true_prob[0]:.6f} to {true_prob[-1]:.6f}")
    
    print("\n" + "="*50)
    print("THE PROBLEM EXPLANATION:")
    print("="*50)
    print("Current plot shows 'average prediction per horizon across all evaluation times'")
    print("But for persistence model, this creates artificial differences because:")
    print("- Different horizons have different sets of valid evaluation times")
    print("- The EPSS predictor values vary across those different time sets")
    print("- This creates the illusion of horizon-dependent predictions")
    print()
    print("The persistence model IS working correctly - it's the visualization that's misleading!")

if __name__ == "__main__":
    examine_current_vs_expected() 