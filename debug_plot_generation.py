#!/usr/bin/env python3
"""
Debug script to understand why the plot shows downward trend for persistence model.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)

def debug_plot_generation():
    """Debug the plot generation to understand the downward trend."""
    
    print("🔍 DEBUGGING PLOT GENERATION")
    print("=" * 50)
    
    # Load the stupid model data
    nc_path = Path("ml_plot/eval_plots/models/stupid_predictions.nc")
    ds = xr.open_dataset(nc_path)
    
    target_cve = "CVE-2024-3094"
    cve_idx = list(ds.cve.values).index(target_cve)
    
    print(f"Analyzing {target_cve} (index {cve_idx})")
    
    # Get the data
    y_true_all = ds.true.values
    y_pred_all = ds.pred.values
    mask_h_all = ds.mask_h.values
    eval_mask_all = ds.eval_mask.values
    time_coords = pd.to_datetime(ds.time.values[cve_idx, :])
    horizons = ds.horizon.values
    
    print(f"Total evaluation timestamps: {np.sum(eval_mask_all[cve_idx, :])}")
    
    # Replicate the exact plot generation logic
    true_means = []
    pred_means = []
    horizon_details = []
    
    for h_idx in range(len(horizons)):
        h = horizons[h_idx]
        print(f"\n--- HORIZON {h} (index {h_idx}) ---")
        
        # Get data for this horizon (exact same logic as plot function)
        y_true_h = y_true_all[cve_idx, :, h_idx]
        y_pred_h = y_pred_all[cve_idx, :, h_idx]
        
        # Apply masks (exact same logic)
        mask_h = mask_h_all[cve_idx, :, h_idx] == 1
        mask_eval = eval_mask_all[cve_idx, :] == 1
        valid = mask_h & mask_eval & ~np.isnan(y_true_h) & ~np.isnan(y_pred_h)
        
        print(f"Valid points: {np.sum(valid)}")
        
        if np.any(valid):
            # Get the actual time indices being used
            valid_indices = np.where(valid)[0]
            valid_times = time_coords[valid_indices]
            valid_times_clean = valid_times[~pd.isna(valid_times)]
            
            if len(valid_times_clean) > 0:
                print(f"Time range: {valid_times_clean.min()} to {valid_times_clean.max()}")
                print(f"First 5 timestamps: {valid_indices[:5]}")
                print(f"Last 5 timestamps: {valid_indices[-5:]}")
            
            # Convert to probability space and compute mean (exact same logic)
            true_prob = _log_to_prob(y_true_h[valid])
            pred_prob = _log_to_prob(y_pred_h[valid])
            
            true_mean = np.mean(true_prob)
            pred_mean = np.mean(pred_prob)
            
            print(f"True mean: {true_mean:.6f}")
            print(f"Pred mean: {pred_mean:.6f}")
            
            # Check if predictions are identical (as expected for persistence)
            unique_preds = len(np.unique(y_pred_h[valid]))
            print(f"Unique prediction values: {unique_preds}")
            
            if unique_preds > 1:
                print(f"⚠️  WARNING: Persistence model should have identical predictions!")
                print(f"Prediction range: {y_pred_h[valid].min():.6f} to {y_pred_h[valid].max():.6f}")
                
                # Show first few prediction values
                print(f"First 5 predictions: {y_pred_h[valid][:5]}")
            
            true_means.append(true_mean)
            pred_means.append(pred_mean)
            
            horizon_details.append({
                'horizon': h,
                'valid_count': np.sum(valid),
                'true_mean': true_mean,
                'pred_mean': pred_mean,
                'unique_preds': unique_preds,
                'time_range_start': valid_times_clean.min() if len(valid_times_clean) > 0 else None,
                'time_range_end': valid_times_clean.max() if len(valid_times_clean) > 0 else None
            })
        else:
            print("No valid points")
            true_means.append(np.nan)
            pred_means.append(np.nan)
    
    # Analyze the pattern
    print(f"\n🔍 PATTERN ANALYSIS")
    print("=" * 30)
    
    df_details = pd.DataFrame(horizon_details)
    print(df_details)
    
    # Check if the issue is temporal bias
    print(f"\nTemporal Analysis:")
    print(f"Horizon 1 time range: {df_details.iloc[0]['time_range_start']} to {df_details.iloc[0]['time_range_end']}")
    print(f"Horizon 30 time range: {df_details.iloc[-1]['time_range_start']} to {df_details.iloc[-1]['time_range_end']}")
    
    # Check trend
    valid_pred_means = [x for x in pred_means if not np.isnan(x)]
    if len(valid_pred_means) > 1:
        trend_slope = np.polyfit(range(len(valid_pred_means)), valid_pred_means, 1)[0]
        print(f"Predicted values trend slope: {trend_slope:.6f}")
        print(f"Trend direction: {'Downward' if trend_slope < 0 else 'Upward'}")
    
    # Check if all predictions should be identical for persistence
    print(f"\nPersistence Model Validation:")
    all_unique_counts = [d['unique_preds'] for d in horizon_details if 'unique_preds' in d]
    if all_unique_counts:
        max_unique = max(all_unique_counts)
        print(f"Maximum unique predictions in any horizon: {max_unique}")
        if max_unique > 1:
            print(f"❌ ERROR: Persistence model should have identical predictions within each horizon!")
        else:
            print(f"✅ OK: All horizons have identical predictions")
    
    return horizon_details

def main():
    results = debug_plot_generation()
    return results

if __name__ == "__main__":
    main() 