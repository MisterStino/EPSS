#!/usr/bin/env python3
"""
Examine if truncating 30 days from time dimension solves the horizon bias issue.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)

def examine_truncation_solution():
    """Examine if truncating 30 days solves the bias."""
    
    print("🔍 EXAMINING TRUNCATION SOLUTION")
    print("=" * 60)
    
    # Load data
    nc_path = Path("ml_plot/eval_plots/models/stupid_predictions.nc")
    ds = xr.open_dataset(nc_path)
    
    print(f"Original data shape: {ds.pred.shape}")
    print(f"Dimensions: {ds.pred.dims}")
    
    target_cve = "CVE-2024-3094"
    cve_idx = list(ds.cve.values).index(target_cve)
    
    # Get original data
    y_true_all = ds['true'].values[cve_idx, :, :]  # [time, horizon]
    y_pred_all = ds['pred'].values[cve_idx, :, :]  # [time, horizon]
    mask_h_all = ds['mask_h'].values[cve_idx, :, :]  # [time, horizon]
    eval_mask_all = ds['eval_mask'].values[cve_idx, :]  # [time]
    
    print(f"\nOriginal CVE data shape: {y_true_all.shape} (time, horizon)")
    
    # Find evaluation time range
    eval_times = np.where(eval_mask_all == 1)[0]
    print(f"Original evaluation times: {eval_times[0]} to {eval_times[-1]} ({len(eval_times)} days)")
    
    print("\n" + "="*60)
    print("STEP 1: UNDERSTANDING THE CURRENT PROBLEM")
    print("="*60)
    
    # Show current bias
    print("Current horizon bias (original data):")
    current_means = []
    for h_idx in range(30):
        mask_h = mask_h_all[:, h_idx] == 1
        mask_eval = eval_mask_all == 1
        valid = mask_h & mask_eval & ~np.isnan(y_pred_all[:, h_idx])
        
        if np.any(valid):
            pred_prob = _log_to_prob(y_pred_all[valid, h_idx])
            mean_pred = np.mean(pred_prob)
            current_means.append(mean_pred)
            
            valid_times = np.where(valid)[0]
            print(f"  Horizon {h_idx+1:2d}: {len(valid_times):3d} points, "
                  f"times {valid_times[0]}-{valid_times[-1]}, mean={mean_pred:.6f}")
        else:
            current_means.append(np.nan)
    
    print("\n" + "="*60)
    print("STEP 2: PROPOSED SOLUTION - TRUNCATE LAST 30 DAYS")
    print("="*60)
    
    # Truncate the last 30 time steps
    truncated_length = y_true_all.shape[0] - 30
    y_true_trunc = y_true_all[:truncated_length, :]
    y_pred_trunc = y_pred_all[:truncated_length, :]
    mask_h_trunc = mask_h_all[:truncated_length, :]
    eval_mask_trunc = eval_mask_all[:truncated_length]
    
    print(f"Truncated data shape: {y_true_trunc.shape}")
    
    # Find new evaluation time range
    eval_times_trunc = np.where(eval_mask_trunc == 1)[0]
    if len(eval_times_trunc) > 0:
        print(f"Truncated evaluation times: {eval_times_trunc[0]} to {eval_times_trunc[-1]} ({len(eval_times_trunc)} days)")
    else:
        print("No evaluation times in truncated data!")
        return
    
    print("\n" + "="*60)
    print("STEP 3: ANALYZING TRUNCATED DATA BIAS")
    print("="*60)
    
    # Check if truncation solves the bias
    print("Horizon analysis with truncated data:")
    truncated_means = []
    for h_idx in range(30):
        mask_h = mask_h_trunc[:, h_idx] == 1
        mask_eval = eval_mask_trunc == 1
        valid = mask_h & mask_eval & ~np.isnan(y_pred_trunc[:, h_idx])
        
        if np.any(valid):
            pred_prob = _log_to_prob(y_pred_trunc[valid, h_idx])
            mean_pred = np.mean(pred_prob)
            truncated_means.append(mean_pred)
            
            valid_times = np.where(valid)[0]
            print(f"  Horizon {h_idx+1:2d}: {len(valid_times):3d} points, "
                  f"times {valid_times[0]}-{valid_times[-1]}, mean={mean_pred:.6f}")
        else:
            truncated_means.append(np.nan)
            print(f"  Horizon {h_idx+1:2d}: No valid points")
    
    print("\n" + "="*60)
    print("STEP 4: THEORETICAL ANALYSIS")
    print("="*60)
    
    print("Question: Does truncating 30 days solve the bias?")
    print()
    print("REASONING:")
    print("1. Original problem: Different horizons have different latest evaluation times")
    print("2. Horizon 1 could evaluate up to time 373")
    print("3. Horizon 30 could only evaluate up to time 344")
    print("4. Difference = 373 - 344 = 29 days")
    print()
    print("5. If we truncate 30 days from the end:")
    print("   - New max time = 763 - 30 = 733")
    print("   - New max evaluation time ≈ 374 - 30 = 344")
    print("   - Horizon 1 can now evaluate up to: 344 - 1 = 343")
    print("   - Horizon 30 can now evaluate up to: 344 - 30 = 314")
    print()
    print("6. NEW DIFFERENCE = 343 - 314 = 29 days")
    print("   → The bias still exists! Just shifted earlier.")
    
    print("\n" + "="*60)
    print("STEP 5: WHAT WOULD ACTUALLY SOLVE THE BIAS?")
    print("="*60)
    
    print("To eliminate bias completely, we need:")
    print("ALL horizons to use EXACTLY the same evaluation time points")
    print()
    print("Solution approaches:")
    print("1. Restrict evaluation to times where ALL horizons are valid")
    print("2. Use only evaluation times t where t+30 ≤ max_time")
    print("3. This means: eval_times ≤ max_time - 30")
    
    # Calculate the fair evaluation range
    max_time_idx = len(eval_mask_all) - 1
    fair_max_eval_time = max_time_idx - 30
    
    print(f"\nFair evaluation range calculation:")
    print(f"  Original max time: {max_time_idx}")
    print(f"  Fair max eval time: {fair_max_eval_time}")
    print(f"  Original eval times: {eval_times[0]} to {eval_times[-1]}")
    
    # Find fair evaluation times
    fair_eval_times = eval_times[eval_times <= fair_max_eval_time]
    if len(fair_eval_times) > 0:
        print(f"  Fair eval times: {fair_eval_times[0]} to {fair_eval_times[-1]} ({len(fair_eval_times)} days)")
        print(f"  Days lost: {len(eval_times) - len(fair_eval_times)}")
    else:
        print("  No fair evaluation times available!")
    
    print("\n" + "="*60)
    print("STEP 6: TESTING THE FAIR SOLUTION")
    print("="*60)
    
    if len(fair_eval_times) > 0:
        print("Testing fair evaluation (all horizons use same time points):")
        
        fair_means = []
        for h_idx in range(30):
            # Only use the fair evaluation times
            mask_h = mask_h_all[fair_eval_times, h_idx] == 1
            valid_fair_times = fair_eval_times[mask_h]
            
            if len(valid_fair_times) > 0:
                pred_prob = _log_to_prob(y_pred_all[valid_fair_times, h_idx])
                mean_pred = np.mean(pred_prob)
                fair_means.append(mean_pred)
                
                print(f"  Horizon {h_idx+1:2d}: {len(valid_fair_times):3d} points, mean={mean_pred:.6f}")
            else:
                fair_means.append(np.nan)
                print(f"  Horizon {h_idx+1:2d}: No valid points")
        
        # Check if fair solution eliminates bias
        fair_means_clean = [x for x in fair_means if not np.isnan(x)]
        if len(fair_means_clean) > 1:
            bias_range = max(fair_means_clean) - min(fair_means_clean)
            print(f"\nFair solution bias range: {bias_range:.6f}")
            print(f"Should be ≈0 for persistence model: {'✓ GOOD' if bias_range < 0.001 else '✗ STILL BIASED'}")

if __name__ == "__main__":
    examine_truncation_solution() 