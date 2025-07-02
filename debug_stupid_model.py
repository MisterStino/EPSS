#!/usr/bin/env python3
"""
Diagnostic script to understand what's going wrong with stupid model evaluation.
This script will trace through the entire pipeline step by step.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)

def analyze_stupid_predictions():
    """Comprehensive analysis of the stupid model predictions and evaluation."""
    
    print("🔍 DIAGNOSTIC ANALYSIS: STUPID MODEL EVALUATION")
    print("=" * 60)
    
    # Step 1: Load the predictions file
    print("\n[STEP 1] Loading stupid predictions...")
    nc_path = Path("ml_pipeline/stupid_model/predictions_stream_copy.nc")
    
    if not nc_path.exists():
        print(f"❌ File not found: {nc_path}")
        return
    
    ds = xr.open_dataset(nc_path)
    print(f"✓ Loaded dataset with shape: {ds.pred.shape}")
    print(f"  CVEs: {ds.sizes['cve']}")
    print(f"  Time points: {ds.sizes['time']}")
    print(f"  Horizons: {ds.sizes['horizon']}")
    
    # Step 2: Analyze a specific CVE (CVE-2024-3094)
    print("\n[STEP 2] Analyzing CVE-2024-3094...")
    
    target_cve = "CVE-2024-3094"
    if target_cve not in ds.cve.values:
        print(f"❌ {target_cve} not found in dataset")
        print(f"Available CVEs (first 10): {list(ds.cve.values[:10])}")
        return
    
    # Get CVE index
    cve_idx = list(ds.cve.values).index(target_cve)
    print(f"✓ Found {target_cve} at index {cve_idx}")
    
    # Extract data for this CVE
    cve_data = ds.isel(cve=cve_idx)
    print(f"  CVE data shape: pred={cve_data.pred.shape}, true={cve_data.true.shape}")
    
    # Step 3: Analyze masks
    print("\n[STEP 3] Analyzing masks for this CVE...")
    
    eval_mask = cve_data.eval_mask.values.astype(bool)
    mask_h = cve_data.mask_h.values.astype(bool)
    
    print(f"  eval_mask shape: {eval_mask.shape}")
    print(f"  mask_h shape: {mask_h.shape}")
    print(f"  eval_mask True count: {np.sum(eval_mask)}")
    print(f"  mask_h True count per horizon:")
    
    for h in [0, 9, 19, 29]:  # Sample horizons
        h_count = np.sum(mask_h[:, h])
        combined_count = np.sum(eval_mask & mask_h[:, h])
        print(f"    Horizon {h+1}: mask_h={h_count}, combined={combined_count}")
    
    # Step 4: Check if predictions are identical across horizons
    print("\n[STEP 4] Checking prediction consistency across horizons...")
    
    pred_data = cve_data.pred.values  # [time, horizon]
    true_data = cve_data.true.values  # [time, horizon]
    
    # Find valid evaluation timestamps
    eval_times = np.where(eval_mask)[0]
    print(f"  Found {len(eval_times)} evaluation timestamps")
    
    if len(eval_times) > 0:
        # Check first evaluation timestamp
        t_idx = eval_times[0]
        pred_at_t = pred_data[t_idx, :]  # All horizons for this timestamp
        mask_at_t = mask_h[t_idx, :]     # Valid horizons for this timestamp
        
        print(f"  At timestamp {t_idx}:")
        print(f"    Valid horizons: {np.sum(mask_at_t)}")
        print(f"    Prediction values (first 10 horizons): {pred_at_t[:10]}")
        print(f"    Are all predictions identical? {np.allclose(pred_at_t[mask_at_t], pred_at_t[mask_at_t][0], equal_nan=True)}")
        
        # Check if they're identical in probability space too
        pred_prob = _log_to_prob(pred_at_t[mask_at_t])
        print(f"    In probability space (first 10): {pred_prob[:10]}")
        print(f"    Probability range: {pred_prob.min():.6f} to {pred_prob.max():.6f}")
    
    # Step 5: Simulate the evaluation logic exactly
    print("\n[STEP 5] Simulating evaluation logic...")
    
    horizons = list(range(1, 31))  # [1, 2, ..., 30]
    results = []
    
    for h in horizons[:5]:  # Test first 5 horizons
        idx = h - 1  # Convert to 0-based index
        
        y_true_log = true_data[:, idx]
        y_pred_log = pred_data[:, idx]
        
        # Apply masks exactly as in eval_plot.py
        mask_h_col = mask_h[:, idx] == 1
        mask_eval_col = eval_mask == 1
        valid = mask_h_col & mask_eval_col & ~np.isnan(y_true_log) & ~np.isnan(y_pred_log)
        
        valid_count = np.sum(valid)
        print(f"  Horizon {h}: {valid_count} valid points")
        
        if valid_count > 0:
            # Convert to probability space
            y_t = _log_to_prob(y_true_log[valid])
            y_p = _log_to_prob(y_pred_log[valid])
            
            mae = np.mean(np.abs(y_t - y_p))
            mean_true = np.mean(y_t)
            mean_pred = np.mean(y_p)
            
            print(f"    MAE: {mae:.6f}")
            print(f"    Mean true: {mean_true:.6f}")
            print(f"    Mean pred: {mean_pred:.6f}")
            print(f"    Difference: {abs(mean_true - mean_pred):.6f}")
            
            results.append({
                'horizon': h,
                'valid_count': valid_count,
                'mae': mae,
                'mean_true': mean_true,
                'mean_pred': mean_pred
            })
    
    # Step 6: Analyze the pattern
    print("\n[STEP 6] Pattern analysis...")
    
    if len(results) > 1:
        df_results = pd.DataFrame(results)
        print(df_results)
        
        # Check if valid counts are decreasing
        valid_counts = df_results['valid_count'].values
        if len(valid_counts) > 1:
            is_decreasing = all(valid_counts[i] >= valid_counts[i+1] for i in range(len(valid_counts)-1))
            print(f"\n  Valid counts decreasing with horizon? {is_decreasing}")
            
        # Check if means are changing
        true_means = df_results['mean_true'].values
        pred_means = df_results['mean_pred'].values
        
        true_trend = "increasing" if true_means[-1] > true_means[0] else "decreasing"
        pred_trend = "increasing" if pred_means[-1] > pred_means[0] else "decreasing"
        
        print(f"  True values trend: {true_trend}")
        print(f"  Predicted values trend: {pred_trend}")
        
        if true_trend != pred_trend:
            print("  ⚠️  OPPOSITE TRENDS DETECTED - This explains the diverging lines!")
    
    # Step 7: Deep dive into mask patterns
    print("\n[STEP 7] Deep dive into mask patterns...")
    
    print("  Analyzing how evaluation sets change per horizon...")
    
    for h in [1, 15, 30]:
        idx = h - 1
        mask_h_col = mask_h[:, idx]
        valid_indices = np.where(eval_mask & mask_h_col)[0]
        
        print(f"  Horizon {h}: valid time indices = {valid_indices[:10]}..." if len(valid_indices) > 10 else f"  Horizon {h}: valid time indices = {valid_indices}")
        
        if len(valid_indices) > 0:
            # Check temporal distribution
            time_coords = pd.to_datetime(cve_data.time.values)
            valid_dates = time_coords[valid_indices]
            valid_dates_clean = valid_dates[~pd.isna(valid_dates)]
            
            if len(valid_dates_clean) > 0:
                print(f"    Date range: {valid_dates_clean.min()} to {valid_dates_clean.max()}")
                
                # Check if later horizons have different temporal distribution
                if h == 30:
                    early_horizon_indices = np.where(eval_mask & mask_h[:, 0])[0]
                    print(f"    Horizon 1 had {len(early_horizon_indices)} points")
                    print(f"    Horizon 30 has {len(valid_indices)} points")
                    print(f"    Overlap: {len(set(early_horizon_indices) & set(valid_indices))} points")

def main():
    analyze_stupid_predictions()

if __name__ == "__main__":
    main() 