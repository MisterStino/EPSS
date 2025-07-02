#!/usr/bin/env python3
"""
Rigorous validation of the original stupid model with correct understanding.
The stupid model should use EPSS[t] to predict horizons t+1, t+2, ..., t+30.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)

def validate_original_stupid_model():
    """Validate the original stupid model with correct understanding."""
    
    print("🔬 RIGOROUS VALIDATION: ORIGINAL STUPID MODEL")
    print("=" * 60)
    
    # Load the original stupid model
    print("\n[STEP 1] Loading original stupid model...")
    nc_path = Path("ml_plot/eval_plots/models/stupid_predictions.nc")
    ds = xr.open_dataset(nc_path)
    
    # Load EPSS data to validate against
    print("\n[STEP 2] Loading EPSS data for validation...")
    epss_path = Path("data/epss/processed/epss_processed.parquet")
    epss_df = pd.read_parquet(epss_path)
    epss_df['date'] = pd.to_datetime(epss_df['date'])
    
    # Focus on CVE-2024-3094
    target_cve = "CVE-2024-3094"
    print(f"\n[STEP 3] Analyzing {target_cve}...")
    
    if target_cve not in ds.cve.values:
        print(f"❌ {target_cve} not found")
        return
    
    cve_idx = list(ds.cve.values).index(target_cve)
    cve_data = ds.isel(cve=cve_idx)
    
    # Get EPSS data for this CVE
    cve_epss = epss_df[epss_df['cve'] == target_cve].copy()
    cve_epss = cve_epss.sort_values('date')
    
    print(f"✓ Found {len(cve_epss)} EPSS records for {target_cve}")
    
    # Analyze the algorithm implementation
    print(f"\n[STEP 4] Validating stupid model algorithm...")
    
    pred_data = cve_data.pred.values  # [time, horizon]
    true_data = cve_data.true.values  # [time, horizon] 
    eval_mask = cve_data.eval_mask.values.astype(bool)  # [time]
    mask_h = cve_data.mask_h.values.astype(bool)  # [time, horizon]
    time_coords = pd.to_datetime(cve_data.time.values)
    
    print(f"Evaluation timestamps: {np.sum(eval_mask)}")
    
    # Check several evaluation timestamps
    eval_times = np.where(eval_mask)[0]
    validation_results = []
    
    for i, t_idx in enumerate(eval_times[:5]):  # Check first 5 evaluation times
        eval_date = time_coords[t_idx]
        if pd.isna(eval_date):
            continue
            
        print(f"\n  Validation {i+1}: Timestamp {t_idx}, Date {eval_date.date()}")
        
        # Get predictions for all horizons at this timestamp
        pred_row = pred_data[t_idx, :]
        mask_row = mask_h[t_idx, :]
        
        valid_preds = pred_row[mask_row]
        
        # Find the EPSS value at time t (evaluation date)
        epss_at_t = cve_epss[cve_epss['date'].dt.date == eval_date.date()]
        
        if len(epss_at_t) == 0:
            print(f"    ⚠️  No EPSS data for {eval_date.date()}")
            continue
            
        epss_value_t = epss_at_t.iloc[0]['epss']
        epss_log_t = np.log(epss_value_t + 1e-6)
        
        print(f"    EPSS at t: {epss_value_t:.6f} (log: {epss_log_t:.6f})")
        print(f"    Predictions range: {valid_preds.min():.6f} to {valid_preds.max():.6f}")
        print(f"    All predictions identical? {np.allclose(valid_preds, epss_log_t, atol=1e-6)}")
        
        # Check if predictions match EPSS[t]
        matches_epss_t = np.allclose(valid_preds, epss_log_t, atol=1e-6)
        
        if matches_epss_t:
            print(f"    ✅ CORRECT: Predictions match EPSS[t]")
        else:
            print(f"    ❌ INCORRECT: Predictions don't match EPSS[t]")
            print(f"    Expected: {epss_log_t:.6f}")
            print(f"    Got: {valid_preds[0]:.6f} (first prediction)")
            
            # Check if it matches true[t, h=0] instead (my wrong implementation)
            true_h0 = true_data[t_idx, 0]
            if np.allclose(valid_preds, true_h0, atol=1e-6):
                print(f"    🔍 NOTE: Predictions match true[t, h=0] = {true_h0:.6f}")
                print(f"    This suggests wrong implementation (using t+1 instead of t)")
        
        validation_results.append({
            'timestamp': t_idx,
            'date': eval_date,
            'epss_t': epss_value_t,
            'epss_log_t': epss_log_t,
            'pred_first': valid_preds[0] if len(valid_preds) > 0 else np.nan,
            'matches_epss_t': matches_epss_t,
            'true_h0': true_data[t_idx, 0]
        })
    
    # Step 5: Analyze the evaluation metrics pattern
    print(f"\n[STEP 5] Analyzing evaluation metrics pattern...")
    
    horizons = [1, 5, 10, 15, 20, 25, 30]
    metrics_analysis = []
    
    for h in horizons:
        idx = h - 1  # Convert to 0-based
        
        # Extract data for this horizon
        y_true_log = true_data[:, idx]
        y_pred_log = pred_data[:, idx]
        
        # Apply masks
        mask_h_col = mask_h[:, idx]
        valid = mask_h_col & eval_mask & ~np.isnan(y_true_log) & ~np.isnan(y_pred_log)
        
        if np.sum(valid) > 0:
            # Convert to probability space
            y_t = _log_to_prob(y_true_log[valid])
            y_p = _log_to_prob(y_pred_log[valid])
            
            mae = np.mean(np.abs(y_t - y_p))
            mean_true = np.mean(y_t)
            mean_pred = np.mean(y_p)
            
            metrics_analysis.append({
                'horizon': h,
                'valid_count': np.sum(valid),
                'mae': mae,
                'mean_true': mean_true,
                'mean_pred': mean_pred
            })
            
            print(f"  Horizon {h:2d}: {np.sum(valid):3d} points, MAE={mae:.6f}, True={mean_true:.6f}, Pred={mean_pred:.6f}")
    
    # Step 6: Expected behavior analysis
    print(f"\n[STEP 6] Expected behavior analysis...")
    
    if len(metrics_analysis) > 1:
        df_metrics = pd.DataFrame(metrics_analysis)
        
        # For a correct stupid model:
        # - Horizon 1: Small error (predicting tomorrow with today's value)
        # - Horizon 30: Large error (predicting 30 days ahead with today's value)
        # - MAE should increase with horizon
        
        mae_trend = np.polyfit(df_metrics['horizon'], df_metrics['mae'], 1)[0]
        print(f"  MAE trend slope: {mae_trend:.6f} (should be positive)")
        print(f"  MAE increases with horizon? {mae_trend > 0}")
        
        # Check if horizon 1 has small error (not zero)
        h1_mae = df_metrics[df_metrics['horizon'] == 1]['mae'].iloc[0]
        print(f"  Horizon 1 MAE: {h1_mae:.6f} (should be small but not zero)")
        print(f"  Horizon 1 reasonable? {0.001 < h1_mae < 0.1}")
    
    # Step 7: Summary
    print(f"\n[STEP 7] SUMMARY...")
    
    df_val = pd.DataFrame(validation_results)
    if len(df_val) > 0:
        correct_implementations = df_val['matches_epss_t'].sum()
        total_checked = len(df_val)
        
        print(f"  Checked {total_checked} evaluation timestamps")
        print(f"  Correct implementations: {correct_implementations}/{total_checked}")
        
        if correct_implementations == total_checked:
            print(f"  ✅ VALIDATION PASSED: Original stupid model is correctly implemented")
        elif correct_implementations == 0:
            print(f"  ❌ VALIDATION FAILED: Original stupid model appears to be incorrectly implemented")
        else:
            print(f"  ⚠️  PARTIAL: Mixed results suggest investigation needed")
    
    return validation_results, metrics_analysis

def main():
    results = validate_original_stupid_model()
    return results

if __name__ == "__main__":
    main() 