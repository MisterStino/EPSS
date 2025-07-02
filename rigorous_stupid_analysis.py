#!/usr/bin/env python3
"""
Rigorous analysis of the stupid model evaluation to understand the "mirrored" pattern.
This script will independently compute metrics and compare with the evaluation pipeline.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)

def analyze_cve_rigorously(cve_id="CVE-2024-3094"):
    """Rigorous analysis of a specific CVE to understand evaluation behavior."""
    
    print(f"🔬 RIGOROUS ANALYSIS: {cve_id}")
    print("=" * 60)
    
    # Step 1: Load the stupid model predictions
    print("\n[STEP 1] Loading stupid model predictions...")
    
    # Try different possible paths
    possible_paths = [
        Path("ml_pipeline/stupid_model/simple_stupid_predictions.nc"),
        Path("ml_plot/eval_plots/models/stupid_predictions.nc"),
        Path("outputs/stupid_predictions/stupid_predictions.nc")
    ]
    
    ds = None
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found predictions at: {path}")
            ds = xr.open_dataset(path)
            break
    
    if ds is None:
        print("❌ Could not find stupid predictions file")
        return
    
    print(f"Dataset shape: {ds.pred.shape}")
    print(f"CVEs available: {len(ds.cve.values)}")
    
    # Step 2: Extract data for target CVE
    print(f"\n[STEP 2] Extracting data for {cve_id}...")
    
    if cve_id not in ds.cve.values:
        print(f"❌ {cve_id} not found in dataset")
        available_cves = [str(c) for c in ds.cve.values if "2024-3094" in str(c)]
        if available_cves:
            cve_id = available_cves[0]
            print(f"Using similar CVE: {cve_id}")
        else:
            print("Available CVEs (first 10):", list(ds.cve.values[:10]))
            return
    
    cve_idx = list(ds.cve.values).index(cve_id)
    cve_data = ds.isel(cve=cve_idx)
    
    print(f"✓ Found {cve_id} at index {cve_idx}")
    
    # Step 3: Analyze masks and data structure
    print(f"\n[STEP 3] Analyzing data structure...")
    
    pred_data = cve_data.pred.values  # [time, horizon]
    true_data = cve_data.true.values  # [time, horizon]
    eval_mask = cve_data.eval_mask.values.astype(bool)  # [time]
    mask_h = cve_data.mask_h.values.astype(bool)  # [time, horizon]
    
    print(f"Prediction shape: {pred_data.shape}")
    print(f"True shape: {true_data.shape}")
    print(f"Eval mask shape: {eval_mask.shape}")
    print(f"Horizon mask shape: {mask_h.shape}")
    print(f"Evaluation timestamps: {np.sum(eval_mask)}")
    
    # Step 4: Validate stupid model behavior
    print(f"\n[STEP 4] Validating stupid model behavior...")
    
    eval_times = np.where(eval_mask)[0]
    if len(eval_times) > 0:
        # Check first few evaluation timestamps
        for i, t_idx in enumerate(eval_times[:3]):
            pred_row = pred_data[t_idx, :]
            true_row = true_data[t_idx, :]
            mask_row = mask_h[t_idx, :]
            
            valid_preds = pred_row[mask_row]
            valid_trues = true_row[mask_row]
            
            # For stupid model, all predictions should be identical to true[horizon=0]
            expected_value = true_row[0]  # Current day value
            
            print(f"  Timestamp {t_idx} (eval time {i+1}):")
            print(f"    Expected value (true[h=0]): {expected_value:.6f}")
            print(f"    Predictions range: {valid_preds.min():.6f} to {valid_preds.max():.6f}")
            print(f"    All predictions identical? {np.allclose(valid_preds, expected_value, atol=1e-6)}")
            
            if not np.allclose(valid_preds, expected_value, atol=1e-6):
                print(f"    ⚠️  PROBLEM: Predictions not identical to current value!")
                print(f"    First 5 predictions: {valid_preds[:5]}")
                print(f"    Expected (repeated): {[expected_value] * min(5, len(valid_preds))}")
    
    # Step 5: Independent metric computation
    print(f"\n[STEP 5] Computing metrics independently...")
    
    horizons = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    results = []
    
    for h in horizons:
        idx = h - 1  # Convert to 0-based
        
        # Extract data for this horizon
        y_true_log = true_data[:, idx]
        y_pred_log = pred_data[:, idx]
        
        # Apply masks exactly as in evaluation
        mask_h_col = mask_h[:, idx]
        mask_eval_col = eval_mask
        valid = mask_h_col & mask_eval_col & ~np.isnan(y_true_log) & ~np.isnan(y_pred_log)
        
        valid_count = np.sum(valid)
        
        if valid_count > 0:
            # Convert to probability space
            y_t = _log_to_prob(y_true_log[valid])
            y_p = _log_to_prob(y_pred_log[valid])
            
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            mean_true = np.mean(y_t)
            mean_pred = np.mean(y_p)
            
            # For perfect persistence, MAE should be 0
            print(f"  Horizon {h:2d}: {valid_count:3d} points, MAE={mae:.6f}, True={mean_true:.6f}, Pred={mean_pred:.6f}")
            
            results.append({
                'horizon': h,
                'valid_count': valid_count,
                'mae': mae,
                'rmse': rmse,
                'mean_true': mean_true,
                'mean_pred': mean_pred,
                'raw_true_sample': y_true_log[valid][:3] if valid_count >= 3 else [],
                'raw_pred_sample': y_pred_log[valid][:3] if valid_count >= 3 else []
            })
        else:
            print(f"  Horizon {h:2d}: No valid points")
    
    # Step 6: Analyze the pattern
    print(f"\n[STEP 6] Analyzing the evaluation pattern...")
    
    if len(results) > 1:
        df_results = pd.DataFrame(results)
        
        # Check for the "mirrored" pattern
        true_means = df_results['mean_true'].values
        pred_means = df_results['mean_pred'].values
        
        # Compute trends
        true_trend = np.polyfit(range(len(true_means)), true_means, 1)[0]
        pred_trend = np.polyfit(range(len(pred_means)), pred_means, 1)[0]
        
        print(f"  True values trend slope: {true_trend:.6f}")
        print(f"  Predicted values trend slope: {pred_trend:.6f}")
        print(f"  Trends opposite? {(true_trend > 0 and pred_trend < 0) or (true_trend < 0 and pred_trend > 0)}")
        
        # Check if this is due to evaluation set changes
        valid_counts = df_results['valid_count'].values
        print(f"  Valid counts: {valid_counts}")
        print(f"  Valid counts decreasing? {all(valid_counts[i] >= valid_counts[i+1] for i in range(len(valid_counts)-1))}")
    
    # Step 7: Deep dive into evaluation sets
    print(f"\n[STEP 7] Deep dive into evaluation sets per horizon...")
    
    for h in [1, 15, 30]:
        if h <= len(horizons) and h in [r['horizon'] for r in results]:
            idx = h - 1
            mask_h_col = mask_h[:, idx]
            valid_indices = np.where(eval_mask & mask_h_col)[0]
            
            print(f"  Horizon {h}:")
            print(f"    Valid time indices: {valid_indices[:10]}{'...' if len(valid_indices) > 10 else ''}")
            
            if len(valid_indices) > 0:
                # Get actual dates if available
                if hasattr(cve_data, 'time') and cve_data.time.values is not None:
                    try:
                        time_coords = pd.to_datetime(cve_data.time.values)
                        valid_dates = time_coords[valid_indices]
                        valid_dates_clean = valid_dates[~pd.isna(valid_dates)]
                        
                        if len(valid_dates_clean) > 0:
                            print(f"    Date range: {valid_dates_clean.min()} to {valid_dates_clean.max()}")
                    except:
                        print(f"    Could not parse dates")
                
                # Check temporal bias
                early_indices = valid_indices[:len(valid_indices)//2] if len(valid_indices) > 1 else valid_indices
                late_indices = valid_indices[len(valid_indices)//2:] if len(valid_indices) > 1 else []
                
                if len(early_indices) > 0 and len(late_indices) > 0:
                    early_true = _log_to_prob(true_data[early_indices, idx])
                    late_true = _log_to_prob(true_data[late_indices, idx])
                    
                    early_pred = _log_to_prob(pred_data[early_indices, idx])
                    late_pred = _log_to_prob(pred_data[late_indices, idx])
                    
                    print(f"    Early period true mean: {np.mean(early_true):.6f}")
                    print(f"    Late period true mean: {np.mean(late_true):.6f}")
                    print(f"    Early period pred mean: {np.mean(early_pred):.6f}")
                    print(f"    Late period pred mean: {np.mean(late_pred):.6f}")
    
    # Step 8: Generate diagnostic plot
    print(f"\n[STEP 8] Creating diagnostic plot...")
    
    if len(results) > 1:
        df_results = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Diagnostic Analysis: {cve_id}')
        
        # Plot 1: MAE vs Horizon
        axes[0,0].plot(df_results['horizon'], df_results['mae'], 'o-')
        axes[0,0].set_title('MAE vs Horizon')
        axes[0,0].set_xlabel('Horizon')
        axes[0,0].set_ylabel('MAE')
        axes[0,0].grid(True)
        
        # Plot 2: Mean values vs Horizon  
        axes[0,1].plot(df_results['horizon'], df_results['mean_true'], 'o-', label='True')
        axes[0,1].plot(df_results['horizon'], df_results['mean_pred'], 'x-', label='Predicted')
        axes[0,1].set_title('Mean Values vs Horizon')
        axes[0,1].set_xlabel('Horizon')
        axes[0,1].set_ylabel('Mean Probability')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Plot 3: Valid counts vs Horizon
        axes[1,0].plot(df_results['horizon'], df_results['valid_count'], 'o-')
        axes[1,0].set_title('Valid Points vs Horizon')
        axes[1,0].set_xlabel('Horizon')
        axes[1,0].set_ylabel('Valid Count')
        axes[1,0].grid(True)
        
        # Plot 4: Difference vs Horizon
        diff = df_results['mean_true'] - df_results['mean_pred']
        axes[1,1].plot(df_results['horizon'], diff, 'o-')
        axes[1,1].set_title('True - Predicted vs Horizon')
        axes[1,1].set_xlabel('Horizon')
        axes[1,1].set_ylabel('Difference')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('diagnostic_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Saved diagnostic plot: diagnostic_analysis.png")
    
    return results

def main():
    results = analyze_cve_rigorously()
    return results

if __name__ == "__main__":
    main() 