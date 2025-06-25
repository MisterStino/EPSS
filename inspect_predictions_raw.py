#!/usr/bin/env python3
"""
Inspect raw prediction values from NetCDF file for plotted CVEs.
This will help determine if the issue is in the predictions themselves or in the plotting code.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

def _log_to_prob(arr: np.ndarray) -> np.ndarray:
    """
    Convert from log space back to probability space.
    Same transform as in plotting code.
    """
    transformed = np.exp(arr) - 1e-6
    return np.clip(transformed, 0.0, 1.0)

def inspect_predictions():
    # Load predictions
    preds_path = Path("ml_pipeline/results/predictions/predictions_stream.nc")
    if not preds_path.exists():
        print(f"❌ Predictions file not found: {preds_path}")
        return
    
    ds = xr.open_dataset(preds_path)
    print(f"✅ Loaded predictions from: {preds_path}")
    print(f"   Dimensions: {dict(ds.dims)}")
    print(f"   Variables: {list(ds.data_vars)}")
    
    # Get plotted CVEs
    plotted_cves = ["CVE-1999-0042", "CVE-1999-0154", "CVE-1999-0113", "CVE-1999-0103", "CVE-1999-0070"]
    
    print("\n" + "="*80)
    print("INSPECTING RAW PREDICTION VALUES")
    print("="*80)
    
    for cve in plotted_cves:
        if cve not in ds.cve.values:
            print(f"❌ {cve} not found in predictions")
            continue
            
        print(f"\n🔍 {cve}")
        print("-" * 50)
        
        # Extract data for this CVE
        pred_cve = ds.sel(cve=cve)
        pred_log = pred_cve.pred.values  # Shape: [L, H] - log space
        true_log = pred_cve.true.values  # Shape: [L, H] - log space  
        eval_mask = pred_cve.eval_mask.values.astype(bool)  # [L]
        mask_h = pred_cve.mask_h.values.astype(bool)  # [L, H]
        
        print(f"   Shape of predictions: {pred_log.shape}")
        print(f"   Number of eval timestamps: {eval_mask.sum()}")
        
        # Transform to probability space
        pred_prob = _log_to_prob(pred_log)
        true_prob = _log_to_prob(true_log)
        
        # Look at eval timestamps only
        eval_indices = np.where(eval_mask)[0]
        
        print(f"\n   RAW PREDICTION VALUES (first 5 eval timestamps):")
        for i, idx in enumerate(eval_indices[:5]):
            valid_horizons = mask_h[idx]
            pred_log_row = pred_log[idx][valid_horizons]
            pred_prob_row = pred_prob[idx][valid_horizons]
            true_prob_row = true_prob[idx][valid_horizons]
            
            print(f"     Timestamp {idx}:")
            print(f"       Log space pred:  min={pred_log_row.min():.6f}, max={pred_log_row.max():.6f}, mean={pred_log_row.mean():.6f}")
            print(f"       Prob space pred: min={pred_prob_row.min():.6f}, max={pred_prob_row.max():.6f}, mean={pred_prob_row.mean():.6f}")
            print(f"       True prob:       min={true_prob_row.min():.6f}, max={true_prob_row.max():.6f}, mean={true_prob_row.mean():.6f}")
            
            # Show some actual values
            print(f"       First 5 pred (log):  {pred_log_row[:5]}")
            print(f"       First 5 pred (prob): {pred_prob_row[:5]}")
            print(f"       First 5 true (prob): {true_prob_row[:5]}")
            print()
        
        # Overall statistics for this CVE
        all_eval_preds_log = pred_log[eval_mask][mask_h[eval_mask]]
        all_eval_preds_prob = _log_to_prob(all_eval_preds_log)
        all_eval_true_prob = _log_to_prob(true_log[eval_mask][mask_h[eval_mask]])
        
        print(f"   OVERALL STATISTICS FOR {cve}:")
        print(f"     Total valid predictions: {len(all_eval_preds_prob)}")
        print(f"     Predictions (log):  min={all_eval_preds_log.min():.6f}, max={all_eval_preds_log.max():.6f}, mean={all_eval_preds_log.mean():.6f}")
        print(f"     Predictions (prob): min={all_eval_preds_prob.min():.6f}, max={all_eval_preds_prob.max():.6f}, mean={all_eval_preds_prob.mean():.6f}")
        print(f"     True values (prob): min={all_eval_true_prob.min():.6f}, max={all_eval_true_prob.max():.6f}, mean={all_eval_true_prob.mean():.6f}")
        
        # Check for patterns
        near_zero_count = (all_eval_preds_prob < 0.001).sum()
        print(f"     Predictions < 0.001: {near_zero_count}/{len(all_eval_preds_prob)} ({100*near_zero_count/len(all_eval_preds_prob):.1f}%)")
        
        very_low_count = (all_eval_preds_prob < 0.01).sum()
        print(f"     Predictions < 0.01:  {very_low_count}/{len(all_eval_preds_prob)} ({100*very_low_count/len(all_eval_preds_prob):.1f}%)")

if __name__ == "__main__":
    inspect_predictions() 