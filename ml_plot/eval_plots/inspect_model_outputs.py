#!/usr/bin/env python3
"""
Rigorous Analysis of Model Output Semantics for True Positive Definition

This script analyzes the exact structure and meaning of LSTM model outputs
to understand how predictions relate to ground truth events and define
correct True Positive conditions for early warning evaluation.

Key Questions:
1. What does pred[c, t, h] represent exactly?
2. How do horizons relate to calendar dates?
3. When should we count a prediction as a True Positive?
4. What is the temporal logic of forecasting?

Author: Research Team
Date: 2025
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def analyze_model_semantics(ncfile: Path, output_dir: Path):
    """
    Rigorous analysis of model output semantics and temporal structure.
    Memory-efficient version that processes data in chunks.
    """
    print("🔍 RIGOROUS MODEL SEMANTICS ANALYSIS")
    print("=" * 60)
    
    # Load the NetCDF file
    print(f"Loading NetCDF: {ncfile}")
    ds = xr.open_dataset(ncfile)
    
    print(f"\nDataset structure:")
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Variables: {list(ds.data_vars.keys())}")
    print(f"  Coordinates: {list(ds.coords.keys())}")
    
    # Get dimensions without loading full arrays
    C, T, H = ds.sizes['cve'], ds.sizes['time'], ds.sizes['horizon']
    cve_ids = ds.cve.values    # [C] - CVE identifiers (small array)
    horizons = ds.horizon.values  # [H] - horizon indices (small array)
    
    print(f"\nArray shapes:")
    print(f"  Predictions: ({C}, {T}, {H})")
    print(f"  Ground truth: ({C}, {T}, {H})")
    print(f"  Total memory per array: {C * T * H * 4 / 1024**3:.2f} GB")
    
    # Sample a specific CVE for detailed analysis - load only what we need
    sample_cve_idx = 0
    sample_cve = cve_ids[sample_cve_idx]
    print(f"\n📊 DETAILED ANALYSIS: {sample_cve}")
    print("=" * 40)
    
    # Load data for just this CVE
    pred_sample = ds.pred[sample_cve_idx].values      # [T, H]
    true_sample = ds.true[sample_cve_idx].values      # [T, H]
    mask_h_sample = ds.mask_h[sample_cve_idx].values  # [T, H]
    eval_mask_sample = ds.eval_mask[sample_cve_idx].values  # [T]
    time_coords_sample = ds.time[sample_cve_idx].values  # [T]
    
    # Get valid time points for this CVE
    valid_times = eval_mask_sample > 0
    valid_time_indices = np.where(valid_times)[0]
    
    print(f"Valid evaluation times: {valid_times.sum()}/{T}")
    print(f"First 10 valid time indices: {valid_time_indices[:10]}")
    
    # Analyze temporal structure
    sample_dates = pd.to_datetime(time_coords_sample[valid_time_indices[:10]])
    print(f"\nFirst 10 calendar dates:")
    for i, (t_idx, date) in enumerate(zip(valid_time_indices[:10], sample_dates)):
        print(f"  t={t_idx:3d}: {date.strftime('%Y-%m-%d')}")
    
    # Analyze prediction semantics for a specific anchor time
    anchor_t = valid_time_indices[5] if len(valid_time_indices) > 5 else valid_time_indices[0]
    anchor_date = pd.to_datetime(time_coords_sample[anchor_t])
    
    print(f"\n🎯 PREDICTION SEMANTICS AT ANCHOR TIME t={anchor_t}")
    print(f"Anchor date: {anchor_date.strftime('%Y-%m-%d')}")
    print("-" * 50)
    
    # Show what each horizon predicts
    print("Horizon semantics (what each h predicts):")
    for h in range(min(10, H)):  # Show first 10 horizons
        if mask_h_sample[anchor_t, h]:
            pred_val = pred_sample[anchor_t, h]
            true_val = true_sample[anchor_t, h]
            target_date = anchor_date + pd.Timedelta(days=h+1)
            
            print(f"  h={h:2d}: predicts {target_date.strftime('%Y-%m-%d')} "
                  f"(t+{h+1:2d}) | pred={pred_val:.4f}, true={true_val:.4f}")
    
    # Analyze target building logic
    print(f"\n🔧 TARGET BUILDING VERIFICATION")
    print("-" * 40)
    
    # Check if true[c, t, h] = epss[c, t+h+1] 
    print("Verifying target semantics: true[c,t,h] should equal epss at date t+h+1")
    
    # For this, we need to understand the relationship between true values and time
    # Let's check consistency across multiple time points
    inconsistencies = 0
    total_checks = 0
    
    for t in valid_time_indices[:5]:  # Check first 5 valid times
        for h in range(min(5, H)):  # Check first 5 horizons
            if mask_h_sample[t, h]:
                # Check if there's a corresponding time point at t+h+1
                target_t = t + h + 1
                if target_t < T and eval_mask_sample[target_t]:
                    # Compare true[c,t,h] with true[c,target_t,0] (horizon 0 is "current")
                    predicted_value = true_sample[t, h]
                    actual_value = true_sample[target_t, 0] if mask_h_sample[target_t, 0] else np.nan
                    
                    if not np.isnan(actual_value):
                        diff = abs(predicted_value - actual_value)
                        if diff > 1e-6:  # Allow for floating point precision
                            inconsistencies += 1
                            print(f"    INCONSISTENCY: t={t}, h={h}, target_t={target_t}")
                            print(f"      true[{t},{h}]={predicted_value:.6f} != true[{target_t},0]={actual_value:.6f}")
                        total_checks += 1
    
    print(f"Target consistency check: {inconsistencies}/{total_checks} inconsistencies")
    
    # Analyze the evaluation logic for early warnings
    print(f"\n⚠️  EARLY WARNING EVALUATION LOGIC")
    print("-" * 45)
    
    # Convert to probability space for threshold analysis - only for sample CVE
    prob_pred_sample = np.clip(np.exp(pred_sample) - 1e-6, 0.0, 1.0)
    prob_true_sample = np.clip(np.exp(true_sample) - 1e-6, 0.0, 1.0)
    threshold = 0.7
    
    print(f"Using threshold: {threshold}")
    print(f"Sample predictions in probability space:")
    
    for t in valid_time_indices[:3]:
        for h in range(min(5, H)):
            if mask_h_sample[t, h]:
                pred_prob = prob_pred_sample[t, h]
                true_prob = prob_true_sample[t, h]
                target_date = pd.to_datetime(time_coords_sample[t]) + pd.Timedelta(days=h+1)
                
                print(f"  t={t:2d}, h={h:2d} → {target_date.strftime('%m-%d')}: "
                      f"pred={pred_prob:.3f}, true={true_prob:.3f}, "
                      f"pred≥{threshold}: {pred_prob >= threshold}, "
                      f"true≥{threshold}: {true_prob >= threshold}")
    
    # Define True Positive conditions
    print(f"\n✅ TRUE POSITIVE DEFINITION ANALYSIS")
    print("-" * 45)
    
    print("CRITICAL INSIGHT: For a valid True Positive, we need:")
    print("1. An event occurs at some target date T_event")
    print("2. A model prediction made at anchor time T_anchor predicts this event")
    print("3. The prediction horizon must exactly cover T_event")
    print("4. T_anchor < T_event (prediction must be made before the event)")
    print()
    
    print("Mathematical condition:")
    print("  If event at T_event, then for TP:")
    print("  ∃ T_anchor, h such that:")
    print("    - T_anchor + h + 1 = T_event  (horizon covers event date)")
    print("    - pred[c, T_anchor, h] ≥ threshold")
    print("    - eval_mask[c, T_anchor] = 1  (valid prediction time)")
    print("    - mask_h[c, T_anchor, h] = 1  (valid horizon)")
    print()
    
    print("This means:")
    print("  - T_anchor = T_event - h - 1")
    print("  - For h=0: T_anchor = T_event - 1 (next-day prediction)")
    print("  - For h=29: T_anchor = T_event - 30 (30-day-ahead prediction)")
    
    # Demonstrate with a concrete example
    print(f"\n🎯 CONCRETE EXAMPLE")
    print("-" * 25)
    
    # Find an actual event in the data
    events_found = []
    for t in valid_time_indices:
        for h in [0]:  # Check horizon 0 (next day) for events
            if mask_h_sample[t, h]:
                true_prob = prob_true_sample[t, h]
                if true_prob >= threshold:
                    event_date = pd.to_datetime(time_coords_sample[t]) + pd.Timedelta(days=h+1)
                    events_found.append((t, h, event_date, true_prob))
    
    if events_found:
        t_event_anchor, h_event, event_date, event_prob = events_found[0]
        print(f"Found event: true EPSS ≥ {threshold} on {event_date.strftime('%Y-%m-%d')}")
        print(f"  Event probability: {event_prob:.3f}")
        print(f"  Event detected at t={t_event_anchor}, h={h_event}")
        print()
        
        # Check all possible predictions that could warn about this event
        actual_event_t = t_event_anchor + h_event + 1  # Actual time index of event
        print(f"Checking predictions that could warn about this event:")
        print(f"Event occurs at time index {actual_event_t}")
        
        warnings_found = []
        for h in range(H):
            required_anchor = actual_event_t - h - 1
            if (required_anchor >= 0 and required_anchor < T and 
                eval_mask_sample[required_anchor] and
                mask_h_sample[required_anchor, h]):
                
                pred_prob = prob_pred_sample[required_anchor, h]
                anchor_date = pd.to_datetime(time_coords_sample[required_anchor])
                
                is_warning = pred_prob >= threshold
                lead_time = h + 1
                
                print(f"  h={h:2d}: anchor t={required_anchor:3d} ({anchor_date.strftime('%m-%d')}) "
                      f"pred={pred_prob:.3f} ≥{threshold}: {is_warning} (lead: {lead_time} days)")
                
                if is_warning:
                    warnings_found.append((required_anchor, h, lead_time, pred_prob))
        
        print(f"\nWarnings found for this event: {len(warnings_found)}")
        if warnings_found:
            earliest_warning = min(warnings_found, key=lambda x: x[0])  # Earliest by anchor time
            anchor_t, h, lead_time, pred_prob = earliest_warning
            print(f"Earliest warning: {lead_time} days ahead (anchor t={anchor_t}, h={h}, prob={pred_prob:.3f})")
    else:
        print("No events found in sample data")
    
    # Additional analysis: Check prediction patterns
    print(f"\n📈 PREDICTION PATTERN ANALYSIS")
    print("-" * 35)
    
    # Analyze the "stupid model" behavior
    print("Analyzing prediction consistency (stupid model should repeat current value):")
    
    # Check if predictions are constant across horizons (stupid model characteristic)
    for t in valid_time_indices[:3]:
        if eval_mask_sample[t]:
            pred_values = []
            for h in range(min(10, H)):
                if mask_h_sample[t, h]:
                    pred_values.append(pred_sample[t, h])
            
            if pred_values:
                is_constant = all(abs(v - pred_values[0]) < 1e-6 for v in pred_values)
                print(f"  t={t}: predictions constant across horizons: {is_constant}")
                print(f"    Values: {pred_values[:5]}")  # Show first 5
    
    # Save analysis results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a summary report
    with open(output_dir / "model_semantics_analysis.txt", "w", encoding='utf-8') as f:
        f.write("MODEL SEMANTICS ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Dataset: {ncfile}\n")
        f.write(f"Dimensions: C={C}, T={T}, H={H}\n")
        f.write(f"Sample CVE: {sample_cve}\n")
        f.write(f"Valid evaluation times: {valid_times.sum()}/{T}\n")
        f.write(f"Target consistency check: {inconsistencies}/{total_checks} inconsistencies\n\n")
        
        f.write("TRUE POSITIVE DEFINITION:\n")
        f.write("For an event at calendar date D_event:\n")
        f.write("  TP if EXISTS anchor time t, horizon h such that:\n")
        f.write("    1. t + h + 1 corresponds to D_event\n")
        f.write("    2. pred[c, t, h] >= threshold\n")
        f.write("    3. eval_mask[c, t] = 1\n")
        f.write("    4. mask_h[c, t, h] = 1\n")
        f.write("    5. t < t_event (prediction before event)\n\n")
        
        f.write("LEAD TIME CALCULATION:\n")
        f.write("  Lead time = h + 1 days\n")
        f.write("  (horizon h=0 means 1-day lead time)\n\n")
        
        f.write("KEY INSIGHT:\n")
        f.write("The current evaluation script incorrectly aggregates at CVE level.\n")
        f.write("Correct evaluation should be event-specific:\n")
        f.write("  - Detect events at specific dates\n")
        f.write("  - Check if any prediction horizon covered that exact date\n")
        f.write("  - Count TP only when prediction horizon matches event date\n")
    
    print(f"\n📄 Analysis saved to: {output_dir / 'model_semantics_analysis.txt'}")
    
    ds.close()
    return {
        'dimensions': (C, T, H),
        'sample_cve': sample_cve,
        'valid_times': valid_times.sum(),
        'inconsistencies': inconsistencies,
        'total_checks': total_checks,
        'events_found': len(events_found) if 'events_found' in locals() else 0
    }


def main():
    """Main analysis function."""
    # Default paths
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "ml_plot/eval_plots/models"
    output_dir = project_root / "ml_plot/eval_plots/analysis"
    
    # Analyze different model outputs
    model_files = [
        models_dir / "stupid_predictions.nc",
        models_dir / "predictions_stream_sus_lstm.nc"
    ]
    
    for model_file in model_files:
        if model_file.exists():
            print(f"\n{'='*80}")
            print(f"ANALYZING: {model_file.name}")
            print(f"{'='*80}")
            
            results = analyze_model_semantics(model_file, output_dir / model_file.stem)
            
            print(f"\nSUMMARY for {model_file.name}:")
            print(f"  Dimensions: {results['dimensions']}")
            print(f"  Sample CVE: {results['sample_cve']}")
            print(f"  Valid evaluation times: {results['valid_times']}")
            print(f"  Target consistency: {results['total_checks'] - results['inconsistencies']}/{results['total_checks']}")
        else:
            print(f"⚠️  Model file not found: {model_file}")


if __name__ == "__main__":
    main() 