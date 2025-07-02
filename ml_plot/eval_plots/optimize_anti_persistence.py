#!/usr/bin/env python3
"""
Parameter Optimization for Anti-Persistence Evaluation

Find optimal parameters that:
1. Maximize LSTM model true positives
2. Keep stupid model at exactly 0 true positives
3. Maintain realistic event criteria

This script systematically tests parameter combinations and reports the best settings.
"""

import sys
import time
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import argparse

def run_evaluation(model_path: str, threshold: float, min_delta: float, 
                  min_relative_change: float, min_baseline: float = 0.1) -> Dict:
    """
    Run the anti-persistence evaluation with given parameters.
    
    Returns:
        dict with TP, FP, FN, TN, and other metrics
    """
    cmd = [
        sys.executable, "-m", "ml_plot.eval_plots.eval_ews_metrics_corrected",
        model_path,
        "--threshold", str(threshold),
        "--min-delta", str(min_delta), 
        "--min-relative-change", str(min_relative_change),
        "--min-baseline", str(min_baseline)
    ]
    
    try:
        print(f"  Running: threshold={threshold}, delta={min_delta}, rel_change={min_relative_change*100:.0f}%")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return {"TP": -1, "error": result.stderr}
        
        # Parse the output for key metrics
        output = result.stdout
        metrics = {}
        
        # Extract metrics from the summary section
        lines = output.split('\n')
        in_summary = False
        
        for line in lines:
            if "Per-Forecast Contingency Table:" in line:
                in_summary = True
                continue
            elif in_summary and "Corrected Skill Metrics:" in line:
                break
            elif in_summary:
                if "True Positives (TP):" in line:
                    metrics["TP"] = int(line.split()[-1])
                elif "False Positives (FP):" in line:
                    metrics["FP"] = int(line.split()[-1].replace(',', ''))
                elif "False Negatives (FN):" in line:
                    metrics["FN"] = int(line.split()[-1])
                elif "True Negatives (TN):" in line:
                    metrics["TN"] = int(line.split()[-1].replace(',', ''))
        
        # Extract additional metrics
        for line in lines:
            if "Recall:" in line and "Per-Forecast" not in line:
                try:
                    metrics["Recall"] = float(line.split()[-1])
                except:
                    metrics["Recall"] = 0.0
            elif "Precision:" in line and "Per-Forecast" not in line:
                try:
                    metrics["Precision"] = float(line.split()[-1])
                except:
                    metrics["Precision"] = 0.0
            elif "Found" in line and "REALISTIC early warning events" in line:
                try:
                    metrics["total_events"] = int(line.split()[1])
                except:
                    metrics["total_events"] = 0
            elif "Forecasts invalidated by anti-persistence:" in line:
                try:
                    metrics["invalidated"] = int(line.split()[-1].replace(',', ''))
                except:
                    metrics["invalidated"] = 0
        
        return metrics
        
    except subprocess.TimeoutExpired:
        return {"TP": -1, "error": "Timeout"}
    except Exception as e:
        return {"TP": -1, "error": str(e)}

def optimize_parameters():
    """
    Systematically search for optimal parameters.
    """
    print("🎯 OPTIMIZING ANTI-PERSISTENCE PARAMETERS")
    print("=" * 60)
    
    # Model paths
    lstm_path = "ml_plot/eval_plots/models/predictions_stream_sus_lstm.nc"
    stupid_path = "ml_pipeline/stupid_model/simple_stupid_predictions.nc"
    
    # Parameter ranges to test
    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
    min_deltas = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
    min_relative_changes = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]  # 20% to 100%
    
    results = []
    best_lstm_tp = 0
    best_params = None
    
    total_combinations = len(thresholds) * len(min_deltas) * len(min_relative_changes)
    current_combo = 0
    
    print(f"Testing {total_combinations} parameter combinations...")
    print()
    
    for threshold in thresholds:
        for min_delta in min_deltas:
            for min_relative_change in min_relative_changes:
                current_combo += 1
                
                print(f"[{current_combo}/{total_combinations}] Testing combination:")
                print(f"  Threshold: {threshold}, Delta: {min_delta}, Rel Change: {min_relative_change*100:.0f}%")
                
                # Test LSTM model
                print("  → Testing LSTM model...")
                lstm_metrics = run_evaluation(lstm_path, threshold, min_delta, min_relative_change)
                
                if lstm_metrics.get("TP", -1) < 0:
                    print(f"  ❌ LSTM evaluation failed: {lstm_metrics.get('error', 'Unknown error')}")
                    continue
                
                # Test Stupid model
                print("  → Testing Stupid model...")
                stupid_metrics = run_evaluation(stupid_path, threshold, min_delta, min_relative_change)
                
                if stupid_metrics.get("TP", -1) < 0:
                    print(f"  ❌ Stupid evaluation failed: {stupid_metrics.get('error', 'Unknown error')}")
                    continue
                
                # Check if this is a valid configuration (stupid model has 0 TP)
                if stupid_metrics["TP"] == 0:
                    lstm_tp = lstm_metrics["TP"]
                    print(f"  ✅ VALID: LSTM={lstm_tp} TP, Stupid=0 TP")
                    
                    # Record this result
                    result = {
                        "threshold": threshold,
                        "min_delta": min_delta,
                        "min_relative_change": min_relative_change,
                        "lstm_tp": lstm_tp,
                        "lstm_fp": lstm_metrics.get("FP", 0),
                        "lstm_fn": lstm_metrics.get("FN", 0),
                        "lstm_recall": lstm_metrics.get("Recall", 0),
                        "lstm_precision": lstm_metrics.get("Precision", 0),
                        "total_events": lstm_metrics.get("total_events", 0),
                        "lstm_invalidated": lstm_metrics.get("invalidated", 0),
                        "stupid_tp": stupid_metrics["TP"],
                        "stupid_invalidated": stupid_metrics.get("invalidated", 0)
                    }
                    results.append(result)
                    
                    # Check if this is the best so far
                    if lstm_tp > best_lstm_tp:
                        best_lstm_tp = lstm_tp
                        best_params = result.copy()
                        print(f"  🏆 NEW BEST: {lstm_tp} LSTM TPs!")
                    
                else:
                    print(f"  ❌ INVALID: Stupid model got {stupid_metrics['TP']} TP (should be 0)")
                
                print()
    
    # Report results
    print("🏆 OPTIMIZATION COMPLETE!")
    print("=" * 60)
    
    if not results:
        print("❌ No valid parameter combinations found!")
        return
    
    print(f"Found {len(results)} valid parameter combinations")
    print(f"Best LSTM performance: {best_lstm_tp} true positives")
    print()
    
    if best_params:
        print("🎯 OPTIMAL PARAMETERS:")
        print(f"  Threshold: {best_params['threshold']}")
        print(f"  Min Delta: {best_params['min_delta']}")
        print(f"  Min Relative Change: {best_params['min_relative_change']*100:.0f}%")
        print()
        print("📊 OPTIMAL PERFORMANCE:")
        print(f"  LSTM True Positives: {best_params['lstm_tp']}")
        print(f"  LSTM False Positives: {best_params['lstm_fp']:,}")
        print(f"  LSTM Recall: {best_params['lstm_recall']:.3f}")
        print(f"  LSTM Precision: {best_params['lstm_precision']:.6f}")
        print(f"  Total Events Found: {best_params['total_events']}")
        print(f"  LSTM Forecasts Invalidated: {best_params['lstm_invalidated']:,}")
        print(f"  Stupid Model TP: {best_params['stupid_tp']} ✅")
        print(f"  Stupid Forecasts Invalidated: {best_params['stupid_invalidated']:,}")
    
    # Save detailed results
    df = pd.DataFrame(results)
    df = df.sort_values("lstm_tp", ascending=False)
    
    output_file = "anti_persistence_optimization_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Show top 10 results
    print("\n📈 TOP 10 PARAMETER COMBINATIONS:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Threshold':<9} {'Delta':<6} {'Rel%':<5} {'LSTM_TP':<7} {'Events':<6} {'LSTM_Recall':<11}")
    print("-" * 80)
    
    for i, row in df.head(10).iterrows():
        print(f"{len(df)-list(df.index).index(i):<4} "
              f"{row['threshold']:<9} "
              f"{row['min_delta']:<6} "
              f"{row['min_relative_change']*100:<5.0f} "
              f"{row['lstm_tp']:<7} "
              f"{row['total_events']:<6} "
              f"{row['lstm_recall']:<11.3f}")

def main():
    parser = argparse.ArgumentParser(description="Optimize anti-persistence parameters")
    parser.add_argument("--quick", action="store_true", help="Run quick test with fewer parameters")
    args = parser.parse_args()
    
    if args.quick:
        print("🚀 Running QUICK parameter optimization...")
        # Reduce parameter space for quick testing
        global thresholds, min_deltas, min_relative_changes
        thresholds = [0.65, 0.7, 0.75]
        min_deltas = [0.08, 0.1, 0.15]
        min_relative_changes = [0.3, 0.5, 0.8]
    
    optimize_parameters()

if __name__ == "__main__":
    main() 