#!/usr/bin/env python
"""
analyze_grid.py - Find the best SUS configuration from grid search results

Usage: python analyze_grid.py
"""

import pandas as pd
import glob
from pathlib import Path

def analyze_grid_results():
    """Analyze grid search CSV files and find the best configuration."""
    
    results_dir = Path("results")
    if not results_dir.exists():
        print("❌ No results directory found. Run sweep_sus.ps1 first.")
        return
    
    # Find all grid CSV files
    csv_files = list(results_dir.glob("grid_*.csv"))
    if not csv_files:
        print("❌ No grid CSV files found. Run sweep_sus.ps1 first.")
        return
    
    print(f"📊 Found {len(csv_files)} grid search results")
    
    # Collect all results
    all_results = []
    for csv_file in csv_files:
        try:
            # Read CSV (no header, so add column names)
            df = pd.read_csv(csv_file, header=None, 
                           names=['beta', 'look_ahead', 'z_mode', 'rmse', 'spike_recall'])
            all_results.append(df)
        except Exception as e:
            print(f"⚠️  Skipping {csv_file}: {e}")
    
    if not all_results:
        print("❌ No valid CSV files found.")
        return
    
    # Combine all results
    results = pd.concat(all_results, ignore_index=True)
    
    # Remove duplicates (in case of re-runs)
    results = results.drop_duplicates(['beta', 'look_ahead', 'z_mode'])
    
    print(f"\n📋 Combined Results ({len(results)} configurations):")
    print("=" * 60)
    print(results.to_string(index=False, float_format='%.4f'))
    
    # Compute selection score: max(rmse, 1-recall)
    # Lower is better (penalizes both high RMSE and low recall)
    results['score'] = results[['rmse', 1 - results['spike_recall']]].max(axis=1)
    
    # Sort by score (best first)
    results = results.sort_values('score')
    
    print(f"\n🏆 Top 5 Configurations (by score = max(rmse, 1-recall)):")
    print("=" * 60)
    top5 = results.head(5)
    print(top5[['beta', 'look_ahead', 'z_mode', 'rmse', 'spike_recall', 'score']].to_string(index=False, float_format='%.4f'))
    
    # Best configuration
    best = results.iloc[0]
    print(f"\n🎯 WINNER:")
    print(f"   β = {best['beta']}")
    print(f"   Δ = {best['look_ahead']}")
    print(f"   Z = {best['z_mode']}")
    print(f"   RMSE = {best['rmse']:.4f}")
    print(f"   Spike Recall = {best['spike_recall']:.3f}")
    print(f"   Score = {best['score']:.4f}")
    
    # Generate config update
    print(f"\n📝 Update your SUS_CONFIG:")
    print("=" * 40)
    print("SUS_CONFIG = {")
    print("    'enabled': True,")
    print(f"    'look_ahead': {int(best['look_ahead'])},")
    print(f"    'beta': {best['beta']},")
    print("    'z_norm': None,   # auto-loaded from JSON")
    print("    'seed': 42,")
    print("    'quick_grid': False")
    print("}")
    
    # Show corresponding JSON file
    if best['z_mode'] == 'quant':
        json_file = f"ml_pipeline/work/sus_config_beta{best['beta']}_d{int(best['look_ahead'])}_q0.995.json"
    else:
        json_file = f"ml_pipeline/work/sus_config_beta{best['beta']}_d{int(best['look_ahead'])}_q1.000.json"
    
    print(f"\n📄 Keep this JSON file: {json_file}")
    
    return best

if __name__ == "__main__":
    analyze_grid_results() 