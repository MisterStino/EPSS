#!/usr/bin/env python3
"""
CORRECTED EPSS Predictions Evaluator and Plotter
===============================================

This script provides CORRECT evaluation of EPSS predictions by:
1. Only evaluating on the test window (eval_mask = 1)
2. Only using valid horizon predictions (mask_h = 1) 
3. Properly converting from inverted log space to probabilities
4. Avoiding data leakage by excluding training/validation data

Key corrections from original:
- Uses combined mask (eval_mask & mask_h) for test window only
- Converts values from inverted log space using proper transformation
- Provides horizon-wise and overall metrics for research
"""

from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import warnings, random, sys, os

# Matplotlib in head-less mode
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# Spark for historical EPSS data
from pyspark.sql import SparkSession

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def safe_mape(y_true, y_pred, eps=1e-2):
    """Safe MAPE calculation avoiding division by zero."""
    mask = np.abs(y_true) > eps
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def inv_invlog(x: np.ndarray, eps: float = 1e-6):
    """Convert from inverted log space back to probability space."""
    return np.clip(np.exp(-x) - eps, 0.0, 1.0)

class CorrectedCVEPredEvaluator:
    """
    CORRECTED evaluator that properly handles test windows and value transformations.
    
    Key features:
    - Only evaluates test window data (eval_mask = 1)
    - Applies horizon validity mask (mask_h = 1)
    - Converts from inverted log space to probabilities
    - Provides comprehensive metrics for research
    """
    
    def __init__(
        self,
        nc_path: Path,
        parquet: Path,
        out_root: Path,
        cve_list: list[str] | None = None,
    ):
        if not nc_path.exists():
            raise FileNotFoundError(nc_path)
        if not parquet.exists():
            raise FileNotFoundError(parquet)
            
        # Load data
        self.ds = xr.open_dataset(nc_path)
        self.spark = SparkSession.builder.getOrCreate()
        self.pq = parquet
        self.out_root = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        
        # Extract dimensions
        self.n_cves = self.ds.dims['cve']
        self.n_time = self.ds.dims['time']
        self.n_horizon = self.ds.dims['horizon']
        
        # Pre-compute masks and convert values
        self.eval_mask = self.ds.eval_mask.values  # (cve, time)
        self.mask_h = self.ds.mask_h.values        # (cve, time, horizon)
        self.test_mask = self.eval_mask[:, :, np.newaxis] * self.mask_h  # Combined mask
        
        # Convert from inverted log space to probabilities
        self.pred_prob = inv_invlog(self.ds.pred.values)
        self.true_prob = inv_invlog(self.ds.true.values)
        
        # CVE selection
        if cve_list is None:
            self.cves = self._auto_select_cves()
        else:
            self.cves = list(cve_list)
            
        # Summary
        test_predictions = self.test_mask.sum()
        print(f"🔬 Corrected EPSS Evaluator Initialized:")
        print(f"   → NetCDF: {nc_path}")
        print(f"   → Parquet: {parquet}")
        print(f"   → Output: {out_root.resolve()}")
        print(f"   → Dataset: {self.n_cves:,} CVEs × {self.n_time} time × {self.n_horizon} horizons")
        print(f"   → Test window predictions: {test_predictions:,}")
        print(f"   → Selected CVEs: {len(self.cves)}")
    
    def _auto_select_cves(self) -> list[str]:
        """Auto-select interesting CVEs based on test window variability."""
        print("🔍 Auto-selecting CVEs based on test window variability...")
        
        # Only analyze test window data
        test_true = self.true_prob * self.test_mask
        
        cves = self.ds.cve.values.astype(str)
        picks = []
        
        # Calculate variability for each CVE in test window only
        for i, cve in enumerate(cves[:100]):  # Limit to first 100 for speed
            cve_test_data = test_true[i][self.test_mask[i].astype(bool)]
            if len(cve_test_data) > 10:  # Need sufficient test data
                cve_range = cve_test_data.max() - cve_test_data.min()
                cve_mean = cve_test_data.mean()
                if cve_range > 0.1:  # Some variability
                    picks.append(cve)
                    if len(picks) >= 10:  # Limit selection
                        break
        
        if len(picks) < 5:  # Fallback
            picks = cves[:5].tolist()
            
        print(f"✨ Selected {len(picks)} CVEs: {', '.join(picks)}")
        return picks
    
    @lru_cache(maxsize=256)
    def _hist_df(self, cve: str) -> pd.DataFrame:
        """Fetch CVE's full EPSS history via Spark."""
        sdf = (
            self.spark.read.parquet(str(self.pq))
            .filter(f"cve = '{cve}'")
            .select("date", "epss")
        )
        pdf = sdf.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf.sort_values("date", inplace=True)
        return pdf
    
    def get_overall_test_metrics(self):
        """Calculate overall metrics for the entire test window."""
        print("📊 Calculating overall test window metrics...")
        
        # Extract test window data only
        pred_test = self.pred_prob[self.test_mask.astype(bool)]
        true_test = self.true_prob[self.test_mask.astype(bool)]
        
        # Remove invalid values
        valid_idx = np.isfinite(pred_test) & np.isfinite(true_test)
        pred_clean = pred_test[valid_idx]
        true_clean = true_test[valid_idx]
        
        if len(pred_clean) == 0:
            print("⚠️ No valid test predictions found!")
            return None
            
        # Calculate metrics
        metrics = {
            'n_predictions': len(pred_clean),
            'mae': mean_absolute_error(true_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(true_clean, pred_clean)),
            'r2': r2_score(true_clean, pred_clean),
            'mape': safe_mape(true_clean, pred_clean),
            'smape': 100 * np.mean(2 * np.abs(true_clean - pred_clean) / 
                                  (np.abs(pred_clean) + np.abs(true_clean) + 1e-8))
        }
        
        print(f"✅ Overall Test Window Metrics:")
        print(f"   Predictions evaluated: {metrics['n_predictions']:,}")
        print(f"   MAE:   {metrics['mae']:.6f}")
        print(f"   RMSE:  {metrics['rmse']:.6f}")
        print(f"   R²:    {metrics['r2']:.6f}")
        print(f"   MAPE:  {metrics['mape']:.2f}%")
        print(f"   SMAPE: {metrics['smape']:.2f}%")
        
        return metrics
    
    def get_horizon_test_metrics(self):
        """Calculate metrics for each forecast horizon (test window only)."""
        print("📈 Calculating horizon-wise test window metrics...")
        
        horizon_metrics = []
        
        for h in range(self.n_horizon):
            # Get test window data for this horizon
            h_mask = self.test_mask[:, :, h]  # (cve, time)
            pred_h = self.pred_prob[:, :, h][h_mask.astype(bool)]
            true_h = self.true_prob[:, :, h][h_mask.astype(bool)]
            
            # Remove invalid values
            valid_h = np.isfinite(pred_h) & np.isfinite(true_h)
            if valid_h.sum() == 0:
                continue
                
            pred_h_clean = pred_h[valid_h]
            true_h_clean = true_h[valid_h]
            
            # Calculate metrics
            h_metrics = {
                'horizon': h + 1,  # 1-indexed
                'n_predictions': len(pred_h_clean),
                'mae': mean_absolute_error(true_h_clean, pred_h_clean),
                'rmse': np.sqrt(mean_squared_error(true_h_clean, pred_h_clean)),
                'r2': r2_score(true_h_clean, pred_h_clean),
                'mape': safe_mape(true_h_clean, pred_h_clean),
                'smape': 100 * np.mean(2 * np.abs(true_h_clean - pred_h_clean) / 
                                      (np.abs(pred_h_clean) + np.abs(true_h_clean) + 1e-8))
            }
            
            horizon_metrics.append(h_metrics)
        
        print(f"✅ Horizon-wise metrics calculated for {len(horizon_metrics)} horizons")
        return horizon_metrics
    
    def plot_horizon_metrics(self, horizon_metrics: list, out_path: Path = None):
        """Plot horizon-wise metrics."""
        if not horizon_metrics:
            print("⚠️ No horizon metrics to plot")
            return
            
        horizons = [hm['horizon'] for hm in horizon_metrics]
        mae_list = [hm['mae'] for hm in horizon_metrics]
        rmse_list = [hm['rmse'] for hm in horizon_metrics]
        r2_list = [hm['r2'] for hm in horizon_metrics]
        mape_list = [hm['mape'] for hm in horizon_metrics]
        smape_list = [hm['smape'] for hm in horizon_metrics]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(horizons, mae_list, label="MAE", color="#f39c12", lw=2)
        ax.plot(horizons, rmse_list, label="RMSE", color="#2980b9", lw=2)
        ax.plot(horizons, smape_list, label="SMAPE", color="#27ae60", lw=2)
        ax.plot(horizons, r2_list, label="R²", color="#8e44ad", lw=2)
        
        ax.set_xlabel("Forecast Horizon (Days)", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title("Test Window Performance by Forecast Horizon", fontsize=14)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(frameon=False)
        ax.set_xticks(horizons[::5])  # Every 5th horizon
        ax.set_xlim([1, max(horizons)])
        
        fig.tight_layout()
        
        if out_path is None:
            out_path = self.out_root / "corrected_horizon_metrics.png"
        
        fig.savefig(out_path, dpi=150)
        print(f"📈 Horizon metrics plot saved: {out_path}")
        plt.close(fig)
    
    def get_cve_test_metrics(self, cve_id: str):
        """Calculate metrics for a specific CVE (test window only)."""
        try:
            cve_idx = list(self.ds.cve.values).index(cve_id)
        except ValueError:
            raise ValueError(f"CVE {cve_id} not found in dataset")
        
        # Get test window data for this CVE
        cve_test_mask = self.test_mask[cve_idx]  # (time, horizon)
        pred_cve = self.pred_prob[cve_idx][cve_test_mask.astype(bool)]
        true_cve = self.true_prob[cve_idx][cve_test_mask.astype(bool)]
        
        # Remove invalid values
        valid_idx = np.isfinite(pred_cve) & np.isfinite(true_cve)
        if valid_idx.sum() == 0:
            return None
            
        pred_clean = pred_cve[valid_idx]
        true_clean = true_cve[valid_idx]
        
        return {
            'cve_id': cve_id,
            'n_predictions': len(pred_clean),
            'mae': mean_absolute_error(true_clean, pred_clean),
            'rmse': np.sqrt(mean_squared_error(true_clean, pred_clean)),
            'r2': r2_score(true_clean, pred_clean),
            'mape': safe_mape(true_clean, pred_clean),
            'smape': 100 * np.mean(2 * np.abs(true_clean - pred_clean) / 
                                  (np.abs(pred_clean) + np.abs(true_clean) + 1e-8))
        }
    
    def print_summary_table(self):
        """Print a comprehensive summary table."""
        print("\n" + "=" * 80)
        print("CORRECTED TEST WINDOW EVALUATION SUMMARY")
        print("=" * 80)
        
        # Overall metrics
        overall = self.get_overall_test_metrics()
        if overall:
            print(f"\nOVERALL TEST WINDOW PERFORMANCE:")
            print(f"{'Metric':<10} {'Value':<12} {'Description'}")
            print("-" * 50)
            print(f"{'N':<10} {overall['n_predictions']:>11,} {'Test predictions evaluated'}")
            print(f"{'MAE':<10} {overall['mae']:>11.6f} {'Mean Absolute Error'}")
            print(f"{'RMSE':<10} {overall['rmse']:>11.6f} {'Root Mean Square Error'}")
            print(f"{'R²':<10} {overall['r2']:>11.6f} {'Coefficient of Determination'}")
            print(f"{'MAPE':<10} {overall['mape']:>10.2f}% {'Mean Absolute Percentage Error'}")
            print(f"{'SMAPE':<10} {overall['smape']:>10.2f}% {'Symmetric Mean Absolute Percentage Error'}")
        
        # Horizon metrics summary
        horizon_metrics = self.get_horizon_test_metrics()
        if horizon_metrics:
            print(f"\nHORIZON PERFORMANCE SUMMARY (First 10 days):")
            print(f"{'Day':<4} {'N_Pred':<8} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
            print("-" * 40)
            for hm in horizon_metrics[:10]:
                print(f"{hm['horizon']:<4} {hm['n_predictions']:<8,} {hm['mae']:<8.4f} {hm['rmse']:<8.4f} {hm['r2']:<8.4f}")
        
        # Sample CVE metrics
        print(f"\nSAMPLE CVE PERFORMANCE:")
        print(f"{'CVE':<16} {'N_Pred':<8} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
        print("-" * 50)
        for cve in self.cves[:5]:  # First 5 CVEs
            cve_metrics = self.get_cve_test_metrics(cve)
            if cve_metrics:
                print(f"{cve:<16} {cve_metrics['n_predictions']:<8,} {cve_metrics['mae']:<8.4f} {cve_metrics['rmse']:<8.4f} {cve_metrics['r2']:<8.4f}")

# Main execution
if __name__ == "__main__":
    NC = Path("ml_plot/perf/files/predictions_stream.nc")
    RAW = Path("data/epss/processed/epss_processed.parquet")
    OUT = Path("ml_plot/perf/corrected_results")
    
    print("🚀 Starting CORRECTED EPSS Predictions Evaluation")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = CorrectedCVEPredEvaluator(nc_path=NC, parquet=RAW, out_root=OUT)
    
    # Run comprehensive evaluation
    print("\n📊 Running comprehensive test window evaluation...")
    
    # Overall metrics
    overall_metrics = evaluator.get_overall_test_metrics()
    
    # Horizon metrics
    horizon_metrics = evaluator.get_horizon_test_metrics()
    evaluator.plot_horizon_metrics(horizon_metrics)
    
    # Summary table
    evaluator.print_summary_table()
    
    print(f"\n✅ CORRECTED evaluation complete!")
    print(f"📁 Results saved in: {OUT}")
    print(f"🔬 These metrics are suitable for research publication!") 