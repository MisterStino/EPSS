#!/usr/bin/env python3
"""Publication-Ready EPSS Plotter with intelligent CVE selection"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import random
from collections import defaultdict
import re
import os


def _find_project_root():
    """Find project root by looking for ml_pipeline directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "ml_pipeline").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (looking for ml_pipeline directory)")


def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)


class PublicationPlotter:
    """EPSS plotter with intelligent CVE selection and fixed time coordinates."""
    
    def __init__(self, preds_path=None, epss_path=None):
        # Find project root and build absolute paths
        project_root = _find_project_root()
        
        if preds_path is None:
            preds_path = project_root / "ml_pipeline/results/predictions/predictions_stream.nc"
        else:
            preds_path = Path(preds_path)
            if not preds_path.is_absolute():
                preds_path = project_root / preds_path
                
        if epss_path is None:
            epss_path = project_root / "data/epss/processed/epss_processed.parquet"
        else:
            epss_path = Path(epss_path)
            if not epss_path.is_absolute():
                epss_path = project_root / epss_path
        
        self.preds_path = preds_path
        self.epss_path = epss_path
        self.out_root = project_root / "ml_plot/plots_publication"
        self.out_root.mkdir(parents=True, exist_ok=True)
        
        print("[INFO] Loading NetCDF predictions...")
        self.ds = xr.open_dataset(self.preds_path)
        print("✅ Time coordinates preserved (no corruption fix)")
        
        print("[INFO] Loading EPSS data...")
        self.epss_df = pd.read_parquet(self.epss_path, columns=["cve", "date", "epss"])
        self.epss_df['date'] = pd.to_datetime(self.epss_df['date'])
        
        print("[INFO] Selecting interesting CVEs...")
        self.cves = self._select_interesting_cves()
        print(f"[INFO] Selected {len(self.cves)} CVEs for plotting")
    
    def _select_interesting_cves(self):
        """Select CVEs with interesting variability patterns."""
        print("🔍 Analyzing CVE patterns...")
        
        true_prob = _log_to_prob(self.ds.true.values.copy())
        eval_mask = self.ds.eval_mask.values.astype(bool)
        
        big_jump_cves = []
        all_cves = list(self.ds.cve.values)
        
        for i, cve in enumerate(all_cves):
            test_mask = eval_mask[i]
            if not test_mask.any():
                continue
                
            # Get anchor values (horizon=0)
            anchor_values = true_prob[i, test_mask, 0]
            valid_values = anchor_values[np.isfinite(anchor_values)]
            
            if len(valid_values) == 0:
                continue
                
            p_min, p_max = np.min(valid_values), np.max(valid_values)
            p_range = p_max - p_min
            
            # Look for big jumps (interesting variability)
            if p_min < 0.3 and p_max > 0.7:
                year_match = re.match(r'CVE-(\d{4})-\d+', cve)
                year = int(year_match.group(1)) if year_match else 9999
                big_jump_cves.append((cve, year, p_range))
                print(f"  🎯 Big jump: {cve} (range={p_range:.3f})")
        
        # Sort by range (most interesting first) and select diverse set
        big_jump_cves.sort(key=lambda x: x[2], reverse=True)
        
        if len(big_jump_cves) >= 10:
            selected = [cve[0] for cve in big_jump_cves[:10]]
        elif len(big_jump_cves) >= 5:
            selected = [cve[0] for cve in big_jump_cves]
        else:
            # Fallback: use first few CVEs but warn user
            print("⚠️ Limited interesting CVEs found, using first 5")
            selected = all_cves[:5]
        
        return selected
    
    def plot_all(self):
        """Generate all plots."""
        print(f"[PLOTTING] Generating plots for {len(self.cves)} CVEs...")
        
        for i, cve in enumerate(self.cves):
            print(f"[{i+1}/{len(self.cves)}] {cve}")
            try:
                self._plot_cve(cve)
            except Exception as e:
                print(f"  ❌ Error: {e}")
    
    def _plot_cve(self, cve):
        """Plot one CVE."""
        pred_cve = self.ds.sel(cve=cve)
        time_coord = pred_cve.time.values  # ✅ Original coordinates
        
        pred_prob = _log_to_prob(pred_cve.pred.values)
        true_prob = _log_to_prob(pred_cve.true.values)
        mask_h = pred_cve.mask_h.values.astype(bool)
        eval_mask = pred_cve.eval_mask.values.astype(bool)
        
        # Get EPSS data
        cve_epss = self.epss_df[self.epss_df.cve == cve].sort_values('date')
        if cve_epss.empty:
            print(f"   ⚠ No EPSS data for {cve}")
            return
        
        # Create output directory
        cve_out = self.out_root / cve
        cve_out.mkdir(parents=True, exist_ok=True)
        
        # Plot each test timestamp
        plot_count = 0
        for idx in np.where(eval_mask)[0]:
            anchor_date = time_coord[idx]
            
            if np.isnat(anchor_date):
                continue
                
            out_path = cve_out / f"{cve}-frame-{plot_count}.png"
            
            self._make_plot(cve, cve_epss, anchor_date, 
                          pred_prob[idx], true_prob[idx], mask_h[idx], out_path)
            plot_count += 1
        
        print(f"   ✅ Created {plot_count} plots")
    
    def _make_plot(self, cve, cve_epss, anchor_date, pred_row, true_row, mask_row, out_path):
        """Create single plot."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Time window
        anchor_pd = pd.to_datetime(anchor_date)
        context_start = anchor_pd - pd.Timedelta(days=60)
        horizon_end = anchor_pd + pd.Timedelta(days=len(pred_row))
        
        # Filter EPSS data
        window_epss = cve_epss[
            (cve_epss.date >= context_start) & (cve_epss.date <= horizon_end)
        ]
        
        # Plot EPSS truth
        if not window_epss.empty:
            ax.plot(window_epss.date, window_epss.epss, 
                   color="gold", label="EPSS (truth)", linewidth=2)
        
        # Model cutoff
        ax.axvline(x=anchor_pd, color='red', linestyle=':', 
                  label='Model cutoff', alpha=0.7)
        
        # Predictions
        horizon_dates = anchor_pd + pd.to_timedelta(range(1, len(pred_row)+1), unit="D")
        valid = mask_row.astype(bool)
        
        if valid.any():
            ax.plot(horizon_dates[valid], true_row[valid], 
                   color="green", label="True future", linewidth=2)
            ax.plot(horizon_dates[valid], pred_row[valid], 
                   color="blue", linestyle="--", label="LSTM prediction", linewidth=2)
        
        # Format
        ax.set_title(f"{cve} | {anchor_pd.date()}")
        ax.set_xlabel("Date")
        ax.set_ylabel("EPSS Probability")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    plotter = PublicationPlotter()
    # Plot just the specific CVE
    # plotter._plot_cve("CVE-2005-2594")
    # plotter._plot_cve("HERE WE CAN MAKE AN INTERESTING SELECTION OF CVES")
    # plotter.plot_all()
    # plotter._plot_cve("CVE-2005-2594")
    plotter._plot_cve("CVE-2024-3094")
    print("✅ Plot for CVE-2005-2594 complete!")