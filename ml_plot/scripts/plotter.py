from __future__ import annotations

"""ml_plot.scripts.plotter

Utility to visualise per-timestamp EPSS forecasts saved by
ml_pipeline.lstm_exp_window_eval.

The main entry-point is the EPSSPredictionPlotter class.  Typical usage::

    python -m ml_plot.scripts.plotter  # runs a demo on first three CVEs

or, from another module::

    from ml_plot.scripts.plotter import EPSSPredictionPlotter

    plotter = EPSSPredictionPlotter(
        preds_path="ml_pipeline/results/predictions/predictions_stream.nc",
        epss_path="data/epss/processed/epss_processed.parquet",
    )
    plotter.plot_all()  # writes PNGs under ml_plot/plots/<CVE>/
"""

from pathlib import Path
from typing import Sequence, Optional
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

__all__ = ["EPSSPredictionPlotter"]

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    """Create *path* (directory) if it does not yet exist."""
    path.mkdir(parents=True, exist_ok=True)


def _log_to_prob(arr: np.ndarray) -> np.ndarray:
    """
    Convert from log space back to probability space.
    
    The training pipeline uses: log(p + 1e-6) where p is EPSS probability
    So the inverse transform is: exp(x) - 1e-6, then clip to [0,1]
    
    Note: EPSS is no longer z-scored in the preprocessing pipeline,
    so we only need to undo the log transform.
    """
    transformed = np.exp(arr) - 1e-6
    return np.clip(transformed, 0.0, 1.0)


# -----------------------------------------------------------------------------
# Main visualiser class
# -----------------------------------------------------------------------------

class EPSSPredictionPlotter:
    """Generate per-timestamp forecast plots for selected CVEs.

    Parameters
    ----------
    preds_path : str | Path
        Path to ``predictions_stream.nc`` produced by the LSTM pipeline.
    epss_path : str | Path
        Path to the *long* format Parquet file that contains raw EPSS
        probabilities (one row per (cve,date)).
    out_dir : str | Path, optional
        Directory root where plots will be written.  Defaults to
        ``ml_plot/plots``.
    cves : Sequence[str] | None
        List of CVE IDs to plot.  If *None*, the first three CVEs contained
        in *preds_path* are used.
    """

    def __init__(
        self,
        *,
        preds_path: str = "ml_pipeline/results/predictions/predictions_stream.nc",
        epss_path: str = "data/epss/processed/epss_processed.parquet",
    ):
        self.preds_path = Path(preds_path)
        self.epss_path = Path(epss_path)
        self.out_root = Path("ml_plot/plots")
        _ensure_dir(self.out_root)

        if not self.preds_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {self.preds_path}")
        if not self.epss_path.exists():
            raise FileNotFoundError(f"EPSS parquet file not found: {self.epss_path}")

        self.ds = xr.open_dataset(self.preds_path)
        # Fix time coordinate: the datetime64 values are malformed
        # They contain days-since-epoch stored as nanoseconds-since-epoch
        time_data = self.ds.time.values
        
        # Extract the raw integer values (which are actually days since epoch)
        # The datetime64[ns] values like '1970-01-01T00:00:00.000019436' 
        # have 19436 nanoseconds, but 19436 is actually days since epoch
        time_ints = time_data.view('int64')  # Extract raw nanosecond values
        
        # Convert from nanoseconds to days (the raw values are actually days)
        # Since datetime64[ns] stores nanoseconds since epoch, we extract those
        # but they represent days, so we convert: days_since_epoch = nanoseconds / 1e9 / 86400
        # But actually, the values are just the day numbers stored as ns
        epoch = np.datetime64('1970-01-01')
        
        # The raw values like 19436 are stored as nanoseconds but represent days
        days_since_epoch = time_ints  # These are the actual day numbers
        time_data_fixed = epoch + days_since_epoch.astype('timedelta64[D]')
        self.ds["time"] = (self.ds.time.dims, time_data_fixed)

        # Choose CVEs based on EPSS variability patterns
        self.cves = self._select_cves_by_stats()

        # Load full EPSS time-series once; subset per CVE later.
        # Only required columns: cve, date, epss
        self.epss_df = (
            pd.read_parquet(self.epss_path, columns=["cve", "date", "epss"])
            .assign(date=lambda df: pd.to_datetime(df["date"]))
        )

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def plot_all(self) -> None:
        """Generate PNGs for each CVE in *self.cves*."""
        for cve in self.cves:
            print(f"[PLOT] {cve}")
            self._plot_cve(cve)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_cves_by_stats(self) -> list[str]:
        """
        Select CVEs based on EPSS variability patterns in test data.
        
        Returns up to 18 CVEs total:
        - 12 high-variability "big jump" (min < 0.3 AND max > 0.7) - stratified by year
        - 2 low & flat (mean ≤ 0.3 AND range < 0.05)
        - 2 high & flat (mean ≥ 0.7 AND range < 0.05) 
        - 2 medium, high variability (0.3 < mean < 0.7, max ≤ 0.7, range > 0.3)
        """
        print("[INFO] Analyzing CVEs for variability-based selection...")
        
        # Convert true data to probability space (create copy to avoid in-place modification)
        true_data_copy = self.ds.true.values.copy()  # [CVE, time, horizon]
        true_prob = _log_to_prob(true_data_copy)
        eval_mask = self.ds.eval_mask.values.astype(bool)  # [CVE, time]
        
        # Initialize category lists
        big_jump_candidates = []  # Will stratify this by year
        low_flat_cves = []        # Target: 2
        high_flat_cves = []       # Target: 2
        medium_var_cves = []      # Target: 2
        
        all_cves = list(self.ds.cve.values)
        
        # First pass: categorize all CVEs
        for i, cve in enumerate(all_cves):
            # Get test timestamps for this CVE
            test_mask = eval_mask[i]  # [time]
            if not test_mask.any():
                continue  # Skip CVEs with no test data
            
            # Extract true EPSS values from test timestamps ONLY (not their future horizons)
            # We want the EPSS value ON the test date itself, not predictions FROM that date
            cve_true = true_prob[i]  # [time, horizon]
            test_true = cve_true[test_mask]  # [test_times, horizon]
            
            # Use only horizon=0 (the anchor date itself) for variability analysis
            # This gives us the actual EPSS score on each test timestamp
            test_anchor_values = test_true[:, 0]  # [test_times] - EPSS on test dates only
            
            # Remove any invalid values (NaN, inf, etc.)
            valid_values = test_anchor_values[np.isfinite(test_anchor_values)]
            if len(valid_values) == 0:
                continue
            
            # Compute statistics
            p_min = np.min(valid_values)
            p_max = np.max(valid_values)
            p_mean = np.mean(valid_values)
            p_range = p_max - p_min
            
            # Extract year from CVE ID
            import re
            year_match = re.match(r'CVE-(\d{4})-\d+', cve)
            year = int(year_match.group(1)) if year_match else None
            
            # Categorize based on criteria
            if p_min < 0.3 and p_max > 0.7:
                big_jump_candidates.append((cve, year, p_min, p_max, p_range))
                
            elif len(low_flat_cves) < 2 and p_mean <= 0.3 and p_range < 0.05:
                low_flat_cves.append(cve)
                print(f"  Low flat: {cve} (mean={p_mean:.3f}, range={p_range:.3f})")
                
            elif len(high_flat_cves) < 2 and p_mean >= 0.7 and p_range < 0.05:
                high_flat_cves.append(cve)
                print(f"  High flat: {cve} (mean={p_mean:.3f}, range={p_range:.3f})")
                
            elif (len(medium_var_cves) < 2 and 
                  0.3 < p_mean < 0.7 and p_max <= 0.7 and p_range > 0.3):
                medium_var_cves.append(cve)
                print(f"  Medium var: {cve} (mean={p_mean:.3f}, max={p_max:.3f}, range={p_range:.3f})")
        
        # Stratified selection for big jump CVEs
        big_jump_cves = self._stratify_big_jump_cves(big_jump_candidates)
        
        # Combine all selected CVEs
        selected_cves = big_jump_cves + low_flat_cves + high_flat_cves + medium_var_cves
        
        print(f"[INFO] Selected {len(selected_cves)} CVEs:")
        print(f"  - Big jump (stratified): {len(big_jump_cves)}/12")
        print(f"  - Low & flat: {len(low_flat_cves)}/2") 
        print(f"  - High & flat: {len(high_flat_cves)}/2")
        print(f"  - Medium variability: {len(medium_var_cves)}/2")
        
        if len(selected_cves) == 0:
            print("[WARN] No CVEs found matching criteria, falling back to first 3")
            return all_cves[:3]
            
        return selected_cves
    
    def _stratify_big_jump_cves(self, candidates: list) -> list[str]:
        """
        Stratify big jump CVEs by year: 2 old + 6 recent + 4 random.
        
        Args:
            candidates: List of (cve, year, p_min, p_max, p_range) tuples
        
        Returns:
            List of up to 12 selected CVE IDs
        """
        if not candidates:
            return []
        
        # Group by year
        import random
        from collections import defaultdict
        
        year_groups = defaultdict(list)
        for cve, year, p_min, p_max, p_range in candidates:
            if year is not None:
                year_groups[year].append((cve, p_min, p_max, p_range))
        
        if not year_groups:
            return []
        
        # Get sorted years
        sorted_years = sorted(year_groups.keys())
        min_year, max_year = sorted_years[0], sorted_years[-1]
        
        print(f"[INFO] Big jump CVEs span {min_year}-{max_year} ({len(sorted_years)} years)")
        
        # Define old and recent years dynamically
        old_years = sorted_years[:2]  # First 2 years
        recent_years = sorted_years[-3:] if len(sorted_years) >= 3 else sorted_years[-1:]  # Last 3 years
        
        selected_cves = []
        
        # Select 2 from old years
        old_candidates = []
        for year in old_years:
            old_candidates.extend(year_groups[year])
        
        if old_candidates:
            old_selected = random.sample(old_candidates, min(2, len(old_candidates)))
            for cve, p_min, p_max, p_range in old_selected:
                selected_cves.append(cve)
                print(f"  Big jump (old): {cve} (min={p_min:.3f}, max={p_max:.3f}, range={p_range:.3f})")
        
        # Select up to 6 from recent years
        recent_candidates = []
        for year in recent_years:
            recent_candidates.extend(year_groups[year])
        
        # Remove already selected CVEs
        recent_candidates = [c for c in recent_candidates if c[0] not in selected_cves]
        
        if recent_candidates:
            num_recent = min(6, len(recent_candidates))
            recent_selected = random.sample(recent_candidates, num_recent)
            for cve, p_min, p_max, p_range in recent_selected:
                selected_cves.append(cve)
                print(f"  Big jump (recent): {cve} (min={p_min:.3f}, max={p_max:.3f}, range={p_range:.3f})")
        
        # Fill remaining slots randomly from all candidates
        remaining_slots = 12 - len(selected_cves)
        if remaining_slots > 0:
            all_candidates = [c for c in candidates if c[0] not in selected_cves]
            if all_candidates:
                num_random = min(remaining_slots, len(all_candidates))
                random_selected = random.sample(all_candidates, num_random)
                for cve, year, p_min, p_max, p_range in random_selected:
                    selected_cves.append(cve)
                    print(f"  Big jump (random): {cve} (min={p_min:.3f}, max={p_max:.3f}, range={p_range:.3f})")
        
        print(f"[INFO] Stratified selection: {len(selected_cves)} big jump CVEs")
        print(f"  - Old years ({old_years}): {len([c for c in selected_cves[:2]])}")
        print(f"  - Recent years ({recent_years}): {len([c for c in selected_cves[2:2+min(6, len(recent_candidates))]])}")
        print(f"  - Random: {len(selected_cves) - len([c for c in selected_cves[:2]]) - len([c for c in selected_cves[2:2+min(6, len(recent_candidates) if recent_candidates else 0)]])}")
        
        return selected_cves

    def _plot_cve(self, cve: str) -> None:
        """Generate all plots for one CVE."""
        # Extract data for this CVE
        pred_cve = self.ds.sel(cve=cve)
        time_coord = pred_cve.time.values
        pred_log = pred_cve.pred.values  # In log space
        true_log = pred_cve.true.values  # In log space
        mask_h = pred_cve.mask_h.values.astype(bool)  # [L, H]
        eval_mask = pred_cve.eval_mask.values.astype(bool)  # [L]
        horizon = pred_cve.dims["horizon"]

        # Transform from log space to probability space
        pred_prob = _log_to_prob(pred_log)
        true_prob = _log_to_prob(true_log)

        # Get EPSS data for this CVE (already in probability space)
        cve_epss = self.epss_df[self.epss_df.cve == cve].copy()
        if cve_epss.empty:
            print(f"   ⚠ No EPSS data found for {cve}")
            return
        
        cve_epss['date'] = pd.to_datetime(cve_epss['date'])
        cve_epss = cve_epss.sort_values('date')

        # Create output directory
        cve_out = self.out_root / cve
        cve_out.mkdir(parents=True, exist_ok=True)

        # Generate plots for each test timestamp
        for idx in np.where(eval_mask)[0]:
            anchor_date = time_coord[idx]
            out_path = cve_out / f"{pd.to_datetime(anchor_date).date()}__t{idx:04d}.png"
            
            self._make_single_plot(
                cve=cve,
                cve_epss=cve_epss,
                anchor_date=anchor_date,
                pred_row=pred_prob[idx],
                true_row=true_prob[idx],
                mask_row=mask_h[idx],
                horizon=horizon,
                out_path=out_path,
            )

    def _make_single_plot(
        self,
        *,
        cve: str,
        cve_epss: pd.DataFrame,
        anchor_date: np.datetime64,
        pred_row: np.ndarray,  # (H,)
        true_row: np.ndarray,  # (H,)
        mask_row: np.ndarray,  # (H,)
        horizon: int,
        out_path: Path,
    ) -> None:
        """Render and save one figure for a single (CVE, time-step)."""
        # Base line – focused EPSS curve around test period
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define focused time window around test period
        context_days = 60  # Days of context before anchor
        horizon_end_date = pd.to_datetime(anchor_date) + pd.Timedelta(days=horizon)
        context_start_date = pd.to_datetime(anchor_date) - pd.Timedelta(days=context_days)
        
        # Filter EPSS data to focused time window
        focused_epss = cve_epss[
            (cve_epss.date >= context_start_date) & 
            (cve_epss.date <= horizon_end_date)
        ].copy()
        
        # Plot in correct z-order: background to foreground
        if not focused_epss.empty:
            # 1. Gold line (background) - EPSS truth (focused view)
            ax.plot(focused_epss.date, focused_epss.epss, color="gold", label="EPSS (truth)", linewidth=2, zorder=1)
        else:
            print(f"   ⚠ No EPSS data in focused window ({context_start_date.date()} to {horizon_end_date.date()}) for {cve}")
        
        # 2. Vertical line at anchor date (model cutoff)
        ax.axvline(x=pd.to_datetime(anchor_date), color='red', linestyle=':', alpha=0.7, 
                   label='Anchor (model cutoff)', zorder=4)

        # Horizon dates relative to anchor
        horizon_days = np.arange(1, horizon + 1)
        horiz_dates = pd.to_datetime(anchor_date) + pd.to_timedelta(horizon_days, unit="D")

        # Mask invalid horizons (where mask_h==0)
        valid = mask_row.astype(bool)
        if valid.any():
            # 2. Green line (middle layer) - True horizon
            ax.plot(
                horiz_dates[valid],
                true_row[valid],
                color="green",
                label="True horizon",
                linewidth=2,
                zorder=2,
            )
            # 3. Blue dashed line (foreground) - Predicted horizon (always on top)
            ax.plot(
                horiz_dates[valid],
                pred_row[valid],
                color="blue",
                linestyle="--",
                label="Predicted horizon",
                linewidth=2,
                zorder=3,  # Highest z-order to always be visible
            )

        # Formatting
        anchor_date_str = pd.to_datetime(anchor_date).strftime('%Y-%m-%d')
        ax.set_title(f"{cve} | anchor {anchor_date_str} | {context_days}-day context + {horizon}-day forecast")
        ax.set_xlabel("Calendar Date")
        ax.set_ylabel("EPSS probability")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend()
        
        # Better date formatting for focused view
        ax.tick_params(axis='x', rotation=45)
        fig.autofmt_xdate()
        fig.tight_layout()

        _ensure_dir(out_path.parent)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print("   ✔", out_path)


# -----------------------------------------------------------------------------
# CLI helper (runs a minimal demo)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Default paths – adjust if your repo layout differs.
    default_preds = Path("ml_pipeline/results/predictions/predictions_stream.nc")
    default_epss = Path("data/epss/processed/epss_processed.parquet")

    plotter = EPSSPredictionPlotter(
        preds_path=str(default_preds), 
        epss_path=str(default_epss)
    )
    plotter.plot_all()
