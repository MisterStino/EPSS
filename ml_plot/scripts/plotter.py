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
        max_cves: int = 10,  # Increased from 3 to get more examples
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

        # Choose CVEs (more for better validation)
        all_cves = list(self.ds.cve.values)
        self.cves = all_cves[:max_cves]

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
        # Base line – full EPSS curve
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot in correct z-order: background to foreground
        # 1. Gold line (background) - EPSS truth
        ax.plot(cve_epss.date, cve_epss.epss, color="gold", label="EPSS (truth)", linewidth=2, zorder=1)

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
        ax.set_title(f"{cve}  |  anchor {anchor_date_str}  (next {horizon}-day forecast)")
        ax.set_xlabel("Calendar Date")
        ax.set_ylabel("EPSS probability")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend()
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
        epss_path=str(default_epss),
        max_cves=10
    )
    plotter.plot_all()
