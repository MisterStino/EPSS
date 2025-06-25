# ml_pipeline/vis/plot_preds.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import warnings, random, sys, os

# ─── Matplotlib in head-less mode ───────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy  as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ─── Get Spark only once (lazy) ─────────────────────────────────────
from pyspark.sql import SparkSession
#from t3_spark.session import get_spark_session   # ← your helper

# ─── For the metrics ─────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def safe_mape(y_true, y_pred, eps=1e-2):
        mask = np.abs(y_true) > eps  # only keep values where y_true is not too close to zero
        if not np.any(mask):
            return np.nan  # or 0.0 or raise a warning
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

class CVEPredPlotter:
    """
    Draw **one PNG per forecast anchor** for every CVE in *cve_list*.

    If *cve_list* is None we auto-select (5 variable +2 hi +2 low +2 hi→low).

    Use `.add_cves()` to append a specific CVE *or* (with no arg)
    draw *n_random* additional random ones.
    """

    GOLD  = "#d4af37"
    BLUE  = "#1f77b4"
    GREEN = "#2ca02c"

    # ────────────────────────────────────────────────────────────────
    def __init__(
        self,
        nc_path : Path,
        parquet : Path,
        out_root: Path,
        cve_list: list[str] | None = None,
        eps: float = 1e-6,
    ):
        if not nc_path.exists():
            raise FileNotFoundError(nc_path)
        if not parquet.exists():
            raise FileNotFoundError(parquet)

        # heavy things first ----------------------
        self.ds    = xr.open_dataset(nc_path)
        self.spark = SparkSession.builder.getOrCreate() #get_spark_session()                 # <- Spark here
        self.pq    = parquet                            # keep Path

        # light config -----------------------------
        self.out_root   = out_root
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.invlog_eps = eps

        # choose CVEs ------------------------------
        if cve_list is None:
            self.cves: list[str] = self._auto_select_cves()
        else:
            self.cves = list(cve_list)

        print(
            f"🧙 Barry Plotter: “I solemnly swear I am up to good.”\n"
            f"   → NetCDF : {nc_path}\n"
            f"   → Parquet: {parquet}\n"
            f"   → Output : {out_root.resolve()}\n"
            f"   → CVEs   : {', '.join(self.cves)}"
        )

    # ────────────────────────────────────────────────────────────────
    ### public convenience -------------------------------------------------
    def add_cves(self, cve_id: str | None = None, n_random: int = 2):
        """
        • If *cve_id* provided → append it (if not already present).  
        • Else append *n_random* extra random CVEs.
        """
        if cve_id:
            if cve_id not in self.cves:
                self.cves.append(cve_id)
                print(f"✨  Added {cve_id}. Mischief managed!")
            else:
                print(f"⚠️  {cve_id} already in list – skipped.")
            return

        pool = [c for c in self.ds.cve.values.astype(str) if c not in self.cves]
        extra = random.sample(pool, min(n_random, len(pool)))
        self.cves.extend(extra)
        print(f"✨  Added {len(extra)} random CVEs: {', '.join(extra)}")

    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _inv_invlog(x: np.ndarray, eps: float):
        """inverse of  -log(p+eps)  →  p  (clip to [0,1])"""
        return np.clip(np.exp(-x) - eps, 0.0, 1.0)

    # ────────────────────────────────────────────────────────────────
    def _auto_select_cves(self) -> list[str]:
        """Return up to 11 CVEs following the bucket rules."""
        print("🔍  Auto-selecting CVEs by variability …")

        # truth: [C,T,H]  → back to probability
        arr = self._inv_invlog(self.ds.true.values, self.invlog_eps)
        rng   = arr.max((1, 2)) - arr.min((1, 2))
        mean  = arr.mean((1, 2))
        tp    = max(1, arr.shape[1] // 10)
        first = arr[:, :tp, :].mean((1, 2))
        last  = arr[:, -tp:, :].mean((1, 2))

        cves  = self.ds.cve.values.astype(str)
        df    = pd.DataFrame(dict(cve=cves, rng=rng, mean=mean,
                                  first=first, last=last))

        picks: list[str] = []

        # helper
        def take(q: str, k: int):
            subset = (
                df.query(q)
                  .sort_values("rng", ascending=False)
                  .cve.tolist()
            )
            new = [c for c in subset if c not in picks][:k]
            if len(new) < k:
                print(f"⚠️   only {len(new):>2}/{k} for “{q}”")
            picks.extend(new)

        # build buckets
        take("rng >= 0", 5)                           # 5 most variable
        take("rng <= 0.05 and mean >= 0.70", 2)       # steady high
        take("rng <= 0.05 and mean <= 0.30", 2)       # steady low
        take("first >= 0.70 and last <= 0.30", 2)     # high → low

        print(f"✨  Selected CVEs: {', '.join(picks)}")
        return picks

    # ────────────────────────────────────────────────────────────────
    @lru_cache(maxsize=256)
    def _hist_df(self, cve: str) -> pd.DataFrame:
        """Fetch one CVE’s full history via Spark (predicate push-down)."""
        sdf = (
            self.spark.read.parquet(str(self.pq))
            .filter(f"cve = '{cve}'")
            .select("date", "epss")
        )
        pdf = sdf.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf.sort_values("date", inplace=True)
        return pdf

    # ────────────────────────────────────────────────────────────────
    def _plot_single_anchor(
        self,
        cve      : str,
        idx_time : int,
        time_vec : np.ndarray,
        true_row : np.ndarray,
        pred_row : np.ndarray,
        out_dir  : Path,
    ):
        base_date = pd.to_datetime(time_vec[idx_time])
        #base_date = pd.to_datetime(time_vec[idx_time].item())

        if pd.isna(base_date):               # padding row – skip
            return

        horizon_dates = base_date + pd.to_timedelta(
            np.arange(1, 31), unit="D"
        )
        truth_y = self._inv_invlog(true_row, self.invlog_eps)
        pred_y  = self._inv_invlog(pred_row,  self.invlog_eps)

        hist = self._hist_df(cve)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(hist.date, hist.epss,
                color=self.GOLD, lw=2, label="EPSS truth (all)")
        ax.plot(horizon_dates, truth_y,
                color=self.GREEN, lw=1.5, label="30-day truth")
        ax.plot(horizon_dates, pred_y,
                color=self.BLUE, lw=1.5, ls="--", label="30-day forecast")

        ax.set_title(f"{cve}  |  anchor {base_date.date()}  (t={idx_time})")
        ax.set_xlabel("calendar date")
        ax.set_ylabel("EPSS probability")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(ls=":", lw=0.4)
        ax.legend(frameon=False, fontsize=8)

        out_path = out_dir / f"img_{idx_time:04d}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)

    # ────────────────────────────────────────────────────────────────
    def plot_all(self):
        """Loop CVEs → anchors, draw PNGs with tqdm progress bars."""
        for cve in self.cves:
            if cve not in self.ds.cve.values:
                warnings.warn(f"{cve} not found in NetCDF – skipped")
                continue

            da_pred = self.ds.pred     .sel(cve=cve)
            da_true = self.ds.true     .sel(cve=cve)
            da_hmsk = self.ds.mask_h   .sel(cve=cve)
            da_eval = self.ds.eval_mask.sel(cve=cve)
            time_vec = self.ds.time    .sel(cve=cve).values

            print(type(time_vec))
            print(time_vec)

            out_dir = self.out_root / cve
            out_dir.mkdir(exist_ok=True)

            total = int(da_eval.sum().values)
            print(f"\n🪄  Expecto Plotronum!  ({cve}) — {total} anchors")

            for t_idx in tqdm(
                range(len(da_eval)),
                unit="row",
                desc=f"{cve} plotting",
                leave=False,
            ):
                if not da_eval[t_idx] or not da_hmsk[t_idx].any():
                    continue
                self._plot_single_anchor(
                    cve, t_idx, time_vec,
                    da_true[t_idx].values, da_pred[t_idx].values,
                    out_dir
                )

            rel = out_dir.relative_to(self.out_root.parent)
            print(f"✅  {cve}: all charms saved under {rel}")
    def plot_random_forecast(self):
        '''
        # Pick a random CVE from the available ones
        available_cves = [cve for cve in self.cves if cve in self.ds.cve.values]
        if not available_cves:
            print("⚠️ No valid CVEs found in dataset.")
            return

        cve = random.choice(available_cves)

        # Extract data for the selected CVE
        da_pred = self.ds.pred.sel(cve=cve)
        da_true = self.ds.true.sel(cve=cve)
        da_hmsk = self.ds.mask_h.sel(cve=cve)
        da_eval = self.ds.eval_mask.sel(cve=cve)
        time_vec = self.ds.time.sel(cve=cve).values

        # Choose a valid anchor (evaluation = True and has any history mask)
        valid_idxs = [
            idx for idx in range(len(da_eval))
            if bool(da_eval[idx]) and bool(da_hmsk[idx].any())
        ]

        if not valid_idxs:
            print(f"⚠️ No valid anchors for CVE {cve}.")
            return

        idx_time = random.choice(valid_idxs)

        # Ensure output directory exists
        out_dir = Path("random_plot")
        out_dir.mkdir(exist_ok=True, parents=True)

        # Plot using existing method
        self._plot_single_anchor(
            cve=cve,
            idx_time=idx_time,
            time_vec=time_vec,
            true_row=da_true[idx_time].values,
            pred_row=da_pred[idx_time].values,
            out_dir=out_dir
        )

        print(f"✅ Random forecast plotted for CVE: {cve}, anchor index: {idx_time}")
        '''
        # pick a random CVE and valid anchor time index
        cve_idx = random.randint(0, self.ds.pred.shape[0] - 1)
        t_idx = random.randint(0, self.ds.pred.shape[1] - 1)

        cve = self.ds.cve.values[cve_idx]
        time_vec = self.ds.time.sel(cve=cve).values

        pred_row = self.ds.pred[cve_idx, t_idx, :].values
        true_row = self.ds.true[cve_idx, t_idx, :].values

        out_dir = Path("random_plot")
        out_dir.mkdir(exist_ok=True, parents=True)

        self._plot_single_anchor(
            cve=cve,
            idx_time=t_idx,
            time_vec=time_vec,
            true_row=true_row,
            pred_row=pred_row,
            out_dir=out_dir
        )

    def plot_horizon_metrics(
        self,
        mae_list: list,
        rmse_list: list,
        mape_list: list,
        smape_list: list,
        r2_list: list,
        out_path: Path = None
    ):

        horizons = np.arange(len(mae_list))

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(horizons, mae_list, label="MAE", color="#f39c12", lw=2)
        ax.plot(horizons, rmse_list, label="RMSE", color="#2980b9", lw=2)
        ax.plot(horizons, smape_list, label="SMAPE", color="#27ae60", lw=2)
        ax.plot(horizons, r2_list, label="R²", color="#8e44ad", lw=2)
        # Optional: MAPE if you want to include it too
        ax.plot(horizons, mape_list, label="MAPE", color="#c0392b", lw=2)

        ax.set_xlabel("Forecast Horizon (Days)", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title("Forecasting Performance by Horizon", fontsize=14)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.legend(frameon=False)
        ax.set_xticks(horizons)
        ax.set_xlim([0, len(horizons) - 1])

        fig.tight_layout()

        if out_path is not None:
            fig.savefig(out_path, dpi=150)
            print(f"📈 Horizon metrics plot saved to {out_path}")
        else:
            plt.show()

        plt.close(fig)

    def get_metrics_by_horizon(self):
        '''
        num_horizons = self.ds.horizon.size
        mae_list, rmse_list, mape_list, smape_list, r2_list = [], [], [], [], []

        for h in range(num_horizons):
            all_y_true = []
            all_y_pred = []

            for cve in self.cves:
                if cve not in self.ds.cve.values:
                    continue

                da_pred = self.ds.pred.sel(cve=cve)
                da_true = self.ds.true.sel(cve=cve)

                if h not in da_pred.horizon:
                    continue  # just in case

                y_pred = da_pred.sel(horizon=h).values
                y_true = da_true.sel(horizon=h).values

                # Flatten and filter NaNs
                valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
                all_y_true.append(y_true[valid])
                all_y_pred.append(y_pred[valid])

            # Combine all valid predictions from all CVEs for this horizon
            y_true_all = np.concatenate(all_y_true)
            y_pred_all = np.concatenate(all_y_pred)

            # Skip if no data
            if len(y_true_all) == 0:
                continue

            # Compute metrics
            mae = mean_absolute_error(y_true_all, y_pred_all)
            rmse = mean_squared_error(y_true_all, y_pred_all) ** 0.5
            r2 = r2_score(y_true_all, y_pred_all)
            mape_val = safe_mape(y_true_all, y_pred_all)
            smape_val = 100 * np.mean(2 * np.abs(y_true_all - y_pred_all) /
                                    (np.abs(y_pred_all) + np.abs(y_true_all) + 1e-8))

            print(f"Horizon {h:2d} → MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape_val:.2f}%, SMAPE: {smape_val:.2f}%, R2: {r2:.4f}")

            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape_val)
            smape_list.append(smape_val)
            r2_list.append(r2)

            #plotter.plot_horizon_metrics(mae_list=mae_list, rmse_list=rmse_list, mape_list=mape_list, smape_list=smape_list, r2_list=r2_list, out_path=Path("horizon_metrics.png"))
            '''
        da_pred = self.ds.pred     # shape: (cve, anchor_day, horizon)
        da_true = self.ds.true     # shape: (cve, anchor_day, horizon)

        y_pred = da_pred.values    # shape: (10000, 386, 30)
        y_true = da_true.values    # shape: (10000, 386, 30)

        # Flatten cve and anchor_day for each horizon
        mae_list, rmse_list, mape_list, smape_list, r2_list = [], [], [], [], []

        for h in range(y_true.shape[2]):  # for each horizon
            y_p = y_pred[:, :, h].flatten()
            y_t = y_true[:, :, h].flatten()

            # Filter invalid (nan) values
            valid = ~np.isnan(y_p) & ~np.isnan(y_t)
            y_p = y_p[valid]
            y_t = y_t[valid]

            if len(y_p) == 0:
                print(f"Horizon {h}: No valid predictions.")
                continue

            mae = mean_absolute_error(y_t, y_p)
            rmse = mean_squared_error(y_t, y_p) ** 0.5
            r2 = r2_score(y_t, y_p)
            mape_val = safe_mape(y_t, y_p)
            smape_val = 100 * np.mean(2 * np.abs(y_t - y_p) / (np.abs(y_p) + np.abs(y_t) + 1e-8))

            print(f"Horizon {h:2} → MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape_val:.2f}%, SMAPE: {smape_val:.2f}%, R2: {r2:.4f}")

            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape_val)
            smape_list.append(smape_val)
            r2_list.append(r2)

        # Optionally plot the metrics
        self.plot_horizon_metrics(
            mae_list=mae_list,
            rmse_list=rmse_list,
            mape_list=mape_list,
            smape_list=smape_list,
            r2_list=r2_list,
            out_path=Path("horizon_metrics.png")
        )
    def get_metrics_new(self):
        #we are going to have all cves, and for each day
        #we are having 3 plots, horizon 0,14,29
        da_pred = self.ds.pred     # shape: (cve, anchor_day, horizon)
        da_true = self.ds.true     # shape: (cve, anchor_day, horizon)

        y_pred = da_pred.values    # shape: (10000, 386, 30)
        y_true = da_true.values    # shape: (10000, 386, 30)

        # Let's say y_true and y_pred are shaped (num_cve, num_days, num_horizons)
        horizons_to_plot = [0, 14, 29]  # Horizon indices (0-indexed for 1, 15, 30)
        # Store results: metric_name -> [day-wise values per horizon]
        metrics_results = {
            "MAE": {h: [] for h in horizons_to_plot},
            "RMSE": {h: [] for h in horizons_to_plot},
            "MAPE": {h: [] for h in horizons_to_plot},
            "SMAPE": {h: [] for h in horizons_to_plot},
            "R2": {h: [] for h in horizons_to_plot},
        }
        days = y_true.shape[1]
        
        results = {h: [] for h in horizons_to_plot}

        for h in horizons_to_plot:
            print(h)
            for d in range(days):
                y_p = y_pred[:, d, h]
                y_t = y_true[:, d, h]

                valid = ~np.isnan(y_p) & ~np.isnan(y_t)
                if valid.sum() == 0:
                    # Append NaN to all metrics so lists have consistent length
                    for metric in metrics_results:
                        metrics_results[metric][h].append(np.nan)
                    continue

                y_p_valid = y_p[valid]
                y_t_valid = y_t[valid]

                mae = mean_absolute_error(y_t_valid, y_p_valid)
                rmse = mean_squared_error(y_t_valid, y_p_valid) ** 0.5
                r2 = r2_score(y_t_valid, y_p_valid)
                mape_val = safe_mape(y_t_valid, y_p_valid)
                smape_val = 100 * np.mean(2 * np.abs(y_t_valid - y_p_valid) / (np.abs(y_p_valid) + np.abs(y_t_valid) + 1e-8))

                print(f"Horizon: {h} → Day: {d} → MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape_val:.2f}%, SMAPE: {smape_val:.2f}%, R2: {r2:.4f}")

                metrics_results["MAE"][h].append(mae)
                metrics_results["RMSE"][h].append(rmse)
                metrics_results["MAPE"][h].append(mape_val)
                metrics_results["SMAPE"][h].append(smape_val)
                metrics_results["R2"][h].append(r2)

                
            # --- Unified Plot Across Metrics ---
            anchor_days = np.arange(days)
            color_map = {
                "MAE": "#f39c12",
                "RMSE": "#2980b9",
                "SMAPE": "#27ae60",
                "R2": "#8e44ad",
                "MAPE": "#c0392b"
            }

            fig, ax = plt.subplots(figsize=(12, 6))

            for metric, color in color_map.items():
                # Defensive check: skip if data is empty or length mismatch
                
                if len(metrics_results[metric][h]) != days:
                    print(f"Skipping metric {metric}, horizon {h} due to data length mismatch")
                    continue
                ax.plot(anchor_days, metrics_results[metric][h], label=metric, color=color, lw=2)

            ax.set_xlabel("Anchor Day", fontsize=12)
            ax.set_ylabel("Metric Value", fontsize=12)
            ax.set_title(f"Metrics over Anchor Days – Horizon {h + 1}", fontsize=14)
            ax.grid(True, linestyle=":", linewidth=0.5)
            ax.legend(frameon=False)
            ax.set_xlim([0, days - 1])

            fig.tight_layout()

            # Save with unique name per horizon
            filename = f"forecast_metrics_h{h+1}.png"
            fig.savefig(filename, dpi=150)
            print(f"📈 Metrics plot for Horizon {h + 1} saved to {filename}")

            plt.show()  # optional in non-interactive environments
            plt.close(fig)

    def get_metrics(self):
        for cve in self.cves:
            
            if cve not in self.ds.cve.values:
                warnings.warn(f"{cve} not found in NetCDF – skipped")
                continue

            da_pred = self.ds.pred     .sel(cve=cve)
            da_true = self.ds.true     .sel(cve=cve)
            da_horizon = self.ds.horizon     .sel(cve=cve)
            da_hmsk = self.ds.mask_h   .sel(cve=cve)
            da_eval = self.ds.eval_mask.sel(cve=cve)
            time_vec = self.ds.time    .sel(cve=cve).values
            
            # Convert to NumPy
            y_pred = da_pred.values.flatten()
            y_true = da_true.values.flatten()

            # Skip CVEs with only NaNs (optional)
            if np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred)):
                continue

            # Optional: mask invalid values
            valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_true = y_true[valid]
            y_pred = y_pred[valid]

            # Compute metrics
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            r2 = r2_score(y_true, y_pred)
            mape_val = safe_mape(y_true, y_pred)
            smape_val = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))

            print(f"CVE: {cve} → MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape_val:.2f}%, SMAPE: {smape_val:.2f}%, R2: {r2:.4f}")

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NC  = Path("ml_plot\perf\files\predictions_stream.nc")
    RAW = Path("data/epss/processed/epss_processed.parquet")
    OUT = Path("ml_plot/perf/perf_plots")

    plotter = CVEPredPlotter(nc_path=NC, parquet=RAW, out_root=OUT)
    # example of adding two more random CVEs afterwards
    
    # get the metrics for each cve
    # plotter.get_metrics()
    
    # get the metrics for all cves across horizons
    #plotter.get_metrics_by_horizon()
    #plotter.get_metrics_new()
    plotter.add_cves(n_random=2)
    #plotter.plot_random_forecast()
    plotter.plot_all()
