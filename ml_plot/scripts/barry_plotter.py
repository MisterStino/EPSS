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
from t3_spark.session import get_spark_session   # ← your helper


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
        self.spark = get_spark_session()                 # <- Spark here
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


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NC  = Path("ml_pipeline/results/predictions/predictions_stream.nc")
    RAW = Path("data/epss/processed/epss_processed.parquet")
    OUT = Path("ml_plots/plots")

    plotter = CVEPredPlotter(nc_path=NC, parquet=RAW, out_root=OUT)
    # example of adding two more random CVEs afterwards
    plotter.add_cves(n_random=2)
    plotter.plot_all()
