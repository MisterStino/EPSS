# quick_test_vis.py

import os, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

# ───────────────────────── configuration ────────────────────────────
INVLOG_TEST_PATH = Path("data/full_db/ml-sets-sampled/inv-log-scaled/test")
ORIG_TEST_PATH   = Path("data/full_db/ml-sets-sampled/original/test")
PRED_CSV         = Path("predictions_lstm_fullhistory.csv")
OUT_DIR          = Path("models/results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 10
SEED    = 42
random.seed(SEED)
np.random.seed(SEED)


# ─────────────────── 1) Recover (cve, asof) provenance ───────────────────────
class CVEPrefixDataset(Dataset):
    """ Re‐implement the exact prefix logic used at inference time. """
    def __init__(self, df: pd.DataFrame, horizon: int = 10):
        self.cves, self.asof = [], []
        for _, grp in df.groupby("cve", observed=True):
            vals  = grp[["epss", "age_epss_pub"]].values.astype("float32")
            dates = grp["date"].values
            T     = len(vals)
            if T <= horizon:
                continue
            for t in range(T - horizon):
                self.cves.append(grp.cve.iloc[0])
                self.asof.append(dates[t])   # end-of-prefix date
        assert len(self.cves) > 0

    def __len__(self): 
        return len(self.cves)

    def __getitem__(self, idx):
        # we only need (cve, asof) to re-attach predictions
        return self.cves[idx], self.asof[idx]


# ─────────────────── 2) Plotting helper ─────────────────────────────────────
def plot_forecast_segments(full_df: pd.DataFrame,
                           df_invlog_test: pd.DataFrame,
                           selected: list[str],
                           out_dir: Path):
    """
    For each CVE in `selected`, slice its merged truth+predictions
    into 6 time-chunks and produce a 6×1 panel plot showing
    the true inv-log EPSS curve and the 10-day-ahead dashed forecasts.
    """
    plt.rcParams["figure.figsize"] = (16, 8)
    colour_cycle = plt.cm.tab10(np.arange(10))

    for cve in selected:
        # a) TRUE on the *inv-log* scale
        truth = (
            df_invlog_test
              .loc[df_invlog_test.cve == cve, ["date","epss"]]
              .rename(columns={"epss":"truth"})
        )
        truth["date"] = pd.to_datetime(truth["date"])
        truth.sort_values("date", inplace=True)

        # b) MELT predictions
        rows = full_df[ full_df.cve == cve ]
        parts = []
        for h in range(1, HORIZON + 1):
            sub = rows[ ["cve","asof", f"pred_t+{h}"] ].copy()
            sub.rename(columns={f"pred_t+{h}":"pred"}, inplace=True)
            sub["date"]    = pd.to_datetime(sub["asof"]) + pd.to_timedelta(h, unit="D")
            sub["horizon"] = h
            parts.append( sub[["date","asof","horizon","pred"]] )
        pred_long = pd.concat(parts, ignore_index=True)
        pred_long["date"] = pd.to_datetime(pred_long["date"])

        # c) MERGE on the inv-log‐date column
        merged = (
            truth
              .merge(pred_long, on="date", how="left", validate="one_to_many")
              .sort_values("date")
              .reset_index(drop=True)
        )

        # d) slice into six panels
        N     = len(merged)
        chunk = math.ceil(N / 6)

        fig, axes = plt.subplots(6, 1, sharey=True, figsize=(16,18))
        fig.suptitle(f"{cve} – true vs 10-step forecasts (inv-log EPSS)", fontsize=14)

        for i, ax in enumerate(axes):
            seg = merged.iloc[i*chunk : (i+1)*chunk]
            if seg.empty:
                ax.axis("off")
                continue

            # solid black = the *inv-log* truth
            ax.plot(seg.date, seg.truth, color="black", linewidth=2)
            ax.text(seg.date.iloc[-1], seg.truth.iloc[-1],
                    " truth", va="center", fontsize=8, color="black")

            # dashed colored = each horizon’s forecast
            anchors = seg["asof"].dropna().unique()
            for j, asof in enumerate(anchors):
                col      = colour_cycle[j % len(colour_cycle)]
                pred_s = seg[ seg["asof"] == asof ]
                ax.plot(pred_s.date, pred_s.pred,
                        linestyle="--", color=col, linewidth=1)

            ax.set_xlim(seg.date.min(), seg.date.max())
            ax.grid(True, linestyle=":", linewidth=0.4)
            if i == 5: ax.set_xlabel("date")
            if i == 0: ax.set_ylabel("inv-log EPS")

        fig.tight_layout()
        out_path = out_dir / f"{cve}_forecast_segments.svg"
        fig.savefig(out_path)
        plt.close(fig)
        print("✅ saved", out_path)


# ─────────────────── 3) Main driver ────────────────────────────────────────
def main():
    # 1) provenance on inv-log test
    print("⟳ Loading inverse-log scaled test …")
    df_invlog  = pd.read_parquet(INVLOG_TEST_PATH, engine="pyarrow")
    ds         = CVEPrefixDataset(df_invlog, HORIZON)
    prov_df    = pd.DataFrame({
        "cve":  ds.cves,
        "asof": pd.to_datetime(ds.asof)
    })

    # 2) read forecasts & concat
    print("⟳ Reading predictions CSV …")
    pred_df  = pd.read_csv(PRED_CSV)
    if len(pred_df) != len(prov_df):
        raise RuntimeError("❌ Row-count mismatch! Regenerate preds with provenance.")
    full_df  = pd.concat([prov_df, pred_df], axis=1)

    # 3) pick CVEs by spike‐behaviour (using original scale only for stats)
    print("⟳ Loading original EPSS test for selection …")
    df_orig  = pd.read_parquet(ORIG_TEST_PATH, engine="pyarrow")
    stats = (
        df_orig
          .groupby("cve", observed=True)
          .agg(epss_max=("epss","max"),
               epss_t0 =("epss","first"))
          .reset_index()
    )
    quiet      = stats.loc[stats.epss_max   < 0.7, "cve"].tolist()
    always_hi  = stats.loc[stats.epss_t0    >= 0.7, "cve"].tolist()
    spike_late = stats.loc[(stats.epss_t0<0.7)&(stats.epss_max>=0.7), "cve"].tolist()

    random.shuffle(quiet); random.shuffle(always_hi); random.shuffle(spike_late)
    selected = (quiet[:5] + always_hi[:2] + spike_late[:3])[:10]
    print("ℹ️ Selected CVEs:", selected)

    # 4) make plots on the inv-log scale
    plot_forecast_segments(full_df, df_invlog, selected, OUT_DIR)
    print("🎉 All plots written to", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
