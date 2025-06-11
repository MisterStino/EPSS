#!/usr/bin/env python3
"""
End-to-end evaluation pipeline:  NVD → truth,  Mastodon → social signal,
LLM → prediction,  metrics ↝ txt/csv.

Usage
-----
Simply run the script with hardcoded file paths:

    python contained_classify.py

Configuration
-------------
Edit the variables in section 7 to customize:
- NVD_JSONL_PATH: Path to NVD JSONL snapshot file
- MASTODON_CSV_PATH: Path to Mastodon CSV data
- OUTPUT_DIR: Directory for results output  
- EVAL_DATE: Set to "YYYY-MM-DD" for single day, None for all dates
"""

from __future__ import annotations
import json, logging, re, sys
from pathlib import Path
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

# ──────────────────────────────────────────────────────────────────────────
# 1.  CONFIG
# ──────────────────────────────────────────────────────────────────────────
LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
LOGGER = logging.getLogger("eval")

CVSS_PROXY = {          # mid-points of NVD ranges + NONE
    "NONE": 0.0,
    "LOW": 2.0,
    "MEDIUM": 5.45,
    "HIGH": 7.95,
    "CRITICAL": 9.5,
}

CLASS_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]  # model never produces NONE
CVE_REGEX   = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
BOT_HINTS   = ("rss", "bot", "_feed", "auto", "scraper")


# ──────────────────────────────────────────────────────────────────────────
# 2.  NVD JSONL → flat CSV (one row per CVE)
# ──────────────────────────────────────────────────────────────────────────
def severity_from_v2_score(score: float) -> str:
    if score == 0.0:
        return "NONE"
    if score <= 3.9:
        return "LOW"
    if score <= 6.9:
        return "MEDIUM"
    return "HIGH"                       # v2 has no CRITICAL tier


def choose_metric(metrics: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return (cvssData, chosen_version_key)."""
    for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            return metrics[key][0]["cvssData"], key
    return None, ""


def jsonl_to_truth(jsonl_path: Path) -> pd.DataFrame:
    rows = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            cvss_data, ver_key = choose_metric(rec.get("metrics", {}))
            if not cvss_data:
                continue
            base = float(cvss_data.get("baseScore", 0.0))
            if ver_key == "cvssMetricV2":
                sev = severity_from_v2_score(base)
            else:
                sev = str(cvss_data.get("baseSeverity", "")).upper()
            rows.append(
                {
                    "cve": rec["id"].upper(),
                    "cvss_base_score": base,
                    "cvss_base_severity": sev,
                }
            )
    truth = pd.DataFrame(rows).drop_duplicates("cve")
    LOGGER.info("Ground truth rows: %d", len(truth))
    return truth


# ──────────────────────────────────────────────────────────────────────────
# 3.  Mastodon cleaning  → (CVE, date, text, account_type)
# ──────────────────────────────────────────────────────────────────────────
def classify_account(name: str, posts_per_day: float) -> str:
    n_lower = name.lower()
    if any(h in n_lower for h in BOT_HINTS) and posts_per_day >= 10:
        return "BOT_HIGH_CONFIDENCE"
    if any(h in n_lower for h in BOT_HINTS):
        return "BOT_MEDIUM_CONFIDENCE"
    return "HUMAN"


def toots_to_cve_date(toot_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(toot_csv)
    df = df.drop_duplicates()

    df["created_at_dt"] = pd.to_datetime(df["created_at"])
    df["date"] = df["created_at_dt"].dt.date

    # ---- very lightweight account classification ----
    posts_per_acct = df["acct"].value_counts()
    date_spans = df.groupby("acct")["date"].agg(['min', 'max'])
    # Convert to pandas datetime for proper subtraction, then get days
    days_active = pd.Series({
        acct: max((pd.to_datetime(row['max']) - pd.to_datetime(row['min'])).days, 1)
        for acct, row in date_spans.iterrows()
    })
    acct_rate = posts_per_acct / days_active
    acct_type_map = {
        acct: classify_account(acct, rate) for acct, rate in acct_rate.items()
    }
    df["account_type"] = df["acct"].map(acct_type_map)

    # ---- CVE extraction ----
    df["cve_list"] = df["content"].fillna("").str.findall(CVE_REGEX)
    df = df.explode("cve_list")
    df = df[df["cve_list"].notna()].copy()
    df["cve"] = df["cve_list"].str.upper()

    # ---- aggregate to (CVE, date) ----
    agg = (
        df.groupby(["cve", "date"], as_index=False)
        .agg(
            text=("content", "\n".join),
            account_types=("account_type", lambda s: s.mode().iloc[0] if len(s) else "UNKNOWN"),
        )
    )
    LOGGER.info("CVE-date rows from Mastodon: %d", len(agg))
    return agg


# ──────────────────────────────────────────────────────────────────────────
# 4.  Classifier wrapper
# ──────────────────────────────────────────────────────────────────────────
try:
    # Import using absolute import path for module execution
    from models.llm.src.label import VulnerabilitySeverityClassifier
except ImportError as exc:  # pragma: no cover
    LOGGER.error("Cannot import VulnerabilitySeverityClassifier: %s", exc)
    LOGGER.error("Make sure to run this script as a module: python -m data.mastodon.scripts.src.contained_classify")
    raise


def run_model(texts: List[str], batch_size: int = 32) -> List[str]:
    clf = VulnerabilitySeverityClassifier(device="auto")
    if not clf.is_ready():
        raise RuntimeError("Model failed to load")

    preds: List[str] = []
    for i in range(0, len(texts), batch_size):
        sev, _ = clf.predict(texts[i : i + batch_size])
        preds.extend(sev)
    return [s.upper() for s in preds]


# ──────────────────────────────────────────────────────────────────────────
# 5.  Metrics helpers
# ──────────────────────────────────────────────────────────────────────────
def cls_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    return {
        "accuracy": acc,
        "precision": dict(zip(CLASS_ORDER, p)),
        "recall": dict(zip(CLASS_ORDER, r)),
        "f1": dict(zip(CLASS_ORDER, f)),
        "support": dict(zip(CLASS_ORDER, s)),
        "confusion": cm,
    }


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


# ──────────────────────────────────────────────────────────────────────────
# 6.  Main evaluation routine
# ──────────────────────────────────────────────────────────────────────────
def evaluate(
    truth_df: pd.DataFrame,
    social_df: pd.DataFrame,
    eval_dates: Optional[List[date]],
    outdir: Path,
) -> None:
    if eval_dates:
        social_df = social_df[social_df["date"].isin(eval_dates)]

    if social_df.empty:
        LOGGER.warning("No social data left after date filter")
        return

    preds = run_model(social_df["text"].tolist())
    social_df["predicted_severity"] = preds
    social_df["predicted_cvss"] = social_df["predicted_severity"].map(CVSS_PROXY)

    merged = social_df.merge(truth_df, on="cve", how="inner")
    merged = merged[merged["cvss_base_severity"].isin(CLASS_ORDER)]  # drop NONE rows

    cls = cls_metrics(merged["cvss_base_severity"], merged["predicted_severity"])
    reg = reg_metrics(
        merged["cvss_base_score"].values.astype(float),
        merged["predicted_cvss"].values.astype(float),
    )

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_csv = outdir / f"eval_results_{ts}.csv"
    merged.to_csv(out_csv, index=False)

    # --- text report ---
    report = outdir / f"eval_report_{ts}.txt"
    with report.open("w") as fh:
        fh.write(f"Rows evaluated: {len(merged)}\n")
        fh.write(f"Accuracy: {cls['accuracy']:.3%}\n")
        fh.write("Precision/Recall/F1 per class:\n")
        for sev in CLASS_ORDER:
            fh.write(
                f"  {sev:<8} P={cls['precision'][sev]:.2f} "
                f"R={cls['recall'][sev]:.2f} "
                f"F1={cls['f1'][sev]:.2f}  (n={cls['support'][sev]})\n"
            )
        fh.write(
            f"\nRegression: MAE={reg['mae']:.2f} RMSE={reg['rmse']:.2f} R²={reg['r2']:.3f}\n"
        )
    # --- console ---
    print("=== FINAL SCOREBOARD ===")
    print(f"Rows evaluated      : {len(merged):,}")
    print(f"Accuracy            : {cls['accuracy']*100:5.1f}%")
    print(
        "MAE / RMSE / R²     : "
        f"{reg['mae']:.2f} | {reg['rmse']:.2f} | {reg['r2']:.3f}"
    )
    for sev in CLASS_ORDER:
        print(
            f"{sev:<8}: P={cls['precision'][sev]:.2f} "
            f"R={cls['recall'][sev]:.2f} F1={cls['f1'][sev]:.2f} "
            f"(n={cls['support'][sev]})"
        )
    print(f"\nDetailed rows → {out_csv}")
    print(f"Report        → {report}")


# ──────────────────────────────────────────────────────────────────────────
# 7.  CONFIGURATION - Hardcoded file paths and settings
# ──────────────────────────────────────────────────────────────────────────

# File paths (corrected for module execution from workspace root)
NVD_JSONL_PATH = Path("catalogs_processed/2025-06-08_snapshot.jsonl")
MASTODON_CSV_PATH = Path("data/mastodon/mastodon_raw.csv")
OUTPUT_DIR = Path("data/mastodon/scripts/src/out")

# Optional date filter (set to None for all dates, or "YYYY-MM-DD" for single day)
EVAL_DATE = None  # Example: "2025-06-07" for single day evaluation


def main() -> None:
    """Main execution with hardcoded configuration"""
    print("🚀 CVE Social Media Severity Evaluation Pipeline")
    print("=" * 60)
    
    # Validate input files exist
    if not NVD_JSONL_PATH.exists():
        raise FileNotFoundError(f"NVD file not found: {NVD_JSONL_PATH}")
    if not MASTODON_CSV_PATH.exists():
        raise FileNotFoundError(f"Mastodon file not found: {MASTODON_CSV_PATH}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    
    # Process ground truth data
    print(f"📊 Loading NVD ground truth from: {NVD_JSONL_PATH}")
    truth = jsonl_to_truth(NVD_JSONL_PATH)
    
    # Process social media data
    print(f"🐘 Loading Mastodon data from: {MASTODON_CSV_PATH}")
    social = toots_to_cve_date(MASTODON_CSV_PATH)
    
    # Handle date filtering
    dates = None
    if EVAL_DATE:
        try:
            dates = [datetime.strptime(EVAL_DATE, "%Y-%m-%d").date()]
            print(f"📅 Evaluating single date: {EVAL_DATE}")
        except ValueError as exc:
            raise SystemExit(f"Invalid EVAL_DATE format: {exc}") from None
    else:
        print("📅 Evaluating all dates (macro average)")
    
    # Run evaluation
    print("🤖 Starting model evaluation...")
    evaluate(truth, social, dates, OUTPUT_DIR)


if __name__ == "__main__":
    main()
