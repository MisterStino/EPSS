#!/usr/bin/env python3
"""Validation script for stupid model predictions.

This script checks that `stupid_predictions.nc` is structurally
and numerically consistent with the source NetCDF file and the
EPSS parquet table.  It implements the validation plan that was
approved in the previous conversation.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from t3_spark.session import get_spark_session
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_stupid_predictions")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def find_project_root(start: Path | None = None) -> Path:
    """Walk up until we see `ml_pipeline` directory."""
    current = (start or Path(__file__).resolve()).parent
    while current != current.parent:
        if (current / "ml_pipeline").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not locate project root (no ml_pipeline dir found)")


def size_gb(path: Path) -> float:
    return path.stat().st_size / 1024 ** 3


def pick_random_valid_positions(eval_mask: np.ndarray, mask_h: np.ndarray, n: int = 5000, seed: int = 42):
    """Return list of (cve_idx, time_idx, horizon_idx) tuples that are valid."""
    rng = np.random.default_rng(seed)
    valid_coords = np.where(eval_mask[:, :, None] & mask_h)
    total_valid = len(valid_coords[0])
    if total_valid == 0:
        return []
    idx = rng.choice(total_valid, size=min(n, total_valid), replace=False)
    return list(zip(valid_coords[0][idx], valid_coords[1][idx], valid_coords[2][idx]))

# ---------------------------------------------------------------------------
# Core validation logic
# ---------------------------------------------------------------------------

def validate(
    source_nc: Path,
    stupid_nc: Path,
    epss_parquet: Path,
    sample_points: int = 5000,
    coverage_stride: int = 1000,  # iterate over CVEs in batches to count coverage
):
    project_root = find_project_root()
    source_nc = source_nc if source_nc.is_absolute() else project_root / source_nc
    stupid_nc = stupid_nc if stupid_nc.is_absolute() else project_root / stupid_nc
    epss_parquet = epss_parquet if epss_parquet.is_absolute() else project_root / epss_parquet

    logger.info("================ BASIC FILE STATS ================")
    logger.info(f"Source  NetCDF: {source_nc}  | size = {size_gb(source_nc):.2f} GB")
    logger.info(f"Stupid NetCDF: {stupid_nc}  | size = {size_gb(stupid_nc):.2f} GB")
    logger.info(f"EPSS  Parquet: {epss_parquet}")

    # ---------------------------------------------------------------------
    # Open NetCDF files lazily (chunk along CVE to keep memory usage low)
    # ---------------------------------------------------------------------
    logger.info("Opening NetCDF datasets… (lazy loading)")
    ds_src = xr.open_dataset(source_nc, chunks={"cve": 1024})
    ds_stu = xr.open_dataset(stupid_nc, chunks={"cve": 1024})

    # ---------------------------------------------------------------------
    # 1. Shape & coord equality
    # ---------------------------------------------------------------------
    logger.info("\n=== 1. DIM / COORD PARITY ===")
    for dim in ["cve", "time", "horizon"]:
        assert ds_src.sizes[dim] == ds_stu.sizes[dim], f"Mismatch in dimension {dim}"
    logger.info("Dimensions identical ✔")

    for coord in ["cve", "horizon"]:  # one-dimensional coords
        assert np.array_equal(ds_src[coord].values, ds_stu[coord].values), f"Mismatch in {coord} coord"

    # Special handling for 2-D time coordinate with NaT values
    src_time = ds_src["time"].values
    stu_time = ds_stu["time"].values
    assert src_time.shape == stu_time.shape, "Time coord shape mismatch"
    same_mask = (src_time == stu_time) | (np.isnat(src_time) & np.isnat(stu_time))
    assert same_mask.all(), "Time coordinate values differ (ignoring NaT)"

    # dtypes
    assert ds_src.pred.dtype == ds_stu.pred.dtype == np.float32, "pred dtype mismatch"
    logger.info("pred dtype float32 ✔")

    # ---------------------------------------------------------------------
    # 2. Uniform-horizon rule & sample value check
    # ---------------------------------------------------------------------
    logger.info("\n=== 2. UNIFORM HORIZON & VALUE CHECKS ===")
    eval_mask_np = ds_src.eval_mask.values.astype(bool)
    mask_h_np = ds_src.mask_h.values.astype(bool)

    sample_coords = pick_random_valid_positions(eval_mask_np, mask_h_np, n=sample_points)
    logger.info(f"Sampled {len(sample_coords):,} valid positions for value checks")

    # Create Spark session only if we actually need it (lazy)
    spark = None
    violations_uniform = 0
    violations_value = 0

    for (c_idx, t_idx, h_idx) in tqdm(sample_coords, ncols=80, desc="Checking samples"):
        pred_row = ds_stu.pred.values[c_idx, t_idx, :]
        # uniform horizon: ignore NaNs that might originate from mask_h
        finite_vals = pred_row[np.isfinite(pred_row)]
        if len(finite_vals) > 0 and not np.allclose(finite_vals, finite_vals[0]):
            violations_uniform += 1
            continue

        # value correctness – compare to EPSS only for h=0 (any horizon would do)
        anchor_date = pd.to_datetime(ds_src.time.values[c_idx, t_idx]).date()
        cve_id = ds_src.cve.values[c_idx]
        stupid_val = ds_stu.pred.values[c_idx, t_idx, h_idx]

        # Lazy-init spark and fetch epss value
        if spark is None:
            spark = get_spark_session()
        epss_val_row = (
            spark.read.parquet(str(epss_parquet))
            .filter((f"cve = '{cve_id}'") & (f"date = DATE('{anchor_date}')"))
            .select("epss")
            .limit(1)
            .collect()
        )
        if not epss_val_row:
            # acceptable – date truly missing → treat as cannot-validate
            continue
        expected_log = np.log(epss_val_row[0]["epss"] + 1e-6)
        if not np.isclose(stupid_val, expected_log, atol=1e-6):
            violations_value += 1

    if spark is not None:
        spark.stop()

    logger.info(f"Uniform-horizon violations: {violations_uniform:,}")
    logger.info(f"Value-mismatch violations : {violations_value:,}")
    if violations_uniform == 0 and violations_value == 0:
        logger.info("Sample checks passed ✔")
    else:
        logger.warning("Sample checks FAILED – inspect violations ❌")

    # ---------------------------------------------------------------------
    # 3. Coverage metric (iterate in strides along CVE axis)
    # ---------------------------------------------------------------------
    logger.info("\n=== 3. COVERAGE ===")
    total_valid = 0
    total_filled = 0
    n_cve = ds_src.sizes["cve"]

    for start in tqdm(range(0, n_cve, coverage_stride), ncols=80, desc="Coverage loop"):
        stop = min(start + coverage_stride, n_cve)
        slc = slice(start, stop)
        eval_slice = eval_mask_np[slc]
        mask_h_slice = mask_h_np[slc]
        valid_slice = eval_slice[:, :, None] & mask_h_slice
        filled_slice = np.isfinite(ds_stu.pred.values[slc])

        total_valid += valid_slice.sum()
        total_filled += (filled_slice & valid_slice).sum()

    coverage_pct = 100 * total_filled / total_valid if total_valid else 0.0
    logger.info(f"Valid positions           : {total_valid:,}")
    logger.info(f"Filled predictions        : {total_filled:,}")
    logger.info(f"Coverage                  : {coverage_pct:.2f} %")

    # ---------------------------------------------------------------------
    # Outcome summary
    # ---------------------------------------------------------------------
    logger.info("\n=== VALIDATION SUMMARY ===")
    ok = (
        violations_uniform == 0
        and violations_value == 0
        and coverage_pct > 99.0  # threshold – adjust if needed
    )
    if ok:
        logger.info("✅ Validation PASSED – stupid_predictions.nc is consistent")
    else:
        logger.warning("❌ Validation FAILED – investigate the issues above")

    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate stupid model predictions NetCDF file")
    parser.add_argument("--source_nc", default="ml_pipeline/stupid_model/predictions_stream_copy.nc", type=Path)
    parser.add_argument("--stupid_nc", default="ml_pipeline/stupid_model/stupid_predictions.nc", type=Path)
    parser.add_argument("--epss_parquet", default="data/epss/processed/epss_processed.parquet", type=Path)
    parser.add_argument("--samples", default=5000, type=int, help="Number of random positions to sample for value checks")
    args = parser.parse_args()

    sys.exit(validate(args.source_nc, args.stupid_nc, args.epss_parquet, sample_points=args.samples)) 