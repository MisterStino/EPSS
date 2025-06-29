#!/usr/bin/env python
# ╔════════════════════════════════════════════════════════════════════╗
# ║ 00_build_arrow.py — fast Arrow writer for the *Spark-sorted* EPSS ║
# ║                                                                    ║
# ║  Prereqs (already produced by Spark job):                           ║
# ║    • work/epss_sorted/       – Parquet folder, sorted (cve,date)    ║
# ║    • work/vocab.json         – categoricals → id                    ║
# ║    • work/scaler.pkl         – μ, σ for numeric cols                ║
# ║                                                                    ║
# ║  This script now ONLY:                                              ║
# ║    • streams those Parquets once (no transforms, no sort)           ║
# ║    • casts booleans → uint8 (optional)                              ║
# ║    • writes work/epss_stage1.arrow                                  ║
# ╚════════════════════════════════════════════════════════════════════╝

import os, sys, json, joblib
from pathlib import Path

import duckdb, pandas as pd, pyarrow as pa, pyarrow.ipc as ipc

# ───────────────────────── user paths ─────────────────────────
# Folder that holds epss_sorted, vocab.json, scaler.pkl
ROOT = Path("ml_pipeline/data_prep/work")
PARQUET_DIR = ROOT / "epss_sorted"        # <- Spark output
ARROW_OUT   = ROOT / "epss_stage1.arrow"  # <- what the DataLoader mmap's

# Leave SAMPLE_ROWS = None for full build
SAMPLE_ROWS: int | None = None            # e.g. 100_000 for a dev run
BATCH_ROWS  : int       = 1_000_000       # Arrow record-batch size

# ───────────────────────── sanity checks ─────────────────────
if not PARQUET_DIR.exists():
    sys.exit(f"[ERR] {PARQUET_DIR} not found – run Spark job first")

for sidecar in ("vocab.json", "scaler.pkl"):
    if not (ROOT / sidecar).exists():
        sys.exit(f"[ERR] {sidecar} missing in {ROOT}")

print(f"[IO] streaming from {PARQUET_DIR} → {ARROW_OUT}")

if ARROW_OUT.exists():
    ARROW_OUT.unlink()

# ───────────────────────── DuckDB stream  → Arrow  ────────────
con = duckdb.connect(database=":memory:")
lim = f" LIMIT {SAMPLE_ROWS}" if SAMPLE_ROWS else ""

stream_sql = f"""
    SELECT *
    FROM parquet_scan('{PARQUET_DIR.as_posix()}/*.parquet')
    {lim}
"""

sink   = pa.OSFile(str(ARROW_OUT), "wb")
writer = None
rows_written = 0

reader = (con.execute(stream_sql)
             .fetch_record_batch(rows_per_batch=BATCH_ROWS))

for rb in reader:
    df = rb.to_pandas(use_threads=False)

    # --- cast data types for PyTorch compatibility ----
    for col, dt in df.dtypes.items():
        if dt == "bool":
            df[col] = df[col].astype("uint8")
        elif col in ["epss_target", "epss_input"] and dt == "float64":
            df[col] = df[col].astype("float32")

    tbl = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pa.ipc.new_file(sink, tbl.schema)
    writer.write(tbl)

    rows_written += len(df)
    if rows_written % 1_000_000 < len(df):
        print(f"  {rows_written:>12,} rows")

writer.close(); sink.close()
con.close()
print(f"✓ {rows_written:,} rows  →  {ARROW_OUT}")

# ───────────────────────── integrity check ───────────────────
tab = ipc.open_file(pa.memory_map(str(ARROW_OUT), "r")).read_all()
# Check that both EPSS columns are float32
assert tab.schema.field("epss_target").type == pa.float32()
assert tab.schema.field("epss_input").type == pa.float32()
vocab = json.load((ROOT / "vocab.json").open())
scaler = joblib.load(ROOT / "scaler.pkl")
assert len(vocab) == len(vocab)            # dummy: just to prove load works
assert scaler["mean"].keys() == scaler["std"].keys()
print("✓ schema, vocab, scaler  OK – ready for training")

# ───────────────────────── copy files for LSTM step ──────────────────
# FIXED: Standardize to ml_pipeline/work/ for cross-platform consistency
import shutil

LSTM_WORK_DIR = Path("ml_pipeline/work")
LSTM_WORK_DIR.mkdir(parents=True, exist_ok=True)

# Copy necessary files for LSTM step - now consistent with LSTM expectations
shutil.copy2(ROOT / "vocab.json", LSTM_WORK_DIR / "vocab.json")
shutil.copy2(ROOT / "scaler.pkl", LSTM_WORK_DIR / "scaler.pkl") 
shutil.copy2(ROOT / "epss_stage1.arrow", LSTM_WORK_DIR / "epss_stage1.arrow")

print(f"✓ Files copied to {LSTM_WORK_DIR} for LSTM step")
