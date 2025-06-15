# %%
#!/usr/bin/env python
"""
00_build_arrow.py  –  one-off CPU preprocessing for the EPSS LSTM pipeline
Reads raw Parquet, runs Cells 1-4 logic, writes:
  • epss_stage1.arrow  – sorted, all-numeric table
  • vocab.json         – categorical id mapping (train rows only)
  • scaler.pkl         – StandardScaler fit on train numerics

NOTE: This script is structured with # %% cell markers for Jupyter notebook conversion.
Use: jupytext --to notebook 00_build_arrow.py
Then upload the .ipynb to your remote system for execution.

Configuration is hardcoded (no CLI args needed):
- local_execution = True/False (controls paths and sample size)
- Local: sample 100K rows, uses data/full_db/processed/final_full_data.parquet
- Cloud: full dataset, uses data/final_full_data.parquet
"""

import json, joblib
from pathlib import Path

import duckdb, numpy as np, pandas as pd
import pyarrow.feather as feather
from sklearn.preprocessing import StandardScaler

# Define if local or paperspace:
local_execution = True

# Hardware-specific configurations
LOCAL_CONFIG = {
    'sample_size': 100000,  # Row limit for local testing
    'output_dir': 'work',   # Local output directory
}

CLOUD_CONFIG = {
    'sample_size': None,    # No limit for full build
    'output_dir': 'work',   # Cloud output directory
}

# Select configuration based on execution environment
CONFIG = LOCAL_CONFIG if local_execution else CLOUD_CONFIG
print(f"[INFO] Using {'LOCAL' if local_execution else 'CLOUD'} configuration:")
print(f"[INFO] Sample size: {CONFIG['sample_size'] or 'FULL'}, Output: {CONFIG['output_dir']}")

# ───────────────────────────── helpers ──────────────────────────────────────

def transform_epss(arr, eps=1e-6):
    p = np.clip(arr.astype("float64"), eps, 1.0 - eps)
    return np.log(p / (1.0 - p)).astype("float32")   # logit

# %%
# ───────────────────────────── CELL 2: PATH SETUP & CONFIGURATION ──────────────────────────────
# Path configuration (similar to lstm_exp_window_eval.py)
if local_execution:
    PARQUET_PATH = Path("data/full_db/processed/final_full_data.parquet")
else:
    PARQUET_PATH = Path("data/final_full_data.parquet")

OUT_DIR = Path(CONFIG['output_dir'])
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_LIM = f" LIMIT {CONFIG['sample_size']}" if CONFIG['sample_size'] else ""

if not PARQUET_PATH.exists():
    raise FileNotFoundError(f"Parquet path not found: {PARQUET_PATH}")

print(f"[INFO] Input: {PARQUET_PATH}")
print(f"[INFO] Output: {OUT_DIR}")
print(f"[INFO] Sample limit: {CONFIG['sample_size'] or 'None (full dataset)'}")

# Columns to DROP (reduced from 47 to ~28 based on event analysis)
DROP_COLS = [
    # Large text fields (memory intensive, require NLP)
    "description_all", 
    "description_en", 
    "desc_len_all",
    "details_combined", 
    "details_longest", 
    "event_data_merged",
    
    # Metadata/provenance (not predictive features)
    "cve_date_key",
    "original_date", 
    "reconstruction_timestamp",
    "reconstruction_timestamp_raw",
    
    # Truly empty (100% missing - never populated)
    "dominant_event_type",
    "primary_source", 
    
    # Always constant when present (no variance)
    "has_threat",          # Always False
    "has_remediation",     # Always False
    "has_multi_source",    # Always False
    "same_day_multi_source", # Always False
    
    # Low-variance OS/platform flags (mostly zeros)
    "is_windows",
    "is_linux", 
    "is_android",
    "is_ios",
    "is_macos",
    "is_hardware",
    "is_application", 
    "is_os",
    
    # Low-variance technical flags
    "has_v2", 
    "has_v30",
    "has_v31", 
    "has_v40",
    "primary_cvss_ver",
    
    # Mostly empty lists/counts
    "event_types_list",
    "sources_list", 
    "doc_ids",
    "event_type_count",
    "source_count",
    "total_detail_length",
    "weakness_count",
    
    # Complex technical strings (encoded in scores already)
    "primary_cvss_vec",
    "cve_tags",
    
    # Duplicate timestamps (leaky and redundant)
    "date_parsed",      # Duplicate of 'date'
    "last_modified_date", 
    "snapshot_date",
    
    # Duplicate counts
    "reference_count",  # Same as n_refs
]

# Safe timestamps (never in future, safe for training)
TS_SAFE = [
    "published_date",  # CVE publication date - never in future
]

# Potentially leaky timestamps (could be in future at training time)
TS_LEAKY = [
    "last_modified_date",  # NVD modification date
    "snapshot_date",       # Data collection timestamp
]

# Boolean columns - includes sparse event features!
BOOL_COLS = lambda df: [c for c in df.columns
                    if c.startswith(("has_", "is_"))
                    or c in (
                        "same_day_multi_source",
                        # Note: Most has_* and is_* columns are kept
                        # despite being sparse - they capture events!
                    )]

# Categorical columns for embedding
CAT_COLS = [
    "cwe_id",              # 454 weakness types
    "source_identifier",   # 279 reporting organizations  
    "vuln_status",         # 6 NVD status levels
    "canon_severity",      # 5 severity categories
    "primary_cvss_sev",    # 5 CVSS severity labels
    "prev_event_type",     # Event type transitions (sparse but valuable!)
]
SENTINEL = -10.0

# DuckDB loading with schema probing
parquet_glob = str(PARQUET_PATH / "*.parquet") if PARQUET_PATH.is_dir() else str(PARQUET_PATH)
schema_df = duckdb.sql(f"SELECT * FROM parquet_scan('{parquet_glob}') LIMIT 0").df()
drop_real = [c for c in DROP_COLS if c in schema_df.columns]
if drop_real:
    quoted_cols = ', '.join(f'"{c}"' for c in drop_real)
    exclude = f" EXCLUDE ({quoted_cols})"
else:
    exclude = ""
BIG = duckdb.sql(f"SELECT *{exclude} FROM parquet_scan('{parquet_glob}'){SAMPLE_LIM}").df()
BIG['date'] = pd.to_datetime(BIG['date'])

print(f"[INFO] loaded {len(BIG):,} rows × {BIG.shape[1]} columns (DuckDB)")

# %%
# ───────────────────────────── CELL 4: DATA PREPROCESSING ──────────────────────────────
# Dynamic list trimming to only include columns that actually exist
TS_SAFE  = [c for c in TS_SAFE  if c in BIG.columns]
TS_LEAKY = [c for c in TS_LEAKY if c in BIG.columns]
CAT_COLS = [c for c in CAT_COLS if c in BIG.columns]

print(f"[INFO] Available TS_SAFE: {TS_SAFE}")
print(f"[INFO] Available TS_LEAKY: {TS_LEAKY}")
print(f"[INFO] Available CAT_COLS: {CAT_COLS}")

# Timestamp preprocessing
for col in TS_SAFE:
    BIG[f"{col}_delta"] = (BIG['date'] - pd.to_datetime(BIG[col])).dt.days.astype("float32")
for col in TS_LEAKY:
    d = (BIG['date'] - pd.to_datetime(BIG[col])).dt.days.astype("float32")
    d[d < 0] = np.nan
    BIG[f"{col}_delta"] = d
BIG.drop(columns=TS_SAFE + TS_LEAKY, inplace=True)

# Calendar-based split
days = np.sort(BIG['date'].unique())
VAL_CUT, TEST_CUT = pd.to_datetime(days[int(.64*len(days))]), pd.to_datetime(days[int(.80*len(days))])
BIG['flag_train'] = (BIG['date'] <  VAL_CUT).astype("uint8")
BIG['flag_val']   = ((BIG['date'] >= VAL_CUT) & (BIG['date'] < TEST_CUT)).astype("uint8")
BIG['flag_test']  = (BIG['date'] >= TEST_CUT).astype("uint8")

print(f"[INFO] VAL from {VAL_CUT.date()} | TEST from {TEST_CUT.date()}")

# %%
# ───────────────────────────── CELL 5: FEATURE ENGINEERING ──────────────────────────────
# Target transform
BIG['epss'] = transform_epss(BIG['epss'].values)

# Boolean processing
for col in BOOL_COLS(BIG):
    if pd.api.types.is_bool_dtype(BIG[col]):
        BIG[col] = BIG[col].fillna(False).astype("uint8")
    else:
        BIG[col] = BIG[col].fillna(0).astype("uint8")

# Categorical processing (vocab from *true* training rows only)
train_mask = BIG['flag_train'] == 1
VOCAB = {}
for col in CAT_COLS:
    cats = [c for c in BIG.loc[train_mask, col].dropna().unique() if c != "UNK"]
    mapping = {c:i+1 for i,c in enumerate(sorted(cats))}
    mapping["UNK"] = 0
    VOCAB[col] = mapping
    BIG[col] = BIG[col].map(mapping).fillna(0).astype("int32")

# Numeric processing  
NUM_COLS = [c for c,t in BIG.dtypes.items() 
            if (pd.api.types.is_numeric_dtype(t) and 
                c not in BOOL_COLS(BIG)+['flag_train','flag_val','flag_test'])]

# CRITICAL FIX: Separate truly numeric columns from categorical columns
# Exclude categorical columns from standardization (embeddings need integer indices!)
NUMERIC_ONLY_COLS = [c for c in NUM_COLS if c not in CAT_COLS]

scaler = StandardScaler().fit(BIG.loc[train_mask, NUMERIC_ONLY_COLS])
numeric_data = scaler.transform(BIG[NUMERIC_ONLY_COLS]).astype("float32")
BIG[NUMERIC_ONLY_COLS] = numeric_data

# Validation: Ensure categorical columns remained integers after transform
for col in CAT_COLS:
    assert pd.api.types.is_integer_dtype(BIG[col]), f"{col} became non-integer!"
for col in NUMERIC_ONLY_COLS:
    miss = BIG[col].isna()
    BIG[f"{col}_missing"] = miss.astype("uint8")
    BIG.loc[miss, col] = SENTINEL

print(f"[INFO] Processed {len(NUMERIC_ONLY_COLS)} truly numeric columns (standardized)")
print(f"[INFO] Processed {len(BOOL_COLS(BIG))} boolean columns")
print(f"[INFO] Processed {len(CAT_COLS)} categorical columns (kept as integers)")

# %%
# ───────────────────────────── CELL 6: OUTPUT WRITING ──────────────────────────────
# Sort and write outputs
BIG.sort_values(['cve','date']).reset_index(drop=True, inplace=True)
feather.write_feather(BIG, OUT_DIR/'epss_stage1.arrow', compression='lz4')
json.dump(VOCAB, open(OUT_DIR/'vocab.json','w'))
joblib.dump(scaler, OUT_DIR/'scaler.pkl')

print(f"[OK] wrote {len(BIG):,} rows → {OUT_DIR}")
print(f"✓ epss_stage1.arrow - preprocessed data")
print(f"✓ vocab.json - categorical mappings")  
print(f"✓ scaler.pkl - numeric standardization")

# %%
# ───────────────────────────── CELL 7: INTEGRITY CHECK ──────────────────────────────
import pyarrow as pa
import pyarrow.feather as feather

# Quick integrity check of output artifacts
tab = feather.read_table("work/epss_stage1.arrow")
assert tab.schema.field('epss').type == pa.float32()
vocab = json.load(open("work/vocab.json"))
scaler = joblib.load("work/scaler.pkl")
assert set(vocab) == set(CAT_COLS)    # same cat cols
assert scaler.mean_.shape[0] == len(NUMERIC_ONLY_COLS)

# Enhanced validation: Check categorical columns in output file
df = feather.read_table('work/epss_stage1.arrow').to_pandas()
for c in ['cwe_id','source_identifier']:
    print(f"{c}: {df[c].dtype}, sample values: {df[c].unique()[:5]}")
    
print("artefacts OK")
