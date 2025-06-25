from pyspark.sql import SparkSession, functions as F
import json, joblib, pyarrow as pa, numpy as np
from pathlib import Path
import pandas as pd, re

import os
import sys

import os
from pathlib import Path

def norm(p):
    # absolute + POSIX‐style
    return Path(os.path.abspath(p)).as_posix()

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
spark = (SparkSession.builder
         .appName("epss-prep")
         .master("local[*]")
         .config("spark.driver.memory", "24g")
        .config("spark.executor.memory", "24g")
        .config("spark.driver.maxResultSize", "10g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())

# Use environment variable for data path, with fallback to default
import os
DEFAULT_DATA_PATH = "data/prod/prod.parquet"
RAW = norm(os.getenv("EPSS_DATA_PATH", DEFAULT_DATA_PATH))
print(f"[INPUT] Using data from: {RAW}")  
OUTDIR  = Path("ml_pipeline/data_prep/work")
OUTDIR.mkdir(parents=True, exist_ok=True)

# 1·1 read
df = spark.read.parquet(RAW)

# 1·1.5 drop unnecessary columns
DROP_COLS = [
    # Large text fields (memory intensive, require NLP)
    "description_all", 
    "description_en", 

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
    
    
    # Mostly empty lists/counts
    "event_types_list",
    "doc_ids",
    
    # Complex technical strings (encoded in scores already)
    "primary_cvss_vec",
    "cve_tags",
    
    # Duplicate timestamps (leaky and redundant)
    "last_modified_date", 
    "snapshot_date",
    
    # Duplicate counts
    "reference_count",  # Same as n_refs
]



# Check which columns actually exist and drop them
existing_cols = set(df.columns)
cols_to_drop = [col for col in DROP_COLS if col in existing_cols]
cols_not_found = [col for col in DROP_COLS if col not in existing_cols]

if cols_to_drop:
    df = df.drop(*cols_to_drop)
    print(f"[DROP] Removed {len(cols_to_drop)} columns: {', '.join(sorted(cols_to_drop))}")
if cols_not_found:
    print(f"[SKIP] {len(cols_not_found)} columns not found: {', '.join(sorted(cols_not_found))}")

print(f"[INFO] Dataset shape after column removal: {df.count():,} rows × {len(df.columns)} columns")

# 1·2 split dates
ts64, ts80 = (df.select(F.unix_timestamp("date").alias("ts"))
                .approxQuantile("ts", [0.64,0.80], 0.0))
VAL_CUT  = pd.to_datetime(ts64, unit="s").date()
TEST_CUT = pd.to_datetime(ts80, unit="s").date()

df = (df
      .withColumn("flag_train", F.col("date") <  F.lit(str(VAL_CUT)))
      .withColumn("flag_val",   (F.col("date") >= F.lit(str(VAL_CUT))) & (F.col("date") < F.lit(str(TEST_CUT))))
      .withColumn("flag_test",  F.col("date") >= F.lit(str(TEST_CUT))))

# 1·3 deltas
for t in ["published_date","last_modified_date","snapshot_date"]:
    if t in df.columns:
        delta = F.datediff(F.col("date"), F.col(t))
        if "last_modified" in t:
            delta = F.when(delta < 0, None).otherwise(delta)
        df = df.withColumn(f"{t}_delta", delta.cast("int"))

# 1·4 EPSS
df = df.withColumn("epss", F.log(F.col("epss") + 1e-6))

# 1·5 vocab + stats
CAT_COLS = ["cwe_id","source_identifier","vuln_status","canon_severity","primary_cvss_sev","prev_event_type"]
vocab = {}
for c in CAT_COLS:
    if c in df.columns:
        keys = (df.filter("flag_train")
                  .select(c).distinct().dropna().orderBy(c)
                  .rdd.map(lambda r: r[0]).collect())
        vocab[c] = {k:i+1 for i,k in enumerate(keys)} | {"UNK":0}

NUMERIC = [f.name for f in df.schema
           if f.name not in {"cve","date","epss"}|set(CAT_COLS)
           and not f.name.startswith("flag_")
           and f.dataType.simpleString() in {"double","float","int","bigint"}]

stats = (df.filter("flag_train")
           .select(*[F.mean(c).alias(f"mean_{c}") for c in NUMERIC],
                   *[F.stddev_pop(c).alias(f"std_{c}") for c in NUMERIC])
           .first()
           .asDict())

μ = {c: stats[f"mean_{c}"] for c in NUMERIC}
σ = {c: stats[f"std_{c}"] or 1.0 for c in NUMERIC}

# 1·6 encode + z-score
for c, mapping in vocab.items():
    if c in df.columns:
        map_expr = F.create_map([F.lit(k) for kv in mapping.items() for k in kv])
        df = df.withColumn(c, F.coalesce(map_expr.getItem(F.col(c)).cast("int"), F.lit(0)))

for c in NUMERIC:
    df = (df.withColumn(f"{c}_missing", F.col(c).isNull().cast("boolean"))
             .withColumn(c, F.when(F.col(c).isNull(), -100.0)
                               .otherwise(((F.col(c)-μ[c])/σ[c]).cast("float"))))

# Boolean columns - includes sparse event features!
BOOL_COLS = lambda df: [c for c in df.columns
                        if c.startswith(("has_", "is_"))
                        or c in (
                            "same_day_multi_source",
                            # Note: Most has_* and is_* columns are kept
                            # despite being sparse - they capture events!
                        )]

# Apply boolean processing with type safety
bool_candidates = BOOL_COLS(df)
dtypes_dict = dict(df.dtypes)
for c in bool_candidates:
    col_type = dtypes_dict[c]
    if col_type == "boolean":
        df = df.withColumn(c, F.coalesce(F.col(c), F.lit(False)).cast("byte"))
        print(f"[BOOL] Processed {c} as boolean → byte")
    else:
        print(f"[SKIP] {c} matches boolean pattern but is {col_type}, not boolean")

# 1·7 write sorted parquet
(df.repartitionByRange("cve")
   .sortWithinPartitions("cve","date")
   .write.mode("overwrite").option("compression","zstd")
   .parquet(str(OUTDIR/"epss_sorted")))

# 1·8 sidecars
(OUTDIR/"vocab.json").write_text(json.dumps(vocab))
joblib.dump({"mean":μ,"std":σ}, OUTDIR/"scaler.pkl")
(OUTDIR/"splits.json").write_text(json.dumps(
        {"val_cut": str(VAL_CUT), "test_cut": str(TEST_CUT)}))

spark.stop()
print("✔ presort done — ready for iterable dataset")
