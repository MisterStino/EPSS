#!/usr/bin/env python3
"""
Merge NVD snapshot features onto dense EPSS daily table.
Result: data/nvd/processed/nvd_processed.parquet

Assumptions
-----------
• EPSS base:  data/epss/processed/epss_processed.parquet
  (columns at least: cve  | date | epss | ...)

• NVD snapshots: data/nvd/raw/cve_snapshots_irregular.parquet
  (columns: cve_id | snapshot_date (Timestamp) | <feature columns>)

Output schema = all EPSS columns + all NVD feature columns.
Row-count identical to EPSS base.
"""

from pathlib import Path
import pyspark.sql.functions as F
from pyspark.sql import Window
from t3_spark.session import get_spark_session

# ---------------------------------------------------------------- paths ----
EPSS_PATH = "data/epss/processed/epss_processed.parquet"
NVD_RAW   = "data/nvd/raw/cve_snapshots_irregular.parquet"
NVD_OUT   = "data/nvd/processed/nvd_processed.parquet"

# ---------------------------------------------------------------- session --
spark = get_spark_session("merge_nvd_to_epss")

# ---------------------------------------------------------------- helpers --
def normalise_base(df):
    return (df
            .withColumn("cve",  F.upper(F.trim(F.col("cve"))))
            .withColumn("date", F.to_date("date")))

# ---------------------------------------------------------------- 1. load --
epss = spark.read.parquet(EPSS_PATH)
epss = normalise_base(epss)
epss.cache()

nvd  = spark.read.parquet(NVD_RAW)
nvd  = (nvd
        .withColumnRenamed("cve_id", "cve")
        .withColumn("cve",  F.upper(F.trim("cve")))
        .withColumn("event_ts",   F.col("snapshot_date"))
        .withColumn("event_date", F.to_date("snapshot_date"))
        )

# ---------------------------------------------------------------- 2. earliest-of-day dedup ---------------
w_dupe = Window.partitionBy("cve", "event_date").orderBy("event_ts")
nvd_day = (nvd
           .withColumn("rn", F.row_number().over(w_dupe))
           .filter("rn = 1")
           .drop("rn", "event_ts")
           .withColumnRenamed("event_date", "date"))

# ---------------------------------------------------------------- 3. back-fill onto EPSS grid ------------
# Identify NVD feature columns (exclude key + JSON leftovers)
key_cols  = {"cve", "date"}
nvd_feats = [c for c in nvd_day.columns if c not in key_cols]

# Join then window-fill last known value
joined = epss.join(nvd_day, on=["cve", "date"], how="left")
w_fill = Window.partitionBy("cve").orderBy("date") \
               .rowsBetween(Window.unboundedPreceding, 0)

for c in nvd_feats:
    joined = joined.withColumn(c, F.last(c, ignorenulls=True).over(w_fill))

# ---------------------------------------------------------------- 4. validation --------------------------
assert joined.count() == epss.count(), "Row-count changed after merge!"

dup_chk = joined.groupBy("cve", "date").count().filter("count > 1").count()
assert dup_chk == 0, "Duplicate (cve,date) keys after merge"

flag_cols = [c for c in nvd_feats if c.startswith("has_v")]
if flag_cols:
    flag_expr = sum(F.col(c) for c in flag_cols)
    assert joined.filter(flag_expr > 1).count() == 0, \
        "CVSS version flags >1 in some rows"

print("✅ Merge validation passed")

# ---------------------------------------------------------------- 5. write --------------------------------
Path(NVD_OUT).parent.mkdir(parents=True, exist_ok=True)
joined.write.mode("overwrite").parquet(NVD_OUT)
print(f"✅ Written {NVD_OUT}  ({joined.count():,} rows)")

spark.stop()
