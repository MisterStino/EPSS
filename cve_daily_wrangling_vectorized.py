#!/usr/bin/env python3
"""
CVE Snapshot Feature Table  –  Spark, NO forward fill
Author : Data Engineering Pipeline
"""

import json, re, hashlib
from pathlib import Path
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import (
    col, to_timestamp, expr, size, length,
    sum as spark_sum, min as spark_min, max as spark_max,
    when, lit
)
from t3_spark.session import get_spark_session
from cvss import CVSS2, CVSS3, CVSS4
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd

# ---------------------------------------------------------------- config ---
MASTER_CSV   = "catalogs_processed/master_cve_timeseries_dedup.csv"   # or Parquet
OUT_PARQUET  = "catalogs_processed/cve_snapshots_irregular.parquet"
MANIFEST     = "catalogs_processed/cve_snapshot_manifest.json"

spark = get_spark_session("cve_snapshot_builder")

# ---------- robust CSV reading --------------------------------------------
csv_options = {
    "header": "true",
    "multiLine": "true",             # Allow multiline fields
    "quote": '"',                    # Standard quote character
    "escape": '"',                   # Standard escape (double quotes)
    "mode": "PERMISSIVE",
    "columnNameOfCorruptRecord": "_corrupt",
    "maxColumns": "50000",           # Increase column limit
    "ignoreLeadingWhiteSpace": "false",
    "ignoreTrailingWhiteSpace": "false"
}

schema = StructType([
    StructField("cve_id", StringType()),
    StructField("reconstruction_timestamp", StringType()),
    StructField("reconstruction_timestamp_raw", StringType()),
    StructField("source_identifier", StringType()),
    StructField("published_date", StringType()),
    StructField("last_modified_date", StringType()),
    StructField("vuln_status", StringType()),
    StructField("cve_tags", StringType()),
    StructField("weakness_count", IntegerType()),
    StructField("reference_count", IntegerType()),
    StructField("configuration_count", IntegerType()),
    StructField("descriptions_json", StringType()),
    StructField("metrics_json", StringType()),
    StructField("weaknesses_json", StringType()),
    StructField("configurations_json", StringType()),
    StructField("references_json", StringType()),
    StructField("primary_cvss_ver", StringType()),
    StructField("primary_cvss_vec", StringType()),
    StructField("primary_cvss_score", DoubleType()),
    StructField("primary_cvss_sev", StringType())
])

snap = (spark.read
            .options(**csv_options)
            .schema(schema)
            .csv(MASTER_CSV))

# Check for corrupt records only if _corrupt column exists
if "_corrupt" in snap.columns:
    corrupt_count = snap.filter("_corrupt IS NOT NULL").count()
    if corrupt_count > 0:
        raise ValueError(f"Corrupt CSV rows detected: {corrupt_count}; regenerate master as Parquet.")
    print(f"✅ CSV parsing validation passed - no corrupt records found")

# ---------- cast timestamps ------------------------------------------------
snap = (snap
        .withColumn("snapshot_date", to_timestamp("reconstruction_timestamp"))
        .withColumn("published_date", to_timestamp("published_date"))
        .withColumn("last_modified_date", to_timestamp("last_modified_date")))

# ---------- pandas-UDF: canonical CVSS ------------------------------------
@pandas_udf("struct<canon_base:double,canon_severity:string,has_v2:int,has_v30:int,has_v31:int,has_v40:int>",
            PandasUDFType.SCALAR)
def cvss_udf(pdf: pd.Series) -> pd.DataFrame:
    out=[]
    for s in pdf:
        row={"canon_base":None,"canon_severity":None,
             "has_v2":0,"has_v30":0,"has_v31":0,"has_v40":0}
        try:
            m=json.loads(s) if isinstance(s,str) else {}
        except Exception:
            m={}
        for tag,key,cls in [("v40","cvssMetricV40",CVSS4),
                            ("v31","cvssMetricV31",CVSS3),
                            ("v30","cvssMetricV30",CVSS3),
                            ("v2","cvssMetricV2",CVSS2)]:
            blk=m.get(key); 
            if not blk: continue
            vec=blk[0]["cvssData"]["vectorString"] if isinstance(blk,list) else None
            if not isinstance(vec,str): continue
            try: parser=cls(vec)
            except Exception: continue
            base_score = parser.base_score if hasattr(parser,'base_score') else parser.scores()[0]
            row["canon_base"] = float(base_score) if base_score is not None else None
            row["canon_severity"] = (
                parser.severity.lower()                                          # CVSS3/4
                if hasattr(parser, "severity") else
                parser.severities()[0].lower() if hasattr(parser, "severities")  # CVSS2
                else None
            )
            row[f"has_{tag}"]=1
            break
        out.append(row)
    return pd.DataFrame(out)

# ---------- pandas-UDF: parse configurations -------------------------------
@pandas_udf("struct<n_cpes:int,n_vendors:int,is_windows:int,is_linux:int,is_android:int,"
            "is_ios:int,is_macos:int,is_hardware:int,is_application:int,is_os:int>",
            PandasUDFType.SCALAR)
def cfg_udf(pdf: pd.Series) -> pd.DataFrame:
    vendor_re=re.compile(r'cpe:2\.3:[aho]:([^:]+):')
    rows=[]
    for s in pdf:
        try: cfg=json.loads(s) if isinstance(s,str) else {}
        except Exception: cfg={}
        
        # Handle all three configuration data structures from pipeline
        # Process configurations maintaining semantic boundaries
        all_nodes = []
        if isinstance(cfg, dict):
            # Standard NVD 2.0: {"nodes": [...]}
            all_nodes = cfg.get("nodes", [])
        elif isinstance(cfg, list):
            # Historical format: [{"nodes": [...]}] - process each config context
            for cfg_item in cfg:
                if isinstance(cfg_item, dict):
                    all_nodes.extend(cfg_item.get("nodes", []))
        # Note: string case already handled by json.loads above
        nodes = all_nodes
        
        if isinstance(nodes,str):
            try:nodes=json.loads(nodes)
            except: nodes=[]
        flags={k:0 for k in ["is_windows","is_linux","is_android","is_ios","is_macos",
                             "is_hardware","is_application","is_os"]}
        n_cpes=0; vendors=set()
        for n in nodes:
            if not isinstance(n, dict):
                continue
            for cm in n.get("cpeMatch",[]):
                if not isinstance(cm, dict):
                    continue
                crit=str(cm.get("criteria","")).lower()
                if not crit: continue
                n_cpes+=1
                if ":windows:" in crit: flags["is_windows"]=1
                if ":linux:" in crit:   flags["is_linux"]=1
                if ":android:" in crit: flags["is_android"]=1
                if ":ios:" in crit:     flags["is_ios"]=1
                if ":macos:" in crit or ":mac_os:" in crit: flags["is_macos"]=1
                if crit.startswith("cpe:2.3:h:"): flags["is_hardware"]=1
                elif crit.startswith("cpe:2.3:a:"): flags["is_application"]=1
                elif crit.startswith("cpe:2.3:o:"): flags["is_os"]=1
                m=vendor_re.match(crit)
                if m: vendors.add(m.group(1))
        rows.append({"n_cpes":n_cpes,"n_vendors":len(vendors),**flags})
    return pd.DataFrame(rows)

snap = (snap
        .withColumn("cvss", cvss_udf("metrics_json"))
        .select("*","cvss.*").drop("cvss")
        .withColumn("cfg", cfg_udf("configurations_json"))
        .select("*","cfg.*").drop("cfg"))

# ---------- simple JSON extraction via Spark SQL ---------------------------
snap = (snap
        .withColumn("cwe_id", expr("get_json_object(weaknesses_json,'$[0].description[0].value')"))
        .withColumn("n_refs", col("reference_count"))  # Use pre-computed count
        # ---------- NEW multilingual description ----------------------------
        .withColumn(
            "description_all",
            expr("concat_ws(' ', transform("
                 "from_json(descriptions_json,'array<struct<lang:string,value:string>>'),"
                 " x -> x.value))"))
        .withColumn("desc_len_all", length("description_all"))
        # ---------- Keep English subset as before ---------------------------
        .withColumn("description_en",
            expr("concat_ws(' ', transform( "
                 " filter(from_json(descriptions_json,'array<struct<lang:string,value:string>>'), "
                 "        x -> x.lang = 'en'), x -> x.value))"))
        .withColumn("desc_len_en", length("description_en")))

# ---------- drop JSON blobs -------------------------------------------------
json_cols=[c for c in snap.columns if c.endswith("_json")]
snap = snap.drop(*json_cols)

# ---------- validations -----------------------------------------------------
dup = snap.groupBy("cve_id","snapshot_date").agg(spark_sum(lit(1)).alias("cnt")).filter("cnt>1").count()
assert dup==0, "duplicate (cve_id,snapshot_date)"

# Check for temporal anomalies (snapshot before publication)
leak = snap.filter(col("snapshot_date") < col("published_date")).count()
if leak > 0:
    print(f"⚠️  Found {leak:,} temporal anomalies (snapshot before publication)")
    print("   This can occur due to:")
    print("   - Pre-disclosure CVE preparation")
    print("   - NVD data quality issues") 
    print("   - Timezone inconsistencies")
    print("   Continuing with processing...")
    
    # Optional: Show examples for debugging
    examples = snap.filter(col("snapshot_date") < col("published_date")).select(
        "cve_id", "snapshot_date", "published_date"
    ).limit(5)
    print("   Examples:")
    examples.show(truncate=False)

from functools import reduce
flag_cols = ["has_v2","has_v30","has_v31","has_v40"]
flag_expr = reduce(lambda a,b: a + b, [col(c) for c in flag_cols])
multi = snap.filter(flag_expr > 1).count()
assert multi==0, "multiple CVSS versions flagged"

# ---------- write output ----------------------------------------------------
snap.write.mode("overwrite").parquet(OUT_PARQUET)

manifest = {
    "rows": snap.count(),
    "unique_cves": snap.select("cve_id").distinct().count(),
    "min_snapshot": str(snap.agg(spark_min("snapshot_date")).first()[0]),
    "max_snapshot": str(snap.agg(spark_max("snapshot_date")).first()[0]),
    "sha256_first10k": hashlib.sha256(
        "\n".join([r.cve_id for r in snap.limit(10000).collect()]).encode()
    ).hexdigest()
}
Path(MANIFEST).parent.mkdir(parents=True, exist_ok=True)
with open(MANIFEST,"w") as f: json.dump(manifest, f, indent=2)

print("✅ Snapshot table written")
print(json.dumps(manifest, indent=2))
spark.stop()
