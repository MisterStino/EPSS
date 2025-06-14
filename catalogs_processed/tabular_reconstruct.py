#!/usr/bin/env python3
"""
Builds a lossless, long-format master CVE time-series table.

Usage
-----
python -u tabular_reconstruct.py
"""
import csv, gzip, json, sys, logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

SCALARS = {
    "source_identifier":  "sourceIdentifier",
    "published_date":     "published",
    "last_modified_date": "lastModified",
    "vuln_status":        "vulnStatus",
    "cve_tags":           "cveTags",
}

def extract_primary_cvss(metrics: dict) -> dict:
    """Extract primary CVSS data, handling mixed types from reconstruction process."""
    order = ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2")
    for key in order:
        metric_raw = metrics.get(key)
        if not metric_raw:
            continue
            
        # Handle different data types from reconstruction
        metric_data = None
        
        if isinstance(metric_raw, list) and len(metric_raw) > 0:
            # Standard NVD format: [{"cvssData": {...}}]
            metric_data = metric_raw[0]
        elif isinstance(metric_raw, str):
            # Reconstructed from History API as JSON string
            try:
                parsed = json.loads(metric_raw)
                if isinstance(parsed, list) and len(parsed) > 0:
                    metric_data = parsed[0]
                elif isinstance(parsed, dict):
                    metric_data = parsed
            except (json.JSONDecodeError, TypeError):
                continue
        elif isinstance(metric_raw, dict):
            # Direct object (malformed reconstruction)
            metric_data = metric_raw
        
        # Extract CVSS data if we have valid metric_data
        if metric_data and isinstance(metric_data, dict):
            cvss_data = metric_data.get("cvssData", {})
            if cvss_data:
                return {
                    "primary_cvss_ver":   cvss_data.get("version"),
                    "primary_cvss_vec":   cvss_data.get("vectorString"),
                    "primary_cvss_score": cvss_data.get("baseScore"),
                    "primary_cvss_sev":   cvss_data.get("baseSeverity"),
                }
    
    return {k: None for k in (
        "primary_cvss_ver","primary_cvss_vec",
        "primary_cvss_score","primary_cvss_sev")}

def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)

def main(recon_dir, snapshot_date, out_path):
    # Hardcoded configuration

    
    if not recon_dir.is_dir():
        sys.exit(f"No such directory: {recon_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    opener  = gzip.open if out_path.suffix.endswith("gz") else open
    rows_written = 0

    with opener(out_path, "wt", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = (
            ["cve_id","reconstruction_timestamp","reconstruction_timestamp_raw"] +
            list(SCALARS.keys()) +
            ["weakness_count","reference_count","configuration_count"] +
            ["descriptions_json","metrics_json","weaknesses_json",
             "configurations_json","references_json"] +
            ["primary_cvss_ver","primary_cvss_vec",
             "primary_cvss_score","primary_cvss_sev"]
        )
        writer.writerow(header)

        jsonl_files = sorted(recon_dir.glob("CVE-*.jsonl"))
        
        for jf in tqdm(jsonl_files, desc="Processing CVE files", unit="file"):
            for obj in iter_jsonl(jf):
                cve_id = obj["id"]
                ts_raw = obj.get("reconstruction_timestamp")
                
                # FIXED: Handle missing timestamps more intelligently
                if not ts_raw:
                    # Try to use lastModified from the CVE data itself
                    ts_raw = obj.get("lastModified") or f"{snapshot_date}T00:00:00.000"
                    logging.warning("Missing reconstruction_timestamp for %s, using: %s", cve_id, ts_raw)
                
                # ISO to unix-safe sort key
                ts_iso = datetime.fromisoformat(ts_raw.rstrip("Z")).isoformat(timespec="milliseconds")

                row  = [cve_id, ts_iso, ts_raw]
                # scalars
                for csv_col, jpath in SCALARS.items():
                    row.append(obj.get(jpath))
                # counts - handle mixed types robustly
                weaknesses_raw = obj.get("weaknesses", [])
                if isinstance(weaknesses_raw, list):
                    weakness_count = len(weaknesses_raw)
                elif isinstance(weaknesses_raw, str):
                    try:
                        parsed = json.loads(weaknesses_raw)
                        weakness_count = len(parsed) if isinstance(parsed, list) else 0
                    except (json.JSONDecodeError, TypeError):
                        weakness_count = 0
                else:
                    weakness_count = 0
                row.append(weakness_count)
                
                references_raw = obj.get("references", [])
                if isinstance(references_raw, list):
                    reference_count = len(references_raw)
                elif isinstance(references_raw, str):
                    try:
                        parsed = json.loads(references_raw)
                        reference_count = len(parsed) if isinstance(parsed, list) else 0
                    except (json.JSONDecodeError, TypeError):
                        reference_count = 0
                else:
                    reference_count = 0
                row.append(reference_count)
                # configuration_count: handle multiple data types from reconstruction
                cfg_raw = obj.get("configurations", {})
                cfg_cnt = 0
                
                if isinstance(cfg_raw, dict):
                    # Standard NVD 2.0 format: {"nodes": [...]}
                    nodes = cfg_raw.get("nodes", [])
                    for n in nodes:
                        cfg_cnt += len(n.get("cpeMatch", []))
                elif isinstance(cfg_raw, list):
                    # Historical format: array of configuration objects
                    for cfg_item in cfg_raw:
                        if isinstance(cfg_item, dict):
                            nodes = cfg_item.get("nodes", [])
                            for n in nodes:
                                cfg_cnt += len(n.get("cpeMatch", []))
                elif isinstance(cfg_raw, str):
                    # Reconstructed from history API as JSON string
                    try:
                        cfg_parsed = json.loads(cfg_raw)
                        if isinstance(cfg_parsed, dict):
                            nodes = cfg_parsed.get("nodes", [])
                            for n in nodes:
                                cfg_cnt += len(n.get("cpeMatch", []))
                        elif isinstance(cfg_parsed, list):
                            for cfg_item in cfg_parsed:
                                if isinstance(cfg_item, dict):
                                    nodes = cfg_item.get("nodes", [])
                                    for n in nodes:
                                        cfg_cnt += len(n.get("cpeMatch", []))
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        # Malformed JSON string, skip counting
                        pass
                
                row.append(cfg_cnt)
                # json blobs - handle mixed types robustly
                row.append(json.dumps(obj.get("descriptions", []),   ensure_ascii=False))
                
                # Handle metrics JSON serialization for mixed types
                metrics_for_json = obj.get("metrics", {})
                if isinstance(metrics_for_json, str):
                    # Already a JSON string, use as-is
                    row.append(metrics_for_json)
                else:
                    # Dict or other - serialize normally
                    row.append(json.dumps(metrics_for_json, ensure_ascii=False))
                # Handle other JSON fields with potential mixed types
                weaknesses_for_json = obj.get("weaknesses", [])
                if isinstance(weaknesses_for_json, str):
                    row.append(weaknesses_for_json)
                else:
                    row.append(json.dumps(weaknesses_for_json, ensure_ascii=False))
                # Handle configurations JSON serialization for all types
                cfg_for_json = obj.get("configurations", {})
                if isinstance(cfg_for_json, str):
                    # Already a JSON string, use as-is
                    row.append(cfg_for_json)
                else:
                    # Dict, list, or other - serialize normally
                    row.append(json.dumps(cfg_for_json, ensure_ascii=False))
                references_for_json = obj.get("references", [])
                if isinstance(references_for_json, str):
                    row.append(references_for_json)
                else:
                    row.append(json.dumps(references_for_json, ensure_ascii=False))
                # primary CVSS - ensure metrics is always a dict
                metrics_for_cvss = obj.get("metrics", {})
                if isinstance(metrics_for_cvss, str):
                    try:
                        metrics_for_cvss = json.loads(metrics_for_cvss)
                    except (json.JSONDecodeError, TypeError):
                        metrics_for_cvss = {}
                if not isinstance(metrics_for_cvss, dict):
                    metrics_for_cvss = {}
                row.extend(extract_primary_cvss(metrics_for_cvss).values())

                writer.writerow(row)
                rows_written += 1
                
                # Update progress every 1000 rows
                if rows_written % 1000 == 0:
                    tqdm.write(f"Processed {rows_written:,} rows...")

    print(f"✅ finished. rows_written={rows_written:,} → {out_path}")

if __name__ == "__main__":
    import datetime as dt
    
    recon_dir = Path("catalogs_processed/reconstructions")
    snapshot_date = dt.date.today().isoformat()  # Use current date dynamically
    out_path = Path("catalogs_processed/master_cve_timeseries_dedup.csv")
    main(recon_dir, snapshot_date, out_path)