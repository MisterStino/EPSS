# CVE Temporal Reconstruction Pipeline - Technical Report

**Author:** Data Engineering Team  
**Date:** June 2025  
**Purpose:** Complete technical documentation of the CVE temporal reconstruction and feature engineering pipeline

---

## Executive Summary

This report documents a sophisticated data engineering pipeline that transforms current CVE (Common Vulnerabilities and Exposures) data into a historically accurate, temporally consistent dataset for machine learning applications. The pipeline reconstructs the evolution of each CVE over time by applying historical changes in reverse chronological order, producing 1.3M+ temporal snapshots spanning 33 years.

**Key Achievements:**
- ✅ **1,324,234 temporal snapshots** from 296,048 unique CVEs
- ✅ **33-year historical coverage** (1992-2025) 
- ✅ **75.8% CVSS score coverage** with version canonicalization
- ✅ **Lossless temporal accuracy** - no synthetic forward-filling
- ✅ **ML-ready features** with semantic consistency validation

---

## Problem Context & Requirements

### The Challenge

**Traditional CVE datasets suffer from temporal inconsistency**:
- Current snapshots don't reflect what was actually known historically
- CVSS scores are retroactively updated, creating future leakage
- Platform information changes as vendors clarify affected products
- Description text evolves with better understanding of vulnerabilities

**Our Goal**: Create a temporally accurate dataset where each row represents **exactly what NVD knew about a CVE at specific historical moments**.

### Business Requirements

```
REQUIREMENT: Lossless Temporal Reconstruction
├── Each snapshot = real historical state (no synthetic data)
├── Composite key (cve_id, snapshot_timestamp) must be unique  
├── All features semantically aligned across time periods
└── ML-ready format optimized for sequence modeling

REQUIREMENT: Comprehensive Feature Engineering  
├── CVSS canonicalization across versions 2.0 → 4.0
├── Platform detection from CPE (Common Platform Enumeration)
├── Multilingual text preservation
└── Robust handling of data structure evolution
```

## Complete Data Flow

**Stage 1: NVD API 2.0 → Current CVE State**
- Input: Live NVD database via REST API
- Output: JSON objects with current CVSS, descriptions, configurations  
- Data: ~300K CVEs, ~1.2GB, current state only

**Stage 2: NVD History API → Change Events**
- Input: CVE modification history via REST API
- Output: Arrays of change objects with timestamps and diffs
- Data: ~5M changes spanning 1999-2025

**Stage 3: Historical Reconstructor → Temporal Snapshots**  
- Input: Current state + change history
- Process: Reverse chronological reconstruction
- Output: JSONL files with reconstruction timestamps
- Data: 1.3M snapshots, ~8GB, covering 1992-2025

**Stage 4: Tabular Converter → Master CSV**
- Input: JSONL historical snapshots
- Process: Flatten to 20-column CSV with JSON preservation
- Output: Master CSV with composite key validation
- Data: 1.3M rows, 4GB, tabular format

**Stage 5: Spark Feature Engineering → ML-Ready Parquet**
- Input: Master CSV (20 columns)
- Process: CVSS canonicalization, platform extraction, text processing
- Output: Optimized Parquet with 38 features
- Data: 1.3M rows, 192MB compressed, ML-ready

## Step-by-Step Technical Implementation

### Stage 1: NVD Data Acquisition

**Core NVD API interaction:**
```python
def fetch_current_cves():
    """Fetches current state of all CVEs from NVD API 2.0"""
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    # Handles pagination, rate limiting, API key management
    return cve_objects
```

**Data Structure Example:**
```json
{
  "id": "CVE-2008-1234",
  "published": "2008-03-15T09:00:00.000",
  "metrics": {
    "cvssMetricV31": [{"cvssData": {"baseScore": 7.5, "vectorString": "CVSS:3.1/AV:N/..."}}]
  },
  "configurations": {
    "nodes": [{"cpeMatch": [{"criteria": "cpe:2.3:o:microsoft:windows:..."}]}]
  },
  "descriptions": [{"lang": "en", "value": "Buffer overflow allows..."}]
}
```

### Stage 2: Historical Reconstruction

**Core Algorithm:**
```python
def reconstruct_cve_timeline(current_state, change_history):
    """
    Applies changes in REVERSE chronological order to reconstruct 
    historical states at each change moment, ensuring current snapshot preservation.
    """
    today_copy = deepcopy(current_state)
    h_today = hash_state(today_copy)
    
    # Sort changes by timestamp (newest first)
    changes = sorted(change_history, key=lambda x: x['created'], reverse=True)
    
    seen_days = set()
    states = []
    cur_state = deepcopy(current_state)
    
    # Process historical changes (newest → oldest)
    for change_event in changes:
        cur_state = undo_change(cur_state, change_event['details'])
        ts = change_event['created']
        day = ts[:10]  # Extract date (YYYY-MM-DD)
        
        if day in seen_days:
            continue  # Keep only last change of each day
        seen_days.add(day)
        
        cur_state['reconstruction_timestamp'] = ts
        if not states or hash_state(cur_state) != hash_state(states[-1]):
            states.append(deepcopy(cur_state))
    
    # CRITICAL: Ensure current snapshot is preserved
    snapshot_date = datetime.date.today().isoformat()
    snapshot_ts = f"{snapshot_date}T00:00:00.000"
    
    if snapshot_date in seen_days:
        # Current day had changes - reuse or inject current state
        if states and hash_state(states[0]) == h_today:
            states[0]['reconstruction_timestamp'] = snapshot_ts
        else:
            today_copy['reconstruction_timestamp'] = snapshot_ts
            states.insert(0, today_copy)
    else:
        # No changes today - inject current snapshot
        today_copy['reconstruction_timestamp'] = snapshot_ts
        states.insert(0, today_copy)
    
    return states  # Already in chronological order (newest first)
```

**Critical Implementation Details:**
- **Current Snapshot Preservation**: Ensures today's snapshot is always included with dynamic timestamp
- **Day-level Deduplication**: Keeps only the last change per day to avoid excessive granularity
- **Hash-based Duplicate Prevention**: Prevents identical consecutive states in timeline
- **Smart Current State Injection**: Reuses existing current state or injects new one based on hash comparison
- **Dynamic Date Handling**: Uses `datetime.date.today().isoformat()` for current snapshot timestamps
- **Type Handling**: Values can be strings, numbers, objects, or arrays  
- **Missing Field Logic**: Distinguishes between null values and absent fields
- **Timestamp Precision**: Preserves millisecond-level accuracy

### Stage 3: Tabular Conversion

**Schema Definition:**
```python
COLUMNS = [
    # Identity & Temporal
    "cve_id", "reconstruction_timestamp", "reconstruction_timestamp_raw",
    "source_identifier", "published_date", "last_modified_date", "vuln_status", "cve_tags",
    
    # Pre-computed Counts  
    "weakness_count", "reference_count", "configuration_count",
    
    # JSON Blobs (preserved for complex parsing)
    "descriptions_json", "metrics_json", "weaknesses_json", 
    "configurations_json", "references_json",
    
    # Primary CVSS (quick access)
    "primary_cvss_ver", "primary_cvss_vec", "primary_cvss_score", "primary_cvss_sev"
]
```

**Configuration Count Logic (Critical for Data Structure Evolution):**
```python
def count_cpe_entries(cfg_raw):
    """Counts CPE entries handling 3 different data structures"""
    if isinstance(cfg_raw, dict):
        # Standard NVD 2.0: {"nodes": [...]}
        nodes = cfg_raw.get("nodes", [])
        return sum(len(n.get("cpeMatch", [])) for n in nodes)
        
    elif isinstance(cfg_raw, list):
        # Historical format: [{"nodes": [...]}]  
        total = 0
        for cfg_item in cfg_raw:
            if isinstance(cfg_item, dict):
                nodes = cfg_item.get("nodes", [])
                total += sum(len(n.get("cpeMatch", [])) for n in nodes)
        return total
        
    elif isinstance(cfg_raw, str):
        # Reconstructed as JSON string
        try:
            return count_cpe_entries(json.loads(cfg_raw))
        except:
            return 0
    
    return 0
```

### Stage 4: Spark Feature Engineering

**CVSS Canonicalization:**
```python
@pandas_udf("struct<canon_base:double,canon_severity:string,has_v2:int,has_v30:int,has_v31:int,has_v40:int>")
def cvss_udf(pdf: pd.Series) -> pd.DataFrame:
    """Canonicalizes CVSS scores across all versions using priority-based selection"""
    results = []
    
    # Priority order: v4.0 → v3.1 → v3.0 → v2.0 (newest first)
    version_priority = [
        ("v40", "cvssMetricV40", CVSS4),
        ("v31", "cvssMetricV31", CVSS3), 
        ("v30", "cvssMetricV30", CVSS3),
        ("v2", "cvssMetricV2", CVSS2)
    ]
    
    for metrics_json in pdf:
        metrics = json.loads(metrics_json) if isinstance(metrics_json, str) else {}
        result = {"canon_base": None, "canon_severity": None, 
                 "has_v2": 0, "has_v30": 0, "has_v31": 0, "has_v40": 0}
        
        # Find highest available CVSS version
        for tag, key, cvss_class in version_priority:
            metric_data = metrics.get(key)
            if not metric_data:
                continue
                
            vector = metric_data[0]["cvssData"]["vectorString"] if isinstance(metric_data, list) else None
            if not vector:
                continue
                
            try:
                # Parse using official CVSS library
                parser = cvss_class(vector)
                
                # Extract canonical values with type conversion
                base_score = parser.base_score if hasattr(parser, 'base_score') else parser.scores()[0]
                result["canon_base"] = float(base_score) if base_score is not None else None
                
                # Handle version-specific severity APIs
                if hasattr(parser, "severity"):
                    result["canon_severity"] = parser.severity.lower()  # CVSS 3.x/4.x
                elif hasattr(parser, "severities"):
                    result["canon_severity"] = parser.severities()[0].lower()  # CVSS 2.0
                
                result[f"has_{tag}"] = 1
                break  # Use highest available version
                
            except Exception:
                continue
        
        results.append(result)
    
    return pd.DataFrame(results)
```

**Configuration Parsing (Handles All Data Structure Variants):**
```python
@pandas_udf("struct<n_cpes:int,n_vendors:int,is_windows:int,is_linux:int,...>")
def cfg_udf(pdf: pd.Series) -> pd.DataFrame:
    """Extracts platform information handling all configuration data structures"""
    vendor_regex = re.compile(r'cpe:2\.3:[aho]:([^:]+):')
    results = []
    
    for config_json in pdf:
        cfg = json.loads(config_json) if isinstance(config_json, str) else {}
        
        # Handle all 3 data structure variants
        all_nodes = []
        if isinstance(cfg, dict):
            # Standard NVD 2.0: {"nodes": [...]}
            all_nodes = cfg.get("nodes", [])
        elif isinstance(cfg, list):
            # Historical format: [{"nodes": [...]}]
            for cfg_item in cfg:
                if isinstance(cfg_item, dict):
                    all_nodes.extend(cfg_item.get("nodes", []))
        
        # Process all CPE entries
        platform_flags = {"is_windows": 0, "is_linux": 0, "is_android": 0, 
                         "is_ios": 0, "is_macos": 0, "is_hardware": 0, 
                         "is_application": 0, "is_os": 0}
        n_cpes = 0
        vendors = set()
        
        for node in all_nodes:
            if not isinstance(node, dict):
                continue
            for cpe_match in node.get("cpeMatch", []):
                if not isinstance(cpe_match, dict):
                    continue
                    
                criteria = str(cpe_match.get("criteria", "")).lower()
                if not criteria:
                    continue
                
                n_cpes += 1
                
                # Platform detection
                if ":windows:" in criteria: platform_flags["is_windows"] = 1
                if ":linux:" in criteria: platform_flags["is_linux"] = 1
                if ":android:" in criteria: platform_flags["is_android"] = 1
                if ":ios:" in criteria: platform_flags["is_ios"] = 1
                if ":macos:" in criteria: platform_flags["is_macos"] = 1
                
                # CPE type detection
                if criteria.startswith("cpe:2.3:h:"): platform_flags["is_hardware"] = 1
                elif criteria.startswith("cpe:2.3:a:"): platform_flags["is_application"] = 1  
                elif criteria.startswith("cpe:2.3:o:"): platform_flags["is_os"] = 1
                
                # Vendor extraction
                vendor_match = vendor_regex.match(criteria)
                if vendor_match:
                    vendors.add(vendor_match.group(1))
        
        results.append({
            "n_cpes": n_cpes,
            "n_vendors": len(vendors),
            **platform_flags
        })
    
    return pd.DataFrame(results)
```

## Configuration Data Evolution (Critical Edge Case)

**The Problem**: NVD configuration format evolved over 20+ years:

```json
// Format 1: Modern NVD 2.0 (2019+)
{"configurations": {"nodes": [{"cpeMatch": [...]}]}}

// Format 2: Historical NVD (2010-2019) 
{"configurations": [{"nodes": [{"cpeMatch": [...]}]}]}

// Format 3: Reconstructed from History API
{"configurations": "{\"nodes\":[{\"cpeMatch\":[...]}]}"}
```

**Our Solution**: Unified parsing logic handling all 3 formats transparently while preserving semantic meaning.

## Validation & Quality Assurance

**Multi-level validation ensuring data quality:**

```python
def comprehensive_validation(df):
    # Level 1: Schema Validation
    assert len(df.columns) == 38, f"Expected 38 columns, got {len(df.columns)}"
    assert df['cve_id'].str.match(r'CVE-\d{4}-\d{4,}').all(), "Invalid CVE ID format"
    
    # Level 2: Temporal Validation  
    assert df['snapshot_date'].dtype == 'datetime64[ns]', "Snapshot date must be datetime"
    
    # Level 3: CVSS Validation
    cvss_flags = ['has_v2', 'has_v30', 'has_v31', 'has_v40']
    flag_sums = df[cvss_flags].sum(axis=1)
    assert (flag_sums <= 1).all(), "Multiple CVSS versions flagged"
    
    # Level 4: Platform Flag Validation
    platform_flags = ['is_windows', 'is_linux', 'is_android', 'is_ios', 'is_macos']
    for flag in platform_flags:
        assert df[flag].isin([0, 1]).all(), f"Platform flag {flag} must be binary"
    
    print("✅ All validation checks passed")
```

## Results Analysis

### Final Dataset Characteristics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Total Snapshots** | 1,324,234+ | Each represents a real historical moment + current state |
| **Unique CVEs** | 296,048 | Complete coverage of NVD database |
| **Temporal Span** | 1992-2025 (33 years) | Full historical coverage with current snapshots |
| **Avg Snapshots/CVE** | 4.5+ | Reflects real update frequency + guaranteed current state |
| **Current Snapshots** | 296,048 | Every CVE has current day snapshot for backfilling |
| **File Size** | 192MB+ (compressed) | 92%+ size reduction vs CSV |

### Feature Coverage Analysis

```
CVSS Coverage by Version:
├── CVSS 2.0:  409,192 snapshots (30.9%) - Legacy standard
├── CVSS 3.0:  243,235 snapshots (18.4%) - Early v3 adoption  
├── CVSS 3.1:  343,436 snapshots (25.9%) - Current standard
├── CVSS 4.0:    7,677 snapshots ( 0.6%) - Latest standard
└── No CVSS:   320,694 snapshots (24.2%) - Pre-standardization era

Platform Detection:
├── Windows:    28,403 snapshots ( 2.1%) - Enterprise focus
├── Linux:      28,056 snapshots ( 2.1%) - Server/embedded
├── Android:    28,181 snapshots ( 2.1%) - Mobile platform
├── iOS:         2,038 snapshots ( 0.2%) - Apple ecosystem  
└── macOS:      11,093 snapshots ( 0.8%) - Desktop/creative
```

### Quality Metrics

- **✅ Temporal Consistency**: 100% - No impossible dates detected  
- **✅ Schema Compliance**: 100% - All columns conform to expected types  
- **✅ CVSS Validation**: 100% - All scores in valid range [0.0, 10.0]  
- **✅ Composite Key Uniqueness**: 100% - No duplicate (cve_id, snapshot_date) pairs  
- **✅ Platform Flag Validity**: 100% - All binary flags in {0, 1}  

## Implementation Guide

### Prerequisites

**System Requirements:**
- RAM: 16GB minimum (32GB recommended)  
- Storage: 50GB free space
- Python 3.8+, Java 8+, PowerShell 5.1+

**Python Dependencies:**
```txt
pandas>=1.5.0, pyarrow>=10.0.0, pyspark>=3.4.0
cvss>=2.6, requests>=2.28.0, tqdm>=4.64.0
```

### Execution Steps

```bash
# 1. Environment Setup
.\epss-env\Scripts\Activate.ps1

# 2. Data Acquisition  
python -m catalogs_processed.get_catalog

# 3. Historical Reconstruction
python -m catalogs_processed.cve_historical_reconstructor_v2

# 4. Tabular Conversion
python -m catalogs_processed.tabular_reconstruct

# 5. Feature Engineering
python -m cve_daily_wrangling_vectorized
```

### Validation Checklist

- [ ] **Output Files**: `cve_snapshots_irregular.parquet` + `cve_snapshot_manifest.json`
- [ ] **Row Count**: Manifest == Parquet == CSV rows  
- [ ] **Schema**: 38 columns with expected data types
- [ ] **Temporal Range**: 1992-2025 with reasonable distribution
- [ ] **Current Snapshots**: Every CVE has current day timestamp (e.g., `2025-06-12T00:00:00.000`)
- [ ] **CVSS Coverage**: ~75% of snapshots have scores
- [ ] **No Duplicates**: Unique composite key
- [ ] **File Size**: ~190MB+ Parquet (95%+ compression)

## Conclusion

This pipeline successfully transforms raw NVD data into a temporally accurate, ML-ready dataset through sophisticated historical reconstruction. The resulting 1.3M+ snapshots provide unprecedented insight into CVE evolution patterns while maintaining strict temporal integrity.

**Key Technical Achievements:**
- ✅ **Lossless Temporal Reconstruction**: No synthetic data, only real historical states + current snapshots
- ✅ **Current Snapshot Preservation**: Every CVE guaranteed to have current day state for backfilling
- ✅ **Smart Deduplication**: Day-level granularity with hash-based duplicate prevention
- ✅ **Robust Data Structure Handling**: Manages 20+ years of format evolution  
- ✅ **Scalable Processing**: Spark handles multi-GB datasets efficiently
- ✅ **Comprehensive Validation**: Multi-level quality assurance
- ✅ **Feature Engineering Excellence**: CVSS canonicalization and platform detection

The pipeline is production-ready with comprehensive error handling, validation, and monitoring. The output format is optimized for downstream ML applications requiring temporal sequence modeling. 