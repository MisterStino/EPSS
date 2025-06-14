#!/usr/bin/env python3
"""
CVE Daily Feature Engineering Pipeline - CORRECTED VERSION

Transforms deduplicated master CVE timeseries into ML-ready daily features
for exploit prediction modeling. Maintains temporal integrity and semantic correctness.

Requirements:
    pip install pandas pyarrow tqdm python-dateutil cvss

Author: Data Engineering Pipeline
"""

import json
import ast
import re
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================
MASTER_CSV = Path("catalogs_processed/master_cve_timeseries_dedup.csv")
OUT_PARQUET = Path("catalogs_processed/cve_timeseries_daily.parquet")
PARQUET_COMP = "zstd"

# ============================================================================
# ROBUST JSON PARSING - HANDLES RECONSTRUCTION ARTIFACTS
# ============================================================================
def robust_json(x):
    """
    Parse JSON with fallback for double-encoded strings from reconstruction process.
    """
    if pd.isna(x) or x == '' or x == '[]' or x == '{}':
        return None
    if not isinstance(x, str):
        return x
    
    try:
        # First parsing attempt
        obj = json.loads(x)
        # Check if result is still a string (double-encoded)
        if isinstance(obj, str):
            obj = json.loads(obj)
        return obj
    except (json.JSONDecodeError, TypeError):
        try:
            # Fallback to ast.literal_eval for malformed JSON
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None

# ============================================================================
# CVSS STANDARDIZATION - CORRECTED IMPLEMENTATION
# ============================================================================
def extract_cvss_features(metrics_blob):
    """
    Extract CVSS features using proper library usage and robust error handling.
    
    Returns standardized CVSS data regardless of version.
    """
    # Initialize result structure
    result = {
        "cvss_version": None,
        "cvss_vector": None,
        "cvss_base_score": None,
        "cvss_temporal_score": None,
        "cvss_environmental_score": None,
        "cvss_severity": None,
        # Version flags
        "has_cvss_v2": 0,
        "has_cvss_v30": 0,
        "has_cvss_v31": 0,
        "has_cvss_v40": 0,
        # CVSS v2 specific
        "cvss_v2_av": None,
        "cvss_v2_ac": None,
        "cvss_v2_au": None,
        "cvss_v2_c": None,
        "cvss_v2_i": None,
        "cvss_v2_a": None,
        # CVSS v3+ specific
        "cvss_v3_av": None,
        "cvss_v3_ac": None,
        "cvss_v3_pr": None,
        "cvss_v3_ui": None,
        "cvss_v3_s": None,
        "cvss_v3_c": None,
        "cvss_v3_i": None,
        "cvss_v3_a": None,
    }
    
    metrics = robust_json(metrics_blob)
    if not isinstance(metrics, dict):
        return result
    
    # Priority order: newer versions preferred
    version_order = [
        ("V40", "cvssMetricV40", "4.0"),
        ("V31", "cvssMetricV31", "3.1"), 
        ("V30", "cvssMetricV30", "3.0"),
        ("V2", "cvssMetricV2", "2.0")
    ]
    
    for version_tag, metric_key, version_num in version_order:
        metric_data = metrics.get(metric_key)
        if not metric_data:
            continue
            
        # Handle list format (standard NVD structure)
        if isinstance(metric_data, list) and len(metric_data) > 0:
            cvss_data = metric_data[0].get("cvssData", {})
        elif isinstance(metric_data, dict):
            cvss_data = metric_data.get("cvssData", {})
        else:
            continue
            
        vector_string = cvss_data.get("vectorString")
        base_score = cvss_data.get("baseScore")
        base_severity = cvss_data.get("baseSeverity")
        
        if not vector_string:
            continue
            
        try:
            # Use the python-cvss library correctly
            if version_tag == "V2":
                from cvss import CVSS2
                cvss_obj = CVSS2(vector_string)
                result["has_cvss_v2"] = 1
                
                # Extract CVSS v2 specific metrics
                if hasattr(cvss_obj, 'access_vector'):
                    result["cvss_v2_av"] = cvss_obj.access_vector
                if hasattr(cvss_obj, 'access_complexity'):
                    result["cvss_v2_ac"] = cvss_obj.access_complexity
                if hasattr(cvss_obj, 'authentication'):
                    result["cvss_v2_au"] = cvss_obj.authentication
                if hasattr(cvss_obj, 'confidentiality_impact'):
                    result["cvss_v2_c"] = cvss_obj.confidentiality_impact
                if hasattr(cvss_obj, 'integrity_impact'):
                    result["cvss_v2_i"] = cvss_obj.integrity_impact
                if hasattr(cvss_obj, 'availability_impact'):
                    result["cvss_v2_a"] = cvss_obj.availability_impact
                    
            elif version_tag in ("V30", "V31"):
                from cvss import CVSS3
                cvss_obj = CVSS3(vector_string)
                if version_tag == "V30":
                    result["has_cvss_v30"] = 1
                else:
                    result["has_cvss_v31"] = 1
                
                # Extract CVSS v3 specific metrics
                if hasattr(cvss_obj, 'attack_vector'):
                    result["cvss_v3_av"] = cvss_obj.attack_vector
                if hasattr(cvss_obj, 'attack_complexity'):
                    result["cvss_v3_ac"] = cvss_obj.attack_complexity
                if hasattr(cvss_obj, 'privileges_required'):
                    result["cvss_v3_pr"] = cvss_obj.privileges_required
                if hasattr(cvss_obj, 'user_interaction'):
                    result["cvss_v3_ui"] = cvss_obj.user_interaction
                if hasattr(cvss_obj, 'scope'):
                    result["cvss_v3_s"] = cvss_obj.scope
                if hasattr(cvss_obj, 'confidentiality'):
                    result["cvss_v3_c"] = cvss_obj.confidentiality
                if hasattr(cvss_obj, 'integrity'):
                    result["cvss_v3_i"] = cvss_obj.integrity
                if hasattr(cvss_obj, 'availability'):
                    result["cvss_v3_a"] = cvss_obj.availability
                    
            elif version_tag == "V40":
                # CVSS v4.0 support may be limited - use basic extraction
                result["has_cvss_v40"] = 1
            
            # Common fields
            result.update({
                "cvss_version": version_num,
                "cvss_vector": vector_string,
                "cvss_base_score": base_score,
                "cvss_severity": base_severity.lower() if base_severity else None
            })
            
            # Try to get scores from the parsed object
            if hasattr(cvss_obj, 'base_score'):
                result["cvss_base_score"] = cvss_obj.base_score
            if hasattr(cvss_obj, 'temporal_score'):
                result["cvss_temporal_score"] = cvss_obj.temporal_score
            if hasattr(cvss_obj, 'environmental_score'):
                result["cvss_environmental_score"] = cvss_obj.environmental_score
            
            break  # Use first available version
            
        except Exception:
            # Silently continue to next version on parsing errors
            continue
    
    return result

# ============================================================================
# CONFIGURATION PARSING - EXTRACTS CPE AND PLATFORM FEATURES
# ============================================================================
CPE_VENDOR_REGEX = re.compile(r'cpe:2\.3:[aho]:([^:]+):')

def parse_configurations(config_blob):
    """Extract configuration features from CPE data."""
    config_data = robust_json(config_blob)
    if not isinstance(config_data, dict):
        return {
            "n_cpes": 0,
            "n_vendors": 0,
            "is_windows": 0,
            "is_linux": 0,
            "is_android": 0,
            "is_ios": 0,
            "is_macos": 0,
            "is_hardware": 0,
            "is_application": 0,
            "is_os": 0
        }
    
    nodes = config_data.get("nodes", [])
    if isinstance(nodes, str):
        nodes = robust_json(nodes) or []
    
    cpe_count = 0
    vendors = set()
    platform_flags = {
        "is_windows": 0,
        "is_linux": 0, 
        "is_android": 0,
        "is_ios": 0,
        "is_macos": 0,
        "is_hardware": 0,
        "is_application": 0,
        "is_os": 0
    }
    
    for node in nodes:
        if not isinstance(node, dict):
            continue
            
        for cpe_match in node.get("cpeMatch", []):
            if not isinstance(cpe_match, dict):
                continue
                
            criteria = cpe_match.get("criteria", "")
            if not isinstance(criteria, str):
                continue
                
            cpe_count += 1
            
            # Platform detection
            criteria_lower = criteria.lower()
            if ":windows:" in criteria_lower:
                platform_flags["is_windows"] = 1
            if ":linux:" in criteria_lower:
                platform_flags["is_linux"] = 1
            if ":android:" in criteria_lower:
                platform_flags["is_android"] = 1
            if ":ios:" in criteria_lower:
                platform_flags["is_ios"] = 1
            if ":macos:" in criteria_lower or ":mac_os:" in criteria_lower:
                platform_flags["is_macos"] = 1
            
            # CPE type detection
            if criteria.startswith("cpe:2.3:h:"):
                platform_flags["is_hardware"] = 1
            elif criteria.startswith("cpe:2.3:a:"):
                platform_flags["is_application"] = 1
            elif criteria.startswith("cpe:2.3:o:"):
                platform_flags["is_os"] = 1
            
            # Vendor extraction
            vendor_match = CPE_VENDOR_REGEX.match(criteria)
            if vendor_match:
                vendors.add(vendor_match.group(1))
    
    return {
        "n_cpes": cpe_count,
        "n_vendors": len(vendors),
        **platform_flags
    }

# ============================================================================
# TEXT PROCESSING - PRESERVES MULTILINGUAL DESCRIPTIONS
# ============================================================================
def extract_descriptions(desc_blob):
    """Extract description text while preserving multilingual content."""
    descriptions = robust_json(desc_blob)
    if not isinstance(descriptions, list):
        return {
            "description_all": "",
            "description_en": "",
            "desc_len_chars": 0,
            "desc_len_words": 0,
            "n_languages": 0
        }
    
    all_text = []
    en_text = []
    languages = set()
    
    for desc in descriptions:
        if not isinstance(desc, dict):
            continue
            
        lang = desc.get("lang", "").lower()
        value = desc.get("value", "")
        
        if not isinstance(value, str):
            continue
            
        languages.add(lang)
        all_text.append(value)
        
        if lang == "en":
            en_text.append(value)
    
    description_all = " ".join(all_text)
    description_en = " ".join(en_text)
    
    return {
        "description_all": description_all,
        "description_en": description_en,
        "desc_len_chars": len(description_en),
        "desc_len_words": len(description_en.split()),
        "n_languages": len(languages)
    }

# ============================================================================
# WEAKNESS PROCESSING - EXTRACTS CWE INFORMATION
# ============================================================================
def extract_weakness_info(weakness_blob):
    """Extract CWE (Common Weakness Enumeration) information."""
    weaknesses = robust_json(weakness_blob)
    if not isinstance(weaknesses, list) or len(weaknesses) == 0:
        return {
            "primary_cwe": None,
            "n_cwes": 0
        }
    
    primary_cwe = None
    cwe_count = len(weaknesses)
    
    # Extract primary CWE
    try:
        first_weakness = weaknesses[0]
        if isinstance(first_weakness, dict):
            descriptions = first_weakness.get("description", [])
            if isinstance(descriptions, list) and len(descriptions) > 0:
                primary_cwe = descriptions[0].get("value")
    except (IndexError, KeyError, TypeError):
        pass
    
    return {
        "primary_cwe": primary_cwe,
        "n_cwes": cwe_count
    }

# ============================================================================
# REFERENCE PROCESSING
# ============================================================================
def count_references(ref_blob):
    """Count references, handling various JSON structures."""
    refs = robust_json(ref_blob)
    if isinstance(refs, list):
        return len(refs)
    return 0

# ============================================================================
# OPTIMIZED BATCH PROCESSING
# ============================================================================
def process_in_batches(df, batch_size=1000):
    """Process dataframe in batches for better performance and memory management."""
    
    print(f"   🔄 Processing {len(df):,} rows in batches of {batch_size:,}...")
    
    # Pre-allocate result lists
    all_cvss_features = []
    all_config_features = []
    all_desc_features = []
    all_weakness_features = []
    all_ref_counts = []
    
    # Process in batches
    for i in tqdm(range(0, len(df), batch_size), desc="Batch Processing"):
        batch = df.iloc[i:i+batch_size]
        
        # CVSS processing
        cvss_batch = [extract_cvss_features(metrics) for metrics in batch["metrics_json"]]
        all_cvss_features.extend(cvss_batch)
        
        # Configuration processing
        config_batch = [parse_configurations(config) for config in batch["configurations_json"]]
        all_config_features.extend(config_batch)
        
        # Description processing
        desc_batch = [extract_descriptions(desc) for desc in batch["descriptions_json"]]
        all_desc_features.extend(desc_batch)
        
        # Weakness processing
        weakness_batch = [extract_weakness_info(weakness) for weakness in batch["weaknesses_json"]]
        all_weakness_features.extend(weakness_batch)
        
        # Reference counting
        ref_batch = [count_references(ref) for ref in batch["references_json"]]
        all_ref_counts.extend(ref_batch)
    
    return all_cvss_features, all_config_features, all_desc_features, all_weakness_features, all_ref_counts

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================
def validate_temporal_integrity(df):
    """Ensure no future leakage in the dataset."""
    print("🔍 Validating temporal integrity...")
    
    temporal_issues = df[df['date'] < df['published_date']]
    
    if len(temporal_issues) > 0:
        print(f"⚠️  Found {len(temporal_issues)} rows where date < published_date")
    else:
        print("✅ Temporal integrity maintained")
    
    return len(temporal_issues) == 0

def validate_schema_uniqueness(df):
    """Ensure (cve_id, date) uniqueness."""
    print("🔍 Validating schema uniqueness...")
    
    duplicates = df.duplicated(subset=['cve_id', 'date']).sum()
    
    if duplicates > 0:
        print(f"❌ Found {duplicates} duplicate (cve_id, date) pairs")
        return False
    else:
        print("✅ Schema uniqueness maintained")
        return True

def validate_feature_consistency(df):
    """Validate that engineered features are consistent."""
    print("🔍 Validating feature consistency...")
    
    issues = []
    
    # Check CVSS version flags
    cvss_cols = [col for col in df.columns if col.startswith('has_cvss_')]
    if cvss_cols:
        version_sums = df[cvss_cols].sum(axis=1)
        if (version_sums > 1).any():
            issues.append("Multiple CVSS versions flagged for same CVE")
    
    # Check that counts are non-negative
    count_cols = [col for col in df.columns if col.startswith('n_')]
    for col in count_cols:
        if (df[col] < 0).any():
            issues.append(f"Negative counts found in {col}")
    
    if issues:
        print(f"❌ Feature consistency issues: {issues}")
        return False
    else:
        print("✅ Feature consistency validated")
        return True

# ============================================================================
# MAIN PIPELINE - OPTIMIZED VERSION
# ============================================================================
def main():
    """Execute the complete daily feature engineering pipeline."""
    print("🚀 CVE DAILY FEATURE ENGINEERING PIPELINE - OPTIMIZED")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: DATA LOADING & INITIAL VALIDATION
    # ========================================================================
    print("\n1️⃣ Loading and validating master dataset...")
    
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Master CSV not found: {MASTER_CSV}")
    
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    
    print(f"   📊 Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Remove exact duplicates
    print("   🧹 Removing duplicate (cve_id, timestamp) pairs...")
    df_before = len(df)
    df = df.drop_duplicates(subset=['cve_id', 'reconstruction_timestamp'], keep='first')
    df_after = len(df)
    duplicates_removed = df_before - df_after
    
    print(f"   ✅ Removed {duplicates_removed:,} duplicate rows ({duplicates_removed/df_before*100:.1f}%)")
    print(f"   📊 Clean dataset: {len(df):,} rows, {len(df.columns)} columns")
    print(f"   📊 Unique CVEs: {df['cve_id'].nunique():,}")
    print(f"   📊 Date range: {df['reconstruction_timestamp'].min()} to {df['reconstruction_timestamp'].max()}")
    
    # ========================================================================
    # STEP 2: TYPE CONVERSION & TEMPORAL SETUP
    # ========================================================================
    print("\n2️⃣ Converting data types and setting up temporal structure...")
    
    # Convert timestamp columns
    timestamp_cols = ["reconstruction_timestamp", "published_date", "last_modified_date"]
    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"   ✅ Converted {col} to datetime64")
    
    # ========================================================================
    # STEP 3: BATCH FEATURE EXTRACTION
    # ========================================================================
    print("\n3️⃣ Extracting features using optimized batch processing...")
    
    cvss_features, config_features, desc_features, weakness_features, ref_counts = process_in_batches(df, batch_size=5000)
    
    # Convert to DataFrames and concatenate
    print("   🔗 Combining extracted features...")
    
    cvss_df = pd.DataFrame(cvss_features)
    config_df = pd.DataFrame(config_features)
    desc_df = pd.DataFrame(desc_features)
    weakness_df = pd.DataFrame(weakness_features)
    
    # Add reference counts
    df["n_references"] = ref_counts
    
    # Combine all features
    df = pd.concat([df, cvss_df, config_df, desc_df, weakness_df], axis=1)
    
    print(f"   ✅ Added {len(cvss_df.columns) + len(config_df.columns) + len(desc_df.columns) + len(weakness_df.columns) + 1} feature columns")
    
    # ========================================================================
    # STEP 4: DAILY RESAMPLING WITH FORWARD FILL
    # ========================================================================
    print("\n4️⃣ Creating daily grid with forward-fill...")
    
    print(f"   📊 Pre-resampling shape: {df.shape}")
    
    # Set reconstruction_timestamp as index for resampling
    df_indexed = df.set_index('reconstruction_timestamp').sort_index()
    
    # Group by CVE and resample to daily frequency with forward fill
    print("   🔄 Resampling to daily frequency...")
    
    resampled_groups = []
    chunked_cves = []
    
    for cve_id, group in tqdm(df_indexed.groupby('cve_id'), desc="Daily Resampling"):
        # Calculate potential timeline length
        min_date = group.index.min()
        max_date = group.index.max()
        timeline_days = (max_date - min_date).days + 1
        
        # For extremely long timelines, use chunked processing
        if timeline_days > 1095:  # >3 years
            chunked_cves.append((cve_id, timeline_days, len(group)))
            print(f"   🔄 Chunked processing for {cve_id}: {timeline_days} days timeline")
            
            # Process in 1-year chunks to avoid memory issues
            chunk_size_days = 365
            current_date = min_date
            cve_chunks = []
            
            while current_date <= max_date:
                chunk_end = min(current_date + pd.Timedelta(days=chunk_size_days), max_date)
                
                # Get data for this chunk
                chunk_mask = (group.index >= current_date) & (group.index <= chunk_end)
                chunk_data = group[chunk_mask]
                
                if len(chunk_data) > 0:
                    try:
                        # Resample this chunk
                        chunk_resampled = chunk_data.resample('1D').ffill()
                        cve_chunks.append(chunk_resampled)
                    except (MemoryError, np.core._exceptions._ArrayMemoryError):
                        # If even a chunk fails, use original data points only
                        cve_chunks.append(chunk_data)
                
                current_date = chunk_end + pd.Timedelta(days=1)
            
            # Combine all chunks for this CVE
            if cve_chunks:
                combined_cve = pd.concat(cve_chunks)
                # Remove any duplicate dates that might occur at chunk boundaries
                combined_cve = combined_cve[~combined_cve.index.duplicated(keep='last')]
                resampled_groups.append(combined_cve)
        else:
            # Normal processing for reasonable timeline lengths
            try:
                daily_group = group.resample('1D').ffill()
                resampled_groups.append(daily_group)
            except (MemoryError, np.core._exceptions._ArrayMemoryError):
                # Fallback: use original data without resampling
                print(f"   ⚠️  Memory error for {cve_id}, using original data points")
                resampled_groups.append(group)
    
    # Report on chunked processing
    if chunked_cves:
        print(f"\n   🔄 Used chunked processing for {len(chunked_cves)} long-timeline CVEs:")
        for cve_id, days, states in chunked_cves[:10]:  # Show first 10
            print(f"      {cve_id}: {days} days, {states} states")
        if len(chunked_cves) > 10:
            print(f"      ... and {len(chunked_cves) - 10} more")
    
    # Combine all resampled groups
    if not resampled_groups:
        raise ValueError("No CVEs could be processed - unexpected error")
    
    df_daily = pd.concat(resampled_groups)
    
    # Reset index and create date column
    df_daily = df_daily.reset_index()
    df_daily = df_daily.rename(columns={'reconstruction_timestamp': 'date'})
    
    # Add synthetic flag
    df_daily['is_synthetic'] = 1  # Start with all as synthetic
    
    # Mark original observation dates as non-synthetic
    original_dates = df.set_index(['cve_id', 'reconstruction_timestamp']).index
    daily_dates = df_daily.set_index(['cve_id', 'date']).index
    
    # Find intersection of original and daily dates
    for cve_id, orig_date in original_dates:
        mask = (df_daily['cve_id'] == cve_id) & (df_daily['date'] == orig_date)
        df_daily.loc[mask, 'is_synthetic'] = 0
    
    print(f"   📊 Post-resampling shape: {df_daily.shape}")
    print(f"   📊 Synthetic rows: {df_daily['is_synthetic'].sum():,}")
    print(f"   📊 Original rows: {(df_daily['is_synthetic'] == 0).sum():,}")
    
    # ========================================================================
    # STEP 5: VALIDATION
    # ========================================================================
    print("\n5️⃣ Performing comprehensive validation...")
    
    # Validate temporal integrity
    temporal_ok = validate_temporal_integrity(df_daily)
    
    # Validate schema uniqueness
    schema_ok = validate_schema_uniqueness(df_daily)
    
    # Validate feature consistency
    features_ok = validate_feature_consistency(df_daily)
    
    if not all([temporal_ok, schema_ok, features_ok]):
        raise ValueError("Validation failed - check errors above")
    
    # ========================================================================
    # STEP 6: FINAL CLEANUP AND OPTIMIZATION
    # ========================================================================
    print("\n6️⃣ Final cleanup and optimization...")
    
    # Drop heavy JSON blob columns to reduce size
    json_cols = [col for col in df_daily.columns if col.endswith('_json')]
    print(f"   🗑️  Dropping {len(json_cols)} JSON blob columns to reduce size")
    df_daily = df_daily.drop(columns=json_cols)
    
    # Optimize data types for memory efficiency
    print("   🔧 Optimizing data types...")
    
    # Convert boolean flags to int8
    bool_cols = [col for col in df_daily.columns if col.startswith('is_') or col.startswith('has_')]
    for col in bool_cols:
        df_daily[col] = df_daily[col].astype('int8')
    
    # Convert count columns to appropriate integer types
    count_cols = [col for col in df_daily.columns if col.startswith('n_')]
    for col in count_cols:
        df_daily[col] = df_daily[col].astype('int32')
    
    print(f"   ✅ Optimized {len(bool_cols + count_cols)} columns")
    
    # ========================================================================
    # STEP 7: SAVE FINAL DATASET
    # ========================================================================
    print("\n7️⃣ Saving final dataset...")
    
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with compression
    df_daily.to_parquet(OUT_PARQUET, compression=PARQUET_COMP, index=False)
    
    file_size_mb = OUT_PARQUET.stat().st_size / (1024 * 1024)
    
    print(f"   💾 Saved to: {OUT_PARQUET}")
    print(f"   📊 Final shape: {df_daily.shape}")
    print(f"   📊 File size: {file_size_mb:.1f} MB")
    print(f"   📊 Compression: {PARQUET_COMP}")
    
    # ========================================================================
    # STEP 8: FINAL SUMMARY
    # ========================================================================
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"✅ Input: {len(df):,} irregular timeline states")
    print(f"✅ Output: {len(df_daily):,} daily feature vectors")
    print(f"✅ CVEs: {df_daily['cve_id'].nunique():,}")
    print(f"✅ Features: {len(df_daily.columns)} columns")
    print(f"✅ Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
    print(f"✅ Synthetic data: {(df_daily['is_synthetic'].sum() / len(df_daily) * 100):.1f}%")
    print("\n🚀 READY FOR ML TRAINING!")

if __name__ == "__main__":
    main() 