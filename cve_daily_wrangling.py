#!/usr/bin/env python3
"""
CVE Daily Feature Engineering Pipeline

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
    
    Args:
        x: JSON string, potentially double-encoded
        
    Returns:
        Parsed object or None if unparseable
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
# CVSS STANDARDIZATION - HANDLES VERSION HETEROGENEITY
# ============================================================================
def canonical_cvss(metrics_blob):
    """
    Extract and standardize CVSS data across versions using python-cvss library.
    
    Creates uniform feature representation regardless of CVSS version while
    preserving version information through one-hot encoding.
    
    Args:
        metrics_blob: JSON string containing CVSS metrics
        
    Returns:
        Dict with canonical features and version flags
    """
    # Initialize result structure
    version_flags = {f"cvss_ver_{v}": 0 for v in ("V40", "V31", "V30", "V2")}
    result = {
        "canon_base": None,
        "canon_temporal": None, 
        "canon_environmental": None,
        "canon_severity": None,
        **version_flags
    }
    
    metrics = robust_json(metrics_blob)
    if not isinstance(metrics, dict):
        return result
    
    # Priority order: newer versions preferred for canonical representation
    version_order = [
        ("V40", "cvssMetricV40"),
        ("V31", "cvssMetricV31"), 
        ("V30", "cvssMetricV30"),
        ("V2", "cvssMetricV2")
    ]
    
    for version_tag, metric_key in version_order:
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
        if not isinstance(vector_string, str):
            continue
            
        try:
            # Import CVSS parser based on version
            if version_tag == "V2":
                from cvss import CVSS2
                parser = CVSS2(vector_string)
            elif version_tag in ("V30", "V31"):
                from cvss import CVSS3
                parser = CVSS3(vector_string)
            elif version_tag == "V40":
                from cvss import CVSS4
                parser = CVSS4(vector_string)
            else:
                continue
            
            # Extract standardized scores
            scores = parser.scores()
            base_score = scores[0] if scores else None
            temporal_score = scores[1] if len(scores) > 1 else None
            environmental_score = scores[2] if len(scores) > 2 else None
            
            # Update result with canonical values
            result.update({
                "canon_base": base_score,
                "canon_temporal": temporal_score,
                "canon_environmental": environmental_score,
                "canon_severity": parser.severity.lower() if hasattr(parser, 'severity') else None,
                f"cvss_ver_{version_tag}": 1
            })
            
            # Add version-specific sub-metrics if available
            if hasattr(parser, 'metrics'):
                metrics_dict = parser.metrics()
                # Add prefix to avoid conflicts
                for key, value in metrics_dict.items():
                    result[f"cvss_{key.lower()}"] = value
            
            break  # Use first available version
            
        except Exception as e:
            print(f"Warning: CVSS parsing failed for {vector_string}: {e}")
            continue
    
    return result

# ============================================================================
# CONFIGURATION PARSING - EXTRACTS CPE AND PLATFORM FEATURES
# ============================================================================
CPE_VENDOR_REGEX = re.compile(r'cpe:2\.3:[aho]:([^:]+):')

def parse_configurations(config_blob):
    """
    Extract configuration features from CPE data.
    
    CPE (Common Platform Enumeration) data indicates affected platforms
    and is strongly correlated with exploitation patterns.
    
    Args:
        config_blob: JSON string containing configuration data
        
    Returns:
        Dict with configuration features
    """
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
    """
    Extract description text while preserving multilingual content.
    
    Args:
        desc_blob: JSON string containing description array
        
    Returns:
        Dict with concatenated descriptions
    """
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
    """
    Extract CWE (Common Weakness Enumeration) information.
    
    Args:
        weakness_blob: JSON string containing weakness data
        
    Returns:
        Dict with weakness features
    """
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
# VALIDATION FUNCTIONS
# ============================================================================
def validate_temporal_integrity(df):
    """Ensure no future leakage in the dataset."""
    print("🔍 Validating temporal integrity...")
    
    # Check that reconstruction_timestamp <= any future processing date
    # This would be checked against actual processing date in production
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
    
    # Check CVSS version flags sum to <= 1
    cvss_cols = [col for col in df.columns if col.startswith('cvss_ver_')]
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
# MAIN PIPELINE
# ============================================================================
def main():
    """Execute the complete daily feature engineering pipeline."""
    print("🚀 CVE DAILY FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # ========================================================================
    # STEP 1: DATA LOADING & INITIAL VALIDATION
    # ========================================================================
    print("\n1️⃣ Loading and validating master dataset...")
    
    if not MASTER_CSV.exists():
        raise FileNotFoundError(f"Master CSV not found: {MASTER_CSV}")
    
    df = pd.read_csv(MASTER_CSV, low_memory=False)
    
    print(f"   📊 Loaded {len(df):,} rows, {len(df.columns)} columns")
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
    
    # Convert numeric columns
    numeric_cols = ["primary_cvss_score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"   ✅ Converted {col} to numeric")
    
    # ========================================================================
    # STEP 3: CVSS CANONICALIZATION
    # ========================================================================
    print("\n3️⃣ Canonicalizing CVSS metrics across versions...")
    
    print("   🔄 Processing CVSS metrics...")
    cvss_features = []
    
    for idx, metrics_json in enumerate(tqdm(df["metrics_json"], desc="CVSS Processing")):
        cvss_dict = canonical_cvss(metrics_json)
        cvss_features.append(cvss_dict)
    
    # Add CVSS features to dataframe
    cvss_df = pd.DataFrame(cvss_features)
    df = pd.concat([df, cvss_df], axis=1)
    
    print(f"   ✅ Added {len(cvss_df.columns)} CVSS features")
    
    # ========================================================================
    # STEP 4: CONFIGURATION PARSING
    # ========================================================================
    print("\n4️⃣ Parsing configuration and CPE data...")
    
    print("   🔄 Processing configurations...")
    config_features = []
    
    for config_json in tqdm(df["configurations_json"], desc="Config Processing"):
        config_dict = parse_configurations(config_json)
        config_features.append(config_dict)
    
    config_df = pd.DataFrame(config_features)
    df = pd.concat([df, config_df], axis=1)
    
    print(f"   ✅ Added {len(config_df.columns)} configuration features")
    
    # ========================================================================
    # STEP 5: TEXT AND WEAKNESS PROCESSING
    # ========================================================================
    print("\n5️⃣ Processing descriptions and weakness data...")
    
    # Process descriptions
    print("   🔄 Processing descriptions...")
    desc_features = []
    for desc_json in tqdm(df["descriptions_json"], desc="Description Processing"):
        desc_dict = extract_descriptions(desc_json)
        desc_features.append(desc_dict)
    
    desc_df = pd.DataFrame(desc_features)
    df = pd.concat([df, desc_df], axis=1)
    
    # Process weaknesses
    print("   🔄 Processing weaknesses...")
    weakness_features = []
    for weakness_json in tqdm(df["weaknesses_json"], desc="Weakness Processing"):
        weakness_dict = extract_weakness_info(weakness_json)
        weakness_features.append(weakness_dict)
    
    weakness_df = pd.DataFrame(weakness_features)
    df = pd.concat([df, weakness_df], axis=1)
    
    # Count references
    print("   🔄 Counting references...")
    df["n_references"] = df["references_json"].apply(count_references)
    
    print(f"   ✅ Added text and weakness features")
    
    # ========================================================================
    # STEP 6: DAILY RESAMPLING WITH FORWARD FILL
    # ========================================================================
    print("\n6️⃣ Creating daily grid with forward-fill...")
    
    print("   📊 Pre-resampling shape:", df.shape)
    
    # Set reconstruction_timestamp as index for resampling
    df_indexed = df.set_index('reconstruction_timestamp').sort_index()
    
    # Group by CVE and resample to daily frequency with forward fill
    print("   🔄 Resampling to daily frequency...")
    
    resampled_groups = []
    
    for cve_id, group in tqdm(df_indexed.groupby('cve_id'), desc="Daily Resampling"):
        # Resample to daily frequency and forward fill
        daily_group = group.resample('1D').ffill()
        resampled_groups.append(daily_group)
    
    # Combine all resampled groups
    df_daily = pd.concat(resampled_groups)
    
    # Reset index and create date column
    df_daily = df_daily.reset_index()
    df_daily = df_daily.rename(columns={'reconstruction_timestamp': 'date'})
    
    # Add synthetic flag
    # Note: After resampling, we need to identify which rows are synthetic
    # We'll mark all forward-filled rows as synthetic
    df_daily['is_synthetic'] = 1  # Start with all as synthetic
    
    # Mark original observation dates as non-synthetic
    original_dates = df.set_index(['cve_id', 'reconstruction_timestamp']).index
    daily_dates = df_daily.set_index(['cve_id', 'date']).index
    
    # Find intersection of original and daily dates
    for cve_id, orig_date in original_dates:
        # Find matching rows in daily data
        mask = (df_daily['cve_id'] == cve_id) & (df_daily['date'] == orig_date)
        df_daily.loc[mask, 'is_synthetic'] = 0
    
    print(f"   📊 Post-resampling shape: {df_daily.shape}")
    print(f"   📊 Synthetic rows: {df_daily['is_synthetic'].sum():,}")
    print(f"   📊 Original rows: {(df_daily['is_synthetic'] == 0).sum():,}")
    
    # ========================================================================
    # STEP 7: VALIDATION
    # ========================================================================
    print("\n7️⃣ Performing comprehensive validation...")
    
    # Validate temporal integrity
    temporal_ok = validate_temporal_integrity(df_daily)
    
    # Validate schema uniqueness
    schema_ok = validate_schema_uniqueness(df_daily)
    
    # Validate feature consistency
    features_ok = validate_feature_consistency(df_daily)
    
    if not all([temporal_ok, schema_ok, features_ok]):
        raise ValueError("Validation failed - check errors above")
    
    # ========================================================================
    # STEP 8: FINAL CLEANUP AND OPTIMIZATION
    # ========================================================================
    print("\n8️⃣ Final cleanup and optimization...")
    
    # Drop heavy JSON blob columns to reduce size
    json_cols = [col for col in df_daily.columns if col.endswith('_json')]
    print(f"   🗑️  Dropping {len(json_cols)} JSON blob columns to reduce size")
    df_daily = df_daily.drop(columns=json_cols)
    
    # Drop redundant columns that were replaced by engineered features
    redundant_cols = ['primary_cvss_ver', 'primary_cvss_vec', 'primary_cvss_score', 'primary_cvss_sev']
    redundant_cols = [col for col in redundant_cols if col in df_daily.columns]
    if redundant_cols:
        print(f"   🗑️  Dropping {len(redundant_cols)} redundant columns")
        df_daily = df_daily.drop(columns=redundant_cols)
    
    # Optimize data types for memory efficiency
    print("   🔧 Optimizing data types...")
    
    # Convert boolean flags to int8
    bool_cols = [col for col in df_daily.columns if col.startswith('is_') or col.startswith('cvss_ver_')]
    for col in bool_cols:
        df_daily[col] = df_daily[col].astype('int8')
    
    # Convert count columns to appropriate integer types
    count_cols = [col for col in df_daily.columns if col.startswith('n_')]
    for col in count_cols:
        df_daily[col] = df_daily[col].astype('int32')
    
    print(f"   ✅ Optimized {len(bool_cols + count_cols)} columns")
    
    # ========================================================================
    # STEP 9: SAVE FINAL DATASET
    # ========================================================================
    print("\n9️⃣ Saving final dataset...")
    
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with compression
    df_daily.to_parquet(OUT_PARQUET, compression=PARQUET_COMP, index=False)
    
    file_size_mb = OUT_PARQUET.stat().st_size / (1024 * 1024)
    
    print(f"   💾 Saved to: {OUT_PARQUET}")
    print(f"   📊 Final shape: {df_daily.shape}")
    print(f"   📊 File size: {file_size_mb:.1f} MB")
    print(f"   📊 Compression: {PARQUET_COMP}")
    
    # ========================================================================
    # STEP 10: FINAL SUMMARY
    # ========================================================================
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"✅ Input: {len(df):,} irregular timeline states")
    print(f"✅ Output: {len(df_daily):,} daily feature vectors")
    print(f"✅ CVEs: {df_daily['cve_id'].nunique():,}")
    print(f"✅ Features: {len(df_daily.columns)} columns")
    print(f"✅ Date range: {df_daily['date'].min()} to {df_daily['date'].max()}")
    print(f"✅ Synthetic data: {(df_daily['is_synthetic'].sum() / len(df_daily) * 100):.1f}%")
    print("\n🚀 READY FOR ML TRAINING!")

if __name__ == "__main__":
    main() 