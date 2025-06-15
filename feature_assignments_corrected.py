#!/usr/bin/env python
"""
Exact feature assignments for 00_build_arrow.py
Based on revised analysis understanding event sparsity
"""

# ====================================================================
# CORRECTED FEATURE ASSIGNMENTS FOR 00_build_arrow.py
# ====================================================================

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

print("✅ FEATURE ASSIGNMENTS READY!")
print(f"📊 Columns to drop: {len(DROP_COLS)}")
print(f"📅 Safe timestamps: {len(TS_SAFE)}")  
print(f"⚠️  Leaky timestamps: {len(TS_LEAKY)}")
print(f"🔵 Boolean pattern detection: Dynamic")
print(f"🏷️  Categorical features: {len(CAT_COLS)}")

print(f"\n💡 KEY CHANGES FROM ORIGINAL ANALYSIS:")
print("• KEPT event features (has_discovery, has_release, etc.)")
print("• KEPT event sequence and stage features")
print("• KEPT cumulative counts and event progressions")
print("• These sparse features (98% 'missing') are actually the most predictive!")

# ====================================================================
# COPY-PASTE READY CODE FOR 00_build_arrow.py
# ====================================================================

copy_paste_code = '''
# Updated feature assignments based on event sparsity analysis
DROP_COLS = [
    # Large text fields
    "description_all", "description_en", "desc_len_all",
    "details_combined", "details_longest", "event_data_merged",
    
    # Metadata/provenance
    "cve_date_key", "original_date", 
    "reconstruction_timestamp", "reconstruction_timestamp_raw",
    
    # Truly empty
    "dominant_event_type", "primary_source", 
    
    # Always constant
    "has_threat", "has_remediation", "has_multi_source", "same_day_multi_source",
    
    # Low-variance flags
    "is_windows", "is_linux", "is_android", "is_ios", "is_macos",
    "is_hardware", "is_application", "is_os",
    "has_v2", "has_v30", "has_v31", "has_v40", "primary_cvss_ver", 
    
    # Mostly empty
    "event_types_list", "sources_list", "doc_ids",
    "event_type_count", "source_count", "total_detail_length", "weakness_count",
    
    # Technical/duplicate  
    "primary_cvss_vec", "cve_tags",
    "date_parsed", "last_modified_date", "snapshot_date", "reference_count",
]

TS_SAFE = ["published_date"]

TS_LEAKY = ["last_modified_date", "snapshot_date"]

BOOL_COLS = lambda df: [c for c in df.columns
                       if c.startswith(("has_", "is_"))
                       or c in ("same_day_multi_source",)]

CAT_COLS = [
    "cwe_id", "source_identifier", "vuln_status", 
    "canon_severity", "primary_cvss_sev", "prev_event_type",
]
'''

print(f"\n" + "="*60)
print("COPY-PASTE READY CODE:")
print("="*60)
print(copy_paste_code) 