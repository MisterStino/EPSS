"""
Naming Convention Utilities for Multi-Source Data Pipeline

This module defines and enforces consistent naming conventions across all data sources:
EPSS, CSAF, NVD, Reddit, GitHub, etc.

Convention: {source}_{descriptor}_{type}
- source: epss, csaf, nvd, reddit, github, etc.
- descriptor: what the feature represents
- type: semantic type (score, count, flag, date, id, list)

Reserved columns (no prefix): cve, date, epss
"""

import re
from typing import Dict, List, Tuple, Optional
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

# Reserved columns that should never have source prefixes
RESERVED_COLUMNS = {'cve', 'date', 'epss'}

# Semantic type suffixes and their expected data types
SEMANTIC_TYPES = {
    'score': 'double',      # Numeric ratings/probabilities
    'count': 'integer',     # Counts/frequencies
    'flag': 'boolean',      # Binary indicators
    'date': 'date',         # Temporal fields
    'timestamp': 'timestamp', # Precise timestamps
    'id': 'string',         # Identifiers
    'list': 'string',       # JSON arrays/lists
    'text': 'string',       # Long text fields
    'category': 'string',   # Categorical variables
    'ratio': 'double',      # Ratios/percentages
    'length': 'integer',    # String lengths
    'sequence': 'integer',  # Sequence numbers
    'age': 'integer',       # Age in days/units
}

# Valid source prefixes
VALID_SOURCES = {
    'epss', 'csaf', 'nvd', 'reddit', 'github', 'kev', 
    'exploitdb', 'mastodon', 'honeypot', 'osint'
}

def standardize_column_name(original_name: str, source: str) -> str:
    """
    Convert any column name to standardized format: {source}_{descriptor}_{type}
    
    Args:
        original_name: Original column name (e.g., "CVE_ID", "Post Count", "hasVulnerability")
        source: Data source (e.g., "reddit", "nvd", "csaf")
    
    Returns:
        Standardized column name (e.g., "reddit_post_count")
    """
    if original_name.lower() in RESERVED_COLUMNS:
        return original_name.lower()
    
    # Clean the original name
    cleaned = original_name.strip()
    
    # Convert to snake_case
    # Handle camelCase -> snake_case
    cleaned = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', cleaned)
    # Handle spaces and special characters
    cleaned = re.sub(r'[^a-zA-Z0-9]+', '_', cleaned)
    # Convert to lowercase
    cleaned = cleaned.lower().strip('_')
    
    # Remove redundant source prefix if already present
    if cleaned.startswith(f"{source}_"):
        cleaned = cleaned[len(source)+1:]
    
    # Infer semantic type from the descriptor
    semantic_type = infer_semantic_type(cleaned, original_name)
    
    # Build final name
    if semantic_type and not cleaned.endswith(f"_{semantic_type}"):
        standardized = f"{source}_{cleaned}_{semantic_type}"
    else:
        standardized = f"{source}_{cleaned}"
    
    return standardized

def infer_semantic_type(descriptor: str, original_name: str) -> Optional[str]:
    """
    Infer semantic type from descriptor and original name patterns
    """
    desc_lower = descriptor.lower()
    orig_lower = original_name.lower()
    
    # Score patterns
    if any(word in desc_lower for word in ['score', 'rating', 'probability', 'epss', 'cvss']):
        return 'score'
    
    # Count patterns
    if any(word in desc_lower for word in ['count', 'number', 'num', 'total']) and 'account' not in desc_lower:
        return 'count'
    
    # Flag patterns  
    if (desc_lower.startswith('has_') or desc_lower.startswith('is_') or 
        desc_lower.startswith('can_') or 'flag' in desc_lower or
        any(word in desc_lower for word in ['available', 'exists', 'present'])):
        return 'flag'
    
    # Date patterns
    if any(word in desc_lower for word in ['date', 'time', 'published', 'updated', 'created', 'modified']):
        if 'timestamp' in desc_lower or 'datetime' in desc_lower:
            return 'timestamp'
        return 'date'
    
    # ID patterns
    if any(word in desc_lower for word in ['id', 'identifier', 'key']) and 'kid' not in desc_lower:
        return 'id'
    
    # List patterns
    if any(word in desc_lower for word in ['list', 'array', 'json', 'combined', 'merged']):
        return 'list'
    
    # Length patterns
    if any(word in desc_lower for word in ['length', 'len', 'size']) and 'array' not in desc_lower:
        return 'length'
    
    # Sequence patterns
    if any(word in desc_lower for word in ['sequence', 'seq', 'order', 'index']):
        return 'sequence'
    
    # Age patterns
    if any(word in desc_lower for word in ['age', 'days_since', 'since']):
        return 'age'
    
    # Ratio patterns
    if any(word in desc_lower for word in ['ratio', 'rate', 'percent', 'proportion']):
        return 'ratio'
    
    # Text patterns (for long text fields)
    if any(word in desc_lower for word in ['description', 'detail', 'text', 'content', 'message']):
        return 'text'
    
    # Category patterns
    if any(word in desc_lower for word in ['type', 'category', 'kind', 'class', 'status', 'state']):
        return 'category'
    
    return None

def apply_naming_convention(df: DataFrame, source: str, column_mapping: Optional[Dict[str, str]] = None) -> DataFrame:
    """
    Apply naming convention to entire DataFrame
    
    Args:
        df: Input DataFrame
        source: Data source identifier
        column_mapping: Optional manual mapping for specific columns
    
    Returns:
        DataFrame with standardized column names
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source '{source}'. Must be one of: {VALID_SOURCES}")
    
    # Build column mapping
    rename_mapping = {}
    
    for original_col in df.columns:
        if column_mapping and original_col in column_mapping:
            # Use manual mapping if provided
            new_name = column_mapping[original_col]
        else:
            # Use automatic standardization
            new_name = standardize_column_name(original_col, source)
        
        rename_mapping[original_col] = new_name
    
    # Apply renaming
    for old_name, new_name in rename_mapping.items():
        df = df.withColumnRenamed(old_name, new_name)
    
    return df

def validate_column_names(columns: List[str]) -> List[Tuple[str, str]]:
    """
    Validate column names against conventions and return violations
    
    Returns:
        List of (column_name, violation_reason) tuples
    """
    violations = []
    
    for col_name in columns:
        # Skip reserved columns
        if col_name in RESERVED_COLUMNS:
            continue
        
        # Must contain at least one underscore (source_descriptor)
        if '_' not in col_name:
            violations.append((col_name, "Missing source prefix"))
            continue
        
        parts = col_name.split('_')
        
        # Must have at least 2 parts (source_descriptor)
        if len(parts) < 2:
            violations.append((col_name, "Invalid format - need source_descriptor"))
            continue
        
        source = parts[0]
        
        # Validate source
        if source not in VALID_SOURCES:
            violations.append((col_name, f"Invalid source '{source}'"))
        
        # Check for invalid characters
        if not re.match(r'^[a-z0-9_]+$', col_name):
            violations.append((col_name, "Contains invalid characters"))
    
    return violations

def get_columns_by_source(columns: List[str]) -> Dict[str, List[str]]:
    """
    Group columns by their source prefix
    """
    by_source = {'reserved': []}
    
    for col_name in columns:
        if col_name in RESERVED_COLUMNS:
            by_source['reserved'].append(col_name)
        elif '_' in col_name:
            source = col_name.split('_')[0]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(col_name)
        else:
            if 'unknown' not in by_source:
                by_source['unknown'] = []
            by_source['unknown'].append(col_name)
    
    return by_source

def get_columns_by_type(columns: List[str]) -> Dict[str, List[str]]:
    """
    Group columns by their semantic type suffix
    """
    by_type = {}
    
    for col_name in columns:
        if col_name in RESERVED_COLUMNS:
            continue
            
        # Find semantic type
        found_type = None
        for semantic_type in SEMANTIC_TYPES:
            if col_name.endswith(f"_{semantic_type}"):
                found_type = semantic_type
                break
        
        if found_type:
            if found_type not in by_type:
                by_type[found_type] = []
            by_type[found_type].append(col_name)
        else:
            if 'untyped' not in by_type:
                by_type['untyped'] = []
            by_type['untyped'].append(col_name)
    
    return by_type

# Example usage and testing
if __name__ == "__main__":
    # Test standardization
    test_cases = [
        ("CVE_ID", "reddit", "cve"),  # Reserved column
        ("Post Count", "reddit", "reddit_post_count"),
        ("hasVulnerability", "csaf", "csaf_has_vulnerability_flag"),
        ("discovery_date", "csaf", "csaf_discovery_date"),
        ("CVSS_Score", "nvd", "nvd_cvss_score"),
        ("exploitDB_type", "exploitdb", "exploitdb_type_category"),
    ]
    
    print("🧪 Testing Naming Convention")
    print("=" * 50)
    
    for original, source, expected in test_cases:
        result = standardize_column_name(original, source)
        status = "✅" if result == expected else "❌"
        print(f"{status} {original} ({source}) -> {result}")
        if result != expected:
            print(f"    Expected: {expected}") 