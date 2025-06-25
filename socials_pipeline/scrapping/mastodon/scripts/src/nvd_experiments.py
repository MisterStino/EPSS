#!/usr/bin/env python3
"""
NVD CVE Data Preprocessing for Machine Learning Experiments
Prepares clean CVE vulnerability data for prediction tasks by:
- Removing validation-only columns
- Filtering to recent vulnerabilities (2021+)
- Ensuring complete CVSS data
- Creating ground truth validation sets
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

def load_and_inspect_data(file_path: str) -> pd.DataFrame:
    """
    Load the CVE dataset and inspect its structure
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Loaded DataFrame with basic inspection output
    """
    print("=== LOADING CVE DATASET ===")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {len(df):,} records from {file_path}")
        print(f"✓ Dataset shape: {df.shape}")
        print(f"✓ Columns: {len(df.columns)}")
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Error: File {file_path} not found")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise

def identify_validation_columns() -> List[str]:
    """
    Identify columns that were created for validation but aren't needed for ML
    
    Returns:
        List of column names to drop
    """
    validation_columns = [
        # Internal processing flags
        'cvss_library_parsed',
        'cvss_library_error',
        'processing_error',
        
        # Validation-specific scores (duplicates of original data)
        'cvss_base_score_validated',
        'cvss_severity_validated',
        'cvss_clean_vector',
        
        # Additional scoring not needed for base prediction
        'cvss_temporal_score',
        'cvss_environmental_score',
        
        # Quality assurance metadata
        'cvss_score_validation_match',
        'cvss_available_versions'
    ]
    
    print("=== VALIDATION COLUMNS TO DROP ===")
    for col in validation_columns:
        print(f"  - {col}: Validation/QA column not needed for ML")
    
    return validation_columns

def identify_core_cvss_columns() -> List[str]:
    """
    Identify core CVSS columns needed for prediction tasks
    
    Returns:
        List of essential CVSS column names
    """
    core_cvss_columns = [
        # Primary prediction targets
        'cvss_base_score',      # Regression target (0.0-10.0)
        'cvss_base_severity',   # Classification target (Low/Medium/High/Critical)
        
        # CVSS vector and version information
        'cvss_vector_string',   # Full CVSS vector for detailed analysis
        'cvss_version',         # CVSS version (2.0, 3.0, 3.1, 4.0)
        
        # Detailed CVSS metrics for feature engineering
        'cvss_exploitability_score',
        'cvss_impact_score',
        'cvss_attack_vector',
        'cvss_attack_complexity',
        'cvss_privileges_required',
        'cvss_user_interaction',
        'cvss_confidentiality_impact',
        'cvss_integrity_impact',
        'cvss_availability_impact'
    ]
    
    print("=== CORE CVSS COLUMNS FOR ML ===")
    for col in core_cvss_columns:
        print(f"  ✓ {col}: Essential for prediction tasks")
    
    return core_cvss_columns

def drop_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove validation-only columns that aren't needed for ML experiments
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with validation columns removed
    """
    print("\n=== DROPPING VALIDATION COLUMNS ===")
    
    validation_cols = identify_validation_columns()
    
    # Check which validation columns actually exist in the dataset
    existing_validation_cols = [col for col in validation_cols if col in df.columns]
    missing_validation_cols = [col for col in validation_cols if col not in df.columns]
    
    if existing_validation_cols:
        print(f"Dropping {len(existing_validation_cols)} validation columns:")
        for col in existing_validation_cols:
            print(f"  - {col}")
        df_cleaned = df.drop(columns=existing_validation_cols)
    else:
        print("No validation columns found to drop")
        df_cleaned = df.copy()
    
    if missing_validation_cols:
        print(f"Note: {len(missing_validation_cols)} expected validation columns not found in dataset")
    
    print(f"✓ Dataset shape after dropping validation columns: {df_cleaned.shape}")
    return df_cleaned

def filter_by_publication_date(df: pd.DataFrame, min_year: int = 2021) -> pd.DataFrame:
    """
    Filter dataset to include only vulnerabilities published from specified year onwards
    
    Args:
        df: Input DataFrame
        min_year: Minimum publication year to include
        
    Returns:
        Filtered DataFrame
    """
    print(f"\n=== FILTERING BY PUBLICATION DATE (>= {min_year}) ===")
    
    initial_count = len(df)
    
    # Convert published_date to datetime
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    
    # Create date filter
    min_date = pd.Timestamp(f'{min_year}-01-01')
    date_filter = df['published_date'] >= min_date
    
    # Apply filter
    df_filtered = df[date_filter].copy()
    
    # Report filtering results
    filtered_count = len(df_filtered)
    removed_count = initial_count - filtered_count
    
    print(f"✓ Records before date filtering: {initial_count:,}")
    print(f"✓ Records after date filtering: {filtered_count:,}")
    print(f"✓ Records removed (pre-{min_year}): {removed_count:,}")
    print(f"✓ Retention rate: {(filtered_count/initial_count)*100:.1f}%")
    
    return df_filtered

def remove_missing_cvss_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing values in core CVSS columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with complete CVSS data only
    """
    print("\n=== REMOVING MISSING CVSS DATA ===")
    
    core_cvss_cols = identify_core_cvss_columns()
    
    # Check which core CVSS columns exist in dataset
    existing_cvss_cols = [col for col in core_cvss_cols if col in df.columns]
    missing_cvss_cols = [col for col in core_cvss_cols if col not in df.columns]
    
    if missing_cvss_cols:
        print(f"Warning: {len(missing_cvss_cols)} expected CVSS columns not found:")
        for col in missing_cvss_cols:
            print(f"  - {col}")
    
    initial_count = len(df)
    
    # Focus on most critical CVSS columns that must be present
    critical_cvss_cols = ['cvss_base_score', 'cvss_base_severity']
    existing_critical_cols = [col for col in critical_cvss_cols if col in df.columns]
    
    if not existing_critical_cols:
        print("✗ Error: No critical CVSS columns found in dataset")
        raise ValueError("Cannot proceed without cvss_base_score or cvss_base_severity columns")
    
    # Remove rows with missing critical CVSS data
    print(f"Checking for missing values in critical CVSS columns: {existing_critical_cols}")
    
    for col in existing_critical_cols:
        missing_count = df[col].isna().sum()
        print(f"  - {col}: {missing_count:,} missing values")
    
    # Keep only rows with complete critical CVSS data
    df_complete = df.dropna(subset=existing_critical_cols).copy()
    
    complete_count = len(df_complete)
    removed_count = initial_count - complete_count
    
    print(f"✓ Records before CVSS filtering: {initial_count:,}")
    print(f"✓ Records after CVSS filtering: {complete_count:,}")
    print(f"✓ Records removed (missing CVSS): {removed_count:,}")
    print(f"✓ Retention rate: {(complete_count/initial_count)*100:.1f}%")
    
    return df_complete

def create_validation_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create validation dataset with CVE IDs and ground truth CVSS labels
    
    Args:
        df: Preprocessed DataFrame with complete data
        
    Returns:
        Validation DataFrame with cve_id and CVSS ground truth columns
    """
    print("\n=== CREATING VALIDATION GROUND TRUTH ===")
    
    # Define ground truth columns for validation
    ground_truth_columns = ['cve_id']
    
    # Add available CVSS columns for ground truth
    cvss_ground_truth_cols = [
        'cvss_base_score',      # Primary regression target
        'cvss_base_severity',   # Primary classification target
        'cvss_vector_string',   # Full vector for detailed analysis
        'cvss_version',         # Version information
        'cvss_exploitability_score',  # Additional metrics
        'cvss_impact_score'
    ]
    
    # Check which ground truth columns exist
    existing_gt_cols = [col for col in cvss_ground_truth_cols if col in df.columns]
    ground_truth_columns.extend(existing_gt_cols)
    
    print(f"Ground truth columns selected:")
    for col in ground_truth_columns:
        print(f"  ✓ {col}")
    
    # Create validation dataset
    validation_df = df[ground_truth_columns].copy()
    
    print(f"✓ Validation ground truth dataset created")
    print(f"✓ Shape: {validation_df.shape}")
    print(f"✓ Records: {len(validation_df):,}")
    
    # Summary statistics for ground truth
    if 'cvss_base_score' in validation_df.columns:
        score_stats = validation_df['cvss_base_score'].describe()
        print(f"✓ CVSS Base Score range: {score_stats['min']:.1f} - {score_stats['max']:.1f}")
        print(f"✓ CVSS Base Score mean: {score_stats['mean']:.1f}")
    
    if 'cvss_base_severity' in validation_df.columns:
        severity_counts = validation_df['cvss_base_severity'].value_counts()
        print(f"✓ CVSS Severity distribution:")
        for severity, count in severity_counts.items():
            print(f"    {severity}: {count:,} ({count/len(validation_df)*100:.1f}%)")
    
    return validation_df

def save_processed_datasets(df_main: pd.DataFrame, df_validation: pd.DataFrame, 
                          output_dir: str = ".") -> Tuple[str, str]:
    """
    Save the processed main dataset and validation ground truth
    
    Args:
        df_main: Main processed dataset
        df_validation: Validation ground truth dataset
        output_dir: Directory to save files
        
    Returns:
        Tuple of (main_file_path, validation_file_path)
    """
    print("\n=== SAVING PROCESSED DATASETS ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output paths
    main_file = f"{output_dir}/nvd_cve_processed_{timestamp}.csv"
    validation_file = f"{output_dir}/nvd_cve_ground_truth_{timestamp}.csv"
    
    # Save datasets
    df_main.to_csv(main_file, index=False)
    df_validation.to_csv(validation_file, index=False)
    
    print(f"✓ Main processed dataset saved: {main_file}")
    print(f"  - Records: {len(df_main):,}")
    print(f"  - Columns: {len(df_main.columns)}")
    print(f"  - Size: {Path(main_file).stat().st_size / (1024*1024):.1f} MB")
    
    print(f"✓ Validation ground truth saved: {validation_file}")
    print(f"  - Records: {len(df_validation):,}")
    print(f"  - Columns: {len(df_validation.columns)}")
    print(f"  - Size: {Path(validation_file).stat().st_size / (1024*1024):.1f} MB")
    
    return main_file, validation_file

def main():
    """
    Main preprocessing pipeline for NVD CVE data
    """
    print("NVD CVE Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Configuration
    input_file = "nvd_truth.csv"  # Renamed CVE dataset
    min_publication_year = 2021
    
    try:
        # Step 1: Load and inspect data
        df = load_and_inspect_data(input_file)
        
        # Step 2: Drop validation-only columns
        df = drop_validation_columns(df)
        
        # Step 3: Filter by publication date (2021+)
        df = filter_by_publication_date(df, min_publication_year)
        
        # Step 4: Remove records with missing CVSS data
        df = remove_missing_cvss_data(df)
        
        # Step 5: Create validation ground truth dataset
        validation_df = create_validation_ground_truth(df)
        
        # Step 6: Save processed datasets
        main_file, validation_file = save_processed_datasets(df, validation_df)
        
        # Final summary
        print(f"\n=== PREPROCESSING COMPLETE ===")
        print(f"✓ Successfully processed {len(df):,} CVE records")
        print(f"✓ Date range: {min_publication_year}+ vulnerabilities")
        print(f"✓ Complete CVSS data ensured")
        print(f"✓ Ready for machine learning experiments")
        print(f"\nOutput files:")
        print(f"  📄 Main dataset: {main_file}")
        print(f"  📊 Ground truth: {validation_file}")
        
        return df, validation_df
        
    except Exception as e:
        print(f"\n✗ PREPROCESSING FAILED: {e}")
        raise

if __name__ == "__main__":
    main() 