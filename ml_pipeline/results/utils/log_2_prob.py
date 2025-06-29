#!/usr/bin/env python3
"""
Log to Probability Conversion Script
===================================

This script converts EPSS predictions and true values from log space back to 
original probability space (0-1 range).

The transformation used in the pipeline:
- Forward:  log(epss + 1e-6)  
- Inverse:  clip(exp(log_epss) - 1e-6, 0.0, 1.0)

Usage: python -m ml_pipeline.results.utils.log_2_prob
"""

import numpy as np
import xarray as xr
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

# =============================================================================
# CONFIGURATION: Modify these settings as needed
# =============================================================================
PREDICTIONS_FILE_PATH = Path("ml_pipeline/results/predictions/predictions_stream_replay.nc")

# Clipping Analysis Configuration
ENABLE_CLIPPING_ANALYSIS = True  # Set to False to disable detailed clipping tracking

# =============================================================================
# CORE TRANSFORMATION FUNCTIONS
# =============================================================================

def log_to_prob(log_values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Convert log-transformed EPSS values back to probability space.
    
    This exactly reverses the transformation used in the ML pipeline:
    - Pipeline forward transform: log(epss + eps)  
    - This inverse transform: clip(exp(log_epss) - eps, 0.0, 1.0)
    
    Parameters
    ----------
    log_values : np.ndarray
        Array of log-transformed values
    eps : float, default=1e-6
        Epsilon value used in original transformation
        
    Returns  
    -------
    np.ndarray
        Values converted back to probability space [0, 1]
    """
    # Apply inverse log transformation
    prob_values = np.exp(log_values) - eps
    
    # Clip to ensure valid probability range
    # This is essential for numerical stability
    prob_values = np.clip(prob_values, 0.0, 1.0)
    
    return prob_values.astype(np.float32)


def log_to_prob_with_tracking(log_values: np.ndarray, 
                             coordinates: dict = None,
                             variable_name: str = "unknown",
                             track_clipping: bool = False,
                             eps: float = 1e-6):
    """
    Enhanced version that can track individual clipping events.
    
    This function maintains identical core functionality to log_to_prob() while
    optionally collecting detailed information about each clipping event.
    
    Parameters
    ----------
    log_values : np.ndarray
        Array of log-transformed values
    coordinates : dict, optional
        Dictionary containing coordinate arrays: {'cve': [...], 'time': [...], 'horizon': [...]}
    variable_name : str, default='unknown'
        Name of the variable being processed ('pred' or 'true')
    track_clipping : bool, default=False
        Whether to collect detailed clipping event information
    eps : float, default=1e-6
        Epsilon value used in original transformation
        
    Returns
    -------
    tuple
        (transformed_values, clipping_events) where:
        - transformed_values: np.ndarray in probability space [0, 1]
        - clipping_events: list of dict with detailed clipping information
    """
    # Step 1: Core transformation (identical to original function)
    prob_values_raw = np.exp(log_values) - eps
    
    # Step 2: Collect clipping events (if requested and coordinates provided)
    clipping_events = []
    if track_clipping and coordinates is not None:
        clipping_events = _collect_clipping_events(
            prob_values_raw, log_values, coordinates, variable_name
        )
    
    # Step 3: Apply clipping (identical to original function)
    prob_values_clipped = np.clip(prob_values_raw, 0.0, 1.0)
    
    return prob_values_clipped.astype(np.float32), clipping_events


def _collect_clipping_events(prob_values_raw: np.ndarray, 
                           log_values_orig: np.ndarray,
                           coordinates: dict, 
                           variable_name: str) -> list:
    """
    Efficiently collect all clipping events with full coordinate details.
    
    This function uses vectorized numpy operations to efficiently identify
    clipping events and map them to their coordinate positions.
    
    Parameters
    ----------
    prob_values_raw : np.ndarray
        Raw probability values before clipping (may be outside [0,1])
    log_values_orig : np.ndarray
        Original log-transformed values
    coordinates : dict
        Coordinate mapping: {'cve': [...], 'time': [...], 'horizon': [...]}
    variable_name : str
        Variable being processed ('pred' or 'true')
        
    Returns
    -------
    list
        List of dictionaries, each containing detailed clipping event information
    """
    clipping_events = []
    
    # Step 1: Find all positions where clipping occurs using vectorized operations
    below_zero_mask = prob_values_raw < 0.0
    above_one_mask = prob_values_raw > 1.0
    
    # Step 2: Process below-zero clipping events
    if np.any(below_zero_mask):
        below_indices = np.where(below_zero_mask)
        
        # Use vectorized approach for efficiency
        for i in range(len(below_indices[0])):
            cve_idx = below_indices[0][i]
            time_idx = below_indices[1][i] 
            horizon_idx = below_indices[2][i]
            
            # Extract values for this specific position
            original_log = float(log_values_orig[cve_idx, time_idx, horizon_idx])
            original_prob = float(prob_values_raw[cve_idx, time_idx, horizon_idx])
            clip_magnitude = abs(original_prob)  # How far below zero
            
            event = {
                'variable': variable_name,
                'cve_index': int(cve_idx),
                'time_index': int(time_idx),
                'horizon_index': int(horizon_idx),
                'cve_id': str(coordinates['cve'][cve_idx]),
                'original_log_value': original_log,
                'original_prob_value': original_prob,
                'clipped_value': 0.0,
                'clip_type': 'below_zero',
                'clip_magnitude': clip_magnitude
            }
            clipping_events.append(event)
    
    # Step 3: Process above-one clipping events  
    if np.any(above_one_mask):
        above_indices = np.where(above_one_mask)
        
        for i in range(len(above_indices[0])):
            cve_idx = above_indices[0][i]
            time_idx = above_indices[1][i]
            horizon_idx = above_indices[2][i]
            
            # Extract values for this specific position
            original_log = float(log_values_orig[cve_idx, time_idx, horizon_idx])
            original_prob = float(prob_values_raw[cve_idx, time_idx, horizon_idx])
            clip_magnitude = original_prob - 1.0  # How far above one
            
            event = {
                'variable': variable_name,
                'cve_index': int(cve_idx),
                'time_index': int(time_idx),
                'horizon_index': int(horizon_idx),
                'cve_id': str(coordinates['cve'][cve_idx]),
                'original_log_value': original_log,
                'original_prob_value': original_prob,
                'clipped_value': 1.0,
                'clip_type': 'above_one',
                'clip_magnitude': clip_magnitude
            }
            clipping_events.append(event)
    
    return clipping_events


def _save_clipping_analysis(clipping_events: list, input_path: Path, total_values: int):
    """
    Save detailed clipping analysis to timestamped CSV file.
    
    This function creates a comprehensive CSV report of all clipping events
    with full context and metadata for future analysis.
    
    Parameters
    ----------
    clipping_events : list
        List of clipping event dictionaries
    input_path : Path
        Path to original input file (for naming and metadata)
    total_values : int
        Total number of values in both pred and true arrays
        
    Returns
    -------
    Path
        Path to the created CSV file
    """
    # Early return if no clipping occurred
    if not clipping_events:
        print("   ✅ No clipping events detected - no CSV file created")
        return None
    
    # Create output directory
    output_dir = Path("ml_pipeline/results/utils")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped filename for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_stem = input_path.stem
    csv_path = output_dir / f"clipping_analysis_{input_stem}_{timestamp}.csv"
    
    # Convert to DataFrame for easy manipulation and export
    df = pd.DataFrame(clipping_events)
    
    # Add metadata columns for comprehensive analysis
    df['analysis_timestamp'] = datetime.now().isoformat()
    df['input_file'] = input_path.name
    df['total_values_processed'] = total_values
    df['clipping_percentage'] = (len(clipping_events) / total_values) * 100
    
    # Sort by severity (largest clip magnitude first) for easier analysis
    df = df.sort_values('clip_magnitude', ascending=False)
    
    # Save with informative header comment
    with open(csv_path, 'w', newline='') as f:
        f.write("# EPSS Log-to-Probability Clipping Analysis Report\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Input file: {input_path.name}\n")
        f.write(f"# Total values processed: {total_values:,}\n")
        f.write(f"# Clipping events found: {len(clipping_events):,}\n")
        f.write(f"# Clipping percentage: {(len(clipping_events) / total_values) * 100:.4f}%\n")
        f.write("#\n")
        
        # Write the DataFrame
        df.to_csv(f, index=False)
    
    # Print detailed summary for immediate feedback
    below_zero_count = len(df[df.clip_type == 'below_zero'])
    above_one_count = len(df[df.clip_type == 'above_one'])
    max_negative = df[df.clip_type == 'below_zero']['clip_magnitude'].max() if below_zero_count > 0 else 0
    max_positive = df[df.clip_type == 'above_one']['clip_magnitude'].max() if above_one_count > 0 else 0
    
    print(f"\n📋 Clipping Analysis Report: {csv_path.name}")
    print(f"   📊 Summary Statistics:")
    print(f"      • Total events: {len(clipping_events):,} / {total_values:,} values ({(len(clipping_events) / total_values) * 100:.4f}%)")
    print(f"      • Below zero: {below_zero_count:,} events (max magnitude: {max_negative:.6f})")
    print(f"      • Above one: {above_one_count:,} events (max magnitude: {max_positive:.6f})")
    
    # Variable breakdown
    var_breakdown = df.groupby('variable').size()
    print(f"   📈 By Variable:")
    for var, count in var_breakdown.items():
        print(f"      • {var}: {count:,} events")
    
    return csv_path


def validate_transformation(original: np.ndarray, recovered: np.ndarray, 
                          tolerance: float = 1e-6) -> bool:
    """
    Validate that the inverse transformation is working correctly.
    
    Parameters
    ----------
    original : np.ndarray
        Original probability values
    recovered : np.ndarray  
        Values after log -> inverse log transformation
    tolerance : float
        Maximum allowed error
        
    Returns
    -------
    bool
        True if transformation is accurate within tolerance
    """
    max_error = np.max(np.abs(original - recovered))
    return max_error < tolerance


def test_transformation():
    """Test the transformation with realistic EPSS values."""
    print("🧪 Testing transformation accuracy...")
    
    # Test with realistic EPSS probability values
    test_epss = np.array([0.00001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.99999])
    
    # Apply forward transformation (as done in pipeline)
    log_vals = np.log(test_epss + 1e-6)
    
    # Apply our inverse transformation
    recovered = log_to_prob(log_vals)
    
    # Calculate error
    max_error = np.max(np.abs(test_epss - recovered))
    
    print(f"   Original EPSS:  {test_epss}")
    print(f"   Recovered EPSS: {recovered}")
    print(f"   Max error:      {max_error:.2e}")
    
    if validate_transformation(test_epss, recovered):
        print("   ✅ Transformation is accurate!")
        return True
    else:
        print("   ❌ Transformation has errors!")
        return False


# =============================================================================
# NETCDF FILE PROCESSING
# =============================================================================

def process_predictions_file(input_path: Path, save_clipping_details: bool = True, output_suffix: str = "_prob") -> Path:
    """
    Process a predictions NetCDF file to convert log values to probabilities.
    
    This function now includes optional clipping analysis that creates detailed
    reports of any values that needed to be clipped during transformation.
    
    Parameters
    ----------
    input_path : Path
        Path to input predictions file
    save_clipping_details : bool, default=True
        Whether to generate detailed clipping analysis CSV file
    output_suffix : str, default="_prob"
        Suffix to add to output filename (before file extension)
        
    Returns
    -------
    Path
        Path to output file with converted values
    """
    print(f"📂 Loading predictions file: {input_path}")
    
    # Load the dataset
    ds = xr.open_dataset(input_path)
    
    # Display file information
    print(f"   Dataset dimensions: {dict(ds.dims)}")
    print(f"   Data variables: {list(ds.data_vars)}")
    print(f"   Coordinates: {list(ds.coords)}")
    
    # Verify expected structure
    required_vars = {'pred', 'true'}
    if not required_vars.issubset(set(ds.data_vars)):
        raise ValueError(f"Missing required variables. Expected {required_vars}, found {set(ds.data_vars)}")
    
    print(f"\n🔄 Converting values from log space to probability space...")
    
    # Create a copy to avoid modifying the original
    ds_prob = ds.copy(deep=True)
    
    # Prepare coordinates for clipping analysis (if requested)
    clipping_events_all = []
    if save_clipping_details:
        coordinates = {
            'cve': ds.cve.values,
            'time': ds.time.values,
            'horizon': ds.horizon.values
        }
        print("   📊 Clipping analysis enabled - collecting detailed event data...")
    
    # Process 'pred' variable
    print("   Converting 'pred' values...")
    pred_log = ds['pred'].values
    
    if save_clipping_details:
        pred_prob, pred_clipping_events = log_to_prob_with_tracking(
            pred_log, coordinates, 'pred', track_clipping=True
        )
        clipping_events_all.extend(pred_clipping_events)
        if pred_clipping_events:
            print(f"      • Found {len(pred_clipping_events):,} clipping events in 'pred'")
    else:
        pred_prob = log_to_prob(pred_log)
    
    # Check for any invalid values after transformation
    n_invalid_pred = np.sum(~np.isfinite(pred_prob))
    if n_invalid_pred > 0:
        print(f"   ⚠️  Warning: {n_invalid_pred} invalid values in pred after transformation")
    
    # Assign transformed values back to dataset
    ds_prob['pred'] = (ds['pred'].dims, pred_prob, ds['pred'].attrs)
    
    # Process 'true' variable
    print("   Converting 'true' values...")
    true_log = ds['true'].values  
    
    if save_clipping_details:
        true_prob, true_clipping_events = log_to_prob_with_tracking(
            true_log, coordinates, 'true', track_clipping=True
        )
        clipping_events_all.extend(true_clipping_events)
        if true_clipping_events:
            print(f"      • Found {len(true_clipping_events):,} clipping events in 'true'")
    else:
        true_prob = log_to_prob(true_log)
    
    # Check for any invalid values after transformation
    n_invalid_true = np.sum(~np.isfinite(true_prob))
    if n_invalid_true > 0:
        print(f"   ⚠️  Warning: {n_invalid_true} invalid values in true after transformation")
    
    # Assign transformed values back to dataset
    ds_prob['true'] = (ds['true'].dims, true_prob, ds['true'].attrs)
    
    # Save clipping analysis if requested and events were found
    if save_clipping_details:
        total_values = pred_log.size + true_log.size
        if clipping_events_all:
            print(f"   📋 Total clipping events across both variables: {len(clipping_events_all):,}")
        _save_clipping_analysis(clipping_events_all, input_path, total_values)
    
    # Update attributes to reflect transformation
    ds_prob.attrs.update({
        'transformation_applied': 'log_to_probability_conversion',
        'transformation_formula': 'clip(exp(log_values) - 1e-6, 0.0, 1.0)',
        'value_range': '[0.0, 1.0] probability space',
        'original_file': str(input_path.name)
    })
    
    # Create output path with custom suffix
    output_path = input_path.parent / (input_path.stem + output_suffix + input_path.suffix)
    
    print(f"💾 Saving transformed data to: {output_path}")
    
    # Save the transformed dataset
    # Use simple encoding to avoid compatibility issues
    encoding = {}
    for var in ds_prob.data_vars:
        if var in ['pred', 'true']:
            encoding[var] = {'dtype': 'float32', 'zlib': True, 'complevel': 4}
        else:
            # Use clean encoding for other variables
            encoding[var] = {'dtype': ds[var].dtype}
    
    ds_prob.to_netcdf(output_path, encoding=encoding)
    
    # Display statistics
    print(f"\n📊 Transformation Statistics:")
    print(f"   pred: min={pred_prob.min():.6f}, max={pred_prob.max():.6f}, mean={pred_prob.mean():.6f}")
    print(f"   true: min={true_prob.min():.6f}, max={true_prob.max():.6f}, mean={true_prob.mean():.6f}")
    
    # Clean up
    ds.close()
    ds_prob.close()
    
    print(f"✅ Successfully created: {output_path}")
    return output_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("🔄 EPSS Log-to-Probability Conversion")
    print("=" * 50)
    
    # Show configuration
    print(f"📋 Configuration:")
    print(f"   • Input file: {PREDICTIONS_FILE_PATH}")
    print(f"   • Clipping analysis: {'Enabled' if ENABLE_CLIPPING_ANALYSIS else 'Disabled'}")
    
    # Test transformation accuracy first
    if not test_transformation():
        print("❌ Transformation test failed. Aborting.")
        sys.exit(1)
    
    print(f"\n📁 Processing file: {PREDICTIONS_FILE_PATH}")
    
    # Check if input file exists
    if not PREDICTIONS_FILE_PATH.exists():
        print(f"❌ Error: File not found: {PREDICTIONS_FILE_PATH}")
        print(f"   Please update PREDICTIONS_FILE_PATH at the top of this script.")
        sys.exit(1)
    
    try:
        # Process the file with configured options
        output_path = process_predictions_file(
            PREDICTIONS_FILE_PATH, 
            save_clipping_details=ENABLE_CLIPPING_ANALYSIS
        )
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"   Input:  {PREDICTIONS_FILE_PATH}")
        print(f"   Output: {output_path}")
        
        if ENABLE_CLIPPING_ANALYSIS:
            print(f"   📊 Clipping analysis CSV saved to: ml_pipeline/results/utils/")
        
        print(f"\n💡 The output file contains the same structure as input,")
        print(f"   but with 'pred' and 'true' values converted to probability space [0-1]")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
