#!/usr/bin/env python3
"""
CORRECTED Early Warning System Evaluation for EPSS Forecasting

This module implements the CORRECT evaluation pipeline for early warning capability
based on rigorous analysis of model semantics and temporal structure.

CRITICAL CORRECTIONS:
1. Event-specific evaluation (not CVE-level aggregation)
2. Exact horizon-to-date matching for True Positives
3. Proper temporal logic: T_anchor + h + 1 = T_event
4. Individual event counting for contingency tables
5. Correct lead time calculation: lead_time = h + 1

Author: Research Team  
Date: 2025
"""

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _find_project_root():
    """Find project root by looking for ml_pipeline directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "ml_pipeline").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (looking for ml_pipeline directory)")


def _log_to_prob(arr: np.ndarray) -> np.ndarray:
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero or invalid."""
    if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return default
    return numerator / denominator


def load_and_align_data(ncfile: Path, epss_parquet: Path, threshold: float = 0.7):
    """
    Load NetCDF predictions and align with EPSS ground truth data.
    Memory-efficient approach using PyArrow streaming.
    """
    print("§1 Loading NetCDF data...")
    start_time = time.time()
    
    # Load NetCDF predictions
    ds = xr.open_dataset(ncfile)
    C, T, H = ds.pred.shape
    print(f"NetCDF shape: C={C}, T={T}, H={H}")
    
    # Extract metadata
    cves = ds.cve.values.astype(str)
    
    # Handle time coordinate - it's a 2D array (C, T) where all CVEs share same calendar
    time_arr = ds.time.values  # Shape (C, T)
    # All CVEs have the same time grid, so use the first CVE's time grid
    dates = pd.to_datetime(time_arr[0])  # Convert first CVE's time grid to datetime
    
    # Create mappings
    cve2idx = {cve: i for i, cve in enumerate(cves)}
    # Use nanosecond timestamps for mapping (consistent with previous implementation)
    date2tidx = {int(date.value): i for i, date in enumerate(dates)}
    
    print(f"§2 NetCDF loaded in {time.time() - start_time:.2f}s")
    print(f"   Date range: {dates.min().date()} to {dates.max().date()}")
    
    # Determine target date range for filtering
    min_date = dates.min() - pd.Timedelta(days=1)  # Buffer
    max_date = dates.max() + pd.Timedelta(days=H+1)  # Account for horizons
    target_cves = set(cves)
    target_dates_dt = pd.date_range(min_date, max_date, freq='D')
    
    print(f"§3 Target filtering: {len(target_cves)} CVEs, {len(target_dates_dt)} dates")
    print(f"   Date range: {min_date.date()} to {max_date.date()}")
    
    # Memory-efficient EPSS loading using PyArrow streaming
    print("§4 Loading EPSS data with streaming approach...")
    start_time = time.time()
    
    try:
        import pyarrow.parquet as pq
        
        # Use PyArrow's batch reader for memory efficiency
        parquet_file = pq.ParquetFile(epss_parquet)
        
        print(f"   Parquet metadata: {parquet_file.metadata.num_rows:,} total rows")
        print(f"   Processing in streaming batches...")
        
        epss_chunks = []
        batch_size = 500_000  # Smaller batches for memory safety
        processed_rows = 0
        max_rows = 3_000_000  # Limit total processing to 3M rows
        
        # Stream through the file in batches
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['cve', 'date', 'epss']):
            if processed_rows >= max_rows:
                print(f"   Reached processing limit of {max_rows:,} rows")
                break
                
            # Convert batch to pandas with minimal memory footprint
            batch_df = batch.to_pandas()
            processed_rows += len(batch_df)
            
            print(f"   Processing batch: {len(batch_df):,} rows (total: {processed_rows:,})")
            
            # Immediate filtering to reduce memory usage
            batch_df['date'] = pd.to_datetime(batch_df['date'])
            
            # Filter to our target CVEs and date range
            cve_mask = batch_df['cve'].isin(target_cves)
            date_mask = (batch_df['date'] >= min_date) & (batch_df['date'] <= max_date)
            filtered_batch = batch_df[cve_mask & date_mask]
            
            if len(filtered_batch) > 0:
                # Final filtering to exact dates
                target_dates_set = set(target_dates_dt)
                exact_date_mask = filtered_batch['date'].dt.normalize().isin(target_dates_set)
                filtered_batch = filtered_batch[exact_date_mask]
                
                if len(filtered_batch) > 0:
                    # Convert to category for memory efficiency
                    filtered_batch = filtered_batch.astype({"cve": "category"}).copy()
                    epss_chunks.append(filtered_batch)
                    print(f"     Kept {len(filtered_batch):,} relevant rows")
            
            # Force garbage collection
            del batch_df, filtered_batch
            
        # Combine all filtered chunks
        if epss_chunks:
            epss_df = pd.concat(epss_chunks, ignore_index=True)
            print(f"   Combined EPSS: {len(epss_df):,} rows match our criteria")
        else:
            epss_df = pd.DataFrame(columns=['cve', 'date', 'epss'])
            print("   No matching EPSS data found")
            
    except Exception as e:
        print(f"   Streaming approach failed ({e})")
        print("   Falling back to minimal sample...")
        
        try:
            # Ultra-conservative fallback
            epss_df = pd.read_parquet(
                epss_parquet,
                columns=['cve', 'date', 'epss'],
                engine='pyarrow'
            ).head(1_000_000)  # Only 1M rows
            
            epss_df['date'] = pd.to_datetime(epss_df['date'])
            cve_mask = epss_df['cve'].isin(target_cves)
            date_mask = (epss_df['date'] >= min_date) & (epss_df['date'] <= max_date)
            epss_df = epss_df[cve_mask & date_mask].copy()
            
            target_dates_set = set(target_dates_dt)
            exact_date_mask = epss_df['date'].dt.normalize().isin(target_dates_set)
            epss_df = epss_df[exact_date_mask]
            epss_df = epss_df.astype({"cve": "category"})
            
            print(f"   Minimal fallback: {len(epss_df):,} rows")
        except Exception as e2:
            raise RuntimeError(f"Could not load EPSS data even with minimal fallback: {e2}")
    
    print(f"EPSS data loaded and filtered: {len(epss_df):,} rows in {time.time() - start_time:.2f}s")
    
    # Validate EPSS data
    if len(epss_df) == 0:
        raise ValueError("No overlapping CVE×date pairs found between NetCDF and EPSS data")
    
    print(f"§5 Data alignment complete!")
    
    return ds, epss_df, cve2idx, date2tidx, C, T, H


def build_event_mask(epss_df: pd.DataFrame, cve2idx: Dict, date2tidx: Dict, 
                    C: int, T: int, H: int, threshold: float = 0.7, min_delta: float = 0.1,
                    min_relative_change: float = 0.5, require_sustained: bool = False,
                    min_baseline: float = 0.1):
    """
    Build binary event mask from EPSS data using REALISTIC early warning criteria.
    
    ENHANCED REALISTIC EVENT DEFINITION:
    An event occurs when ALL of the following criteria are met:
    1. EPSS[t] >= threshold (crosses high-risk threshold)
    2. EPSS[t] - EPSS[t-1] >= min_delta (absolute increase)
    3. (EPSS[t] - EPSS[t-1]) / EPSS[t-1] >= min_relative_change (relative increase)
    4. EPSS[t-1] >= min_baseline (not starting from near-zero)
    5. [Optional] Sustained increase over multiple days
    
    This prevents models from getting credit for:
    - Predicting already-high scores (persistence)
    - Tiny absolute changes that cross threshold
    - Huge relative changes from near-zero baselines
    - Single-day spikes that immediately revert
    
    Args:
        epss_df: EPSS ground truth data
        cve2idx: CVE to index mapping  
        date2tidx: Date to time index mapping
        C, T, H: Dimensions
        threshold: EPSS threshold for high-risk classification (default: 0.7)
        min_delta: Minimum absolute EPSS increase (default: 0.1)
        min_relative_change: Minimum relative increase (default: 0.5 = 50%)
        require_sustained: Require increase sustained for 2+ days (default: False)
        min_baseline: Minimum baseline EPSS to avoid near-zero artifacts (default: 0.1)
    
    Returns:
        event_rise: (C, T+H+1) boolean array where True = realistic early warning event
    """
    print("§5 Building REALISTIC early warning event mask...")
    start_time = time.time()
    
    # Estimate memory usage and warn if large
    mask_memory_gb = (C * (T + H + 1) * 4) / (1024**3)  # 4 bytes per float32
    print(f"   Event mask memory estimate: {mask_memory_gb:.2f} GB")
    print(f"   REALISTIC EVENT CRITERIA:")
    print(f"     1. EPSS ≥ {threshold} (high-risk threshold)")
    print(f"     2. Absolute increase ≥ {min_delta}")
    print(f"     3. Relative increase ≥ {min_relative_change*100:.0f}%")
    print(f"     4. Baseline EPSS ≥ {min_baseline} (avoid near-zero)")
    if require_sustained:
        print(f"     5. Sustained increase for 2+ days")
    
    if mask_memory_gb > 4.0:
        print("   WARNING: Large memory usage expected. Consider reducing data size.")
    
    # Create padded EPSS value array to handle horizon shifts safely
    print(f"   Creating EPSS value array shape: ({C}, {T + H + 1})")
    try:
        epss_array = np.full((C, T + H + 1), np.nan, dtype=np.float32)  # Initialize with NaN
    except MemoryError as e:
        raise MemoryError(f"Cannot allocate EPSS array of size {C}×{T + H + 1}: {e}")
    
    print("   Mapping EPSS data to indices...")
    # Vectorized assignment using fancy indexing
    cve_indices = epss_df["cve"].map(cve2idx).to_numpy(dtype=np.int32, na_value=-1)
    date_indices = epss_df["date"].astype('int64').map(date2tidx).to_numpy(dtype=np.int32, na_value=-1)
    epss_values = epss_df["epss"].to_numpy()
    
    # Filter out any unmapped entries
    valid_mask = (cve_indices >= 0) & (date_indices >= 0)
    cve_indices = cve_indices[valid_mask]
    date_indices = date_indices[valid_mask]
    epss_values = epss_values[valid_mask]
    
    print(f"   Mapping {len(cve_indices):,} valid EPSS records...")
    
    # Fill EPSS values into the array
    epss_array[cve_indices, date_indices] = epss_values
    
    print("   Computing REALISTIC early warning event detection...")
    
    # Compute EPSS changes (current - previous)
    epss_change = np.full_like(epss_array, np.nan)
    epss_change[:, 1:] = epss_array[:, 1:] - epss_array[:, :-1]  # change from t-1 to t
    
    # Compute relative changes
    epss_relative_change = np.full_like(epss_array, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        epss_relative_change[:, 1:] = epss_change[:, 1:] / epss_array[:, :-1]
    
    # REALISTIC EVENT CRITERIA (all must be true):
    
    # 1. Current EPSS >= threshold
    criterion_1 = epss_array >= threshold
    
    # 2. Absolute increase >= min_delta
    criterion_2 = epss_change >= min_delta
    
    # 3. Relative increase >= min_relative_change (and not NaN/Inf)
    criterion_3 = (epss_relative_change >= min_relative_change) & np.isfinite(epss_relative_change)
    
    # 4. Baseline EPSS >= min_baseline (previous day not near-zero)
    baseline_check = np.full_like(epss_array, False, dtype=bool)
    baseline_check[:, 1:] = epss_array[:, :-1] >= min_baseline
    criterion_4 = baseline_check
    
    # 5. [Optional] Sustained increase check
    if require_sustained:
        print("     Computing sustained increase requirement...")
        # Check if increase continues for at least one more day
        sustained_check = np.full_like(epss_array, False, dtype=bool)
        # For each position, check if next day also has positive change
        for t in range(1, epss_array.shape[1] - 1):
            current_increase = epss_change[:, t] > 0
            next_increase = epss_change[:, t + 1] > 0
            sustained_check[:, t] = current_increase & next_increase
        criterion_5 = sustained_check
    else:
        criterion_5 = np.ones_like(epss_array, dtype=bool)  # Always true if not required
    
    # Combine all criteria (ensure all are boolean arrays)
    event_mask = (criterion_1.astype(bool) & criterion_2.astype(bool) & criterion_3.astype(bool) & 
                  criterion_4.astype(bool) & criterion_5.astype(bool) & 
                  ~np.isnan(epss_array) & ~np.isnan(epss_change))
    
    # Extract event_rise (remove the extra padding dimension)
    event_rise = event_mask[:, 1:]  # shape (C, T+H) - remove first column used for change calculation
    
    # Free the large intermediate arrays
    del epss_array, epss_change, epss_relative_change, criterion_1, criterion_2, criterion_3, criterion_4, criterion_5, event_mask
    
    num_events = np.sum(event_rise)
    num_cves_with_events = np.sum(np.any(event_rise, axis=1))
    
    print(f"REALISTIC event mask built in {time.time() - start_time:.2f}s")
    print(f"Found {num_events} REALISTIC early warning events across {num_cves_with_events} CVEs")
    print(f"  (Required: ALL criteria must be met simultaneously)")
    
    return event_rise


def evaluate_per_forecast_corrected(ds: xr.Dataset, event_rise: np.ndarray, epss_df: pd.DataFrame,
                                   cve2idx: Dict, date2tidx: Dict, C: int, T: int, H: int, threshold: float = 0.7):
    """
    CORRECTED per-forecast evaluation with REALISTIC timing constraints.
    
    CRITICAL FIX: Ensures forecasts are made BEFORE events become observable.
    This eliminates the "stupid" model's artificial advantage from persistence forecasting.
    
    The key insight: A forecast is only valid if made when the current EPSS state
    does NOT already suggest the event will occur (no obvious uptrend).
    
    Args:
        ds: NetCDF dataset with predictions
        event_rise: Boolean array of events
        epss_df: EPSS ground truth data for timing validation
        cve2idx, date2tidx: Mapping dictionaries
        C, T, H: Dimensions
        threshold: EPSS threshold
    
    Returns:
        contingency_stats: Dict with TP, FP, FN, TN counts
        lead_time_stats: Dict with lead time analysis  
        forecast_arrays: Dict with P, O arrays for further analysis
    """
    print("§6 Evaluating per-forecast with ANTI-PERSISTENCE constraints...")
    start_time = time.time()
    
    # Estimate memory usage for forecast arrays
    forecast_memory_gb = (C * T * H * 4) / (1024**3)  # 4 bytes per bool/float32
    print(f"   Forecast arrays memory estimate: {forecast_memory_gb:.2f} GB each")
    
    if forecast_memory_gb > 2.0:
        print("   WARNING: Large memory usage for forecast arrays.")
    
    # Load masks
    print("   Loading validity masks...")
    eval_mask = ds.eval_mask.values.astype(bool)  # [C, T]
    mask_h = ds.mask_h.values.astype(bool)        # [C, T, H]
    
    print("   Converting predictions to probability space...")
    # Convert log predictions to probabilities (process in chunks if needed)
    if forecast_memory_gb > 3.0:
        print("     Using chunked conversion to save memory...")
        pred_probs = np.zeros((C, T, H), dtype=np.float32)
        chunk_size = max(1, C // 4)  # Process in quarters
        for c_start in range(0, C, chunk_size):
            c_end = min(c_start + chunk_size, C)
            pred_probs[c_start:c_end] = _log_to_prob(ds.pred.values[c_start:c_end])
    else:
        pred_probs = _log_to_prob(ds.pred.values)  # [C, T, H]
    
    # Reconstruct EPSS values at forecast times for timing validation
    print("   Reconstructing EPSS values for timing validation...")
    time_arr = ds.time.values  # Shape (C, T) 
    dates = pd.to_datetime(time_arr[0])  # All CVEs share same time grid
    
    # Create EPSS array for timing checks
    epss_at_forecast_time = np.full((C, T), np.nan, dtype=np.float32)
    
    # Map EPSS data to forecast time grid
    cve_indices = epss_df["cve"].map(cve2idx).to_numpy(dtype=np.int32, na_value=-1)
    date_indices = epss_df["date"].astype('int64').map(date2tidx).to_numpy(dtype=np.int32, na_value=-1)
    epss_values = epss_df["epss"].to_numpy()
    
    # Filter valid mappings
    valid_mask = (cve_indices >= 0) & (date_indices >= 0)
    cve_indices = cve_indices[valid_mask]
    date_indices = date_indices[valid_mask]
    epss_values = epss_values[valid_mask]
    
    # Fill EPSS values
    epss_at_forecast_time[cve_indices, date_indices] = epss_values
    
    print("   Building ANTI-PERSISTENCE forecast mask...")
    # Basic forecast mask: prediction >= threshold
    basic_forecast_mask = (pred_probs >= threshold) & mask_h & eval_mask[:, :, None]
    
    # CRITICAL: Anti-persistence constraint
    # A forecast is only valid if made when current EPSS does NOT already suggest the event
    print("   Applying ANTI-PERSISTENCE constraints...")
    
    anti_persistence_mask = np.ones_like(basic_forecast_mask, dtype=bool)
    
    for h in range(H):
        print(f"     Processing anti-persistence for horizon {h+1}...")
        
        # For each forecast at time t predicting event at t+h+1,
        # check that the event is NOT already obvious from EPSS[t]
        
        for t in range(T):
            # Current EPSS values at forecast time t
            current_epss = epss_at_forecast_time[:, t]  # Shape (C,)
            
            # ANTI-PERSISTENCE RULES:
            # 1. If current EPSS is already >= threshold, no credit for predicting high EPSS
            # 2. If current EPSS is close to threshold (within 0.1), no credit
            # 3. If there's recent upward trend, no credit for predicting continuation
            
            # Rule 1: Current EPSS already high
            already_high = current_epss >= (threshold - 0.05)  # Within 5% of threshold
            
            # Rule 2: Recent upward trend check (if we have previous data)
            recent_trend_up = np.zeros(C, dtype=bool)
            if t > 0:
                prev_epss = epss_at_forecast_time[:, t-1]
                # If EPSS increased in last day, don't give credit for predicting continuation
                recent_increase = (current_epss - prev_epss) > 0.02  # 2% increase
                recent_trend_up = recent_increase & ~np.isnan(prev_epss) & ~np.isnan(current_epss)
            
            # Rule 3: Check for sustained recent trend (if we have more history)
            sustained_trend_up = np.zeros(C, dtype=bool)
            if t > 2:
                # Check if EPSS has been increasing for 2+ days
                epss_t_minus_2 = epss_at_forecast_time[:, t-2]
                epss_t_minus_1 = epss_at_forecast_time[:, t-1]
                
                increase_1 = (epss_t_minus_1 - epss_t_minus_2) > 0.01
                increase_2 = (current_epss - epss_t_minus_1) > 0.01
                
                sustained_trend_up = (increase_1 & increase_2 & 
                                    ~np.isnan(epss_t_minus_2) & 
                                    ~np.isnan(epss_t_minus_1) & 
                                    ~np.isnan(current_epss))
            
            # Combine anti-persistence rules
            invalid_forecast = already_high | recent_trend_up | sustained_trend_up
            
            # Apply to forecast mask
            anti_persistence_mask[:, t, h] = ~invalid_forecast
    
    # Apply anti-persistence constraint
    P = basic_forecast_mask & anti_persistence_mask
    
    print("   Building observation mask O...")
    # O[c,t,h] = 1 if event occurred at target date τ = t+h+1
    try:
        O = np.zeros((C, T, H), dtype=bool)
    except MemoryError as e:
        raise MemoryError(f"Cannot allocate observation mask of size {C}×{T}×{H}: {e}")
        
    for h in range(H):
        # For horizon h, target dates are at indices t+h+1
        # event_rise has shape (C, T+H), so we slice appropriately
        target_start = h + 1
        target_end = h + 1 + T
        if target_end <= event_rise.shape[1]:
            O[:, :, h] = event_rise[:, target_start:target_end]
    
    # Apply same validity masks to observations
    O = O & mask_h & eval_mask[:, :, None]
    
    # ADDITIONAL REALISTIC CONSTRAINT: Minimum useful lead time
    print("   Enforcing minimum useful lead time...")
    min_useful_lead_time = 4  # At least 4 days lead time for useful early warning
    
    for h in range(min(min_useful_lead_time, H)):
        # For horizons 0, 1, 2, 3 (lead times 1, 2, 3, 4 days), set to invalid
        P[:, :, h] = False
        O[:, :, h] = False
    
    print("   Computing contingency table...")
    # Vectorized contingency table computation
    TP = np.sum(P & O)
    FP = np.sum(P & ~O)
    FN = np.sum(~P & O)
    TN = np.sum(~P & ~O)
    
    # Count how many forecasts were invalidated by anti-persistence
    total_basic_forecasts = np.sum(basic_forecast_mask)
    total_anti_persistence_forecasts = np.sum(P)
    invalidated_by_timing = total_basic_forecasts - total_anti_persistence_forecasts
    
    print(f"Per-forecast evaluation completed in {time.time() - start_time:.2f}s")
    print(f"Contingency Table (ANTI-PERSISTENCE evaluation):")
    print(f"  True Positives (TP):  {TP}")
    print(f"  False Positives (FP): {FP}")
    print(f"  False Negatives (FN): {FN}")
    print(f"  True Negatives (TN):  {TN}")
    print(f"  Minimum lead time: {min_useful_lead_time} days")
    print(f"  Forecasts invalidated by anti-persistence: {invalidated_by_timing:,}")
    
    # Lead time analysis for TP forecasts
    print("§7 Computing lead time statistics...")
    tp_mask = P & O
    if np.any(tp_mask):
        # Lead time is h+1 for each TP forecast
        c_idx, t_idx, h_idx = np.where(tp_mask)
        lead_times = h_idx + 1  # h+1 is the lead time
        
        pct_ge7 = 100 * np.mean(lead_times >= 7)
        median_lt = np.median(lead_times)
        mean_lt = np.mean(lead_times)
        max_lt = np.max(lead_times)
        min_lt = np.min(lead_times)
        
        print(f"Lead time statistics from {len(lead_times)} TP forecasts:")
        print(f"  % warned ≥7 days: {pct_ge7:.1f}%")
        print(f"  Median lead time: {median_lt:.1f} days")
        print(f"  Mean lead time: {mean_lt:.1f} days")
        print(f"  Min lead time: {min_lt:.1f} days")
        print(f"  Max lead time: {max_lt:.1f} days")
        
        lead_time_stats = {
            "num_tp_forecasts": len(lead_times),
            "pct_warned_ge7d": pct_ge7,
            "median_lead": median_lt,
            "mean_lead": mean_lt,
            "min_lead": min_lt,
            "max_lead": max_lt,
            "all_leads": lead_times.tolist()
        }
    else:
        print("No TP forecasts found - no lead time statistics available")
        lead_time_stats = {
            "num_tp_forecasts": 0,
            "pct_warned_ge7d": 0.0,
            "median_lead": 0.0,
            "mean_lead": 0.0,
            "min_lead": 0.0,
            "max_lead": 0.0,
            "all_leads": []
        }
    
    contingency_stats = {
        "TP": int(TP),
        "FP": int(FP), 
        "FN": int(FN),
        "TN": int(TN),
        "invalidated_by_timing": int(invalidated_by_timing)
    }
    
    forecast_arrays = {
        "P": P,
        "O": O,
        "pred_probs": pred_probs
    }
    
    return contingency_stats, lead_time_stats, forecast_arrays


def compute_corrected_metrics(contingency_stats: Dict, lead_time_stats: Dict):
    """
    Compute corrected skill metrics based on per-forecast evaluation.
    """
    print("§8 Computing corrected skill metrics...")
    
    TP = contingency_stats["TP"]
    FP = contingency_stats["FP"]
    FN = contingency_stats["FN"]
    TN = contingency_stats["TN"]
    
    print(f"Final Contingency Table (per-forecast):")
    print(f"  True Positives (TP):  {TP}")
    print(f"  False Positives (FP): {FP}")
    print(f"  False Negatives (FN): {FN}")
    print(f"  True Negatives (TN):  {TN}")
    
    # Compute metrics
    precision = _safe_divide(TP, TP + FP, 0.0)
    recall = _safe_divide(TP, TP + FN, 0.0)
    f1 = _safe_divide(2 * precision * recall, precision + recall, 0.0)
    
    # Specificity and other metrics
    specificity = _safe_divide(TN, TN + FP, 0.0)
    
    # TSS (True Skill Statistic) = Sensitivity + Specificity - 1
    tss = recall + specificity - 1
    
    # CSI (Critical Success Index) = TP / (TP + FP + FN)
    csi = _safe_divide(TP, TP + FP + FN, 0.0)
    
    # Event rate and forecast rate
    event_rate = _safe_divide(TP + FN, TP + FN + TN + FP, 0.0)
    forecast_rate = _safe_divide(TP + FP, TP + FN + TN + FP, 0.0)
    
    metrics = {
        **contingency_stats,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "TSS": tss,
        "CSI": csi,
        "Event_Rate": event_rate,
        "Forecast_Rate": forecast_rate,
        **{k: v for k, v in lead_time_stats.items() if k != "all_leads"}
    }
    
    return metrics


def compute_probabilistic_metrics(forecast_arrays: Dict):
    """
    Compute probabilistic metrics (PR-AUC, ROC-AUC) from forecast arrays.
    """
    print("§9 Computing probabilistic metrics...")
    
    P = forecast_arrays["P"]
    O = forecast_arrays["O"]
    pred_probs = forecast_arrays["pred_probs"]
    
    # Flatten arrays for sklearn metrics
    y_true = O.ravel()
    y_scores = pred_probs.ravel()
    
    # Only use valid forecasts (where masks allow)
    valid_mask = P.ravel() | O.ravel()  # Either forecast was made or event occurred
    
    if np.sum(valid_mask) > 0 and np.sum(y_true[valid_mask]) > 0:
        try:
            pr_auc = average_precision_score(y_true[valid_mask], y_scores[valid_mask])
            roc_auc = roc_auc_score(y_true[valid_mask], y_scores[valid_mask])
            
            print(f"Probabilistic metrics:")
            print(f"  PR-AUC:  {pr_auc:.3f}")
            print(f"  ROC-AUC: {roc_auc:.3f}")
            
            return {"PR_AUC": pr_auc, "ROC_AUC": roc_auc}
        except Exception as e:
            print(f"Could not compute probabilistic metrics: {e}")
            return {"PR_AUC": 0.0, "ROC_AUC": 0.5}
    else:
        print("Insufficient data for probabilistic metrics")
        return {"PR_AUC": 0.0, "ROC_AUC": 0.5}


def save_corrected_results(metrics: Dict, output_path: Path):
    """Save corrected evaluation results."""
    print("§10 Saving corrected results...")
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_csv(output_path)
    print(f"Corrected results saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CORRECTED EARLY WARNING SYSTEM EVALUATION SUMMARY")
    print("="*60)
    print(f"Per-Forecast Contingency Table:")
    print(f"  True Positives (TP):  {metrics['TP']:6d}")
    print(f"  False Positives (FP): {metrics['FP']:6d}")
    print(f"  False Negatives (FN): {metrics['FN']:6d}")
    print(f"  True Negatives (TN):  {metrics['TN']:6d}")
    print(f"\nCorrected Skill Metrics:")
    print(f"  Precision:    {metrics['Precision']:.3f}")
    print(f"  Recall:       {metrics['Recall']:.3f}")
    print(f"  F1 Score:     {metrics['F1']:.3f}")
    print(f"  Specificity:  {metrics['Specificity']:.3f}")
    print(f"  TSS:          {metrics['TSS']:.3f}")
    print(f"  CSI:          {metrics['CSI']:.3f}")
    if "PR_AUC" in metrics:
        print(f"  PR-AUC:       {metrics['PR_AUC']:.3f}")
        print(f"  ROC-AUC:      {metrics['ROC_AUC']:.3f}")
    print(f"\nForecast Characteristics:")
    print(f"  Event Rate:    {metrics['Event_Rate']:.4f}")
    print(f"  Forecast Rate: {metrics['Forecast_Rate']:.4f}")
    print(f"\nLead-time Statistics:")
    print(f"  TP forecasts:     {metrics['num_tp_forecasts']}")
    print(f"  % warned ≥7 days: {metrics['pct_warned_ge7d']:.1f}%")
    print(f"  Median lead time: {metrics['median_lead']:.1f} days")
    print(f"  Mean lead time:   {metrics['mean_lead']:.1f} days")


def main():
    """Main corrected evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description='CORRECTED early warning system evaluation for EPSS forecasting'
    )
    parser.add_argument('ncfile', type=Path, 
                       help='Input NetCDF (.nc) prediction file')
    parser.add_argument('--epss-parquet', type=Path, 
                       default=None,
                       help='EPSS ground truth parquet file')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='EPSS threshold for event detection (default: 0.7)')
    parser.add_argument('--min-delta', type=float, default=0.1,
                       help='Minimum absolute EPSS increase required (default: 0.1)')
    parser.add_argument('--min-relative-change', type=float, default=0.5,
                       help='Minimum relative EPSS increase required (default: 0.5 = 50%%)')
    parser.add_argument('--min-baseline', type=float, default=0.1,
                       help='Minimum baseline EPSS to avoid near-zero artifacts (default: 0.1)')
    parser.add_argument('--require-sustained', action='store_true',
                       help='Require sustained increase for 2+ days')
    parser.add_argument('--output', type=Path, default=Path('ews_metrics_corrected.csv'),
                       help='Output CSV file for results')
    parser.add_argument('--max-memory-gb', type=float, default=8.0,
                       help='Maximum memory usage in GB (default: 8.0)')
    
    args = parser.parse_args()
    
    # Check system memory
    try:
        import psutil
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)
        total_gb = memory_info.total / (1024**3)
        print(f"System Memory: {available_gb:.1f} GB available / {total_gb:.1f} GB total")
        
        if available_gb < args.max_memory_gb:
            print(f"WARNING: Available memory ({available_gb:.1f} GB) < requested limit ({args.max_memory_gb:.1f} GB)")
            print("Consider reducing --max-memory-gb or closing other applications")
    except ImportError:
        print("psutil not available - cannot check system memory")
    
    # Validate inputs
    if not args.ncfile.exists():
        raise FileNotFoundError(f"NetCDF file not found: {args.ncfile}")
    
    if args.threshold <= 0 or args.threshold >= 1:
        raise ValueError(f"Threshold must be between 0 and 1, got: {args.threshold}")
    
    if args.min_delta <= 0:
        raise ValueError(f"Minimum delta must be positive, got: {args.min_delta}")
    
    if args.min_relative_change <= 0:
        raise ValueError(f"Minimum relative change must be positive, got: {args.min_relative_change}")
    
    if args.min_baseline < 0:
        raise ValueError(f"Minimum baseline must be non-negative, got: {args.min_baseline}")
    
    # Set default EPSS parquet path
    if args.epss_parquet is None:
        project_root = _find_project_root()
        args.epss_parquet = project_root / "data/epss/processed/epss_processed.parquet"
    
    if not args.epss_parquet.exists():
        raise FileNotFoundError(f"EPSS parquet file not found: {args.epss_parquet}")
    
    print("Starting REALISTIC Per-Forecast Early Warning System Evaluation...")
    print(f"NetCDF file: {args.ncfile}")
    print(f"EPSS parquet: {args.epss_parquet}")
    print(f"REALISTIC CRITERIA:")
    print(f"  Threshold: {args.threshold}")
    print(f"  Min absolute increase: {args.min_delta}")
    print(f"  Min relative increase: {args.min_relative_change*100:.0f}%")
    print(f"  Min baseline: {args.min_baseline}")
    print(f"  Require sustained: {args.require_sustained}")
    print(f"Memory limit: {args.max_memory_gb:.1f} GB")
    
    total_start = time.time()
    
    try:
        # Load and align data
        ds, epss_df, cve2idx, date2tidx, C, T, H = load_and_align_data(
            args.ncfile, args.epss_parquet, args.threshold
        )
        
        # Estimate total memory requirements
        event_mask_gb = (C * (T + H + 1) * 4) / (1024**3)  # Updated for float32
        forecast_gb = (C * T * H * 4) / (1024**3)
        total_estimated_gb = event_mask_gb + forecast_gb * 3  # P, O, pred_probs
        
        print(f"\nMemory Requirements Estimate:")
        print(f"  Event mask: {event_mask_gb:.2f} GB")
        print(f"  Forecast arrays: {forecast_gb:.2f} GB each (×3 = {forecast_gb*3:.2f} GB)")
        print(f"  Total estimated: {total_estimated_gb:.2f} GB")
        
        if total_estimated_gb > args.max_memory_gb:
            print(f"WARNING: Estimated memory ({total_estimated_gb:.2f} GB) exceeds limit ({args.max_memory_gb:.1f} GB)")
            print("Proceeding with memory-efficient chunked processing...")
        
        # Build REALISTIC early warning event mask
        event_rise = build_event_mask(
            epss_df, cve2idx, date2tidx, C, T, H, 
            args.threshold, args.min_delta, args.min_relative_change,
            args.require_sustained, args.min_baseline
        )
        
        # Evaluate per-forecast with ANTI-PERSISTENCE logic
        contingency_stats, lead_time_stats, forecast_arrays = evaluate_per_forecast_corrected(
            ds, event_rise, epss_df, cve2idx, date2tidx, C, T, H, args.threshold
        )
        
        # Compute corrected metrics
        metrics = compute_corrected_metrics(contingency_stats, lead_time_stats)
        
        # Add configuration to results
        metrics.update({
            "threshold": args.threshold,
            "min_delta": args.min_delta,
            "min_relative_change": args.min_relative_change,
            "min_baseline": args.min_baseline,
            "require_sustained": args.require_sustained
        })
        
        # Compute probabilistic metrics
        prob_metrics = compute_probabilistic_metrics(forecast_arrays)
        metrics.update(prob_metrics)
        
        # Save results
        save_corrected_results(metrics, args.output)
        
        print(f"\nTotal execution time: {time.time() - total_start:.2f} seconds")
        print("REALISTIC per-forecast evaluation complete!")
        
    except Exception as e:
        print(f"\n❌ REALISTIC evaluation failed with error: {e}")
        raise
    finally:
        # Cleanup
        if 'ds' in locals():
            ds.close()


if __name__ == '__main__':
    main() 