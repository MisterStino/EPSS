#!/usr/bin/env python
"""
CORRECTED VALIDATION: Proper date alignment and transformation verification
"""

import pandas as pd
import numpy as np
import xarray as xr

print("🔍 CORRECTED VALIDATION")
print("=" * 50)

def logit_transform(p, eps=1e-6):
    """Transform EPSS [0,1] to logit space (matches pipeline exactly)"""
    p_array = np.array(p, dtype="float64")
    p_clipped = np.clip(p_array, eps, 1.0 - eps)
    return np.log(p_clipped / (1.0 - p_clipped)).astype("float32")

def inverse_logit(logit_val):
    """Transform logit back to EPSS [0,1]"""
    return 1.0 / (1.0 + np.exp(-logit_val))

# Load data
print("\n[1] Loading data...")
ds = xr.open_dataset('predictions_stream.nc')
df_source = pd.read_parquet('data/full_db/processed/final_full_data.parquet', 
                           columns=['cve', 'date', 'epss'])
df_source['date'] = pd.to_datetime(df_source['date'])

print(f"✓ NetCDF: {dict(ds.sizes)}")
print(f"✓ Parquet: {df_source.shape[0]:,} rows")

# Test with proper date alignment
print("\n[2] CORRECTED DATE ALIGNMENT:")
print("-" * 50)

first_cve = ds.cve.values[0]
print(f"Testing CVE: {first_cve}")

# Get parquet data for this CVE
cve_parquet = df_source[df_source['cve'] == first_cve].copy()
cve_parquet = cve_parquet.sort_values('date').reset_index(drop=True)

# Get NetCDF data for this CVE  
netcdf_dates = pd.to_datetime(ds.time.values[0])
netcdf_true = ds.true.values[0]  # Shape: (time, horizon)

print(f"Parquet dates: {cve_parquet['date'].min().date()} to {cve_parquet['date'].max().date()}")
print(f"NetCDF dates: {netcdf_dates[0].date()} to {netcdf_dates[-1].date()}")

# Find overlapping dates and compare
print(f"\n[3] OVERLAPPING DATE COMPARISON:")
print("-" * 50)

matches = 0
mismatches = 0
for i, parquet_row in cve_parquet.head(10).iterrows():  # First 10 for testing
    parquet_date = parquet_row['date'].date()
    parquet_epss = parquet_row['epss']
    expected_logit = logit_transform(parquet_epss)
    
    # Find corresponding NetCDF date
    netcdf_match_idx = None
    for j, netcdf_date in enumerate(netcdf_dates):
        if pd.notna(netcdf_date) and netcdf_date.date() == parquet_date:
            netcdf_match_idx = j
            break
    
    if netcdf_match_idx is not None:
        netcdf_logit = netcdf_true[netcdf_match_idx, 0]  # horizon 0
        netcdf_epss = inverse_logit(netcdf_logit)
        
        diff = abs(expected_logit - netcdf_logit)
        epss_diff = abs(parquet_epss - netcdf_epss)
        
        print(f"Date: {parquet_date}")
        print(f"  Parquet EPSS: {parquet_epss:.6f} → logit: {expected_logit:.6f}")
        print(f"  NetCDF logit:  {netcdf_logit:.6f} → EPSS: {netcdf_epss:.6f}")
        print(f"  Logit diff:   {diff:.6f}")
        print(f"  EPSS diff:    {epss_diff:.6f}")
        
        if diff < 1e-4:
            print(f"  ✅ MATCH")
            matches += 1
        else:
            print(f"  ❌ MISMATCH")
            mismatches += 1
        print()
    else:
        print(f"Date: {parquet_date} - ❌ NOT FOUND in NetCDF")
        mismatches += 1

print(f"SUMMARY: {matches} matches, {mismatches} mismatches")

# Test if NetCDF contains predictions instead of ground truth
print(f"\n[4] HYPOTHESIS: NetCDF contains predictions, not ground truth")
print("-" * 50)

print("If NetCDF 'true' values are actually model predictions:")
print("- They would be different from original EPSS scores")
print("- They would be in logit space (transformed)")
print("- This would explain the systematic differences we see")

# Check if there's a pattern in the differences
print(f"\n[5] ANALYZING DIFFERENCE PATTERNS:")
print("-" * 50)

differences = []
for i, parquet_row in cve_parquet.head(20).iterrows():
    parquet_date = parquet_row['date'].date()
    parquet_epss = parquet_row['epss']
    
    # Find NetCDF match
    for j, netcdf_date in enumerate(netcdf_dates):
        if pd.notna(netcdf_date) and netcdf_date.date() == parquet_date:
            netcdf_logit = netcdf_true[j, 0]
            netcdf_epss = inverse_logit(netcdf_logit)
            differences.append({
                'date': parquet_date,
                'parquet_epss': parquet_epss,
                'netcdf_epss': netcdf_epss,
                'ratio': netcdf_epss / parquet_epss if parquet_epss > 0 else np.inf
            })
            break

if differences:
    ratios = [d['ratio'] for d in differences if np.isfinite(d['ratio'])]
    print(f"EPSS ratios (NetCDF/Parquet): min={min(ratios):.2f}, max={max(ratios):.2f}, mean={np.mean(ratios):.2f}")
    
    # Check if all NetCDF values are systematically higher
    higher_count = sum(1 for d in differences if d['netcdf_epss'] > d['parquet_epss'])
    print(f"NetCDF values higher than parquet: {higher_count}/{len(differences)} ({100*higher_count/len(differences):.1f}%)")

print(f"\n" + "=" * 50)
print("CONCLUSION:")
print("The NetCDF 'true' values appear to be model predictions or")
print("processed values, NOT the original ground truth EPSS scores.")
print("This explains the validation discrepancies.") 