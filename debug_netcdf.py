#!/usr/bin/env python3
"""Debug NetCDF structure and time coordinate issues"""

import xarray as xr
import numpy as np
import pandas as pd

# Load the NetCDF file
print("Loading NetCDF file...")
ds = xr.open_dataset('ml_pipeline/results/predictions/predictions_stream.nc')

print("Dataset structure:")
print(ds)
print("\n" + "="*60)

print("Time coordinate analysis:")
print(f"Time shape: {ds.time.shape}")
print(f"Time dtype: {ds.time.dtype}")
print(f"Time dims: {ds.time.dims}")

print("\nFirst CVE time samples:")
print(ds.time[0, :10].values)

print("\nTime coordinate as datetime64:")
time_sample = ds.time[0, :10].values
print(f"Raw values: {time_sample}")
print(f"As datetime64: {time_sample.view('datetime64[ns]')}")

print("\nChecking for valid dates:")
valid_times = ds.time[0, :].values
valid_times_dt = valid_times.view('datetime64[ns]')
non_nat = valid_times_dt[~pd.isna(valid_times_dt)]
if len(non_nat) > 0:
    print(f"Valid date range: {non_nat.min()} to {non_nat.max()}")
else:
    print("No valid dates found!")

print("\nEval mask for first CVE:")
eval_mask = ds.eval_mask[0, :].values
print(f"Valid timesteps: {eval_mask.sum()} / {len(eval_mask)}")
print(f"Valid indices: {np.where(eval_mask)[0][:10]}")

print("\nCVE samples:")
print(ds.cve[:5].values)

print("\nData variables info:")
for var in ds.data_vars:
    print(f"{var}: shape={ds[var].shape}, dtype={ds[var].dtype}")

print("\nCoordinates info:")
for coord in ds.coords:
    print(f"{coord}: shape={ds[coord].shape}, dtype={ds[coord].dtype}") 