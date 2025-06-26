# Predictions Directory

This directory contains model prediction outputs in NetCDF format.

## Files
- `predictions_stream.nc` - Main prediction file with xarray Dataset containing:
  - **pred**: Model predictions in log space `[CVE, time, horizon]`
  - **true**: Ground truth EPSS values in log space `[CVE, time, horizon]`
  - **mask_h**: Horizon validity mask `[CVE, time, horizon]`
  - **eval_mask**: Test window mask `[CVE, time]`
  - **Coordinates**: CVE IDs, calendar dates, horizon indices

## Usage
Load with: `ds = xr.open_dataset("predictions_stream.nc")`

Convert to probability space: `prob = np.clip(np.exp(-log_values) - 1e-6, 0.0, 1.0)` 