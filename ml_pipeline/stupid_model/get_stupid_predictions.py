#!/usr/bin/env python3
"""
Stupid Model Predictions Generator

This script generates predictions for a naive baseline model that simply repeats
the current day's true EPSS value for the entire 30-day prediction horizon.

This serves as a critical baseline for evaluating the performance of sophisticated
LSTM models by establishing the minimum performance threshold.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from t3_spark.session import get_spark_session
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StupidModelGenerator:
    """Generates naive baseline predictions by repeating current EPSS values."""
    
    def __init__(self):
        self.project_root = self._find_project_root()
        self.nc_path = self.project_root / "ml_pipeline/stupid_model/predictions_stream_copy.nc"
        self.epss_path = self.project_root / "data/epss/processed/epss_processed.parquet"
        self.output_path = self.project_root / "ml_pipeline/stupid_model/stupid_predictions.nc"
        
        logger.info(f"Initialized with:")
        logger.info(f"  NetCDF path: {self.nc_path}")
        logger.info(f"  EPSS path: {self.epss_path}")
        logger.info(f"  Output path: {self.output_path}")
    
    def _find_project_root(self):
        """Find project root by looking for ml_pipeline directory."""
        current = Path(__file__).resolve().parent
        while current != current.parent:
            if (current / "ml_pipeline").exists():
                return current
            current = current.parent
        raise FileNotFoundError("Could not find project root")
    
    def generate_stupid_predictions(self):
        """Main method to generate stupid model predictions."""
        logger.info("Starting stupid model prediction generation...")
        
        # Step 1: Load NetCDF data
        logger.info("Step 1: Loading NetCDF predictions file...")
        ds = xr.open_dataset(self.nc_path)
        logger.info(f"  NetCDF shape: {ds.pred.shape}")
        logger.info(f"  CVEs: {ds.sizes['cve']}, Time points: {ds.sizes['time']}, Horizon: {ds.sizes['horizon']}")
        
        # Step 2: Initialize Spark and load EPSS data
        logger.info("Step 2: Loading EPSS data with Spark...")
        spark = get_spark_session()
        epss_spark_df = spark.read.parquet(str(self.epss_path))
        logger.info(f"  EPSS rows: {epss_spark_df.count():,}")
        
        # Convert to pandas for easier manipulation (we'll filter first to reduce size)
        logger.info("Step 3: Converting relevant EPSS data to pandas...")
        # Get unique CVEs from NetCDF
        nc_cves = set(ds.cve.values)
        epss_pandas_df = epss_spark_df.filter(
            epss_spark_df.cve.isin(list(nc_cves))
        ).toPandas()
        epss_pandas_df['date'] = pd.to_datetime(epss_pandas_df['date'])
        spark.stop()
        
        logger.info(f"  Filtered EPSS rows: {len(epss_pandas_df):,}")
        logger.info(f"  EPSS date range: {epss_pandas_df['date'].min()} to {epss_pandas_df['date'].max()}")
        
        # Step 4: Validate data alignment
        logger.info("Step 4: Validating data alignment...")
        self._validate_data_alignment(ds, epss_pandas_df)
        
        # Step 5: Generate stupid predictions
        logger.info("Step 5: Generating stupid model predictions...")
        stupid_predictions = self._create_stupid_predictions(ds, epss_pandas_df)
        
        # Step 6: Validate shapes and attach to dataset
        logger.info("Step 6: Validating shapes and creating output dataset...")
        output_ds = self._create_output_dataset(ds, stupid_predictions)
        
        # Step 7: Save the result
        logger.info("Step 7: Saving stupid predictions...")
        # Use compression to keep file size similar to original
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in output_ds.data_vars}
        # Preserve original time encoding if present
        if "time" in output_ds.coords:
            encoding["time"] = {"dtype": "int64"}

        output_ds.to_netcdf(self.output_path, encoding=encoding)
        logger.info(f"  Saved to: {self.output_path}")
        
        # Step 8: Final validation
        logger.info("Step 8: Final validation...")
        self._final_validation(output_ds, ds)
        
        logger.info("✅ Stupid model generation completed successfully!")
        return output_ds
    
    def _validate_data_alignment(self, ds, epss_df):
        """Validate that NetCDF and EPSS data can be properly aligned."""
        logger.info("  Checking CVE alignment...")
        
        nc_cves = set(ds.cve.values)
        epss_cves = set(epss_df['cve'].unique())
        
        common_cves = nc_cves.intersection(epss_cves)
        missing_in_epss = nc_cves - epss_cves
        
        logger.info(f"    NetCDF CVEs: {len(nc_cves):,}")
        logger.info(f"    EPSS CVEs: {len(epss_cves):,}")
        logger.info(f"    Common CVEs: {len(common_cves):,}")
        logger.info(f"    Missing in EPSS: {len(missing_in_epss):,}")
        
        if len(missing_in_epss) > 0:
            logger.warning(f"  Some CVEs from NetCDF not found in EPSS data: {list(missing_in_epss)[:5]}...")
        
        # Check time alignment for a sample CVE
        sample_cve = list(common_cves)[0]
        cve_idx = list(ds.cve.values).index(sample_cve)
        nc_times = pd.to_datetime(ds.time.values[cve_idx])
        nc_times_clean = nc_times[~pd.isna(nc_times)]
        
        cve_epss = epss_df[epss_df['cve'] == sample_cve]
        
        logger.info(f"  Sample CVE {sample_cve}:")
        logger.info(f"    NetCDF time range: {nc_times_clean.min()} to {nc_times_clean.max()}")
        logger.info(f"    EPSS time range: {cve_epss['date'].min()} to {cve_epss['date'].max()}")
    
    def _create_stupid_predictions(self, ds, epss_df):
        """Create the actual stupid predictions."""
        logger.info("  Creating stupid predictions array...")
        
        # Initialize with same shape as original predictions
        stupid_preds = np.full_like(ds.pred.values, np.nan, dtype=np.float32)
        
        # Create EPSS lookup for efficiency
        epss_lookup = {}
        for _, row in epss_df.iterrows():
            key = (row['cve'], row['date'].date())
            # Convert EPSS to log space (add small epsilon to avoid log(0))
            epss_log = np.log(row['epss'] + 1e-6)
            epss_lookup[key] = epss_log
        
        logger.info(f"  Created EPSS lookup with {len(epss_lookup):,} entries")
        
        # Process each CVE
        processed_count = 0
        filled_predictions = 0
        
        for cve_idx, cve in enumerate(ds.cve.values):
            if cve_idx % 1000 == 0:
                logger.info(f"    Processing CVE {cve_idx:,}/{len(ds.cve):,}...")
            
            # Get valid evaluation times for this CVE
            eval_mask = ds.eval_mask.values[cve_idx, :].astype(bool)
            time_coords = pd.to_datetime(ds.time.values[cve_idx, :])
            
            for time_idx in np.where(eval_mask)[0]:
                anchor_date = time_coords[time_idx]
                
                if pd.isna(anchor_date):
                    continue
                
                # Look up EPSS value at time t (anchor date)
                lookup_key = (cve, anchor_date.date())
                if lookup_key in epss_lookup:
                    epss_log_value = epss_lookup[lookup_key]
                    
                    # Fill entire horizon with this value
                    # Use mask_h to respect original masking
                    horizon_mask = ds.mask_h.values[cve_idx, time_idx, :].astype(bool)
                    
                    for horizon_idx in range(len(horizon_mask)):
                        if horizon_mask[horizon_idx]:
                            stupid_preds[cve_idx, time_idx, horizon_idx] = epss_log_value
                            filled_predictions += 1
            
            processed_count += 1
        
        logger.info(f"  Processed {processed_count:,} CVEs")
        logger.info(f"  Filled {filled_predictions:,} prediction values")
        
        return stupid_preds
    
    def _create_output_dataset(self, original_ds, stupid_predictions):
        """Create output dataset with stupid predictions."""
        logger.info("  Creating output dataset...")
        
        # Validate shapes
        assert stupid_predictions.shape == original_ds.pred.shape, \
            f"Shape mismatch: {stupid_predictions.shape} vs {original_ds.pred.shape}"
        
        # Create new dataset with stupid predictions
        output_ds = original_ds.copy(deep=True)
        output_ds = output_ds.assign(pred=(['cve', 'time', 'horizon'], stupid_predictions))
        
        # Add metadata
        output_ds.attrs['title'] = 'Stupid Model Predictions'
        output_ds.attrs['description'] = 'Naive baseline predictions that repeat current EPSS value for 30-day horizon'
        output_ds.attrs['model_type'] = 'stupid_baseline'
        
        logger.info(f"  Output dataset shape: {output_ds.pred.shape}")
        
        return output_ds
    
    def _final_validation(self, output_ds, original_ds):
        """Perform final validation of the output."""
        logger.info("  Performing final validation...")
        
        # Check dimensions
        assert output_ds.pred.dims == original_ds.pred.dims
        assert output_ds.pred.shape == original_ds.pred.shape
        assert output_ds.true.shape == original_ds.true.shape
        
        # Check that we have predictions where eval_mask is True
        eval_mask = original_ds.eval_mask.values.astype(bool)
        mask_h = original_ds.mask_h.values.astype(bool)
        
        valid_positions = eval_mask[:, :, np.newaxis] & mask_h
        stupid_values = output_ds.pred.values[valid_positions]
        
        finite_count = np.sum(np.isfinite(stupid_values))
        total_valid = np.sum(valid_positions)
        
        logger.info(f"  Valid positions: {total_valid:,}")
        logger.info(f"  Finite predictions: {finite_count:,}")
        logger.info(f"  Coverage: {finite_count/total_valid*100:.1f}%")
        
        # Check value ranges (should be in log space)
        finite_values = stupid_values[np.isfinite(stupid_values)]
        if len(finite_values) > 0:
            logger.info(f"  Prediction range: {finite_values.min():.3f} to {finite_values.max():.3f}")
        
        # Compare a sample with true values
        sample_idx = np.where(valid_positions)
        if len(sample_idx[0]) > 0:
            sample_pos = (sample_idx[0][0], sample_idx[1][0], sample_idx[2][0])
            stupid_val = output_ds.pred.values[sample_pos]
            true_val = original_ds.true.values[sample_pos]
            logger.info(f"  Sample comparison - Stupid: {stupid_val:.3f}, True: {true_val:.3f}")


def main():
    """Main execution function."""
    generator = StupidModelGenerator()
    output_ds = generator.generate_stupid_predictions()
    return output_ds


if __name__ == "__main__":
    main()