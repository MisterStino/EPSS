#!/usr/bin/env python3
"""Quick test of event marking on a subset of CVEs."""

import sys
import time
sys.path.append('.')

from eval_plot_events import mark_all_events, validate_event_marking
import xarray as xr
from pathlib import Path

def quick_test():
    print("Loading dataset...")
    ds = xr.open_dataset("models/predictions_stream_sus_lstm.nc")
    
    # Test on first 100 CVEs
    print("Creating subset (first 100 CVEs)...")
    ds_subset = ds.isel(cve=slice(0, 100))
    
    print("Running event marking...")
    start_time = time.time()
    
    event_df = mark_all_events(
        ds_subset, 
        threshold=0.7,
        test_only=True,
        horizon_idx=0
    )
    
    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.2f} seconds")
    
    # Show results
    print("\nResults summary:")
    print(f"Total CVEs: {len(event_df)}")
    print(f"CVEs with events: {(event_df['num_events'] > 0).sum()}")
    print(f"Total events: {event_df['num_events'].sum()}")
    
    # Show CVEs with events
    with_events = event_df[event_df['num_events'] > 0]
    if len(with_events) > 0:
        print(f"\nCVEs with events:")
        for _, row in with_events.iterrows():
            print(f"  {row['cve_id']}: {row['num_events']} events at times {row['tau_true']}")
    
    # Save subset results
    event_df.to_csv("event_markings_subset.csv", index=False)
    print(f"\nSaved results to event_markings_subset.csv")
    
    # Quick validation on a few CVEs
    if len(with_events) > 0:
        sample_cves = with_events['cve_id'].head(2).tolist()
        print(f"\nValidating on: {sample_cves}")
        
        validate_event_marking(
            ds_subset, event_df,
            sample_cves=sample_cves,
            output_dir=Path("validation_plots_subset")
        )
        print("Validation plots saved to validation_plots_subset/")
    
    ds.close()
    ds_subset.close()
    
    return event_df

if __name__ == "__main__":
    event_df = quick_test() 