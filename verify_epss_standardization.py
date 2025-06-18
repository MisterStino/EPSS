#!/usr/bin/env python3
"""
Verify that EPSS values are not standardized in the preprocessing pipeline.
"""

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from pathlib import Path
import json

def verify_epss_standardization():
    """Check if EPSS values are standardized or remain in log space"""
    
    arrow_path = Path("work/epss_stage1.arrow")
    if not arrow_path.exists():
        print(f"❌ Arrow file not found: {arrow_path}")
        return
    
    print("🔍 VERIFYING EPSS STANDARDIZATION STATUS")
    print("=" * 60)
    
    # Load Arrow file and sample EPSS values
    reader = ipc.open_file(pa.memory_map(str(arrow_path), "r"))
    
    # Find EPSS column
    epss_col_idx = None
    for i, field in enumerate(reader.schema):
        if field.name == "epss":
            epss_col_idx = i
            break
    
    if epss_col_idx is None:
        print("❌ EPSS column not found!")
        return
    
    # Sample multiple batches to get distribution
    epss_values = []
    n_batches = min(5, reader.num_record_batches)  # Sample first 5 batches
    
    for b in range(n_batches):
        batch = reader.get_batch(b)
        epss_col = batch.columns[epss_col_idx]
        batch_values = [epss_col[i].as_py() for i in range(len(epss_col)) if epss_col[i].as_py() is not None]
        epss_values.extend(batch_values)
    
    epss_array = np.array(epss_values)
    
    print(f"📊 Sampled {len(epss_values)} EPSS values from {n_batches} batches")
    print(f"📈 EPSS Statistics:")
    print(f"   Min:    {epss_array.min():.6f}")
    print(f"   Max:    {epss_array.max():.6f}")
    print(f"   Mean:   {epss_array.mean():.6f}")
    print(f"   Std:    {epss_array.std():.6f}")
    print(f"   Median: {np.median(epss_array):.6f}")
    
    # Check if values look standardized (mean ~0, std ~1) or log-transformed
    is_standardized = abs(epss_array.mean()) < 0.1 and abs(epss_array.std() - 1.0) < 0.1
    is_log_space = epss_array.min() < -1 and epss_array.max() < 10  # Typical log(p + 1e-6) range
    
    print(f"\n🔍 ANALYSIS:")
    if is_standardized:
        print("❌ VALUES APPEAR STANDARDIZED (mean ≈ 0, std ≈ 1)")
        print("❌ This would be problematic for learning EPSS patterns")
    elif is_log_space:
        print("✅ VALUES APPEAR TO BE IN LOG SPACE (not standardized)")
        print("✅ This is correct - EPSS retains its log-transformed distribution")
    else:
        print("❓ VALUES HAVE UNEXPECTED DISTRIBUTION")
    
    # Convert sample back to probability space to verify
    print(f"\n📈 SAMPLE CONVERSION TO PROBABILITY SPACE:")
    sample_indices = np.random.choice(len(epss_array), min(10, len(epss_array)), replace=False)
    for i, idx in enumerate(sample_indices):
        log_val = epss_array[idx]
        prob_val = max(0, np.exp(log_val) - 1e-6)
        print(f"   Sample {i+1}: log={log_val:.6f} → prob={prob_val:.6f}")
        
        # Check if probability is reasonable (0-1 range)
        if not (0 <= prob_val <= 1):
            print(f"   ⚠️  Converted probability {prob_val:.6f} is outside [0,1] range!")
    
    # Check scaler.pkl to see what was actually standardized
    scaler_path = arrow_path.parent / "scaler.pkl"
    if scaler_path.exists():
        import joblib
        scaler_data = joblib.load(scaler_path)
        print(f"\n📋 SCALER.PKL CONTENTS:")
        print(f"   Standardized columns: {list(scaler_data['mean'].keys())}")
        
        if 'epss' in scaler_data['mean']:
            print("❌ EPSS found in scaler - IT WAS STANDARDIZED!")
            print(f"   EPSS mean: {scaler_data['mean']['epss']:.6f}")
            print(f"   EPSS std:  {scaler_data['std']['epss']:.6f}")
        else:
            print("✅ EPSS not found in scaler - IT WAS NOT STANDARDIZED!")
    
    return {
        'epss_stats': {
            'min': epss_array.min(),
            'max': epss_array.max(), 
            'mean': epss_array.mean(),
            'std': epss_array.std()
        },
        'is_standardized': is_standardized,
        'is_log_space': is_log_space
    }

if __name__ == "__main__":
    result = verify_epss_standardization()
    
    print(f"\n" + "=" * 60)
    print("🎯 CONCLUSION")
    print("=" * 60)
    
    if result and result['is_log_space'] and not result['is_standardized']:
        print("✅ EPSS is correctly preserved in log space (not standardized)")
        print("✅ Model will receive meaningful EPSS values as input")
        print("✅ No additional fixes needed for standardization")
    elif result and result['is_standardized']:
        print("❌ EPSS appears to be standardized - this is problematic")
        print("❌ Model would receive normalized values instead of log EPSS")
        print("❌ Need to fix preprocessing to exclude EPSS from standardization")
    else:
        print("❓ Unable to determine EPSS standardization status")
        print("❓ Manual inspection of preprocessing pipeline needed") 