#!/usr/bin/env python3
"""
Test the fixed CVEIterableDataset to verify EPSS is included as input feature.
"""

import sys
from pathlib import Path

# Add the ml_pipeline directory to path
sys.path.insert(0, str(Path("ml_pipeline").resolve()))

from training.dataset_iterable import CVEIterableDataset

def test_fixed_features():
    """Test that EPSS is now included in numeric features"""
    
    arrow_path = Path("work/epss_stage1.arrow")
    if not arrow_path.exists():
        print(f"❌ Arrow file not found: {arrow_path}")
        return
    
    print("🧪 TESTING FIXED FEATURE CATEGORIZATION")
    print("=" * 60)
    
    # Create dataset with fixed logic
    try:
        dataset = CVEIterableDataset(arrow_path, horizon=30)
        
        print(f"✅ Dataset created successfully")
        print(f"📊 Numeric features ({len(dataset.num_cols)}): {dataset.num_cols}")
        print(f"🔘 Boolean features ({len(dataset.bool_cols)}): {dataset.bool_cols[:5]}{'...' if len(dataset.bool_cols) > 5 else ''}")
        print(f"🏷️  Categorical features ({len(dataset.cat_cols)}): {dataset.cat_cols}")
        
        # Check if EPSS is in numeric features
        if 'epss' in dataset.num_cols:
            print("\n🎉 SUCCESS: 'epss' is now included in numeric features!")
            epss_idx = dataset.num_cols.index('epss')
            print(f"📍 EPSS position in numeric features: index {epss_idx}")
        else:
            print("\n❌ FAILED: 'epss' still not in numeric features")
            print(f"🔍 Current numeric features: {dataset.num_cols}")
            
        # Test data loading
        print(f"\n🔄 Testing data loading...")
        sample = next(iter(dataset))
        num_tensor, bool_tensor, cat_tensor, epss_tensor, flag_tensor, date_tensor = sample
        
        print(f"✅ Sample loaded successfully")
        print(f"📊 Numeric tensor shape: {num_tensor.shape}")
        print(f"🔘 Boolean tensor shape: {bool_tensor.shape}")  
        print(f"🏷️  Categorical tensor shape: {cat_tensor.shape}")
        print(f"🎯 EPSS target shape: {epss_tensor.shape}")
        
        # Show first few values of numeric features (should include EPSS)
        if num_tensor.shape[0] > 0:
            first_row = num_tensor[0]
            print(f"📈 First row numeric features: {first_row[:10].tolist()}...")
            if 'epss' in dataset.num_cols:
                epss_idx = dataset.num_cols.index('epss')
                epss_input_val = first_row[epss_idx].item()
                epss_target_val = epss_tensor[0].item()
                print(f"🎯 EPSS as input feature: {epss_input_val:.6f}")
                print(f"🎯 EPSS as target: {epss_target_val:.6f}")
                print(f"✅ Input and target should be identical: {abs(epss_input_val - epss_target_val) < 1e-6}")
        
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_features() 