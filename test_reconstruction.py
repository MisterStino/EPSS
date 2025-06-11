#!/usr/bin/env python3

from catalogs_processed.cve_historical_reconstructor_v2 import NVDHelper
import logging

def test_reconstruction():
    print("Testing reconstruction phase with new error handling...")
    
    # Create NVDHelper instance
    nvd = NVDHelper(
        verbose=True,
        workers=12,
        api_key='0f3e80d0-fe35-4118-a2eb-df87148e355c',
    )
    
    try:
        # Test just the reconstruction phase
        nvd.reconstruct_all_with_history()
        print("✅ Reconstruction completed successfully!")
        
        # Check if manifests were created
        manifest_path = nvd.manifest_dir / "reconstruction_manifest.csv"
        if manifest_path.exists():
            print(f"✅ Reconstruction manifest created: {manifest_path}")
            
            # Show first few lines of manifest
            with open(manifest_path) as f:
                lines = f.readlines()[:10]
                print("\n📊 First few manifest entries:")
                for line in lines:
                    print(f"   {line.strip()}")
        else:
            print("❌ Reconstruction manifest not found")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_reconstruction() 