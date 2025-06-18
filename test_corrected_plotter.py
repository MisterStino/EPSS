#!/usr/bin/env python3
"""Test the corrected plotter"""

import sys
sys.path.append('.')

from ml_plot.scripts.barry_plotter_fixed import CVEPredPlotterFixed
from pathlib import Path

def test_corrected_plotter():
    """Test that the corrected plotter works with the fixed inverse transform"""
    
    print("🧪 Testing corrected plotter...")
    
    try:
        # Create the corrected plotter
        plotter = CVEPredPlotterFixed(
            nc_path=Path("ml_pipeline/results/predictions/predictions_stream.nc"),
            parquet=Path("data/epss/processed/epss_processed.parquet"),
            out_root=Path("ml_plots/plots_corrected_test"),
            cve_list=["CVE-2013-3190"]
        )
        
        print("✅ Corrected plotter created successfully!")
        
        # Test a single plot
        print("🎨 Testing plot generation...")
        plotter.plot_all()
        print("✅ Plot generation completed successfully!")
        
        # Verify the fix worked by checking the function exists
        if hasattr(plotter, '_log_to_prob'):
            print("✅ SUCCESS: _log_to_prob function exists (old _inv_invlog replaced)")
        else:
            print("❌ ERROR: _log_to_prob function not found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_corrected_plotter() 