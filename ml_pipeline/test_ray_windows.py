#!/usr/bin/env python
"""
Test Ray Tune Setup (Cross-Platform)
Simple script to verify Ray works before running full HPO.
Works on Windows, Linux, and macOS.
"""

import sys
import os
from pathlib import Path

def test_ray_basic():
    """Test basic Ray functionality."""
    print("Testing basic Ray functionality...")
    
    try:
        import ray
        print("✓ Ray imported successfully")
        
        # Test Ray initialization
        if ray.is_initialized():
            ray.shutdown()
        
        # Platform-aware dashboard setting
        import platform
        is_windows = platform.system().lower() == 'windows'
        
        ray.init(
            include_dashboard=not is_windows,  # Enable on Linux/Mac
            ignore_reinit_error=True,
            log_to_driver=False,
            num_cpus=2,  # Limit for testing
            num_gpus=None
        )
        print("✓ Ray initialized successfully")
        
        # Test simple task
        @ray.remote
        def simple_task(x):
            return x * 2
        
        future = simple_task.remote(21)
        result = ray.get(future)
        
        if result == 42:
            print("✓ Ray remote task works")
        else:
            print(f"✗ Ray remote task failed: expected 42, got {result}")
            return False
        
        ray.shutdown()
        print("✓ Ray shutdown successfully")
        return True
        
    except Exception as e:
        print(f"✗ Ray basic test failed: {e}")
        return False

def test_ray_tune():
    """Test Ray Tune functionality."""
    print("\nTesting Ray Tune functionality...")
    
    try:
        import ray
        from ray import tune
        from ray.tune.search.optuna import OptunaSearch
        import optuna
        
        print("✓ Ray Tune imports successful")
        
        # Initialize Ray
        if ray.is_initialized():
            ray.shutdown()
        
        # Platform-aware dashboard setting  
        import platform
        is_windows = platform.system().lower() == 'windows'
        
        ray.init(
            include_dashboard=not is_windows,  # Enable on Linux/Mac
            ignore_reinit_error=True,
            log_to_driver=False,
            num_cpus=2,
            num_gpus=None
        )
        
        # Simple trainable function
        def simple_trainable(config):
            from ray.air import session
            import time
            
            for i in range(3):
                # Simple computation
                result = config["x"] ** 2 + config["y"] ** 2
                session.report({"score": result, "iteration": i})
                time.sleep(0.1)
        
        # Test search algorithm
        search_algorithm = OptunaSearch(
            metric="score",
            mode="min",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run a simple tune
        tuner = tune.Tuner(
            simple_trainable,
            param_space={"x": tune.uniform(-5, 5), "y": tune.uniform(-5, 5)},
            tune_config=tune.TuneConfig(
                search_alg=search_algorithm,
                num_samples=3,
                max_concurrent_trials=1
            ),
                     run_config=tune.RunConfig(
             name="test_tune",
             storage_path=str(Path("./test_ray_results").absolute()),
             log_to_file=False
         )
        )
        
        results = tuner.fit()
        best_result = results.get_best_result("score", "min")
        
        if best_result:
            print("✓ Ray Tune optimization successful")
            print(f"  Best score: {best_result.metrics['score']:.4f}")
        else:
            print("✗ Ray Tune optimization failed")
            return False
        
        ray.shutdown()
        print("✓ Ray Tune test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Ray Tune test failed: {e}")
        return False

def test_data_files():
    """Test if required data files exist."""
    print("\nTesting data file availability...")
    
    arrow_path = Path("ml_pipeline/work/epss_stage1.arrow")
    vocab_path = Path("ml_pipeline/work/vocab.json")
    
    if arrow_path.exists():
        print(f"✓ Arrow file exists: {arrow_path}")
        print(f"  Size: {arrow_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"✗ Arrow file missing: {arrow_path}")
        return False
    
    if vocab_path.exists():
        print(f"✓ Vocabulary file exists: {vocab_path}")
        
        # Check vocab content
        try:
            import json
            with open(vocab_path, 'r') as f:
                vocab = json.load(f)
            print(f"  Categorical columns: {len(vocab)}")
            for col, values in vocab.items():
                print(f"    {col}: {len(values)} unique values")
        except Exception as e:
            print(f"  Warning: Could not read vocab file: {e}")
    else:
        print(f"✗ Vocabulary file missing: {vocab_path}")
        return False
    
    return True

def test_imports():
    """Test if all required imports work."""
    print("\nTesting imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("! CUDA not available (will use CPU)")
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed
        print("✓ Dataset imports successful")
    except Exception as e:
        print(f"✗ Dataset import failed: {e}")
        return False
    
    try:
        import numpy as np
        import json
        from pathlib import Path
        from functools import partial
        print("✓ Standard library imports successful")
    except Exception as e:
        print(f"✗ Standard library import failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    import platform
    system = platform.system()
    
    print("="*60)
    print(f"RAY TUNE COMPATIBILITY TEST ({system.upper()})")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data_files),
        ("Ray Basic", test_ray_basic),
        ("Ray Tune", test_ray_tune)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! You can run the hyperparameter optimization.")
        print("Next step: python -m ml_pipeline.lstm_tune_ray")
    else:
        print("\n❌ Some tests failed. Please fix the issues above before running HPO.")
        print("Common fixes:")
        print("  - Install missing packages: pip install ray[tune] optuna")
        print("  - Prepare data: python -m ml_pipeline.data_prep.00_build_arrow")
        print("  - Check CUDA installation if using GPU")
    
    # Cleanup
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except:
        pass
    
    # Clean up test results
    import shutil
    test_dir = Path("test_ray_results")
    if test_dir.exists():
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    main() 