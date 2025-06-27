#!/usr/bin/env python
"""
Setup script for Ray Tune hyperparameter optimization
Installs dependencies and validates environment
"""

import subprocess
import sys
import importlib
from pathlib import Path

def check_and_install_packages():
    """Check and install required packages for Ray Tune"""
    
    required_packages = [
        ("ray[default]==2.7.0", "ray"),
        ("bayesian-optimization", "bayes_opt"),
        ("xarray", "xarray"),
        ("tqdm", "tqdm"),
        ("pandas", "pandas"),
        ("torch", "torch"),
        ("numpy", "numpy")
    ]
    
    print("🔍 CHECKING RAY TUNE DEPENDENCIES")
    print("=" * 50)
    
    missing_packages = []
    
    for package_spec, import_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"✅ {import_name:20s}: Already installed")
        except ImportError:
            print(f"❌ {import_name:20s}: Missing")
            missing_packages.append(package_spec)
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"   Installing: {package}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ Successfully installed: {package}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {package}: {e}")
                return False
    else:
        print("\n✅ All dependencies already satisfied!")
    
    return True

def validate_ray_setup():
    """Validate Ray can initialize properly"""
    
    print("\n🚀 VALIDATING RAY SETUP")
    print("=" * 50)
    
    try:
        import ray
        from ray import tune, air
        from ray.tune.search.bayesopt import BayesOptSearch
        from ray.tune.schedulers import ASHAScheduler
        import ray.train.torch
        
        print("✅ Ray imports successful")
        
        # Test Ray initialization
        try:
            ray.init(
                num_gpus=0,  # CPU only for test
                num_cpus=2,
                object_store_memory=100_000_000,  # 100MB
                ignore_reinit_error=True,
                configure_logging=False
            )
            print("✅ Ray initialization successful")
            
            # Test basic functionality
            def dummy_trainable(config):
                import time
                time.sleep(0.1)
                from ray.air import session
                session.report({"test_metric": config["x"] ** 2})
            
            # Quick test run
            tuner = tune.Tuner(
                dummy_trainable,
                param_space={"x": tune.uniform(0, 1)},
                tune_config=tune.TuneConfig(num_samples=2)
            )
            
            results = tuner.fit()
            best = results.get_best_result(metric="test_metric", mode="min")
            
            print(f"✅ Ray Tune test successful (best x={best.config['x']:.3f})")
            
            ray.shutdown()
            
        except Exception as e:
            print(f"❌ Ray functionality test failed: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Ray import failed: {e}")
        return False
    
    return True

def validate_data_files():
    """Check if required data files exist"""
    
    print("\n📁 VALIDATING DATA FILES")
    print("=" * 50)
    
    work_dir = Path("work")
    arrow_path = work_dir / "epss_stage1.arrow"
    vocab_path = work_dir / "vocab.json"
    
    files_valid = True
    
    if arrow_path.exists():
        size_mb = arrow_path.stat().st_size / (1024 * 1024)
        print(f"✅ Arrow file found: {arrow_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Arrow file missing: {arrow_path}")
        files_valid = False
    
    if vocab_path.exists():
        print(f"✅ Vocab file found: {vocab_path}")
    else:
        print(f"❌ Vocab file missing: {vocab_path}")
        files_valid = False
    
    if not files_valid:
        print("\n⚠️  Data files missing. Please run:")
        print("   python -m ml_pipeline.data_prep.00_build_arrow")
    
    return files_valid

def validate_gpu():
    """Check GPU availability"""
    
    print("\n🎮 CHECKING GPU AVAILABILITY")
    print("=" * 50)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            print(f"✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   Primary GPU: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 6:  # 6GB minimum
                print("✅ GPU memory sufficient for Ray Tune trials")
                return True
            else:
                print("⚠️  GPU memory may be insufficient for concurrent trials")
                return False
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            return False
            
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False

def create_launch_script():
    """Create a convenient launch script"""
    
    print("\n📝 CREATING LAUNCH SCRIPT")
    print("=" * 50)
    
    launch_script = """#!/bin/bash
# Ray Tune Launch Script for EPSS LSTM Hyperparameter Optimization

echo "🚀 Starting Ray Tune Hyperparameter Optimization"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "ml_pipeline/lstm_ray_tune.py" ]; then
    echo "❌ Error: Please run from the project root directory"
    exit 1
fi

# Check data files
if [ ! -f "work/epss_stage1.arrow" ]; then
    echo "❌ Error: Arrow file not found. Please run data preparation first."
    echo "   python -m ml_pipeline.data_prep.00_build_arrow"
    exit 1
fi

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1

# Launch Ray Tune
echo "🎯 Launching hyperparameter search..."
python -m ml_pipeline.lstm_ray_tune

echo "✅ Ray Tune completed. Check ml_pipeline/results/ for best configuration."
"""

    script_path = Path("run_ray_tune.sh")
    with open(script_path, 'w') as f:
        f.write(launch_script)
    
    # Make executable on Unix systems
    try:
        import stat
        script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
        print(f"✅ Launch script created: {script_path}")
        print("   Usage: ./run_ray_tune.sh")
    except:
        print(f"✅ Launch script created: {script_path}")
        print("   Usage: bash run_ray_tune.sh")

def main():
    """Main setup and validation"""
    
    print("🔧 RAY TUNE SETUP & VALIDATION")
    print("=" * 60)
    
    success = True
    
    # 1. Install dependencies
    if not check_and_install_packages():
        success = False
    
    # 2. Validate Ray functionality
    if not validate_ray_setup():
        success = False
    
    # 3. Check data files
    if not validate_data_files():
        success = False
    
    # 4. Check GPU
    gpu_available = validate_gpu()
    
    # 5. Create launch script
    create_launch_script()
    
    # Final summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 RAY TUNE SETUP COMPLETE!")
        print("✅ All dependencies installed")
        print("✅ Ray functionality validated")
        print("✅ Data files present")
        if gpu_available:
            print("✅ GPU ready for accelerated training")
        else:
            print("⚠️  No GPU available (will use CPU)")
        
        print("\n🚀 READY TO LAUNCH:")
        print("   python -m ml_pipeline.lstm_ray_tune")
        print("   # OR")
        print("   ./run_ray_tune.sh")
        
        print("\n📊 After completion, retrain with best config:")
        print("   python -m ml_pipeline.lstm_final_retrain")
        
    else:
        print("❌ SETUP INCOMPLETE")
        print("   Please resolve the issues above before proceeding")
    
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 