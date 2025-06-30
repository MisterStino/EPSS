#!/usr/bin/env python
"""
Simple execution script for EPSS TCN hyperparameter optimization.
Provides easy configuration options for different use cases.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run EPSS TCN hyperparameter optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        choices=["quick", "standard", "extensive"], 
        default="standard",
        help="Optimization mode: quick (10 trials), standard (50 trials), extensive (100 trials)"
    )
    
    parser.add_argument(
        "--gpu-memory", 
        choices=["small", "medium", "large"], 
        default="large",
        help="GPU memory size: small (8-16GB), medium (24-32GB), large (48-90GB)"
    )
    
    parser.add_argument(
        "--concurrent-trials", 
        type=int, 
        default=2,
        help="Number of concurrent trials to run"
    )
    
    parser.add_argument(
        "--spike-percentile", 
        type=float, 
        default=95.0,
        help="Percentile for spike threshold definition (90-99)"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show configuration without running optimization"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (90.0 <= args.spike_percentile <= 99.0):
        print("Error: spike-percentile must be between 90 and 99")
        sys.exit(1)
    
    # Check prerequisites with robust path detection
    def find_project_root():
        current = Path.cwd()
        if (current / "ml_pipeline").exists():
            return current
        for parent in current.parents:
            if (parent / "ml_pipeline").exists():
                return parent
        for path in ["/notebooks/EPSS", "/notebooks", Path.home() / "EPSS"]:
            path = Path(path)
            if path.exists() and (path / "ml_pipeline").exists():
                return path
        raise FileNotFoundError("Could not find project root")
    
    try:
        project_root = find_project_root()
        print(f"Project root: {project_root}")
    except FileNotFoundError:
        print("Error: Could not find project root with ml_pipeline directory")
        sys.exit(1)
    
    # Check for required files
    arrow_candidates = [
        project_root / "ml_pipeline" / "work" / "epss_stage1.arrow",
        project_root / "ml_pipeline" / "data_prep" / "work" / "epss_stage1.arrow",
    ]
    vocab_candidates = [
        project_root / "ml_pipeline" / "work" / "vocab.json",
        project_root / "ml_pipeline" / "data_prep" / "work" / "vocab.json",
    ]
    
    arrow_found = any(p.exists() for p in arrow_candidates)
    vocab_found = any(p.exists() for p in vocab_candidates)
    
    if not arrow_found:
        print(f"Error: Arrow file not found in any of: {arrow_candidates}")
        print("Please run: python -m ml_pipeline.data_prep.00_build_arrow")
        sys.exit(1)
    
    if not vocab_found:
        print(f"Error: Vocabulary file not found in any of: {vocab_candidates}")
        print("Try running: python -m ml_pipeline.generate_missing_vocab")
        sys.exit(1)
    
    # Configure based on mode
    mode_configs = {
        "quick": {"num_samples": 10, "max_steps": 5000},
        "standard": {"num_samples": 50, "max_steps": 20000},
        "extensive": {"num_samples": 100, "max_steps": 40000}
    }
    
    # Configure based on GPU memory - TCN specific parameters
    gpu_configs = {
        "small": {
            "nb_filters": [64, 128],
            "levels": [4, 6],
            "kernel_sizes": [3, 5],
            "batch_sizes": [128, 256]
        },
        "medium": {
            "nb_filters": [128, 256],
            "levels": [6, 8],
            "kernel_sizes": [3, 5, 7],
            "batch_sizes": [256, 512]
        },
        "large": {
            "nb_filters": [128, 256, 512],
            "levels": [6, 8, 10],
            "kernel_sizes": [3, 5, 7],
            "batch_sizes": [256, 512, 768]
        }
    }
    
    config = mode_configs[args.mode]
    gpu_config = gpu_configs[args.gpu_memory]
    
    print("="*80)
    print("EPSS TCN HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"GPU Memory: {args.gpu_memory}")
    print(f"Number of trials: {config['num_samples']}")
    print(f"Max concurrent trials: {args.concurrent_trials}")
    print(f"Max steps per trial: {config['max_steps']}")
    print(f"Spike percentile: {args.spike_percentile}%")
    print(f"TCN filters: {gpu_config['nb_filters']}")
    print(f"TCN levels: {gpu_config['levels']}")
    print(f"Kernel sizes: {gpu_config['kernel_sizes']}")
    print(f"Batch sizes: {gpu_config['batch_sizes']}")
    print("="*80)
    
    if args.dry_run:
        print("Dry run mode - configuration shown above")
        print("Remove --dry-run to start optimization")
        return
    
    # Import and modify the main script configuration
    from ml_pipeline.tcn_tune_ray import main as run_optimization
    import ml_pipeline.tcn_tune_ray as hpo_module
    from ray import tune
    
    # Temporarily modify the configuration - TCN specific search space
    original_search_space = {
        "nb_filters": tune.choice(gpu_config['nb_filters']),
        "levels": tune.choice(gpu_config['levels']),
        "kernel_size": tune.choice(gpu_config['kernel_sizes']),
        "dropout": tune.uniform(0.05, 0.3),
        "lr": tune.qloguniform(1e-5, 1e-2, q=1e-6),
        "batch_size": tune.choice(gpu_config['batch_sizes']),
        "epochs": tune.choice([8, 12, 16, 20])
    }
    
    # Monkey patch the configuration
    def patched_main():
        # Suppress warnings
        import os
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
        os.environ["RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS"] = "0"
        
        import ray
        from ray.tune.search.optuna import OptunaSearch
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune import CLIReporter
        import optuna
        
        # Initialize Ray (disable dashboard only on Windows to avoid handle errors)
        import platform
        is_windows = platform.system().lower() == 'windows'
        
        if not ray.is_initialized():
            ray.init(
                include_dashboard=not is_windows,  # Enable dashboard on Linux/Mac
                ignore_reinit_error=True,
                log_to_driver=False,  # Reduce logging overhead
                num_cpus=None,  # Auto-detect
                num_gpus=None   # Auto-detect
            )
        
        # Setup Bayesian optimization with OptunaSearch
        search_algorithm = OptunaSearch(
            metric=["val_loss", "val_spike_recall"],
            mode=["min", "max"],
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f"epss_tcn_hpo_{args.mode}"
        )
        
        # Setup ASHA scheduler for early stopping
        scheduler = ASHAScheduler(
            time_attr="global_step",
            metric="val_loss",
            mode="min",
            max_t=config['max_steps'],
            grace_period=min(2000, config['max_steps'] // 10),
            reduction_factor=2
        )
        
        # Setup reporter - TCN specific parameter names
        reporter = CLIReporter(
            parameter_columns={
                "nb_filters": "filters",
                "levels": "levels",
                "kernel_size": "kernel",
                "dropout": "dropout",
                "lr": "lr",
                "batch_size": "batch",
                "epochs": "epochs"
            },
            metric_columns=[
                "val_loss", 
                "val_spike_recall", 
                "train_loss", 
                "train_spike_recall", 
                "epoch",
                "global_step"
            ],
            max_progress_rows=20,
            max_error_rows=5
        )
        
        # Create and run tuner
        tuner = tune.Tuner(
            hpo_module.train_tcn_with_tune,
            param_space=original_search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_algorithm,
                scheduler=scheduler,
                num_samples=config['num_samples'],
                max_concurrent_trials=args.concurrent_trials,
            ),
            run_config=tune.RunConfig(
                name=f"epss_tcn_{args.mode}_hpo",
                progress_reporter=reporter,
                stop={"global_step": config['max_steps']},
                failure_config=tune.FailureConfig(max_failures=3),
                storage_path=str(Path("./ray_results").absolute()),  # Use absolute path
                log_to_file=True
            )
        )
        
        print("Starting TCN hyperparameter optimization...")
        results = tuner.fit()
        
        # The rest follows the same pattern as the original main function
        return results
    
    # Also patch the spike percentile if different from default
    if args.spike_percentile != 95.0:
        original_define_spike_threshold = hpo_module.define_spike_threshold
        def patched_define_spike_threshold(epss_scores, percentile=args.spike_percentile):
            return original_define_spike_threshold(epss_scores, percentile)
        hpo_module.define_spike_threshold = patched_define_spike_threshold
    
    print("Starting TCN optimization...")
    
    # Platform-specific dashboard message
    import platform
    if platform.system().lower() == 'windows':
        print("Dashboard disabled on Windows to avoid handle errors")
        print("Monitor progress in the terminal output below")
    else:
        print("Monitor progress at: http://localhost:8265 (Ray Dashboard)")
        print("Dashboard will be available once optimization starts")
    
    print("Press Ctrl+C to stop early if needed")
    print()
    
    try:
        patched_main()
        print("\n" + "="*80)
        print("TCN OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Next steps:")
        print("1. Review results in ./ray_results/")
        print("2. Train final models: python -m ml_pipeline.use_best_tcn_config")
        print("3. Check best configurations: ml_pipeline/results/best_tcn_configs.json")
        
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("TCN OPTIMIZATION INTERRUPTED BY USER")
        print("="*80)
        print("Partial results may be available in ./ray_results/")
        
    except Exception as e:
        print(f"\n" + "="*80)
        print("TCN OPTIMIZATION FAILED")
        print("="*80)
        print(f"Error: {e}")
        print("Check logs in ./ray_results/ for details")
        sys.exit(1)

if __name__ == "__main__":
    main() 