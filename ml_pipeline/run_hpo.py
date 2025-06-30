#!/usr/bin/env python
"""
Simple execution script for EPSS LSTM hyperparameter optimization.
Provides easy configuration options for different use cases.
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run EPSS LSTM hyperparameter optimization",
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
    
    # Check prerequisites
    arrow_path = Path("ml_pipeline/work/epss_stage1.arrow")
    vocab_path = Path("ml_pipeline/work/vocab.json")
    
    if not arrow_path.exists():
        print(f"Error: Arrow file not found: {arrow_path}")
        print("Please run: python -m ml_pipeline.data_prep.00_build_arrow")
        sys.exit(1)
    
    if not vocab_path.exists():
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print("Please run: python -m ml_pipeline.data_prep.00_build_arrow")
        sys.exit(1)
    
    # Configure based on mode
    mode_configs = {
        "quick": {"num_samples": 10, "max_steps": 5000},
        "standard": {"num_samples": 50, "max_steps": 20000},
        "extensive": {"num_samples": 100, "max_steps": 40000}
    }
    
    # Configure based on GPU memory
    gpu_configs = {
        "small": {
            "hidden_sizes": [256, 512, 768],
            "batch_sizes": [128, 256, 512]
        },
        "medium": {
            "hidden_sizes": [512, 768, 1024],
            "batch_sizes": [256, 512, 768]
        },
        "large": {
            "hidden_sizes": [512, 768, 1024, 1536],
            "batch_sizes": [256, 512, 768, 1024]
        }
    }
    
    config = mode_configs[args.mode]
    gpu_config = gpu_configs[args.gpu_memory]
    
    print("="*80)
    print("EPSS LSTM HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"GPU Memory: {args.gpu_memory}")
    print(f"Number of trials: {config['num_samples']}")
    print(f"Max concurrent trials: {args.concurrent_trials}")
    print(f"Max steps per trial: {config['max_steps']}")
    print(f"Spike percentile: {args.spike_percentile}%")
    print(f"Hidden sizes: {gpu_config['hidden_sizes']}")
    print(f"Batch sizes: {gpu_config['batch_sizes']}")
    print("="*80)
    
    if args.dry_run:
        print("Dry run mode - configuration shown above")
        print("Remove --dry-run to start optimization")
        return
    
    # Import and modify the main script configuration
    from ml_pipeline.lstm_tune_ray import main as run_optimization
    import ml_pipeline.lstm_tune_ray as hpo_module
    from ray import tune
    
    # Temporarily modify the configuration
    original_search_space = {
        "hidden_size": tune.choice(gpu_config['hidden_sizes']),
        "layers": tune.choice([2, 3, 4]),
        "dropout": tune.uniform(0.05, 0.3),
        "lr": tune.qloguniform(1e-5, 1e-2, q=1e-6),
        "batch_size": tune.choice(gpu_config['batch_sizes']),
        "epochs": tune.choice([8, 12, 16, 20])
    }
    
    # Monkey patch the configuration
    def patched_main():
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
            study_name=f"epss_lstm_hpo_{args.mode}"
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
        
        # Setup reporter
        reporter = CLIReporter(
            parameter_columns={
                "hidden_size": "hidden",
                "layers": "layers",
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
            hpo_module.train_lstm_with_tune,
            param_space=original_search_space,
            tune_config=tune.TuneConfig(
                search_alg=search_algorithm,
                scheduler=scheduler,
                num_samples=config['num_samples'],
                max_concurrent_trials=args.concurrent_trials,
            ),
            run_config=ray.air.RunConfig(
                name=f"epss_lstm_{args.mode}_hpo",
                progress_reporter=reporter,
                stop={"global_step": config['max_steps']},
                failure_config=ray.air.FailureConfig(max_failures=3),
                storage_path="./ray_results",
                log_to_file=True
            )
        )
        
        print("Starting hyperparameter optimization...")
        results = tuner.fit()
        
        # The rest is identical to the original main function
        return hpo_module.main.__wrapped__(results) if hasattr(hpo_module.main, '__wrapped__') else None
    
    # Also patch the spike percentile if different from default
    if args.spike_percentile != 95.0:
        original_define_spike_threshold = hpo_module.define_spike_threshold
        def patched_define_spike_threshold(epss_scores, percentile=args.spike_percentile):
            return original_define_spike_threshold(epss_scores, percentile)
        hpo_module.define_spike_threshold = patched_define_spike_threshold
    
    print("Starting optimization...")
    
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
        run_optimization()
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("Next steps:")
        print("1. Review results in ./ray_results/")
        print("2. Train final models: python -m ml_pipeline.use_best_config")
        print("3. Check best configurations: ml_pipeline/results/best_configs.json")
        
    except KeyboardInterrupt:
        print("\n" + "="*80)
        print("OPTIMIZATION INTERRUPTED BY USER")
        print("="*80)
        print("Partial results may be available in ./ray_results/")
        
    except Exception as e:
        print(f"\n" + "="*80)
        print("OPTIMIZATION FAILED")
        print("="*80)
        print(f"Error: {e}")
        print("Check logs in ./ray_results/ for details")
        sys.exit(1)

if __name__ == "__main__":
    main() 