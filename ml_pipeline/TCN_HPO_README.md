# Comprehensive TCN Hyperparameter Optimization Pipeline

This document explains how to use the advanced Ray Tune-enabled TCN training pipeline for hyperparameter optimization with spike-aware metrics, Bayesian optimization, and early reporting.

## Overview

The TCN (Temporal Convolutional Network) HPO pipeline provides state-of-the-art hyperparameter optimization for EPSS forecasting using:

- **Multi-objective optimization**: Optimizes both validation loss and spike recall simultaneously
- **Bayesian optimization**: Uses OptunaSearch with TPE sampler for intelligent hyperparameter exploration
- **Early metric reporting**: Reports metrics every 100 batches for fine-grained early stopping
- **Spike-aware metrics**: Custom spike recall metric based on 95th percentile EPSS threshold
- **TCN-specific search spaces**: Optimized for dilated convolutional architectures
- **ASHA scheduling**: Aggressive early stopping for efficient resource utilization

## TCN vs LSTM Pipeline Differences

### Architecture Parameters

| Parameter | LSTM Pipeline | TCN Pipeline | Description |
|-----------|---------------|--------------|-------------|
| Model Capacity | `hidden_size` (256-1536) | `nb_filters` (64-512) | Number of hidden units vs convolutional filters |
| Model Depth | `layers` (2-4) | `levels` (4-10) | LSTM layers vs TCN dilated levels |
| Regularization | `dropout` (0.05-0.3) | `dropout` (0.05-0.3) | Same dropout range |
| Kernel Size | N/A | `kernel_size` (3,5,7) | TCN-specific: convolution kernel size |

### Memory Characteristics

- **TCN**: Generally more memory-efficient due to dilated convolutions
- **LSTM**: Higher memory usage due to recurrent state maintenance
- **Batch Size**: TCN can typically handle larger batch sizes

### Computational Patterns

- **TCN**: Parallel computation, better GPU utilization
- **LSTM**: Sequential computation, harder to parallelize
- **Training Speed**: TCN typically faster per epoch

## Prerequisites

1. **Install dependencies**:
   ```bash
   .\epss-env\Scripts\Activate.ps1
   pip install "ray[tune]==2.42.0" "optuna>=3.0.0" "pytorch-tcn>=0.0.1"
   ```

2. **Ensure data is prepared**:
   - `ml_pipeline/work/epss_stage1.arrow` must exist
   - `ml_pipeline/work/vocab.json` must exist
   - Run `python -m ml_pipeline.data_prep.00_build_arrow` if needed

## Quick Start

### 1. Basic TCN Hyperparameter Optimization

```bash
# Standard mode: 50 trials with balanced search space
python -m ml_pipeline.run_tcn_hpo

# Quick test: 10 trials for faster experimentation
python -m ml_pipeline.run_tcn_hpo --mode quick

# Extensive search: 100 trials for thorough optimization
python -m ml_pipeline.run_tcn_hpo --mode extensive
```

### 2. GPU Memory-Specific Optimization

```bash
# Small GPU (8-16GB): Conservative parameters
python -m ml_pipeline.run_tcn_hpo --gpu-memory small

# Medium GPU (24-32GB): Balanced parameters
python -m ml_pipeline.run_tcn_hpo --gpu-memory medium

# Large GPU (48-90GB): Aggressive parameters
python -m ml_pipeline.run_tcn_hpo --gpu-memory large
```

### 3. Advanced Configuration

```bash
# Custom concurrent trials and spike threshold
python -m ml_pipeline.run_tcn_hpo \
    --mode standard \
    --gpu-memory large \
    --concurrent-trials 4 \
    --spike-percentile 99.0

# Dry run to see configuration without training
python -m ml_pipeline.run_tcn_hpo --dry-run
```

## Search Space Configuration

### GPU Memory Configurations

#### Small GPU (8-16GB)
```python
{
    "nb_filters": [64, 128],
    "levels": [4, 6],
    "kernel_sizes": [3, 5],
    "batch_sizes": [128, 256]
}
```

#### Medium GPU (24-32GB)
```python
{
    "nb_filters": [128, 256],
    "levels": [6, 8],
    "kernel_sizes": [3, 5, 7],
    "batch_sizes": [256, 512]
}
```

#### Large GPU (48-90GB)
```python
{
    "nb_filters": [128, 256, 512],
    "levels": [6, 8, 10],
    "kernel_sizes": [3, 5, 7],
    "batch_sizes": [256, 512, 768]
}
```

### Complete Search Space

```python
search_space = {
    # TCN Architecture
    "nb_filters": tune.choice([64, 128, 256, 512]),  # Filters per TCN level
    "levels": tune.choice([4, 6, 8, 10]),           # Number of dilated levels
    "kernel_size": tune.choice([3, 5, 7]),          # Convolution kernel size
    "dropout": tune.uniform(0.05, 0.3),             # Dropout rate
    
    # Training Hyperparameters
    "lr": tune.qloguniform(1e-5, 5e-3, q=1e-6),    # Learning rate
    "batch_size": tune.choice([128, 256, 512, 768]), # Batch size
    
    # SUS Parameters (if enabled)
    "sus_beta": tune.uniform(0.3, 0.7),             # SUS significance
    "sus_look_ahead": tune.choice([5, 10, 15, 20])  # SUS window
}
```

## Using Best Configurations

### 1. Train with Best Loss Configuration

```bash
# Train with configuration that achieved lowest validation loss
python -m ml_pipeline.use_best_tcn_config --config best_loss --epochs 50
```

### 2. Train with Best Recall Configuration

```bash
# Train with configuration that achieved highest spike recall
python -m ml_pipeline.use_best_tcn_config --config best_recall --epochs 50
```

### 3. Advanced Training Options

```bash
# Train without saving model (for testing)
python -m ml_pipeline.use_best_tcn_config --config best_loss --epochs 20 --no-save

# Extended training for final model
python -m ml_pipeline.use_best_tcn_config --config best_recall --epochs 100
```

## Output Files and Results

### HPO Results
- `./ray_results/`: Ray Tune experiment logs and checkpoints
- `ml_pipeline/results/best_tcn_configs.json`: Best configurations found

### Final Model Training
- `ml_pipeline/results/best_tcn_{config_name}_final.pt`: Model checkpoint
- `ml_pipeline/results/tcn_{config_name}_training_history.csv`: Training metrics
- Model checkpoint includes:
  - Best hyperparameters
  - Model state dict
  - Training history
  - Final test metrics

## Monitoring and Analysis

### 1. Ray Dashboard (Linux/Mac)
```bash
# Monitor progress in real-time
# Dashboard available at: http://localhost:8265
```

### 2. Console Output
```bash
# Real-time progress reporting every 100 batches
# Epoch-level validation metrics
# Final results summary
```

### 3. Log Files
```bash
# Detailed logs in ./ray_results/{experiment_name}/
# Individual trial logs and metrics
# Error logs for failed trials
```

## Performance Optimization Tips

### 1. TCN-Specific Optimizations

- **Parallel Processing**: TCN benefits more from larger batch sizes than LSTM
- **Memory Efficiency**: Increase `levels` before `nb_filters` for better memory usage
- **Kernel Size**: Larger kernels capture longer dependencies but require more memory

### 2. GPU Memory Management

```python
# Monitor GPU memory usage
import torch
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### 3. Batch Size Tuning

- Start with smaller batch sizes and increase until OOM
- TCN typically handles larger batches than LSTM
- Use mixed precision training for memory savings

## Troubleshooting

### Common Issues

1. **OOM Errors**: Reduce batch size or model capacity
2. **Slow Training**: Increase batch size, check GPU utilization
3. **Poor Convergence**: Try different learning rates, check data quality
4. **Ray Worker Crashes**: Reduce concurrent trials, check system resources

### Debug Mode

```bash
# Run single trial for debugging
python -m ml_pipeline.run_tcn_hpo --mode quick --concurrent-trials 1

# Check data pipeline
python -c "
from ml_pipeline.tcn_tune_ray import load_vocab_and_paths
vocab, arrow_path = load_vocab_and_paths()
print(f'Vocab size: {len(vocab)}')
print(f'Arrow file exists: {arrow_path.exists()}')
"
```

## Comparison with LSTM Pipeline

### When to Use TCN vs LSTM

**Use TCN when:**
- You have large datasets (better parallelization)
- GPU memory is limited (more efficient)
- You need faster training
- Long sequence modeling is important

**Use LSTM when:**
- You need explicit sequential modeling
- Working with smaller datasets
- Interpretability is crucial
- You have proven LSTM baselines

### Performance Comparison

| Metric | TCN | LSTM | Winner |
|--------|-----|------|--------|
| Training Speed | ⚡⚡⚡ | ⚡⚡ | TCN |
| Memory Efficiency | ⚡⚡⚡ | ⚡⚡ | TCN |
| Long Dependencies | ⚡⚡⚡ | ⚡⚡ | TCN |
| Sequential Modeling | ⚡⚡ | ⚡⚡⚡ | LSTM |
| Interpretability | ⚡⚡ | ⚡⚡⚡ | LSTM |

## Next Steps

1. **Run HPO**: Start with standard mode to find best configurations
2. **Analyze Results**: Compare loss vs recall optimized models
3. **Final Training**: Train final models with best configurations
4. **Evaluation**: Compare TCN vs LSTM performance on your specific task
5. **Production**: Deploy best performing model

## Support and Contributing

For issues or questions:
1. Check Ray Tune logs in `./ray_results/`
2. Verify data pipeline integrity
3. Monitor system resources during training
4. Compare with LSTM pipeline behavior

The TCN pipeline follows the same architectural patterns as the LSTM pipeline for consistency and maintainability. 