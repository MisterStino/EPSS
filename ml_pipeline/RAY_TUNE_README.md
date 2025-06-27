# Ray Tune Hyperparameter Optimization for EPSS LSTM

This directory contains a complete Ray Tune implementation for hyperparameter optimization of the EPSS LSTM forecaster, following official Ray documentation best practices.

## 🚀 Quick Start

1. **Setup and validate environment:**
   ```bash
   python ml_pipeline/setup_ray_tune.py
   ```

2. **Launch hyperparameter search:**
   ```bash
   python -m ml_pipeline.lstm_ray_tune
   # OR use the generated launch script:
   ./run_ray_tune.sh
   ```

3. **Retrain with best configuration:**
   ```bash
   python -m ml_pipeline.lstm_final_retrain
   ```

## 📁 File Structure

```
ml_pipeline/
├── lstm_ray_tune.py          # Main Ray Tune implementation
├── lstm_final_retrain.py     # Final training with best config
├── setup_ray_tune.py         # Environment setup and validation
├── RAY_TUNE_README.md        # This documentation
└── results/                  # Generated results
    ├── best_config.json      # Optimal hyperparameters
    ├── final_model.pt        # Production-ready model
    └── final_results_summary.json
```

## 🎯 Implementation Details

### Key Features

✅ **Deterministic Training Function**: Follows Ray docs pattern with proper seeding and reproducibility  
✅ **Resource-Aware Scheduling**: Fractional GPU allocation for concurrent trials  
✅ **Memory Constraint Checking**: Rejects configs that would cause OOM  
✅ **Mixed Precision Training**: FP16 for better GPU utilization  
✅ **Early Stopping**: ASHA scheduler with aggressive pruning  
✅ **Bayesian Optimization**: Efficient search in continuous spaces  
✅ **Checkpoint Management**: Automatic saving and best model selection  

### Search Space

| Hyperparameter | Type | Range | Distribution |
|---------------|------|--------|--------------|
| `hidden_size` | int | 384-1024 | Quantized log-uniform (steps of 64) |
| `lstm_layers` | int | 2-4 | Categorical choice |
| `dropout` | float | 0.1-0.5 | Uniform |
| `lr` | float | 1e-4 to 3e-3 | Log-uniform |
| `weight_decay` | float | 1e-5 to 1e-2 | Log-uniform |
| `clip_grad` | float | [0.5, 1.0, 2.0] | Categorical choice |
| `batch_size` | int | [448, 640, 768, 896] | Categorical choice |
| `num_workers` | int | [4, 6, 8] | Categorical choice |
| `emb_dim` | int | [8, 16] | Categorical choice |

### Scheduler Configuration

- **Algorithm**: ASHAScheduler (Ray's recommended default)
- **Rungs**: 1 → 3 → 9 → 12 epochs
- **Reduction Factor**: 3 (keeps 1/3 of trials at each rung)
- **Grace Period**: 1 epoch (all trials run at least once)

### Resource Allocation

- **GPU**: 0.85 per trial (allows ~2 concurrent trials on 48GB GPU)
- **CPU**: 3 cores per trial
- **Memory**: Conservative OOM protection with constraint checking

## 🧠 Architecture Decisions

### Why This Approach?

1. **Function-Based Trainable**: More flexible than Lightning integration for custom streaming datasets
2. **Per-Batch Padding**: Maintains memory efficiency during hyperparameter search
3. **ASHA + BayesOpt**: Best of both worlds - aggressive early stopping + intelligent search
4. **Fractional GPU**: Maximizes hardware utilization without memory conflicts

### Ray AIR Integration

Following Ray 2.x best practices:
- Uses `session.report()` instead of deprecated `tune.report()`
- Proper checkpoint management with `session.get_checkpoint_dir()`
- AIR RunConfig for comprehensive experiment tracking

## 📊 Expected Results

### Search Budget
- **60 trials** (~1 GPU-day on modern hardware)
- **Early termination** of poor performers after 1-3 epochs
- **~5-10 trials** complete the full 12 epochs

### Performance Gains
Based on similar LSTM hyperparameter optimization studies:
- **5-15% validation loss improvement** over default parameters
- **Better generalization** through proper regularization tuning
- **Stable convergence** with optimized learning rates

## 🔧 Advanced Usage

### Custom Search Space

Modify the `search_space` dict in `lstm_ray_tune.py`:

```python
search_space = {
    "hidden_size": tune.qloguniform(256, 1536, 64),  # Wider range
    "lstm_layers": tune.choice([2, 3, 4, 5]),        # Deeper models
    # ... add your parameters
}
```

### Resource Tuning

For different GPU memory configurations:

```python
# For 24GB GPUs (e.g., RTX 3090)
trainable_with_resources = tune.with_resources(
    train_tune,
    {"gpu": 0.8, "cpu": 2}  # More conservative
)

# For 80GB GPUs (e.g., A100)
trainable_with_resources = tune.with_resources(
    train_tune,
    {"gpu": 0.4, "cpu": 4}  # Up to 2.5 concurrent trials
)
```

### Extending the Search

Add new hyperparameters in the `train_tune()` function:

```python
def train_tune(config):
    # ... existing code ...
    
    # New: Learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=config["lr_patience"],
        factor=config["lr_factor"]
    )
    
    # Use in training loop...
```

## 🐛 Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce `batch_size` choices in search space
- Lower `hidden_size` upper bound
- Decrease GPU allocation fraction

**Slow Startup:**
- Check `num_workers` - too many can cause contention
- Verify Arrow file is memory-mapped properly
- Reduce `prefetch_factor` if needed

**Non-Reproducible Results:**
```python
# Ensure in train_tune():
ray.train.torch.enable_reproducibility()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**DataLoader Hangs:**
- Set `num_workers=0` for debugging
- Check Windows multiprocessing compatibility
- Verify proper cleanup in dataset iterator

### Performance Monitoring

Monitor with `nvidia-smi`:
```bash
watch -n1 nvidia-smi
```

Check Ray dashboard (if enabled):
```
http://localhost:8265
```

## 📈 Production Deployment

### Loading the Best Model

```python
import torch
from pathlib import Path

# Load the optimized model
checkpoint = torch.load("ml_pipeline/results/final_model.pt")
model_state = checkpoint["model_state_dict"]
best_config = checkpoint["config"]

# Recreate model with optimal architecture
model = Seq2SeqLSTM(
    n_num=checkpoint["model_info"]["n_num"],
    n_bool=checkpoint["model_info"]["n_bool"],
    cat_sizes=checkpoint["model_info"]["cat_sizes"],
    hidden=best_config["hidden_size"],
    layers=best_config["lstm_layers"],
    # ... other params
)
model.load_state_dict(model_state)
```

### Performance Validation

The final model includes:
- **Test metrics** (MSE, MAE) from held-out evaluation
- **Training history** for convergence analysis
- **Complete configuration** for reproducibility

## 🔬 Research Extensions

### Multi-Objective Optimization
```python
# Add to search space for Pareto optimization
search_space["model_size_weight"] = tune.uniform(0.0, 0.1)

# Modify objective in train_tune():
session.report({
    "val_loss": val_loss,
    "model_size": total_params / 1e6,  # Millions of parameters
    "combined_metric": val_loss + config["model_size_weight"] * total_params / 1e6
})
```

### Population-Based Training
```python
from ray.tune.schedulers import PopulationBasedTraining

scheduler = PopulationBasedTraining(
    time_attr="epoch",
    perturbation_interval=4,
    hyperparam_mutations={
        "lr": tune.uniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.1, 0.5),
    }
)
```

## 📚 References

- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [PyTorch Hyperparameter Tuning Tutorial](https://docs.pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
- [ASHA Paper](https://arxiv.org/abs/1810.05934)
- [Bayesian Optimization](https://github.com/fmfn/BayesianOptimization)

---

**Note**: This implementation follows the exact patterns from the Ray documentation walk-through, ensuring compatibility with Ray 2.x and adherence to current best practices. 