# ML Pipeline Results Directory

This directory contains training outputs and model artifacts:

## Structure
- `loss_history.csv` - Training and validation loss per epoch
- `checkpoint.pt` - Model weights and training configuration
- `predictions/` - Model predictions in NetCDF format

## Files Generated
- **Training History**: CSV file with epoch-wise loss metrics
- **Model Checkpoint**: PyTorch state dict with hyperparameters
- **Predictions**: xarray Dataset with test predictions, ground truth, and masks

## Note
This directory is automatically created by the training script if it doesn't exist. 