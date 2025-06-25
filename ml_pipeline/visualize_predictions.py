#!/usr/bin/env python3
"""
Visualization of LSTM Predictions
=================================

Creates comprehensive visualizations of the prediction analysis results.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def create_prediction_visualizations():
    """Create comprehensive visualizations of prediction results"""
    
    # Load data
    netcdf_path = "ml_pipeline/results/predictions/predictions_stream.nc"
    if not Path(netcdf_path).exists():
        print(f"❌ Predictions file not found: {netcdf_path}")
        return
    
    print("📊 Creating prediction visualizations...")
    ds = xr.open_dataset(netcdf_path)
    
    # Create output directory
    viz_dir = Path("ml_pipeline/results/visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Overall Prediction vs Truth Scatter Plot
    print("  → Creating prediction vs truth scatter plot...")
    create_pred_vs_truth_plot(ds, viz_dir)
    
    # 2. Horizon Performance Plot
    print("  → Creating horizon performance plot...")
    create_horizon_performance_plot(ds, viz_dir)
    
    # 3. Error Distribution Plot
    print("  → Creating error distribution plot...")
    create_error_distribution_plot(ds, viz_dir)
    
    # 4. CVE Performance Distribution
    print("  → Creating CVE performance distribution...")
    create_cve_performance_plot(ds, viz_dir)
    
    # 5. Sample CVE Time Series
    print("  → Creating sample CVE time series...")
    create_sample_timeseries_plot(ds, viz_dir)
    
    # 6. Model Bias Analysis
    print("  → Creating model bias analysis...")
    create_bias_analysis_plot(ds, viz_dir)
    
    print(f"✓ All visualizations saved to: {viz_dir}")


def create_pred_vs_truth_plot(ds, output_dir):
    """Create scatter plot of predictions vs ground truth"""
    
    # Get valid data
    pred = ds.pred.values
    true = ds.true.values
    mask_h = ds.mask_h.values
    mask_e = ds.eval_mask.values
    
    # Sample data for visualization (too much data to plot all)
    combined_mask = mask_h * mask_e[:, :, np.newaxis]
    valid_mask = combined_mask > 0
    
    valid_pred = pred[valid_mask]
    valid_true = true[valid_mask]
    
    # Sample for plotting
    n_sample = min(50000, len(valid_pred))
    idx = np.random.choice(len(valid_pred), n_sample, replace=False)
    sample_pred = valid_pred[idx]
    sample_true = valid_true[idx]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(sample_true, sample_pred, alpha=0.5, s=1)
    ax1.plot([sample_true.min(), sample_true.max()], 
             [sample_true.min(), sample_true.max()], 'r--', alpha=0.8)
    ax1.set_xlabel('True EPSS (log scale)')
    ax1.set_ylabel('Predicted EPSS (log scale)')
    ax1.set_title('Predictions vs Ground Truth')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation
    corr = np.corrcoef(sample_pred, sample_true)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='white'))
    
    # Hexbin plot for density
    hb = ax2.hexbin(sample_true, sample_pred, gridsize=50, cmap='Blues')
    ax2.plot([sample_true.min(), sample_true.max()], 
             [sample_true.min(), sample_true.max()], 'r--', alpha=0.8)
    ax2.set_xlabel('True EPSS (log scale)')
    ax2.set_ylabel('Predicted EPSS (log scale)')
    ax2.set_title('Prediction Density')
    plt.colorbar(hb, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_vs_truth.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_horizon_performance_plot(ds, output_dir):
    """Create plot showing performance across forecast horizons"""
    
    pred = ds.pred.values
    true = ds.true.values
    mask_h = ds.mask_h.values
    mask_e = ds.eval_mask.values
    
    horizon_metrics = []
    
    for h in range(len(ds.horizon)):
        h_mask = mask_h[:, :, h] * mask_e
        valid_idx = h_mask > 0
        
        if valid_idx.sum() == 0:
            continue
            
        h_pred = pred[:, :, h][valid_idx]
        h_true = true[:, :, h][valid_idx]
        
        h_mse = np.mean((h_pred - h_true) ** 2)
        h_mae = np.mean(np.abs(h_pred - h_true))
        h_corr = np.corrcoef(h_pred, h_true)[0, 1] if len(h_pred) > 1 else 0
        
        horizon_metrics.append({
            'horizon': h + 1,
            'mse': h_mse,
            'mae': h_mae,
            'correlation': h_corr
        })
    
    horizon_df = pd.DataFrame(horizon_metrics)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # MAE and MSE
    ax1.plot(horizon_df.horizon, horizon_df.mae, 'o-', label='MAE', linewidth=2)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(horizon_df.horizon, horizon_df.mse, 's-', color='red', label='MSE', linewidth=2)
    
    ax1.set_xlabel('Forecast Horizon (days)')
    ax1.set_ylabel('Mean Absolute Error', color='blue')
    ax1_twin.set_ylabel('Mean Squared Error', color='red')
    ax1.set_title('Prediction Error by Forecast Horizon')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Correlation
    ax2.plot(horizon_df.horizon, horizon_df.correlation, 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Forecast Horizon (days)')
    ax2.set_ylabel('Correlation')
    ax2.set_title('Prediction Correlation by Forecast Horizon')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "horizon_performance.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_error_distribution_plot(ds, output_dir):
    """Create plots showing error distribution"""
    
    pred = ds.pred.values
    true = ds.true.values
    mask_h = ds.mask_h.values
    mask_e = ds.eval_mask.values
    
    combined_mask = mask_h * mask_e[:, :, np.newaxis]
    valid_mask = combined_mask > 0
    
    errors = pred[valid_mask] - true[valid_mask]
    abs_errors = np.abs(errors)
    
    # Sample for plotting
    n_sample = min(100000, len(errors))
    idx = np.random.choice(len(errors), n_sample, replace=False)
    sample_errors = errors[idx]
    sample_abs_errors = abs_errors[idx]
    
    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Error histogram
    ax1.hist(sample_errors, bins=100, alpha=0.7, density=True)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    ax1.axvline(x=sample_errors.mean(), color='orange', linestyle='-', alpha=0.8, 
                label=f'Mean: {sample_errors.mean():.3f}')
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Density')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Absolute error histogram
    ax2.hist(sample_abs_errors, bins=100, alpha=0.7, density=True)
    ax2.axvline(x=sample_abs_errors.mean(), color='orange', linestyle='-', alpha=0.8,
                label=f'Mean: {sample_abs_errors.mean():.3f}')
    ax2.axvline(x=np.median(sample_abs_errors), color='green', linestyle='--', alpha=0.8,
                label=f'Median: {np.median(sample_abs_errors):.3f}')
    ax2.set_xlabel('Absolute Prediction Error')
    ax2.set_ylabel('Density')
    ax2.set_title('Absolute Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot for normality check
    from scipy import stats
    stats.probplot(sample_errors, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normal Distribution)')
    ax3.grid(True, alpha=0.3)
    
    # Error vs magnitude
    sample_true = true[valid_mask][idx]
    ax4.scatter(sample_true, sample_errors, alpha=0.5, s=1)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax4.set_xlabel('True EPSS Value')
    ax4.set_ylabel('Prediction Error')
    ax4.set_title('Error vs True Value')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_cve_performance_plot(ds, output_dir):
    """Create plots showing CVE-level performance distribution"""
    
    # Load CVE analysis if available
    cve_file = "ml_pipeline/results/cve_performance_analysis.csv"
    if not Path(cve_file).exists():
        print("CVE analysis file not found, skipping CVE performance plot")
        return
    
    cve_df = pd.read_csv(cve_file)
    
    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # MAE distribution
    ax1.hist(cve_df.mae, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=cve_df.mae.mean(), color='red', linestyle='--', 
                label=f'Mean: {cve_df.mae.mean():.3f}')
    ax1.axvline(x=cve_df.mae.median(), color='green', linestyle='--',
                label=f'Median: {cve_df.mae.median():.3f}')
    ax1.set_xlabel('Mean Absolute Error')
    ax1.set_ylabel('Number of CVEs')
    ax1.set_title('CVE MAE Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Correlation distribution
    valid_corr = cve_df.correlation.dropna()
    ax2.hist(valid_corr, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(x=valid_corr.mean(), color='red', linestyle='--',
                label=f'Mean: {valid_corr.mean():.3f}')
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Correlation')
    ax2.set_ylabel('Number of CVEs')
    ax2.set_title('CVE Correlation Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Timesteps vs performance
    ax3.scatter(cve_df.n_timesteps, cve_df.mae, alpha=0.6, s=10)
    ax3.set_xlabel('Number of Timesteps')
    ax3.set_ylabel('Mean Absolute Error')
    ax3.set_title('Performance vs Sequence Length')
    ax3.grid(True, alpha=0.3)
    
    # Performance scatter
    valid_cve_df = cve_df.dropna(subset=['correlation'])
    ax4.scatter(valid_cve_df.correlation, valid_cve_df.mae, alpha=0.6, s=10)
    ax4.set_xlabel('Correlation')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('MAE vs Correlation')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "cve_performance_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_sample_timeseries_plot(ds, output_dir):
    """Create sample time series plots for individual CVEs"""
    
    # Find CVEs with good data coverage
    mask_e = ds.eval_mask.values
    timesteps_per_cve = mask_e.sum(axis=1)
    
    # Get CVEs with at least 50 timesteps
    good_cves = np.where(timesteps_per_cve >= 50)[0]
    
    if len(good_cves) == 0:
        print("No CVEs with sufficient data for time series plot")
        return
    
    # Sample a few CVEs
    n_samples = min(6, len(good_cves))
    sample_indices = np.random.choice(good_cves, n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, cve_idx in enumerate(sample_indices):
        ax = axes[i]
        
        # Get data for this CVE
        cve_mask = mask_e[cve_idx] > 0
        if cve_mask.sum() == 0:
            continue
        
        # Get predictions and truth for first horizon (1-day ahead)
        cve_pred = ds.pred.values[cve_idx, cve_mask, 0]
        cve_true = ds.true.values[cve_idx, cve_mask, 0]
        
        # Create time axis
        time_idx = np.arange(len(cve_pred))
        
        # Plot
        ax.plot(time_idx, cve_true, label='True', linewidth=2, alpha=0.8)
        ax.plot(time_idx, cve_pred, label='Predicted', linewidth=2, alpha=0.8)
        
        cve_id = ds.cve.values[cve_idx]
        ax.set_title(f'CVE: {cve_id}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('EPSS (log scale)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and show metrics
        mae = np.mean(np.abs(cve_pred - cve_true))
        corr = np.corrcoef(cve_pred, cve_true)[0, 1]
        ax.text(0.02, 0.98, f'MAE: {mae:.3f}\nCorr: {corr:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_timeseries.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_bias_analysis_plot(ds, output_dir):
    """Create plots analyzing model bias patterns"""
    
    pred = ds.pred.values
    true = ds.true.values
    mask_h = ds.mask_h.values
    mask_e = ds.eval_mask.values
    
    # Get first horizon data
    pred_1d = pred[:, :, 0]
    true_1d = true[:, :, 0]
    mask_1d = mask_e
    
    # Calculate bias by true value bins
    valid_mask = mask_1d > 0
    valid_pred = pred_1d[valid_mask]
    valid_true = true_1d[valid_mask]
    
    # Sample for analysis
    n_sample = min(50000, len(valid_pred))
    idx = np.random.choice(len(valid_pred), n_sample, replace=False)
    sample_pred = valid_pred[idx]
    sample_true = valid_true[idx]
    
    # Create bins
    n_bins = 20
    true_bins = np.linspace(sample_true.min(), sample_true.max(), n_bins + 1)
    bin_centers = (true_bins[:-1] + true_bins[1:]) / 2
    
    # Calculate bias in each bin
    bin_bias = []
    bin_mae = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (sample_true >= true_bins[i]) & (sample_true < true_bins[i + 1])
        if mask.sum() > 10:  # Need sufficient samples
            bias = np.mean(sample_pred[mask] - sample_true[mask])
            mae = np.mean(np.abs(sample_pred[mask] - sample_true[mask]))
            count = mask.sum()
        else:
            bias = np.nan
            mae = np.nan
            count = 0
        
        bin_bias.append(bias)
        bin_mae.append(mae)
        bin_counts.append(count)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bias by true value
    valid_bins = ~np.isnan(bin_bias)
    ax1.plot(np.array(bin_centers)[valid_bins], np.array(bin_bias)[valid_bins], 
             'o-', linewidth=2, markersize=6)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    ax1.set_xlabel('True EPSS Value (binned)')
    ax1.set_ylabel('Mean Bias (Pred - True)')
    ax1.set_title('Model Bias by True Value Range')
    ax1.grid(True, alpha=0.3)
    
    # MAE by true value
    ax2.plot(np.array(bin_centers)[valid_bins], np.array(bin_mae)[valid_bins], 
             'o-', color='orange', linewidth=2, markersize=6)
    ax2.set_xlabel('True EPSS Value (binned)')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Model Error by True Value Range')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "bias_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    create_prediction_visualizations() 