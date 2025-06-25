#!/usr/bin/env python3
"""
Simple EPSS Confusion Matrix Analysis
====================================

Analyzes the EPSS forecasting model as a binary classification problem.
Generates confusion matrices and key metrics for overall and per-horizon performance.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def _log_to_prob(arr):
    """Convert from inverted log space to probability space"""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)

def main():
    """Main analysis function"""
    print("🎯 EPSS Confusion Matrix Analysis")
    print("="*50)
    
    # Create output directory
    output_dir = Path("confusion_matrix")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n🔬 Loading predictions data...")
    ds = xr.open_dataset("ml_plot/perf/files/predictions_stream.nc")
    
    # Convert to probability space
    pred_prob = _log_to_prob(ds.pred.values)
    true_prob = _log_to_prob(ds.true.values)
    
    # Extract masks
    eval_mask = ds.eval_mask.values.astype(bool)
    mask_h = ds.mask_h.values.astype(bool)
    test_mask = eval_mask[:, :, np.newaxis] & mask_h
    
    n_cves, n_time, n_horizon = pred_prob.shape
    print(f"📊 Dataset: {n_cves:,} CVEs × {n_time} time × {n_horizon} horizons")
    print(f"📈 Total test predictions: {test_mask.sum():,}")
    
    # Overall analysis
    print("\n🔍 OVERALL CLASSIFICATION ANALYSIS")
    print("="*50)
    
    # Get all test data
    pred_test = pred_prob[test_mask]
    true_test = true_prob[test_mask]
    
    # Remove invalid values
    valid_idx = np.isfinite(pred_test) & np.isfinite(true_test)
    pred_clean = pred_test[valid_idx]
    true_clean = true_test[valid_idx]
    
    # Binarize at 0.7 threshold
    threshold = 0.7
    y_pred = (pred_clean >= threshold).astype(int)
    y_true = (true_clean >= threshold).astype(int)
    
    print(f"📊 Valid test samples: {len(y_true):,}")
    print(f"   • Positive cases (≥{threshold}): {y_true.sum():,} ({y_true.mean():.1%})")
    print(f"   • Negative cases (<{threshold}): {(1-y_true).sum():,} ({(1-y_true.mean()):.1%})")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Print results
    print(f"\n📊 CONFUSION MATRIX:")
    print(f"                    PREDICTED")
    print(f"                 Low     High")
    print(f"   ACTUAL  Low  [{tn:>6}] [{fp:>6}]")
    print(f"          High  [{fn:>6}] [{tp:>6}]")
    
    print(f"\n📈 KEY METRICS:")
    print(f"   • True Positives (TP): {tp:,}")
    print(f"   • True Negatives (TN): {tn:,}")
    print(f"   • False Positives (FP): {fp:,}")
    print(f"   • False Negatives (FN): {fn:,}")
    print(f"   • Accuracy: {accuracy:.3f}")
    print(f"   • Precision: {precision:.3f}")
    print(f"   • Recall: {recall:.3f}")
    print(f"   • F1-Score: {f1:.3f}")
    print(f"   • Specificity: {specificity:.3f}")
    
    # Save overall results
    overall_results = pd.DataFrame({
        'Metric': ['TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1', 'Specificity'],
        'Value': [tp, tn, fp, fn, accuracy, precision, recall, f1, specificity]
    })
    overall_results.to_csv(output_dir / "overall_metrics.csv", index=False)
    
    # Per-horizon analysis
    print("\n🔍 PER-HORIZON ANALYSIS")
    print("="*50)
    
    horizon_results = []
    
    for h in range(n_horizon):
        print(f"  Horizon {h+1:2d}/30...", end=" ")
        
        # Get data for this horizon
        h_mask = test_mask[:, :, h]
        if not h_mask.any():
            continue
            
        pred_h = pred_prob[:, :, h][h_mask]
        true_h = true_prob[:, :, h][h_mask]
        
        # Remove invalid values
        valid_idx = np.isfinite(pred_h) & np.isfinite(true_h)
        if valid_idx.sum() < 100:
            continue
            
        pred_clean_h = pred_h[valid_idx]
        true_clean_h = true_h[valid_idx]
        
        # Binarize
        y_pred_h = (pred_clean_h >= threshold).astype(int)
        y_true_h = (true_clean_h >= threshold).astype(int)
        
        if y_true_h.sum() == 0:
            print("No positive cases")
            continue
        
        # Calculate metrics
        cm_h = confusion_matrix(y_true_h, y_pred_h)
        tn_h, fp_h, fn_h, tp_h = cm_h.ravel()
        
        acc_h = accuracy_score(y_true_h, y_pred_h)
        prec_h = precision_score(y_true_h, y_pred_h, zero_division=0)
        rec_h = recall_score(y_true_h, y_pred_h, zero_division=0)
        f1_h = f1_score(y_true_h, y_pred_h, zero_division=0)
        spec_h = tn_h / (tn_h + fp_h) if (tn_h + fp_h) > 0 else 0
        
        horizon_results.append({
            'horizon': h + 1,
            'n_samples': len(y_true_h),
            'n_positive': y_true_h.sum(),
            'tp': tp_h,
            'tn': tn_h,
            'fp': fp_h,
            'fn': fn_h,
            'accuracy': acc_h,
            'precision': prec_h,
            'recall': rec_h,
            'f1': f1_h,
            'specificity': spec_h
        })
        
        print(f"F1={f1_h:.3f}")
    
    # Save per-horizon results
    horizon_df = pd.DataFrame(horizon_results)
    horizon_df.to_csv(output_dir / "per_horizon_metrics.csv", index=False)
    
    print(f"\n📊 HORIZON SUMMARY:")
    print(f"   • Best F1: {horizon_df['f1'].max():.3f}")
    print(f"   • Worst F1: {horizon_df['f1'].min():.3f}")
    print(f"   • Average F1: {horizon_df['f1'].mean():.3f}")
    
    # Create plots
    print("\n🎨 Creating visualizations...")
    
    # 1. Overall confusion matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low (<0.7)', 'High (≥0.7)'],
                yticklabels=['Low (<0.7)', 'High (≥0.7)'],
                ax=ax1)
    ax1.set_title('Confusion Matrix - Raw Counts')
    ax1.set_xlabel('Predicted EPSS')
    ax1.set_ylabel('Actual EPSS')
    
    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Low (<0.7)', 'High (≥0.7)'],
                yticklabels=['Low (<0.7)', 'High (≥0.7)'],
                ax=ax2)
    ax2.set_title('Confusion Matrix - Normalized')
    ax2.set_xlabel('Predicted EPSS')
    ax2.set_ylabel('Actual EPSS')
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Horizon trends
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Metrics
    ax1.plot(horizon_df.horizon, horizon_df.f1, 'o-', label='F1-Score', color='blue')
    ax1.plot(horizon_df.horizon, horizon_df.precision, 's-', label='Precision', color='red')
    ax1.plot(horizon_df.horizon, horizon_df.recall, '^-', label='Recall', color='green')
    ax1.set_xlabel('Horizon (days)')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Classification Metrics vs Horizon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, 30)
    
    # Accuracy
    ax2.plot(horizon_df.horizon, horizon_df.accuracy, 'o-', color='purple')
    ax2.set_xlabel('Horizon (days)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Horizon')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 30)
    
    # Confusion components (log scale)
    ax3.plot(horizon_df.horizon, horizon_df.tp, 'o-', label='TP', color='green')
    ax3.plot(horizon_df.horizon, horizon_df.tn, 's-', label='TN', color='blue')
    ax3.plot(horizon_df.horizon, horizon_df.fp, '^-', label='FP', color='red')
    ax3.plot(horizon_df.horizon, horizon_df.fn, 'd-', label='FN', color='orange')
    ax3.set_xlabel('Horizon (days)')
    ax3.set_ylabel('Count (log scale)')
    ax3.set_title('Confusion Matrix Components')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(1, 30)
    ax3.set_yscale('log')
    
    # Sample sizes
    ax4.plot(horizon_df.horizon, horizon_df.n_samples, 'o-', color='gray')
    ax4.set_xlabel('Horizon (days)')
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Sample Size vs Horizon')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(1, 30)
    
    plt.tight_layout()
    plt.savefig(output_dir / "horizon_trends.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print(f"📊 Files generated:")
    print(f"   - overall_metrics.csv")
    print(f"   - per_horizon_metrics.csv")
    print(f"   - confusion_matrix.png")
    print(f"   - horizon_trends.png")

if __name__ == "__main__":
    main() 