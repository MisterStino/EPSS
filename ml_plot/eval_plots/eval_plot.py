import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for faster saving
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def _log_to_prob(arr):
    """Convert log-space predictions to probability space."""
    return np.clip(np.exp(arr) - 1e-6, 0.0, 1.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + eps
    return 100 * np.mean(2.0 * numerator / denominator)


def compute_metrics_per_cve(
    ds: xr.Dataset,
    true_var: str,
    pred_var: str,
    horizons: list[int],
    dim_cve: str = 'cve',
    dim_horizon: str = 'horizon'
) -> pd.DataFrame:
    records = []
    y_true_all = ds[true_var].values
    y_pred_all = ds[pred_var].values
    mask_h_all = ds['mask_h'].values  # horizon mask
    eval_mask_all = ds['eval_mask'].values  # time mask
    cve_ids = ds[dim_cve].values

    for cve_idx, cve_id in enumerate(tqdm(cve_ids, desc="Computing metrics")):
        for h in horizons:
            idx = h - 1
            y_true_log = y_true_all[cve_idx, :, idx]
            y_pred_log = y_pred_all[cve_idx, :, idx]
            
            # Apply masks: only where both horizon and eval masks are 1
            mask_h = mask_h_all[cve_idx, :, idx] == 1
            mask_eval = eval_mask_all[cve_idx, :] == 1
            valid = mask_h & mask_eval & ~np.isnan(y_true_log) & ~np.isnan(y_pred_log)

            if not np.any(valid):
                metrics = {m: np.nan for m in ['MAE', 'RMSE', 'SMAPE (%)', 'R2']}
            else:
                # Convert from log-space to probability space
                y_t = _log_to_prob(y_true_log[valid])
                y_p = _log_to_prob(y_pred_log[valid])
                
                metrics = {
                    'MAE': mean_absolute_error(y_t, y_p),
                    'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
                    'SMAPE (%)': smape(y_t, y_p),
                    'R2': r2_score(y_t, y_p),
                }

            record = {'CVE': str(cve_id), 'Horizon': h}
            record.update(metrics)
            records.append(record)

    return pd.DataFrame(records)


def plot_single_cve(
    df_cve: pd.DataFrame,
    metrics: list[str],
    horizons_to_plot: list[int],
    output_dir: Path
):
    cve = df_cve['CVE'].iloc[0]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4))
    fig.suptitle(f'Metrics for CVE {cve}', fontsize=16)

    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        axes[i].plot(df_cve['Horizon'], df_cve[metric], marker='o')
        axes[i].set_title(metric)
        axes[i].set_xlabel('Horizon')
        axes[i].set_xticks(horizons_to_plot)
        axes[i].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / f'metrics_cve_{cve}.png')
    plt.close(fig)


def plot_metrics(
    df: pd.DataFrame,
    horizons_to_plot=None,
    output_dir: Path | None = None,
    filter_cves: list[str] | None = None
):
    if horizons_to_plot is None:
        horizons_to_plot = list(range(1, 31))

    metrics = ['MAE', 'RMSE', 'SMAPE (%)', 'R2']
    cves = df['CVE'].unique().tolist()

    if filter_cves is not None:
        cves = [c for c in cves if c in filter_cves]

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=8) as executor:
        for cve in cves:
            df_cve = df[(df['CVE'] == cve) & (df['Horizon'].isin(horizons_to_plot))]
            df_cve = df_cve.sort_values('Horizon')
            executor.submit(plot_single_cve, df_cve, metrics, horizons_to_plot, output_dir)


def plot_preds_vs_true(
    ds: xr.Dataset,
    true_var: str,
    pred_var: str,
    output_dir: Path,
    filter_cves: list[str] | None = None,
    dim_cve: str = 'cve',
    dim_horizon: str = 'horizon'
):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cve_ids = [str(c) for c in ds[dim_cve].values]
    cve_ids = all_cve_ids if filter_cves is None else [c for c in all_cve_ids if c in filter_cves]

    horizons = ds[dim_horizon].values
    y_true_all = ds[true_var].values
    y_pred_all = ds[pred_var].values
    mask_h_all = ds['mask_h'].values
    eval_mask_all = ds['eval_mask'].values

    for cve_id in tqdm(cve_ids, desc="Plotting pred vs true"):
        cve_idx = all_cve_ids.index(cve_id)
        fig, ax = plt.subplots(figsize=(8, 5))

        # Compute masked averages for each horizon
        true_means = []
        pred_means = []
        
        for h_idx in range(len(horizons)):
            # Get data for this horizon
            y_true_h = y_true_all[cve_idx, :, h_idx]
            y_pred_h = y_pred_all[cve_idx, :, h_idx]
            
            # Apply masks
            mask_h = mask_h_all[cve_idx, :, h_idx] == 1
            mask_eval = eval_mask_all[cve_idx, :] == 1
            valid = mask_h & mask_eval & ~np.isnan(y_true_h) & ~np.isnan(y_pred_h)
            
            if np.any(valid):
                # Convert to probability space and compute mean
                true_prob = _log_to_prob(y_true_h[valid])
                pred_prob = _log_to_prob(y_pred_h[valid])
                true_means.append(np.mean(true_prob))
                pred_means.append(np.mean(pred_prob))
            else:
                true_means.append(np.nan)
                pred_means.append(np.nan)

        # Convert to arrays for plotting
        true_means = np.array(true_means)
        pred_means = np.array(pred_means)

        ax.plot(horizons, true_means, marker='o', label='True (probability)')
        ax.plot(horizons, pred_means, marker='x', label='Predicted (probability)')

        ax.set_title(f'CVE: {cve_id}')
        ax.set_xlabel('Horizon')
        ax.set_ylabel('EPSS Score (Probability)')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        fig.savefig(output_dir / f'pred_vs_truecve{cve_id}.png')
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Compute and save metrics by forecast horizon and CVE from a NetCDF file.')
    parser.add_argument('ncfile', type=Path, help='Input NetCDF (.nc) path')
    parser.add_argument('--true-var', default='true', help='Name of true-value variable in NetCDF')
    parser.add_argument('--pred-var', default='pred', help='Name of prediction variable in NetCDF')
    parser.add_argument('--horizons', nargs='+', type=int, default=list(range(1, 31)), help='Horizons to evaluate')
    parser.add_argument('--out-csv', type=Path, default=None, help='Path to save metrics table as CSV')
    parser.add_argument('--plot', action='store_true', help='If set, generate metric plots')
    parser.add_argument('--plot-dir', type=Path, default=None, help='Directory to save plots (used only if --plot is set)')
    parser.add_argument('--plot-preds-vs-true', action='store_true', help='If set, generate prediction vs true plots')
    parser.add_argument('--plot-preds-vs-true-dir', type=Path, default=None, help='Directory to save pred-vs-true plots')
    parser.add_argument('--filter-recent-cves', action='store_true', help='Filter out CVEs from 2025 with insufficient data')

    parser.add_argument(
        '--cves',
        nargs='+',
        type=str,
        default=None,
        help='List of CVE IDs to plot (e.g. CVE-1999-0070 CVE-1999-0508)'
    )

    args = parser.parse_args()

    ds = xr.open_dataset(args.ncfile)
    df = compute_metrics_per_cve(ds, args.true_var, args.pred_var, args.horizons)

    # Filter out CVEs with insufficient data if requested
    if args.filter_recent_cves:
        # Count valid metrics per CVE (excluding NaN)
        cve_valid_counts = df.groupby('CVE')[['MAE', 'RMSE', 'SMAPE (%)', 'R2']].apply(
            lambda x: x.notna().any(axis=1).sum()
        )
        # Dynamic threshold: at least 50% of tested horizons OR minimum 2 horizons
        min_required = max(2, len(args.horizons) // 2)
        valid_cves = cve_valid_counts[cve_valid_counts >= min_required].index
        df = df[df['CVE'].isin(valid_cves)]
        print(f"Filtered to {len(valid_cves)} CVEs with sufficient data (≥{min_required} valid horizons out of {len(args.horizons)} tested)")

    print('\nMetrics by CVE and Horizon:')
    print(df)

    if args.out_csv:
        # Ensure output directory exists
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f'Saved metrics to {args.out_csv}')

    if args.plot:
        if args.plot_dir is None:
            raise ValueError("You must specify --plot-dir to save plots when --plot is set.")
        plot_metrics(df, horizons_to_plot=args.horizons, output_dir=args.plot_dir, filter_cves=args.cves)

    if args.plot_preds_vs_true:
        if args.plot_preds_vs_true_dir is None:
            raise ValueError("You must specify --plot-preds-vs-true-dir to save prediction vs true plots.")
        plot_preds_vs_true(ds, args.true_var, args.pred_var, args.plot_preds_vs_true_dir, filter_cves=args.cves)


if __name__ == '__main__':
    main()

#python eval_plot.py predictions_stream.nc --true-var true --pred-var pred --out-csv metrics_output.csv --plot --plot-dir plots/ --plot-preds-vs-true --plot-preds-vs-true-dir plots/pred_vs_true/ --cves CVE-2024-3094
#python eval_plot.py no_data_predictions_stream.nc --true-var true --pred-var pred --out-csv metrics_output.csv --plot --plot-dir plots/ --plot-preds-vs-true --plot-preds-vs-true-dir plots/pred_vs_true/ --cves CVE-2024-3094
#python eval_plot.py full_data_predictions_stream.nc --true-var true --pred-var pred --out-csv metrics_output.csv --plot --plot-dir plots/ --plot-preds-vs-true --plot-preds-vs-true-dir plots/pred_vs_true/ --cves CVE-2024-3094
#python eval_plot.py predictions_stream_sus_lstm.nc --true-var true --pred-var pred --out-csv metrics_output.csv --plot --plot-dir plots/ --plot-preds-vs-true --plot-preds-vs-true-dir plots/pred_vs_true/ --cves CVE-2024-3094
#python eval_plot.py tcn_model_sus.nc --true-var true --pred-var pred --out-csv metrics_output.csv --plot --plot-dir plots/ --plot-preds-vs-true --plot-preds-vs-true-dir plots/pred_vs_true/ --cves CVE-2024-3094
