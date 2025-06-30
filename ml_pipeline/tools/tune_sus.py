import subprocess
import sys
import json
from pathlib import Path
import csv
import argparse
import itertools

# ---------------------------------------------------------------------------
# Stochastic-Under-Sampling (SUS) tuner
#
# Quickly finds near-optimal (β, Δ, Z) by:
#   1. Pre-computing Z with `compute_weight_quantile.py` (quantile = 0.995)
#   2. Running `lstm_exp_window_eval.py` for 2 epochs (quick-grid mode)
#   3. Reading the resulting CSV written by the trainer and collecting metrics
#   4. Printing an ordered summary so you can pick the best hyper-parameters
#
# NOTE
# ────
# • Designed for short, cheap runs. Increase `--epochs` later for full training.
# • Requires that the Arrow file & vocab are already generated.
# • Always invoked as a module: `python -m ml_pipeline.tools.tune_sus`.
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SUS hyper-parameter tuner")
    p.add_argument("--arrow", type=str, default="ml_pipeline/work/epss_stage1.arrow",
                   help="Path to Arrow file (default: ml_pipeline/work/epss_stage1.arrow)")
    p.add_argument("--betas", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0],
                   help="Space-separated list of β values to try")
    p.add_argument("--look_aheads", type=int, nargs="+", default=[1, 2, 5, 10],
                   help="Space-separated list of Δ (look-ahead) values to try")
    p.add_argument("--epochs", type=int, default=2,
                   help="Epochs to train each configuration (must be 2 for quick mode)")
    p.add_argument("--quantile", type=float, default=0.995,
                   help="Quantile for Z computation (default: 0.995)")
    return p.parse_args()


def run_subprocess(cmd, cwd=None):
    """Run a subprocess and stream its output."""
    print("\n▶", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def compute_Z(arrow_path: Path, beta: float, look_ahead: int, quantile: float):
    """Call the weight-quantile utility to compute & save Z."""
    cmd = [
        sys.executable, "-m", "ml_pipeline.tools.compute_weight_quantile",
        "--arrow", str(arrow_path),
        "--beta", str(beta),
        "--look-ahead", str(look_ahead),
        "--quantile", str(quantile)
    ]
    run_subprocess(cmd)

    out_path = arrow_path.parent / f"sus_config_beta{beta}_d{look_ahead}_q{quantile}.json"
    with open(out_path) as fh:
        z_val = json.load(fh)["Z"]
    return z_val, out_path


def train_quick(beta: float, look_ahead: int, epochs: int):
    """Run the LSTM trainer in quick-grid mode."""
    cmd = [
        sys.executable, "-m", "ml_pipeline.lstm_exp_window_eval",
        "--beta", str(beta),
        "--look-ahead", str(look_ahead),
        "--epochs", str(epochs)
    ]
    run_subprocess(cmd)

    # The trainer always writes a CSV tagged as grid_B{β}_D{Δ}_quant.csv
    tag = f"B{beta}_D{look_ahead}_quant"
    csv_path = Path("results") / f"grid_{tag}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected results CSV not found: {csv_path}")

    # Read the last row (the most recent run)
    with open(csv_path) as fh:
        row = list(csv.reader(fh))[-1]
    # Columns: beta, look_ahead, z_mode, va_loss, recall
    return float(row[3]), float(row[4])  # (val_loss, recall)


def main():
    args = parse_args()
    arrow_path = Path(args.arrow)
    if not arrow_path.exists():
        sys.exit(f"Arrow file not found: {arrow_path}\nRun: python -m ml_pipeline.data_prep.00_build_arrow")

    combos = list(itertools.product(sorted(set(args.betas)), sorted(set(args.look_aheads))))
    print(f"Searching {len(combos)} combinations…\n")

    summary = []  # (val_loss, beta, look_ahead, Z, recall)

    for beta, la in combos:
        print("=" * 80)
        print(f"β = {beta}, Δ = {la}")
        print("-" * 80)
        Z, cfg_path = compute_Z(arrow_path, beta, la, args.quantile)
        print(f"Z computed: {Z:.6f} (config saved to {cfg_path})")

        val_loss, recall = train_quick(beta, la, args.epochs)
        summary.append((val_loss, beta, la, Z, recall))

    # Sort by val_loss ascending (lower is better)
    summary.sort(key=lambda t: t[0])

    print("\n" + "#" * 80)
    print("TUNING SUMMARY (best ∶ lowest validation loss)")
    print("val_loss  β     Δ   Z         spike_recall")
    for vl, b, la, z, rec in summary:
        print(f"{vl:8.4f}  {b:<4.2f}  {la:<3d}  {z:<10.6f}  {rec:.3f}")

    best = summary[0]
    print("\nBEST CONFIG → β = {0}, Δ = {1}, Z = {2:.6f} (val_loss = {3:.4f}, recall = {4:.3f})".format(
        best[1], best[2], best[3], best[0], best[4]))
    print("\nTo run full training: \n  python -m ml_pipeline.lstm_exp_window_eval --beta {0} --look-ahead {1}".format(best[1], best[2]))


if __name__ == "__main__":
    main() 