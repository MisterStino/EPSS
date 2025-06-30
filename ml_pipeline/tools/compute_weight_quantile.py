#!/usr/bin/env python3
"""
Weight Quantile Computation for SUS Sampling
============================================

Pre-computes the normalization constant Z = quantile(weights^β, 0.995)
needed for Stochastic Under-Sampling (SUS) to ensure p ≤ 1.

This script must be run once before training to determine the appropriate
Z value based on the actual distribution of weight magnitudes in the data.
"""

import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from tqdm import tqdm


def compute_weights_from_arrow(arrow_path: Path, 
                              horizon: int = 30,
                              look_ahead: int = 5,
                              max_cves: int = 1000) -> List[float]:
    """
    Compute weights from Arrow file using same logic as CVEIterableDatasetSUS
    
    Args:
        arrow_path: Path to Arrow file  
        horizon: Window length
        look_ahead: Days ahead for weight computation
        max_cves: Limit CVEs processed (for speed)
        
    Returns:
        List of computed weights
    """
    print(f"Computing weights from {arrow_path}")
    print(f"Parameters: horizon={horizon}, look_ahead={look_ahead}")
    
    weights = []
    
    with ipc.open_file(pa.memory_map(str(arrow_path), "r")) as reader:
        schema = reader.schema
        col2idx = {f.name: i for i, f in enumerate(schema)}
        
        if "epss_target" not in col2idx:
            raise ValueError("epss_target column not found in Arrow file")
        
        cur_cve = None
        buf_eps = []
        processed_cves = 0
        
        def process_cve():
            """Process current CVE and compute weights"""
            nonlocal weights, processed_cves
            
            if len(buf_eps) < horizon + look_ahead:
                return  # Sequence too short
                
            eps_np = np.array(buf_eps, dtype=np.float32)
            T = len(eps_np)
            
            # Slide window and compute weights
            for t in range(horizon - 1, T - look_ahead):
                s = t - horizon + 1
                
                # Extract window including future data for weight computation
                window_eps = eps_np[s:t+1+look_ahead]
                
                # Compute weight: |future_max - current|
                if len(window_eps) >= look_ahead + 1:
                    current = window_eps[-look_ahead - 1]
                    future_window = window_eps[-look_ahead:]
                    future_max = np.max(future_window)
                    weight = abs(float(future_max - current))
                    weights.append(weight)
            
            processed_cves += 1
            buf_eps.clear()
        
        print("Processing CVEs...")
        for b in tqdm(range(reader.num_record_batches), desc="Batches"):
            batch = reader.get_batch(b)
            cols = batch.columns
            
            for r in range(batch.num_rows):
                cve = cols[col2idx["cve"]][r].as_py()
                
                if cur_cve is None:
                    cur_cve = cve
                elif cve != cur_cve:
                    process_cve()
                    cur_cve = cve
                    
                    if processed_cves >= max_cves:
                        break
                
                eps_value = cols[col2idx["epss_target"]][r].as_py()
                buf_eps.append(eps_value)
            
            if processed_cves >= max_cves:
                break
        
        # Process final CVE
        process_cve()
    
    print(f"✓ Processed {processed_cves} CVEs")
    print(f"✓ Computed {len(weights)} weights")
    
    return weights


def main():
    parser = argparse.ArgumentParser(description="Compute weight quantile for SUS sampling")
    parser.add_argument("--arrow", type=str, required=True,
                       help="Path to Arrow file")
    parser.add_argument("--horizon", type=int, default=30,
                       help="Window horizon (default: 30)")
    parser.add_argument("--look-ahead", type=int, default=5,
                       help="Look-ahead for weight computation (default: 5)")
    parser.add_argument("--beta", type=float, default=3.0,
                       help="Beta exponent for probability function (default: 3.0)")
    parser.add_argument("--quantile", type=float, default=0.995,
                       help="Quantile for Z computation (default: 0.995)")
    parser.add_argument("--max-cves", type=int, default=1000,
                       help="Maximum CVEs to process (default: 1000)")
    parser.add_argument("--output", type=str, 
                       help="Output JSON file (default: auto-generated)")
    
    args = parser.parse_args()
    
    arrow_path = Path(args.arrow)
    if not arrow_path.exists():
        raise FileNotFoundError(f"Arrow file not found: {arrow_path}")
    
    # Compute weights
    weights = compute_weights_from_arrow(
        arrow_path, 
        horizon=args.horizon,
        look_ahead=args.look_ahead,
        max_cves=args.max_cves
    )
    
    if not weights:
        raise ValueError("No weights computed - check data and parameters")
    
    # Convert to numpy for quantile computation
    weights_np = np.array(weights)
    
    # Compute statistics
    print("\n📊 Weight Statistics:")
    print(f"   Count: {len(weights):,}")
    print(f"   Min: {weights_np.min():.6f}")
    print(f"   Max: {weights_np.max():.6f}")
    print(f"   Mean: {weights_np.mean():.6f}")
    print(f"   Std: {weights_np.std():.6f}")
    print(f"   Median: {np.median(weights_np):.6f}")
    
    # Compute quantiles
    quantiles = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
    print(f"\n📈 Weight Quantiles:")
    for q in quantiles:
        val = np.quantile(weights_np, q)
        print(f"   {q*100:5.1f}%: {val:.6f}")
    
    # Compute Z = quantile(weights^beta, target_quantile)
    weights_beta = weights_np ** args.beta
    Z = np.quantile(weights_beta, args.quantile)
    
    print(f"\n🎯 SUS Parameters:")
    print(f"   Beta (β): {args.beta}")
    print(f"   Target quantile: {args.quantile}")
    print(f"   Z = quantile(weights^{args.beta}, {args.quantile}) = {Z:.6f}")
    
    # Verify Z bounds
    max_weight_beta = weights_beta.max()
    if Z < max_weight_beta:
        print(f"⚠️  WARNING: Z ({Z:.6f}) < max(weights^β) ({max_weight_beta:.6f})")
        print(f"   This means some probabilities will be > 1.0")
        print(f"   Consider using a higher quantile (e.g., 0.999)")
    else:
        print(f"✓ Z properly bounds probabilities (Z ≥ max(weights^β))")
    
    # Create output data
    output_data = {
        "arrow_path": str(arrow_path),
        "horizon": args.horizon,
        "look_ahead": args.look_ahead,
        "beta": args.beta,
        "target_quantile": args.quantile,
        "n_weights": len(weights),
        "weight_stats": {
            "min": float(weights_np.min()),
            "max": float(weights_np.max()),
            "mean": float(weights_np.mean()),
            "std": float(weights_np.std()),
            "median": float(np.median(weights_np))
        },
        "weight_quantiles": {str(q): float(np.quantile(weights_np, q)) for q in quantiles},
        "Z": float(Z),
        "max_weight_beta": float(max_weight_beta),
        "Z_is_valid": bool(Z >= max_weight_beta)
    }
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = arrow_path.parent / f"sus_config_beta{args.beta}_d{args.look_ahead}_q{args.quantile}.json"
    
    # Save configuration
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Configuration saved to: {output_path}")
    print(f"\n🚀 To use in training:")
    print(f"   Z = {Z:.6f}")
    print(f"   CVEIterableDatasetSUS(..., beta={args.beta}, z_norm={Z:.6f})")


if __name__ == "__main__":
    main() 