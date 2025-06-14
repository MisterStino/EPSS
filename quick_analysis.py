import pandas as pd
import numpy as np

# Load data and analyze sequence lengths
df = pd.read_parquet('data/full_db/sampled/final_full_data_sampled.parquet')
seq_lengths = df.groupby('cve').size()

print("=== SEQUENCE LENGTH ANALYSIS ===")
print(f"L_max (longest sequence): {seq_lengths.max()}")
print(f"Mean sequence length: {seq_lengths.mean():.1f}")
print(f"Median sequence length: {seq_lengths.median():.0f}")
print(f"75th percentile: {seq_lengths.quantile(0.75):.0f}")
print(f"95th percentile: {seq_lengths.quantile(0.95):.0f}")
print(f"Total CVEs: {len(seq_lengths):,}")

# Memory analysis for different batch sizes
L_max = seq_lengths.max()
horizon = 10

print(f"\n=== MEMORY ANALYSIS (L_max={L_max}) ===")
for batch_size in [32, 64, 128, 256, 512]:
    # Memory components (float32 = 4 bytes)
    X_mem = batch_size * L_max * 2 * 4 / (1024**2)  # [B, L_max, 2]
    Y_mem = batch_size * L_max * horizon * 4 / (1024**2)  # [B, L_max, H]
    masks_mem = batch_size * L_max * (1 + horizon + 1) * 4 / (1024**2)  # 3 masks
    
    # Forward pass memory
    forward_mem = X_mem + Y_mem + masks_mem
    
    # Total with gradients + optimizer states (Adam stores 2x params)
    total_mem = forward_mem * 3  # Conservative estimate
    
    print(f"Batch {batch_size:3d}: {forward_mem:6.1f} MB forward, {total_mem:6.1f} MB total")

print(f"\n=== 16GB GPU RECOMMENDATIONS ===")
print("Available GPU memory: ~14GB (after OS/driver overhead)")
print("Recommended batch sizes:")
print("  - Conservative: 32-64 (safe, allows other processes)")
print("  - Aggressive: 128-256 (max performance, monitor memory)")
print("  - If OOM: Reduce batch size or implement gradient accumulation") 