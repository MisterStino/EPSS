#!/usr/bin/env python3
"""Compare signals between coworker commit data and our GitHub event data."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to data files
commit_csv = os.path.join(os.path.dirname(__file__), '..', 'raw', 'github_commit_timestamps_9k.csv')
bq_csv = os.path.join(os.path.dirname(__file__), '..', 'raw', 'github_bq.csv')

# Load data
print("Loading coworker commit data...")
commit_df = pd.read_csv(commit_csv, parse_dates=['commit_date'])
print(f"Loaded {len(commit_df)} commit records.")
print("Loading our GitHub event data...")
bq_df = pd.read_csv(bq_csv, parse_dates=['date'])
print(f"Loaded {len(bq_df)} event records.")

# Standardize CVE IDs to uppercase
commit_df['cve_id'] = commit_df['cve_id'].str.upper()
bq_df['cve_id'] = bq_df['cve_id'].str.upper()

# Rename date columns for consistency
commit_df = commit_df.rename(columns={'commit_date': 'date'})

# Compute total events per record
event_cols = [col for col in bq_df.columns if col not in ['date', 'cve_id']]
bq_df['total_events'] = bq_df[event_cols].sum(axis=1)

# Merge datasets on CVE ID and date
print("Merging datasets...")
df = pd.merge(commit_df, bq_df, on=['cve_id', 'date'], how='outer')
df = df.fillna(0)
print(f"Merged dataset has {len(df)} records.")

# Compute correlation between counts
corr = df['commit_count'].corr(df['total_events'])
print(f"Correlation between coworker commit_count and total GitHub events: {corr:.3f}")

# Binary activity signals
df['commit_signal'] = df['commit_count'] > 0
df['event_signal'] = df['total_events'] > 0

# Compute confusion matrix
tp = ((df['commit_signal']) & (df['event_signal'])).sum()
fp = ((~df['commit_signal']) & (df['event_signal'])).sum()
fn = ((df['commit_signal']) & (~df['event_signal'])).sum()
tn = ((~df['commit_signal']) & (~df['event_signal'])).sum()
precision = tp / (tp + fp) if tp + fp > 0 else np.nan
recall = tp / (tp + fn) if tp + fn > 0 else np.nan
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

print("\nBinary signal agreement metrics:")
print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}\n")

# Create output directory for plots
out_dir = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(out_dir, exist_ok=True)

# Scatter plot of counts (sampled for speed)
print("Creating scatter plot...")
sample = df.sample(min(1000, len(df)), random_state=42)
plt.figure(figsize=(6, 4))
sns.scatterplot(x='commit_count', y='total_events', data=sample)
plt.xlabel('Coworker commit_count')
plt.ylabel('Total GitHub events')
plt.title('Commit count vs. GitHub events')
plt.tight_layout()
scatter_path = os.path.join(out_dir, 'commit_vs_events_scatter.png')
plt.savefig(scatter_path)
print(f"Saved scatter plot to {scatter_path}")

# Identify top CVEs with highest disagreement
print("Identifying top CVEs with disagreement...")
df['agreement'] = df['commit_signal'] == df['event_signal']
disagree_counts = df[~df['agreement']].groupby('cve_id').size().sort_values(ascending=False)
top_disagree = disagree_counts.head(5).index.tolist()

for cve in top_disagree:
    sub = df[df['cve_id'] == cve].sort_values('date')
    plt.figure(figsize=(8, 3))
    plt.plot(sub['date'], sub['commit_count'], label='commit_count')
    plt.plot(sub['date'], sub['total_events'], label='total_events')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title(f'{cve} signals over time')
    plt.legend()
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'{cve}_signals.png')
    plt.savefig(fig_path)
    print(f"Saved time series plot for {cve} to {fig_path}")

print("Analysis complete.")