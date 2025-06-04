#!/usr/bin/env python3
"""Step 1: Load and merge GitHub event and EPSS score data, and validate merge."""

import pandas as pd
import os

# Paths
tool_dir = os.path.dirname(__file__)
bq_csv = os.path.join(tool_dir, '..', 'raw', 'github_bq.csv')
epss_parquet = os.path.join(tool_dir, '..', '..', 'epss', 'epss_parquet', 'epss_all.parquet')

# Load GitHub data
print("Loading GitHub event data from:", bq_csv)
gh_df = pd.read_csv(bq_csv, parse_dates=['date'])
gh_df['cve_id'] = gh_df['cve_id'].str.upper()
print("GitHub data shape:", gh_df.shape)
print("GitHub columns:", gh_df.columns.tolist())
print(gh_df.head(), "\n")

# Load EPSS data
print("Loading EPSS data from:", epss_parquet)
epss_df = pd.read_parquet(epss_parquet)
epss_df = epss_df.rename(columns={'cve':'cve_id'})
# ensure date type
if epss_df['date'].dtype == object:
    epss_df['date'] = pd.to_datetime(epss_df['date'])
epss_df['cve_id'] = epss_df['cve_id'].str.upper()
print("EPSS data shape:", epss_df.shape)
print("EPSS columns:", epss_df.columns.tolist())
print(epss_df.head(), "\n")

# Merge on cve_id and date
merged = pd.merge(gh_df, epss_df, on=['cve_id', 'date'], how='inner')
print("Merged data shape:", merged.shape)
print("Merged columns:", merged.columns.tolist())
print(merged.head()) 