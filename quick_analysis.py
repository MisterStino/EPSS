import pandas as pd
import pyarrow.feather as feather
import json
import joblib

# Load data
tab = feather.read_table('work/epss_stage1.arrow')
df = tab.to_pandas()
vocab = json.load(open('work/vocab.json'))

print('🔍 COMPREHENSIVE ANALYSIS')
print('='*50)
print(f'Shape: {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'CVEs: {df.cve.nunique():,} unique')
print(f'Date range: {df.date.min()} to {df.date.max()}')
print(f'EPSS: [{df.epss.min():.3f}, {df.epss.max():.3f}] (logit-transformed)')

print('\n📊 Column Types:')
numeric_cols = [c for c in df.columns if df[c].dtype in ["float32", "float64", "int32", "int64"]]
bool_cols = [c for c in df.columns if c.startswith(("has_", "is_"))]
missing_cols = [c for c in df.columns if c.endswith("_missing")]

print(f'  Numeric: {len(numeric_cols)}')
print(f'  Boolean: {len(bool_cols)}')
print(f'  Missing flags: {len(missing_cols)}')

print('\n🎯 Train/Val/Test Split:')
print(f'  Train: {(df.flag_train==1).sum():,} ({(df.flag_train==1).mean()*100:.1f}%)')
print(f'  Val:   {(df.flag_val==1).sum():,} ({(df.flag_val==1).mean()*100:.1f}%)')
print(f'  Test:  {(df.flag_test==1).sum():,} ({(df.flag_test==1).mean()*100:.1f}%)')

print('\n🏷️ Categorical Encoding:')
for col, mapping in vocab.items():
    if col in df.columns:
        print(f'  {col}: {len(mapping)} vocab, range [{df[col].min()}, {df[col].max()}]')

print('\n📈 Sample of Event Features:')
event_cols = ['has_discovery', 'has_release', 'event_sequence', 'cumulative_source_count']
for col in event_cols:
    if col in df.columns:
        non_zero = (df[col] != 0).sum() if col in ['has_discovery', 'has_release'] else df[col].notna().sum()
        print(f'  {col}: {non_zero:,} non-zero/non-null ({non_zero/len(df)*100:.1f}%)')

print('\n✅ Data Integrity Checks:')
print(f'  ✓ All CVEs have data: {df.cve.notna().all()}')
print(f'  ✓ All dates valid: {df.date.notna().all()}')
print(f'  ✓ EPSS transformed: {df.epss.min() < 0 and df.epss.max() > 0}')
print(f'  ✓ Flags sum to 1: {(df[["flag_train", "flag_val", "flag_test"]].sum(axis=1) == 1).all()}')

# Check a few CVE time series
print('\n📅 Sample CVE Time Series:')
sample_cves = df.cve.unique()[:3]
for cve in sample_cves:
    cve_data = df[df.cve == cve]
    print(f'  {cve}: {len(cve_data)} time steps, {cve_data.date.min()} to {cve_data.date.max()}') 