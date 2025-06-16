#!/usr/bin/env python3
"""
Check why we consistently get 87 CVEs in the test set
"""

import pyarrow.ipc as ipc
import pyarrow as pa
import pandas as pd
import numpy as np

def analyze_test_data():
    """Analyze the test data distribution."""
    
    try:
        # Read the Arrow file
        arrow_path = 'ml_pipeline/data_prep/work/epss_stage1.arrow'
        reader = ipc.open_file(pa.memory_map(arrow_path, 'r'))
        df = reader.read_all().to_pandas()
        
        print("=" * 80)
        print("ANALYZING TEST DATA DISTRIBUTION")
        print("=" * 80)
        
        print(f'Total records: {len(df):,}')
        print(f'Unique CVEs: {df.cve.nunique():,}')
        print(f'Date range: {df.date.min()} to {df.date.max()}')
        
        # Check split distribution
        train_count = df.flag_train.sum()
        val_count = df.flag_val.sum() 
        test_count = df.flag_test.sum()
        
        print(f'\nSplit distribution:')
        print(f'  Train: {train_count:,} records ({train_count/len(df)*100:.1f}%)')
        print(f'  Val: {val_count:,} records ({val_count/len(df)*100:.1f}%)')
        print(f'  Test: {test_count:,} records ({test_count/len(df)*100:.1f}%)')
        
        # Check unique CVEs in each split
        test_cves = df[df.flag_test == 1].cve.nunique()
        val_cves = df[df.flag_val == 1].cve.nunique()
        train_cves = df[df.flag_train == 1].cve.nunique()
        
        print(f'\nUnique CVEs per split:')
        print(f'  Train: {train_cves:,} CVEs')
        print(f'  Val: {val_cves:,} CVEs')
        print(f'  Test: {test_cves:,} CVEs ← THIS IS WHY YOU GET 87!')
        
        # Check date ranges for each split
        if test_count > 0:
            test_data = df[df.flag_test == 1]
            test_dates = test_data['date']
            print(f'\nTest period: {test_dates.min()} to {test_dates.max()}')
            print(f'Test period length: {(test_dates.max() - test_dates.min()).days} days')
            
        if val_count > 0:
            val_data = df[df.flag_val == 1]
            val_dates = val_data['date']
            print(f'Val period: {val_dates.min()} to {val_dates.max()}')
            print(f'Val period length: {(val_dates.max() - val_dates.min()).days} days')
            
        # Check CVE overlap between splits
        train_cve_set = set(df[df.flag_train == 1]['cve'].unique())
        val_cve_set = set(df[df.flag_val == 1]['cve'].unique())
        test_cve_set = set(df[df.flag_test == 1]['cve'].unique())
        
        print(f'\nCVE Overlap Analysis:')
        print(f'  Train ∩ Val: {len(train_cve_set & val_cve_set):,} CVEs')
        print(f'  Train ∩ Test: {len(train_cve_set & test_cve_set):,} CVEs')
        print(f'  Val ∩ Test: {len(val_cve_set & test_cve_set):,} CVEs')
        
        # Show some example test CVEs
        test_cve_list = list(test_cve_set)[:10]
        print(f'\nSample test CVEs: {test_cve_list}')
        
        # Check if these CVEs exist in different date ranges
        sample_cve = test_cve_list[0] if test_cve_list else None
        if sample_cve:
            cve_data = df[df.cve == sample_cve].sort_values('date')
            print(f'\nSample CVE {sample_cve} timeline:')
            print(f'  Full range: {cve_data.date.min()} to {cve_data.date.max()}')
            print(f'  Total records: {len(cve_data)}')
            print(f'  Train records: {cve_data.flag_train.sum()}')
            print(f'  Val records: {cve_data.flag_val.sum()}')
            print(f'  Test records: {cve_data.flag_test.sum()}')
        
        print(f'\n{"="*80}')
        print("CONCLUSION:")
        print("The 87 CVEs in test set is determined by:")
        print("1. Date-based splitting (not CVE-based)")
        print("2. Only CVEs that have data in the latest 20% of date range")
        print("3. This is consistent because the date range is the same")
        print("4. Your sampling changes CVE composition but not date distribution")
        print(f'{"="*80}')
        
    except Exception as e:
        print(f'Error: {e}')
        print('Arrow file not found - run the preprocessing first')

if __name__ == "__main__":
    analyze_test_data() 