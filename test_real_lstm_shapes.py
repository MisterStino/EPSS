#!/usr/bin/env python
"""Test real LSTM components with synthetic data"""

import numpy as np
import pandas as pd
import torch
import sys
import os

# Add models path
sys.path.append(os.path.join('models', 'models'))

def create_test_data():
    """Create synthetic data matching real schema"""
    print("Creating test data...")
    
    base_date = pd.Timestamp('2024-01-01')
    data = []
    
    # CVE-1: 3 steps
    for i in range(3):
        data.append({
            'cve': 'CVE-2024-0001',
            'date': base_date + pd.Timedelta(days=i),
            'epss': 0.001 + i * 0.001,
            'published_date': base_date - pd.Timedelta(days=5),
            'last_modified_date': base_date + pd.Timedelta(days=i-1),
            'snapshot_date': base_date + pd.Timedelta(days=i),
            'age_epss_pub': i + 10,
            'desc_len_en': 100 + i * 10,
            'has_discovery': i % 2,
            'has_release': 1,
            'is_disputed': 0,
            'has_v2': i == 0,
            'has_v30': i == 1,
            'has_v31': i == 2,
            'primary_cvss_ver': '3.1',
            'primary_cvss_sev': 'HIGH',
            'dominant_event_type': 'PUBLISHED',
            'cwe_id': f'CWE-{100+i}',
            'vuln_status': 'PUBLISHED',
        })
    
    # CVE-2: 5 steps
    for i in range(5):
        data.append({
            'cve': 'CVE-2024-0002',
            'date': base_date + pd.Timedelta(days=10+i),
            'epss': 0.005 + i * 0.002,
            'published_date': base_date + pd.Timedelta(days=5),
            'last_modified_date': base_date + pd.Timedelta(days=8+i),
            'snapshot_date': base_date + pd.Timedelta(days=12+i),
            'age_epss_pub': i + 20,
            'desc_len_en': 200 + i * 20,
            'has_discovery': 1,
            'has_release': i % 2,
            'is_disputed': 0,
            'has_v2': 0,
            'has_v30': 0,
            'has_v31': i < 3,
            'primary_cvss_ver': '4.0',
            'primary_cvss_sev': 'CRITICAL',
            'dominant_event_type': 'MODIFIED',
            'cwe_id': f'CWE-{200+i}',
            'vuln_status': 'PUBLISHED',
        })
    
    df = pd.DataFrame(data)
    for col in ['date', 'published_date', 'last_modified_date', 'snapshot_date']:
        df[col] = pd.to_datetime(df[col])
    
    print(f"Created {len(df)} rows, {len(df['cve'].unique())} CVEs")
    return df

def test_real_components():
    """Test with real components"""
    print("Testing with REAL LSTM components")
    print("="*50)
    
    # CRITICAL: Patch pandas.read_parquet to avoid loading massive dataset
    import pandas as pd
    original_read_parquet = pd.read_parquet
    
    def mock_read_parquet(path, **kwargs):
        print(f"INTERCEPTED: Would load {path} but using synthetic data instead")
        return create_test_data()  # Return our tiny synthetic data
    
    # Apply the patch
    pd.read_parquet = mock_read_parquet
    
    # Import real components (now safe!)
    try:
        from lstm_exp_window_eval import (
            transform_epss, CVEDataset, Seq2SeqLSTM, masked_mse,
            DROP_COLS, TS_SAFE, TS_LEAKY, BOOL_COLS, CAT_COLS, SENTINEL
        )
        print("Successfully imported real components (with mock data)")
    except Exception as e:
        print(f"Import failed: {e}")
        return False
    finally:
        # Restore original function
        pd.read_parquet = original_read_parquet
    
    # Use our synthetic data
    df = create_test_data()
    
    # Apply real preprocessing
    print("\nApplying real preprocessing...")
    
    # Drop columns
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    
    # Timestamp deltas
    for col in TS_SAFE:
        if col in df.columns:
            df[f"{col}_delta"] = (df["date"] - df[col]).dt.days.astype("float32")
    
    for col in TS_LEAKY:
        if col in df.columns:
            d = (df["date"] - df[col]).dt.days.astype("float32")
            d[d < 0] = np.nan
            df[f"{col}_delta"] = d
    
    df = df.drop(columns=[c for c in TS_SAFE + TS_LEAKY if c in df.columns])
    
    # Calendar split
    days = np.sort(df["date"].unique())
    VAL_CUT = pd.to_datetime(days[int(0.64 * len(days))])
    TEST_CUT = pd.to_datetime(days[int(0.80 * len(days))])
    
    df["flag_train"] = (df["date"] < VAL_CUT).astype("uint8")
    df["flag_val"] = ((df["date"] >= VAL_CUT) & (df["date"] < TEST_CUT)).astype("uint8")
    df["flag_test"] = (df["date"] >= TEST_CUT).astype("uint8")
    
    # EPSS transform
    df["epss"] = transform_epss(df["epss"].values, mode="logit", eps=1e-6)
    print(f"EPSS transform: range [{df['epss'].min():.3f}, {df['epss'].max():.3f}]")
    
    # Feature engineering
    bool_cols = BOOL_COLS(df)
    for col in bool_cols:
        df[col] = df[col].fillna(0).astype("uint8")
    
    train_mask = df["flag_train"] == 1
    VOCAB = {}
    for col in CAT_COLS:
        if col in df.columns:
            cats = df.loc[train_mask, col].dropna().unique()
            VOCAB[col] = {"UNK": 0, **{c: i + 1 for i, c in enumerate(sorted(cats))}}
            df[col] = df[col].map(VOCAB[col]).fillna(0).astype("int16")
    
    NUM_COLS = [c for c, t in df.dtypes.items()
                if np.issubdtype(t, np.number)
                and c not in bool_cols + ["flag_train", "flag_val", "flag_test"]]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(df.loc[train_mask, NUM_COLS])
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    
    for col in NUM_COLS:
        miss = df[col].isna()
        df[f"{col}_missing"] = miss.astype("uint8")
        df.loc[miss, col] = SENTINEL
    
    print(f"Features: {len(NUM_COLS)} numeric, {len(bool_cols)} boolean")
    
    # Test dataset
    print("\nTesting CVEDataset...")
    L_max = df.groupby("cve", observed=True).size().max()
    horizon = 5
    
    # Set global NUM_COLS
    sys.modules['lstm_exp_window_eval'].NUM_COLS = NUM_COLS
    
    dataset = CVEDataset(df, L_max, horizon, "flag_train")
    print(f"Dataset: {len(dataset)} CVEs, L_max={L_max}")
    
    # Test shapes
    item = dataset[0]
    print("Dataset item shapes:")
    print(f"  num: {item[0].shape}")
    print(f"  boo: {item[1].shape}")
    print(f"  cat: {item[2].shape}")
    print(f"  Y: {item[3].shape}")
    print(f"  masks: {item[4].shape}, {item[5].shape}, {item[6].shape}")
    
    # Test model
    print("\nTesting Seq2SeqLSTM...")
    cat_sizes = [len(VOCAB[col]) for col in CAT_COLS if col in VOCAB]
    
    model = Seq2SeqLSTM(
        n_num=len(NUM_COLS),
        n_bool=len(bool_cols),
        cat_sizes=cat_sizes,
        horizon=horizon,
        hidden=16,
        layers=1,
        emb_dim=4
    )
    
    # Forward pass
    with torch.no_grad():
        pred = model(
            item[0].unsqueeze(0),
            item[1].unsqueeze(0),
            item[2].unsqueeze(0).long()
        )
    
    print(f"Prediction shape: {pred.shape}")
    expected = (1, L_max, horizon)
    print(f"Expected: {expected}")
    
    if pred.shape == expected:
        print("✅ Shape test PASSED!")
    else:
        print("❌ Shape test FAILED!")
        return False
    
    # Test loss
    loss = masked_mse(
        pred,
        item[3].unsqueeze(0),
        item[4].unsqueeze(0),
        item[5].unsqueeze(0),
        item[6].unsqueeze(0)
    )
    print(f"Loss: {loss.item():.6f}")
    
    print("\n🎉 ALL TESTS PASSED!")
    return True

if __name__ == "__main__":
    success = test_real_components()
    if not success:
        exit(1) 