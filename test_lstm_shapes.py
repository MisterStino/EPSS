#!/usr/bin/env python
"""
Comprehensive shape and data lineage tests for LSTM pipeline
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import json

def create_test_data():
    """Create small, controlled test dataset"""
    data = []
    base_date = pd.Timestamp('2024-01-01')
    
    # CVE-1: 3 time steps
    for i in range(3):
        data.append({
            'cve': 'CVE-1', 'date': base_date + pd.Timedelta(days=i),
            'epss': 0.001 * (i+1), 'published_date': base_date - pd.Timedelta(days=5),
            'last_modified_date': base_date + pd.Timedelta(days=i-1),
            'desc_len': 100 + i*10, 'has_discovery': i%2, 
            'primary_cvss_sev': 'HIGH', 'cwe_id': f'CWE-{100+i}'
        })
    
    # CVE-2: 5 time steps  
    for i in range(5):
        data.append({
            'cve': 'CVE-2', 'date': base_date + pd.Timedelta(days=10+i),
            'epss': 0.005 * (i+1), 'published_date': base_date + pd.Timedelta(days=5),
            'last_modified_date': base_date + pd.Timedelta(days=8+i),
            'desc_len': 200 + i*20, 'has_discovery': 1,
            'primary_cvss_sev': 'CRITICAL', 'cwe_id': f'CWE-{200+i}'
        })
    
    df = pd.DataFrame(data)
    for col in ['date', 'published_date', 'last_modified_date']:
        df[col] = pd.to_datetime(df[col])
    
    return df

def test_preprocessing(df):
    """Test data preprocessing steps"""
    print("=== TESTING PREPROCESSING ===")
    print(f"Raw data: {df.shape}")
    
    # Test timestamp deltas
    df['published_date_delta'] = (df['date'] - df['published_date']).dt.days.astype('float32')
    
    delta = (df['date'] - df['last_modified_date']).dt.days.astype('float32')
    negatives = (delta < 0).sum()
    delta[delta < 0] = np.nan
    df['last_modified_date_delta'] = delta
    print(f"Clipped {negatives} negative deltas")
    
    # Drop original timestamps
    df.drop(columns=['published_date', 'last_modified_date'], inplace=True)
    
    # Add flags
    days = np.sort(df['date'].unique())
    val_cut = pd.to_datetime(days[int(0.6 * len(days))])
    test_cut = pd.to_datetime(days[int(0.8 * len(days))])
    
    df['flag_train'] = (df['date'] < val_cut).astype('uint8')
    df['flag_val'] = ((df['date'] >= val_cut) & (df['date'] < test_cut)).astype('uint8')
    df['flag_test'] = (df['date'] >= test_cut).astype('uint8')
    
    print(f"Processed data: {df.shape}")
    return df

def test_feature_engineering(df):
    """Test feature engineering"""
    print("\n=== TESTING FEATURE ENGINEERING ===")
    
    # Transform EPSS
    eps = 1e-6
    p = np.clip(df['epss'].values, eps, 1.0 - eps)
    df['epss'] = np.log(p / (1.0 - p)).astype('float32')
    
    # Categorize columns
    bool_cols = ['has_discovery']  
    cat_cols = ['primary_cvss_sev', 'cwe_id']
    num_cols = ['desc_len', 'published_date_delta', 'last_modified_date_delta']
    
    # Process booleans
    for col in bool_cols:
        df[col] = df[col].fillna(0).astype('uint8')
    
    # Process categoricals (train-only vocab)
    train_mask = df['flag_train'] == 1
    vocab = {}
    for col in cat_cols:
        cats = df.loc[train_mask, col].dropna().unique()
        vocab[col] = {'UNK': 0, **{c: i+1 for i, c in enumerate(sorted(cats))}}
        df[col] = df[col].map(vocab[col]).fillna(0).astype('int16')
    
    # Process numerics
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Handle missing
    for col in num_cols:
        miss = df[col].isna()
        df[f'{col}_missing'] = miss.astype('uint8')
        df.loc[miss, col] = -10.0  # sentinel
    
    missing_cols = [c for c in df.columns if c.endswith('_missing')]
    
    print(f"Features: {len(num_cols)} numeric, {len(bool_cols)} boolean, {len(cat_cols)} categorical")
    print(f"Missing flags: {len(missing_cols)}")
    
    return df, num_cols, bool_cols, cat_cols, vocab

class TestDataset:
    """Test dataset class"""
    def __init__(self, df, L_max, horizon, flag_col, num_cols, bool_cols, cat_cols):
        self.num, self.boo, self.cat = [], [], []
        self.Y, self.m_t, self.m_h, self.m_eval = [], [], [], []
        
        for cve, idx in df.groupby('cve').groups.items():
            idx = np.array(idx)
            T = len(idx)
            pad = L_max - T
            
                                      # Features with padding
             num_data = df.loc[idx, num_cols].values.astype('float32')
             boo_data = df.loc[idx, bool_cols].values.astype('uint8')
             cat_data = df.loc[idx, cat_cols].values.astype('int64')  # PyTorch embeddings need int64
            
            self.num.append(torch.from_numpy(np.pad(num_data, ((0,pad),(0,0)))))
            self.boo.append(torch.from_numpy(np.pad(boo_data, ((0,pad),(0,0)))))
            self.cat.append(torch.from_numpy(np.pad(cat_data, ((0,pad),(0,0)))))
            
            # Masks
            self.m_t.append(torch.from_numpy(np.r_[np.ones(T), np.zeros(pad)].astype('float32')))
            
            flag_data = df.loc[idx, flag_col].values.astype('float32')
            self.m_eval.append(torch.from_numpy(np.r_[flag_data, np.zeros(pad)].astype('float32')))
            
            # Targets
            epss_data = df.loc[idx, 'epss'].values.astype('float32')
            y = np.zeros((L_max, horizon), 'float32')
            mh = np.zeros_like(y)
            
            for t in range(T):
                k = min(horizon, T - t - 1)
                if k:
                    y[t, :k] = epss_data[t+1:t+1+k]
                    mh[t, :k] = 1
            
            self.Y.append(torch.from_numpy(y))
            self.m_h.append(torch.from_numpy(mh))
    
    def __len__(self):
        return len(self.num)
    
    def __getitem__(self, i):
        return (self.num[i], self.boo[i], self.cat[i], 
                self.Y[i], self.m_t[i], self.m_h[i], self.m_eval[i])

class TestModel(nn.Module):
    """Test model"""
    def __init__(self, n_num, n_bool, cat_sizes, horizon=3, hidden=16, emb_dim=4):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(s, emb_dim) for s in cat_sizes])
        in_dim = n_num + n_bool + emb_dim * len(cat_sizes)
        
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
    
    def forward(self, num, boo, cat):
        e = torch.cat([emb(cat[..., i]) for i, emb in enumerate(self.emb)], dim=-1)
        x = torch.cat([num, boo.float(), e], dim=-1)
        h, _ = self.lstm(x)
        return self.head(h)

def test_shapes():
    """Main shape testing function"""
    print("🚀 TESTING LSTM PIPELINE SHAPES")
    print("="*50)
    
    # 1. Create test data
    df = create_test_data()
    print(f"Created test data: {len(df)} rows")
    print(f"CVE lengths: {df.groupby('cve').size().to_dict()}")
    
    # 2. Test preprocessing  
    df = test_preprocessing(df)
    
    # 3. Test feature engineering
    df, num_cols, bool_cols, cat_cols, vocab = test_feature_engineering(df)
    
    # 4. Test dataset creation
    print("\n=== TESTING DATASET CREATION ===")
    L_max = df.groupby('cve').size().max()
    horizon = 3
    
    print(f"L_max: {L_max}")
    
    dataset = TestDataset(df, L_max, horizon, 'flag_train', num_cols, bool_cols, cat_cols)
    print(f"Dataset size: {len(dataset)} CVEs")
    
    # Test item shapes
    item = dataset[0]
    expected_shapes = {
        'num': (L_max, len(num_cols)),
        'boo': (L_max, len(bool_cols)), 
        'cat': (L_max, len(cat_cols)),
        'Y': (L_max, horizon),
        'm_t': (L_max,),
        'm_h': (L_max, horizon),
        'm_eval': (L_max,)
    }
    
    actual_shapes = {name: tensor.shape for name, tensor in zip(
        ['num', 'boo', 'cat', 'Y', 'm_t', 'm_h', 'm_eval'], item)}
    
    print("\nItem shapes:")
    for name in expected_shapes:
        exp, act = expected_shapes[name], actual_shapes[name]
        status = "✅" if exp == act else "❌"
        print(f"  {name}: expected {exp}, actual {act} {status}")
        assert exp == act, f"Shape mismatch for {name}"
    
    # 5. Test model forward pass
    print("\n=== TESTING MODEL FORWARD PASS ===")
    cat_sizes = [len(vocab[col]) for col in cat_cols]
    model = TestModel(len(num_cols), len(bool_cols), cat_sizes, horizon)
    
    # Single item test
    with torch.no_grad():
        pred = model(item[0].unsqueeze(0), item[1].unsqueeze(0), item[2].unsqueeze(0))
    
    expected_pred_shape = (1, L_max, horizon)
    print(f"Single prediction shape: expected {expected_pred_shape}, actual {pred.shape}")
    assert pred.shape == expected_pred_shape
    
    # Batch test
    batch_size = 2
    batch_items = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    
    num_batch = torch.stack([item[0] for item in batch_items])
    boo_batch = torch.stack([item[1] for item in batch_items])  
    cat_batch = torch.stack([item[2] for item in batch_items])
    
    with torch.no_grad():
        pred_batch = model(num_batch, boo_batch, cat_batch)
    
    expected_batch_shape = (len(batch_items), L_max, horizon)
    print(f"Batch prediction shape: expected {expected_batch_shape}, actual {pred_batch.shape}")
    assert pred_batch.shape == expected_batch_shape
    
    # 6. Test masking
    print("\n=== TESTING MASKING ===")
    for i, item in enumerate(batch_items):
        m_t, m_eval, m_h = item[4], item[6], item[5]
        real_steps = int(m_t.sum())
        eval_steps = int(m_eval.sum()) 
        total_preds = int(m_h.sum())
        print(f"  CVE {i}: {real_steps} real, {eval_steps} eval, {total_preds} predictions")
    
    print("\n🎉 ALL SHAPE TESTS PASSED!")
    print("="*50)
    
    return True

if __name__ == "__main__":
    test_shapes() 