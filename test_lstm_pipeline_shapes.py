#!/usr/bin/env python
"""
Comprehensive unit tests for LSTM pipeline shape transformations and data lineage.
Uses controlled synthetic data to validate every assumption.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_synthetic_dataset() -> pd.DataFrame:
    """
    Create controlled synthetic dataset that mimics real structure but is trackable.
    
    Design:
    - 3 CVEs with different sequence lengths (3, 5, 7 time steps)
    - Predictable date sequence
    - Known values for all features
    - Includes edge cases for testing
    """
    print("🔬 Creating synthetic dataset...")
    
    # Define our test CVEs with different sequence lengths
    cve_data = []
    base_date = pd.Timestamp('2024-01-01')
    
    # CVE-1: Short sequence (3 time steps)
    for i in range(3):
        cve_data.append({
            'cve': 'CVE-2024-0001',
            'date': base_date + pd.Timedelta(days=i),
            'epss': 0.001 + i * 0.001,  # 0.001, 0.002, 0.003
            'age_epss_pub': i + 10,     # 10, 11, 12
            
            # Timestamps (some will create negative deltas for testing)
            'published_date': base_date - pd.Timedelta(days=5),  # Always before observation
            'last_modified_date': base_date + pd.Timedelta(days=i-1),  # Mix of past/future
            'snapshot_date': base_date + pd.Timedelta(days=i),   # Mix of past/future
            
            # Numeric features
            'desc_len_en': 100 + i * 10,
            'n_vendors': 1 + i,
            'n_refs': 2 + i,
            
            # Boolean features  
            'has_discovery': i % 2,
            'has_release': 1,
            'is_disputed': 0,
            
            # Categorical features
            'primary_cvss_ver': '3.1',
            'primary_cvss_sev': 'HIGH' if i > 1 else 'MEDIUM',
            'vuln_status': 'PUBLISHED',
            'cwe_id': f'CWE-{100+i}',
        })
    
    # CVE-2: Medium sequence (5 time steps) 
    for i in range(5):
        cve_data.append({
            'cve': 'CVE-2024-0002', 
            'date': base_date + pd.Timedelta(days=10+i),
            'epss': 0.005 + i * 0.002,  # 0.005, 0.007, 0.009, 0.011, 0.013
            'age_epss_pub': i + 20,
            
            'published_date': base_date + pd.Timedelta(days=5),
            'last_modified_date': base_date + pd.Timedelta(days=8+i),
            'snapshot_date': base_date + pd.Timedelta(days=12+i),
            
            'desc_len_en': 200 + i * 20,
            'n_vendors': 2 + i,
            'n_refs': 3 + i,
            
            'has_discovery': 1,
            'has_release': i % 2,
            'is_disputed': i == 4,  # Only last one
            
            'primary_cvss_ver': '4.0',
            'primary_cvss_sev': 'CRITICAL' if i > 2 else 'HIGH',
            'vuln_status': 'PUBLISHED',
            'cwe_id': f'CWE-{200+i}',
        })
    
    # CVE-3: Long sequence (7 time steps)
    for i in range(7):
        cve_data.append({
            'cve': 'CVE-2024-0003',
            'date': base_date + pd.Timedelta(days=20+i),
            'epss': 0.010 + i * 0.005,  # 0.010 to 0.040
            'age_epss_pub': i + 30,
            
            'published_date': base_date + pd.Timedelta(days=15),
            'last_modified_date': base_date + pd.Timedelta(days=18+i),
            'snapshot_date': base_date + pd.Timedelta(days=22+i),
            
            'desc_len_en': 300 + i * 30,
            'n_vendors': 3 + i,
            'n_refs': 4 + i,
            
            'has_discovery': i % 3 == 0,
            'has_release': 1,
            'is_disputed': 0,
            
            'primary_cvss_ver': '3.1' if i < 4 else '4.0',
            'primary_cvss_sev': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][i % 4],
            'vuln_status': 'PUBLISHED',
            'cwe_id': f'CWE-{300+i}',
        })
    
    df = pd.DataFrame(cve_data)
    df['date'] = pd.to_datetime(df['date'])
    df['published_date'] = pd.to_datetime(df['published_date'])
    df['last_modified_date'] = pd.to_datetime(df['last_modified_date'])
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    print(f"✅ Created synthetic dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"   CVE-1: 3 time steps, CVE-2: 5 time steps, CVE-3: 7 time steps")
    print(f"   L_max expected: {df.groupby('cve').size().max()}")
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPONENTS (SIMPLIFIED VERSIONS FOR TESTING)
# ═══════════════════════════════════════════════════════════════════════════════

def transform_epss(arr: np.ndarray, mode: str = "logit", eps: float = 1e-6) -> np.ndarray:
    """Transform EPSS probabilities."""
    p = np.clip(arr.astype("float64"), eps, 1.0 - eps)
    if mode == "logit":
        out = np.log(p / (1.0 - p))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return out.astype("float32")

def get_column_categorization(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize columns like the real pipeline."""
    DROP_COLS = []  # No drops in synthetic data
    
    TS_SAFE = ["published_date"]
    TS_LEAKY = ["last_modified_date", "snapshot_date"]
    
    BOOL_COLS = [c for c in df.columns if c.startswith(("has_", "is_"))]
    
    CAT_COLS = [
        "primary_cvss_ver", "primary_cvss_sev", 
        "vuln_status", "cwe_id"
    ]
    
    return {
        'drop': DROP_COLS,
        'ts_safe': TS_SAFE,
        'ts_leaky': TS_LEAKY,
        'bool': BOOL_COLS,
        'cat': CAT_COLS
    }

class TestCVEDataset:
    """Simplified dataset class for testing."""
    def __init__(self, df: pd.DataFrame, L_max: int, horizon: int, flag_col: str, 
                 num_cols: List[str], bool_cols: List[str], cat_cols: List[str]):
        
        self.num, self.boo, self.cat = [], [], []
        self.Y, self.m_t, self.m_h, self.m_eval = [], [], [], []
        
        # Convert to numpy arrays
        num_data = df[num_cols].to_numpy("float32")
        boo_data = df[bool_cols].to_numpy("uint8") 
        cat_data = df[cat_cols].to_numpy("int16")
        epss_data = df["epss"].to_numpy("float32").reshape(-1, 1)
        flag_data = df[flag_col].to_numpy("float32")
        
        # Group by CVE and create sequences
        for cve, group_idx in df.groupby("cve", sort=False).groups.items():
            idx = np.array(group_idx)
            T = len(idx)
            pad = L_max - T
            
            # Features with padding
            self.num.append(torch.from_numpy(np.pad(num_data[idx], ((0, pad), (0, 0)))))
            self.boo.append(torch.from_numpy(np.pad(boo_data[idx], ((0, pad), (0, 0)))))
            self.cat.append(torch.from_numpy(np.pad(cat_data[idx], ((0, pad), (0, 0)))))
            
            # Masks
            self.m_t.append(torch.from_numpy(np.r_[np.ones(T), np.zeros(pad)].astype("float32")))
            self.m_eval.append(torch.from_numpy(np.r_[flag_data[idx], np.zeros(pad)].astype("float32")))
            
            # Targets and horizon mask
            y = np.zeros((L_max, horizon), "float32")
            mh = np.zeros_like(y)
            for t in range(T):
                k = min(horizon, T - t - 1)
                if k:
                    y[t, :k] = epss_data[idx][t + 1:t + 1 + k, 0]
                    mh[t, :k] = 1
            
            self.Y.append(torch.from_numpy(y))
            self.m_h.append(torch.from_numpy(mh))
    
    def __len__(self):
        return len(self.num)
    
    def __getitem__(self, i):
        return (self.num[i], self.boo[i], self.cat[i], 
                self.Y[i], self.m_t[i], self.m_h[i], self.m_eval[i])

class TestSeq2SeqLSTM(nn.Module):
    """Simplified model for testing."""
    def __init__(self, n_num: int, n_bool: int, cat_sizes: List[int], 
                 horizon: int = 5, hidden: int = 8, emb_dim: int = 4):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(s, emb_dim) for s in cat_sizes])
        in_dim = n_num + n_bool + emb_dim * len(cat_sizes)
        
        self.lstm = nn.LSTM(in_dim, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
    
    def forward(self, num, boo, cat):
        # Embeddings
        e = torch.cat([emb(cat[..., i]) for i, emb in enumerate(self.emb)], dim=-1)
        
        # Concatenation
        x = torch.cat([num, boo.float(), e], dim=-1)
        
        # LSTM
        h, _ = self.lstm(x)
        
        # Prediction head
        return self.head(h)

# ═══════════════════════════════════════════════════════════════════════════════
# SHAPE TESTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def test_data_preprocessing_shapes(df: pd.DataFrame) -> pd.DataFrame:
    """Test Phase 1: Data preprocessing and shape transformations."""
    print("\n" + "="*60)
    print("🧪 TESTING PHASE 1: DATA PREPROCESSING SHAPES")
    print("="*60)
    
    original_shape = df.shape
    print(f"📊 Initial data shape: {original_shape}")
    
    # Test column categorization
    cols = get_column_categorization(df)
    print(f"✅ Column categorization:")
    for cat, col_list in cols.items():
        print(f"   {cat}: {len(col_list)} columns - {col_list}")
    
    # Test timestamp delta conversion
    print(f"\n🕐 Testing timestamp delta conversion...")
    df_test = df.copy()
    
    # Safe timestamps (should never be negative)
    for col in cols['ts_safe']:
        delta = (df_test['date'] - df_test[col]).dt.days
        df_test[f"{col}_delta"] = delta.astype("float32")
        print(f"   {col}_delta: min={delta.min()}, max={delta.max()}")
        assert delta.min() >= 0, f"Negative delta in {col}!"
    
    # Leaky timestamps (may have negatives that need clipping)
    for col in cols['ts_leaky']:
        delta = (df_test['date'] - df_test[col]).dt.days.astype("float32")
        negative_count = (delta < 0).sum()
        delta[delta < 0] = np.nan  # Clip negatives
        df_test[f"{col}_delta"] = delta
        print(f"   {col}_delta: {negative_count} negatives clipped to NaN")
    
    # Drop original timestamp columns
    df_test.drop(columns=cols['ts_safe'] + cols['ts_leaky'], inplace=True)
    
    # Test calendar split
    print(f"\n📅 Testing calendar split...")
    days = np.sort(df_test["date"].unique())
    VAL_CUT = pd.to_datetime(days[int(0.6 * len(days))])
    TEST_CUT = pd.to_datetime(days[int(0.8 * len(days))])
    
    df_test["flag_train"] = (df_test["date"] < VAL_CUT).astype("uint8")
    df_test["flag_val"] = ((df_test["date"] >= VAL_CUT) & (df_test["date"] < TEST_CUT)).astype("uint8")
    df_test["flag_test"] = (df_test["date"] >= TEST_CUT).astype("uint8")
    
    train_count = df_test["flag_train"].sum()
    val_count = df_test["flag_val"].sum()
    test_count = df_test["flag_test"].sum()
    print(f"   Train: {train_count} rows, Val: {val_count} rows, Test: {test_count} rows")
    
    # Test EPSS transform
    print(f"\n📐 Testing EPSS transform...")
    original_epss = df_test["epss"].copy()
    df_test["epss"] = transform_epss(df_test["epss"].values, mode="logit")
    print(f"   Original EPSS range: [{original_epss.min():.6f}, {original_epss.max():.6f}]")
    print(f"   Logit EPSS range: [{df_test['epss'].min():.3f}, {df_test['epss'].max():.3f}]")
    
    final_shape = df_test.shape
    print(f"\n📊 Final preprocessed shape: {final_shape}")
    print(f"   Added columns: {final_shape[1] - original_shape[1]}")
    
    return df_test

def test_feature_engineering_shapes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Test Phase 2: Feature engineering and categorization."""
    print("\n" + "="*60)
    print("🧪 TESTING PHASE 2: FEATURE ENGINEERING SHAPES")
    print("="*60)
    
    df_test = df.copy()
    cols = get_column_categorization(df)
    
    # Test boolean processing
    print(f"🔘 Testing boolean features...")
    for col in cols['bool']:
        original_nulls = df_test[col].isna().sum()
        df_test[col] = df_test[col].fillna(0).astype("uint8")
        print(f"   {col}: {original_nulls} nulls filled with 0")
    
    # Test categorical processing with train-only vocab
    print(f"\n📋 Testing categorical features...")
    train_mask = df_test["flag_train"] == 1
    VOCAB = {}
    
    for col in cols['cat']:
        # Build vocab from training data only
        train_cats = df_test.loc[train_mask, col].dropna().unique()
        VOCAB[col] = {"UNK": 0}
        for i, cat in enumerate(sorted(train_cats)):
            VOCAB[col][cat] = i + 1
        
        # Apply mapping
        df_test[col] = df_test[col].map(VOCAB[col]).fillna(0).astype("int16")
        print(f"   {col}: {len(VOCAB[col])} unique values (including UNK)")
    
    # Test numeric processing
    print(f"\n🔢 Testing numeric features...")
    NUM_COLS = [c for c, t in df_test.dtypes.items()
                if np.issubdtype(t, np.number)
                and c not in cols['bool'] + ["flag_train", "flag_val", "flag_test"]]
    
    print(f"   Identified {len(NUM_COLS)} numeric columns: {NUM_COLS}")
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(df_test.loc[train_mask, NUM_COLS])
    df_test[NUM_COLS] = scaler.transform(df_test[NUM_COLS])
    
    # Handle missing values
    SENTINEL = -10.0
    for col in NUM_COLS:
        miss = df_test[col].isna()
        df_test[f"{col}_missing"] = miss.astype("uint8")
        df_test.loc[miss, col] = SENTINEL
        print(f"   {col}: {miss.sum()} missing values → sentinel + flag")
    
    # Final column counts
    missing_cols = [c for c in df_test.columns if c.endswith('_missing')]
    
    feature_info = {
        'num_cols': NUM_COLS,
        'bool_cols': cols['bool'], 
        'cat_cols': cols['cat'],
        'missing_cols': missing_cols,
        'vocab': VOCAB
    }
    
    print(f"\n📊 Final feature counts:")
    print(f"   Numeric: {len(NUM_COLS)} columns")
    print(f"   Boolean: {len(cols['bool'])} columns") 
    print(f"   Categorical: {len(cols['cat'])} columns")
    print(f"   Missing flags: {len(missing_cols)} columns")
    print(f"   Total dataset: {df_test.shape}")
    
    return df_test, feature_info

def test_dataset_creation_shapes(df: pd.DataFrame, feature_info: Dict) -> Tuple:
    """Test Phase 3: Dataset creation and sequence padding."""
    print("\n" + "="*60)
    print("🧪 TESTING PHASE 3: DATASET CREATION SHAPES")
    print("="*60)
    
    # Test L_max calculation
    L_max = df.groupby("cve", observed=True).size().max()
    cve_lengths = df.groupby("cve", observed=True).size()
    print(f"📏 Sequence length analysis:")
    print(f"   L_max (max sequence): {L_max}")
    print(f"   CVE lengths: {cve_lengths.to_dict()}")
    
    # Create test datasets for each split
    horizon = 5  # Small for testing
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        flag_col = f"flag_{split}"
        datasets[split] = TestCVEDataset(
            df, L_max, horizon, flag_col,
            feature_info['num_cols'],
            feature_info['bool_cols'], 
            feature_info['cat_cols']
        )
        print(f"✅ {split} dataset: {len(datasets[split])} CVEs")
    
    # Test individual dataset item shapes
    print(f"\n🔍 Testing individual dataset item shapes:")
    train_item = datasets['train'][0]  # First CVE
    
    expected_shapes = {
        'num': (L_max, len(feature_info['num_cols'])),
        'boo': (L_max, len(feature_info['bool_cols'])),
        'cat': (L_max, len(feature_info['cat_cols'])),
        'Y': (L_max, horizon),
        'm_t': (L_max,),
        'm_h': (L_max, horizon),
        'm_eval': (L_max,)
    }
    
    actual_shapes = {
        'num': train_item[0].shape,
        'boo': train_item[1].shape,
        'cat': train_item[2].shape,
        'Y': train_item[3].shape,
        'm_t': train_item[4].shape,
        'm_h': train_item[5].shape,
        'm_eval': train_item[6].shape
    }
    
    print(f"   Expected vs Actual shapes:")
    for name in expected_shapes:
        expected = expected_shapes[name]
        actual = actual_shapes[name]
        status = "✅" if expected == actual else "❌"
        print(f"   {name:8}: expected {expected}, actual {actual} {status}")
        assert expected == actual, f"Shape mismatch for {name}!"
    
    # Test padding behavior
    print(f"\n🧩 Testing padding behavior:")
    for i, (split, dataset) in enumerate(datasets.items()):
        item = dataset[0]  # First CVE in each split
        m_t = item[4]  # Temporal mask
        real_length = int(m_t.sum().item())
        pad_length = len(m_t) - real_length
        print(f"   {split}: {real_length} real + {pad_length} padding = {len(m_t)} total")
    
    return datasets, L_max, horizon

def test_model_forward_shapes(datasets: Dict, feature_info: Dict, L_max: int, horizon: int):
    """Test Phase 4: Model forward pass shapes."""
    print("\n" + "="*60)
    print("🧪 TESTING PHASE 4: MODEL FORWARD PASS SHAPES")
    print("="*60)
    
    # Create test model with small dimensions
    cat_sizes = [len(feature_info['vocab'][col]) for col in feature_info['cat_cols']]
    
    model = TestSeq2SeqLSTM(
        n_num=len(feature_info['num_cols']),
        n_bool=len(feature_info['bool_cols']),
        cat_sizes=cat_sizes,
        horizon=horizon,
        hidden=8,  # Small for testing
        emb_dim=4  # Small for testing
    )
    
    print(f"🏗️  Model architecture:")
    print(f"   Input dims: {len(feature_info['num_cols'])} num + {len(feature_info['bool_cols'])} bool + {len(cat_sizes)} cat")
    print(f"   Cat sizes: {cat_sizes}")
    print(f"   Expected total input dim: {len(feature_info['num_cols']) + len(feature_info['bool_cols']) + 4*len(cat_sizes)}")
    
    # Test with single item (batch size 1)
    print(f"\n🔍 Testing single item forward pass:")
    item = datasets['train'][0]
    
    # Add batch dimension
    num = item[0].unsqueeze(0)  # [1, L_max, n_num]
    boo = item[1].unsqueeze(0)  # [1, L_max, n_bool]
    cat = item[2].unsqueeze(0)  # [1, L_max, n_cat]
    
    print(f"   Input shapes:")
    print(f"     num: {num.shape}")
    print(f"     boo: {boo.shape}")
    print(f"     cat: {cat.shape}")
    
    # Forward pass
    with torch.no_grad():
        pred = model(num, boo, cat)
    
    expected_pred_shape = (1, L_max, horizon)
    print(f"   Output shape: expected {expected_pred_shape}, actual {pred.shape}")
    assert pred.shape == expected_pred_shape, f"Prediction shape mismatch!"
    print(f"   ✅ Forward pass successful!")
    
    # Test with batch
    print(f"\n🔍 Testing batch forward pass:")
    batch_size = 2
    
    # Create batch by stacking first two items
    items = [datasets['train'][i] for i in range(min(batch_size, len(datasets['train'])))]
    
    num_batch = torch.stack([item[0] for item in items], 0)
    boo_batch = torch.stack([item[1] for item in items], 0)
    cat_batch = torch.stack([item[2] for item in items], 0)
    
    print(f"   Batch input shapes:")
    print(f"     num: {num_batch.shape}")
    print(f"     boo: {boo_batch.shape}")
    print(f"     cat: {cat_batch.shape}")
    
    with torch.no_grad():
        pred_batch = model(num_batch, boo_batch, cat_batch)
    
    expected_batch_shape = (batch_size, L_max, horizon)
    print(f"   Batch output shape: expected {expected_batch_shape}, actual {pred_batch.shape}")
    assert pred_batch.shape == expected_batch_shape, f"Batch prediction shape mismatch!"
    print(f"   ✅ Batch forward pass successful!")
    
    return model

def test_masking_behavior(datasets: Dict, L_max: int, horizon: int):
    """Test Phase 5: Masking behavior and loss computation."""
    print("\n" + "="*60)
    print("🧪 TESTING PHASE 5: MASKING BEHAVIOR")
    print("="*60)
    
    # Test masking logic with known data
    print(f"🎭 Testing mask behavior per CVE:")
    
    for split, dataset in datasets.items():
        print(f"\n   {split.upper()} dataset:")
        
        for cve_idx in range(len(dataset)):
            item = dataset[cve_idx]
            m_t = item[4]      # Temporal mask
            m_h = item[5]      # Horizon mask  
            m_eval = item[6]   # Evaluation mask
            
            real_timesteps = int(m_t.sum().item())
            eval_timesteps = int(m_eval.sum().item())
            total_predictions = int(m_h.sum().item())
            
            print(f"     CVE {cve_idx}: {real_timesteps} real steps, {eval_timesteps} eval steps, {total_predictions} total predictions")
            
            # Verify mask properties
            assert (m_t >= 0).all() and (m_t <= 1).all(), "Temporal mask not binary!"
            assert (m_eval >= 0).all() and (m_eval <= 1).all(), "Eval mask not binary!"
            assert (m_h >= 0).all() and (m_h <= 1).all(), "Horizon mask not binary!"
            
            # Eval mask should be subset of temporal mask
            assert (m_eval <= m_t).all(), "Eval mask extends beyond temporal mask!"
    
    # Test combined masking for loss
    print(f"\n🎯 Testing combined masking for loss:")
    
    def test_masked_loss(pred, true, m_t, m_h, m_eval):
        """Test version of masked loss."""
        mask = m_t * m_eval  # [B, L]
        err = (pred - true) ** 2  # [B, L, H]
        combined_mask = mask.unsqueeze(-1) * m_h  # [B, L, H]
        total_loss = (err * combined_mask).sum()
        total_valid = combined_mask.sum()
        return total_loss / total_valid if total_valid > 0 else torch.tensor(0.0)
    
    # Create dummy predictions and test loss
    batch_size = 2
    items = [datasets['train'][i] for i in range(min(batch_size, len(datasets['train'])))]
    
    pred = torch.randn(batch_size, L_max, horizon)
    true = torch.stack([item[3] for item in items], 0)  # Y values
    m_t = torch.stack([item[4] for item in items], 0)
    m_h = torch.stack([item[5] for item in items], 0)
    m_eval = torch.stack([item[6] for item in items], 0)
    
    loss = test_masked_loss(pred, true, m_t, m_h, m_eval)
    print(f"   Test loss computation: {loss.item():.6f}")
    assert torch.isfinite(loss), "Loss is not finite!"
    print(f"   ✅ Masked loss computation successful!")

def run_comprehensive_shape_tests():
    """Run all shape tests in sequence."""
    print("🚀 STARTING COMPREHENSIVE LSTM PIPELINE SHAPE TESTS")
    print("="*80)
    
    try:
        # Phase 1: Create synthetic data
        df_raw = create_synthetic_dataset()
        
        # Phase 2: Test preprocessing
        df_processed = test_data_preprocessing_shapes(df_raw)
        
        # Phase 3: Test feature engineering
        df_features, feature_info = test_feature_engineering_shapes(df_processed)
        
        # Phase 4: Test dataset creation
        datasets, L_max, horizon = test_dataset_creation_shapes(df_features, feature_info)
        
        # Phase 5: Test model forward pass
        model = test_model_forward_shapes(datasets, feature_info, L_max, horizon)
        
        # Phase 6: Test masking behavior
        test_masking_behavior(datasets, L_max, horizon)
        
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! SHAPE LINEAGE VALIDATED!")
        print("="*80)
        print(f"✅ Data preprocessing: shapes validated")
        print(f"✅ Feature engineering: {len(feature_info['num_cols'])} num + {len(feature_info['bool_cols'])} bool + {len(feature_info['cat_cols'])} cat features")
        print(f"✅ Dataset creation: {len(datasets['train'])} CVEs, L_max={L_max}, horizon={horizon}")
        print(f"✅ Model forward pass: input dim → LSTM → {horizon} predictions")
        print(f"✅ Masking behavior: temporal, evaluation, and horizon masks working correctly")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_shape_tests()
    if not success:
        exit(1) 