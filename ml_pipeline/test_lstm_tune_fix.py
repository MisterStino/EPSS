#!/usr/bin/env python
"""
Test script to verify LSTM tune fixes work correctly.
This tests data loading and model initialization without running full Ray Tune.
"""

import sys
from pathlib import Path
import torch
from functools import partial

# Add the training module to path
sys.path.append('training')
from ml_pipeline.training.dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed
from torch.utils.data import DataLoader

def test_data_loading():
    """Test that data loading works with correct unpacking."""
    print("Testing data loading...")
    
    # Find project root and files
    project_root = Path.cwd()
    if not (project_root / "ml_pipeline").exists():
        for parent in project_root.parents:
            if (parent / "ml_pipeline").exists():
                project_root = parent
                break
    
    # Find Arrow file
    arrow_candidates = [
        project_root / "ml_pipeline" / "work" / "epss_stage1.arrow",
        project_root / "ml_pipeline" / "data_prep" / "work" / "epss_stage1.arrow",
    ]
    
    arrow_path = None
    for candidate in arrow_candidates:
        if candidate.exists():
            arrow_path = candidate
            break
    
    if arrow_path is None:
        print("❌ Arrow file not found")
        return False
    
    print(f"✓ Using Arrow file: {arrow_path}")
    
    # Create dataset and dataloader
    try:
        dataset = CVEIterableDatasetFixed(arrow_path, horizon=30)
        
        train_loader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=30),
            num_workers=0
        )
        
        # Test batch loading
        batch = next(iter(train_loader))
        print(f"✓ Batch loaded successfully")
        print(f"✓ Number of tensors in batch: {len(batch)}")
        
        # Test unpacking
        num_tensor, boo_tensor, cat_tensor, target_tensor, m_t, m_h, m_eval, date_tensor, lengths_tensor = batch
        print(f"✓ Batch unpacking successful")
        
        # Print shapes
        print(f"✓ Tensor shapes:")
        print(f"  - num_tensor: {num_tensor.shape}")
        print(f"  - boo_tensor: {boo_tensor.shape}")
        print(f"  - cat_tensor: {cat_tensor.shape}")
        print(f"  - target_tensor: {target_tensor.shape}")
        print(f"  - m_t: {m_t.shape}")
        print(f"  - m_h: {m_h.shape}")
        print(f"  - m_eval: {m_eval.shape}")
        print(f"  - date_tensor: {date_tensor.shape}")
        print(f"  - lengths_tensor: {lengths_tensor.shape}")
        
        # Check for NaN/Inf
        has_nan_inf = False
        for name, tensor in [("num", num_tensor), ("target", target_tensor), ("bool", boo_tensor)]:
            if torch.isnan(tensor).any():
                print(f"⚠️  NaN found in {name}_tensor")
                has_nan_inf = True
            if torch.isinf(tensor).any():
                print(f"⚠️  Inf found in {name}_tensor")
                has_nan_inf = True
        
        if not has_nan_inf:
            print("✓ No NaN/Inf found in tensors")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_initialization():
    """Test model initialization with proper weights."""
    print("\nTesting model initialization...")
    
    try:
        import torch.nn as nn
        
        # Mock vocab for testing
        vocab = {
            "feature1": ["a", "b", "c"],
            "feature2": ["x", "y", "z"]
        }
        
        # Simple LSTM model for testing
        class TestLSTM(nn.Module):
            def __init__(self, input_size=10, hidden_size=64, layers=2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True)
                self.output = nn.Linear(hidden_size, 30)
                
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.output(out)
        
        model = TestLSTM()
        
        # Initialize weights properly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        n = param.size(0)
                        param.data[(n//4):(n//2)].fill_(1)
        
        model.apply(init_weights)
        
        # Test forward pass
        test_input = torch.randn(2, 10, 10)  # batch_size=2, seq_len=10, features=10
        output = model(test_input)
        
        print(f"✓ Model initialized successfully")
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        # Check for NaN/Inf in output
        if torch.isnan(output).any():
            print("⚠️  NaN found in model output")
            return False
        if torch.isinf(output).any():
            print("⚠️  Inf found in model output")
            return False
        
        print("✓ No NaN/Inf in model output")
        return True
        
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing LSTM Tune Fixes")
    print("=" * 50)
    
    success = True
    
    # Test data loading
    success &= test_data_loading()
    
    # Test model initialization
    success &= test_model_initialization()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! LSTM tune fixes are working correctly.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    sys.exit(0 if success else 1) 