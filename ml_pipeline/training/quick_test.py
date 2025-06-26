#!/usr/bin/env python
"""Quick test to validate dataset with real Arrow file"""

from pathlib import Path
from functools import partial
from torch.utils.data import DataLoader
from dataset_iterable_fixed import CVEIterableDatasetFixed, pad_and_mask_fixed

# Test with correct path
ARROW_PATH = Path("../../work/epss_stage1.arrow")
VOCAB_PATH = Path("../../work/vocab.json")

print(f"Testing dataset with path: {ARROW_PATH.resolve()}")
print(f"Arrow file exists: {ARROW_PATH.exists()}")
print(f"Vocab file exists: {VOCAB_PATH.exists()}")

if ARROW_PATH.exists() and VOCAB_PATH.exists():
    print("\n🚀 Creating dataset...")
    ds = CVEIterableDatasetFixed(ARROW_PATH, horizon=30)
    print("✓ Dataset created successfully!")
    
    print("\n🚀 Testing DataLoader...")
    dl = DataLoader(
        ds,
        batch_size=2,
        collate_fn=partial(pad_and_mask_fixed, flag_kind="train", horizon=30),
        num_workers=0,
        pin_memory=False
    )
    
    print("✓ DataLoader created successfully!")
    
    print("\n🚀 Loading first batch...")
    batch = next(iter(dl))
    
    print(f"✓ Batch loaded! Shapes: {[t.shape for t in batch]}")
    
    # Test date conversion
    dates = batch[-1]  # date_pad
    print(f"✓ Date sample (nanoseconds): {dates[0, :3]}")
    print(f"✓ Date sample (datetime): {dates[0, :3].numpy().view('datetime64[ns]')}")
    
    print("\n🎉 ALL TESTS PASSED - DATASET IS WORKING WITH REAL DATA!")
    
else:
    print("❌ Required files not found!")
    print(f"Arrow: {ARROW_PATH.exists()} at {ARROW_PATH.resolve()}")
    print(f"Vocab: {VOCAB_PATH.exists()} at {VOCAB_PATH.resolve()}") 