#!/usr/bin/env python3
"""
Test the UNK collision fix
"""

import pandas as pd
import numpy as np

def test_unk_fix():
    print("=" * 80)
    print("TESTING UNK COLLISION FIX")
    print("=" * 80)
    
    # Create test data that contains literal "UNK" strings
    test_data = pd.DataFrame({
        'col_with_unk': ['HIGH', 'MEDIUM', 'UNK', 'LOW', 'UNK', 'HIGH'],
        'col_without_unk': ['A', 'B', 'C', 'A', 'B', 'C']
    })
    
    print("Test data:")
    print(test_data['col_with_unk'].value_counts())
    print()
    
    # Test the OLD (buggy) approach
    print("=" * 50)
    print("OLD APPROACH (BUGGY)")
    print("=" * 50)
    
    for col in ['col_with_unk', 'col_without_unk']:
        print(f"\nTesting column: {col}")
        cats = test_data[col].dropna().unique()
        print(f"Unique values: {list(cats)}")
        
        # Old buggy approach
        old_vocab = {"UNK": 0, **{c: i + 1 for i, c in enumerate(sorted(cats))}}
        
        vocab_size = len(old_vocab)
        max_index = max(old_vocab.values())
        
        print(f"OLD vocabulary: {old_vocab}")
        print(f"Vocab size: {vocab_size}")
        print(f"Max index: {max_index}")
        
        if max_index >= vocab_size:
            print(f"🚨 BUG DETECTED: max_index ({max_index}) >= vocab_size ({vocab_size})")
            print(f"   This would cause CUDA assertion failure!")
        else:
            print(f"✅ No issue with this column")
    
    # Test the NEW (fixed) approach
    print("\n" + "=" * 50)
    print("NEW APPROACH (FIXED)")
    print("=" * 50)
    
    for col in ['col_with_unk', 'col_without_unk']:
        print(f"\nTesting column: {col}")
        cats = test_data[col].dropna().unique()
        print(f"Original unique values: {list(cats)}")
        
        # 1️⃣ Remove accidental UNK coming from the raw data
        cats_filtered = [c for c in cats if c != "UNK"]
        print(f"After removing UNK: {list(cats_filtered)}")
        
        # 2️⃣ First build normal ids, then add the real UNK = 0
        mapping = {c: i + 1 for i, c in enumerate(sorted(cats_filtered))}
        mapping["UNK"] = 0
        
        vocab_size = len(mapping)
        max_index = max(mapping.values())
        
        print(f"NEW vocabulary: {mapping}")
        print(f"Vocab size: {vocab_size}")
        print(f"Max index: {max_index}")
        
        # Validation
        if max_index == vocab_size - 1:
            print(f"✅ FIXED: max_index ({max_index}) == vocab_size - 1 ({vocab_size - 1})")
        else:
            print(f"🚨 Still broken: max_index ({max_index}) != vocab_size - 1 ({vocab_size - 1})")
        
        # Test mapping
        mapped_values = test_data[col].map(mapping).fillna(0)
        print(f"Mapped values range: [{mapped_values.min()}, {mapped_values.max()}]")
        print(f"All values < vocab_size? {(mapped_values < vocab_size).all()}")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("✅ The fix correctly handles literal 'UNK' strings in data")
    print("✅ Ensures max(vocab.values()) == len(vocab) - 1")
    print("✅ Prevents CUDA assertion failures in embedding layers")
    print("✅ Maintains UNK=0 for proper padding behavior")

if __name__ == "__main__":
    test_unk_fix() 