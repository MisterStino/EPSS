#!/usr/bin/env python3
"""
Analyze vocabulary size vs maximum indices mismatch
"""

import json
import numpy as np

def analyze_vocab_mismatch():
    print("=" * 80)
    print("ANALYZING VOCABULARY SIZE VS MAXIMUM INDICES MISMATCH")
    print("=" * 80)
    
    # Load the vocabulary created by the LSTM script
    with open("vocab.json", "r") as f:
        VOCAB = json.load(f)
    
    print("Vocabulary Analysis:")
    print("=" * 50)
    
    for col, vocab in VOCAB.items():
        print(f"\nColumn: {col}")
        print(f"  Vocabulary: {vocab}")
        
        # Key metrics
        vocab_size = len(vocab)
        max_index = max(vocab.values())
        min_index = min(vocab.values())
        
        print(f"  Vocabulary size (len): {vocab_size}")
        print(f"  Maximum index: {max_index}")
        print(f"  Minimum index: {min_index}")
        
        # The critical check
        print(f"  Embedding rows created: [0, {vocab_size-1}]")
        print(f"  Data contains indices: [{min_index}, {max_index}]")
        
        # Check for the mismatch
        if max_index >= vocab_size:
            print(f"  🚨 MISMATCH DETECTED!")
            print(f"     Index {max_index} >= vocab_size {vocab_size}")
            print(f"     Embedding has rows [0-{vocab_size-1}] but data needs row {max_index}")
            print(f"     This causes: srcIndex ({max_index}) >= srcSelectDimSize ({vocab_size})")
        else:
            print(f"  ✅ No mismatch: max_index ({max_index}) < vocab_size ({vocab_size})")
        
        # Check for potential issues
        expected_max = vocab_size - 1
        if max_index != expected_max:
            print(f"  ⚠️  Unexpected: max_index should be {expected_max} for optimal embedding size")
        
        # Check for UNK mapping
        unk_index = vocab.get("UNK", "NOT_FOUND")
        print(f"  UNK mapped to: {unk_index}")
        
        # Look for duplicate values (key collision evidence)
        values = list(vocab.values())
        unique_values = set(values)
        if len(values) != len(unique_values):
            print(f"  🚨 DUPLICATE VALUES DETECTED!")
            print(f"     This indicates key collision during vocabulary creation!")
            
            # Find duplicates
            from collections import Counter
            value_counts = Counter(values)
            duplicates = {v: count for v, count in value_counts.items() if count > 1}
            print(f"     Duplicate indices: {duplicates}")
        else:
            print(f"  ✅ No duplicate values in vocabulary")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Overall analysis
    total_issues = 0
    for col, vocab in VOCAB.items():
        vocab_size = len(vocab)
        max_index = max(vocab.values())
        if max_index >= vocab_size:
            total_issues += 1
            print(f"❌ {col}: max_index ({max_index}) >= vocab_size ({vocab_size})")
    
    if total_issues > 0:
        print(f"\n🔥 FOUND {total_issues} VOCABULARY MISMATCHES!")
        print("These will cause CUDA assertion failures during embedding lookup.")
        print("\nThe fix is to ensure: max(vocab.values()) < len(vocab)")
    else:
        print("\n✅ No vocabulary mismatches found in current vocab.json")
        print("The issue might be elsewhere or in a different vocabulary version.")

if __name__ == "__main__":
    analyze_vocab_mismatch() 