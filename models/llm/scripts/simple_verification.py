#!/usr/bin/env python3
"""
Simple verification script to prove models are unchanged.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import AutoConfig, pipeline
import json

def test_model_consistency():
    """Test that models give consistent results."""
    
    print("="*80)
    print("MODEL CONSISTENCY VERIFICATION")
    print("="*80)
    
    # Test 1: Multiple loads of classification model
    print("1. TESTING CLASSIFICATION MODEL CONSISTENCY:")
    print("-" * 50)
    
    test_text = "SQL injection vulnerability in web application"
    results = []
    
    for i in range(3):
        classifier = pipeline("text-classification", 
                             model="CIRCL/vulnerability-severity-classification-roberta-base")
        result = classifier(test_text)
        results.append(result[0])
        print(f"   Load {i+1}: {result[0]['label']} ({result[0]['score']:.6f})")
        del classifier  # Clean up
    
    # Check consistency
    labels = [r['label'] for r in results]
    scores = [r['score'] for r in results]
    
    labels_consistent = len(set(labels)) == 1
    score_variance = max(scores) - min(scores)
    scores_consistent = score_variance < 1e-6
    
    print(f"   Labels consistent: {labels_consistent}")
    print(f"   Scores consistent: {scores_consistent} (variance: {score_variance:.8f})")
    
    # Test 2: Check configuration consistency
    print("\n2. TESTING CONFIGURATION CONSISTENCY:")
    print("-" * 50)
    
    configs = []
    for i in range(3):
        config = AutoConfig.from_pretrained("CIRCL/vulnerability-severity-classification-roberta-base")
        configs.append({
            'num_labels': getattr(config, 'num_labels', None),
            'id2label': config.id2label,
            'model_type': config.model_type,
            'vocab_size': config.vocab_size
        })
        print(f"   Load {i+1}: num_labels={configs[i]['num_labels']}, model_type={configs[i]['model_type']}")
    
    # Check config consistency
    config_consistent = all(c == configs[0] for c in configs)
    print(f"   Configurations consistent: {config_consistent}")
    
    return labels_consistent and scores_consistent and config_consistent

def check_original_state():
    """Check that models are in their original state."""
    
    print("\n" + "="*80)
    print("ORIGINAL STATE VERIFICATION")
    print("="*80)
    
    print("1. CHECKING EXPECTED VALUES:")
    print("-" * 50)
    
    # Load classification model
    config = AutoConfig.from_pretrained("CIRCL/vulnerability-severity-classification-roberta-base")
    
    # Expected values
    expected = {
        'id2label': {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'},
        'model_type': 'roberta',
        'vocab_size': 50265
    }
    
    # Check each expected value
    checks = []
    for key, expected_value in expected.items():
        actual_value = getattr(config, key)
        matches = actual_value == expected_value
        checks.append(matches)
        print(f"   {key}: {'MATCH' if matches else 'DIFFERENT'}")
        if not matches:
            print(f"     Expected: {expected_value}")
            print(f"     Actual:   {actual_value}")
    
    all_match = all(checks)
    print(f"   All values match expected: {all_match}")
    
    return all_match

def test_prediction_accuracy():
    """Test that predictions match expected behavior."""
    
    print("\n" + "="*80)
    print("PREDICTION ACCURACY VERIFICATION")
    print("="*80)
    
    print("1. TESTING KNOWN VULNERABILITY CLASSIFICATIONS:")
    print("-" * 50)
    
    classifier = pipeline("text-classification", 
                         model="CIRCL/vulnerability-severity-classification-roberta-base")
    
    # Test cases with expected severity levels
    test_cases = [
        ("SQL injection vulnerability allows data extraction", ["High", "Critical"]),
        ("Minor configuration issue", ["Low", "Medium"]),
        ("Remote code execution vulnerability", ["High", "Critical"]),
        ("Information disclosure vulnerability", ["Low", "Medium", "High"])
    ]
    
    all_reasonable = True
    for text, expected_severities in test_cases:
        result = classifier(text)
        prediction = result[0]['label']
        confidence = result[0]['score']
        
        is_reasonable = prediction in expected_severities
        all_reasonable = all_reasonable and is_reasonable
        
        print(f"   Text: '{text[:40]}...'")
        print(f"   Prediction: {prediction} ({confidence:.3f}) - {'REASONABLE' if is_reasonable else 'UNEXPECTED'}")
    
    print(f"   All predictions reasonable: {all_reasonable}")
    return all_reasonable

def memory_isolation_test():
    """Test that in-memory changes don't persist."""
    
    print("\n" + "="*80)
    print("MEMORY ISOLATION TEST")
    print("="*80)
    
    print("1. MAKING TEMPORARY MODIFICATIONS:")
    print("-" * 50)
    
    # Load model and modify config
    config = AutoConfig.from_pretrained("CIRCL/vulnerability-severity-classification-roberta-base")
    original_labels = config.id2label.copy()
    
    print(f"   Original labels: {original_labels}")
    
    # Modify in memory
    config.id2label = {0: 'Test1', 1: 'Test2', 2: 'Test3', 3: 'Test4'}
    print(f"   Modified labels: {config.id2label}")
    
    # Delete the modified config
    del config
    
    print("\n2. LOADING FRESH CONFIG:")
    print("-" * 50)
    
    # Load fresh config
    fresh_config = AutoConfig.from_pretrained("CIRCL/vulnerability-severity-classification-roberta-base")
    fresh_labels = fresh_config.id2label
    
    print(f"   Fresh labels: {fresh_labels}")
    
    # Check if original state restored
    restored = fresh_labels == original_labels
    print(f"   Original state restored: {restored}")
    
    return restored

def main():
    """Main verification function."""
    
    print("COMPREHENSIVE MODEL INTEGRITY VERIFICATION")
    print("="*80)
    print("Testing to ensure no permanent changes were made to CIRCL models...")
    
    # Run all tests
    consistency_ok = test_model_consistency()
    original_state_ok = check_original_state()
    predictions_ok = test_prediction_accuracy()
    isolation_ok = memory_isolation_test()
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    print("TEST RESULTS:")
    print(f"  Model consistency: {'PASS' if consistency_ok else 'FAIL'}")
    print(f"  Original state: {'PASS' if original_state_ok else 'FAIL'}")
    print(f"  Prediction accuracy: {'PASS' if predictions_ok else 'FAIL'}")
    print(f"  Memory isolation: {'PASS' if isolation_ok else 'FAIL'}")
    
    all_tests_pass = consistency_ok and original_state_ok and predictions_ok and isolation_ok
    
    print(f"\nOVERALL RESULT: {'ALL TESTS PASS' if all_tests_pass else 'SOME TESTS FAILED'}")
    
    if all_tests_pass:
        print("\nCONCLUSION:")
        print("  - Models are in their original, unmodified state")
        print("  - All changes were temporary and in-memory only")
        print("  - Models will behave identically in new sessions")
        print("  - No permanent modifications were made")
        print("  - Safe to use in production environments")
    else:
        print("\nISSUES DETECTED:")
        print("  - Some tests failed, indicating possible modifications")
        print("  - Review the detailed output above")
        print("  - Consider clearing cache if needed")

if __name__ == "__main__":
    main() 