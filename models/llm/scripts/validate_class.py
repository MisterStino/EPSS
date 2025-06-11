#!/usr/bin/env python3
"""
Validation script to compare results between:
1. Original classify.py approach (direct model usage)
2. Our new VulnerabilitySeverityClassifier class

This ensures our class implementation is correct and produces identical results.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Import our class
from label import VulnerabilitySeverityClassifier

def original_classify_approach(descriptions):
    """
    Replicate the original classify.py approach exactly.
    This is the reference implementation to compare against.
    """
    print("🔄 Loading model using original approach...")
    
    model_name = "CIRCL/vulnerability-severity-classification-roberta-base"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Move to device if not using device_map
        if not torch.cuda.is_available():
            model = model.to("cpu")
        
        # Severity labels (same as original)
        SEVERITY_LABELS = ["low", "medium", "high", "critical"]
        
        # CVSS score mapping (same as original)
        SEVERITY_TO_CVSS = {
            "low": 3.0,      # 0.1-3.9
            "medium": 6.0,   # 4.0-6.9  
            "high": 8.0,     # 7.0-8.9
            "critical": 9.5  # 9.0-10.0
        }
        
        print(f"✅ Original model loaded on {model.device}")
        
        # Tokenize descriptions (same parameters as original)
        inputs = tokenizer(
            descriptions, 
            truncation=True, 
            padding=True, 
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Get predictions (same as original)
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_indices = torch.argmax(probabilities, dim=-1)
        
        # Convert to severity labels and CVSS scores (same as original)
        predicted_severities = [SEVERITY_LABELS[idx] for idx in predicted_indices.cpu().numpy()]
        predicted_cvss = [SEVERITY_TO_CVSS[severity] for severity in predicted_severities]
        
        return predicted_severities, predicted_cvss, probabilities.cpu().numpy()
        
    except Exception as e:
        print(f"❌ Original approach failed: {e}")
        return None, None, None

def class_based_approach(descriptions):
    """
    Use our new VulnerabilitySeverityClassifier class.
    """
    print("🔄 Loading model using class-based approach...")
    
    # Initialize our class with same parameters as original
    classifier = VulnerabilitySeverityClassifier(
        device="auto",  # Same auto-detection as original
        max_length=512  # Same max_length as original
    )
    
    if not classifier.is_ready():
        print("❌ Class-based approach failed: Model not ready")
        return None, None, None
    
    print(f"✅ Class model loaded on {classifier.get_model_info()['device']}")
    
    # Get predictions with probabilities
    predicted_severities, predicted_cvss, probabilities = classifier.predict(
        descriptions, 
        return_probabilities=True
    )
    
    return predicted_severities, predicted_cvss, np.array(probabilities)

def compare_results(original_results, class_results, descriptions):
    """
    Compare results between original and class-based approaches.
    """
    orig_sev, orig_cvss, orig_probs = original_results
    class_sev, class_cvss, class_probs = class_results
    
    if orig_sev is None or class_sev is None:
        print("❌ Cannot compare - one approach failed")
        return False
    
    print(f"\n" + "="*80)
    print("DETAILED COMPARISON RESULTS")
    print("="*80)
    
    # Compare predictions for each description
    all_match = True
    
    print(f"{'Description':<50} {'Orig Sev':<10} {'Class Sev':<10} {'Orig CVSS':<10} {'Class CVSS':<10} {'Match':<8}")
    print("-" * 100)
    
    for i, desc in enumerate(descriptions):
        desc_short = desc[:47] + "..." if len(desc) > 50 else desc
        orig_s = orig_sev[i]
        class_s = class_sev[i]
        orig_c = orig_cvss[i]
        class_c = class_cvss[i]
        
        severity_match = orig_s == class_s
        cvss_match = abs(orig_c - class_c) < 0.001  # Allow tiny floating point differences
        row_match = severity_match and cvss_match
        all_match = all_match and row_match
        
        match_symbol = "✅" if row_match else "❌"
        
        print(f"{desc_short:<50} {orig_s:<10} {class_s:<10} {orig_c:<10.1f} {class_c:<10.1f} {match_symbol:<8}")
    
    # Compare probabilities
    print(f"\n" + "-"*50)
    print("PROBABILITY COMPARISON")
    print("-"*50)
    
    prob_differences = np.abs(orig_probs - class_probs)
    max_prob_diff = np.max(prob_differences)
    mean_prob_diff = np.mean(prob_differences)
    
    print(f"Maximum probability difference: {max_prob_diff:.6f}")
    print(f"Mean probability difference: {mean_prob_diff:.6f}")
    
    # Probabilities should be nearly identical (allowing for tiny numerical differences)
    probs_match = max_prob_diff < 1e-5
    
    print(f"Probabilities match: {'✅' if probs_match else '❌'}")
    
    # Overall result
    print(f"\n" + "="*50)
    print("OVERALL VALIDATION RESULT")
    print("="*50)
    
    overall_success = all_match and probs_match
    
    if overall_success:
        print("🎉 SUCCESS: Class implementation produces IDENTICAL results to original!")
        print("✅ Severities match perfectly")
        print("✅ CVSS scores match perfectly") 
        print("✅ Probabilities match within numerical precision")
    else:
        print("❌ FAILURE: Results differ between implementations")
        if not all_match:
            print("❌ Severity/CVSS predictions don't match")
        if not probs_match:
            print("❌ Probabilities differ significantly")
    
    return overall_success

def main():
    """Main validation function."""
    
    print("="*80)
    print("VALIDATING VulnerabilitySeverityClassifier CLASS")
    print("="*80)
    print("Comparing results with original classify.py approach...")
    
    # Test descriptions (same variety as original script)
    test_descriptions = [
        "Buffer overflow vulnerability in network parsing component allows remote code execution",
        "SQL injection vulnerability in user input validation allows database manipulation", 
        "Cross-site scripting vulnerability enables malicious script execution in browsers",
        "Authentication bypass vulnerability allows unauthorized administrative access",
        "Information disclosure vulnerability exposes sensitive configuration data",
        "Denial of service vulnerability causes application crash through malformed input",
        "Privilege escalation vulnerability allows local users to gain root access",
        "Directory traversal vulnerability enables access to arbitrary files",
        "Command injection vulnerability allows execution of arbitrary system commands",
        "Cryptographic weakness in random number generation reduces security"
    ]
    
    print(f"\nTesting with {len(test_descriptions)} vulnerability descriptions...")
    
    # Run original approach
    print(f"\n1. Running original classify.py approach...")
    original_results = original_classify_approach(test_descriptions)
    
    # Run class-based approach  
    print(f"\n2. Running class-based approach...")
    class_results = class_based_approach(test_descriptions)
    
    # Compare results
    print(f"\n3. Comparing results...")
    success = compare_results(original_results, class_results, test_descriptions)
    
    # Additional validation with edge cases
    print(f"\n4. Testing edge cases...")
    edge_cases = [
        "",  # Empty string
        "Short vuln",  # Very short description
        "A" * 1000,  # Very long description
        "Normal vulnerability description with standard length and content"
    ]
    
    # Filter out empty string for original approach (it might handle differently)
    edge_cases_filtered = [desc for desc in edge_cases if desc.strip()]
    
    print(f"Testing {len(edge_cases_filtered)} edge cases...")
    
    try:
        orig_edge = original_classify_approach(edge_cases_filtered)
        class_edge = class_based_approach(edge_cases_filtered)
        edge_success = compare_results(orig_edge, class_edge, edge_cases_filtered)
        
        if edge_success:
            print("✅ Edge cases also pass validation!")
        else:
            print("⚠️ Edge cases show differences")
            
    except Exception as e:
        print(f"⚠️ Edge case testing failed: {e}")
        edge_success = False
    
    # Final result
    print(f"\n" + "="*80)
    print("FINAL VALIDATION RESULT")
    print("="*80)
    
    if success:
        print("🎉 VALIDATION PASSED!")
        print("✅ VulnerabilitySeverityClassifier class is correctly implemented")
        print("✅ Results are identical to original classify.py approach")
        print("✅ Class can be used as a drop-in replacement")
    else:
        print("❌ VALIDATION FAILED!")
        print("❌ Class implementation differs from original")
        print("❌ Investigation needed to fix discrepancies")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 