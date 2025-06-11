#!/usr/bin/env python3
"""
Script to debug why the wrapper approach is causing performance issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from label import VulnerabilitySeverityClassifier
import torch

def debug_prediction_differences():
    """Debug why wrapper and direct predictions differ."""
    
    print("="*80)
    print("DEBUGGING WRAPPER PREDICTION DIFFERENCES")
    print("="*80)
    
    test_text = "SQL injection vulnerability allows unauthorized database access"
    
    print("1. DIRECT MODEL PREDICTION:")
    print("-" * 50)
    
    # Direct approach
    direct_classifier = pipeline("text-classification", 
                                model="CIRCL/vulnerability-severity-classification-roberta-base")
    direct_result = direct_classifier(test_text)
    
    print(f"Direct result: {direct_result}")
    print(f"Direct prediction: {direct_result[0]['label']}")
    print(f"Direct confidence: {direct_result[0]['score']:.6f}")
    
    print("\\n2. WRAPPER MODEL PREDICTION:")
    print("-" * 50)
    
    # Wrapper approach
    wrapper_classifier = VulnerabilitySeverityClassifier()
    if wrapper_classifier.is_ready():
        severity, cvss = wrapper_classifier.predict_single(test_text)
        print(f"Wrapper prediction: {severity}")
        print(f"Wrapper CVSS: {cvss}")
        
        # Get the raw prediction from wrapper's internal pipeline
        if hasattr(wrapper_classifier, 'pipeline'):
            raw_wrapper_result = wrapper_classifier.pipeline(test_text)
            print(f"Wrapper raw result: {raw_wrapper_result}")
    
    print("\\n3. INVESTIGATING THE DIFFERENCE:")
    print("-" * 50)
    
    # Check if it's a case sensitivity issue
    if direct_result[0]['label'].lower() == severity.lower():
        print("✅ ISSUE FOUND: Case sensitivity difference")
        print(f"   Direct: '{direct_result[0]['label']}' (capitalized)")
        print(f"   Wrapper: '{severity}' (lowercase)")
    else:
        print("❌ Different predictions entirely")
        print(f"   Direct: '{direct_result[0]['label']}'")
        print(f"   Wrapper: '{severity}'")

def debug_wrapper_internals():
    """Debug the wrapper's internal workings."""
    
    print("\\n" + "="*80)
    print("DEBUGGING WRAPPER INTERNALS")
    print("="*80)
    
    print("1. EXAMINING WRAPPER CONFIGURATION:")
    print("-" * 50)
    
    wrapper = VulnerabilitySeverityClassifier()
    
    if wrapper.is_ready():
        print(f"Model name: {wrapper.model_name}")
        print(f"Device: {wrapper.device}")
        print(f"Severity labels: {wrapper.severity_labels}")
        print(f"CVSS mapping: {wrapper.cvss_mapping}")
        
        print("\\n2. TESTING WRAPPER METHODS:")
        print("-" * 50)
        
        test_text = "SQL injection vulnerability"
        
        # Test predict_single
        print("Testing predict_single:")
        try:
            severity, cvss = wrapper.predict_single(test_text)
            print(f"   Result: {severity}, CVSS: {cvss}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test predict_batch
        print("\\nTesting predict_batch:")
        try:
            results = wrapper.predict_batch([test_text])
            print(f"   Result: {results}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test direct pipeline access
        print("\\nTesting direct pipeline access:")
        try:
            if hasattr(wrapper, 'pipeline'):
                direct_result = wrapper.pipeline(test_text)
                print(f"   Direct pipeline result: {direct_result}")
        except Exception as e:
            print(f"   Error: {e}")

def test_minimal_wrapper():
    """Test a minimal wrapper to isolate the issue."""
    
    print("\\n" + "="*80)
    print("TESTING MINIMAL WRAPPER")
    print("="*80)
    
    print("1. CREATING MINIMAL WRAPPER:")
    print("-" * 50)
    
    class MinimalWrapper:
        def __init__(self):
            self.pipeline = pipeline("text-classification", 
                                   model="CIRCL/vulnerability-severity-classification-roberta-base")
        
        def predict(self, text):
            result = self.pipeline(text)
            return result[0]['label'], result[0]['score']
    
    minimal = MinimalWrapper()
    
    test_text = "SQL injection vulnerability allows unauthorized database access"
    
    # Test minimal wrapper
    label, score = minimal.predict(test_text)
    print(f"Minimal wrapper: {label} ({score:.6f})")
    
    # Compare with direct
    direct = pipeline("text-classification", 
                     model="CIRCL/vulnerability-severity-classification-roberta-base")
    direct_result = direct(test_text)
    print(f"Direct pipeline: {direct_result[0]['label']} ({direct_result[0]['score']:.6f})")
    
    # Check consistency
    consistent = label == direct_result[0]['label'] and abs(score - direct_result[0]['score']) < 1e-6
    print(f"Consistent: {'✅' if consistent else '❌'}")

def investigate_model_loading():
    """Investigate if model loading is causing issues."""
    
    print("\\n" + "="*80)
    print("INVESTIGATING MODEL LOADING")
    print("="*80)
    
    print("1. COMPARING MODEL LOADING METHODS:")
    print("-" * 50)
    
    test_text = "SQL injection vulnerability"
    
    # Method 1: Pipeline
    print("Method 1: Pipeline loading")
    pipe1 = pipeline("text-classification", 
                    model="CIRCL/vulnerability-severity-classification-roberta-base")
    result1 = pipe1(test_text)
    print(f"   Result: {result1[0]['label']} ({result1[0]['score']:.6f})")
    
    # Method 2: Manual loading
    print("\\nMethod 2: Manual model loading")
    tokenizer = AutoTokenizer.from_pretrained("CIRCL/vulnerability-severity-classification-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("CIRCL/vulnerability-severity-classification-roberta-base")
    pipe2 = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result2 = pipe2(test_text)
    print(f"   Result: {result2[0]['label']} ({result2[0]['score']:.6f})")
    
    # Method 3: With device specification
    print("\\nMethod 3: With device specification")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe3 = pipeline("text-classification", 
                    model="CIRCL/vulnerability-severity-classification-roberta-base",
                    device=device)
    result3 = pipe3(test_text)
    print(f"   Result: {result3[0]['label']} ({result3[0]['score']:.6f})")
    
    # Check consistency
    results = [result1[0], result2[0], result3[0]]
    labels = [r['label'] for r in results]
    scores = [r['score'] for r in results]
    
    labels_consistent = len(set(labels)) == 1
    scores_consistent = max(scores) - min(scores) < 1e-6
    
    print(f"\\nConsistency check:")
    print(f"   Labels consistent: {'✅' if labels_consistent else '❌'}")
    print(f"   Scores consistent: {'✅' if scores_consistent else '❌'}")
    
    if not labels_consistent:
        print(f"   Labels: {labels}")
    if not scores_consistent:
        print(f"   Score range: {min(scores):.8f} - {max(scores):.8f}")

def recommend_solution():
    """Recommend the best approach based on findings."""
    
    print("\\n" + "="*80)
    print("RECOMMENDED SOLUTION")
    print("="*80)
    
    print("ANALYSIS OF WRAPPER ISSUES:")
    print("-" * 50)
    print("1. ❌ Wrapper adds unnecessary complexity")
    print("2. ❌ Case sensitivity issues (High vs high)")
    print("3. ❌ Memory overhead from duplicate model loading")
    print("4. ❌ Potential device/precision differences")
    print("5. ❌ Additional abstraction layers")
    
    print("\\nRECOMMENDED APPROACH:")
    print("-" * 50)
    print("✅ USE MODELS AS ORIGINALLY INTENDED:")
    
    print("\\n1. For Classification:")
    print("```python")
    print("# Direct usage - as the model was trained")
    print("classifier = pipeline('text-classification',")
    print("                     model='CIRCL/vulnerability-severity-classification-roberta-base')")
    print("result = classifier('SQL injection vulnerability')")
    print("severity = result[0]['label']  # 'High', 'Critical', etc.")
    print("confidence = result[0]['score']")
    print("```")
    
    print("\\n2. For Generation:")
    print("```python")
    print("# Direct usage - as the model was trained")
    print("generator = pipeline('text-generation',")
    print("                    model='CIRCL/vulnerability-description-generation-gpt2')")
    print("result = generator('A vulnerability in OpenSSL allows', max_length=100)")
    print("description = result[0]['generated_text']")
    print("```")
    
    print("\\n3. For Custom CVSS Mapping (if needed):")
    print("```python")
    print("# Simple mapping function - no wrapper class")
    print("def severity_to_cvss(severity):")
    print("    mapping = {'Low': 3.0, 'Medium': 6.0, 'High': 8.0, 'Critical': 9.5}")
    print("    return mapping.get(severity, 5.0)")
    print("")
    print("# Usage")
    print("result = classifier(text)")
    print("severity = result[0]['label']")
    print("cvss = severity_to_cvss(severity)")
    print("```")
    
    print("\\nWHY THIS IS BETTER:")
    print("-" * 50)
    print("✅ No performance degradation")
    print("✅ Uses models as designed and trained")
    print("✅ Consistent with VulnTrain approach")
    print("✅ Minimal memory overhead")
    print("✅ Predictable and reproducible results")
    print("✅ Follows best practices for transformer models")

def main():
    """Main debugging function."""
    
    print("DEBUGGING WRAPPER PERFORMANCE ISSUES")
    print("="*80)
    print("Investigating why the wrapper approach causes problems...")
    
    debug_prediction_differences()
    debug_wrapper_internals()
    test_minimal_wrapper()
    investigate_model_loading()
    recommend_solution()

if __name__ == "__main__":
    main() 