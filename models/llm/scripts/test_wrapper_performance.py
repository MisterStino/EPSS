#!/usr/bin/env python3
"""
Script to test whether custom wrapper approaches degrade model performance
compared to using the models as originally intended.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from label import VulnerabilitySeverityClassifier
import time
import torch
import numpy as np

def test_classification_performance():
    """Test classification performance: wrapper vs direct usage."""
    
    print("="*80)
    print("TESTING CLASSIFICATION MODEL PERFORMANCE")
    print("="*80)
    
    # Test vulnerabilities with known expected severities
    test_cases = [
        {
            "text": "SQL injection vulnerability allows unauthorized database access",
            "expected_range": ["High", "Critical"],
            "description": "SQL injection - should be high severity"
        },
        {
            "text": "Remote code execution vulnerability in web server",
            "expected_range": ["High", "Critical"], 
            "description": "RCE - should be high/critical severity"
        },
        {
            "text": "Information disclosure through error messages",
            "expected_range": ["Low", "Medium"],
            "description": "Info disclosure - should be low/medium severity"
        },
        {
            "text": "Cross-site scripting vulnerability in user input",
            "expected_range": ["Medium", "High"],
            "description": "XSS - should be medium/high severity"
        },
        {
            "text": "Buffer overflow allows arbitrary code execution",
            "expected_range": ["High", "Critical"],
            "description": "Buffer overflow - should be high/critical severity"
        }
    ]
    
    print("1. TESTING DIRECT MODEL USAGE (AS INTENDED):")
    print("-" * 50)
    
    # Direct pipeline usage (as the model was designed)
    direct_classifier = pipeline("text-classification", 
                                model="CIRCL/vulnerability-severity-classification-roberta-base")
    
    direct_results = []
    direct_times = []
    
    for i, case in enumerate(test_cases):
        start_time = time.time()
        result = direct_classifier(case["text"])
        end_time = time.time()
        
        prediction = result[0]
        is_reasonable = prediction['label'] in case['expected_range']
        
        direct_results.append({
            'case': i,
            'text': case["text"][:50] + "...",
            'prediction': prediction['label'],
            'confidence': prediction['score'],
            'reasonable': is_reasonable,
            'time': end_time - start_time
        })
        direct_times.append(end_time - start_time)
        
        print(f"   Case {i+1}: {prediction['label']} ({prediction['score']:.3f}) - {'✅' if is_reasonable else '❌'}")
        print(f"           Time: {end_time - start_time:.4f}s")
    
    print("\\n2. TESTING WRAPPER APPROACH:")
    print("-" * 50)
    
    # Our custom wrapper
    wrapper_classifier = VulnerabilitySeverityClassifier()
    
    wrapper_results = []
    wrapper_times = []
    
    if wrapper_classifier.is_ready():
        for i, case in enumerate(test_cases):
            start_time = time.time()
            severity, cvss = wrapper_classifier.predict_single(case["text"])
            end_time = time.time()
            
            is_reasonable = severity in case['expected_range']
            
            wrapper_results.append({
                'case': i,
                'text': case["text"][:50] + "...",
                'prediction': severity,
                'cvss': cvss,
                'reasonable': is_reasonable,
                'time': end_time - start_time
            })
            wrapper_times.append(end_time - start_time)
            
            print(f"   Case {i+1}: {severity} (CVSS: {cvss}) - {'✅' if is_reasonable else '❌'}")
            print(f"           Time: {end_time - start_time:.4f}s")
    
    print("\\n3. PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    # Compare accuracy
    direct_accuracy = sum(1 for r in direct_results if r['reasonable']) / len(direct_results)
    wrapper_accuracy = sum(1 for r in wrapper_results if r['reasonable']) / len(wrapper_results)
    
    print(f"Accuracy:")
    print(f"   Direct model: {direct_accuracy:.2%} ({sum(1 for r in direct_results if r['reasonable'])}/{len(direct_results)})")
    print(f"   Wrapper:      {wrapper_accuracy:.2%} ({sum(1 for r in wrapper_results if r['reasonable'])}/{len(wrapper_results)})")
    
    # Compare speed
    direct_avg_time = np.mean(direct_times)
    wrapper_avg_time = np.mean(wrapper_times)
    
    print(f"\\nSpeed:")
    print(f"   Direct model: {direct_avg_time:.4f}s average")
    print(f"   Wrapper:      {wrapper_avg_time:.4f}s average")
    print(f"   Overhead:     {((wrapper_avg_time - direct_avg_time) / direct_avg_time * 100):+.1f}%")
    
    # Compare predictions directly
    print(f"\\n4. PREDICTION CONSISTENCY:")
    print("-" * 50)
    
    consistent_predictions = 0
    for i in range(len(test_cases)):
        direct_pred = direct_results[i]['prediction']
        wrapper_pred = wrapper_results[i]['prediction']
        is_consistent = direct_pred == wrapper_pred
        consistent_predictions += is_consistent
        
        print(f"   Case {i+1}: Direct={direct_pred}, Wrapper={wrapper_pred} - {'✅' if is_consistent else '❌'}")
    
    consistency_rate = consistent_predictions / len(test_cases)
    print(f"\\nConsistency rate: {consistency_rate:.2%}")
    
    return {
        'direct_accuracy': direct_accuracy,
        'wrapper_accuracy': wrapper_accuracy,
        'direct_speed': direct_avg_time,
        'wrapper_speed': wrapper_avg_time,
        'consistency': consistency_rate
    }

def test_generation_performance():
    """Test generation performance: direct vs wrapper approaches."""
    
    print("\\n" + "="*80)
    print("TESTING GENERATION MODEL PERFORMANCE")
    print("="*80)
    
    test_prompts = [
        "A vulnerability in OpenSSL allows",
        "Buffer overflow in Apache",
        "SQL injection vulnerability",
        "Cross-site scripting flaw",
        "Remote code execution bug"
    ]
    
    print("1. TESTING DIRECT MODEL USAGE (AS INTENDED):")
    print("-" * 50)
    
    # Direct pipeline usage
    direct_generator = pipeline("text-generation", 
                               model="CIRCL/vulnerability-description-generation-gpt2")
    
    direct_results = []
    direct_times = []
    
    for i, prompt in enumerate(test_prompts):
        start_time = time.time()
        result = direct_generator(prompt, max_new_tokens=50, num_return_sequences=1)
        end_time = time.time()
        
        generated = result[0]['generated_text'][len(prompt):].strip()
        
        direct_results.append({
            'prompt': prompt,
            'generated': generated,
            'time': end_time - start_time
        })
        direct_times.append(end_time - start_time)
        
        print(f"   Prompt {i+1}: '{prompt}'")
        print(f"   Generated: '{generated[:60]}...'")
        print(f"   Time: {end_time - start_time:.4f}s")
    
    print("\\n2. TESTING WITH CUSTOM GENERATION CONFIGS:")
    print("-" * 50)
    
    # Test with our custom generation configs
    custom_configs = [
        {"temperature": 0.3, "top_p": 0.8, "repetition_penalty": 1.2},
        {"temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.1},
        {"temperature": 0.5, "top_p": 0.85, "repetition_penalty": 1.15}
    ]
    
    custom_results = []
    custom_times = []
    
    for i, prompt in enumerate(test_prompts[:3]):  # Test fewer for time
        config = custom_configs[i]
        
        start_time = time.time()
        result = direct_generator(prompt, max_new_tokens=50, **config)
        end_time = time.time()
        
        generated = result[0]['generated_text'][len(prompt):].strip()
        
        custom_results.append({
            'prompt': prompt,
            'generated': generated,
            'config': config,
            'time': end_time - start_time
        })
        custom_times.append(end_time - start_time)
        
        print(f"   Prompt {i+1}: '{prompt}'")
        print(f"   Config: {config}")
        print(f"   Generated: '{generated[:60]}...'")
        print(f"   Time: {end_time - start_time:.4f}s")
    
    print("\\n3. GENERATION PERFORMANCE COMPARISON:")
    print("-" * 50)
    
    direct_avg_time = np.mean(direct_times)
    custom_avg_time = np.mean(custom_times)
    
    print(f"Speed:")
    print(f"   Direct generation: {direct_avg_time:.4f}s average")
    print(f"   Custom configs:    {custom_avg_time:.4f}s average")
    print(f"   Overhead:          {((custom_avg_time - direct_avg_time) / direct_avg_time * 100):+.1f}%")
    
    return {
        'direct_speed': direct_avg_time,
        'custom_speed': custom_avg_time
    }

def test_memory_usage():
    """Test memory usage of different approaches."""
    
    print("\\n" + "="*80)
    print("TESTING MEMORY USAGE")
    print("="*80)
    
    if torch.cuda.is_available():
        print("1. GPU MEMORY USAGE:")
        print("-" * 50)
        
        # Baseline memory
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated()
        print(f"   Baseline GPU memory: {baseline_memory / 1024**2:.1f} MB")
        
        # Direct model loading
        direct_classifier = pipeline("text-classification", 
                                    model="CIRCL/vulnerability-severity-classification-roberta-base")
        direct_memory = torch.cuda.memory_allocated()
        print(f"   Direct model memory: {direct_memory / 1024**2:.1f} MB")
        print(f"   Direct model overhead: {(direct_memory - baseline_memory) / 1024**2:.1f} MB")
        
        # Wrapper approach
        wrapper_classifier = VulnerabilitySeverityClassifier()
        wrapper_memory = torch.cuda.memory_allocated()
        print(f"   Wrapper memory: {wrapper_memory / 1024**2:.1f} MB")
        print(f"   Wrapper overhead: {(wrapper_memory - baseline_memory) / 1024**2:.1f} MB")
        
        # Memory difference
        memory_diff = wrapper_memory - direct_memory
        print(f"   Additional wrapper overhead: {memory_diff / 1024**2:.1f} MB")
        
        return {
            'baseline': baseline_memory,
            'direct': direct_memory,
            'wrapper': wrapper_memory,
            'wrapper_overhead': memory_diff
        }
    else:
        print("   GPU not available, skipping GPU memory test")
        return {}

def analyze_performance_impact():
    """Analyze the overall performance impact of wrapper approaches."""
    
    print("\\n" + "="*80)
    print("PERFORMANCE IMPACT ANALYSIS")
    print("="*80)
    
    print("1. THEORETICAL CONCERNS:")
    print("-" * 50)
    print("   ❓ Does wrapper add computational overhead?")
    print("   ❓ Does custom logic interfere with model predictions?")
    print("   ❓ Does instruction-like usage degrade accuracy?")
    print("   ❓ Does memory usage increase significantly?")
    
    # Run performance tests
    classification_perf = test_classification_performance()
    generation_perf = test_generation_performance()
    memory_usage = test_memory_usage()
    
    print("\\n2. ACTUAL FINDINGS:")
    print("-" * 50)
    
    # Analyze classification performance
    accuracy_impact = classification_perf['wrapper_accuracy'] - classification_perf['direct_accuracy']
    speed_impact = ((classification_perf['wrapper_speed'] - classification_perf['direct_speed']) / 
                   classification_perf['direct_speed'] * 100)
    
    print(f"Classification Model:")
    print(f"   Accuracy impact: {accuracy_impact:+.1%}")
    print(f"   Speed impact: {speed_impact:+.1f}%")
    print(f"   Prediction consistency: {classification_perf['consistency']:.1%}")
    
    # Analyze generation performance
    gen_speed_impact = ((generation_perf['custom_speed'] - generation_perf['direct_speed']) / 
                       generation_perf['direct_speed'] * 100)
    
    print(f"\\nGeneration Model:")
    print(f"   Speed impact: {gen_speed_impact:+.1f}%")
    
    # Analyze memory usage
    if memory_usage:
        memory_impact = memory_usage['wrapper_overhead'] / 1024**2
        print(f"\\nMemory Usage:")
        print(f"   Additional overhead: {memory_impact:.1f} MB")
    
    print("\\n3. PERFORMANCE VERDICT:")
    print("-" * 50)
    
    # Determine if performance is significantly impacted
    significant_accuracy_loss = abs(accuracy_impact) > 0.1  # >10% accuracy change
    significant_speed_loss = abs(speed_impact) > 20  # >20% speed change
    low_consistency = classification_perf['consistency'] < 0.8  # <80% consistency
    
    if significant_accuracy_loss:
        print("   ❌ SIGNIFICANT ACCURACY IMPACT DETECTED")
    elif abs(accuracy_impact) > 0.05:
        print("   ⚠️  MINOR ACCURACY IMPACT DETECTED")
    else:
        print("   ✅ NO SIGNIFICANT ACCURACY IMPACT")
    
    if significant_speed_loss:
        print("   ❌ SIGNIFICANT SPEED IMPACT DETECTED")
    elif abs(speed_impact) > 10:
        print("   ⚠️  MINOR SPEED IMPACT DETECTED")
    else:
        print("   ✅ NO SIGNIFICANT SPEED IMPACT")
    
    if low_consistency:
        print("   ❌ LOW PREDICTION CONSISTENCY")
    else:
        print("   ✅ HIGH PREDICTION CONSISTENCY")
    
    # Overall recommendation
    major_issues = significant_accuracy_loss or significant_speed_loss or low_consistency
    minor_issues = (abs(accuracy_impact) > 0.05 or abs(speed_impact) > 10 or 
                   classification_perf['consistency'] < 0.9)
    
    print("\\n4. RECOMMENDATION:")
    print("-" * 50)
    
    if major_issues:
        print("   ❌ WRAPPER APPROACH NOT RECOMMENDED")
        print("   - Significant performance degradation detected")
        print("   - Use models as originally intended")
    elif minor_issues:
        print("   ⚠️  WRAPPER APPROACH ACCEPTABLE WITH CAUTION")
        print("   - Minor performance impact detected")
        print("   - Monitor performance in production")
    else:
        print("   ✅ WRAPPER APPROACH RECOMMENDED")
        print("   - No significant performance impact")
        print("   - Safe to use in production")

def main():
    """Main function to test wrapper performance impact."""
    
    print("TESTING WRAPPER PERFORMANCE IMPACT")
    print("="*80)
    print("Investigating whether custom wrapper approaches degrade model performance...")
    
    analyze_performance_impact()

if __name__ == "__main__":
    main() 