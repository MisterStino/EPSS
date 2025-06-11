#!/usr/bin/env python3
"""
Script to test whether CIRCL models have built-in methods to modify their internal instructions/behavior.
This explores deeper customization beyond just changing input prompts.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoConfig, pipeline, GenerationConfig
)
from label import VulnerabilitySeverityClassifier
import torch
import json

def explore_classification_model_modification():
    """Explore ways to modify the classification model's behavior."""
    
    print("="*80)
    print("EXPLORING CLASSIFICATION MODEL MODIFICATION OPTIONS")
    print("="*80)
    
    # Load model and config
    model_name = "CIRCL/vulnerability-severity-classification-roberta-base"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print("1. CURRENT MODEL CONFIGURATION:")
    print("-" * 50)
    print(f"Model type: {config.model_type}")
    print(f"Number of labels: {config.num_labels}")
    print(f"Label mappings: {config.id2label}")
    print(f"Problem type: {config.problem_type}")
    
    print("\\n2. AVAILABLE CONFIGURATION PARAMETERS:")
    print("-" * 50)
    
    # Get all configurable parameters
    config_dict = config.to_dict()
    modifiable_params = []
    
    for key, value in config_dict.items():
        if not key.startswith('_') and key not in ['transformers_version', 'torch_dtype']:
            modifiable_params.append((key, value, type(value).__name__))
    
    print("Parameters that could potentially be modified:")
    for param, value, param_type in modifiable_params[:15]:  # Show first 15
        print(f"  {param}: {value} ({param_type})")
    
    print("\\n3. TESTING CONFIGURATION MODIFICATIONS:")
    print("-" * 50)
    
    # Test 1: Can we change label mappings?
    print("Test 1: Modifying label mappings...")
    try:
        original_labels = config.id2label.copy()
        config.id2label = {
            "0": "Minimal",
            "1": "Moderate", 
            "2": "Severe",
            "3": "Extreme"
        }
        config.label2id = {v: int(k) for k, v in config.id2label.items()}
        
        print(f"  ✅ Successfully changed labels: {config.id2label}")
        
        # Test if this affects predictions
        classifier = VulnerabilitySeverityClassifier()
        if classifier.is_ready():
            # Update the classifier's labels
            classifier.severity_labels = list(config.id2label.values())
            severity, cvss = classifier.predict_single("SQL injection vulnerability")
            print(f"  Prediction with new labels: {severity}")
        
        # Restore original
        config.id2label = original_labels
        config.label2id = {v: int(k) for k, v in config.id2label.items()}
        
    except Exception as e:
        print(f"  ❌ Failed to modify labels: {e}")
    
    # Test 2: Can we change the problem type?
    print("\\nTest 2: Modifying problem type...")
    try:
        original_problem_type = config.problem_type
        config.problem_type = "multi_label_classification"
        print(f"  ✅ Changed problem type to: {config.problem_type}")
        
        # Restore
        config.problem_type = original_problem_type
        
    except Exception as e:
        print(f"  ❌ Failed to modify problem type: {e}")
    
    # Test 3: Model methods and attributes
    print("\\n4. MODEL METHODS FOR BEHAVIOR MODIFICATION:")
    print("-" * 50)
    
    model_methods = [method for method in dir(model) if not method.startswith('_')]
    behavior_methods = [m for m in model_methods if any(keyword in m.lower() 
                       for keyword in ['config', 'set', 'update', 'modify', 'change'])]
    
    print("Methods that might modify behavior:")
    for method in behavior_methods:
        try:
            method_obj = getattr(model, method)
            if callable(method_obj):
                print(f"  📝 {method}() - {method_obj.__doc__[:100] if method_obj.__doc__ else 'No docs'}...")
        except:
            print(f"  📝 {method} - Could not inspect")
    
    return config, model, tokenizer

def explore_generation_model_modification():
    """Explore ways to modify the generation model's behavior."""
    
    print("\\n" + "="*80)
    print("EXPLORING GENERATION MODEL MODIFICATION OPTIONS")
    print("="*80)
    
    model_name = "CIRCL/vulnerability-description-generation-gpt2"
    
    print("1. CURRENT GENERATION CONFIGURATION:")
    print("-" * 50)
    
    # Load with pipeline to see current config
    pipe = pipeline("text-generation", model=model_name)
    
    print(f"Current generation config:")
    gen_config = pipe.model.generation_config
    print(f"  Max length: {gen_config.max_length}")
    print(f"  Do sample: {gen_config.do_sample}")
    print(f"  Temperature: {getattr(gen_config, 'temperature', 'Not set')}")
    print(f"  Top-p: {getattr(gen_config, 'top_p', 'Not set')}")
    print(f"  Top-k: {getattr(gen_config, 'top_k', 'Not set')}")
    
    print("\\n2. TESTING GENERATION CONFIG MODIFICATIONS:")
    print("-" * 50)
    
    # Test 1: Modify generation parameters
    print("Test 1: Modifying generation parameters...")
    try:
        # Create custom generation config
        custom_config = GenerationConfig(
            max_length=200,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            length_penalty=1.0,
            early_stopping=True,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        print(f"  ✅ Created custom config: temperature={custom_config.temperature}")
        
        # Test generation with custom config
        test_prompt = "A vulnerability in the system allows"
        
        # Original generation
        original_result = pipe(test_prompt, max_length=50)
        print(f"  Original: '{original_result[0]['generated_text'][len(test_prompt):].strip()[:50]}...'")
        
        # Custom generation
        custom_result = pipe(test_prompt, generation_config=custom_config)
        print(f"  Custom: '{custom_result[0]['generated_text'][len(test_prompt):].strip()[:50]}...'")
        
    except Exception as e:
        print(f"  ❌ Failed to modify generation config: {e}")
    
    # Test 2: Explore model's generation methods
    print("\\n3. GENERATION MODEL METHODS:")
    print("-" * 50)
    
    model = pipe.model
    generation_methods = [method for method in dir(model) if 'generat' in method.lower()]
    
    print("Available generation methods:")
    for method in generation_methods:
        try:
            method_obj = getattr(model, method)
            if callable(method_obj):
                print(f"  🔧 {method}()")
        except:
            print(f"  🔧 {method}")
    
    return pipe

def test_advanced_customization():
    """Test advanced customization techniques."""
    
    print("\\n" + "="*80)
    print("TESTING ADVANCED CUSTOMIZATION TECHNIQUES")
    print("="*80)
    
    print("1. TOKENIZER MODIFICATIONS:")
    print("-" * 50)
    
    # Test tokenizer customization
    tokenizer = AutoTokenizer.from_pretrained("CIRCL/vulnerability-severity-classification-roberta-base")
    
    print(f"Original vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    
    # Test adding special tokens (this would require retraining in practice)
    print("\\nTest: Adding custom tokens...")
    try:
        # Add custom tokens
        custom_tokens = ["<CRITICAL>", "<HIGH>", "<MEDIUM>", "<LOW>"]
        num_added = tokenizer.add_tokens(custom_tokens)
        print(f"  ✅ Added {num_added} custom tokens")
        print(f"  New vocab size: {tokenizer.vocab_size}")
        
        # Test tokenization with custom tokens
        test_text = "<CRITICAL> SQL injection vulnerability"
        tokens = tokenizer.tokenize(test_text)
        print(f"  Tokenized: {tokens}")
        
    except Exception as e:
        print(f"  ❌ Failed to add custom tokens: {e}")
    
    print("\\n2. MODEL ARCHITECTURE INSPECTION:")
    print("-" * 50)
    
    # Load model and inspect architecture
    model = AutoModelForSequenceClassification.from_pretrained(
        "CIRCL/vulnerability-severity-classification-roberta-base"
    )
    
    print("Model layers and components:")
    for name, module in model.named_modules():
        if len(name.split('.')) <= 2:  # Top-level components only
            print(f"  🏗️  {name}: {type(module).__name__}")
    
    print("\\n3. TESTING LAYER MODIFICATIONS:")
    print("-" * 50)
    
    # Test if we can modify specific layers
    print("Test: Accessing classifier layer...")
    try:
        if hasattr(model, 'classifier'):
            classifier_layer = model.classifier
            print(f"  ✅ Classifier layer: {classifier_layer}")
            print(f"  Input features: {classifier_layer.in_features}")
            print(f"  Output features: {classifier_layer.out_features}")
            
            # Could theoretically replace this layer (but would break the model)
            print(f"  💡 This layer maps {classifier_layer.in_features} features to {classifier_layer.out_features} classes")
            
    except Exception as e:
        print(f"  ❌ Failed to access classifier layer: {e}")

def test_practical_modifications():
    """Test practical modifications that actually work."""
    
    print("\\n" + "="*80)
    print("PRACTICAL MODIFICATIONS THAT WORK")
    print("="*80)
    
    print("1. CLASSIFICATION MODEL - WORKING MODIFICATIONS:")
    print("-" * 50)
    
    # Test our custom class modifications
    classifier = VulnerabilitySeverityClassifier()
    
    if classifier.is_ready():
        print("✅ Custom CVSS mapping modification:")
        
        # Show current mapping
        print(f"  Current mapping: {classifier.cvss_mapping}")
        
        # Modify CVSS mapping
        new_mapping = {
            "low": 2.5,
            "medium": 5.5, 
            "high": 7.5,
            "critical": 9.8
        }
        
        classifier.update_cvss_mapping(new_mapping)
        print(f"  New mapping: {classifier.cvss_mapping}")
        
        # Test prediction with new mapping
        severity, cvss = classifier.predict_single("SQL injection vulnerability")
        print(f"  Prediction: {severity} → CVSS {cvss} (using new mapping)")
    
    print("\\n2. GENERATION MODEL - WORKING MODIFICATIONS:")
    print("-" * 50)
    
    pipe = pipeline("text-generation", model="CIRCL/vulnerability-description-generation-gpt2")
    
    print("✅ Runtime generation parameter modification:")
    
    test_prompt = "A buffer overflow vulnerability"
    
    # Different generation styles
    configs = [
        {"temperature": 0.3, "top_p": 0.8, "max_length": 80, "name": "Conservative"},
        {"temperature": 0.9, "top_p": 0.95, "max_length": 80, "name": "Creative"},
        {"temperature": 1.2, "top_k": 40, "max_length": 80, "name": "Diverse"}
    ]
    
    for config in configs:
        name = config.pop("name")
        try:
            result = pipe(test_prompt, **config, num_return_sequences=1)
            generated = result[0]['generated_text'][len(test_prompt):].strip()
            print(f"  {name}: '{generated[:60]}...'")
        except Exception as e:
            print(f"  {name}: Failed - {e}")

def main():
    """Main function to explore model modification capabilities."""
    
    print("INVESTIGATING MODEL MODIFICATION CAPABILITIES")
    print("="*80)
    print("Exploring whether CIRCL models have built-in methods to modify behavior...")
    
    # Explore classification model
    config, model, tokenizer = explore_classification_model_modification()
    
    # Explore generation model
    pipe = explore_generation_model_modification()
    
    # Test advanced techniques
    test_advanced_customization()
    
    # Test practical modifications
    test_practical_modifications()
    
    print("\\n" + "="*80)
    print("SUMMARY: WHAT CAN BE MODIFIED")
    print("="*80)
    
    print("✅ WHAT WORKS (Built-in modification methods):")
    print("  1. Generation parameters (temperature, top_p, max_length, etc.)")
    print("  2. Label name mappings (cosmetic changes)")
    print("  3. CVSS score mappings (through custom wrapper)")
    print("  4. Runtime behavior via custom classes")
    print("  5. Post-processing and result formatting")
    
    print("\\n⚠️  WHAT'S LIMITED (Requires careful handling):")
    print("  1. Adding new tokens (requires model retraining)")
    print("  2. Changing number of output classes")
    print("  3. Modifying core architecture")
    
    print("\\n❌ WHAT DOESN'T WORK (Fundamental limitations):")
    print("  1. Changing the core task (classification ↔ generation)")
    print("  2. Adding new capabilities not in training data")
    print("  3. Instruction-following behavior")
    print("  4. Changing model architecture post-training")

if __name__ == "__main__":
    main() 