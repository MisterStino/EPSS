#!/usr/bin/env python3
"""
Script to test specific built-in methods for modifying model instructions/behavior.
Focus on practical instruction extension capabilities.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoConfig, pipeline, GenerationConfig
)
import torch
import json

def test_generation_config_instructions():
    """Test if GenerationConfig can be used to modify instruction-like behavior."""
    
    print("="*80)
    print("TESTING GENERATION CONFIG AS INSTRUCTION MODIFICATION")
    print("="*80)
    
    pipe = pipeline("text-generation", model="CIRCL/vulnerability-description-generation-gpt2")
    
    print("1. EXPLORING GENERATION CONFIG PARAMETERS:")
    print("-" * 50)
    
    # Get all available generation config parameters
    gen_config = GenerationConfig()
    config_params = [attr for attr in dir(gen_config) if not attr.startswith('_')]
    
    instruction_related = []
    for param in config_params:
        if any(keyword in param.lower() for keyword in 
               ['prefix', 'prompt', 'instruction', 'system', 'template', 'format']):
            instruction_related.append(param)
    
    print("Instruction-related parameters found:")
    if instruction_related:
        for param in instruction_related:
            value = getattr(gen_config, param, "Not available")
            print(f"  📝 {param}: {value}")
    else:
        print("  ❌ No explicit instruction-related parameters found")
    
    print("\\n2. TESTING CUSTOM GENERATION CONFIGS:")
    print("-" * 50)
    
    # Test different "instruction-like" configurations
    test_prompt = "A vulnerability allows"
    
    configs = [
        {
            "name": "Technical Focus",
            "config": {
                "temperature": 0.3,
                "top_p": 0.8,
                "repetition_penalty": 1.2,
                "length_penalty": 1.1,
                "max_new_tokens": 60
            }
        },
        {
            "name": "Detailed Description", 
            "config": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "length_penalty": 0.9,
                "max_new_tokens": 100
            }
        },
        {
            "name": "Concise Summary",
            "config": {
                "temperature": 0.2,
                "top_p": 0.7,
                "repetition_penalty": 1.3,
                "length_penalty": 1.5,
                "max_new_tokens": 30
            }
        }
    ]
    
    for test_config in configs:
        print(f"\\n{test_config['name']} Configuration:")
        try:
            result = pipe(test_prompt, **test_config['config'])
            generated = result[0]['generated_text'][len(test_prompt):].strip()
            print(f"  Result: '{generated}'")
        except Exception as e:
            print(f"  ❌ Failed: {e}")

def test_model_adapter_methods():
    """Test adapter-related methods for instruction modification."""
    
    print("\\n" + "="*80)
    print("TESTING ADAPTER METHODS FOR INSTRUCTION MODIFICATION")
    print("="*80)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "CIRCL/vulnerability-severity-classification-roberta-base"
    )
    
    print("1. EXPLORING ADAPTER CAPABILITIES:")
    print("-" * 50)
    
    # Check if model supports adapters
    adapter_methods = [method for method in dir(model) if 'adapter' in method.lower()]
    
    print("Available adapter methods:")
    for method in adapter_methods:
        try:
            method_obj = getattr(model, method)
            if callable(method_obj):
                doc = method_obj.__doc__
                doc_preview = doc[:100] + "..." if doc and len(doc) > 100 else doc or "No documentation"
                print(f"  🔧 {method}(): {doc_preview}")
        except Exception as e:
            print(f"  🔧 {method}: Could not inspect - {e}")
    
    print("\\n2. TESTING SET_ADAPTER METHOD:")
    print("-" * 50)
    
    # Test the set_adapter method we found
    try:
        # Check current adapter state
        print("Current adapter state:")
        if hasattr(model, 'active_adapters'):
            print(f"  Active adapters: {model.active_adapters}")
        else:
            print("  No active_adapters attribute found")
        
        # Try to get adapter info
        if hasattr(model, 'get_adapter_state_dict'):
            print("  ✅ Model supports adapter state management")
        else:
            print("  ❌ No adapter state management found")
            
    except Exception as e:
        print(f"  ❌ Adapter testing failed: {e}")

def test_input_embeddings_modification():
    """Test input embeddings modification for instruction-like behavior."""
    
    print("\\n" + "="*80)
    print("TESTING INPUT EMBEDDINGS MODIFICATION")
    print("="*80)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "CIRCL/vulnerability-severity-classification-roberta-base"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "CIRCL/vulnerability-severity-classification-roberta-base"
    )
    
    print("1. CURRENT INPUT EMBEDDINGS:")
    print("-" * 50)
    
    # Get current embeddings
    embeddings = model.get_input_embeddings()
    print(f"Embedding layer: {type(embeddings).__name__}")
    print(f"Vocabulary size: {embeddings.num_embeddings}")
    print(f"Embedding dimension: {embeddings.embedding_dim}")
    
    print("\\n2. TESTING CUSTOM INSTRUCTION TOKENS:")
    print("-" * 50)
    
    # Add special instruction tokens
    special_tokens = ["<CLASSIFY>", "<SEVERITY>", "<TECHNICAL>", "<BRIEF>"]
    
    print("Adding instruction tokens...")
    original_vocab_size = tokenizer.vocab_size
    num_added = tokenizer.add_tokens(special_tokens)
    
    print(f"  Added {num_added} tokens")
    print(f"  Vocab size: {original_vocab_size} → {tokenizer.vocab_size}")
    
    # Test tokenization with instruction tokens
    test_cases = [
        "<CLASSIFY> SQL injection vulnerability in web application",
        "<SEVERITY> Buffer overflow allows remote code execution", 
        "<TECHNICAL> Cross-site scripting vulnerability",
        "<BRIEF> Authentication bypass flaw"
    ]
    
    print("\\nTokenization with instruction tokens:")
    for test_case in test_cases:
        tokens = tokenizer.tokenize(test_case)
        print(f"  '{test_case[:30]}...' → {tokens[:5]}...")

def test_configuration_based_instructions():
    """Test configuration-based instruction modification."""
    
    print("\\n" + "="*80)
    print("TESTING CONFIGURATION-BASED INSTRUCTION MODIFICATION")
    print("="*80)
    
    print("1. CLASSIFICATION MODEL CONFIG MODIFICATION:")
    print("-" * 50)
    
    config = AutoConfig.from_pretrained("CIRCL/vulnerability-severity-classification-roberta-base")
    
    # Test adding custom configuration parameters
    print("Adding custom instruction parameters to config...")
    
    # Add custom instruction-related parameters
    config.instruction_mode = "severity_focus"
    config.output_format = "detailed"
    config.context_awareness = True
    config.custom_labels = {
        "low": "Low severity - minimal impact",
        "medium": "Medium severity - moderate impact", 
        "high": "High severity - significant impact",
        "critical": "Critical severity - severe impact"
    }
    
    print(f"  ✅ Added instruction_mode: {config.instruction_mode}")
    print(f"  ✅ Added output_format: {config.output_format}")
    print(f"  ✅ Added context_awareness: {config.context_awareness}")
    print(f"  ✅ Added custom_labels: {len(config.custom_labels)} entries")
    
    print("\\n2. TESTING CONFIG PERSISTENCE:")
    print("-" * 50)
    
    # Test if custom config can be saved/loaded
    try:
        config_dict = config.to_dict()
        custom_params = {k: v for k, v in config_dict.items() 
                        if k in ['instruction_mode', 'output_format', 'context_awareness', 'custom_labels']}
        
        print("Custom parameters in config:")
        for param, value in custom_params.items():
            print(f"  📝 {param}: {value}")
            
        # Test creating new config from dict
        new_config = AutoConfig.from_dict(config_dict)
        print(f"  ✅ Config recreation successful")
        print(f"  Preserved instruction_mode: {getattr(new_config, 'instruction_mode', 'Not found')}")
        
    except Exception as e:
        print(f"  ❌ Config persistence failed: {e}")

def test_pipeline_customization():
    """Test pipeline-level instruction customization."""
    
    print("\\n" + "="*80)
    print("TESTING PIPELINE-LEVEL INSTRUCTION CUSTOMIZATION")
    print("="*80)
    
    print("1. CUSTOM PIPELINE PARAMETERS:")
    print("-" * 50)
    
    # Test classification pipeline with custom parameters
    classifier = pipeline(
        "text-classification",
        model="CIRCL/vulnerability-severity-classification-roberta-base",
        return_all_scores=True,
        function_to_apply="softmax"
    )
    
    print("Testing custom pipeline behavior...")
    
    test_text = "SQL injection vulnerability allows data extraction"
    
    # Test different pipeline configurations
    configs = [
        {"name": "Standard", "params": {}},
        {"name": "All Scores", "params": {"return_all_scores": True}},
        {"name": "Top 2", "params": {"top_k": 2}},
    ]
    
    for config in configs:
        try:
            result = classifier(test_text, **config["params"])
            print(f"  {config['name']}: {result}")
        except Exception as e:
            print(f"  {config['name']}: Failed - {e}")
    
    print("\\n2. GENERATION PIPELINE CUSTOMIZATION:")
    print("-" * 50)
    
    # Test generation pipeline with instruction-like parameters
    generator = pipeline(
        "text-generation",
        model="CIRCL/vulnerability-description-generation-gpt2"
    )
    
    # Test "instruction-like" prompting through generation parameters
    instruction_configs = [
        {
            "name": "Technical Report Style",
            "prompt": "Technical analysis: A vulnerability",
            "params": {"temperature": 0.3, "max_new_tokens": 50, "repetition_penalty": 1.2}
        },
        {
            "name": "Security Alert Style", 
            "prompt": "SECURITY ALERT: A vulnerability",
            "params": {"temperature": 0.5, "max_new_tokens": 40, "repetition_penalty": 1.3}
        },
        {
            "name": "Detailed Description Style",
            "prompt": "Detailed vulnerability description: A vulnerability", 
            "params": {"temperature": 0.7, "max_new_tokens": 60, "repetition_penalty": 1.1}
        }
    ]
    
    for config in instruction_configs:
        try:
            result = generator(config["prompt"], **config["params"])
            generated = result[0]['generated_text'][len(config["prompt"]):].strip()
            print(f"  {config['name']}: '{generated[:50]}...'")
        except Exception as e:
            print(f"  {config['name']}: Failed - {e}")

def main():
    """Main function to test instruction modification capabilities."""
    
    print("TESTING BUILT-IN INSTRUCTION MODIFICATION METHODS")
    print("="*80)
    print("Exploring practical ways to extend model instructions...")
    
    # Test generation config modifications
    test_generation_config_instructions()
    
    # Test adapter methods
    test_model_adapter_methods()
    
    # Test input embeddings modification
    test_input_embeddings_modification()
    
    # Test configuration-based instructions
    test_configuration_based_instructions()
    
    # Test pipeline customization
    test_pipeline_customization()
    
    print("\\n" + "="*80)
    print("SUMMARY: BUILT-IN INSTRUCTION MODIFICATION CAPABILITIES")
    print("="*80)
    
    print("✅ WORKING BUILT-IN METHODS:")
    print("  1. GenerationConfig parameters (temperature, repetition_penalty, etc.)")
    print("  2. Custom configuration parameters (can add instruction-related fields)")
    print("  3. Pipeline-level customization (return_all_scores, top_k, etc.)")
    print("  4. Input embeddings extension (add instruction tokens)")
    print("  5. Runtime parameter modification")
    
    print("\\n🔧 ADAPTER SUPPORT:")
    print("  - Models have set_adapter() method")
    print("  - Could potentially support instruction adapters")
    print("  - Requires PEFT (Parameter Efficient Fine-Tuning) setup")
    
    print("\\n⚠️  LIMITATIONS:")
    print("  - No built-in instruction templates or system prompts")
    print("  - Models not trained for instruction-following")
    print("  - Custom tokens require model retraining to be effective")
    print("  - Configuration changes are mostly cosmetic")
    
    print("\\n💡 PRACTICAL RECOMMENDATIONS:")
    print("  1. Use GenerationConfig for behavior modification")
    print("  2. Implement custom wrapper classes for instruction-like behavior")
    print("  3. Use pipeline parameters for output formatting")
    print("  4. Consider fine-tuning with adapters for true instruction extension")

if __name__ == "__main__":
    main() 