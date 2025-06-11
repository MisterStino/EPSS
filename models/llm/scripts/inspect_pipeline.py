#!/usr/bin/env python3
"""
Script to inspect what the pipeline object contains for the CIRCL vulnerability generation model.
This will show you exactly what pipeline does under the hood.
"""

from transformers import pipeline
import inspect

def inspect_pipeline_object():
    """Inspect the pipeline object in detail."""
    
    print("="*80)
    print("INSPECTING CIRCL VULNERABILITY GENERATION PIPELINE")
    print("="*80)
    
    # Create the pipeline
    print("\n1. Creating pipeline...")
    pipe = pipeline("text-generation", model="CIRCL/vulnerability-description-generation-gpt2")
    
    print(f"✅ Pipeline created successfully!")
    print(f"Pipeline type: {type(pipe)}")
    
    # Basic pipeline information
    print(f"\n2. Basic Pipeline Information:")
    print("-" * 50)
    print(f"Pipeline class: {pipe.__class__.__name__}")
    print(f"Pipeline module: {pipe.__class__.__module__}")
    print(f"Task: {pipe.task}")
    
    # Model information
    print(f"\n3. Model Information:")
    print("-" * 50)
    print(f"Model class: {pipe.model.__class__.__name__}")
    print(f"Model type: {type(pipe.model)}")
    print(f"Model config type: {type(pipe.model.config)}")
    print(f"Model device: {pipe.model.device}")
    
    # Tokenizer information
    print(f"\n4. Tokenizer Information:")
    print("-" * 50)
    print(f"Tokenizer class: {pipe.tokenizer.__class__.__name__}")
    print(f"Tokenizer type: {type(pipe.tokenizer)}")
    print(f"Vocab size: {pipe.tokenizer.vocab_size}")
    print(f"Model max length: {pipe.tokenizer.model_max_length}")
    
    # Configuration details
    print(f"\n5. Model Configuration:")
    print("-" * 50)
    config = pipe.model.config
    print(f"Architecture: {config.architectures}")
    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Number of attention heads: {config.num_attention_heads}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Max position embeddings: {config.max_position_embeddings}")
    
    # Pipeline attributes
    print(f"\n6. Pipeline Attributes:")
    print("-" * 50)
    attributes = [attr for attr in dir(pipe) if not attr.startswith('_')]
    for attr in attributes:
        try:
            value = getattr(pipe, attr)
            if not callable(value):
                print(f"{attr}: {value}")
        except:
            print(f"{attr}: <could not access>")
    
    # Pipeline methods
    print(f"\n7. Pipeline Methods:")
    print("-" * 50)
    methods = [method for method in dir(pipe) if callable(getattr(pipe, method)) and not method.startswith('_')]
    for method in methods:
        print(f"- {method}()")
    
    # Inspect the __call__ method (what happens when you use pipe())
    print(f"\n8. Pipeline __call__ Method Signature:")
    print("-" * 50)
    call_signature = inspect.signature(pipe.__call__)
    print(f"pipe{call_signature}")
    
    # Show default parameters
    print(f"\n9. Default Generation Parameters:")
    print("-" * 50)
    # Try to access default parameters
    try:
        if hasattr(pipe, '_forward_params'):
            print(f"Forward params: {pipe._forward_params}")
        if hasattr(pipe, 'call_count'):
            print(f"Call count: {pipe.call_count}")
    except:
        pass
    
    # Show what the pipeline actually contains
    print(f"\n10. Pipeline Internal Structure:")
    print("-" * 50)
    print(f"Pipeline object contents:")
    for key, value in pipe.__dict__.items():
        if not key.startswith('_'):
            print(f"  {key}: {type(value)} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    # Test a simple generation to see the process
    print(f"\n11. Test Generation Process:")
    print("-" * 50)
    test_prompt = "A vulnerability in the system allows"
    print(f"Input prompt: '{test_prompt}'")
    
    # Generate with verbose output
    result = pipe(test_prompt, max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7)
    print(f"Generated result type: {type(result)}")
    print(f"Generated result: {result}")
    
    # Show tokenization process
    print(f"\n12. Tokenization Process:")
    print("-" * 50)
    tokens = pipe.tokenizer.encode(test_prompt)
    print(f"Input tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Decoded tokens: {pipe.tokenizer.decode(tokens)}")
    
    return pipe

def inspect_model_config_file():
    """Show the actual config.json content that pipeline reads."""
    print(f"\n" + "="*80)
    print("MODEL CONFIG FILE INSPECTION")
    print("="*80)
    
    from transformers import AutoConfig
    
    # Load the config directly
    config = AutoConfig.from_pretrained("CIRCL/vulnerability-description-generation-gpt2")
    
    print(f"\n13. Raw Model Configuration (config.json):")
    print("-" * 50)
    
    # Show all config attributes
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    # Run the inspection
    pipeline_obj = inspect_pipeline_object()
    inspect_model_config_file()
    
    print(f"\n" + "="*80)
    print("INSPECTION COMPLETE!")
    print("="*80)
    print(f"You can now use 'pipeline_obj' to experiment further.")
    print(f"Try: pipeline_obj('Your prompt here', max_length=100)") 