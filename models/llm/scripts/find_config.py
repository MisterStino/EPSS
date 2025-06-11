#!/usr/bin/env python3
"""
Script to locate and inspect the actual config files for CIRCL models.
This shows you exactly where the files are stored and what they contain.
"""

import os
import json
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
import platform

def find_huggingface_cache():
    """Find the HuggingFace cache directory."""
    
    # Default cache locations by OS
    if platform.system() == "Windows":
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    else:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Check if HF_HOME environment variable is set
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        cache_dir = Path(hf_home) / "hub"
    
    # Check if HUGGINGFACE_HUB_CACHE is set
    hf_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_cache:
        cache_dir = Path(hf_cache)
    
    return cache_dir

def find_model_cache_dir(model_name):
    """Find the cache directory for a specific model."""
    cache_dir = find_huggingface_cache()
    
    # Convert model name to cache directory format
    # "CIRCL/vulnerability-description-generation-gpt2" becomes
    # "models--CIRCL--vulnerability-description-generation-gpt2"
    cache_model_name = "models--" + model_name.replace("/", "--")
    
    model_cache_dir = cache_dir / cache_model_name
    
    return model_cache_dir

def explore_model_files(model_name):
    """Explore all files in a model's cache directory."""
    print(f"\n{'='*80}")
    print(f"EXPLORING FILES FOR: {model_name}")
    print(f"{'='*80}")
    
    model_dir = find_model_cache_dir(model_name)
    
    print(f"Cache directory: {model_dir}")
    print(f"Directory exists: {model_dir.exists()}")
    
    if not model_dir.exists():
        print(f"❌ Model cache directory not found. Model may not be downloaded yet.")
        return None
    
    print(f"\n📁 Directory contents:")
    print("-" * 50)
    
    # List all files and subdirectories
    for item in sorted(model_dir.rglob("*")):
        relative_path = item.relative_to(model_dir)
        if item.is_file():
            size = item.stat().st_size
            size_str = f"{size:,} bytes"
            print(f"📄 {relative_path} ({size_str})")
        else:
            print(f"📁 {relative_path}/")
    
    return model_dir

def read_config_file(model_name):
    """Read and display the config.json file."""
    print(f"\n{'='*80}")
    print(f"CONFIG.JSON CONTENT FOR: {model_name}")
    print(f"{'='*80}")
    
    model_dir = find_model_cache_dir(model_name)
    
    # Look for config.json in various possible locations
    possible_config_paths = [
        model_dir / "config.json",
        model_dir / "snapshots" / "main" / "config.json",
    ]
    
    # Also check in snapshot directories
    if model_dir.exists():
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot_dir in snapshots_dir.iterdir():
                if snapshot_dir.is_dir():
                    possible_config_paths.append(snapshot_dir / "config.json")
    
    config_path = None
    for path in possible_config_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path:
        print(f"📍 Config file location: {config_path}")
        print(f"\n📄 Config file contents:")
        print("-" * 50)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Pretty print the JSON
            print(json.dumps(config_data, indent=2))
            
        except Exception as e:
            print(f"❌ Error reading config file: {e}")
    else:
        print(f"❌ config.json not found in any expected location")
        print(f"Searched paths:")
        for path in possible_config_paths:
            print(f"  - {path}")

def compare_configs():
    """Compare configs using transformers library vs direct file access."""
    print(f"\n{'='*80}")
    print(f"COMPARING CONFIG ACCESS METHODS")
    print(f"{'='*80}")
    
    model_name = "CIRCL/vulnerability-description-generation-gpt2"
    
    print(f"\n1. Using transformers.AutoConfig.from_pretrained():")
    print("-" * 50)
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"✅ Config loaded successfully")
        print(f"Model type: {config.model_type}")
        print(f"Architecture: {config.architectures}")
        print(f"Vocab size: {config.vocab_size}")
        print(f"Max position embeddings: {config.n_positions}")
        
        # Show where transformers thinks the config is
        print(f"\nConfig object type: {type(config)}")
        print(f"Config class: {config.__class__.__name__}")
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
    
    print(f"\n2. Direct file access:")
    print("-" * 50)
    read_config_file(model_name)

def show_cache_structure():
    """Show the overall cache structure."""
    print(f"\n{'='*80}")
    print(f"HUGGINGFACE CACHE STRUCTURE")
    print(f"{'='*80}")
    
    cache_dir = find_huggingface_cache()
    print(f"Cache directory: {cache_dir}")
    print(f"Cache exists: {cache_dir.exists()}")
    
    if cache_dir.exists():
        print(f"\n📁 Models in cache:")
        print("-" * 50)
        
        model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
        
        for model_dir in sorted(model_dirs):
            model_name = model_dir.name.replace("models--", "").replace("--", "/")
            print(f"📦 {model_name}")
            print(f"   📍 {model_dir}")
            
            # Check for config.json
            config_found = False
            for item in model_dir.rglob("config.json"):
                print(f"   📄 config.json: {item}")
                config_found = True
                break
            
            if not config_found:
                print(f"   ❌ No config.json found")
            
            print()

def main():
    """Main function to explore config file locations."""
    
    print("="*80)
    print("HUGGINGFACE CONFIG FILE LOCATION EXPLORER")
    print("="*80)
    
    # Models to explore
    models_to_check = [
        "CIRCL/vulnerability-description-generation-gpt2",
        "CIRCL/vulnerability-severity-classification-roberta-base"
    ]
    
    # Show overall cache structure
    show_cache_structure()
    
    # Explore each model
    for model_name in models_to_check:
        explore_model_files(model_name)
        read_config_file(model_name)
    
    # Compare access methods
    compare_configs()
    
    print(f"\n{'='*80}")
    print("SUMMARY: WHERE TO FIND CONFIG FILES")
    print("="*80)
    print(f"1. 🏠 Local cache: {find_huggingface_cache()}")
    print(f"2. 🌐 Online: https://huggingface.co/CIRCL/vulnerability-description-generation-gpt2/blob/main/config.json")
    print(f"3. 🐍 Python: AutoConfig.from_pretrained('model-name')")
    print(f"4. 📁 Direct access: Look in snapshots/[hash]/config.json")

if __name__ == "__main__":
    main() 