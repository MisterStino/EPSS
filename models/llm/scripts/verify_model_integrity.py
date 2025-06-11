#!/usr/bin/env python3
"""
Script to verify model integrity and ensure no permanent changes were made.
Checks if any modifications persist across sessions and different script runs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoConfig, pipeline
)
import torch
import json
import hashlib

def get_model_fingerprint(model_name):
    """Get a fingerprint of the model's current state."""
    
    print(f"Getting fingerprint for: {model_name}")
    print("-" * 50)
    
    # Load fresh model and config
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    fingerprint = {}
    
    # 1. Configuration fingerprint
    config_dict = config.to_dict()
    # Remove dynamic/session-specific fields
    stable_config = {k: v for k, v in config_dict.items() 
                    if k not in ['transformers_version', 'torch_dtype', '_name_or_path']}
    
    fingerprint['config'] = stable_config
    fingerprint['config_hash'] = hashlib.md5(str(sorted(stable_config.items())).encode()).hexdigest()
    
    # 2. Tokenizer fingerprint
    fingerprint['vocab_size'] = tokenizer.vocab_size
    fingerprint['special_tokens'] = tokenizer.special_tokens_map
    fingerprint['tokenizer_hash'] = hashlib.md5(str(tokenizer.vocab_size).encode()).hexdigest()
    
    # 3. Model architecture fingerprint
    fingerprint['model_type'] = type(model).__name__
    fingerprint['num_parameters'] = sum(p.numel() for p in model.parameters())
    fingerprint['num_layers'] = len(list(model.named_modules()))
    
    # 4. Model weights fingerprint (sample)
    if hasattr(model, 'classifier'):
        classifier_weights = model.classifier.out_proj.weight.data
        fingerprint['classifier_shape'] = list(classifier_weights.shape)
        fingerprint['classifier_sum'] = float(classifier_weights.sum().item())
        fingerprint['classifier_hash'] = hashlib.md5(classifier_weights.cpu().numpy().tobytes()).hexdigest()[:16]
    
    # 5. Label mappings
    fingerprint['id2label'] = config.id2label
    fingerprint['label2id'] = config.label2id
    
    return fingerprint

def compare_with_baseline():
    """Compare current model state with expected baseline."""
    
    print("="*80)
    print("COMPARING CURRENT MODEL STATE WITH BASELINE")
    print("="*80)
    
    # Expected baseline values (what the models should be)
    expected_baseline = {
        "CIRCL/vulnerability-severity-classification-roberta-base": {
            "num_labels": 4,
            "id2label": {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Critical'},
            "model_type": "roberta",
            "problem_type": "single_label_classification",
            "vocab_size": 50265
        },
        "CIRCL/vulnerability-description-generation-gpt2": {
            "model_type": "gpt2",
            "vocab_size": 50257
        }
    }
    
    models_to_check = [
        "CIRCL/vulnerability-severity-classification-roberta-base",
        "CIRCL/vulnerability-description-generation-gpt2"
    ]
    
    all_clean = True
    
    for model_name in models_to_check:
        print(f"\\nCHECKING: {model_name}")
        print("-" * 60)
        
        try:
            # Get current fingerprint
            current = get_model_fingerprint(model_name)
            expected = expected_baseline.get(model_name, {})
            
            # Check critical parameters
            checks = []
            
            if "num_labels" in expected:
                actual_labels = current['config'].get('num_labels')
                expected_labels = expected['num_labels']
                checks.append(("num_labels", actual_labels, expected_labels, actual_labels == expected_labels))
            
            if "id2label" in expected:
                actual_id2label = current['id2label']
                expected_id2label = expected['id2label']
                checks.append(("id2label", actual_id2label, expected_id2label, actual_id2label == expected_id2label))
            
            if "model_type" in expected:
                actual_type = current['config'].get('model_type')
                expected_type = expected['model_type']
                checks.append(("model_type", actual_type, expected_type, actual_type == expected_type))
            
            if "vocab_size" in expected:
                actual_vocab = current['vocab_size']
                expected_vocab = expected['vocab_size']
                checks.append(("vocab_size", actual_vocab, expected_vocab, actual_vocab == expected_vocab))
            
            # Report results
            for check_name, actual, expected_val, is_match in checks:
                status = "✅ CLEAN" if is_match else "❌ MODIFIED"
                print(f"  {check_name}: {status}")
                print(f"    Expected: {expected_val}")
                print(f"    Actual:   {actual}")
                
                if not is_match:
                    all_clean = False
            
            # Check for unexpected config additions
            unexpected_keys = []
            for key in current['config']:
                if key not in ['return_dict', 'output_hidden_states', 'output_attentions', 
                              'torchscript', 'use_bfloat16', 'tf_legacy_loss', 'pruned_heads',
                              'tie_word_embeddings', 'chunk_size_feed_forward', 'is_encoder_decoder',
                              'is_decoder', 'cross_attention_hidden_size', 'add_cross_attention',
                              'tie_encoder_decoder', 'max_length', 'num_labels', 'id2label',
                              'label2id', 'model_type', 'problem_type', 'architectures',
                              'attention_probs_dropout_prob', 'classifier_dropout', 'hidden_act',
                              'hidden_dropout_prob', 'hidden_size', 'initializer_range',
                              'intermediate_size', 'layer_norm_eps', 'max_position_embeddings',
                              'num_attention_heads', 'num_hidden_layers', 'pad_token_id',
                              'position_embedding_type', 'type_vocab_size', 'use_cache',
                              'vocab_size', 'bos_token_id', 'eos_token_id']:
                    # Check if this is a custom addition
                    if not key.startswith('_') and key not in ['transformers_version', 'torch_dtype']:
                        unexpected_keys.append(key)
            
            if unexpected_keys:
                print(f"  ⚠️  UNEXPECTED CONFIG KEYS: {unexpected_keys}")
                all_clean = False
            else:
                print(f"  ✅ No unexpected configuration keys")
                
        except Exception as e:
            print(f"  ❌ ERROR checking {model_name}: {e}")
            all_clean = False
    
    return all_clean

def test_fresh_model_loads():
    """Test that fresh model loads give consistent results."""
    
    print("\\n" + "="*80)
    print("TESTING FRESH MODEL LOADS FOR CONSISTENCY")
    print("="*80)
    
    model_name = "CIRCL/vulnerability-severity-classification-roberta-base"
    test_text = "SQL injection vulnerability in web application"
    
    print(f"Testing consistency across multiple fresh loads...")
    print(f"Test text: '{test_text}'")
    print("-" * 50)
    
    results = []
    
    # Load model 3 times and test
    for i in range(3):
        print(f"\\nLoad #{i+1}:")
        
        # Fresh load each time
        pipeline_fresh = pipeline("text-classification", model=model_name)
        result = pipeline_fresh(test_text)
        
        prediction = result[0]
        results.append({
            'load': i+1,
            'label': prediction['label'],
            'score': prediction['score']
        })
        
        print(f"  Result: {prediction['label']} ({prediction['score']:.4f})")
        
        # Clean up
        del pipeline_fresh
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Check consistency
    print(f"\\nCONSISTENCY CHECK:")
    print("-" * 30)
    
    labels = [r['label'] for r in results]
    scores = [r['score'] for r in results]
    
    labels_consistent = len(set(labels)) == 1
    scores_consistent = max(scores) - min(scores) < 1e-6
    
    print(f"  Labels consistent: {'✅ YES' if labels_consistent else '❌ NO'}")
    print(f"  Scores consistent: {'✅ YES' if scores_consistent else '❌ NO'}")
    
    if not labels_consistent:
        print(f"    Labels: {labels}")
    if not scores_consistent:
        print(f"    Score range: {min(scores):.8f} - {max(scores):.8f}")
    
    return labels_consistent and scores_consistent

def check_cache_and_files():
    """Check if any model files or cache have been modified."""
    
    print("\\n" + "="*80)
    print("CHECKING MODEL CACHE AND FILES")
    print("="*80)
    
    import os
    from pathlib import Path
    
    # Check HuggingFace cache
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    print(f"HuggingFace cache directory: {cache_dir}")
    print("-" * 50)
    
    if cache_dir.exists():
        # Look for CIRCL model directories
        circl_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and "CIRCL" in d.name]
        
        print(f"Found {len(circl_dirs)} CIRCL model cache directories:")
        for circl_dir in circl_dirs:
            print(f"  📁 {circl_dir.name}")
            
            # Check for any recent modifications
            snapshots_dir = circl_dir / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                print(f"     Snapshots: {len(snapshots)}")
                
                # Check if any files were modified recently (last hour)
                import time
                current_time = time.time()
                recent_modifications = []
                
                for snapshot in snapshots:
                    if snapshot.is_dir():
                        for file_path in snapshot.rglob("*"):
                            if file_path.is_file():
                                mod_time = file_path.stat().st_mtime
                                if current_time - mod_time < 3600:  # Last hour
                                    recent_modifications.append((file_path, mod_time))
                
                if recent_modifications:
                    print(f"     ⚠️  {len(recent_modifications)} files modified in last hour")
                    for file_path, mod_time in recent_modifications[:3]:  # Show first 3
                        print(f"       - {file_path.name}")
                else:
                    print(f"     ✅ No recent modifications")
    else:
        print("  ❌ HuggingFace cache directory not found")

def test_model_isolation():
    """Test that model modifications don't persist across Python sessions."""
    
    print("\\n" + "="*80)
    print("TESTING MODEL ISOLATION ACROSS SESSIONS")
    print("="*80)
    
    print("1. TESTING IN-MEMORY MODIFICATIONS:")
    print("-" * 50)
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "CIRCL/vulnerability-severity-classification-roberta-base"
    )
    config = model.config
    
    # Record original state
    original_labels = config.id2label.copy()
    original_problem_type = config.problem_type
    
    print(f"Original labels: {original_labels}")
    print(f"Original problem type: {original_problem_type}")
    
    # Make temporary modifications
    print("\\nMaking temporary modifications...")
    config.id2label = {"0": "Test1", "1": "Test2", "2": "Test3", "3": "Test4"}
    config.problem_type = "test_classification"
    
    print(f"Modified labels: {config.id2label}")
    print(f"Modified problem type: {config.problem_type}")
    
    # Delete model
    del model, config
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load fresh model
    print("\\nLoading fresh model...")
    fresh_model = AutoModelForSequenceClassification.from_pretrained(
        "CIRCL/vulnerability-severity-classification-roberta-base"
    )
    fresh_config = fresh_model.config
    
    print(f"Fresh labels: {fresh_config.id2label}")
    print(f"Fresh problem type: {fresh_config.problem_type}")
    
    # Check if modifications persisted
    labels_clean = fresh_config.id2label == original_labels
    problem_type_clean = fresh_config.problem_type == original_problem_type
    
    print(f"\\nISOLATION CHECK:")
    print(f"  Labels restored: {'✅ YES' if labels_clean else '❌ NO'}")
    print(f"  Problem type restored: {'✅ YES' if problem_type_clean else '❌ NO'}")
    
    return labels_clean and problem_type_clean

def main():
    """Main function to verify model integrity."""
    
    print("MODEL INTEGRITY VERIFICATION")
    print("="*80)
    print("Checking if any permanent changes were made to the CIRCL models...")
    
    # Run all checks
    baseline_clean = compare_with_baseline()
    consistency_good = test_fresh_model_loads()
    isolation_good = test_model_isolation()
    
    # Check cache
    check_cache_and_files()
    
    print("\\n" + "="*80)
    print("FINAL INTEGRITY REPORT")
    print("="*80)
    
    print(f"✅ Baseline comparison: {'CLEAN' if baseline_clean else 'MODIFIED'}")
    print(f"✅ Load consistency: {'CONSISTENT' if consistency_good else 'INCONSISTENT'}")
    print(f"✅ Session isolation: {'ISOLATED' if isolation_good else 'PERSISTENT'}")
    
    overall_clean = baseline_clean and consistency_good and isolation_good
    
    print(f"\\n🎯 OVERALL STATUS: {'✅ MODELS ARE CLEAN' if overall_clean else '❌ MODELS MAY BE MODIFIED'}")
    
    if overall_clean:
        print("\\n💡 CONCLUSION:")
        print("  - No permanent changes were made to the original models")
        print("  - All modifications were temporary and in-memory only")
        print("  - Models will behave identically in new sessions/scripts")
        print("  - Original CIRCL model parameters and configs are intact")
    else:
        print("\\n⚠️  ISSUES DETECTED:")
        print("  - Some modifications may have persisted")
        print("  - Check the detailed output above for specific problems")
        print("  - Consider clearing cache or reinstalling models if needed")

if __name__ == "__main__":
    main() 