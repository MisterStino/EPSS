#!/usr/bin/env python3
"""
Comprehensive Embedding Validator for CUDA Assertion Debugging
This script replicates the exact LSTM training setup and adds extensive validation
to identify where out-of-bounds categorical indices are introduced.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import duckdb
from pathlib import Path
from tqdm import tqdm
import json
import traceback

class EmbeddingValidator:
    """Comprehensive validator for embedding layer inputs"""
    
    def __init__(self, vocab_sizes):
        self.vocab_sizes = vocab_sizes
        self.validation_log = []
        
    def validate_tensor(self, tensor, vocab_idx, step_name, extra_info=""):
        """Validate a categorical tensor against vocabulary bounds"""
        if not isinstance(tensor, torch.Tensor):
            return True
            
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        vocab_size = self.vocab_sizes[vocab_idx]
        
        log_entry = {
            'step': step_name,
            'vocab_idx': vocab_idx,
            'vocab_size': vocab_size,
            'tensor_shape': list(tensor.shape),
            'tensor_dtype': str(tensor.dtype),
            'min_val': min_val,
            'max_val': max_val,
            'valid': max_val < vocab_size,
            'extra_info': extra_info
        }
        
        self.validation_log.append(log_entry)
        
        if max_val >= vocab_size:
            print(f"🚨 VALIDATION FAILED at {step_name}:")
            print(f"   Vocab {vocab_idx}: size={vocab_size}, tensor_max={max_val}")
            print(f"   Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            print(f"   Extra info: {extra_info}")
            
            # Find all out-of-bounds indices
            oob_mask = tensor >= vocab_size
            oob_indices = torch.where(oob_mask)
            print(f"   Out-of-bounds positions: {len(oob_indices[0])} total")
            if len(oob_indices[0]) > 0:
                print(f"   First few OOB values: {tensor[oob_mask][:10].tolist()}")
            
            return False
            
        print(f"✅ VALIDATION PASSED at {step_name}: vocab_{vocab_idx} max={max_val} < {vocab_size}")
        return True
    
    def save_log(self, filename="embedding_validation_log.json"):
        """Save validation log to file"""
        with open(filename, 'w') as f:
            json.dump(self.validation_log, f, indent=2)
        print(f"Validation log saved to {filename}")

def comprehensive_embedding_validation():
    print("=" * 80)
    print("COMPREHENSIVE EMBEDDING VALIDATION")
    print("=" * 80)
    
    # Configuration
    local_execution = True
    LOCAL_CONFIG = {
        'batch_size': 64,
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'sequence_length': 30,
        'data_path': 'data/full_db/v1/data/minimal_v1_timeseries_sample.parquet'
    }
    
    config = LOCAL_CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load and preprocess data exactly as main script
    print("\n[STEP 1] Loading data...")
    PARQUET_DIR = Path(config['data_path'])
    if PARQUET_DIR.is_dir():
        parquet_glob = str(PARQUET_DIR / "*.parquet")
    else:
        parquet_glob = str(PARQUET_DIR)
    
    DROP_COLS = [
        'cve_id', 'date', 'epss_score', 'epss_percentile', 
        'cvss_base_score', 'cvss_exploitability_score', 'cvss_impact_score'
    ]
    
    try:
        df = duckdb.execute(f"""
            SELECT * FROM read_parquet('{parquet_glob}')
        """).df()
        print(f"Loaded {len(df)} rows")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Drop unnecessary columns
    df = df.drop(columns=[col for col in DROP_COLS if col in df.columns])
    
    # Handle boolean columns
    bool_cols = ['has_exploit', 'has_github_poc', 'in_cisa_kev']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)
    
    print(f"Data shape after preprocessing: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Create vocabularies and validate
    print("\n[STEP 2] Creating vocabularies...")
    categorical_cols = ['dominant_event_type', 'primary_cwe_category']
    vocabularies = {}
    vocab_sizes = {}
    
    for col in categorical_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            vocab = {'UNK': 0}
            vocab.update({val: i+1 for i, val in enumerate(unique_vals)})
            vocabularies[col] = vocab
            vocab_sizes[col] = len(vocab)
            print(f"Vocabulary for {col}: size={len(vocab)}")
            print(f"  Values: {list(vocab.keys())[:10]}...")  # Show first 10
    
    # Initialize validator
    validator = EmbeddingValidator([vocab_sizes.get('dominant_event_type', 3), 
                                   vocab_sizes.get('primary_cwe_category', 10)])
    
    # 3. Temporal splitting
    print("\n[STEP 3] Temporal splitting...")
    df['date'] = pd.to_datetime(df['date'] if 'date' in df.columns else df.index)
    df = df.sort_values('date')
    
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 4. Feature engineering and categorical encoding
    print("\n[STEP 4] Feature engineering...")
    
    def encode_categorical_features(df, vocabularies, validator, split_name):
        """Encode categorical features with validation"""
        df_encoded = df.copy()
        
        for i, col in enumerate(categorical_cols):
            if col in df.columns:
                vocab = vocabularies[col]
                # Map values to indices
                df_encoded[col] = df[col].fillna('UNK').map(vocab).fillna(0)
                
                # Validate immediately after encoding
                tensor_vals = torch.tensor(df_encoded[col].values, dtype=torch.long)
                validator.validate_tensor(
                    tensor_vals, i, f"categorical_encoding_{split_name}_{col}",
                    f"After mapping {col} values to indices"
                )
        
        return df_encoded
    
    train_df_encoded = encode_categorical_features(train_df, vocabularies, validator, "train")
    val_df_encoded = encode_categorical_features(val_df, vocabularies, validator, "val")
    
    # 5. Numerical feature scaling
    print("\n[STEP 5] Scaling numerical features...")
    numerical_cols = [col for col in train_df_encoded.columns 
                     if col not in categorical_cols + ['date', 'cve_id']]
    
    scaler = StandardScaler()
    if numerical_cols:
        train_df_encoded[numerical_cols] = scaler.fit_transform(train_df_encoded[numerical_cols])
        val_df_encoded[numerical_cols] = scaler.transform(val_df_encoded[numerical_cols])
    
    # 6. Create dataset with validation
    print("\n[STEP 6] Creating dataset...")
    
    class ValidatedCVEDataset(Dataset):
        def __init__(self, df, sequence_length, validator, split_name):
            self.df = df.reset_index(drop=True)
            self.sequence_length = sequence_length
            self.validator = validator
            self.split_name = split_name
            
            # Separate categorical and numerical columns
            self.categorical_cols = ['dominant_event_type', 'primary_cwe_category']
            self.numerical_cols = [col for col in df.columns 
                                 if col not in self.categorical_cols + ['date', 'cve_id']]
            
            print(f"Dataset {split_name}: {len(df)} samples")
            print(f"  Categorical cols: {self.categorical_cols}")
            print(f"  Numerical cols: {len(self.numerical_cols)} columns")
        
        def __len__(self):
            return max(0, len(self.df) - self.sequence_length + 1)
        
        def __getitem__(self, idx):
            # Get sequence
            end_idx = idx + self.sequence_length
            sequence_df = self.df.iloc[idx:end_idx]
            
            # Extract categorical features
            categorical_features = []
            for i, col in enumerate(self.categorical_cols):
                if col in sequence_df.columns:
                    cat_tensor = torch.tensor(sequence_df[col].values, dtype=torch.long)
                    
                    # Validate before returning
                    self.validator.validate_tensor(
                        cat_tensor, i, f"dataset_getitem_{self.split_name}",
                        f"Sample {idx}, column {col}, sequence {idx}-{end_idx}"
                    )
                    
                    categorical_features.append(cat_tensor)
            
            # Extract numerical features
            if self.numerical_cols:
                numerical_tensor = torch.tensor(
                    sequence_df[self.numerical_cols].values, 
                    dtype=torch.float32
                )
            else:
                numerical_tensor = torch.zeros((self.sequence_length, 1), dtype=torch.float32)
            
            return {
                'categorical': categorical_features,
                'numerical': numerical_tensor,
                'idx': idx
            }
    
    # Create datasets
    train_dataset = ValidatedCVEDataset(train_df_encoded, config['sequence_length'], validator, "train")
    val_dataset = ValidatedCVEDataset(val_df_encoded, config['sequence_length'], validator, "val")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # 7. Create data loaders with validation
    print("\n[STEP 7] Creating data loaders...")
    
    def validated_collate_fn(batch):
        """Custom collate function with validation"""
        batch_size = len(batch)
        seq_len = batch[0]['numerical'].shape[0]
        num_features = batch[0]['numerical'].shape[1]
        
        # Stack numerical features
        numerical_batch = torch.stack([item['numerical'] for item in batch])
        
        # Stack categorical features
        categorical_batch = []
        for cat_idx in range(len(batch[0]['categorical'])):
            cat_tensors = [item['categorical'][cat_idx] for item in batch]
            cat_batch = torch.stack(cat_tensors)
            
            # Validate batch
            validator.validate_tensor(
                cat_batch, cat_idx, f"collate_fn_batch",
                f"Batch size {batch_size}, cat_idx {cat_idx}"
            )
            
            categorical_batch.append(cat_batch)
        
        return {
            'categorical': categorical_batch,
            'numerical': numerical_batch,
            'indices': [item['idx'] for item in batch]
        }
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=validated_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        collate_fn=validated_collate_fn
    )
    
    # 8. Create model with embedding validation hooks
    print("\n[STEP 8] Creating model with validation hooks...")
    
    class ValidatedLSTMModel(nn.Module):
        def __init__(self, vocab_sizes, numerical_input_size, hidden_size, num_layers, dropout, validator):
            super().__init__()
            self.validator = validator
            self.vocab_sizes = vocab_sizes
            
            # Embedding layers
            self.embeddings = nn.ModuleList([
                nn.Embedding(vocab_size, 16, padding_idx=0) 
                for vocab_size in vocab_sizes
            ])
            
            # Calculate total input size
            embedding_size = len(vocab_sizes) * 16
            total_input_size = embedding_size + numerical_input_size
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=total_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            # Output layer
            self.output_layer = nn.Linear(hidden_size, 1)
            
            print(f"Model created:")
            print(f"  Vocab sizes: {vocab_sizes}")
            print(f"  Embedding dims: {[16] * len(vocab_sizes)}")
            print(f"  Total input size: {total_input_size}")
            print(f"  Hidden size: {hidden_size}")
        
        def forward(self, categorical_features, numerical_features):
            batch_size, seq_len = numerical_features.shape[:2]
            
            # Process categorical features with validation
            embeddings = []
            for i, (cat_tensor, embedding_layer) in enumerate(zip(categorical_features, self.embeddings)):
                # Validate before embedding
                self.validator.validate_tensor(
                    cat_tensor, i, f"model_forward_pre_embedding",
                    f"Before embedding layer {i}, batch_size={batch_size}"
                )
                
                # Apply embedding
                try:
                    embedded = embedding_layer(cat_tensor)
                    embeddings.append(embedded)
                    print(f"✅ Embedding {i} successful: {cat_tensor.shape} -> {embedded.shape}")
                except Exception as e:
                    print(f"🚨 EMBEDDING FAILED at layer {i}:")
                    print(f"   Input shape: {cat_tensor.shape}")
                    print(f"   Input range: [{cat_tensor.min()}, {cat_tensor.max()}]")
                    print(f"   Vocab size: {self.vocab_sizes[i]}")
                    print(f"   Error: {e}")
                    raise
            
            # Concatenate all features
            all_embeddings = torch.cat(embeddings, dim=-1)
            combined_features = torch.cat([all_embeddings, numerical_features], dim=-1)
            
            # LSTM forward pass
            lstm_out, _ = self.lstm(combined_features)
            
            # Output
            output = self.output_layer(lstm_out[:, -1, :])
            return output
    
    # Initialize model
    model = ValidatedLSTMModel(
        vocab_sizes=[vocab_sizes.get('dominant_event_type', 3), 
                    vocab_sizes.get('primary_cwe_category', 10)],
        numerical_input_size=len(numerical_cols) if numerical_cols else 1,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        validator=validator
    ).to(device)
    
    print(f"Model moved to device: {device}")
    
    # 9. Test with a few batches
    print("\n[STEP 9] Testing with training batches...")
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 3:  # Test first 3 batches
                break
                
            print(f"\n--- Testing Batch {batch_idx + 1} ---")
            
            # Move to device with validation
            categorical_features = []
            for i, cat_tensor in enumerate(batch['categorical']):
                # Validate before device transfer
                validator.validate_tensor(
                    cat_tensor, i, f"pre_device_transfer_batch_{batch_idx}",
                    f"Before moving to {device}"
                )
                
                cat_tensor_gpu = cat_tensor.to(device)
                
                # Validate after device transfer
                validator.validate_tensor(
                    cat_tensor_gpu, i, f"post_device_transfer_batch_{batch_idx}",
                    f"After moving to {device}"
                )
                
                categorical_features.append(cat_tensor_gpu)
            
            numerical_features = batch['numerical'].to(device)
            
            # Forward pass
            try:
                output = model(categorical_features, numerical_features)
                print(f"✅ Batch {batch_idx + 1} forward pass successful!")
                print(f"   Output shape: {output.shape}")
            except Exception as e:
                print(f"🚨 FORWARD PASS FAILED on batch {batch_idx + 1}:")
                print(f"   Error: {e}")
                print(f"   Traceback:")
                traceback.print_exc()
                break
    
    # 10. Save validation results
    print("\n[STEP 10] Saving validation results...")
    validator.save_log("comprehensive_embedding_validation.json")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    total_validations = len(validator.validation_log)
    failed_validations = sum(1 for log in validator.validation_log if not log['valid'])
    
    print(f"Total validations: {total_validations}")
    print(f"Failed validations: {failed_validations}")
    print(f"Success rate: {(total_validations - failed_validations) / total_validations * 100:.2f}%")
    
    if failed_validations > 0:
        print("\nFailed validation details:")
        for log in validator.validation_log:
            if not log['valid']:
                print(f"  {log['step']}: vocab_{log['vocab_idx']} max={log['max_val']} >= {log['vocab_size']}")
    
    print(f"\nDetailed log saved to: comprehensive_embedding_validation.json")

if __name__ == "__main__":
    try:
        comprehensive_embedding_validation()
    except Exception as e:
        print(f"Script failed with error: {e}")
        traceback.print_exc() 