#!/usr/bin/env python
"""
Generate missing vocab.json file by examining the Arrow file.
This script extracts categorical columns and creates the vocabulary mapping.
"""

import json
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
from pathlib import Path
import sys
import os

def find_project_root():
    """Find the project root directory containing ml_pipeline."""
    current = Path.cwd()
    
    # Try current directory first
    if (current / "ml_pipeline").exists():
        return current
    
    # Try parent directories
    for parent in current.parents:
        if (parent / "ml_pipeline").exists():
            return parent
    
    # Try common paths
    for path in ["/notebooks/EPSS", "/notebooks", Path.home() / "EPSS"]:
        path = Path(path)
        if path.exists() and (path / "ml_pipeline").exists():
            return path
    
    raise FileNotFoundError("Could not find project root with ml_pipeline directory")

def find_arrow_file(project_root):
    """Find the Arrow file in various possible locations."""
    possible_paths = [
        project_root / "ml_pipeline" / "work" / "epss_stage1.arrow",
        project_root / "ml_pipeline" / "data_prep" / "work" / "epss_stage1.arrow",
        project_root / "work" / "epss_stage1.arrow",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError(f"Could not find epss_stage1.arrow in any of: {possible_paths}")

def extract_categorical_columns(arrow_path):
    """Extract categorical columns and their unique values from Arrow file."""
    print(f"Examining Arrow file: {arrow_path}")
    
    # Open Arrow file
    with ipc.open_file(pa.memory_map(str(arrow_path), "r")) as reader:
        schema = reader.schema
        
        # Identify likely categorical columns
        categorical_columns = []
        for field in schema:
            field_name = field.name
            # Skip known non-categorical columns
            if field_name in ['cve', 'date', 'epss_target', 'epss_input', 'flag_train', 'flag_val', 'flag_test']:
                continue
            
            # Check if it's a string/categorical type or low-cardinality integer
            if (pa.types.is_string(field.type) or 
                pa.types.is_large_string(field.type) or
                (pa.types.is_integer(field.type) and not field_name.startswith(('epss', 'has_', 'is_')))):
                categorical_columns.append(field_name)
        
        print(f"Found potential categorical columns: {categorical_columns}")
        
        # Extract unique values for each categorical column
        vocab = {}
        
        # Sample batches to find unique values
        unique_values = {col: set() for col in categorical_columns}
        
        print("Scanning batches to extract unique values...")
        for batch_idx in range(min(reader.num_record_batches, 100)):  # Sample first 100 batches
            batch = reader.get_batch(batch_idx)
            
            for col in categorical_columns:
                if col in [f.name for f in batch.schema]:
                    col_data = batch.column(col)
                    # Convert to Python and add to set
                    values = col_data.to_pylist()
                    unique_values[col].update(v for v in values if v is not None)
            
            if batch_idx % 10 == 0:
                print(f"  Processed batch {batch_idx}/{min(reader.num_record_batches, 100)}")
        
        # Create vocabulary mapping (value -> id)
        for col in categorical_columns:
            values = sorted(list(unique_values[col]))
            # Reserve 0 for unknown/padding
            vocab[col] = {str(value): idx + 1 for idx, value in enumerate(values)}
            vocab[col]["<UNK>"] = 0  # Unknown token
            print(f"  {col}: {len(values)} unique values")
    
    return vocab

def main():
    """Main function to generate vocab.json."""
    print("="*60)
    print("GENERATING MISSING VOCAB.JSON")
    print("="*60)
    
    try:
        # Find project root
        project_root = find_project_root()
        print(f"Project root: {project_root}")
        
        # Find Arrow file
        arrow_path = find_arrow_file(project_root)
        print(f"Arrow file: {arrow_path}")
        
        # Extract vocabulary
        vocab = extract_categorical_columns(arrow_path)
        
        # Determine output path
        work_dir = project_root / "ml_pipeline" / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = work_dir / "vocab.json"
        
        # Save vocabulary
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f, indent=2)
        
        print(f"\n✓ Vocabulary saved to: {vocab_path}")
        print(f"✓ Categorical columns: {len(vocab)}")
        for col, values in vocab.items():
            print(f"  {col}: {len(values)} values")
        
        # Also copy to data_prep/work if it exists
        data_prep_work = project_root / "ml_pipeline" / "data_prep" / "work"
        if data_prep_work.exists():
            data_prep_vocab = data_prep_work / "vocab.json"
            with open(data_prep_vocab, 'w') as f:
                json.dump(vocab, f, indent=2)
            print(f"✓ Also saved to: {data_prep_vocab}")
        
        print("\n🎉 vocab.json generated successfully!")
        print("You can now run the hyperparameter optimization.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project directory")
        print("2. Ensure epss_stage1.arrow exists")
        print("3. Check file permissions")
        sys.exit(1)

if __name__ == "__main__":
    main() 