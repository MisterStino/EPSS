#!/usr/bin/env python3
"""
Debug script to verify feature categorization and test EPSS inclusion fix.
"""

import json
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.types as patypes
from pathlib import Path
import re

def analyze_current_features():
    """Analyze current feature categorization that excludes EPSS"""
    
    arrow_path = Path("work/epss_stage1.arrow")
    if not arrow_path.exists():
        print(f"❌ Arrow file not found: {arrow_path}")
        return
        
    # Load schema and vocab
    file = ipc.open_file(pa.memory_map(str(arrow_path), "r"))
    schema = file.schema
    
    vocab_path = arrow_path.parent / "vocab.json"
    with vocab_path.open() as fh:
        vocab = json.load(fh)
    
    # Current categorization logic (PROBLEMATIC)
    BOOL_RE = re.compile(r"^(has_|is_)")
    names = [f.name for f in schema]
    
    cat_cols = list(vocab.keys())
    bool_cols = [n for n in names if BOOL_RE.match(n) and n not in cat_cols]
    flag_cols = ["flag_train", "flag_val", "flag_test"]
    
    # PROBLEM: EPSS is in reserved set
    reserved = set(cat_cols + bool_cols + flag_cols + ["cve", "date", "epss"])
    num_cols = [
        f.name for f in schema
        if f.name not in reserved and
           (patypes.is_integer(f.type) or patypes.is_floating(f.type))
    ]
    
    print("🔍 CURRENT FEATURE CATEGORIZATION (PROBLEMATIC)")
    print("=" * 60)
    print(f"📊 Numeric features ({len(num_cols)}): {num_cols[:10]}{'...' if len(num_cols) > 10 else ''}")
    print(f"🔘 Boolean features ({len(bool_cols)}): {bool_cols[:5]}{'...' if len(bool_cols) > 5 else ''}")
    print(f"🏷️  Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"🚫 Reserved (excluded): {sorted(reserved)}")
    print()
    print("❌ PROBLEM: 'epss' is in reserved set - NOT fed as input to model!")
    print("❌ Model only sees static metadata, no EPSS context")
    print("❌ This explains why predictions are always ~baseline")
    
    return {
        'num_cols': num_cols,
        'bool_cols': bool_cols, 
        'cat_cols': cat_cols,
        'reserved': reserved,
        'schema': schema
    }

def propose_fix():
    """Propose fixed feature categorization that includes EPSS as input"""
    
    print("\n🔧 PROPOSED FIX")
    print("=" * 60)
    print("1. INCLUDE 'epss' in numeric features (input to model)")
    print("2. STILL use 'epss' as target Y (for loss computation)")
    print("3. Model gets EPSS context + can learn temporal patterns")
    print()
    
    # Show the fixed logic
    print("FIXED CATEGORIZATION LOGIC:")
    print("```python")
    print("# FIXED: Remove 'epss' from reserved set")
    print("reserved = set(cat_cols + bool_cols + flag_cols + ['cve', 'date'])")
    print("# Now EPSS will be included in num_cols")
    print("num_cols = [f.name for f in schema")
    print("            if f.name not in reserved and")
    print("               (patypes.is_integer(f.type) or patypes.is_floating(f.type))]")
    print("```")
    print()
    print("✅ RESULT: Model receives EPSS as input feature")
    print("✅ RESULT: Model can learn from EPSS history")
    print("✅ RESULT: Model can self-correct over time")
    print("✅ RESULT: Predictions should improve dramatically")

def verify_epss_values():
    """Verify EPSS values in the arrow file"""
    
    arrow_path = Path("work/epss_stage1.arrow")
    if not arrow_path.exists():
        print(f"❌ Arrow file not found: {arrow_path}")
        return
        
    print("\n📊 EPSS VALUES IN ARROW FILE")
    print("=" * 60)
    
    reader = ipc.open_file(pa.memory_map(str(arrow_path), "r"))
    
    # Sample first batch to check EPSS values
    batch = reader.get_batch(0)
    epss_col_idx = None
    for i, field in enumerate(reader.schema):
        if field.name == "epss":
            epss_col_idx = i
            break
    
    if epss_col_idx is None:
        print("❌ EPSS column not found in schema!")
        return
        
    epss_values = batch.columns[epss_col_idx]
    sample_values = [epss_values[i].as_py() for i in range(min(10, len(epss_values)))]
    
    print(f"✅ EPSS column found at index {epss_col_idx}")
    print(f"📈 Sample EPSS values (log space): {sample_values}")
    print(f"📈 These are log(epss + 1e-6) transformed values")
    print(f"📈 Original EPSS ≈ exp(log_value) - 1e-6")
    
    # Show what original values would be
    import math
    orig_sample = [max(0, math.exp(v) - 1e-6) for v in sample_values if v is not None]
    print(f"📈 Corresponding original EPSS: {[f'{v:.6f}' for v in orig_sample]}")

if __name__ == "__main__":
    print("🚨 EPSS FEATURE INCLUSION DEBUG")
    print("=" * 60)
    
    current = analyze_current_features()
    propose_fix()
    verify_epss_values()
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY")
    print("=" * 60)
    print("ROOT CAUSE: EPSS excluded from model input features")
    print("IMPACT: Model predicts baseline (~0) without EPSS context") 
    print("FIX: Include EPSS in numeric features while keeping as target")
    print("EXPECTED RESULT: Dramatically improved predictions") 