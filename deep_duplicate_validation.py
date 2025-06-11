#!/usr/bin/env python3
"""
Deep Duplicate Validation - JSON Blob Content Analysis

Rigorous comparison of duplicate rows including deep parsing and comparison
of all JSON blob contents to ensure true identity.
"""

import pandas as pd
import json
import hashlib
import sys

def deep_json_comparison(json_str1, json_str2, field_name):
    """
    Deep comparison of two JSON strings after parsing
    Returns (are_identical, differences, parse_errors)
    """
    try:
        # Parse both JSON strings
        obj1 = json.loads(json_str1) if json_str1 and json_str1.strip() else None
        obj2 = json.loads(json_str2) if json_str2 and json_str2.strip() else None
        
        # Handle None cases
        if obj1 is None and obj2 is None:
            return True, "Both are None/empty", []
        if obj1 is None or obj2 is None:
            return False, f"One is None: {obj1} vs {obj2}", []
        
        # Deep comparison using normalized JSON
        # Convert both to normalized JSON strings (sorted keys, consistent formatting)
        norm1 = json.dumps(obj1, sort_keys=True, separators=(',', ':'))
        norm2 = json.dumps(obj2, sort_keys=True, separators=(',', ':'))
        
        if norm1 == norm2:
            return True, "Identical", []
        else:
            # Show first difference for debugging
            diff_info = f"Different content lengths: {len(norm1)} vs {len(norm2)}"
            return False, diff_info, []
            
    except json.JSONDecodeError as e1:
        try:
            obj2 = json.loads(json_str2)
            return False, f"Parse error in first JSON: {e1}", [str(e1)]
        except json.JSONDecodeError as e2:
            return False, f"Parse errors in both: {e1}, {e2}", [str(e1), str(e2)]
    except Exception as e:
        return False, f"Comparison error: {e}", [str(e)]

def calculate_json_hash(json_str, field_name):
    """Calculate a normalized hash of JSON content"""
    try:
        if not json_str or not json_str.strip():
            return "empty"
        
        obj = json.loads(json_str)
        # Create normalized JSON string (sorted keys, no whitespace)
        normalized = json.dumps(obj, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(normalized.encode()).hexdigest()
    except:
        return f"parse_error_{hash(json_str)}"

def analyze_duplicate_row_group(group_df, group_name):
    """
    Thoroughly analyze a group of duplicate rows
    """
    print(f"\n{'='*80}")
    print(f"DEEP ANALYSIS: {group_name}")
    print(f"Number of duplicate rows: {len(group_df)}")
    print(f"{'='*80}")
    
    json_columns = ['descriptions_json', 'metrics_json', 'weaknesses_json', 
                   'configurations_json', 'references_json']
    
    all_identical = True
    analysis_results = {}
    
    # First, compare scalar columns
    print(f"\n📊 SCALAR COLUMNS COMPARISON:")
    scalar_columns = [col for col in group_df.columns if col not in json_columns and col != 'composite_key']
    
    for col in scalar_columns:
        values = group_df[col].tolist()
        unique_values = list(set(str(v) for v in values))
        
        if len(unique_values) == 1:
            print(f"  ✅ {col:25} | Identical: '{unique_values[0][:50]}'")
        else:
            print(f"  ❌ {col:25} | Different: {unique_values}")
            all_identical = False
    
    # Now deep-dive into JSON columns
    print(f"\n📦 JSON BLOB DEEP COMPARISON:")
    
    for json_col in json_columns:
        print(f"\n🔍 Analyzing {json_col}:")
        
        json_values = group_df[json_col].tolist()
        
        # Calculate hashes for quick comparison
        hashes = [calculate_json_hash(val, json_col) for val in json_values]
        unique_hashes = list(set(hashes))
        
        if len(unique_hashes) == 1:
            print(f"    ✅ Content hashes identical: {unique_hashes[0]}")
        else:
            print(f"    ❌ Different content hashes found: {len(unique_hashes)} variants")
            all_identical = False
            
            # Deep comparison between first two different values
            if len(json_values) >= 2:
                val1, val2 = json_values[0], json_values[1]
                is_identical, differences, errors = deep_json_comparison(val1, val2, json_col)
                
                print(f"    📋 Deep comparison result: {'Identical' if is_identical else 'Different'}")
                if not is_identical:
                    print(f"    📋 Differences: {differences[:200]}...")
                if errors:
                    print(f"    ⚠️  Parse errors: {errors}")
        
        # Sample the actual JSON content structure
        if json_values and json_values[0]:
            try:
                sample_obj = json.loads(json_values[0])
                if isinstance(sample_obj, dict):
                    print(f"    📋 Structure: dict with {len(sample_obj)} keys: {list(sample_obj.keys())[:5]}")
                elif isinstance(sample_obj, list):
                    print(f"    📋 Structure: array with {len(sample_obj)} elements")
                    if sample_obj and isinstance(sample_obj[0], dict):
                        print(f"    📋 First element keys: {list(sample_obj[0].keys())[:5]}")
                else:
                    print(f"    📋 Structure: {type(sample_obj).__name__}")
            except:
                print(f"    ⚠️  Could not parse JSON for structure analysis")
        
        analysis_results[json_col] = {
            'unique_hashes': len(unique_hashes),
            'identical': len(unique_hashes) == 1
        }
    
    print(f"\n🎯 GROUP SUMMARY:")
    if all_identical:
        print("    ✅ ALL COLUMNS AND JSON BLOBS ARE TRULY IDENTICAL")
        print("    → This confirms perfect duplicates - same data repeated")
    else:
        print("    ❌ DIFFERENCES FOUND - Not perfect duplicates")
        print("    → These might represent different states with same timestamp")
    
    return all_identical, analysis_results

def run_comprehensive_duplicate_analysis():
    """Run comprehensive analysis of duplicate rows"""
    print("🚀 COMPREHENSIVE DUPLICATE ANALYSIS")
    print("="*80)
    print("Goal: Deep validation of duplicate row content including JSON blob parsing")
    print("="*80)
    
    # Load data
    df = pd.read_csv("catalogs_processed/master_cve_timeseries_20250610.csv")
    
    # Find duplicates
    df['composite_key'] = df['cve_id'] + '|' + df['reconstruction_timestamp']
    duplicate_mask = df.duplicated(subset=['composite_key'], keep=False)
    duplicate_rows = df[duplicate_mask].copy()
    
    if len(duplicate_rows) == 0:
        print("✅ No duplicates found - analysis complete")
        return
    
    # Group duplicates
    duplicate_groups = duplicate_rows.groupby('composite_key')
    
    print(f"📊 Found {len(duplicate_groups)} duplicate composite keys")
    print(f"📊 Total duplicate rows: {len(duplicate_rows)}")
    
    # Analyze sample of duplicate groups
    sample_size = min(10, len(duplicate_groups))
    sample_groups = list(duplicate_groups)[:sample_size]
    
    print(f"\n🔍 ANALYZING {sample_size} SAMPLE DUPLICATE GROUPS:")
    
    perfect_duplicates = 0
    imperfect_duplicates = 0
    
    for i, (composite_key, group_df) in enumerate(sample_groups):
        cve_id, timestamp = composite_key.split('|', 1)
        group_name = f"{cve_id} at {timestamp}"
        
        is_perfect, results = analyze_duplicate_row_group(group_df, group_name)
        
        if is_perfect:
            perfect_duplicates += 1
        else:
            imperfect_duplicates += 1
    
    # Summary statistics
    print(f"\n🎯 FINAL DEEP ANALYSIS RESULTS:")
    print(f"="*50)
    print(f"Sample groups analyzed: {sample_size}")
    print(f"Perfect duplicates (identical in ALL aspects): {perfect_duplicates}")
    print(f"Imperfect duplicates (some differences): {imperfect_duplicates}")
    
    if perfect_duplicates > 0:
        print(f"\n❌ CONFIRMED: Perfect duplicates found")
        print(f"   → Same CVE+timestamp with identical JSON content")
        print(f"   → This indicates a data generation/processing bug")
    
    if imperfect_duplicates > 0:
        print(f"\n⚠️  ATTENTION: Imperfect duplicates found")
        print(f"   → Same CVE+timestamp but different content")
        print(f"   → This might indicate timestamp precision issues")
    
    # Quick validation of larger sample
    print(f"\n🔍 QUICK HASH VALIDATION OF ALL DUPLICATES:")
    validation_sample = list(duplicate_groups)[:100]  # Larger sample
    
    perfect_count = 0
    for composite_key, group_df in validation_sample:
        # Quick hash check
        json_columns = ['descriptions_json', 'metrics_json', 'weaknesses_json', 
                       'configurations_json', 'references_json']
        
        all_hashes_identical = True
        for col in json_columns:
            hashes = [calculate_json_hash(val, col) for val in group_df[col].tolist()]
            if len(set(hashes)) > 1:
                all_hashes_identical = False
                break
        
        if all_hashes_identical:
            perfect_count += 1
    
    print(f"Perfect duplicates in {len(validation_sample)} sample: {perfect_count}")
    print(f"Percentage of perfect duplicates: {(perfect_count/len(validation_sample))*100:.1f}%")

if __name__ == "__main__":
    run_comprehensive_duplicate_analysis() 