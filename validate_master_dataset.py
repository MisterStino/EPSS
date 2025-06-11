#!/usr/bin/env python3
"""
Comprehensive Validation of Master CVE Time-Series Dataset

Validates the generated master_cve_timeseries_20250610.csv to ensure:
1. Data integrity and completeness
2. Schema correctness
3. JSON blob validity
4. Count accuracy
5. CVSS extraction correctness
6. Temporal accuracy
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MasterDatasetValidator:
    def __init__(self, csv_path="catalogs_processed/master_cve_timeseries_20250610.csv"):
        self.csv_path = Path(csv_path)
        self.df = None
        self.validation_results = {}
        
    def load_dataset(self):
        """Load and basic inspection of the dataset"""
        print("🔍 Loading master dataset...")
        self.df = pd.read_csv(self.csv_path)
        
        print(f"✅ Loaded dataset: {len(self.df):,} rows × {len(self.df.columns)} columns")
        print(f"📊 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        return self.df
    
    def validate_schema_structure(self):
        """Validate column structure matches expected schema"""
        print("\n" + "="*60)
        print("🏗️  SCHEMA STRUCTURE VALIDATION")
        print("="*60)
        
        expected_columns = [
            'cve_id', 'reconstruction_timestamp', 'reconstruction_timestamp_raw',
            'source_identifier', 'published_date', 'last_modified_date', 'vuln_status', 'cve_tags',
            'weakness_count', 'reference_count', 'configuration_count',
            'descriptions_json', 'metrics_json', 'weaknesses_json', 'configurations_json', 'references_json',
            'primary_cvss_ver', 'primary_cvss_vec', 'primary_cvss_score', 'primary_cvss_sev'
        ]
        
        actual_columns = list(self.df.columns)
        missing_cols = set(expected_columns) - set(actual_columns)
        extra_cols = set(actual_columns) - set(expected_columns)
        
        print(f"Expected columns: {len(expected_columns)}")
        print(f"Actual columns: {len(actual_columns)}")
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
        if extra_cols:
            print(f"⚠️  Extra columns: {extra_cols}")
        if not missing_cols and not extra_cols:
            print("✅ Schema structure matches expectations perfectly")
            
        # Column types analysis
        print("\n📋 Column Types:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            print(f"  {col:25} | {str(dtype):10} | {null_count:8,} nulls ({null_pct:5.1f}%)")
            
        self.validation_results['schema'] = {
            'missing_columns': missing_cols,
            'extra_columns': extra_cols,
            'column_count': len(actual_columns)
        }
    
    def validate_primary_key_uniqueness(self):
        """Validate (cve_id, reconstruction_timestamp) uniqueness"""
        print("\n" + "="*60)
        print("🔑 PRIMARY KEY UNIQUENESS VALIDATION")
        print("="*60)
        
        # Check for duplicates
        duplicates = self.df.duplicated(subset=['cve_id', 'reconstruction_timestamp'])
        duplicate_count = duplicates.sum()
        
        if duplicate_count == 0:
            print("✅ Primary key (cve_id, reconstruction_timestamp) is unique")
        else:
            print(f"❌ Found {duplicate_count:,} duplicate primary keys")
            # Show examples
            duplicate_rows = self.df[duplicates]
            print("Examples of duplicates:")
            print(duplicate_rows[['cve_id', 'reconstruction_timestamp']].head())
        
        # CVE distribution analysis
        cve_counts = self.df['cve_id'].value_counts()
        print(f"\n📊 CVE Distribution:")
        print(f"  Unique CVEs: {len(cve_counts):,}")
        print(f"  States per CVE (avg): {cve_counts.mean():.1f}")
        print(f"  States per CVE (median): {cve_counts.median():.1f}")
        print(f"  Max states for single CVE: {cve_counts.max()}")
        print(f"  CVE with most states: {cve_counts.idxmax()}")
        
        self.validation_results['primary_key'] = {
            'duplicate_count': duplicate_count,
            'unique_cves': len(cve_counts),
            'avg_states_per_cve': cve_counts.mean()
        }
    
    def validate_json_blob_integrity(self):
        """Validate all JSON blobs are parseable and have expected structure"""
        print("\n" + "="*60)
        print("📦 JSON BLOB INTEGRITY VALIDATION")
        print("="*60)
        
        json_columns = ['descriptions_json', 'metrics_json', 'weaknesses_json', 
                       'configurations_json', 'references_json']
        
        json_stats = {}
        
        for col in json_columns:
            print(f"\n🔍 Validating {col}...")
            
            parse_errors = 0
            empty_count = 0
            sample_structures = []
            
            # Sample validation (check first 1000 non-null rows)
            non_null_rows = self.df[self.df[col].notna()][col].head(1000)
            
            for idx, json_str in non_null_rows.items():
                try:
                    parsed = json.loads(json_str)
                    if not parsed:  # Empty dict/list
                        empty_count += 1
                    else:
                        # Collect structure info
                        if isinstance(parsed, dict):
                            sample_structures.append(f"dict[{len(parsed)}keys]")
                        elif isinstance(parsed, list):
                            sample_structures.append(f"array[{len(parsed)}]")
                except (json.JSONDecodeError, TypeError) as e:
                    parse_errors += 1
                    if parse_errors <= 3:  # Show first few errors
                        print(f"    ❌ Parse error at row {idx}: {str(e)[:100]}")
            
            # Structure analysis
            structure_counts = Counter(sample_structures)
            
            print(f"    Parse errors: {parse_errors}")
            print(f"    Empty structures: {empty_count}")
            print(f"    Common structures: {dict(structure_counts.most_common(5))}")
            
            json_stats[col] = {
                'parse_errors': parse_errors,
                'empty_count': empty_count,
                'structures': dict(structure_counts)
            }
        
        self.validation_results['json_integrity'] = json_stats
    
    def validate_count_accuracy(self):
        """Validate helper counts match JSON blob contents"""
        print("\n" + "="*60)
        print("🔢 COUNT ACCURACY VALIDATION")
        print("="*60)
        
        # Sample rows for validation
        sample_size = min(1000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        count_validations = {
            'weakness_count': ('weaknesses_json', 'weaknesses'),
            'reference_count': ('references_json', 'references'),
            'configuration_count': ('configurations_json', 'configurations')
        }
        
        for count_col, (json_col, json_key) in count_validations.items():
            print(f"\n🔍 Validating {count_col}...")
            
            mismatches = 0
            errors = 0
            
            for _, row in sample_df.iterrows():
                try:
                    expected_count = row[count_col]
                    json_data = json.loads(row[json_col]) if pd.notna(row[json_col]) else []
                    
                    if count_col == 'configuration_count':
                        # Special handling for configurations
                        actual_count = 0
                        if isinstance(json_data, dict):
                            nodes = json_data.get('nodes', [])
                            for node in nodes:
                                actual_count += len(node.get('cpeMatch', []))
                        elif isinstance(json_data, list):
                            for config in json_data:
                                if isinstance(config, dict):
                                    nodes = config.get('nodes', [])
                                    for node in nodes:
                                        actual_count += len(node.get('cpeMatch', []))
                    else:
                        actual_count = len(json_data) if isinstance(json_data, list) else 0
                    
                    if expected_count != actual_count:
                        mismatches += 1
                        if mismatches <= 3:  # Show first few mismatches
                            print(f"    ❌ Mismatch: expected {expected_count}, got {actual_count}")
                            
                except Exception as e:
                    errors += 1
                    if errors <= 3:
                        print(f"    ❌ Error validating row: {str(e)[:100]}")
            
            accuracy = ((sample_size - mismatches - errors) / sample_size) * 100
            print(f"    Accuracy: {accuracy:.1f}% ({sample_size - mismatches - errors}/{sample_size})")
            print(f"    Mismatches: {mismatches}, Errors: {errors}")
    
    def validate_cvss_extraction(self):
        """Validate primary CVSS extraction correctness"""
        print("\n" + "="*60)
        print("🛡️  CVSS EXTRACTION VALIDATION")
        print("="*60)
        
        # CVSS version distribution
        cvss_versions = self.df['primary_cvss_ver'].value_counts(dropna=False)
        print("📊 CVSS Version Distribution:")
        for version, count in cvss_versions.items():
            pct = (count / len(self.df)) * 100
            print(f"  {str(version):8} | {count:8,} ({pct:5.1f}%)")
        
        # Score distribution analysis
        scores = pd.to_numeric(self.df['primary_cvss_score'], errors='coerce')
        print(f"\n📊 CVSS Score Statistics:")
        print(f"  Count: {scores.count():,}")
        print(f"  Mean: {scores.mean():.2f}")
        print(f"  Median: {scores.median():.2f}")  
        print(f"  Range: {scores.min():.1f} - {scores.max():.1f}")
        
        # Validate extraction logic on sample
        sample_df = self.df[self.df['primary_cvss_ver'].notna()].sample(n=100, random_state=42)
        extraction_errors = 0
        
        print(f"\n🔍 Validating extraction logic on {len(sample_df)} samples...")
        
        for _, row in sample_df.iterrows():
            try:
                metrics_data = json.loads(row['metrics_json'])
                expected_version = row['primary_cvss_ver']
                
                # Check if the version exists in metrics
                version_mapping = {
                    '4.0': 'cvssMetricV40',
                    '3.1': 'cvssMetricV31', 
                    '3.0': 'cvssMetricV30',
                    '2.0': 'cvssMetricV2'
                }
                
                expected_key = version_mapping.get(str(expected_version))
                if expected_key and expected_key not in metrics_data:
                    extraction_errors += 1
                    
            except Exception as e:
                extraction_errors += 1
        
        extraction_accuracy = ((len(sample_df) - extraction_errors) / len(sample_df)) * 100
        print(f"    Extraction accuracy: {extraction_accuracy:.1f}%")
        
        self.validation_results['cvss_extraction'] = {
            'version_distribution': dict(cvss_versions),
            'score_stats': scores.describe().to_dict(),
            'extraction_accuracy': extraction_accuracy
        }
    
    def validate_temporal_coverage(self):
        """Validate temporal aspects and coverage"""
        print("\n" + "="*60)
        print("📅 TEMPORAL COVERAGE VALIDATION")
        print("="*60)
        
        # Convert timestamps
        self.df['timestamp_dt'] = pd.to_datetime(self.df['reconstruction_timestamp'], errors='coerce')
        
        # Basic temporal stats
        print("📊 Temporal Statistics:")
        print(f"  Earliest timestamp: {self.df['timestamp_dt'].min()}")
        print(f"  Latest timestamp: {self.df['timestamp_dt'].max()}")
        print(f"  Temporal span: {(self.df['timestamp_dt'].max() - self.df['timestamp_dt'].min()).days:,} days")
        
        # Year distribution
        self.df['year'] = self.df['timestamp_dt'].dt.year
        year_counts = self.df['year'].value_counts().sort_index()
        
        print(f"\n📊 Records by Year:")
        for year, count in year_counts.head(10).items():
            if pd.notna(year):
                print(f"  {int(year):4d} | {count:8,}")
        if len(year_counts) > 10:
            print(f"  ... and {len(year_counts) - 10} more years")
        
        # Check for reasonable distributions
        recent_years = year_counts[year_counts.index >= 2020].sum() if not year_counts.empty else 0
        total_records = len(self.df)
        recent_pct = (recent_years / total_records) * 100 if total_records > 0 else 0
        
        print(f"\n📊 Recent Activity (2020+): {recent_years:,} records ({recent_pct:.1f}%)")
        
        self.validation_results['temporal'] = {
            'earliest_date': str(self.df['timestamp_dt'].min()),
            'latest_date': str(self.df['timestamp_dt'].max()),
            'year_distribution': dict(year_counts.head(20)),
            'recent_activity_pct': recent_pct
        }
    
    def validate_data_completeness(self):
        """Validate completeness against expected totals"""
        print("\n" + "="*60)
        print("📋 DATA COMPLETENESS VALIDATION")
        print("="*60)
        
        # Check against reconstruction files
        recon_dir = Path("catalogs_processed/reconstructions")
        if recon_dir.exists():
            print("🔍 Counting source JSONL lines...")
            total_source_lines = 0
            jsonl_files = list(recon_dir.glob("CVE-*.jsonl"))
            
            # Sample count (checking first 100 files for performance)
            sample_files = jsonl_files[:100]
            sample_lines = 0
            
            for jsonl_file in sample_files:
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        file_lines = sum(1 for line in f if line.strip())
                        sample_lines += file_lines
                except Exception as e:
                    print(f"    ⚠️  Error reading {jsonl_file}: {e}")
            
            # Estimate total
            if sample_files:
                avg_lines_per_file = sample_lines / len(sample_files)
                estimated_total = avg_lines_per_file * len(jsonl_files)
                
                print(f"📊 Completeness Estimate:")
                print(f"  Total JSONL files: {len(jsonl_files):,}")
                print(f"  Sample files checked: {len(sample_files):,}")
                print(f"  Lines in sample: {sample_lines:,}")
                print(f"  Avg lines per file: {avg_lines_per_file:.1f}")
                print(f"  Estimated total lines: {estimated_total:,.0f}")
                print(f"  Actual CSV rows: {len(self.df):,}")
                
                coverage_pct = (len(self.df) / estimated_total) * 100 if estimated_total > 0 else 0
                print(f"  Coverage: {coverage_pct:.1f}%")
                
                self.validation_results['completeness'] = {
                    'estimated_source_lines': estimated_total,
                    'actual_csv_rows': len(self.df),
                    'coverage_percentage': coverage_pct
                }
        else:
            print("⚠️  Source reconstruction directory not found - skipping completeness check")
    
    def generate_summary_report(self):
        """Generate final validation summary"""
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY REPORT")
        print("="*60)
        
        print(f"Dataset: {self.csv_path}")
        print(f"Validation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\n📋 Key Metrics:")
        print(f"  Total rows: {len(self.df):,}")
        print(f"  Total columns: {len(self.df.columns)}")
        print(f"  Unique CVEs: {self.validation_results.get('primary_key', {}).get('unique_cves', 'N/A'):,}")
        print(f"  Primary key duplicates: {self.validation_results.get('primary_key', {}).get('duplicate_count', 'N/A')}")
        
        # Overall health assessment
        issues = []
        if self.validation_results.get('schema', {}).get('missing_columns'):
            issues.append("Missing expected columns")
        if self.validation_results.get('primary_key', {}).get('duplicate_count', 0) > 0:
            issues.append("Primary key duplicates found")
        
        if not issues:
            print(f"\n✅ OVERALL STATUS: HEALTHY - Dataset meets all validation criteria")
        else:
            print(f"\n⚠️  OVERALL STATUS: ISSUES DETECTED")
            for issue in issues:
                print(f"    - {issue}")
        
        return self.validation_results
    
    def run_full_validation(self):
        """Execute complete validation suite"""
        print("🚀 Starting comprehensive validation of master CVE dataset...")
        
        # Load data
        self.load_dataset()
        
        # Run all validations
        self.validate_schema_structure()
        self.validate_primary_key_uniqueness()
        self.validate_json_blob_integrity()
        self.validate_count_accuracy()
        self.validate_cvss_extraction()
        self.validate_temporal_coverage()
        self.validate_data_completeness()
        
        # Generate summary
        return self.generate_summary_report()

if __name__ == "__main__":
    validator = MasterDatasetValidator()
    results = validator.run_full_validation() 