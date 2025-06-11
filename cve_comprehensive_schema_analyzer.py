#!/usr/bin/env python3
"""
Comprehensive CVE Schema Analyzer
=================================

Advanced analysis of CVE reconstruction data to build complete master schema
for CSV flattening without data loss.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ComprehensiveCVESchemaAnalyzer:
    def __init__(self, reconstructions_dir="catalogs_processed/reconstructions"):
        self.reconstructions_dir = Path(reconstructions_dir)
        
        # Core tracking structures
        self.property_paths = set()                    # All discovered property paths
        self.property_types = defaultdict(set)         # path -> {types}
        self.property_counts = Counter()               # path -> occurrence count
        self.array_lengths = defaultdict(set)          # array_path -> {lengths}
        self.object_key_counts = defaultdict(set)      # object_path -> {key_counts}
        
        # Advanced analysis
        self.value_samples = defaultdict(list)         # path -> [sample_values]
        self.null_occurrences = defaultdict(int)       # path -> null_count
        self.conditional_properties = defaultdict(set) # parent_path -> {child_paths}
        
        # Statistics
        self.files_analyzed = 0
        self.timeline_states_analyzed = 0
        self.unique_cves = set()
        
    def analyze_comprehensive_sample(self, max_files=200):
        """Comprehensive analysis of sample files"""
        logging.info(f"=== COMPREHENSIVE ANALYSIS: {max_files} files ===")
        
        files = list(self.reconstructions_dir.glob("CVE-*.jsonl"))[:max_files]
        logging.info(f"Found {len(files)} files to analyze")
        
        for file_path in files:
            self._analyze_file_comprehensive(file_path)
            self.files_analyzed += 1
            
            if self.files_analyzed % 50 == 0:
                logging.info(f"Progress: {self.files_analyzed} files, "
                           f"{self.timeline_states_analyzed} states, "
                           f"{len(self.property_paths)} unique paths")
        
        logging.info("=== ANALYSIS COMPLETE ===")
        self._print_summary_stats()
    
    def _analyze_file_comprehensive(self, file_path):
        """Deep analysis of single CVE file"""
        cve_id = file_path.stem
        self.unique_cves.add(cve_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        timeline_state = json.loads(line)
                        self._analyze_state_structure(timeline_state, "", cve_id, line_num)
                        self.timeline_states_analyzed += 1
                    except json.JSONDecodeError as e:
                        logging.warning(f"JSON error in {file_path}:{line_num} - {e}")
    
    def _analyze_state_structure(self, obj, path, cve_id, line_num):
        """Recursive deep structure analysis"""
        if isinstance(obj, dict):
            # Track this object path
            if path:
                self.property_paths.add(path)
                self.property_counts[path] += 1
                self.property_types[path].add(f"object[{len(obj)}keys]")
                self.object_key_counts[path].add(len(obj))
                
                # Sample object keys for pattern analysis
                if len(self.value_samples[path]) < 5:
                    self.value_samples[path].append(sorted(obj.keys()))
            
            # Analyze each property
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                self.property_paths.add(current_path)
                self.property_counts[current_path] += 1
                
                # Track parent-child relationships
                if path:
                    self.conditional_properties[path].add(current_path)
                
                # Analyze value
                if value is None:
                    self.null_occurrences[current_path] += 1
                    self.property_types[current_path].add("null")
                else:
                    value_type = self._get_detailed_type(value, current_path)
                    self.property_types[current_path].add(value_type)
                    
                    # Sample non-null values
                    if len(self.value_samples[current_path]) < 3:
                        if isinstance(value, (str, int, float, bool)):
                            self.value_samples[current_path].append(value)
                
                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    self._analyze_state_structure(value, current_path, cve_id, line_num)
                    
        elif isinstance(obj, list):
            # Track array characteristics
            if path:
                self.property_paths.add(path)
                self.property_counts[path] += 1
                self.property_types[path].add(f"array[{len(obj)}]")
                self.array_lengths[path].add(len(obj))
            
            # Analyze array elements
            for idx, item in enumerate(obj):
                array_element_path = f"{path}[{idx}]" if path else f"[{idx}]"
                self._analyze_state_structure(item, array_element_path, cve_id, line_num)
    
    def _get_detailed_type(self, value, path):
        """Enhanced type analysis with context"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            # Analyze string patterns for common formats
            if "T" in value and ":" in value and len(value) > 15:
                return "datetime_string"
            elif value.startswith("CVE-"):
                return "cve_id_string"
            elif value.startswith("cpe:"):
                return "cpe_string"
            elif value.startswith("CWE-"):
                return "cwe_string"
            elif "." in value and len(value.split(".")) == 4:
                return "version_string"
            else:
                return "string"
        elif isinstance(value, list):
            return f"array[{len(value)}]"
        elif isinstance(value, dict):
            return f"object[{len(value)}keys]"
        else:
            return type(value).__name__
    
    def _print_summary_stats(self):
        """Print comprehensive summary statistics"""
        print(f"\n{'='*60}")
        print("COMPREHENSIVE CVE SCHEMA ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Files analyzed: {self.files_analyzed}")
        print(f"Timeline states: {self.timeline_states_analyzed}")
        print(f"Unique CVEs: {len(self.unique_cves)}")
        print(f"Total property paths: {len(self.property_paths)}")
        print(f"Properties with null values: {len(self.null_occurrences)}")
    
    def generate_master_schema_report(self):
        """Generate comprehensive master schema documentation"""
        report = []
        report.append("COMPREHENSIVE CVE MASTER SCHEMA REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Statistics:")
        report.append(f"  Files: {self.files_analyzed}")
        report.append(f"  Timeline States: {self.timeline_states_analyzed}")
        report.append(f"  Unique CVEs: {len(self.unique_cves)}")
        report.append(f"  Property Paths: {len(self.property_paths)}")
        report.append("")
        
        # Group properties by hierarchy level
        top_level = sorted([p for p in self.property_paths if '.' not in p and '[' not in p])
        nested_level_2 = sorted([p for p in self.property_paths if p.count('.') == 1 and '[' not in p])
        nested_level_3 = sorted([p for p in self.property_paths if p.count('.') == 2 and '[' not in p])
        array_elements = sorted([p for p in self.property_paths if '[' in p])
        
        report.append("1. TOP-LEVEL PROPERTIES (Core CVE Structure)")
        report.append("-" * 50)
        for prop in top_level:
            count = self.property_counts[prop]
            types = sorted(self.property_types[prop])
            percentage = (count / self.timeline_states_analyzed) * 100
            null_count = self.null_occurrences.get(prop, 0)
            
            report.append(f"  {prop}:")
            report.append(f"    Occurrences: {count}/{self.timeline_states_analyzed} ({percentage:.1f}%)")
            report.append(f"    Types: {types}")
            if null_count > 0:
                report.append(f"    Null values: {null_count}")
            if prop in self.value_samples:
                report.append(f"    Sample values: {self.value_samples[prop][:3]}")
            report.append("")
        
        report.append("2. SECOND-LEVEL NESTED PROPERTIES")
        report.append("-" * 50)
        for prop in nested_level_2:
            count = self.property_counts[prop]
            types = sorted(self.property_types[prop])
            percentage = (count / self.timeline_states_analyzed) * 100
            
            report.append(f"  {prop}: {count} occurrences ({percentage:.1f}%) - {types}")
        report.append("")
        
        report.append("3. THIRD-LEVEL NESTED PROPERTIES")
        report.append("-" * 50)
        for prop in nested_level_3[:20]:  # Limit to first 20 for readability
            count = self.property_counts[prop]
            types = sorted(self.property_types[prop])
            report.append(f"  {prop}: {count} occurrences - {types}")
        report.append("")
        
        report.append("4. ARRAY CHARACTERISTICS")
        report.append("-" * 50)
        for array_path in sorted(self.array_lengths.keys()):
            if '[' not in array_path:  # Only show array containers, not elements
                lengths = sorted(self.array_lengths[array_path])
                count = self.property_counts[array_path]
                report.append(f"  {array_path}: {count} occurrences, lengths: {lengths}")
        report.append("")
        
        report.append("5. CONDITIONAL PROPERTY PATTERNS")
        report.append("-" * 50)
        for parent, children in sorted(self.conditional_properties.items()):
            if len(children) > 1:  # Only show parents with multiple children
                report.append(f"  {parent} contains: {sorted(children)}")
        report.append("")
        
        # Write comprehensive report
        report_content = "\n".join(report)
        with open("cve_master_schema_report.txt", "w", encoding="utf-8") as f:
            f.write(report_content)
        
        return report_content
    
    def identify_csv_flattening_requirements(self):
        """Identify specific requirements for CSV flattening"""
        print(f"\n{'='*60}")
        print("CSV FLATTENING REQUIREMENTS ANALYSIS")
        print(f"{'='*60}")
        
        # Arrays that need special handling
        array_properties = [p for p in self.property_paths if any('array[' in t for t in self.property_types[p])]
        print(f"\nArrays requiring flattening ({len(array_properties)}):")
        for prop in sorted(array_properties)[:10]:  # Show first 10
            max_length = max([int(t.split('[')[1].split(']')[0]) for t in self.property_types[prop] if 'array[' in t])
            print(f"  {prop}: max length {max_length}")
        
        # Objects requiring expansion
        object_properties = [p for p in self.property_paths if any('object[' in t for t in self.property_types[p])]
        print(f"\nObjects requiring flattening ({len(object_properties)}):")
        for prop in sorted(object_properties)[:10]:
            max_keys = max([int(t.split('[')[1].split(']')[0].replace('keys', '')) for t in self.property_types[prop] if 'object[' in t])
            print(f"  {prop}: max {max_keys} keys")
        
        # Optional fields analysis
        total_states = self.timeline_states_analyzed
        optional_fields = []
        required_fields = []
        
        for prop in self.property_paths:
            if '.' not in prop and '[' not in prop:  # Top-level only
                count = self.property_counts[prop]
                percentage = (count / total_states) * 100
                if percentage < 95:
                    optional_fields.append((prop, percentage))
                else:
                    required_fields.append((prop, percentage))
        
        print(f"\nRequired fields (>95% presence): {len(required_fields)}")
        for prop, pct in sorted(required_fields, key=lambda x: x[1], reverse=True):
            print(f"  {prop}: {pct:.1f}%")
        
        print(f"\nOptional fields (<95% presence): {len(optional_fields)}")
        for prop, pct in sorted(optional_fields, key=lambda x: x[1], reverse=True):
            print(f"  {prop}: {pct:.1f}%")
    
    def run_comprehensive_analysis(self, max_files=200):
        """Run complete comprehensive analysis workflow"""
        print("🔍 COMPREHENSIVE CVE SCHEMA ANALYSIS")
        print("=" * 60)
        
        # Step 1: Deep structural analysis
        self.analyze_comprehensive_sample(max_files)
        
        # Step 2: Generate master schema report
        print("\n📋 Generating master schema report...")
        report = self.generate_master_schema_report()
        
        # Step 3: CSV flattening requirements
        self.identify_csv_flattening_requirements()
        
        print(f"\n✅ COMPREHENSIVE ANALYSIS COMPLETE!")
        print(f"📊 Master schema report: cve_master_schema_report.txt")
        print(f"🎯 Ready for CSV schema design!")

if __name__ == "__main__":
    analyzer = ComprehensiveCVESchemaAnalyzer()
    analyzer.run_comprehensive_analysis(max_files=200) 