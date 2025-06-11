#!/usr/bin/env python3
"""
CVE Schema Analyzer - Comprehensive Data Structure Analysis
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class CVESchemaAnalyzer:
    def __init__(self, reconstructions_dir="catalogs_processed/reconstructions"):
        self.reconstructions_dir = Path(reconstructions_dir)
        self.all_properties = set()
        self.property_types = defaultdict(set)
        self.property_occurrence_count = Counter()
        self.files_analyzed = 0
        self.timeline_states_analyzed = 0
        
    def analyze_sample_files(self, max_files=200):
        """Analyze first N files to understand schema patterns"""
        logging.info(f"Starting analysis of first {max_files} files")
        
        files = list(self.reconstructions_dir.glob("CVE-*.jsonl"))[:max_files]
        
        for file_path in files:
            self._analyze_single_file(file_path)
            self.files_analyzed += 1
            
            if self.files_analyzed % 50 == 0:
                logging.info(f"Progress: {self.files_analyzed} files, {self.timeline_states_analyzed} states")
    
    def _analyze_single_file(self, file_path):
        """Analyze one CVE reconstruction file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    timeline_state = json.loads(line)
                    self._analyze_structure(timeline_state, "")
                    self.timeline_states_analyzed += 1
    
    def _analyze_structure(self, obj, path=""):
        """Recursively analyze object structure"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                if not path:  # Top-level property
                    self.all_properties.add(key)
                
                self.property_occurrence_count[current_path] += 1
                value_type = self._get_type_info(value)
                self.property_types[current_path].add(value_type)
                
                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    self._analyze_structure(value, current_path)
                    
        elif isinstance(obj, list):
            for item in obj:
                self._analyze_structure(item, path)
    
    def _get_type_info(self, value):
        """Get detailed type information"""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return f"array[{len(value)}]"
        elif isinstance(value, dict):
            return f"object[{len(value)}keys]"
        else:
            return type(value).__name__
    
    def generate_report(self):
        """Generate comprehensive schema analysis report"""
        report = []
        report.append("CVE SCHEMA ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"Files analyzed: {self.files_analyzed}")
        report.append(f"Timeline states: {self.timeline_states_analyzed}")
        report.append(f"Unique top-level properties: {len(self.all_properties)}")
        report.append("")
        
        # Top-level properties
        report.append("TOP-LEVEL PROPERTIES:")
        for prop in sorted(self.all_properties):
            count = self.property_occurrence_count[prop]
            types = sorted(self.property_types[prop])
            report.append(f"  {prop}: {count} occurrences, types: {types}")
        report.append("")
        
        # All property paths
        report.append("ALL PROPERTY PATHS (nested included):")
        for path in sorted(self.property_types.keys()):
            count = self.property_occurrence_count[path]
            types = sorted(self.property_types[path])
            report.append(f"  {path}: {count} occurrences, types: {types}")
        
        return "\n".join(report)
    
    def run_analysis(self, max_files=200):
        """Run complete analysis"""
        print("🔍 Starting CVE Schema Analysis")
        self.analyze_sample_files(max_files)
        
        report = self.generate_report()
        
        # Save to file
        with open("cve_schema_report.txt", "w") as f:
            f.write(report)
        
        # Print summary
        print(f"\n✅ Analysis complete!")
        print(f"Files analyzed: {self.files_analyzed}")
        print(f"Timeline states: {self.timeline_states_analyzed}")
        print(f"Top-level properties found: {len(self.all_properties)}")
        print(f"Report saved to: cve_schema_report.txt")
        
        return report

if __name__ == "__main__":
    analyzer = CVESchemaAnalyzer()
    analyzer.run_analysis(max_files=200) 