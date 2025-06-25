#!/usr/bin/env python3
"""
CVE Catalog JSONL to CSV Converter
Converts complex nested CVE vulnerability data from JSONL format to a flattened CSV 
suitable for data analysis and machine learning applications.

Handles:
- Multiple CVSS versions (prioritizes latest available)
- Nested JSON structures flattening
- Variable field presence across records
- Multi-language descriptions (extracts English)
- CISA KEV (Known Exploited Vulnerabilities) indicators
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

def extract_primary_description(descriptions: List[Dict[str, str]]) -> str:
    """
    Extract primary English description from multi-language descriptions list
    
    Args:
        descriptions: List of description objects with lang/value pairs
        
    Returns:
        English description text or first available description
    """
    if not descriptions:
        return ""
    
    # Prioritize English description
    for desc in descriptions:
        if desc.get('lang', '').lower() == 'en':
            return desc.get('value', '')
    
    # Fallback to first available description
    return descriptions[0].get('value', '')

def extract_cvss_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and prioritize CVSS metrics from multiple versions
    Priority order: v4.0 > v3.1 > v3.0 > v2.0
    
    Args:
        metrics: Nested metrics dictionary containing multiple CVSS versions
        
    Returns:
        Flattened dictionary with primary CVSS metrics
    """
    cvss_data = {
        'cvss_version': None,
        'cvss_base_score': None,
        'cvss_base_severity': None,
        'cvss_vector_string': None,
        'cvss_exploitability_score': None,
        'cvss_impact_score': None,
        'cvss_attack_vector': None,
        'cvss_attack_complexity': None,
        'cvss_privileges_required': None,
        'cvss_user_interaction': None,
        'cvss_confidentiality_impact': None,
        'cvss_integrity_impact': None,
        'cvss_availability_impact': None
    }
    
    # Priority order for CVSS versions
    version_priority = ['cvssMetricV40', 'cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2']
    
    for version_key in version_priority:
        if version_key in metrics and metrics[version_key]:
            metric = metrics[version_key][0]  # Take first (primary) metric
            cvss_info = metric.get('cvssData', {})
            
            cvss_data['cvss_version'] = cvss_info.get('version')
            cvss_data['cvss_base_score'] = cvss_info.get('baseScore')
            cvss_data['cvss_base_severity'] = cvss_info.get('baseSeverity')
            cvss_data['cvss_vector_string'] = cvss_info.get('vectorString')
            cvss_data['cvss_exploitability_score'] = metric.get('exploitabilityScore')
            cvss_data['cvss_impact_score'] = metric.get('impactScore')
            
            # Version-specific field mapping
            if version_key == 'cvssMetricV2':
                cvss_data['cvss_attack_vector'] = cvss_info.get('accessVector')
                cvss_data['cvss_attack_complexity'] = cvss_info.get('accessComplexity')
                cvss_data['cvss_privileges_required'] = cvss_info.get('authentication')
                cvss_data['cvss_user_interaction'] = 'Required' if cvss_info.get('userInteractionRequired') else 'None'
            else:
                cvss_data['cvss_attack_vector'] = cvss_info.get('attackVector')
                cvss_data['cvss_attack_complexity'] = cvss_info.get('attackComplexity')
                cvss_data['cvss_privileges_required'] = cvss_info.get('privilegesRequired')
                cvss_data['cvss_user_interaction'] = cvss_info.get('userInteraction')
            
            cvss_data['cvss_confidentiality_impact'] = cvss_info.get('confidentialityImpact')
            cvss_data['cvss_integrity_impact'] = cvss_info.get('integrityImpact')
            cvss_data['cvss_availability_impact'] = cvss_info.get('availabilityImpact')
            
            break  # Use first available version (highest priority)
    
    return cvss_data

def extract_primary_weakness(weaknesses: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract primary weakness classification (CWE)
    
    Args:
        weaknesses: List of weakness objects with source/type/description
        
    Returns:
        Dictionary with primary weakness information
    """
    weakness_data = {
        'primary_weakness_source': None,
        'primary_weakness_type': None,
        'primary_weakness_description': None
    }
    
    if weaknesses:
        primary = weaknesses[0]  # Take first (primary) weakness
        weakness_data['primary_weakness_source'] = primary.get('source')
        weakness_data['primary_weakness_type'] = primary.get('type')
        
        descriptions = primary.get('description', [])
        weakness_data['primary_weakness_description'] = extract_primary_description(descriptions)
    
    return weakness_data

def count_nested_elements(data: List[Any]) -> int:
    """
    Count elements in nested list structures
    
    Args:
        data: List or nested structure to count
        
    Returns:
        Total count of elements
    """
    if not data:
        return 0
    return len(data)

def extract_cisa_kev_info(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract CISA Known Exploited Vulnerabilities information
    
    Args:
        record: CVE record dictionary
        
    Returns:
        Dictionary with CISA KEV status and details
    """
    return {
        'cisa_kev_listed': 'cisaExploitAdd' in record,
        'cisa_exploit_add_date': record.get('cisaExploitAdd'),
        'cisa_action_due_date': record.get('cisaActionDue'),
        'cisa_required_action': record.get('cisaRequiredAction'),
        'cisa_vulnerability_name': record.get('cisaVulnerabilityName')
    }

def convert_cve_record_to_flat_dict(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single CVE record from nested JSON to flat dictionary for CSV
    
    Args:
        record: Single CVE record from JSONL
        
    Returns:
        Flattened dictionary suitable for CSV row
    """
    # Extract core metadata
    flat_record = {
        'cve_id': record.get('id'),
        'source_identifier': record.get('sourceIdentifier'),
        'published_date': record.get('published'),
        'last_modified_date': record.get('lastModified'),
        'vulnerability_status': record.get('vulnStatus'),
        'primary_description': extract_primary_description(record.get('descriptions', [])),
        'description_length': len(extract_primary_description(record.get('descriptions', []))),
        'reference_count': count_nested_elements(record.get('references', [])),
        'configuration_count': count_nested_elements(record.get('configurations', [])),
        'has_vendor_comments': len(record.get('vendorComments', [])) > 0,
        'has_evaluator_comments': 'evaluatorComment' in record
    }
    
    # Extract CVSS metrics (prioritized by version)
    cvss_metrics = extract_cvss_metrics(record.get('metrics', {}))
    flat_record.update(cvss_metrics)
    
    # Extract primary weakness classification
    weakness_data = extract_primary_weakness(record.get('weaknesses', []))
    flat_record.update(weakness_data)
    
    # Extract CISA KEV information
    cisa_data = extract_cisa_kev_info(record)
    flat_record.update(cisa_data)
    
    return flat_record

def convert_jsonl_to_csv(input_file: str, output_file: str) -> None:
    """
    Convert CVE catalog JSONL file to flattened CSV format
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output CSV file
    """
    print(f"Converting CVE catalog from {input_file} to {output_file}")
    
    # Define CSV column headers in logical order
    csv_headers = [
        # Core CVE metadata
        'cve_id', 'source_identifier', 'published_date', 'last_modified_date', 
        'vulnerability_status', 'primary_description', 'description_length',
        
        # CVSS metrics (prioritized latest version)
        'cvss_version', 'cvss_base_score', 'cvss_base_severity', 'cvss_vector_string',
        'cvss_exploitability_score', 'cvss_impact_score', 'cvss_attack_vector',
        'cvss_attack_complexity', 'cvss_privileges_required', 'cvss_user_interaction',
        'cvss_confidentiality_impact', 'cvss_integrity_impact', 'cvss_availability_impact',
        
        # Weakness classification
        'primary_weakness_source', 'primary_weakness_type', 'primary_weakness_description',
        
        # Reference and configuration counts
        'reference_count', 'configuration_count', 'has_vendor_comments', 'has_evaluator_comments',
        
        # CISA KEV information
        'cisa_kev_listed', 'cisa_exploit_add_date', 'cisa_action_due_date', 
        'cisa_required_action', 'cisa_vulnerability_name'
    ]
    
    processed_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as jsonl_file, \
             open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
            
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
            csv_writer.writeheader()
            
            for line_num, line in enumerate(jsonl_file, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    flat_record = convert_cve_record_to_flat_dict(record)
                    csv_writer.writerow(flat_record)
                    processed_count += 1
                    
                    # Progress indication for large files
                    if processed_count % 10000 == 0:
                        print(f"Processed {processed_count:,} records...")
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing record at line {line_num}: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"ERROR: Input file {input_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR during conversion: {e}")
        sys.exit(1)
    
    print(f"Conversion completed successfully!")
    print(f"Total records processed: {processed_count:,}")
    print(f"Output saved to: {output_file}")

def main():
    """Main execution function"""
    # Define input and output paths
    input_file = "../../../../catalogs_processed/2025-06-08_snapshot.jsonl"
    output_dir = Path(".")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate descriptive output filename with current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cve_catalog_flattened_{current_time}.csv"
    
    # Validate input file exists
    if not Path(input_file).exists():
        print(f"ERROR: Input file {input_file} does not exist")
        sys.exit(1)
    
    # Perform conversion
    convert_jsonl_to_csv(str(input_file), str(output_file))
    
    print(f"\n=== CONVERSION SUMMARY ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Timestamp: {current_time}")

if __name__ == "__main__":
    main() 