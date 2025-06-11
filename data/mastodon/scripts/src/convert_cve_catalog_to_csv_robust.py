#!/usr/bin/env python3
"""
Enhanced CVE Catalog JSONL to CSV Converter with Robust CVSS Processing
Converts complex nested CVE vulnerability data from JSONL format to a flattened CSV 
using the professional cvss library for robust CVSS vector parsing and validation.

Enhanced Features:
- Robust CVSS vector parsing using cvss library
- Comprehensive validation and error handling
- Standardized CVSS score calculation
- Automatic severity classification
- Clean vector string normalization
- Enhanced error reporting and logging
"""

import json
import csv
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from cvss import CVSS2, CVSS3, CVSS4

# Configure logging for robust error tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cvss_conversion.log')
    ]
)
logger = logging.getLogger(__name__)

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

def parse_cvss_vector_with_library(vector_string: str, version: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Parse CVSS vector string using the robust cvss library
    
    Args:
        vector_string: CVSS vector string to parse
        version: CVSS version identifier
        
    Returns:
        Tuple of (cvss_object, extracted_data_dict)
    """
    cvss_data = {
        'cvss_library_parsed': False,
        'cvss_library_error': None,
        'cvss_base_score_validated': None,
        'cvss_severity_validated': None,
        'cvss_clean_vector': None,
        'cvss_temporal_score': None,
        'cvss_environmental_score': None
    }
    
    if not vector_string:
        return None, cvss_data
    
    try:
        # Determine CVSS version and create appropriate object
        cvss_obj = None
        
        if version.startswith('2.'):
            cvss_obj = CVSS2(vector_string)
            scores = cvss_obj.scores()
            cvss_data.update({
                'cvss_library_parsed': True,
                'cvss_base_score_validated': scores[0] if len(scores) > 0 else None,
                'cvss_temporal_score': scores[1] if len(scores) > 1 else None,
                'cvss_environmental_score': scores[2] if len(scores) > 2 else None,
                'cvss_clean_vector': cvss_obj.clean_vector(),
                'cvss_severity_validated': cvss_obj.severities()[0] if hasattr(cvss_obj, 'severities') else None
            })
            
        elif version.startswith('3.'):
            cvss_obj = CVSS3(vector_string)
            scores = cvss_obj.scores()
            severities = cvss_obj.severities()
            cvss_data.update({
                'cvss_library_parsed': True,
                'cvss_base_score_validated': scores[0] if len(scores) > 0 else None,
                'cvss_temporal_score': scores[1] if len(scores) > 1 else None,
                'cvss_environmental_score': scores[2] if len(scores) > 2 else None,
                'cvss_clean_vector': cvss_obj.clean_vector(),
                'cvss_severity_validated': severities[0] if len(severities) > 0 else None
            })
            
        elif version.startswith('4.'):
            cvss_obj = CVSS4(vector_string)
            cvss_data.update({
                'cvss_library_parsed': True,
                'cvss_base_score_validated': cvss_obj.base_score,
                'cvss_severity_validated': cvss_obj.severity,
                'cvss_clean_vector': str(cvss_obj)  # CVSS4 string representation
            })
            
        else:
            logger.warning(f"Unsupported CVSS version: {version}")
            cvss_data['cvss_library_error'] = f"Unsupported version: {version}"
            
        return cvss_obj, cvss_data
        
    except Exception as e:
        logger.warning(f"CVSS parsing error for vector '{vector_string}' (v{version}): {str(e)}")
        cvss_data.update({
            'cvss_library_parsed': False,
            'cvss_library_error': str(e)[:100]  # Truncate long error messages
        })
        return None, cvss_data

def extract_enhanced_cvss_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and validate CVSS metrics using the robust cvss library
    Enhanced with comprehensive validation and multiple version support
    
    Args:
        metrics: Nested metrics dictionary containing multiple CVSS versions
        
    Returns:
        Enhanced dictionary with validated CVSS metrics
    """
    # Initialize comprehensive CVSS data structure
    cvss_data = {
        # Original fields from manual parsing
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
        'cvss_availability_impact': None,
        
        # Enhanced fields from cvss library
        'cvss_library_parsed': False,
        'cvss_library_error': None,
        'cvss_base_score_validated': None,
        'cvss_severity_validated': None,
        'cvss_clean_vector': None,
        'cvss_temporal_score': None,
        'cvss_environmental_score': None,
        'cvss_score_validation_match': None,
        'cvss_available_versions': []
    }
    
    if not metrics:
        return cvss_data
    
    # Track all available CVSS versions
    available_versions = []
    for version_key in ['cvssMetricV40', 'cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2']:
        if version_key in metrics and metrics[version_key]:
            available_versions.append(version_key)
    
    cvss_data['cvss_available_versions'] = available_versions
    
    # Priority order for CVSS versions (latest first)
    version_priority = ['cvssMetricV40', 'cvssMetricV31', 'cvssMetricV30', 'cvssMetricV2']
    
    for version_key in version_priority:
        if version_key in metrics and metrics[version_key]:
            try:
                metric = metrics[version_key][0]  # Take first (primary) metric
                cvss_info = metric.get('cvssData', {})
                
                # Extract basic CVSS information
                version = cvss_info.get('version')
                vector_string = cvss_info.get('vectorString')
                original_base_score = cvss_info.get('baseScore')
                
                cvss_data.update({
                    'cvss_version': version,
                    'cvss_base_score': original_base_score,
                    'cvss_base_severity': cvss_info.get('baseSeverity'),
                    'cvss_vector_string': vector_string,
                    'cvss_exploitability_score': metric.get('exploitabilityScore'),
                    'cvss_impact_score': metric.get('impactScore')
                })
                
                # Extract version-specific fields with robust handling
                if version_key == 'cvssMetricV2':
                    cvss_data.update({
                        'cvss_attack_vector': cvss_info.get('accessVector'),
                        'cvss_attack_complexity': cvss_info.get('accessComplexity'),
                        'cvss_privileges_required': cvss_info.get('authentication'),
                        'cvss_user_interaction': 'Required' if cvss_info.get('userInteractionRequired') else 'None'
                    })
                else:
                    cvss_data.update({
                        'cvss_attack_vector': cvss_info.get('attackVector'),
                        'cvss_attack_complexity': cvss_info.get('attackComplexity'),
                        'cvss_privileges_required': cvss_info.get('privilegesRequired'),
                        'cvss_user_interaction': cvss_info.get('userInteraction')
                    })
                
                cvss_data.update({
                    'cvss_confidentiality_impact': cvss_info.get('confidentialityImpact'),
                    'cvss_integrity_impact': cvss_info.get('integrityImpact'),
                    'cvss_availability_impact': cvss_info.get('availabilityImpact')
                })
                
                # Enhanced validation using cvss library
                if vector_string and version:
                    cvss_obj, library_data = parse_cvss_vector_with_library(vector_string, version)
                    cvss_data.update(library_data)
                    
                    # Validate score consistency between manual extraction and library
                    if (library_data['cvss_base_score_validated'] is not None and 
                        original_base_score is not None):
                        score_diff = abs(float(library_data['cvss_base_score_validated']) - float(original_base_score))
                        cvss_data['cvss_score_validation_match'] = score_diff < 0.1  # Allow small floating point differences
                        
                        if score_diff >= 0.1:
                            logger.warning(f"Score mismatch detected: original={original_base_score}, validated={library_data['cvss_base_score_validated']}")
                
                break  # Use first available version (highest priority)
                
            except Exception as e:
                logger.error(f"Error processing CVSS version {version_key}: {str(e)}")
                continue
    
    return cvss_data

def extract_primary_weakness(weaknesses: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Extract primary weakness classification (CWE) with enhanced error handling
    
    Args:
        weaknesses: List of weakness objects with source/type/description
        
    Returns:
        Dictionary with primary weakness information
    """
    weakness_data = {
        'primary_weakness_source': None,
        'primary_weakness_type': None,
        'primary_weakness_description': None,
        'weakness_count': 0
    }
    
    if not weaknesses:
        return weakness_data
    
    weakness_data['weakness_count'] = len(weaknesses)
    
    try:
        primary = weaknesses[0]  # Take first (primary) weakness
        weakness_data.update({
            'primary_weakness_source': primary.get('source'),
            'primary_weakness_type': primary.get('type')
        })
        
        descriptions = primary.get('description', [])
        weakness_data['primary_weakness_description'] = extract_primary_description(descriptions)
        
    except Exception as e:
        logger.warning(f"Error extracting weakness data: {str(e)}")
    
    return weakness_data

def count_nested_elements(data: List[Any]) -> int:
    """
    Count elements in nested list structures with error handling
    
    Args:
        data: List or nested structure to count
        
    Returns:
        Total count of elements
    """
    try:
        return len(data) if data else 0
    except (TypeError, AttributeError):
        return 0

def extract_cisa_kev_info(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract CISA Known Exploited Vulnerabilities information with validation
    
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
    Enhanced with robust CVSS processing and comprehensive error handling
    
    Args:
        record: Single CVE record from JSONL
        
    Returns:
        Flattened dictionary suitable for CSV row
    """
    try:
        # Extract core metadata with error handling
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
        
        # Extract enhanced CVSS metrics using cvss library
        cvss_metrics = extract_enhanced_cvss_metrics(record.get('metrics', {}))
        flat_record.update(cvss_metrics)
        
        # Extract enhanced weakness classification
        weakness_data = extract_primary_weakness(record.get('weaknesses', []))
        flat_record.update(weakness_data)
        
        # Extract CISA KEV information
        cisa_data = extract_cisa_kev_info(record)
        flat_record.update(cisa_data)
        
        return flat_record
        
    except Exception as e:
        logger.error(f"Error processing CVE record {record.get('id', 'UNKNOWN')}: {str(e)}")
        # Return minimal record with error information
        return {
            'cve_id': record.get('id', 'ERROR'),
            'processing_error': str(e)[:200]  # Truncate long error messages
        }

def convert_jsonl_to_csv_robust(input_file: str, output_file: str) -> Dict[str, int]:
    """
    Convert CVE catalog JSONL file to flattened CSV format with robust error handling
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output CSV file
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Starting robust CVE catalog conversion from {input_file} to {output_file}")
    
    # Define comprehensive CSV column headers
    csv_headers = [
        # Core CVE metadata
        'cve_id', 'source_identifier', 'published_date', 'last_modified_date', 
        'vulnerability_status', 'primary_description', 'description_length',
        
        # Original CVSS metrics
        'cvss_version', 'cvss_base_score', 'cvss_base_severity', 'cvss_vector_string',
        'cvss_exploitability_score', 'cvss_impact_score', 'cvss_attack_vector',
        'cvss_attack_complexity', 'cvss_privileges_required', 'cvss_user_interaction',
        'cvss_confidentiality_impact', 'cvss_integrity_impact', 'cvss_availability_impact',
        
        # Enhanced CVSS validation fields
        'cvss_library_parsed', 'cvss_library_error', 'cvss_base_score_validated',
        'cvss_severity_validated', 'cvss_clean_vector', 'cvss_temporal_score',
        'cvss_environmental_score', 'cvss_score_validation_match', 'cvss_available_versions',
        
        # Enhanced weakness classification
        'primary_weakness_source', 'primary_weakness_type', 'primary_weakness_description',
        'weakness_count',
        
        # Reference and configuration counts
        'reference_count', 'configuration_count', 'has_vendor_comments', 'has_evaluator_comments',
        
        # CISA KEV information
        'cisa_kev_listed', 'cisa_exploit_add_date', 'cisa_action_due_date', 
        'cisa_required_action', 'cisa_vulnerability_name',
        
        # Error tracking
        'processing_error'
    ]
    
    # Initialize processing statistics
    stats = {
        'total_processed': 0,
        'successful_records': 0,
        'cvss_parsed_successfully': 0,
        'cvss_validation_matches': 0,
        'error_records': 0,
        'json_decode_errors': 0
    }
    
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
                    
                    # Update statistics
                    stats['total_processed'] += 1
                    
                    if 'processing_error' not in flat_record:
                        stats['successful_records'] += 1
                        
                        if flat_record.get('cvss_library_parsed'):
                            stats['cvss_parsed_successfully'] += 1
                            
                        if flat_record.get('cvss_score_validation_match'):
                            stats['cvss_validation_matches'] += 1
                    else:
                        stats['error_records'] += 1
                    
                    csv_writer.writerow(flat_record)
                    
                    # Progress indication for large files
                    if stats['total_processed'] % 10000 == 0:
                        logger.info(f"Processed {stats['total_processed']:,} records...")
                        
                except json.JSONDecodeError as e:
                    stats['json_decode_errors'] += 1
                    logger.error(f"JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    stats['error_records'] += 1
                    logger.error(f"Unexpected error processing line {line_num}: {e}")
                    continue
    
    except FileNotFoundError:
        logger.error(f"Input file {input_file} not found")
        raise
    except Exception as e:
        logger.error(f"Critical error during conversion: {e}")
        raise
    
    logger.info("Robust conversion completed successfully!")
    logger.info(f"Processing Statistics: {stats}")
    
    return stats

def main():
    """Main execution function with comprehensive logging and error handling"""
    # Define input and output paths
    input_file = "../../../../catalogs_processed/2025-06-08_snapshot.jsonl"
    output_dir = Path(".")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate descriptive output filename with current timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"cve_catalog_robust_{current_time}.csv"
    
    # Validate input file exists
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} does not exist")
        sys.exit(1)
    
    try:
        # Perform robust conversion
        stats = convert_jsonl_to_csv_robust(str(input_file), str(output_file))
        
        # Display comprehensive summary
        print(f"\n=== ROBUST CONVERSION SUMMARY ===")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Timestamp: {current_time}")
        print(f"\n=== PROCESSING STATISTICS ===")
        print(f"Total Records Processed: {stats['total_processed']:,}")
        print(f"Successful Records: {stats['successful_records']:,}")
        print(f"CVSS Library Parsed: {stats['cvss_parsed_successfully']:,}")
        print(f"CVSS Validation Matches: {stats['cvss_validation_matches']:,}")
        print(f"Error Records: {stats['error_records']:,}")
        print(f"JSON Decode Errors: {stats['json_decode_errors']:,}")
        
        if stats['cvss_parsed_successfully'] > 0:
            success_rate = (stats['cvss_validation_matches'] / stats['cvss_parsed_successfully']) * 100
            print(f"CVSS Validation Success Rate: {success_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 