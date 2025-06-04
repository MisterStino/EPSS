#!/usr/bin/env python3
"""
CVE Description Extractor

Extracts official CVE descriptions from cvelistV5 repository and adds them 
to the existing CVE catalog as a new 'description' column.

Usage:
    python extract_cve_descriptions.py

Input:  catalogs_raw/cve_catalog.csv (296,387 CVEs)
Output: catalogs_raw/cve_catalog_with_descriptions.csv (same + description column)
"""

import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cve_file_path(cve_id: str) -> Path:
    """
    Convert CVE ID to file path in cvelistV5 repository.
    
    Args:
        cve_id: CVE ID like 'CVE-2024-0999'
        
    Returns:
        Path to the JSON file
        
    Example:
        CVE-2024-0999 -> cvelistV5/cves/2024/0xxx/CVE-2024-0999.json
    """
    # Parse CVE ID: CVE-YYYY-NNNNN
    parts = cve_id.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid CVE ID format: {cve_id}")
    
    year = parts[1]
    number = parts[2]
    
    # Determine subdirectory based on number
    # Examples: 0999 -> 0xxx, 1234 -> 1xxx, 12345 -> 12xxx
    if len(number) <= 4:
        subdir = number[0] + "xxx"
    else:
        # For 5+ digit numbers, use first 2 digits
        subdir = number[:2] + "xxx"
    
    return Path(f"../cvelistV5/cves/{year}/{subdir}/{cve_id}.json")

def extract_description_from_json(json_file_path: Path) -> str:
    """
    Extract English description from CVE JSON file.
    
    Args:
        json_file_path: Path to CVE JSON file
        
    Returns:
        English description text, or empty string if not found
    """
    try:
        if not json_file_path.exists():
            return ""
            
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Navigate to descriptions: containers.cna.descriptions[]
        containers = data.get('containers', {})
        cna = containers.get('cna', {})
        descriptions = cna.get('descriptions', [])
        
        # Find English description
        for desc in descriptions:
            if desc.get('lang') == 'en':
                return desc.get('value', '').strip()
        
        # If no English description, take the first available
        if descriptions:
            return descriptions[0].get('value', '').strip()
            
        return ""
        
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        logger.warning(f"Error processing {json_file_path}: {e}")
        return ""

def extract_descriptions_for_catalog(catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract descriptions for all CVEs in the catalog.
    
    Args:
        catalog_df: DataFrame with CVE_ID column
        
    Returns:
        DataFrame with added 'description' column
    """
    logger.info(f"Extracting descriptions for {len(catalog_df):,} CVEs...")
    
    descriptions = []
    missing_files = 0
    empty_descriptions = 0
    
    for cve_id in tqdm(catalog_df['CVE_ID'], desc="Extracting descriptions"):
        try:
            json_path = get_cve_file_path(cve_id)
            description = extract_description_from_json(json_path)
            
            if not json_path.exists():
                missing_files += 1
            elif not description:
                empty_descriptions += 1
                
            descriptions.append(description)
            
        except Exception as e:
            logger.warning(f"Error processing {cve_id}: {e}")
            descriptions.append("")
            missing_files += 1
    
    # Add description column
    result_df = catalog_df.copy()
    result_df['description'] = descriptions
    
    # Statistics
    total_cves = len(catalog_df)
    with_descriptions = sum(1 for d in descriptions if d.strip())
    
    logger.info(f"Extraction complete:")
    logger.info(f"  Total CVEs: {total_cves:,}")
    logger.info(f"  With descriptions: {with_descriptions:,} ({with_descriptions/total_cves*100:.1f}%)")
    logger.info(f"  Missing files: {missing_files:,}")
    logger.info(f"  Empty descriptions: {empty_descriptions:,}")
    
    return result_df

def main():
    """Main execution function."""
    logger.info("🔍 CVE Description Extractor Starting")
    logger.info("=" * 60)
    
    # Step 1: Load existing catalog
    catalog_path = "cve_catalog.csv"
    logger.info(f"📂 Loading catalog: {catalog_path}")
    
    if not os.path.exists(catalog_path):
        logger.error(f"Catalog file not found: {catalog_path}")
        return
    
    df = pd.read_csv(catalog_path)
    logger.info(f"   Loaded {len(df):,} CVEs")
    logger.info(f"   Columns: {list(df.columns)}")
    
    # Step 2: Check cvelistV5 repository
    cvelist_path = Path("../cvelistV5")
    if not cvelist_path.exists():
        logger.error(f"cvelistV5 repository not found at: {cvelist_path}")
        logger.error("Please ensure the cvelistV5 repository is cloned in the workspace")
        return
    
    logger.info(f"✅ Found cvelistV5 repository at: {cvelist_path}")
    
    # Step 3: Extract descriptions
    logger.info(f"🔄 Extracting descriptions from JSON files...")
    enhanced_df = extract_descriptions_for_catalog(df)
    
    # Step 4: Save enhanced catalog
    output_path = "cve_catalog_with_descriptions.csv"
    logger.info(f"💾 Saving enhanced catalog: {output_path}")
    enhanced_df.to_csv(output_path, index=False)
    
    # Step 5: Show sample results
    logger.info(f"📊 Sample results:")
    sample_with_desc = enhanced_df[enhanced_df['description'].str.len() > 0].head(3)
    
    for _, row in sample_with_desc.iterrows():
        logger.info(f"   {row['CVE_ID']}: {row['description'][:100]}...")
    
    logger.info("=" * 60)
    logger.info("✅ CVE Description Extraction Complete!")
    logger.info(f"   Enhanced catalog saved to: {output_path}")

if __name__ == "__main__":
    main() 