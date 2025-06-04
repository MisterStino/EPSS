#!/usr/bin/env python3
"""
CVE Description Extractor - Batch Version

Extracts official CVE descriptions from cvelistV5 repository and adds them 
to the existing CVE catalog. Processes in batches and saves progress incrementally.

Usage:
    python extract_cve_descriptions_batch.py

Input:  cve_catalog.csv (296,387 CVEs)
Output: cve_catalog_with_descriptions.csv (same + description column)
"""

import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_cve_file_path(cve_id: str) -> Path:
    """Convert CVE ID to file path in cvelistV5 repository."""
    parts = cve_id.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid CVE ID format: {cve_id}")
    
    year = parts[1]
    number = parts[2]
    
    if len(number) <= 4:
        subdir = number[0] + "xxx"
    else:
        subdir = number[:2] + "xxx"
    
    return Path(f"../cvelistV5/cves/{year}/{subdir}/{cve_id}.json")

def extract_description_from_json(json_file_path: Path) -> str:
    """Extract English description from CVE JSON file."""
    try:
        if not json_file_path.exists():
            return ""
            
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
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
        
    except Exception as e:
        return ""

def process_batch(batch_df: pd.DataFrame, batch_num: int, total_batches: int) -> pd.DataFrame:
    """Process a batch of CVEs and extract descriptions."""
    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_df):,} CVEs)")
    
    descriptions = []
    missing_files = 0
    empty_descriptions = 0
    
    for cve_id in tqdm(batch_df['CVE_ID'], desc=f"Batch {batch_num}", leave=False):
        try:
            json_path = get_cve_file_path(cve_id)
            description = extract_description_from_json(json_path)
            
            if not json_path.exists():
                missing_files += 1
            elif not description:
                empty_descriptions += 1
                
            descriptions.append(description)
            
        except Exception as e:
            descriptions.append("")
            missing_files += 1
    
    # Add description column to batch
    result_batch = batch_df.copy()
    result_batch['description'] = descriptions
    
    # Log batch statistics
    with_descriptions = sum(1 for d in descriptions if d.strip())
    logger.info(f"Batch {batch_num} complete: {with_descriptions}/{len(batch_df)} with descriptions")
    
    return result_batch

def main():
    """Main execution function with batch processing."""
    logger.info("🔍 CVE Description Extractor (Batch Version) Starting")
    logger.info("=" * 70)
    
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
        return
    
    logger.info(f"✅ Found cvelistV5 repository")
    
    # Step 3: Process in batches
    batch_size = 10000  # Process 10K CVEs at a time
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    logger.info(f"🔄 Processing {len(df):,} CVEs in {total_batches} batches of {batch_size:,}")
    
    all_results = []
    start_time = time.time()
    
    for i in range(total_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        # Process batch
        batch_result = process_batch(batch_df, i + 1, total_batches)
        all_results.append(batch_result)
        
        # Save intermediate progress every 5 batches
        if (i + 1) % 5 == 0 or i == total_batches - 1:
            logger.info(f"💾 Saving progress after batch {i + 1}...")
            combined_df = pd.concat(all_results, ignore_index=True)
            temp_output = f"cve_catalog_progress_batch_{i + 1}.csv"
            combined_df.to_csv(temp_output, index=False)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            progress = (i + 1) / total_batches
            eta_seconds = (elapsed / progress) - elapsed if progress > 0 else 0
            eta_minutes = eta_seconds / 60
            
            logger.info(f"   Progress: {progress*100:.1f}% | ETA: {eta_minutes:.1f} minutes")
    
    # Step 4: Combine all results and save final output
    logger.info(f"🔗 Combining all batches...")
    final_df = pd.concat(all_results, ignore_index=True)
    
    output_path = "cve_catalog_with_descriptions.csv"
    logger.info(f"💾 Saving final catalog: {output_path}")
    final_df.to_csv(output_path, index=False)
    
    # Step 5: Final statistics
    total_cves = len(final_df)
    with_descriptions = len(final_df[final_df['description'].str.len() > 0])
    
    logger.info("=" * 70)
    logger.info("✅ CVE Description Extraction Complete!")
    logger.info(f"   Total CVEs: {total_cves:,}")
    logger.info(f"   With descriptions: {with_descriptions:,} ({with_descriptions/total_cves*100:.1f}%)")
    logger.info(f"   Output saved to: {output_path}")
    logger.info(f"   Total time: {(time.time() - start_time)/60:.1f} minutes")
    
    # Clean up intermediate files
    logger.info("🧹 Cleaning up intermediate files...")
    for i in range(total_batches):
        temp_file = f"cve_catalog_progress_batch_{i + 1}.csv"
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    main() 