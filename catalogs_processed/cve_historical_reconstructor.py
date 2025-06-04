import os
import json
import csv
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set
import logging
from pathlib import Path
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
HISTORY_API_URL = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0"
VULN_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
API_KEY = os.getenv("NVD_API_KEY")
HEADERS = {"apiKey": API_KEY} if API_KEY else {}

class CVEHistoricalReconstructor:
    def __init__(self, catalog_path: str = None, cache_dir: str = "catalogs_processed/cache"):
        self.catalog_path = catalog_path or self._find_catalog()
        self.current_cves = {}
        self.session = None
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _find_catalog(self) -> str:
        """Find the CVE catalog file."""
        possible_paths = [
            'processed_data/enriched_catalog.csv',
            '../processed_data/enriched_catalog.csv',
            os.path.join('..', 'cve_catalog', 'cve_catalog.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find CVE catalog file")
    
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def load_current_cve_catalog(self) -> Dict[str, Dict]:
        """Load current CVE snapshots from catalog."""
        logger.info(f"Loading current CVE catalog from {self.catalog_path}")
        
        df = pd.read_csv(self.catalog_path)
        current_cves = {}
        
        for _, row in df.iterrows():
            cve_id = row['CVE_ID']
            current_cves[cve_id] = {
                'CVE_ID': cve_id,
                'state': row.get('state', ''),
                'assigner': row.get('assigner', ''),
                'date_published': row.get('date_published', ''),
                'date_updated': row.get('date_updated', ''),
                'cvss_score': row.get('cvss_score', ''),
                'cvss_version': row.get('cvss_version', ''),
                'cwes': row.get('cwes', ''),
                'references': row.get('references', '')
            }
        
        logger.info(f"Loaded {len(current_cves)} current CVE snapshots")
        self.current_cves = current_cves
        return current_cves
    
    async def get_full_current_cve(self, cve_id: str) -> Optional[Dict]:
        """Get full current CVE data from NVD Vulnerabilities API."""
        params = {"cveId": cve_id}
        
        try:
            async with self.session.get(VULN_API_URL, params=params, headers=HEADERS) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("vulnerabilities"):
                        return data["vulnerabilities"][0]["cve"]
        except Exception as e:
            logger.error(f"Failed to get current CVE {cve_id}: {e}")
        
        return None
    
    async def get_change_history(self, cve_id: str) -> List[Dict]:
        """Get complete change history for a CVE with caching."""
        # Check cache first
        cache_file = self.cache_dir / f"changes_{cve_id.replace('-', '_')}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_changes = json.load(f)
                logger.debug(f"Loaded {len(cached_changes)} changes from cache for {cve_id}")
                return cached_changes
            except Exception as e:
                logger.warning(f"Failed to load cache for {cve_id}: {e}")
        
        # Fetch from API if not cached
        params = {"cveId": cve_id}
        
        try:
            async with self.session.get(HISTORY_API_URL, params=params, headers=HEADERS) as response:
                if response.status == 200:
                    data = await response.json()
                    changes = []
                    
                    for change_wrapper in data.get("cveChanges", []):
                        change = change_wrapper.get("change", {})
                        changes.append(change)
                    
                    # Sort by creation date (oldest first)
                    changes.sort(key=lambda x: x.get("created", ""))
                    
                    # Cache the results
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(changes, f, indent=2)
                        logger.debug(f"Cached {len(changes)} changes for {cve_id}")
                    except Exception as e:
                        logger.warning(f"Failed to cache changes for {cve_id}: {e}")
                    
                    return changes
        except Exception as e:
            logger.error(f"Failed to get change history for {cve_id}: {e}")
        
        return []
    
    def parse_change_date(self, date_str: str) -> datetime:
        """Parse NVD date string to datetime object."""
        try:
            # Handle different date formats from NVD
            if date_str.endswith('Z'):
                return datetime.fromisoformat(date_str[:-1])
            else:
                return datetime.fromisoformat(date_str)
        except:
            return datetime.min
    
    def apply_reverse_change(self, current_snapshot: Dict, change: Dict) -> Dict:
        """Apply a change in reverse (newValue → oldValue)."""
        reconstructed = current_snapshot.copy()
        
        for detail in change.get("details", []):
            change_type = detail.get("type", "")
            old_value = detail.get("oldValue", "")
            new_value = detail.get("newValue", "")
            
            # Map NVD change types to our catalog fields
            field_mapping = {
                "Description": "description",
                "CWE": "cwes", 
                "CVSS V2": "cvss_v2_score",
                "CVSS V3.1": "cvss_v3_score",
                "CVSS V4.0": "cvss_v4_score",
                "Reference": "references",
                "CPE Configuration": "cpes"
            }
            
            if change_type in field_mapping:
                field_name = field_mapping[change_type]
                
                # Apply reverse change: replace current value with old value
                if field_name in reconstructed:
                    logger.debug(f"Reversing {change_type}: '{new_value}' → '{old_value}'")
                    reconstructed[field_name] = old_value
                else:
                    # Add the field if it doesn't exist
                    reconstructed[field_name] = old_value
            
            # Handle special cases for complex fields
            elif change_type.startswith("CVSS"):
                # Extract CVSS score from complex CVSS data
                if "baseScore" in old_value:
                    try:
                        # Parse CVSS score from JSON-like string
                        import re
                        score_match = re.search(r'"baseScore":\s*([0-9.]+)', old_value)
                        if score_match:
                            score = float(score_match.group(1))
                            if "V2" in change_type:
                                reconstructed["cvss_v2_score"] = score
                            elif "V3" in change_type:
                                reconstructed["cvss_v3_score"] = score
                            elif "V4" in change_type:
                                reconstructed["cvss_v4_score"] = score
                    except:
                        pass
        
        return reconstructed
    
    async def reconstruct_cve_at_date(self, cve_id: str, target_date: datetime) -> Optional[Dict]:
        """
        Reconstruct CVE snapshot at a specific date.
        
        Algorithm:
        1. Check if reconstruction is already cached
        2. Start with current CVE snapshot
        3. Get all change history
        4. Apply changes in reverse chronological order
        5. Stop when we reach target_date
        6. Cache the result
        """
        logger.info(f"Reconstructing {cve_id} at {target_date}")
        
        # Check if reconstruction is already cached
        date_str = target_date.strftime("%Y%m%d")
        snapshot_cache_file = self.cache_dir / f"snapshot_{cve_id.replace('-', '_')}_{date_str}.json"
        
        if snapshot_cache_file.exists():
            try:
                with open(snapshot_cache_file, 'r', encoding='utf-8') as f:
                    cached_snapshot = json.load(f)
                logger.debug(f"Loaded cached reconstruction for {cve_id} at {target_date}")
                return cached_snapshot
            except Exception as e:
                logger.warning(f"Failed to load cached snapshot for {cve_id}: {e}")
        
        # Step 1: Get current snapshot
        if cve_id in self.current_cves:
            current_snapshot = self.current_cves[cve_id].copy()
        else:
            logger.warning(f"CVE {cve_id} not found in catalog, fetching from API")
            current_snapshot = await self.get_full_current_cve(cve_id)
            if not current_snapshot:
                logger.error(f"Could not get current snapshot for {cve_id}")
                return None
        
        # Step 2: Get change history
        changes = await self.get_change_history(cve_id)
        if not changes:
            logger.info(f"No change history for {cve_id}, returning current snapshot")
            # Cache the current snapshot
            try:
                with open(snapshot_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(current_snapshot, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to cache snapshot for {cve_id}: {e}")
            return current_snapshot
        
        # Step 3: Apply changes in reverse chronological order
        reconstructed = current_snapshot.copy()
        changes_applied = 0
        
        # Process changes from newest to oldest
        for change in reversed(changes):
            change_date = self.parse_change_date(change.get("created", ""))
            
            # If this change happened after our target date, reverse it
            if change_date > target_date:
                reconstructed = self.apply_reverse_change(reconstructed, change)
                changes_applied += 1
                logger.debug(f"Applied reverse change from {change_date}")
            else:
                # We've reached changes that happened before target date, stop
                break
        
        logger.info(f"Reconstructed {cve_id} at {target_date}: applied {changes_applied} reverse changes")
        
        # Add reconstruction metadata
        reconstructed["reconstruction_date"] = target_date.isoformat()
        reconstructed["changes_applied"] = changes_applied
        reconstructed["total_changes"] = len(changes)
        
        # Cache the reconstructed snapshot
        try:
            with open(snapshot_cache_file, 'w', encoding='utf-8') as f:
                json.dump(reconstructed, f, indent=2)
            logger.debug(f"Cached reconstruction for {cve_id} at {target_date}")
        except Exception as e:
            logger.warning(f"Failed to cache reconstruction for {cve_id}: {e}")
        
        return reconstructed
    
    async def reconstruct_multiple_cves_at_date(self, cve_ids: List[str], target_date: datetime) -> Dict[str, Dict]:
        """Reconstruct multiple CVEs at a specific date."""
        results = {}
        
        # Add progress bar
        with tqdm(total=len(cve_ids), desc=f"Reconstructing CVEs at {target_date.strftime('%Y-%m-%d')}", unit="CVE") as pbar:
            for cve_id in cve_ids:
                try:
                    reconstructed = await self.reconstruct_cve_at_date(cve_id, target_date)
                    if reconstructed:
                        results[cve_id] = reconstructed
                        pbar.set_postfix({"Success": len(results), "Current": cve_id})
                    else:
                        pbar.set_postfix({"Success": len(results), "Failed": cve_id})
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Failed to reconstruct {cve_id}: {e}")
                    pbar.set_postfix({"Success": len(results), "Error": cve_id})
                
                # Update progress bar
                pbar.update(1)
        
        return results
    
    async def batch_reconstruct_cves(self, cve_ids: List[str], date_range: List[datetime], 
                                   output_dir: str = "catalogs_processed", 
                                   save_each_date: bool = True) -> Dict[str, Dict[str, Dict]]:
        """
        Batch reconstruct CVEs across multiple dates.
        
        Args:
            cve_ids: List of CVE IDs to reconstruct
            date_range: List of target dates for reconstruction
            output_dir: Directory to save results
            save_each_date: Whether to save CSV for each date
            
        Returns:
            Dict mapping date_str -> {cve_id -> reconstructed_snapshot}
        """
        all_results = {}
        
        logger.info(f"Starting batch reconstruction for {len(cve_ids)} CVEs across {len(date_range)} dates")
        
        # Check cache status
        cache_status = self.get_cache_status(cve_ids, date_range)
        logger.info(f"Change history cache hit rate: {cache_status['changes_cache_hit_rate']:.1%}")
        if 'snapshots_cache_hit_rate' in cache_status:
            logger.info(f"Snapshot cache hit rate: {cache_status['snapshots_cache_hit_rate']:.1%}")
        
        # Progress bar for dates
        with tqdm(total=len(date_range), desc="Processing dates", unit="date", position=0) as date_pbar:
            for target_date in date_range:
                date_str = target_date.strftime('%Y-%m-%d')
                date_pbar.set_description(f"Processing {date_str}")
                
                # Reconstruct all CVEs for this date
                results = await self.reconstruct_multiple_cves_at_date(cve_ids, target_date)
                all_results[date_str] = results
                
                # Save results for this date if requested
                if save_each_date and results:
                    self.save_reconstructed_snapshots(results, target_date, output_dir)
                
                date_pbar.set_postfix({
                    "Date": date_str, 
                    "CVEs": len(results), 
                    "Total": f"{sum(len(r) for r in all_results.values())}"
                })
                date_pbar.update(1)
        
        logger.info(f"Batch reconstruction complete. Processed {len(all_results)} dates")
        return all_results
    
    def save_reconstructed_snapshots(self, snapshots: Dict[str, Dict], target_date: datetime, output_dir: str = "catalogs_processed"):
        """Save reconstructed snapshots to CSV."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        date_str = target_date.strftime("%Y%m%d")
        filename = output_path / f"cve_snapshots_{date_str}.csv"
        
        if not snapshots:
            logger.warning("No snapshots to save")
            return
        
        # Convert to list of dictionaries for CSV
        rows = []
        for cve_id, snapshot in snapshots.items():
            # Add composite key
            snapshot["composite_key"] = f"{date_str}_{cve_id}"
            rows.append(snapshot)
        
        # Write to CSV
        fieldnames = list(rows[0].keys()) if rows else []
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Saved {len(rows)} reconstructed snapshots to {filename}")

    def get_cache_status(self, cve_ids: List[str], target_dates: List[datetime] = None) -> Dict[str, any]:
        """Check which CVEs are cached vs need to be fetched."""
        changes_cached = []
        changes_need_fetch = []
        
        # Check change history cache
        for cve_id in cve_ids:
            cache_file = self.cache_dir / f"changes_{cve_id.replace('-', '_')}.json"
            if cache_file.exists():
                changes_cached.append(cve_id)
            else:
                changes_need_fetch.append(cve_id)
        
        result = {
            "changes_cached": changes_cached,
            "changes_need_fetch": changes_need_fetch,
            "changes_cache_hit_rate": len(changes_cached) / len(cve_ids) if cve_ids else 0
        }
        
        # Check snapshot cache if target dates provided
        if target_dates:
            snapshots_cached = 0
            snapshots_total = len(cve_ids) * len(target_dates)
            
            for cve_id in cve_ids:
                for target_date in target_dates:
                    date_str = target_date.strftime("%Y%m%d")
                    snapshot_cache_file = self.cache_dir / f"snapshot_{cve_id.replace('-', '_')}_{date_str}.json"
                    if snapshot_cache_file.exists():
                        snapshots_cached += 1
            
            result.update({
                "snapshots_cached": snapshots_cached,
                "snapshots_total": snapshots_total,
                "snapshots_cache_hit_rate": snapshots_cached / snapshots_total if snapshots_total else 0
            })
        
        logger.info(f"Cache status: {len(changes_cached)} change histories cached, {len(changes_need_fetch)} need fetch")
        if target_dates:
            logger.info(f"Snapshot cache: {snapshots_cached}/{snapshots_total} ({result['snapshots_cache_hit_rate']:.1%})")
        
        return result

# Example usage and testing
async def test_reconstruction():
    """Test the reconstruction system."""
    async with CVEHistoricalReconstructor() as reconstructor:
        # Load current catalog
        reconstructor.load_current_cve_catalog()
        
        # Test with a specific CVE and date
        test_cve = "CVE-2021-36767"
        test_date = datetime(2022, 6, 1)  # June 1, 2022
        
        print(f"Testing reconstruction of {test_cve} at {test_date}")
        
        # Reconstruct
        reconstructed = await reconstructor.reconstruct_cve_at_date(test_cve, test_date)
        
        if reconstructed:
            print(f"\n✅ Reconstruction successful!")
            print(f"Changes applied: {reconstructed.get('changes_applied', 0)}")
            print(f"Total changes: {reconstructed.get('total_changes', 0)}")
            
            # Show key fields
            print(f"\nReconstructed snapshot:")
            for key in ['CVE_ID', 'cvss_score', 'cwes', 'reconstruction_date']:
                if key in reconstructed:
                    value = str(reconstructed[key])[:100]
                    print(f"  {key}: {value}")
        else:
            print("❌ Reconstruction failed")

if __name__ == "__main__":

    async def main():
        async with CVEHistoricalReconstructor() as reconstructor:

            # get historical reconstructed cves for feb 2 2022 , until march 15 2025
            start_date = datetime(2022, 2, 2)
            end_date = datetime(2025, 3, 15)
            
            # Generate monthly date range
            date_range = []
            current_date = start_date
            while current_date <= end_date:
                date_range.append(current_date)
                # Add roughly 30 days
                current_date += timedelta(days=30)
            
            cves = reconstructor.load_current_cve_catalog()
            cve_ids = list(cves.keys())[300]  # Limit to first 100 for testing
            
            logger.info(f"Processing {len(cve_ids)} CVEs across {len(date_range)} dates")
            await reconstructor.batch_reconstruct_cves(cve_ids, date_range)
    
    asyncio.run(main())