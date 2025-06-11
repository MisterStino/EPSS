import os
import json
import csv
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Set, Tuple
import logging
from pathlib import Path
from tqdm import tqdm
import tempfile
import hashlib

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
        
        # Error tracking
        self.errors = []
        self.error_log_file = self.cache_dir / "processing_errors.csv"
        self._initialize_error_log()
        
    def _initialize_error_log(self):
        """Initialize the error log CSV file with headers."""
        if not self.error_log_file.exists():
            self._atomic_write_csv(self.error_log_file, [], [
                'timestamp', 'cve_id', 'operation', 'error_type', 'error_message', 'details'
            ])
    
    def _log_error(self, cve_id: str, operation: str, error_type: str, error_message: str, details: str = ""):
        """Log an error to both memory and CSV file."""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'cve_id': cve_id,
            'operation': operation,
            'error_type': error_type,
            'error_message': str(error_message),
            'details': details
        }
        
        self.errors.append(error_record)
        
        # Append to CSV atomically
        try:
            self._atomic_append_csv(self.error_log_file, [error_record])
        except Exception as e:
            logger.error(f"Failed to log error to CSV: {e}")
    
    def _atomic_write_json(self, file_path: Path, data: dict) -> bool:
        """Atomically write JSON data to file using .tmp pattern."""
        tmp_file = file_path.with_suffix(file_path.suffix + '.tmp')
        
        try:
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            tmp_file.replace(file_path)
            return True
            
        except Exception as e:
            # Clean up temp file if it exists
            if tmp_file.exists():
                tmp_file.unlink()
            raise e
    
    def _atomic_write_csv(self, file_path: Path, data: List[Dict], fieldnames: List[str]) -> bool:
        """Atomically write CSV data to file using .tmp pattern."""
        tmp_file = file_path.with_suffix(file_path.suffix + '.tmp')
        
        try:
            with open(tmp_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            # Atomic rename
            tmp_file.replace(file_path)
            return True
            
        except Exception as e:
            # Clean up temp file if it exists
            if tmp_file.exists():
                tmp_file.unlink()
            raise e
    
    def _atomic_append_csv(self, file_path: Path, data: List[Dict]) -> bool:
        """Atomically append CSV data to existing file."""
        if not file_path.exists():
            return False
        
        # Read existing data
        existing_data = []
        fieldnames = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                existing_data = list(reader)
        except Exception as e:
            logger.error(f"Failed to read existing CSV {file_path}: {e}")
            return False
        
        # Combine with new data
        combined_data = existing_data + data
        
        # Write atomically
        return self._atomic_write_csv(file_path, combined_data, fieldnames)
    
    def _validate_json_file(self, file_path: Path) -> Tuple[bool, str]:
        """Validate that a JSON file is properly formatted and complete."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic validation
            if not isinstance(data, (dict, list)):
                return False, "Invalid JSON structure"
            
            if isinstance(data, list) and len(data) == 0:
                return False, "Empty data list"
            
            return True, "Valid"
            
        except json.JSONDecodeError as e:
            return False, f"JSON decode error: {e}"
        except Exception as e:
            return False, f"File validation error: {e}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file contents for integrity checking."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""
    
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
        """Get complete change history for a CVE with atomic caching and error handling."""
        cache_file = self.cache_dir / f"changes_{cve_id.replace('-', '_')}.json"
        
        # Check cache first
        if cache_file.exists():
            try:
                # Validate cached file
                is_valid, validation_msg = self._validate_json_file(cache_file)
                if is_valid:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_changes = json.load(f)
                    logger.debug(f"Loaded {len(cached_changes)} changes from cache for {cve_id}")
                    return cached_changes
                else:
                    logger.warning(f"Invalid cached file for {cve_id}: {validation_msg}")
                    self._log_error(cve_id, "cache_validation", "invalid_cache", validation_msg)
                    # Remove invalid cache file
                    cache_file.unlink()
            except Exception as e:
                error_msg = f"Failed to load cache for {cve_id}: {e}"
                logger.warning(error_msg)
                self._log_error(cve_id, "cache_load", "cache_error", str(e))
        
        # Fetch from API if not cached or cache invalid
        params = {"cveId": cve_id}
        
        try:
            async with self.session.get(HISTORY_API_URL, params=params, headers=HEADERS) as response:
                if response.status == 200:
                    data = await response.json()
                    changes = []
                    
                    # Extract changes
                    for change_wrapper in data.get("cveChanges", []):
                        change = change_wrapper.get("change", {})
                        changes.append(change)
                    
                    # Sort by creation date (oldest first)
                    changes.sort(key=lambda x: x.get("created", ""))
                    
                    # Atomic cache write with validation
                    try:
                        if self._atomic_write_json(cache_file, changes):
                            # Validate written file
                            is_valid, validation_msg = self._validate_json_file(cache_file)
                            if is_valid:
                                logger.debug(f"Atomically cached {len(changes)} changes for {cve_id}")
                            else:
                                logger.error(f"Cache validation failed for {cve_id}: {validation_msg}")
                                self._log_error(cve_id, "cache_write_validation", "validation_failed", validation_msg)
                                # Remove invalid file
                                if cache_file.exists():
                                    cache_file.unlink()
                        else:
                            self._log_error(cve_id, "cache_write", "write_failed", "Atomic write returned False")
                            
                    except Exception as e:
                        error_msg = f"Failed to atomically cache changes for {cve_id}: {e}"
                        logger.warning(error_msg)
                        self._log_error(cve_id, "cache_write", "write_error", str(e))
                    
                    return changes
                    
                elif response.status == 429:
                    # Rate limit
                    error_msg = f"Rate limited for {cve_id}"
                    logger.warning(error_msg)
                    self._log_error(cve_id, "api_fetch", "rate_limit", "HTTP 429")
                    
                elif response.status == 404:
                    # CVE not found
                    error_msg = f"CVE {cve_id} not found (404)"
                    logger.warning(error_msg)
                    self._log_error(cve_id, "api_fetch", "not_found", "HTTP 404")
                    
                else:
                    # Other HTTP error
                    error_msg = f"HTTP {response.status} for {cve_id}"
                    logger.error(error_msg)
                    self._log_error(cve_id, "api_fetch", "http_error", f"HTTP {response.status}")
                    
        except asyncio.TimeoutError:
            error_msg = f"Timeout fetching change history for {cve_id}"
            logger.error(error_msg)
            self._log_error(cve_id, "api_fetch", "timeout", "Request timeout")
            
        except Exception as e:
            error_msg = f"Failed to get change history for {cve_id}: {e}"
            logger.error(error_msg)
            self._log_error(cve_id, "api_fetch", "exception", str(e))
        
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
        Reconstruct CVE snapshot at a specific date with atomic operations and error handling.
        """
        logger.info(f"Reconstructing {cve_id} at {target_date}")
        
        # Generate cache file path
        date_str = target_date.strftime("%Y%m%d")
        snapshot_cache_file = self.cache_dir / f"snapshot_{cve_id.replace('-', '_')}_{date_str}.json"
        
        # Check if reconstruction is already cached
        if snapshot_cache_file.exists():
            try:
                # Validate cached snapshot
                is_valid, validation_msg = self._validate_json_file(snapshot_cache_file)
                if is_valid:
                    with open(snapshot_cache_file, 'r', encoding='utf-8') as f:
                        cached_snapshot = json.load(f)
                    logger.debug(f"Loaded cached reconstruction for {cve_id} at {target_date}")
                    return cached_snapshot
                else:
                    logger.warning(f"Invalid cached snapshot for {cve_id}: {validation_msg}")
                    self._log_error(cve_id, "snapshot_cache_validation", "invalid_cache", validation_msg)
                    # Remove invalid cache
                    snapshot_cache_file.unlink()
            except Exception as e:
                error_msg = f"Failed to load cached snapshot for {cve_id}: {e}"
                logger.warning(error_msg)
                self._log_error(cve_id, "snapshot_cache_load", "cache_error", str(e))
        
        try:
            # Step 1: Get current snapshot
            if cve_id in self.current_cves:
                current_snapshot = self.current_cves[cve_id].copy()
            else:
                logger.warning(f"CVE {cve_id} not found in catalog, fetching from API")
                current_snapshot = await self.get_full_current_cve(cve_id)
                if not current_snapshot:
                    error_msg = f"Could not get current snapshot for {cve_id}"
                    logger.error(error_msg)
                    self._log_error(cve_id, "current_snapshot", "fetch_failed", "API returned None")
                    return None
            
            # Step 2: Get change history
            changes = await self.get_change_history(cve_id)
            if not changes:
                logger.info(f"No change history for {cve_id}, using current snapshot")
                # Cache the current snapshot atomically
                try:
                    if self._atomic_write_json(snapshot_cache_file, current_snapshot):
                        logger.debug(f"Atomically cached current snapshot for {cve_id}")
                    else:
                        self._log_error(cve_id, "current_snapshot_cache", "write_failed", "Atomic write returned False")
                except Exception as e:
                    error_msg = f"Failed to cache current snapshot for {cve_id}: {e}"
                    logger.warning(error_msg)
                    self._log_error(cve_id, "current_snapshot_cache", "write_error", str(e))
                
                return current_snapshot
            
            # Step 3: Apply changes in reverse chronological order
            reconstructed = current_snapshot.copy()
            changes_applied = 0
            
            # Process changes from newest to oldest
            for i, change in enumerate(reversed(changes)):
                try:
                    change_date = self.parse_change_date(change.get("created", ""))
                    
                    # If this change happened after our target date, reverse it
                    if change_date > target_date:
                        reconstructed = self.apply_reverse_change(reconstructed, change)
                        changes_applied += 1
                        logger.debug(f"Applied reverse change {i+1} from {change_date}")
                    else:
                        # We've reached changes that happened before target date, stop
                        break
                        
                except Exception as e:
                    error_msg = f"Failed to apply change {i} for {cve_id}: {e}"
                    logger.error(error_msg)
                    self._log_error(cve_id, "change_application", "processing_error", 
                                  f"Change {i}: {str(e)}")
                    # Continue with next change
                    continue
            
            logger.info(f"Reconstructed {cve_id} at {target_date}: applied {changes_applied} reverse changes")
            
            # Add reconstruction metadata
            reconstructed["reconstruction_date"] = target_date.isoformat()
            reconstructed["changes_applied"] = changes_applied
            reconstructed["total_changes"] = len(changes)
            reconstructed["reconstruction_timestamp"] = datetime.now().isoformat()
            reconstructed["cache_file_hash"] = self._calculate_file_hash(snapshot_cache_file) if snapshot_cache_file.exists() else ""
            
            # Atomic cache write with validation
            try:
                if self._atomic_write_json(snapshot_cache_file, reconstructed):
                    # Validate written file
                    is_valid, validation_msg = self._validate_json_file(snapshot_cache_file)
                    if is_valid:
                        logger.debug(f"Atomically cached reconstruction for {cve_id} at {target_date}")
                    else:
                        logger.error(f"Reconstruction cache validation failed for {cve_id}: {validation_msg}")
                        self._log_error(cve_id, "reconstruction_cache_validation", "validation_failed", validation_msg)
                        # Remove invalid file
                        if snapshot_cache_file.exists():
                            snapshot_cache_file.unlink()
                else:
                    self._log_error(cve_id, "reconstruction_cache", "write_failed", "Atomic write returned False")
                    
            except Exception as e:
                error_msg = f"Failed to atomically cache reconstruction for {cve_id}: {e}"
                logger.warning(error_msg)
                self._log_error(cve_id, "reconstruction_cache", "write_error", str(e))
            
            return reconstructed
            
        except Exception as e:
            error_msg = f"Fatal error reconstructing {cve_id} at {target_date}: {e}"
            logger.error(error_msg)
            self._log_error(cve_id, "reconstruction", "fatal_error", str(e))
            return None
    
    async def reconstruct_multiple_cves_at_date(self, cve_ids: List[str], target_date: datetime) -> Dict[str, Dict]:
        """Reconstruct multiple CVEs at a specific date with comprehensive error tracking."""
        results = {}
        failures = 0
        
        # Add progress bar
        with tqdm(total=len(cve_ids), desc=f"Reconstructing CVEs at {target_date.strftime('%Y-%m-%d')}", unit="CVE") as pbar:
            for cve_id in cve_ids:
                try:
                    reconstructed = await self.reconstruct_cve_at_date(cve_id, target_date)
                    if reconstructed:
                        results[cve_id] = reconstructed
                        pbar.set_postfix({
                            "Success": len(results), 
                            "Failed": failures, 
                            "Current": cve_id[:12]  # Truncate for display
                        })
                    else:
                        failures += 1
                        self._log_error(cve_id, "batch_reconstruction", "reconstruction_failed", 
                                      f"Returned None for date {target_date}")
                        pbar.set_postfix({
                            "Success": len(results), 
                            "Failed": failures, 
                            "Error": cve_id[:12]
                        })
                    
                    # Rate limiting
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    failures += 1
                    error_msg = f"Exception during reconstruction of {cve_id}: {e}"
                    logger.error(error_msg)
                    self._log_error(cve_id, "batch_reconstruction", "exception", str(e))
                    pbar.set_postfix({
                        "Success": len(results), 
                        "Failed": failures, 
                        "Exception": cve_id[:12]
                    })
                
                # Update progress bar
                pbar.update(1)
        
        logger.info(f"Batch reconstruction complete: {len(results)} success, {failures} failures")
        return results
    
    async def batch_reconstruct_cves(self, cve_ids: List[str], date_range: List[datetime], 
                                   output_dir: str = "catalogs_processed", 
                                   save_each_date: bool = True) -> Dict[str, Dict[str, Dict]]:
        """
        Batch reconstruct CVEs across multiple dates with comprehensive error tracking.
        
        Args:
            cve_ids: List of CVE IDs to reconstruct
            date_range: List of target dates for reconstruction
            output_dir: Directory to save results
            save_each_date: Whether to save CSV for each date
            
        Returns:
            Dict mapping date_str -> {cve_id -> reconstructed_snapshot}
        """
        all_results = {}
        total_success = 0
        total_failures = 0
        
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
                
                try:
                    # Reconstruct all CVEs for this date
                    results = await self.reconstruct_multiple_cves_at_date(cve_ids, target_date)
                    all_results[date_str] = results
                    
                    # Update counters
                    date_success = len(results)
                    date_failures = len(cve_ids) - date_success
                    total_success += date_success
                    total_failures += date_failures
                    
                    # Save results for this date if requested
                    if save_each_date and results:
                        try:
                            self.save_reconstructed_snapshots(results, target_date, output_dir)
                        except Exception as e:
                            error_msg = f"Failed to save snapshots for {date_str}: {e}"
                            logger.error(error_msg)
                            self._log_error("N/A", "batch_save", "save_error", f"Date {date_str}: {str(e)}")
                    
                    date_pbar.set_postfix({
                        "Date": date_str, 
                        "Success": date_success,
                        "Failed": date_failures,
                        "Total": f"{total_success}/{total_success + total_failures}"
                    })
                    
                except Exception as e:
                    error_msg = f"Fatal error processing date {date_str}: {e}"
                    logger.error(error_msg)
                    self._log_error("N/A", "batch_date_processing", "fatal_error", f"Date {date_str}: {str(e)}")
                    all_results[date_str] = {}
                    total_failures += len(cve_ids)
                    
                    date_pbar.set_postfix({
                        "Date": date_str, 
                        "Error": "FATAL",
                        "Total": f"{total_success}/{total_success + total_failures}"
                    })
                
                date_pbar.update(1)
        
        # Final summary
        logger.info(f"Batch reconstruction complete:")
        logger.info(f"  - Dates processed: {len(all_results)}")
        logger.info(f"  - Total reconstructions successful: {total_success}")
        logger.info(f"  - Total failures: {total_failures}")
        logger.info(f"  - Success rate: {total_success/(total_success + total_failures)*100:.1f}%")
        
        # Error summary
        error_summary = self.get_error_summary()
        if error_summary['total_errors'] > 0:
            logger.warning(f"Total errors encountered: {error_summary['total_errors']}")
            logger.warning(f"Error log saved to: {error_summary['error_log_file']}")
            logger.info(f"Error breakdown by type: {error_summary['error_types']}")
        
        return all_results
    
    def save_reconstructed_snapshots(self, snapshots: Dict[str, Dict], target_date: datetime, output_dir: str = "catalogs_processed"):
        """Save reconstructed snapshots to CSV with atomic operations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        date_str = target_date.strftime("%Y%m%d")
        filename = output_path / f"cve_snapshots_{date_str}.csv"
        
        if not snapshots:
            logger.warning("No snapshots to save")
            self._log_error("N/A", "save_snapshots", "no_data", f"No snapshots for {target_date}")
            return
        
        try:
            # Convert to list of dictionaries for CSV
            rows = []
            for cve_id, snapshot in snapshots.items():
                try:
                    # Add composite key
                    snapshot_copy = snapshot.copy()
                    snapshot_copy["composite_key"] = f"{date_str}_{cve_id}"
                    rows.append(snapshot_copy)
                except Exception as e:
                    error_msg = f"Failed to process snapshot for {cve_id}: {e}"
                    logger.error(error_msg)
                    self._log_error(cve_id, "snapshot_processing", "processing_error", str(e))
                    continue
            
            if not rows:
                logger.warning("No valid snapshots to save after processing")
                self._log_error("N/A", "save_snapshots", "no_valid_data", f"No valid snapshots for {target_date}")
                return
            
            # Get fieldnames from first row
            fieldnames = list(rows[0].keys()) if rows else []
            
            # Atomic CSV write
            if self._atomic_write_csv(filename, rows, fieldnames):
                logger.info(f"Atomically saved {len(rows)} reconstructed snapshots to {filename}")
            else:
                error_msg = f"Failed to atomically save snapshots to {filename}"
                logger.error(error_msg)
                self._log_error("N/A", "save_snapshots", "write_failed", f"Atomic CSV write failed for {filename}")
                
        except Exception as e:
            error_msg = f"Fatal error saving snapshots for {target_date}: {e}"
            logger.error(error_msg)
            self._log_error("N/A", "save_snapshots", "fatal_error", str(e))

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

    def get_error_summary(self) -> Dict[str, any]:
        """Get comprehensive error summary and statistics."""
        if not self.errors:
            return {"total_errors": 0, "error_types": {}, "operations": {}, "cves_with_errors": []}
        
        error_types = {}
        operations = {}
        cve_errors = {}
        
        for error in self.errors:
            # Count by error type
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Count by operation
            operation = error['operation']
            operations[operation] = operations.get(operation, 0) + 1
            
            # Track CVEs with errors
            cve_id = error['cve_id']
            if cve_id != "N/A":
                if cve_id not in cve_errors:
                    cve_errors[cve_id] = []
                cve_errors[cve_id].append(error)
        
        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "operations": operations,
            "cves_with_errors": list(cve_errors.keys()),
            "cve_error_details": cve_errors,
            "error_log_file": str(self.error_log_file)
        }

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
            cve_ids = list(cves.keys())[:100]  # Limit to first 100 for testing
            
            logger.info(f"Processing {len(cve_ids)} CVEs across {len(date_range)} dates")
            await reconstructor.batch_reconstruct_cves(cve_ids, date_range)
    
    asyncio.run(main())