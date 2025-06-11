import os
import json
import csv
import asyncio
import aiohttp
import logging
import math
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
HISTORY_API_URL = "https://services.nvd.nist.gov/rest/json/cvehistory/2.0"
API_KEY = os.getenv("NVD_API_KEY")
HEADERS = {"apiKey": API_KEY} if API_KEY else {}

class CVEChunkProcessor:
    """
    Production-ready CVE change history processor with atomic file operations.
    
    This class implements the industry-standard .tmp pattern for data integrity:
    - Creates .tmp files during processing
    - Only renames to final when 100% complete
    - Handles full pagination for each chunk
    - Ensures no partial data in your pipeline
    
    Key Features:
    - Atomic file operations (.tmp pattern)
    - Complete pagination handling
    - Resumable processing (skip completed chunks)
    - Data integrity validation
    - Comprehensive error handling
    
    Example Usage:
        async with CVEChunkProcessor() as processor:
            await processor.process_single_chunk(
                datetime(2022, 2, 1),
                datetime(2022, 5, 31)
            )
    """
    
    def __init__(self, output_dir: str = "catalogs_processed/chunk_results"):
        """
        Initialize the CVE Chunk Processor.
        
        Args:
            output_dir: Directory where chunk result files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        
        # API constraints from NVD documentation
        self.max_date_range_days = 120
        self.max_results_per_page = 5000
        
        logger.info(f"CVE Chunk Processor initialized. Output directory: {self.output_dir}")
        
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for large responses
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def generate_chunk_filename(self, start_date: datetime, end_date: datetime) -> str:
        """
        Generate human-readable filename for a date chunk.
        
        Args:
            start_date: Start of the chunk
            end_date: End of the chunk
            
        Returns:
            Filename like "2022_02_01_to_2022_05_31"
        """
        start_str = start_date.strftime('%Y_%m_%d')
        end_str = end_date.strftime('%Y_%m_%d')
        return f"{start_str}_to_{end_str}"
    
    def format_datetime_for_api(self, dt: datetime) -> str:
        """
        Format datetime for NVD API compliance.
        
        Args:
            dt: DateTime object to format
            
        Returns:
            API-compliant datetime string like "2022-02-01T00:00:00.000Z"
        """
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    async def get_api_page(
        self,
        start_date: datetime,
        end_date: datetime,
        start_index: int,
        event_name: Optional[str] = None
    ) -> Dict:
        """
        Get a single page from the NVD CVE Change History API.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            start_index: Starting index for pagination
            event_name: Optional event filter
            
        Returns:
            Raw API response data
            
        Raises:
            Exception: If API request fails
        """
        params = {
            "changeStartDate": self.format_datetime_for_api(start_date),
            "changeEndDate": self.format_datetime_for_api(end_date),
            "startIndex": start_index,
            "resultsPerPage": self.max_results_per_page
        }
        
        if event_name:
            params["eventName"] = event_name
        
        logger.debug(f"API request: start_index={start_index}, params={params}")
        
        async with self.session.get(HISTORY_API_URL, params=params, headers=HEADERS) as response:
            if response.status == 200:
                data = await response.json()
                return data
            else:
                error_text = await response.text()
                raise Exception(f"API request failed with status {response.status}: {error_text}")
    
    def extract_changes_from_response(self, api_response: Dict) -> List[Dict]:
        """
        Extract change records from API response.
        
        Args:
            api_response: Raw API response
            
        Returns:
            List of change records
        """
        changes = []
        for change_wrapper in api_response.get("cveChanges", []):
            change = change_wrapper.get("change", {})
            changes.append(change)
        return changes
    
    def write_changes_to_csv(self, changes: List[Dict], file_path: Path) -> None:
        """
        Write change records to CSV file.
        
        Args:
            changes: List of change records
            file_path: Path where to write the CSV file
        """
        if not changes:
            logger.warning("No changes to write")
            return
        
        # Determine all possible fieldnames from the data
        all_fieldnames = set()
        for change in changes:
            all_fieldnames.update(change.keys())
        
        # Sort fieldnames for consistent output
        fieldnames = sorted(list(all_fieldnames))
        
        logger.info(f"Writing {len(changes)} changes to {file_path}")
        logger.debug(f"CSV fields: {fieldnames}")
        
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for change in changes:
                # Handle nested objects by converting to JSON strings
                row = {}
                for field in fieldnames:
                    value = change.get(field, '')
                    if isinstance(value, (dict, list)):
                        row[field] = json.dumps(value)
                    else:
                        row[field] = value
                writer.writerow(row)
        
        logger.info(f"✅ Successfully wrote {len(changes)} records to {file_path}")
    
    async def process_single_chunk(
        self,
        start_date: datetime,
        end_date: datetime,
        event_name: Optional[str] = None
    ) -> bool:
        """
        Process a single date chunk with atomic file operations.
        
        This implements the industry-standard .tmp pattern:
        1. Check if chunk already complete (final file exists)
        2. Create .tmp file
        3. Get ALL pages for the chunk
        4. Validate data completeness
        5. Write complete dataset to .tmp file
        6. Atomically rename .tmp to final (commit)
        
        Args:
            start_date: Start of date range (must be ≤120 days from end_date)
            end_date: End of date range
            event_name: Optional event filter
            
        Returns:
            True if chunk was processed, False if already complete
            
        Raises:
            Exception: If chunk processing fails
        """
        # Validate date range
        days_in_range = (end_date - start_date).days + 1
        if days_in_range > self.max_date_range_days:
            raise ValueError(f"Date range too large: {days_in_range} days (max: {self.max_date_range_days})")
        
        # Generate file paths
        chunk_name = self.generate_chunk_filename(start_date, end_date)
        tmp_file = self.output_dir / f"{chunk_name}.csv.tmp"
        final_file = self.output_dir / f"{chunk_name}.csv"
        
        logger.info(f"Processing chunk: {chunk_name} ({days_in_range} days)")
        
        # Step 1: Check if already complete
        if final_file.exists():
            logger.info(f"✅ Chunk {chunk_name} already complete, skipping")
            return False
        
        # Step 2: Clean up any existing .tmp file (from previous failed run)
        if tmp_file.exists():
            logger.info(f"Removing existing tmp file: {tmp_file}")
            tmp_file.unlink()
        
        try:
            # Step 3: Get first page to determine pagination requirements
            logger.info(f"Getting first page to determine total pages...")
            first_page_data = await self.get_api_page(start_date, end_date, start_index=0, event_name=event_name)
            
            total_results = first_page_data.get('totalResults', 0)
            results_per_page = first_page_data.get('resultsPerPage', self.max_results_per_page)
            total_pages = math.ceil(total_results / results_per_page) if total_results > 0 else 1
            
            logger.info(f"📊 Chunk {chunk_name}: {total_results} total results across {total_pages} pages")
            
            # Step 4: Collect all changes from all pages
            all_changes = []
            
            # Add changes from first page
            first_page_changes = self.extract_changes_from_response(first_page_data)
            all_changes.extend(first_page_changes)
            logger.info(f"Page 1/{total_pages}: {len(first_page_changes)} changes")
            
            # Get remaining pages if needed
            for page_num in range(2, total_pages + 1):
                start_index = (page_num - 1) * results_per_page
                
                logger.info(f"Fetching page {page_num}/{total_pages} (start_index={start_index})...")
                page_data = await self.get_api_page(start_date, end_date, start_index, event_name)
                
                page_changes = self.extract_changes_from_response(page_data)
                all_changes.extend(page_changes)
                
                logger.info(f"Page {page_num}/{total_pages}: {len(page_changes)} changes")
                
                # Rate limiting between pages
                await asyncio.sleep(0.1)
            
            # Step 5: Validate data completeness
            actual_count = len(all_changes)
            if actual_count != total_results:
                raise ValueError(f"Data validation failed: expected {total_results} changes, got {actual_count}")
            
            logger.info(f"✅ Data validation passed: {actual_count} changes collected")
            
            # Step 6: Write complete dataset to .tmp file
            logger.info(f"Writing complete dataset to temporary file: {tmp_file}")
            self.write_changes_to_csv(all_changes, tmp_file)
            
            # Step 7: Atomic commit (rename .tmp to final)
            logger.info(f"Committing chunk: {tmp_file} → {final_file}")
            tmp_file.rename(final_file)
            
            logger.info(f"🎉 Chunk {chunk_name} completed successfully: {actual_count} changes")
            return True
            
        except Exception as e:
            # Clean up .tmp file on error
            if tmp_file.exists():
                logger.error(f"Cleaning up tmp file due to error: {tmp_file}")
                tmp_file.unlink()
            
            logger.error(f"❌ Failed to process chunk {chunk_name}: {e}")
            raise
    
    def get_completed_chunks(self) -> List[str]:
        """
        Get list of already completed chunks (files without .tmp extension).
        
        Returns:
            List of completed chunk names
        """
        completed = []
        for file_path in self.output_dir.glob("*.csv"):
            if not file_path.name.endswith('.tmp'):
                chunk_name = file_path.stem  # filename without extension
                completed.append(chunk_name)
        
        return sorted(completed)
    
    def get_pending_tmp_files(self) -> List[Path]:
        """
        Get list of pending .tmp files (incomplete chunks).
        
        Returns:
            List of .tmp file paths
        """
        return list(self.output_dir.glob("*.csv.tmp"))


# Testing and example usage
async def test_single_chunk():
    """
    Test processing a single chunk.
    
    This will process approximately 120 days of CVE change data.
    """
    async with CVEChunkProcessor() as processor:
        
        # Test with a 120-day chunk from early 2022
        start_date = datetime(2022, 2, 1)
        end_date = datetime(2022, 5, 31)  # About 120 days
        
        logger.info("Starting single chunk test...")
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Check current state
        completed = processor.get_completed_chunks()
        pending = processor.get_pending_tmp_files()
        
        logger.info(f"Completed chunks: {len(completed)}")
        logger.info(f"Pending .tmp files: {len(pending)}")
        
        # Process the chunk
        was_processed = await processor.process_single_chunk(start_date, end_date)
        
        if was_processed:
            logger.info("✅ Chunk processing completed successfully!")
        else:
            logger.info("ℹ️ Chunk was already complete")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_single_chunk()) 