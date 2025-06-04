#!/usr/bin/env python3
"""
Download all vulnerability feeds from https://vulnerability.circl.lu/dumps/
Stores data on external SSD drive at W:\raw_circl
"""

import os
import requests
import shutil
from pathlib import Path
from tqdm import tqdm
import time
from urllib.parse import urljoin
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log'),
        logging.StreamHandler()
    ]
)

# Base URL and target directory
BASE_URL = "https://vulnerability.circl.lu/dumps/"
TARGET_DIR = Path("W:/raw_circl")

# All available feeds from CIRCL with their expected file extensions
FEEDS = {
    "capec": ".ndjson",
    "circl": ".ndjson", 
    "csaf_certbund": ".ndjson",
    "csaf_cisa": ".ndjson",
    "csaf_cisco": ".ndjson",
    "csaf_microsoft": ".ndjson",
    "csaf_ncscnl": ".ndjson",
    "csaf_nozominetworks": ".ndjson",
    "csaf_opensuse": ".ndjson",
    "csaf_ox": ".ndjson",
    "csaf_redhat": ".ndjson",
    "csaf_sick": ".ndjson",
    "csaf_siemens": ".ndjson",
    "csaf_suse": ".ndjson",
    "cvelistv5": ".ndjson",
    "cwec": ".ndjson",
    "emb3d": ".ndjson",
    "fkie_nvd": ".ndjson",
    "github": ".ndjson",
    "gna-1": ".ndjson",
    "gsd": ".ndjson",
    "jvndb": ".ndjson",
    "nvd": ".ndjson",
    "ossf_malicious_packages": ".ndjson",
    "pysec": ".ndjson",
    "tailscale": ".ndjson",
    "variot": ".ndjson",
    "vulnrichment": ".ndjson"
}

def create_target_directory():
    """Create the target directory if it doesn't exist"""
    try:
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Target directory created/verified: {TARGET_DIR}")
        return True
    except Exception as e:
        logging.error(f"Failed to create target directory {TARGET_DIR}: {e}")
        return False

def get_file_size(url):
    """Get the file size from URL headers"""
    try:
        response = requests.head(url, timeout=30)
        if response.status_code == 200:
            content_length = response.headers.get('content-length')
            if content_length:
                return int(content_length)
    except Exception as e:
        logging.warning(f"Could not get file size for {url}: {e}")
    return None

def check_url_exists(url):
    """Check if URL exists and is accessible"""
    try:
        response = requests.head(url, timeout=30)
        return response.status_code == 200
    except Exception as e:
        logging.warning(f"Could not check URL {url}: {e}")
        return False

def download_file(url, local_path, retries=3):
    """Download a file with progress bar and retry logic"""
    for attempt in range(retries):
        try:
            # Get file size for progress bar
            file_size = get_file_size(url)
            
            # Check if file already exists and has the same size
            if local_path.exists() and file_size:
                existing_size = local_path.stat().st_size
                if existing_size == file_size:
                    logging.info(f"File already exists with correct size: {local_path.name}")
                    return True
                else:
                    logging.info(f"File exists but size differs (existing: {existing_size}, expected: {file_size})")
            
            logging.info(f"Downloading {url} to {local_path}")
            
            # Download with streaming to handle large files
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Use temporary file during download
            temp_path = local_path.with_suffix(local_path.suffix + '.tmp')
            
            total_size = file_size or int(response.headers.get('content-length', 0))
            
            with open(temp_path, 'wb') as f:
                if total_size > 0:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_path.name) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No content-length header, download without progress bar
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                    logging.info(f"Downloaded {local_path.name} (size unknown)")
            
            # Move temporary file to final location
            shutil.move(temp_path, local_path)
            logging.info(f"Successfully downloaded: {local_path.name}")
            return True
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Download attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"Failed to download {url} after {retries} attempts")
                
        except Exception as e:
            logging.error(f"Unexpected error downloading {url}: {e}")
            break
    
    return False

def try_different_extensions(base_url, feed, local_path_base):
    """Try downloading with different file extensions"""
    # Common extensions to try
    extensions = ['.ndjson', '.json', '.json.gz', '.ndjson.gz', '.gz', '']
    
    for ext in extensions:
        url = base_url + feed + ext
        local_path = local_path_base.with_suffix(ext if ext else '')
        
        logging.info(f"Trying URL: {url}")
        
        if check_url_exists(url):
            logging.info(f"Found accessible URL: {url}")
            if download_file(url, local_path):
                return True
        else:
            logging.debug(f"URL not accessible: {url}")
    
    return False

def download_all_feeds():
    """Download all vulnerability feeds"""
    if not create_target_directory():
        return False
    
    success_count = 0
    total_feeds = len(FEEDS)
    
    logging.info(f"Starting download of {total_feeds} feeds to {TARGET_DIR}")
    
    for i, (feed, expected_ext) in enumerate(FEEDS.items(), 1):
        logging.info(f"Processing feed {i}/{total_feeds}: {feed}")
        
        # Construct base URL and local path
        base_url = BASE_URL
        local_path_base = TARGET_DIR / feed
        
        # Try with expected extension first, then others
        url_with_ext = base_url + feed + expected_ext
        local_path = local_path_base.with_suffix(expected_ext)
        
        logging.info(f"Trying primary URL: {url_with_ext}")
        
        if check_url_exists(url_with_ext):
            if download_file(url_with_ext, local_path):
                success_count += 1
            else:
                logging.warning(f"Failed to download {feed} with expected extension {expected_ext}")
        else:
            logging.info(f"Primary URL not found, trying alternative extensions for {feed}")
            if try_different_extensions(base_url, feed, local_path_base):
                success_count += 1
            else:
                logging.error(f"Could not find any working URL for feed: {feed}")
        
        # Small delay between downloads to be respectful
        time.sleep(1)
    
    logging.info(f"Download completed: {success_count}/{total_feeds} feeds downloaded successfully")
    
    if success_count < total_feeds:
        logging.warning(f"Failed to download {total_feeds - success_count} feeds")
        return False
    
    return True

def main():
    """Main function"""
    start_time = time.time()
    
    logging.info("Starting CIRCL vulnerability feeds download")
    logging.info(f"Target directory: {TARGET_DIR}")
    logging.info(f"Number of feeds to download: {len(FEEDS)}")
    
    # Check if target drive is accessible
    if not TARGET_DIR.parent.exists():
        logging.error(f"Target drive {TARGET_DIR.parent} is not accessible. Please ensure your external SSD is connected.")
        return
    
    success = download_all_feeds()
    
    end_time = time.time()
    duration = end_time - start_time
    
    logging.info(f"Script completed in {duration:.2f} seconds")
    
    if success:
        logging.info("All feeds downloaded successfully!")
    else:
        logging.error("Some downloads failed. Check the log for details.")

if __name__ == "__main__":
    main()

