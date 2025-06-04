#!/usr/bin/env python3
"""
SANS DShield Feeds Downloader

This script systematically downloads cybersecurity threat feeds from SANS DShield,
including SSH honeypot logs, web honeypot data, and threat intelligence feeds.

Features:
- Rate limiting compliance (max once per hour)
- Robust error handling and retry logic
- Organized file storage with metadata
- Preparation for Python pandas analysis
- Comprehensive logging and progress tracking

Author: Expert Full Stack Developer
Date: 2025-01-31
Target Directory: W:\sans
"""

import requests
import json
import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib
import argparse
from urllib.parse import urlparse
import shutil

class SANSFeedsDownloader:
    """
    Downloads and manages SANS DShield threat intelligence feeds
    """
    
    def __init__(self, base_dir: str = r"W:\sans"):
        """
        Initialize the downloader
        
        Args:
            base_dir: Base directory for storing downloaded files
        """
        self.base_dir = Path(base_dir)
        self.metadata_file = self.base_dir / "download_metadata.json"
        self.log_file = self.base_dir / "download.log"
        
        # Rate limiting: 1 hour minimum between downloads
        self.rate_limit_seconds = 3600
        
        # Feed configurations
        self.feeds_config = {
            "ssh_daily": {
                "url_template": "https://feeds.dshield.org/feeds/ssh_daily_{date}",
                "description": "Daily SSH honeypot logs (IPs, usernames, passwords)",
                "directory": "ssh_logs",
                "date_format": "%Y-%m-%d",
                "days_back": 1,  # Get previous day's data
                "file_extension": "txt",
                "expected_format": "tab_separated"
            },
            "url_summary": {
                "url": "https://isc.sans.edu/feeds/urlsummary.txt",
                "description": "URL summary with first/last seen dates and frequencies",
                "directory": "web_honeypot",
                "file_extension": "txt",
                "expected_format": "tab_separated"
            },
            "url_categories": {
                "url": "https://isc.sans.edu/feeds/urlcategories.txt", 
                "description": "URL categories with vulnerability classifications",
                "directory": "web_honeypot",
                "file_extension": "txt",
                "expected_format": "tab_separated"
            },
            "threat_intel": {
                "url": "https://isc.sans.edu/feeds/threatintel.txt",
                "description": "Threat intelligence database dump with IP labels",
                "directory": "threat_intel",
                "file_extension": "txt",
                "expected_format": "tab_separated"
            }
        }
        
        self.setup_logging()
        self.setup_directories()
        self.load_metadata()
    
    def setup_logging(self):
        """Configure comprehensive logging"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*60)
        self.logger.info("SANS DShield Feeds Downloader Started")
        self.logger.info(f"Base directory: {self.base_dir}")
        self.logger.info("="*60)
    
    def setup_directories(self):
        """Create organized directory structure"""
        directories = set()
        for feed_config in self.feeds_config.values():
            directories.add(feed_config["directory"])
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {dir_path}")
        
        # Create analysis directory for processed files
        (self.base_dir / "analysis").mkdir(exist_ok=True)
        (self.base_dir / "archive").mkdir(exist_ok=True)
    
    def load_metadata(self):
        """Load download metadata for rate limiting and tracking"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                self.logger.info(f"Loaded metadata for {len(self.metadata)} feeds")
            else:
                self.metadata = {}
                self.logger.info("No existing metadata found, starting fresh")
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            self.metadata = {}
    
    def save_metadata(self):
        """Save download metadata"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            self.logger.debug("Metadata saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def can_download(self, feed_name: str) -> Tuple[bool, str]:
        """
        Check if we can download a feed based on rate limiting
        
        Args:
            feed_name: Name of the feed to check
            
        Returns:
            Tuple of (can_download, reason)
        """
        if feed_name not in self.metadata:
            return True, "First download"
        
        last_download = self.metadata[feed_name].get("last_download_time")
        if not last_download:
            return True, "No previous download time"
        
        last_download_dt = datetime.fromisoformat(last_download)
        time_since_last = datetime.now() - last_download_dt
        
        if time_since_last.total_seconds() >= self.rate_limit_seconds:
            return True, f"Rate limit satisfied ({time_since_last})"
        else:
            wait_time = self.rate_limit_seconds - time_since_last.total_seconds()
            return False, f"Rate limited. Wait {wait_time:.0f} seconds"
    
    def generate_filename(self, feed_name: str, config: Dict, download_date: str = None) -> str:
        """
        Generate appropriate filename for the feed
        
        Args:
            feed_name: Name of the feed
            config: Feed configuration
            download_date: Date for date-based feeds
            
        Returns:
            Generated filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if feed_name == "ssh_daily" and download_date:
            # For SSH daily logs, include the data date and download timestamp
            return f"ssh_daily_{download_date}_{timestamp}.{config['file_extension']}"
        else:
            # For other feeds, include download timestamp to track versions
            return f"{feed_name}_{timestamp}.{config['file_extension']}"
    
    def download_file(self, url: str, filepath: Path, feed_name: str) -> Dict:
        """
        Download a file with comprehensive error handling and metadata collection
        
        Args:
            url: URL to download
            filepath: Local file path to save
            feed_name: Name of the feed for logging
            
        Returns:
            Dictionary with download results and metadata
        """
        result = {
            "success": False,
            "url": url,
            "filepath": str(filepath),
            "download_time": datetime.now().isoformat(),
            "file_size": 0,
            "file_hash": None,
            "http_status": None,
            "error": None,
            "headers": {}
        }
        
        try:
            self.logger.info(f"Downloading {feed_name}: {url}")
            
            headers = {
                'User-Agent': 'SANSFeedsDownloader/1.0 (research@cybersec.edu)',
                'Accept': 'text/plain, text/csv, application/octet-stream',
                'Accept-Encoding': 'gzip, deflate'
            }
            
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            result["http_status"] = response.status_code
            result["headers"] = dict(response.headers)
            
            if response.status_code == 200:
                # Download the file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Calculate file metadata
                result["file_size"] = filepath.stat().st_size
                
                # Calculate file hash for integrity verification
                with open(filepath, 'rb') as f:
                    file_hash = hashlib.sha256()
                    for chunk in iter(lambda: f.read(4096), b""):
                        file_hash.update(chunk)
                    result["file_hash"] = file_hash.hexdigest()
                
                result["success"] = True
                self.logger.info(f"Downloaded {feed_name}: {result['file_size']} bytes")
                
                # Analyze file content for Python readiness
                self.analyze_file_format(filepath, feed_name)
                
            elif response.status_code == 404:
                result["error"] = "File not found (404) - may not be available yet"
                self.logger.warning(f"{feed_name}: File not found (404): {url}")
            else:
                result["error"] = f"HTTP {response.status_code}: {response.reason}"
                self.logger.error(f"{feed_name}: HTTP {response.status_code}: {url}")
                
        except requests.exceptions.Timeout:
            result["error"] = "Download timeout"
            self.logger.error(f"{feed_name}: Download timeout: {url}")
        except requests.exceptions.ConnectionError:
            result["error"] = "Connection error"
            self.logger.error(f"{feed_name}: Connection error: {url}")
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"{feed_name}: Unexpected error: {e}")
            
        return result
    
    def analyze_file_format(self, filepath: Path, feed_name: str):
        """
        Analyze downloaded file format for Python pandas compatibility
        
        Args:
            filepath: Path to the downloaded file
            feed_name: Name of the feed for logging
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first few lines to analyze format
                lines = [f.readline().strip() for _ in range(10) if f.readline()]
                
            if not lines:
                self.logger.warning(f"{feed_name}: Empty file")
                return
                
            # Detect delimiter
            first_line = lines[0]
            tab_count = first_line.count('\t')
            comma_count = first_line.count(',')
            pipe_count = first_line.count('|')
            space_count = first_line.count(' ')
            
            delimiter = '\t'  # Default to tab
            if comma_count > tab_count and comma_count > pipe_count:
                delimiter = ','
            elif pipe_count > tab_count and pipe_count > comma_count:
                delimiter = '|'
            elif space_count > tab_count and space_count > comma_count and space_count > pipe_count:
                delimiter = ' '
            
            # Check for headers
            has_headers = False
            if lines:
                # Common header indicators
                header_indicators = ['ip', 'url', 'date', 'time', 'source', 'target', 'port', 'username', 'password']
                first_line_lower = lines[0].lower()
                if any(indicator in first_line_lower for indicator in header_indicators):
                    has_headers = True
            
            analysis = {
                "delimiter": delimiter,
                "has_headers": has_headers,
                "line_count": len(lines),
                "sample_lines": lines[:3],
                "encoding": "utf-8"
            }
            
            # Save analysis for Python processing
            analysis_file = filepath.parent / f"{filepath.stem}_format_analysis.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"{feed_name}: Format analysis saved - delimiter: '{delimiter}', headers: {has_headers}")
            
        except Exception as e:
            self.logger.error(f"{feed_name}: Error analyzing file format: {e}")
    
    def download_ssh_daily(self, target_date: str = None) -> Dict:
        """
        Download SSH daily logs for a specific date
        
        Args:
            target_date: Date in YYYY-MM-DD format (defaults to yesterday)
            
        Returns:
            Download result dictionary
        """
        config = self.feeds_config["ssh_daily"]
        
        if not target_date:
            # Default to yesterday's data
            yesterday = datetime.now() - timedelta(days=config["days_back"])
            target_date = yesterday.strftime(config["date_format"])
        
        # Check rate limiting
        feed_key = f"ssh_daily_{target_date}"
        can_download, reason = self.can_download(feed_key)
        if not can_download:
            self.logger.warning(f"SSH Daily ({target_date}): {reason}")
            return {"success": False, "error": reason, "feed": feed_key}
        
        # Generate URL and filename
        url = config["url_template"].format(date=target_date)
        filename = self.generate_filename("ssh_daily", config, target_date)
        filepath = self.base_dir / config["directory"] / filename
        
        # Download file
        result = self.download_file(url, filepath, f"SSH Daily ({target_date})")
        
        # Update metadata
        if result["success"]:
            self.metadata[feed_key] = {
                "last_download_time": result["download_time"],
                "last_successful_download": result,
                "target_date": target_date,
                "feed_type": "ssh_daily"
            }
            self.save_metadata()
            
            # Create copy as latest file (Windows-compatible alternative to symlink)
            latest_file = self.base_dir / config["directory"] / "ssh_daily_latest.txt"
            try:
                shutil.copy2(filepath, latest_file)
                self.logger.info(f"Created latest file copy: {latest_file}")
            except Exception as e:
                self.logger.warning(f"Could not create latest file copy: {e}")
        
        return result
    
    def download_feed(self, feed_name: str) -> Dict:
        """
        Download a standard feed (non-date-based)
        
        Args:
            feed_name: Name of the feed to download
            
        Returns:
            Download result dictionary
        """
        if feed_name not in self.feeds_config:
            return {"success": False, "error": f"Unknown feed: {feed_name}"}
        
        config = self.feeds_config[feed_name]
        
        # Check rate limiting
        can_download, reason = self.can_download(feed_name)
        if not can_download:
            self.logger.warning(f"{feed_name}: {reason}")
            return {"success": False, "error": reason, "feed": feed_name}
        
        # Generate filename and path
        filename = self.generate_filename(feed_name, config)
        filepath = self.base_dir / config["directory"] / filename
        
        # Download file
        result = self.download_file(config["url"], filepath, feed_name)
        
        # Update metadata
        if result["success"]:
            self.metadata[feed_name] = {
                "last_download_time": result["download_time"],
                "last_successful_download": result,
                "feed_type": feed_name
            }
            self.save_metadata()
            
            # Create copy as latest file (Windows-compatible alternative to symlink)
            latest_file = self.base_dir / config["directory"] / f"{feed_name}_latest.txt"
            try:
                shutil.copy2(filepath, latest_file)
                self.logger.info(f"Created latest file copy: {latest_file}")
            except Exception as e:
                self.logger.warning(f"Could not create latest file copy: {e}")
        
        return result
    
    def download_all_feeds(self, include_ssh_daily: bool = True, ssh_date: str = None) -> Dict:
        """
        Download all configured feeds
        
        Args:
            include_ssh_daily: Whether to include SSH daily logs
            ssh_date: Specific date for SSH logs (YYYY-MM-DD)
            
        Returns:
            Dictionary with results for all feeds
        """
        results = {}
        
        self.logger.info("Starting download of all SANS DShield feeds")
        
        # Download standard feeds
        standard_feeds = ["url_summary", "url_categories", "threat_intel"]
        for feed_name in standard_feeds:
            self.logger.info(f"Processing feed: {feed_name}")
            results[feed_name] = self.download_feed(feed_name)
            
            # Add delay between downloads to be respectful
            time.sleep(2)
        
        # Download SSH daily logs if requested
        if include_ssh_daily:
            self.logger.info("Processing SSH daily logs")
            results["ssh_daily"] = self.download_ssh_daily(ssh_date)
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results: Dict):
        """
        Generate a comprehensive summary report of the download session
        
        Args:
            results: Dictionary of download results
        """
        report_file = self.base_dir / f"download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("SANS DShield Feeds Download Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Download Time: {datetime.now().isoformat()}\n")
            f.write(f"Base Directory: {self.base_dir}\n\n")
            
            successful = 0
            failed = 0
            total_size = 0
            
            for feed_name, result in results.items():
                f.write(f"Feed: {feed_name}\n")
                f.write("-" * 30 + "\n")
                
                if result.get("success"):
                    successful += 1
                    file_size = result.get("file_size", 0)
                    total_size += file_size
                    f.write(f"Status: SUCCESS\n")
                    f.write(f"File: {result.get('filepath', 'Unknown')}\n")
                    f.write(f"Size: {file_size:,} bytes\n")
                    f.write(f"Hash: {result.get('file_hash', 'N/A')}\n")
                else:
                    failed += 1
                    f.write(f"Status: FAILED\n")
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                
                f.write(f"URL: {result.get('url', 'N/A')}\n\n")
            
            f.write("Summary\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Feeds: {len(results)}\n")
            f.write(f"Successful: {successful}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"Total Downloaded: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)\n")
        
        self.logger.info(f"Summary report saved: {report_file}")
        self.logger.info(f"Download session complete: {successful}/{len(results)} successful")
    
    def list_available_files(self):
        """List all downloaded files with metadata"""
        print("\nAvailable SANS DShield Feed Files:")
        print("=" * 60)
        
        for feed_name, config in self.feeds_config.items():
            feed_dir = self.base_dir / config["directory"]
            if feed_dir.exists():
                print(f"\n{feed_name.upper()} ({config['description']}):")
                print("-" * 40)
                
                files = list(feed_dir.glob(f"{feed_name}_*.{config['file_extension']}"))
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                if files:
                    for file_path in files[:5]:  # Show latest 5 files
                        stat = file_path.stat()
                        size_mb = stat.st_size / 1024 / 1024
                        mod_time = datetime.fromtimestamp(stat.st_mtime)
                        print(f"  {file_path.name} ({size_mb:.2f} MB, {mod_time.strftime('%Y-%m-%d %H:%M')})")
                    
                    if len(files) > 5:
                        print(f"  ... and {len(files) - 5} more files")
                else:
                    print("  No files found")

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Download SANS DShield threat intelligence feeds")
    parser.add_argument("--base-dir", default=r"W:\sans", help="Base directory for downloads")
    parser.add_argument("--feed", choices=["url_summary", "url_categories", "threat_intel", "ssh_daily", "all"], 
                       default="all", help="Specific feed to download")
    parser.add_argument("--ssh-date", help="Specific date for SSH logs (YYYY-MM-DD)")
    parser.add_argument("--list", action="store_true", help="List available downloaded files")
    parser.add_argument("--force", action="store_true", help="Force download ignoring rate limits")
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = SANSFeedsDownloader(base_dir=args.base_dir)
    
    if args.list:
        downloader.list_available_files()
        return
    
    # Override rate limiting if forced
    if args.force:
        downloader.rate_limit_seconds = 0
        downloader.logger.warning("Rate limiting disabled - use responsibly!")
    
    # Download feeds based on arguments
    if args.feed == "all":
        results = downloader.download_all_feeds(ssh_date=args.ssh_date)
    elif args.feed == "ssh_daily":
        results = {"ssh_daily": downloader.download_ssh_daily(args.ssh_date)}
    else:
        results = {args.feed: downloader.download_feed(args.feed)}
    
    # Print results summary
    successful = sum(1 for r in results.values() if r.get("success"))
    total = len(results)
    print(f"\nDownload Complete: {successful}/{total} feeds successful")
    
    if successful > 0:
        print(f"\nFiles saved to: {downloader.base_dir}")
        print("\nFor Python analysis, check the format analysis JSON files in each directory.")
        print("Example usage:")
        print("  import pandas as pd")
        print("  df = pd.read_csv('path/to/file.txt', delimiter='\\t')  # Use detected delimiter")

if __name__ == "__main__":
    main() 