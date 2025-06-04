#!/usr/bin/env python3
"""
Simple SANS DShield Feeds Downloader

Downloads cybersecurity threat feeds from SANS DShield to W:\sans directory.
Minimal, straightforward implementation.

Author: Expert Full Stack Developer  
Date: 2025-01-31
"""

import requests
import os
from datetime import datetime, timedelta
from pathlib import Path

def download_file(url, filepath):
    """Download a file from URL to filepath"""
    try:
        print(f"Downloading: {url}")
        
        headers = {
            'User-Agent': 'SANSDownloader/1.0 (research@example.com)'
        }
        
        response = requests.get(url, headers=headers, timeout=60)
        
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(filepath)
            print(f"✓ Downloaded: {filepath.name} ({file_size:,} bytes)")
            return True
        else:
            print(f"✗ Failed: HTTP {response.status_code} - {url}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading {url}: {e}")
        return False

def main():
    """Download all SANS DShield feeds"""
    
    # Base directory
    base_dir = Path(r"W:\sans")
    
    # Create directories
    ssh_dir = base_dir / "ssh_logs"
    web_dir = base_dir / "web_honeypot" 
    threat_dir = base_dir / "threat_intel"
    
    for directory in [ssh_dir, web_dir, threat_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("SANS DShield Feeds Downloader")
    print("=" * 40)
    
    # Get yesterday's date for SSH logs
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Define feeds to download
    feeds = [
        {
            "url": f"https://feeds.dshield.org/feeds/ssh_daily_{yesterday}",
            "filepath": ssh_dir / f"ssh_daily_{yesterday}.txt",
            "name": "SSH Daily Logs"
        },
        {
            "url": "https://isc.sans.edu/feeds/urlsummary.txt",
            "filepath": web_dir / "urlsummary.txt", 
            "name": "URL Summary"
        },
        {
            "url": "https://isc.sans.edu/feeds/urlcategories.txt",
            "filepath": web_dir / "urlcategories.txt",
            "name": "URL Categories"
        },
        {
            "url": "https://isc.sans.edu/feeds/threatintel.txt", 
            "filepath": threat_dir / "threatintel.txt",
            "name": "Threat Intelligence"
        }
    ]
    
    # Download each feed
    successful = 0
    total = len(feeds)
    
    for feed in feeds:
        print(f"\n{feed['name']}:")
        if download_file(feed["url"], feed["filepath"]):
            successful += 1
    
    # Summary
    print("\n" + "=" * 40)
    print(f"Download Complete: {successful}/{total} feeds successful")
    print(f"Files saved to: {base_dir}")
    
    if successful > 0:
        print("\nDownloaded files:")
        for feed in feeds:
            if feed["filepath"].exists():
                size_mb = feed["filepath"].stat().st_size / (1024 * 1024)
                print(f"  {feed['filepath']} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main() 