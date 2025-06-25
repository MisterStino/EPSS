#!/usr/bin/env python3
"""
Download large ML files from external storage.
Run this after cloning the repository to get large data files.
"""

import requests
import os
from pathlib import Path

# URLs for your large files (upload to cloud storage)
LARGE_FILES = {
    "ml_pipeline/data_prep/work/epss_stage1.arrow": "https://your-cloud-storage.com/epss_stage1.arrow",
    "ml_pipeline/results/predictions/predictions_stream.nc": "https://your-cloud-storage.com/predictions_stream.nc"
}

def download_file(url: str, local_path: str):
    """Download a file from URL to local path."""
    print(f"Downloading {url} -> {local_path}")
    
    # Create directory if it doesn't exist
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Download file
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print(f"✅ Downloaded: {local_path}")

def main():
    """Download all large files."""
    print("🔄 Downloading large ML files...")
    
    for local_path, url in LARGE_FILES.items():
        if not os.path.exists(local_path):
            download_file(url, local_path)
        else:
            print(f"⏭️  Skipping {local_path} (already exists)")
    
    print("✅ All large files downloaded!")

if __name__ == "__main__":
    main() 