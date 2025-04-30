import requests
import csv
import os
import json
from datetime import datetime
import time
import logging
from random import uniform

# Configuration
API_URL = "https://euvdservices.enisa.europa.eu/api/vulnerabilities"
OUTPUT_DIR = "euvd_cve_data"
CSV_FILE = "euvd_cve_full_dataset.csv"
CHECKPOINT_FILE = "last_page.txt"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
SIZE = 100  # Max results per page
REQUEST_DELAY = (1, 3)  # Random delay range (seconds)
LOG_FILE = "full_dataset_scraper.log"

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def read_last_page():
    """Read the last processed page from the checkpoint file."""
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0  # Start from page 0 if file doesn't exist or is invalid

def save_last_page(page):
    """Save the current page to the checkpoint file."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(page))

def fetch_cves(page=0):
    """Fetch CVEs from EUVD API with pagination, no filtering."""
    params = {
        "page": page,
        "size": SIZE
    }
    headers = {
        "accept": "application/json",
        "User-Agent": USER_AGENT
    }
    
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(API_URL, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            if page == 0:
                logging.debug(f"Raw API response (page 0): {json.dumps(data, indent=2)}")
            return data
        except requests.HTTPError as e:
            if e.response.status_code == 429:
                logging.warning(f"Rate limit hit on attempt {attempt + 1}, retrying...")
                time.sleep(2 ** attempt)
            elif e.response.status_code in (502, 503, 504):
                logging.warning(f"Server error {e.response.status_code} on attempt {attempt + 1}, retrying after delay...")
                time.sleep(10 + (2 ** attempt))  # Longer delay for server errors
            else:
                logging.error(f"Error fetching data (page {page}): {e}")
                return None
        except requests.RequestException as e:
            logging.error(f"Error fetching data (page {page}): {e}")
            return None
    logging.error(f"Failed to fetch page {page} after {retries} attempts")
    return None

def parse_cve_data(cve):
    """Extract metadata from a CVE item."""
    euvd_id = cve.get("id", "unknown")
    try:
        logging.debug(f"Processing CVE {euvd_id}: {json.dumps(cve, indent=2)}")

        # Extract cve_id, use euvd_id as fallback
        aliases = cve.get("aliases", "")
        cve_id = ""
        if aliases:
            alias_list = [alias for alias in aliases.split("\n") if alias and alias.startswith("CVE-")]
            cve_id = alias_list[0] if alias_list else ""

        # Parse severity
        severity = ""
        try:
            base_score_vector = cve.get("baseScoreVector", "")
            if base_score_vector:
                vector_parts = base_score_vector.split("/")
                for part in vector_parts:
                    if part.startswith("S:"):
                        severity = part.split(":")[1]
                        break
        except Exception as e:
            logging.warning(f"Error parsing baseScoreVector for CVE {euvd_id}: {e}")

        # Parse products
        products = []
        try:
            products = [
                f"{p.get('product', {}).get('name', '')} {p.get('product_version', '')}".strip()
                for p in cve.get("enisaIdProduct", [])
                if p.get("product", {}).get("name")
            ]
        except Exception as e:
            logging.warning(f"Error parsing enisaIdProduct for CVE {euvd_id}: {e}")

        # Parse vendors
        vendors = []
        try:
            vendors = [
                v.get("vendor", {}).get("name", "")
                for v in cve.get("enisaIdVendor", [])
                if v.get("vendor", {}).get("name")
            ]
        except Exception as e:
            logging.warning(f"Error parsing enisaIdVendor for CVE {euvd_id}: {e}")

        cve_data = {
            "euvd_id": euvd_id,
            "cve_id": cve_id,
            "description": cve.get("description", ""),
            "base_score": str(cve.get("baseScore", "")),
            "severity": severity,
            "affected_products": ";".join(products),
            "vendors": ";".join(vendors),
            "references": cve.get("references", "").replace("\n", ";"),
            "published_date": cve.get("datePublished", ""),
            "modified_date": cve.get("dateUpdated", ""),
            "epss": str(cve.get("epss", "")),
            "assigner": cve.get("assigner", "")
        }
        return cve_data
    except Exception as e:
        logging.error(f"Error parsing CVE {euvd_id}: {e}")
        return None

def save_to_csv(cve_data, output_file, is_first_write=False):
    """Append CVE data to a CSV file."""
    fieldnames = [
        "euvd_id", "cve_id", "description", "base_score", "severity",
        "affected_products", "vendors", "references", "published_date",
        "modified_date", "epss", "assigner"
    ]
    
    mode = "w" if is_first_write else "a"
    with open(output_file, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_first_write:
            writer.writeheader()
        for cve in cve_data:
            writer.writerow(cve)
    
    logging.info(f"Appended {len(cve_data)} CVEs to {output_file}")

def main():
    logging.info("Starting full EUVD CVE dataset scrape...")
    
    output_file = os.path.join(OUTPUT_DIR, CSV_FILE)
    start_page = read_last_page()
    logging.info(f"Starting from page {start_page + 1}")
    
    page = start_page
    is_first_write = not os.path.exists(output_file)
    
    while True:
        data = fetch_cves(page)
        if not data or not data.get("items"):
            logging.info(f"No more CVEs at page {page + 1}")
            break
        
        cve_list = []
        for cve in data.get("items", []):
            parsed_cve = parse_cve_data(cve)
            if parsed_cve:
                cve_list.append(parsed_cve)
        
        if cve_list:
            save_to_csv(cve_list, output_file, is_first_write)
            is_first_write = False
        
        # Update checkpoint after processing page
        save_last_page(page)
        logging.info(f"Completed page {page + 1}, checkpoint saved")
        
        total = data.get("total", 0)
        logging.info(f"Page {page + 1}, Total CVEs: {total}")
        
        page += 1
        time.sleep(uniform(*REQUEST_DELAY))
    
    logging.info("Full dataset scrape completed")

if __name__ == "__main__":
    main()