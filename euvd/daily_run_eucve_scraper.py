import requests
import csv
import os
import json
from datetime import datetime, timedelta
import time
import logging
from random import uniform

# Configuration
API_URL = "https://euvdservices.enisa.europa.eu/api/vulnerabilities"
OUTPUT_DIR = "euvd_cve_data"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
SIZE = 100  # Max results per page
REQUEST_DELAY = (1, 3)  # Random delay range (seconds)
LOG_FILE = "daily_scraper.log"

# Setup logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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
            else:
                logging.error(f"Error fetching data (page {page}): {e}")
                return None
        except requests.RequestException as e:
            logging.error(f"Error fetching data (page {page}): {e}")
            return None
    logging.error(f"Failed to fetch page {page} after {retries} attempts")
    return None

def parse_date_updated(date_str):
    """Parse dateUpdated to YYYY-MM-DD or return None if parsing fails."""
    try:
        if not date_str:
            return None
        for fmt in (
            "%b %d, %Y, %I:%M:%S %p",  # e.g., Apr 28, 2025, 4:31:21 PM
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        logging.warning(f"Failed to parse dateUpdated: {date_str}")
        return None
    except Exception as e:
        logging.error(f"Error parsing dateUpdated {date_str}: {e}")
        return None

def parse_cve_data(cve, target_date, day_before_target):
    """Extract metadata from a CVE item, filter by dateUpdated."""
    euvd_id = cve.get("id", "unknown")
    try:
        logging.debug(f"Processing CVE {euvd_id}: {json.dumps(cve, indent=2)}")

        # Parse dateUpdated
        date_updated = cve.get("dateUpdated", "")
        date_updated_ymd = parse_date_updated(date_updated)
        if not date_updated_ymd:
            logging.warning(f"Skipping CVE {euvd_id}: Invalid dateUpdated format: {date_updated}")
            return None, False

        # Check dateUpdated
        if date_updated_ymd != target_date:
            if date_updated_ymd < day_before_target:
                logging.info(f"Found CVE {euvd_id} with dateUpdated {date_updated_ymd}, earlier than target {target_date}")
                return None, True  # Signal to stop after processing page
            logging.info(f"Skipping CVE {euvd_id}: dateUpdated {date_updated_ymd} does not match target {target_date}")
            return None, False

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
            "modified_date": date_updated,
            "epss": str(cve.get("epss", "")),
            "assigner": cve.get("assigner", "")
        }
        return cve_data, False
    except Exception as e:
        logging.error(f"Error parsing CVE {euvd_id}: {e}")
        return None, False

def save_to_csv(cve_data, date_str):
    """Save CVE data to a CSV file."""
    output_file = os.path.join(OUTPUT_DIR, f"euvd_cve_{date_str}.csv")
    fieldnames = [
        "euvd_id", "cve_id", "description", "base_score", "severity",
        "affected_products", "vendors", "references", "published_date",
        "modified_date", "epss", "assigner"
    ]
    
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cve in cve_data:
            writer.writerow(cve)
    
    logging.info(f"Saved {len(cve_data)} CVEs to {output_file}")

def main():
    logging.info("Starting daily EUVD CVE scrape...")
    
    # Set target date (yesterday) and day before
    today = datetime.utcnow()
    target_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")  # e.g., 2025-04-28
    day_before_target = (today - timedelta(days=2)).strftime("%Y-%m-%d")  # e.g., 2025-04-27
    
    logging.info(f"Fetching CVEs updated on {target_date}")
    
    cve_list = []
    page = 0
    
    while True:
        data = fetch_cves(page)
        if not data or not data.get("items"):
            logging.info(f"No more CVEs at page {page + 1}")
            break
        
        stop_fetching = False
        for cve in data.get("items", []):
            parsed_cve, should_stop = parse_cve_data(cve, target_date, day_before_target)
            if parsed_cve:
                cve_list.append(parsed_cve)
            if should_stop:
                stop_fetching = True
        
        if stop_fetching:
            logging.info(f"Stopping fetch at page {page + 1} due to dateUpdated earlier than {target_date}")
            break
        
        total = data.get("total", 0)
        logging.info(f"Page {page + 1}, Total CVEs: {total}")
        
        page += 1
        time.sleep(uniform(*REQUEST_DELAY))
    
    if cve_list:
        save_to_csv(cve_list, target_date)
    else:
        save_to_csv([], target_date)
    
    logging.info("Daily scrape completed")

if __name__ == "__main__":
    main()