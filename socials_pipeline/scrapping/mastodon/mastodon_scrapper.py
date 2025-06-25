import os
import requests
from bs4 import BeautifulSoup
import re
import time
import csv
import json
from tqdm import tqdm

# Configuration
BASE_URL = "https://infosec.exchange"
TOKEN = os.getenv("MASTODON_ACCESS_TOKEN") or "M_dWGRSwTSk8JWbRnCjs2JpOebi3xJj28bZVoGBZEKk"
HEADERS = {
    'Authorization': f'Bearer {TOKEN}',
    'User-Agent': 'Mozilla/5.0',
}
CVE_PATTERN = re.compile(r"CVE-(\d{4})-(\d{4,7})")

PROGRESS_FILE = "data/mastodon/cve_scraper_progress.json"
CSV_FILE = "data/mastodon/mastodon_raw.csv"
PAGES_PER_RUN = 200
LIMIT_PER_PAGE = 100

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f).get("last_max_id")
    return None

def save_progress(max_id):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"last_max_id": max_id}, f)

def fetch_hashtag_toots(tag="cve", pages=200, limit=100, start_max_id=None):
    url_base = f"{BASE_URL}/api/v1/timelines/tag/{tag}?limit={limit}"
    all_statuses = []
    max_id = start_max_id

    print(f"Fetching up to {pages * limit} posts starting from max_id={start_max_id}...")
    for _ in tqdm(range(pages), desc="Downloading pages"):
        paged_url = url_base + (f"&max_id={max_id}" if max_id else "")
        try:
            response = requests.get(paged_url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            if not data:
                break
            all_statuses.extend(data)
            max_id = data[-1]["id"]
            time.sleep(1)
        except requests.RequestException as e:
            print(f"Error fetching: {e}")
            break

    return all_statuses, max_id

def extract_cves_and_append_to_csv(statuses, output_file=CSV_FILE):
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["id", "acct", "created_at", "url", "content", "cve_ids"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for status in tqdm(statuses, desc="Processing posts"):
            soup = BeautifulSoup(status.get("content", ""), "html.parser")
            text = soup.get_text()
            cve_matches = CVE_PATTERN.findall(text)
            cve_ids = [f"CVE-{year}-{id_}" for year, id_ in cve_matches]
            writer.writerow({
                "id": status["id"],
                "acct": status["account"]["acct"],
                "created_at": status["created_at"],
                "url": status["url"],
                "content": text.strip(),
                "cve_ids": ", ".join(cve_ids)
            })

if __name__ == "__main__":
    if not TOKEN or TOKEN == "your_token_here":
        print("Please set the MASTODON_ACCESS_TOKEN environment variable or replace 'your_token_here'.")
    else:
        last_max_id = load_progress()
        statuses, new_max_id = fetch_hashtag_toots(
            tag="cve", pages=PAGES_PER_RUN, limit=LIMIT_PER_PAGE, start_max_id=last_max_id
        )
        print(f"\nFetched {len(statuses)} statuses tagged with #cve")
        if statuses:
            extract_cves_and_append_to_csv(statuses)
            save_progress(new_max_id)
            print(f"✅ Done! Appended to {CSV_FILE}, progress saved.")
        else:
            print("🛑 No new statuses found. Nothing saved.")