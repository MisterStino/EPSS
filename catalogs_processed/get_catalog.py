import os
import re
import csv
import json
import subprocess
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
REPOS = [
    ("https://github.com/CVEProject/cvelistV5.git", "cvelistV5"),
    ("https://gitlab.com/exploit-database/exploitdb.git", "exploitdb"),
    ("https://github.com/cisagov/kev-data.git", "kev-data"),
]
CVE_DIR = "cvelistV5"
OUTPUT_DIR = "catalogs_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Repo Cloner ---
def clone_or_update_repos(repo_list):
    for url, directory in repo_list:
        if not os.path.isdir(directory):
            print(f"[+] Cloning {url} into {directory}...")
            subprocess.run(["git", "clone", url, directory], check=True)
        else:
            print(f"[+] Updating {directory}...")
            subprocess.run(["git", "-C", directory, "pull"], check=True)

# --- CVE Catalog Scraper ---
def scrape_cve(output_csv=f"{OUTPUT_DIR}/cve_catalog.csv"):
    file_paths = [
        os.path.join(root, filename)
        for root, _, files in os.walk(CVE_DIR)
        for filename in files if filename.endswith(".json")
    ]
    files_processed = 0
    fieldnames_written = False

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[])

        def process(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "cveMetadata" not in data:
                        return None

                    metadata = data["cveMetadata"]
                    containers = data.get("containers", {})
                    cna = containers.get("cna", {})

                    cvss_score, cvss_version = "N/A", "N/A"
                    for metric in cna.get("metrics", []):
                        for key, value in metric.items():
                            if "cvss" in key.lower() and "baseScore" in value:
                                cvss_score = value["baseScore"]
                                cvss_version = value.get("version", key.replace("cvss", "CVSS v"))
                                break
                        if cvss_score != "N/A": break

                    return {
                        "CVE_ID": metadata.get("cveId", ""),
                        "state": metadata.get("state", ""),
                        "assigner": metadata.get("assignerShortName", ""),
                        "date_published": metadata.get("datePublished", ""),
                        "date_updated": metadata.get("dateUpdated", ""),
                        "cvss_score": cvss_score,
                        "cvss_version": cvss_version,
                        "cwes": "; ".join(
                            d.get("cweId", "")
                            for pt in cna.get("problemTypes", [])
                            for d in pt.get("descriptions", [])
                            if d.get("lang") == "en"
                        )
                        #"references": "; ".join(ref.get("url", "") for ref in cna.get("references", [])),
                    }

            except Exception as e:
                print(f"[!] Error parsing {file_path}: {e}")
                return None

        with ThreadPoolExecutor() as executor, tqdm(total=len(file_paths), desc="CVE Scrape") as pbar:
            for future in as_completed([executor.submit(process, f) for f in file_paths]):
                data = future.result()
                if data:
                    if not fieldnames_written:
                        writer.fieldnames = data.keys()
                        writer.writeheader()
                        fieldnames_written = True
                    writer.writerow(data)
                    files_processed += 1
                pbar.update(1)

    print(f"[✔] CVE catalog: {files_processed} records saved to {output_csv}")

# --- Git History Extractor ---
def extract_git_history(output_csv=f"{OUTPUT_DIR}/cve_modification_history.csv"):
    print("[📜] Extracting git log for CVE file modification history...")

    result = subprocess.run(
        ['git', '-C', CVE_DIR, 'log', '--name-only', '--pretty=format:%H|%ad', '--date=iso'],
        stdout=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Git log failed: {result.stderr}")

    history, current_date = defaultdict(list), None
    for line in result.stdout.splitlines():
        line = line.strip()
        if '|' in line:
            _, current_date = line.split('|', 1)
        elif re.match(r'^.*CVE-\d{4}-\d+\.json$', line):
            history[line].append(current_date)

    with open(output_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["CVE_ID", "File_Path", "Modification_Date"])
        writer.writeheader()
        for path, dates in history.items():
            cve_id = Path(path).stem
            for date in dates:
                writer.writerow({"CVE_ID": cve_id, "File_Path": path, "Modification_Date": date})

    print(f"[✔] Git modification history saved to {output_csv}")

# --- ZDI Catalog Scraper ---
def scrape_zdi(output_csv=f"{OUTPUT_DIR}/ZDI_catalog.csv"):
    rows = []
    for year in range(2005, 2026):
        url = f"https://www.zerodayinitiative.com/advisories/published/{year}/"
        print(f"Fetching {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for row in soup.find_all("tr", id="publishedAdvisories"):
                try:
                    tds = row.find_all("td")
                    if len(tds) < 7:
                        continue
                    vendors = ", ".join(dict.fromkeys(v.strip() for v in tds[2].text.split(",")))
                    rows.append([vendors, tds[3].text.strip(), tds[4].text.strip()])
                except Exception as e:
                    print(f"[!] ZDI row error: {e}")
        except Exception as e:
            print(f"[!] Failed to fetch ZDI data for {year}: {e}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["Affected Vendor(s)", "CVE_ID", "CVSS v3.0"])
        csv.writer(f).writerows(rows)
    print(f"[✔] ZDI catalog: {len(rows)} records saved to {output_csv}")

# --- ExploitDB Catalog Scraper ---
def scrape_exploitdb(input_csv='exploitdb/files_exploits.csv', output_csv=f"{OUTPUT_DIR}/exploitDB_catalog.csv"):
    df = pd.read_csv(input_csv)
    df.rename(columns={'codes': 'CVE_ID'}, inplace=True)
    df['CVE_ID'] = df['CVE_ID'].apply(lambda x: re.findall(r'CVE-\d{4}-\d{4,}', str(x)) if pd.notna(x) else [])
    df = df.explode('CVE_ID').dropna(subset=['CVE_ID']).reset_index(drop=True)
    df.drop(columns=[
        'id', 'file', 'description', 'author', 'port', 'tags', 'aliases',
        'screenshot_url', 'application_url', 'source_url',
        'date_added', 'date_published', 'date_updated', 'verified'
    ], errors='ignore', inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"[✔] ExploitDB catalog saved to {output_csv}")

# --- KEV Catalog Scraper ---
def scrape_kev(input_csv='kev-data/known_exploited_vulnerabilities.csv', output_csv=f"{OUTPUT_DIR}/KEV_catalog.csv"):
    df = pd.read_csv(input_csv)
    df.rename(columns={'cveID': 'CVE_ID'}, inplace=True)
    df.drop(columns=[
        col for col in ['vulnerabilityName', 'shortDescription', 'requiredAction', 'dateAdded', 'dueDate', 'notes']
        if col in df.columns
    ], inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"[✔] KEV catalog saved to {output_csv}")

# --- Catalog Merger ---
import os
import pandas as pd

def merge_and_enrich_catalogs(
    cve_csv=f"{OUTPUT_DIR}/cve_catalog.csv",
    exploitdb_csv=f"{OUTPUT_DIR}/exploitDB_catalog.csv",
    kev_csv=f"{OUTPUT_DIR}/KEV_catalog.csv",
    zdi_csv=f"{OUTPUT_DIR}/ZDI_catalog.csv",
    output_csv="processed_data/enriched_catalog.csv"
):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Load input catalogs
    cve_df = pd.read_csv(cve_csv)
    exploit_df = pd.read_csv(exploitdb_csv)
    kev_df = pd.read_csv(kev_csv)
    zdi_df = pd.read_csv(zdi_csv)

    # Clean and deduplicate
    #cve_df = cve_df[cve_df["state"] == "PUBLISHED"].drop_duplicates("CVE_ID")
    exploit_df = exploit_df.drop_duplicates("CVE_ID").rename(columns={col: f"exploitDB_{col}" for col in exploit_df.columns if col != "CVE_ID"})
    kev_df = kev_df.drop_duplicates("CVE_ID").rename(columns={col: f"KEV_{col}" for col in kev_df.columns if col != "CVE_ID"})
    zdi_df = zdi_df.drop_duplicates("CVE_ID").rename(columns={col: f"ZDI_{col}" for col in zdi_df.columns if col != "CVE_ID"})

    # Merge catalogs
    merged = cve_df.merge(exploit_df, on="CVE_ID", how="outer") \
                   .merge(kev_df, on="CVE_ID", how="outer") \
                   .merge(zdi_df, on="CVE_ID", how="outer")

    # Fill and consolidate columns
    merged["cwes"] = merged["cwes"].fillna(merged.get("KEV_cwes"))
    merged["cvss_score"] = merged["cvss_score"].fillna(merged.get("ZDI_CVSS v3.0"))
    merged["Vendor"] = merged.get("ZDI_Affected Vendor(s)", "").fillna(merged.get("KEV_vendorProject"))

    # Drop now redundant columns
    for col in ["KEV_cwes", "ZDI_CVSS v3.0", "ZDI_Affected Vendor(s)", "KEV_vendorProject"]:
        if col in merged.columns:
            merged.drop(columns=col, inplace=True)

    # Save to file
    merged.to_csv(output_csv, index=False)
    print(f"[✔] Enriched catalog saved to {output_csv}")

def merge_social_media_with_catalog(
    cve_catalog_path='processed_data/enriched_catalog.csv',
    mastodon_path='processed_data/mastodon_catalog.csv',
    reddit_path='processed_data/reddit_catalog.csv',
    mastodon_output_path='parquet_preprocessing/merged_mastodon.csv',
    reddit_output_path='parquet_preprocessing/merged_reddit.csv'
):

    # Load CSV files
    df1 = pd.read_csv(cve_catalog_path)
    df2 = pd.read_csv(mastodon_path)
    df3 = pd.read_csv(reddit_path) 

    #df1.drop(columns=['vendor'], inplace=True)

    # Clean column names (strip any leading/trailing whitespace)
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    df3.columns = df3.columns.str.strip()


    # Merge CVE catalog with Mastodon data
    merged_mastodon = pd.merge(df1, df2, left_on='CVE_ID', right_on='cve_ids', how='inner')
    merged_mastodon.drop(columns=['cve_ids'], inplace=True)

    # Merge CVE catalog with Reddit data
    merged_reddit = pd.merge(df1, df3, left_on='CVE_ID', right_on='CVE_ID', how='inner')
    #merged_reddit.drop(columns=['CVE_ID'], inplace=True)

    # Save the merged data
    merged_mastodon.to_csv(mastodon_output_path, index=False)
    merged_reddit.to_csv(reddit_output_path, index=False)

    print(f"Merge completed. Output saved as:\n- {mastodon_output_path}\n- {reddit_output_path}")


# --- Pipeline Runner ---
if __name__ == "__main__":
    clone_or_update_repos(REPOS)
    scrape_zdi()
    scrape_cve()
    extract_git_history()
    scrape_exploitdb()
    scrape_kev()
    merge_and_enrich_catalogs()
    merge_social_media_with_catalog()