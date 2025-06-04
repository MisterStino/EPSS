import os
import subprocess
import requests
from bs4 import BeautifulSoup
import csv
import json
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import re
from pathlib import Path

#For CVE, exploitdb, and KEV catalogs
REPOS = [
    ("https://github.com/CVEProject/cvelistV5.git", "cvelistV5"),
    ("https://gitlab.com/exploit-database/exploitdb.git", "exploitdb"),
    ("https://github.com/cisagov/kev-data.git", "kev-data"),
]

def scrape_repos(repo_list):
    for url, directory in repo_list:
        if not os.path.isdir(directory):
            print(f"[+] Cloning {url} into {directory}...")
            subprocess.run(["git", "clone", "--depth", "1", url, directory], check=True)
        else:
            print(f"[+] Updating {directory}...")
            subprocess.run(["git", "-C", directory, "pull"], check=True)

#CVE catalog
def scrape_cve():

    CVE_DIR = "cvelistV5"
    OUTPUT_CSV = "catalogs_raw/cve_catalog.csv"
    MAX_FILES = None

    files_processed = 0
    fieldnames_written = False

    # Prepare CSV file for writing
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[])
        
        # Collect all JSON file paths
        file_paths = []
        for root, dirs, files in os.walk(CVE_DIR):
            for filename in files:
                if filename.endswith(".json"):
                    file_paths.append(os.path.join(root, filename))

        # Function to process each CVE file
        def process_cve_file(file_path):

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, dict) or "cveMetadata" not in data:
                        return None

                    metadata = data.get("cveMetadata", {})
                    containers = data.get("containers", {})
                    cna = containers.get("cna", {})

                    # Flattening top-level fields
                    cve_id = metadata.get("cveId", "")
                    state = metadata.get("state", "")
                    assigner = metadata.get("assignerShortName", "")
                    date_reserved = metadata.get("dateReserved", "")
                    date_published = metadata.get("datePublished", "")
                    date_updated = metadata.get("dateUpdated", "")

                    # Flattening description and title
                    description = next(
                        (d.get("value", "") for d in cna.get("descriptions", []) if d.get("lang") == "en"),
                        ""
                    )
                    title = cna.get("title", "")

                    # Flattening impact
                    impact = containers.get("impact", "")

                    # Flattening metrics (CVSS Scores)
                    cvss_score = "N/A"
                    cvss_version = "N/A"

                    metrics = cna.get("metrics", [])
                    for metric in metrics:
                        for key, value in metric.items():
                            if "cvss" in key.lower() and "baseScore" in value:
                                cvss_score = value["baseScore"]
                                if "version" in value:
                                    cvss_version = value["version"]
                                else:
                                    cvss_version = key.replace("cvss", "CVSS v")

                                # Assuming we want the first available score, break after the first valid one
                                break
                        if cvss_score != "N/A":
                            break  # Exit once a valid score is found

                    # If no CVSS score was found, set them to a placeholder
                    if cvss_score == "N/A":
                        cvss_version = "N/A"

                    # Flattening CWE(s)
                    cwes = "; ".join(
                        d.get("cweId", "")
                        for pt in cna.get("problemTypes", [])
                        for d in pt.get("descriptions", [])
                        if d.get("lang") == "en"
                    )

                    # Flattening references
                    references = "; ".join(ref.get("url", "") for ref in cna.get("references", []))

                    # Flattening discovery method
                    discovery_method = cna.get("discoveryMethod", "")

                    # Flattening org_id (from providerMetadata)
                    org_id = cna.get("providerMetadata", {}).get("orgId", "")

                    # Combine all the fields into a single dictionary
                    flattened_data = {
                        "CVE_ID": cve_id,
                        "state": state,
                        "assigner": assigner,
                        #"date_reserved": date_reserved,
                        "date_published": date_published,
                        "date_updated": date_updated,
                        #"title": title,
                        #"description": description,
                        #"impact": impact,
                        "cvss_score": cvss_score,
                        "cvss_version": cvss_version,
                        "cwes": cwes,
                        #"references": references,
                        #"discovery_method": discovery_method,
                        #"org_id": org_id
                    }

                    return flattened_data

            except Exception as e:
                print(f"[!] Error parsing {file_path}: {e}")
                return None

        # Function to write to the CSV file safely
        def write_to_csv(flattened_data):
            nonlocal files_processed, fieldnames_written
            if flattened_data:
                # Dynamically update fieldnames
                fieldnames = flattened_data.keys()
                
                # Write headers only once
                if not fieldnames_written:
                    writer.fieldnames = fieldnames
                    writer.writeheader()
                    fieldnames_written = True
                
                # Write the data to CSV
                writer.writerow(flattened_data)
                files_processed += 1
                # Print progress
                print(f"Processed {files_processed} files...", end="\r")

        # Use ThreadPoolExecutor to process files in parallel
        with ThreadPoolExecutor() as executor:
            # Submit tasks for each file processing
            futures = [executor.submit(process_cve_file, file_path) for file_path in file_paths]
            
            # Use tqdm for progress bar
            with tqdm(total=len(futures), desc="Processing CVE files") as progress_bar:
                for future in as_completed(futures):
                    flattened_data = future.result()
                    write_to_csv(flattened_data)
                    progress_bar.update(1)  # Update progress bar

        print(f"\nProcessed {files_processed} files and saved to {OUTPUT_CSV}")

#ZDI catalog (Extracted using beautifulsoup)
def scrape_zdi(output_file):

    list = []

    start_year = 2005
    end_year = 2025

    for year in range(start_year, end_year + 1):
        url = f"https://www.zerodayinitiative.com/advisories/published/{year}/"
        print(f"Fetching data from {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all('tr', id='publishedAdvisories')

            for row in rows:
                try:
                    # Extract data from each <td> element
                    tds = row.find_all('td')
                    if len(tds) < 7:  # Skip rows with insufficient data
                        continue

                    affected_vendor_raw = tds[2].text.strip()

                    # Remove duplicates while preserving order
                    vendors = [v.strip() for v in affected_vendor_raw.split(',')]
                    unique_vendors = []
                    seen = set()
                    for v in vendors:
                        if v and v not in seen:
                            seen.add(v)
                            unique_vendors.append(v)
                    affected_vendor_clean = ", ".join(unique_vendors)

                    cve = tds[3].text.strip()
                    cvss = tds[4].text.strip()
                    published = tds[5].text.strip()
                    updated = tds[6].text.strip()

                    #list.append([affected_vendor_clean, cve, cvss, published, updated])
                    list.append([affected_vendor_clean, cve, cvss])

                except Exception as e:
                    print(f"Error extracting data from row: {e}")
                    continue

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue

    # Write the collected data to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        #writer.writerow(['Affected Vendor(s)', 'CVE_ID', 'CVSS v3.0', 'Published', 'Updated'])
        writer.writerow(['Affected Vendor(s)', 'CVE_ID', 'CVSS v3.0'])

        writer.writerows(list)

    print(f"[✔] Cleaned ZDI: {len(list)} records saved to {output_file}.")

#ExploitDB Catalog
def scrape_exploitDB(input_csv='exploitdb/files_exploits.csv', output_csv='catalogs_raw/exploitDB_catalog.csv'):
    # Read CSV
    df = pd.read_csv(input_csv)

    # Rename 'codes' to 'CVE_ID'
    df.rename(columns={'codes': 'CVE_ID'}, inplace=True)

    # Extract CVE patterns
    def extract_cves(cell):
        if pd.isna(cell):
            return []
        return re.findall(r'CVE-\d{4}-\d{4,}', str(cell))

    df['CVE_ID'] = df['CVE_ID'].apply(extract_cves)

    # Expand list of CVEs into separate rows
    df = df.explode('CVE_ID').reset_index(drop=True)

    # Drop irrelevant columns
    columns_to_drop = [
        'id', 'file', 'description', 'author', 'port',
        'tags', 'aliases', 'screenshot_url', 'application_url',
        'source_url', 'date_added', 'date_published', 'date_updated', 'verified'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Drop empty CVE IDs
    df = df[df['CVE_ID'].notna() & (df['CVE_ID'] != "")]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"[✔] Cleaned ExploitDB catalog saved to: {output_csv}")

#KEV Catalog
def scrape_kev(input_csv='kev-data/known_exploited_vulnerabilities.csv', output_csv='catalogs_raw/KEV_catalog.csv'):
    
    df2 = pd.read_csv(input_csv)

    # Standardize column names
    df2.rename(columns={'cveID': 'CVE_ID'}, inplace=True)

    # Drop unwanted columns, only if they exist
    columns_to_remove = [
        'vulnerabilityName',
        'shortDescription',
        'requiredAction',
        'dateAdded',
        'dueDate',
        'notes'
    ]
    df2.drop(columns=[col for col in columns_to_remove if col in df2.columns], inplace=True)

    df2.to_csv(output_csv, index=False)

    print(f"[✔] Cleaned KEV catalog saved to: {output_csv}")


########################################RUN THIS TO HAVE CATALOGS EXTRACTED + UP TO DATE
#Extract reposotories
scrape_repos(REPOS)

#ZDI
scrape_zdi('catalogs_raw/ZDI_catalog.csv') 

########################################RUN THIS TO CONVERT CATALOGS TO CSV + POLISHING 
#CVE
scrape_cve()

#ExploitDB
scrape_exploitDB()

#KEV
scrape_kev()


import os
import subprocess
import requests
from bs4 import BeautifulSoup
import csv
import json
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import re
from pathlib import Path

#For CVE, exploitdb, and KEV catalogs
REPOS = [
    ("https://github.com/CVEProject/cvelistV5.git", "cvelistV5"),
    ("https://gitlab.com/exploit-database/exploitdb.git", "exploitdb"),
    ("https://github.com/cisagov/kev-data.git", "kev-data"),
]

def scrape_repos(repo_list):
    for url, directory in repo_list:
        if not os.path.isdir(directory):
            print(f"[+] Cloning {url} into {directory}...")
            subprocess.run(["git", "clone", "--depth", "1", url, directory], check=True)
        else:
            print(f"[+] Updating {directory}...")
            subprocess.run(["git", "-C", directory, "pull"], check=True)

#CVE catalog
def scrape_cve():

    CVE_DIR = "cvelistV5"
    OUTPUT_CSV = "catalogs_raw/cve_catalog.csv"
    MAX_FILES = None

    files_processed = 0
    fieldnames_written = False

    # Prepare CSV file for writing
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[])
        
        # Collect all JSON file paths
        file_paths = []
        for root, dirs, files in os.walk(CVE_DIR):
            for filename in files:
                if filename.endswith(".json"):
                    file_paths.append(os.path.join(root, filename))

        # Function to process each CVE file
        def process_cve_file(file_path):

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, dict) or "cveMetadata" not in data:
                        return None

                    metadata = data.get("cveMetadata", {})
                    containers = data.get("containers", {})
                    cna = containers.get("cna", {})

                    # Flattening top-level fields
                    cve_id = metadata.get("cveId", "")
                    state = metadata.get("state", "")
                    assigner = metadata.get("assignerShortName", "")
                    date_reserved = metadata.get("dateReserved", "")
                    date_published = metadata.get("datePublished", "")
                    date_updated = metadata.get("dateUpdated", "")

                    # Flattening description and title
                    description = next(
                        (d.get("value", "") for d in cna.get("descriptions", []) if d.get("lang") == "en"),
                        ""
                    )
                    title = cna.get("title", "")

                    # Flattening impact
                    impact = containers.get("impact", "")

                    # Flattening metrics (CVSS Scores)
                    cvss_score = "N/A"
                    cvss_version = "N/A"

                    metrics = cna.get("metrics", [])
                    for metric in metrics:
                        for key, value in metric.items():
                            if "cvss" in key.lower() and "baseScore" in value:
                                cvss_score = value["baseScore"]
                                if "version" in value:
                                    cvss_version = value["version"]
                                else:
                                    cvss_version = key.replace("cvss", "CVSS v")

                                # Assuming we want the first available score, break after the first valid one
                                break
                        if cvss_score != "N/A":
                            break  # Exit once a valid score is found

                    # If no CVSS score was found, set them to a placeholder
                    if cvss_score == "N/A":
                        cvss_version = "N/A"

                    # Flattening CWE(s)
                    cwes = "; ".join(
                        d.get("cweId", "")
                        for pt in cna.get("problemTypes", [])
                        for d in pt.get("descriptions", [])
                        if d.get("lang") == "en"
                    )

                    # Flattening references
                    references = "; ".join(ref.get("url", "") for ref in cna.get("references", []))

                    # Flattening discovery method
                    discovery_method = cna.get("discoveryMethod", "")

                    # Flattening org_id (from providerMetadata)
                    org_id = cna.get("providerMetadata", {}).get("orgId", "")

                    # Combine all the fields into a single dictionary
                    flattened_data = {
                        "CVE_ID": cve_id,
                        "state": state,
                        "assigner": assigner,
                        #"date_reserved": date_reserved,
                        "date_published": date_published,
                        "date_updated": date_updated,
                        #"title": title,
                        #"description": description,
                        #"impact": impact,
                        "cvss_score": cvss_score,
                        "cvss_version": cvss_version,
                        "cwes": cwes,
                        #"references": references,
                        #"discovery_method": discovery_method,
                        #"org_id": org_id
                    }

                    return flattened_data

            except Exception as e:
                print(f"[!] Error parsing {file_path}: {e}")
                return None

        # Function to write to the CSV file safely
        def write_to_csv(flattened_data):
            nonlocal files_processed, fieldnames_written
            if flattened_data:
                # Dynamically update fieldnames
                fieldnames = flattened_data.keys()
                
                # Write headers only once
                if not fieldnames_written:
                    writer.fieldnames = fieldnames
                    writer.writeheader()
                    fieldnames_written = True
                
                # Write the data to CSV
                writer.writerow(flattened_data)
                files_processed += 1
                # Print progress
                print(f"Processed {files_processed} files...", end="\r")

        # Use ThreadPoolExecutor to process files in parallel
        with ThreadPoolExecutor() as executor:
            # Submit tasks for each file processing
            futures = [executor.submit(process_cve_file, file_path) for file_path in file_paths]
            
            # Use tqdm for progress bar
            with tqdm(total=len(futures), desc="Processing CVE files") as progress_bar:
                for future in as_completed(futures):
                    flattened_data = future.result()
                    write_to_csv(flattened_data)
                    progress_bar.update(1)  # Update progress bar

        print(f"\nProcessed {files_processed} files and saved to {OUTPUT_CSV}")

#ZDI catalog (Extracted using beautifulsoup)
def scrape_zdi(output_file):

    list = []

    start_year = 2005
    end_year = 2025

    for year in range(start_year, end_year + 1):
        url = f"https://www.zerodayinitiative.com/advisories/published/{year}/"
        print(f"Fetching data from {url}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.find_all('tr', id='publishedAdvisories')

            for row in rows:
                try:
                    # Extract data from each <td> element
                    tds = row.find_all('td')
                    if len(tds) < 7:  # Skip rows with insufficient data
                        continue

                    affected_vendor_raw = tds[2].text.strip()

                    # Remove duplicates while preserving order
                    vendors = [v.strip() for v in affected_vendor_raw.split(',')]
                    unique_vendors = []
                    seen = set()
                    for v in vendors:
                        if v and v not in seen:
                            seen.add(v)
                            unique_vendors.append(v)
                    affected_vendor_clean = ", ".join(unique_vendors)

                    cve = tds[3].text.strip()
                    cvss = tds[4].text.strip()
                    published = tds[5].text.strip()
                    updated = tds[6].text.strip()

                    #list.append([affected_vendor_clean, cve, cvss, published, updated])
                    list.append([affected_vendor_clean, cve, cvss])

                except Exception as e:
                    print(f"Error extracting data from row: {e}")
                    continue

        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue

    # Write the collected data to a CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        #writer.writerow(['Affected Vendor(s)', 'CVE_ID', 'CVSS v3.0', 'Published', 'Updated'])
        writer.writerow(['Affected Vendor(s)', 'CVE_ID', 'CVSS v3.0'])

        writer.writerows(list)

    print(f"[✔] Cleaned ZDI: {len(list)} records saved to {output_file}.")

#ExploitDB Catalog
def scrape_exploitDB(input_csv='exploitdb/files_exploits.csv', output_csv='catalogs_raw/exploitDB_catalog.csv'):
    # Read CSV
    df = pd.read_csv(input_csv)

    # Rename 'codes' to 'CVE_ID'
    df.rename(columns={'codes': 'CVE_ID'}, inplace=True)

    # Extract CVE patterns
    def extract_cves(cell):
        if pd.isna(cell):
            return []
        return re.findall(r'CVE-\d{4}-\d{4,}', str(cell))

    df['CVE_ID'] = df['CVE_ID'].apply(extract_cves)

    # Expand list of CVEs into separate rows
    df = df.explode('CVE_ID').reset_index(drop=True)

    # Drop irrelevant columns
    columns_to_drop = [
        'id', 'file', 'description', 'author', 'port',
        'tags', 'aliases', 'screenshot_url', 'application_url',
        'source_url', 'date_added', 'date_published', 'date_updated', 'verified'
    ]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Drop empty CVE IDs
    df = df[df['CVE_ID'].notna() & (df['CVE_ID'] != "")]

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"[✔] Cleaned ExploitDB catalog saved to: {output_csv}")

#KEV Catalog
def scrape_kev(input_csv='kev-data/known_exploited_vulnerabilities.csv', output_csv='catalogs_raw/KEV_catalog.csv'):
    
    df2 = pd.read_csv(input_csv)

    # Standardize column names
    df2.rename(columns={'cveID': 'CVE_ID'}, inplace=True)

    # Drop unwanted columns, only if they exist
    columns_to_remove = [
        'vulnerabilityName',
        'shortDescription',
        'requiredAction',
        'dateAdded',
        # 'dueDate',
        'notes'
    ]
    df2.drop(columns=[col for col in columns_to_remove if col in df2.columns], inplace=True)

    df2.to_csv(output_csv, index=False)

    print(f"[✔] Cleaned KEV catalog saved to: {output_csv}")


########################################RUN THIS TO HAVE CATALOGS EXTRACTED + UP TO DATE
#Extract reposotories
scrape_repos(REPOS)

#ZDI
scrape_zdi('catalogs_raw/ZDI_catalog.csv') 

########################################RUN THIS TO CONVERT CATALOGS TO CSV + POLISHING 
#CVE
scrape_cve()

#ExploitDB
scrape_exploitDB()

#KEV
scrape_kev()


