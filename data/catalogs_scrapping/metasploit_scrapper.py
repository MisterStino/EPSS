import re
import csv
from pathlib import Path
import subprocess


def run_git_clone(repo_url):
    try:
        # Run the git clone command
        subprocess.run(['git', 'clone', repo_url], check=True)
        print(f"Successfully cloned {repo_url}")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")

repo_url = "https://github.com/rapid7/metasploit-framework.git"
run_git_clone(repo_url)

# Root path to Metasploit modules directory
MODULES_DIR = Path('metasploit-framework/modules')
OUTPUT_CSV = 'data/catalogs/metasploit_catalog.csv'

# Regex patterns
CVE_REGEX = re.compile(r'CVE-\d{4}-\d{4,7}', re.IGNORECASE)
DATE_REGEX = re.compile(r"'DisclosureDate'\s*=>\s*['\"](.+?)['\"]", re.IGNORECASE)

results = []

for path in MODULES_DIR.rglob("*.rb"):
    try:
        text = path.read_text(errors='ignore')

        # Extract CVEs
        cves = sorted(set(CVE_REGEX.findall(text)))

        # Extract Disclosure Date
        date_m = DATE_REGEX.search(text)
        date = date_m.group(1).strip() if date_m else ''

        # Only include rows with a non-empty date
        if date:
            for cve in cves:
                results.append([date, cve])

    except Exception as e:
        print(f"Error reading {path}: {e}")

# Write CSV
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Disclosure Date', 'CVE_ID'])
    w.writerows(results)

