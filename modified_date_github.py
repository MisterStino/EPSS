import subprocess
import csv
import re
from collections import defaultdict

CVE_REPO_PATH = 'cvelistV5'
OUTPUT_CSV = 'catalogs_raw/cve_modification_history.csv'

def parse_git_log():
    print("📜 Running full git log, please wait...")
    result = subprocess.run(
        ['git', '-C', CVE_REPO_PATH, 'log', '--name-only', '--pretty=format:%H|%ad', '--date=iso'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Git log failed: {result.stderr}")

    lines = result.stdout.splitlines()
    history = defaultdict(list)
    current_date = None

    cve_json_re = re.compile(r'^.*CVE-\d{4}-\d+\.json$')
    matched_any = False

    for line in lines:
        line = line.strip()
        if '|' in line:
            _, current_date = line.split('|', 1)
        elif cve_json_re.match(line):
            matched_any = True
            history[line].append(current_date)

    if not matched_any:
        print("⚠️ No matching CVE JSON files found in git log output.")
    else:
        print(f"✅ Found {len(history)} unique CVE files with history.")

    return history

def write_to_csv(history):
    print("📝 Writing results to CSV...")
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['CVE_ID', 'File_Path', 'Modification_Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file_path, dates in history.items():
            cve_id = file_path.replace('.json', '').split('/')[-1]
            for date in dates:
                writer.writerow({
                    'CVE_ID': cve_id,
                    'File_Path': file_path,
                    'Modification_Date': date
                })

    print(f"\n✅ Done! Output written to '{OUTPUT_CSV}'")

if __name__ == "__main__":
    history = parse_git_log()
    write_to_csv(history)
