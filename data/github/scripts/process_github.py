import csv
from collections import defaultdict
import re

def extract_repo_and_sha(commit_url):
    """Extract repo URL and SHA from GitHub commit URL"""
    # Example: https://github.com/owner/repo/commit/sha
    pattern = r'(https://github\.com/[^/]+/[^/]+)/commit/([a-f0-9]{40})'
    match = re.search(pattern, commit_url)
    if match:
        repo_url = match.group(1)  # e.g., https://github.com/securezeron/cve_reports
        sha = match.group(2)       # e.g., 0b77b927e29a0510d913d15a35799041de908637
        return repo_url, sha
    return None, None

def process_cve_file(input_filename, output_filename):
    # Dictionary to store CVE counts and details
    cve_counts = defaultdict(int)
    cve_details = defaultdict(list)
    
    # Read the input CSV file
    with open(input_filename, 'r', newline='') as file:
        reader = csv.DictReader(file)  # Automatically uses the header row
        for row in reader:
            cve_id = row['cve_id'].strip()
            commit_url = row['commit_url'].strip()
            
            # Count occurrences
            cve_counts[cve_id] += 1
            
            # Extract repo URL and SHA
            repo_url, sha = extract_repo_and_sha(commit_url)
            if repo_url and sha:
                cve_details[cve_id].append({
                    'commit_url': commit_url,
                    'repo_url': repo_url,
                    'sha': sha
                })
            else:
                cve_details[cve_id].append({
                    'commit_url': commit_url,
                    'repo_url': 'Invalid URL',
                    'sha': 'Invalid SHA'
                })
    
    # Write results to a new CSV file
    with open(output_filename, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        # Write header
        writer.writerow(['cve_id', 'count', 'commit_url', 'repo_url', 'sha'])
        
        # Write data
        for cve_id, count in cve_counts.items():
            for detail in cve_details[cve_id]:
                writer.writerow([
                    cve_id,
                    count,
                    detail['commit_url'],
                    detail['repo_url'],
                    detail['sha']
                ])
    
    # Also print results to console for verification
    print("CVE ID Counts and Details:")
    for cve_id, count in cve_counts.items():
        print(f"CVE: {cve_id}, Count: {count}")
        for detail in cve_details[cve_id]:
            print(f"  Commit URL: {detail['commit_url']}")
            print(f"    Repo URL: {detail['repo_url']}")
            print(f"    SHA: {detail['sha']}")

if __name__ == "__main__":
    process_cve_file('github_commit_urls.csv', 'cve_results.csv')