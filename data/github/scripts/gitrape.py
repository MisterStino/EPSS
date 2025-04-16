import requests
import csv
import time
from datetime import datetime

def get_github_token():
    # Replace with your GitHub Personal Access Token
    return ""

def run_rest_query(endpoint, query, token, page=1, per_page=100, retries=3):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    url = f"https://api.github.com{endpoint}?q={query}&page={page}&per_page={per_page}"
    for attempt in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            print(f"REST response (page {page}):", result)  # Debug
            return result
        elif response.status_code == 403 and "secondary rate limit" in response.text.lower():
            wait_times = [300, 1800, 3600]  # 5 min, 30 min, 60 min
            wait_time = wait_times[attempt]
            print(f"Hit secondary rate limit on attempt {attempt + 1}/{retries}. Waiting {wait_time // 60} minutes ({wait_time} seconds)...")
            time.sleep(wait_time)
        else:
            raise Exception(f"REST query failed: {response.status_code} - {response.text}")
    raise Exception(f"Failed after {retries} retries due to secondary rate limit")

def get_commit_urls_by_cve(cve_id, start_date, end_date):
    token = get_github_token()
    start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
    end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date

    # REST API for Commits with date range
    commit_query = f"\"{cve_id}\" author-date:{start_str}..{end_str}"
    page = 1
    per_page = 100  # Max allowed by GitHub API
    all_commits = []

    while True:
        commit_result = run_rest_query("/search/commits", commit_query, token, page, per_page)
        commits = commit_result.get("items", [])
        all_commits.extend(commits)

        # Check if there are more pages
        if len(commits) < per_page:
            break
        page += 1
        time.sleep(5)  # 5s between pages

    # Extract commit URLs
    commit_details = []
    for commit in all_commits:
        commit_url = commit.get('html_url', '')  # Use html_url for direct commit link
        commit_info = {
            'cve_id': cve_id,
            'start_date': start_str,
            'end_date': end_str,
            'commit_url': commit_url
        }
        commit_details.append(commit_info)

    return commit_details

def get_latest_end_date(output_file="github_commit_urls.csv"):
    try:
        with open(output_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            end_dates = [row[2] for row in reader if len(row) >= 3]  # Get end_date column
            if end_dates:
                return max(end_dates, key=lambda d: datetime.strptime(d, '%Y-%m-%d'))
            return None
    except FileNotFoundError:
        return None

def write_to_csv(details, output_file="github_commit_urls.csv", is_first_write=False):
    headers = ['cve_id', 'start_date', 'end_date', 'commit_url']
    mode = 'w' if is_first_write else 'a'
    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if is_first_write:
            writer.writerow(headers)
        for detail in details:
            row = [
                detail['cve_id'], detail['start_date'], detail['end_date'],
                detail['commit_url']
            ]
            writer.writerow(row)

def read_cve_file(filename):
    cve_list = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row ("cve_id,start_date,end_date")
        for row in reader:
            if len(row) == 3:
                cve_id, start_date, end_date = row
                cve_list.append((cve_id.strip(), start_date.strip(), end_date.strip()))
    return cve_list

def main():
    input_file = "cve_time_ranges.csv"
    output_file = "github_commit_urls.csv"
    
    try:
        cve_entries = read_cve_file(input_file)
        print(f"Loaded {len(cve_entries)} CVE entries from {input_file}")
        
        # Get the latest end_date from the existing CSV
        latest_end_date = get_latest_end_date(output_file)
        if latest_end_date:
            latest_end = datetime.strptime(latest_end_date, '%Y-%m-%d')
            print(f"Latest end date in {output_file}: {latest_end_date}")
        else:
            latest_end = None
            print(f"No existing data in {output_file}, starting fresh")

        is_first_write = not bool(latest_end_date)  # Write headers if no file exists
        for cve_id, start_date, end_date in cve_entries:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            # Only process if end_date is after the latest recorded end_date
            if latest_end and end_dt <= latest_end:
                print(f"Skipping {cve_id} (end_date {end_date} <= latest {latest_end_date})")
                continue

            print(f"Fetching commit URLs for {cve_id} from {start_date} to {end_date}...")
            commit_details = get_commit_urls_by_cve(cve_id, start_date, end_date)
            print(f"Commits retrieved: {len(commit_details)}")
            if commit_details:
                write_to_csv(commit_details, output_file, is_first_write)
                is_first_write = False
            time.sleep(5)  # 5s between CVE queries
        
        print(f"All new commit URLs saved to {output_file}")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()