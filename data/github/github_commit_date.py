import requests
import csv
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math

def get_github_token():
    # Replace with your GitHub Personal Access Token
    return ""

def run_rest_query(endpoint, query, token, page=1, per_page=100, retries=3):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com{endpoint}?q={query}&page={page}&per_page={per_page}"
    for attempt in range(retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 403 and "secondary rate limit" in response.text.lower():
            wait_times = [300, 1800, 3600]  # 5 min, 30 min, 60 min
            wait_time = wait_times[attempt]
            print(f"Hit secondary rate limit on attempt {attempt + 1}/{retries}. Waiting {wait_time // 60} minutes...")
            time.sleep(wait_time)
        else:
            raise Exception(f"REST query failed: {response.status_code} - {response.text}")
    raise Exception(f"Failed after {retries} retries due to secondary rate limit")

def fetch_commits(cve_id, start_dt, end_dt, all_commits, seen_shas, per_page=100):
    start_str = start_dt.strftime('%Y-%m-%d')
    end_str = end_dt.strftime('%Y-%m-%d')
    commit_query = f"\"{cve_id}\" author-date:{start_str}..{end_str}"
    print(f"Querying {cve_id} from {start_str} to {end_str}...")
    
    page = 1
    while True:
        result = run_rest_query("/search/commits", commit_query, token=get_github_token(), page=page, per_page=per_page)
        commits = result.get("items", [])
        for commit in commits:
            sha = commit.get('sha')
            if sha not in seen_shas:
                all_commits.append(commit)
                seen_shas.add(sha)
        if len(commits) < per_page:
            break
        page += 1
        time.sleep(5)

def get_commit_dates_by_cve(cve_id, start_date, end_date):
    start_dt = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
    end_dt = datetime.strptime(end_date, '%Y-%m-%d') if isinstance(end_date, str) else end_date

    # Check total count for full range
    commit_query = f"\"{cve_id}\" author-date:{start_dt.strftime('%Y-%m-%d')}..{end_dt.strftime('%Y-%m-%d')}"
    print(f"Fetching total count for {cve_id}...")
    result = run_rest_query("/search/commits", commit_query, token=get_github_token(), page=1, per_page=1)
    total_count = result.get("total_count", 0)
    print(f"Estimated total commits: {total_count}")

    all_commits = []
    seen_shas = set()

    # If <= 1000 commits, query full range
    if total_count <= 1000:
        fetch_commits(cve_id, start_dt, end_dt, all_commits, seen_shas)
    else:
        # Split into monthly intervals
        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(current_start + relativedelta(months=1) - relativedelta(days=1), end_dt)
            # Check total count for this month
            month_start_str = current_start.strftime('%Y-%m-%d')
            month_end_str = current_end.strftime('%Y-%m-%d')
            month_query = f"\"{cve_id}\" author-date:{month_start_str}..{month_end_str}"
            print(f"Checking total count for {cve_id} from {month_start_str} to {month_end_str}...")
            result = run_rest_query("/search/commits", month_query, token=get_github_token(), page=1, per_page=1)
            month_count = result.get("total_count", 0)
            print(f"Estimated commits for month: {month_count}")

            if month_count <= 1000:
                # Query the full month
                fetch_commits(cve_id, current_start, current_end, all_commits, seen_shas)
            else:
                # Split into daily intervals
                day_start = current_start
                while day_start <= current_end:
                    day_end = min(day_start, current_end)
                    fetch_commits(cve_id, day_start, day_end, all_commits, seen_shas)
                    day_start = day_start + relativedelta(days=1)

            current_start = current_end + relativedelta(days=1)
            time.sleep(5)

    # Extract commit dates
    commit_details = [
        {
            'cve_id': cve_id,
            'commit_date': commit['commit']['author']['date']
        }
        for commit in all_commits
    ]

    return commit_details

#change output file name if needed
def write_to_csv(details, output_file="github_commit_dates.csv", is_first_write=False):
    headers = ['cve_id', 'commit_date']
    mode = 'w' if is_first_write else 'a'
    with open(output_file, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if is_first_write:
            writer.writerow(headers)
        for detail in details:
            writer.writerow([detail['cve_id'], detail['commit_date']])

def read_cve_file(filename):
    cve_list = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) == 3:
                cve_id, start_date, end_date = row
                cve_list.append((cve_id.strip(), start_date.strip(), end_date.strip()))
    return cve_list

def main():
    #change the input file and output file at needs
    input_file = "cve_time_ranges.csv"
    output_file = "github_commit_dates.csv"

    try:
        cve_entries = read_cve_file(input_file)
        print(f"Loaded {len(cve_entries)} CVE entries")

        is_first_write = True
        for cve_id, start_date, end_date in cve_entries:
            print(f"Fetching commit dates for {cve_id} from {start_date} to {end_date}...")
            commit_details = get_commit_dates_by_cve(cve_id, start_date, end_date)
            print(f"Commits retrieved: {len(commit_details)}")
            if commit_details:
                write_to_csv(commit_details, output_file, is_first_write)
                is_first_write = False
            time.sleep(5)

        print(f"Commit dates saved to {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()