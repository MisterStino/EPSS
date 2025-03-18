import os
import requests
import datetime
import time
import json

def log_error(date_str, error_msg, error_file="temp_error.json"):
    """
    Logs an error occurrence for a particular date to a JSON file.
    The JSON file is a list of objects: [{ "date": "YYYY-MM-DD", "error": "..."}].
    If the file doesn't exist, create it; otherwise, append to it.
    """
    entry = {"date": date_str, "error": error_msg}

    if os.path.exists(error_file):
        # Read existing errors (if any)
        try:
            with open(error_file, 'r') as ef:
                errors = json.load(ef)
        except (json.JSONDecodeError, OSError):
            errors = []
    else:
        errors = []

    errors.append(entry)

    # Write back
    with open(error_file, 'w') as ef:
        json.dump(errors, ef, indent=2)

def download_epss_for_day(date_str, raw_folder="data/raw", error_file="temp_error.json"):
    """
    Downloads the EPSS csv.gz file for a single day (YYYY-MM-DD) into 'raw_folder'.
    If the file already exists, it skips downloading. 
    If the download fails, logs an error to error_file and returns None.

    :param date_str: The date in 'YYYY-MM-DD' format.
    :param raw_folder: Folder where the raw file is stored.
    :param error_file: Where to log download errors if any occur.
    :return: The local file path if successful (or already exists), otherwise None.
    """
    url = f"https://epss.cyentia.com/epss_scores-{date_str}.csv.gz"
    local_filename = f"epss_scores-{date_str}.csv.gz"
    local_path = os.path.join(raw_folder, local_filename)

    # Ensure the raw directory exists
    os.makedirs(raw_folder, exist_ok=True)

    # Skip if already downloaded
    if os.path.exists(local_path):
        print(f"[INFO] {local_filename} already exists. Skipping download.")
        return local_path

    # Attempt download
    print(f"[INFO] Downloading {url} -> {local_path}")
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()  # Raises if 4xx or 5xx error
        with open(local_path, 'wb') as f:
            f.write(response.content)

        print(f"[INFO] Download complete: {local_path}")
        return local_path

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        log_error(date_str, str(e), error_file=error_file)
        return None

def get_all_epss_data(raw_folder="data/raw", error_file="temp_error.json"):
    """
    Downloads EPSS csv.gz files from the earliest available date (2021-04-14)
    up to today's date. Skips any days already downloaded. Logs errors in error_file.

    You can run this any time to ensure you have the latest daily EPSS data up to now.
    """
    start_dt = datetime.date(2021, 4, 14)      # earliest available
    end_dt = datetime.date.today()            # dynamic current day

    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.isoformat()
        saved_file = download_epss_for_day(date_str, raw_folder, error_file)

        # Optional short sleep to respect rate limits
        # time.sleep(1)

        current_dt += datetime.timedelta(days=1)

    print("\n[INFO] Completed downloads from 2021-04-14 through today.")
    print(f"[INFO] Any download errors were logged to: {error_file}")

if __name__ == "__main__":
    # Example usage:
    # This function ensures all daily CSVs from 2021-04-14 to "today" exist.
    get_all_epss_data(raw_folder="data/raw", error_file="temp_error.json")
