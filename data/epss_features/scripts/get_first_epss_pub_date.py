import os
import csv
import time
import requests
import math

def fetch_epss_pub_date_for_cves(
    cve_csv_path='data/general_utils/files/all_unique_cves.csv',
    output_csv_path='data/general_utils/files/epss_pub_date.csv',
    chunk_size=50,
    wait_secs=1.0
):
    """
    Attempts to fetch the earliest EPS publication date for each CVE
    from the EPSS API.
    
    Steps:
      1) Read the cve_csv_path for all CVE IDs.
      2) Chunk them (chunk_size).
      3) For each chunk, do a GET request:
           GET https://api.first.org/data/v1/epss?cve=cve1,cve2,...&scope=public
         parse the JSON "data" array. 
      4) If the response includes a field for "days" or "created", compute epss_pub_date.
         Otherwise, store None or empty string.
      5) Rate-limit: wait wait_secs after each request.
      6) Write a single CSV with columns: [cve, epss_pub_date].
      
    NOTE: 
      - The official doc snippet doesn't show 'days' or 'created' in the response. 
      - If the API doesn't provide it, we can't fill it in accurately.
      - This code will attempt to parse them if they do appear.
    """

    # 1) Read local CSV of CVEs
    with open(cve_csv_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    # Assume first line is header "cve", subsequent lines are cve IDs
    cve_list = lines[1:] if lines and lines[0].lower().startswith("cve") else lines
    cve_list = [c.strip() for c in cve_list if c.strip()]

    print(f"Loaded {len(cve_list)} CVEs from {cve_csv_path}")

    # We'll store results in a dict: cve -> epss_pub_date (string or None)
    results = {}

    # 2) Chunk them
    total_cves = len(cve_list)
    num_chunks = math.ceil(total_cves / chunk_size)

    base_url = "https://api.first.org/data/v1/epss"
    
    for i in range(num_chunks):
        chunk = cve_list[i*chunk_size:(i+1)*chunk_size]
        cves_str = ",".join(chunk)
        # 3) Build the request
        params = {
            "cve": cves_str,
            "scope": "public",
            "limit": 1000,   # up to 1000 per doc
            # you might also use 'envelope': 'false' if you prefer
        }

        try:
            # do GET
            resp = requests.get(base_url, params=params, timeout=30)
            resp.raise_for_status()
            
            data_json = resp.json()
            if isinstance(data_json, dict) and "data" in data_json:
                for item in data_json["data"]:
                    cve_id = item.get("cve", "")
                    # The doc shows 'cve, epss, percentile, date'
                    # Possibly there's 'days' or 'created'? We'll check:
                    days_val = item.get("days", None)
                    created_val = item.get("created", None)
                    
                    # We'll compute epss_pub_date if we have 'days'
                    if days_val is not None:
                        # days_val might be a string or int, parse it
                        try:
                            days_int = int(days_val)
                            # approximate calculation: "today minus days_val"
                            # We'll store a string "today - X days" or do a real date if you want:
                            # e.g. if you want to do real date:
                            # from datetime import date, timedelta
                            # epss_pub_date = str(date.today() - timedelta(days=days_int))
                            # but let's just store the integer or a message
                            epss_pub_date = f"{days_int} days ago"
                        except:
                            epss_pub_date = None
                    elif created_val is not None:
                        # Maybe we can store created_val as the publication date
                        epss_pub_date = created_val
                    else:
                        # no data found
                        epss_pub_date = None
                    
                    results[cve_id] = epss_pub_date

            else:
                print(f"Unexpected JSON structure for chunk {i+1}/{num_chunks}: {data_json}")

        except Exception as e:
            print(f"Error fetching chunk {i+1}/{num_chunks}: {e}")
        
        # Rate limit
        time.sleep(wait_secs)

        print(f"Processed chunk {i+1}/{num_chunks}")

    # 4) Write to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["cve", "epss_pub_date"])
        for cve_id in cve_list:
            epss_pub_date = results.get(cve_id, None)
            writer.writerow([cve_id, epss_pub_date if epss_pub_date else ""])

    print(f"Done. Wrote results to {output_csv_path}")

if __name__ == "__main__":
    fetch_epss_pub_date_for_cves(
        cve_csv_path='data/general_utils/files/all_unique_cves.csv',
        output_csv_path='data/general_utils/files/epss_pub_date.csv',
        chunk_size=50,
        wait_secs=1.0
    )
