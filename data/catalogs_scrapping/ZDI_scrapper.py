import requests
from bs4 import BeautifulSoup
import csv

# List to store the extracted records
advisories = []

for year in range(2005, 2026):
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
                affected_vendor_raw = row.find_all('td')[2].text.strip()

                # Remove duplicates while preserving order
                vendors = [v.strip() for v in affected_vendor_raw.split(',')]
                unique_vendors = []
                seen = set()
                for v in vendors:
                    if v and v not in seen:
                        seen.add(v)
                        unique_vendors.append(v)
                affected_vendor_clean = ", ".join(unique_vendors)

                cve = row.find_all('td')[3].text.strip()
                
                cvss = row.find_all('td')[4].text.strip()
                
                published = row.find_all('td')[5].text.strip()
                updated = row.find_all('td')[6].text.strip()

                advisories.append([affected_vendor_clean, cve, cvss, published, updated])

            except Exception as e:
                print(f"Error extracting data from row: {e}")
                continue

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        continue

with open('data/catalogs/ZDI_catalog.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Affected Vendor(s)', 'CVE_ID', 'CVSS v3.0', 'Published', 'Updated'])
    writer.writerows(advisories)
