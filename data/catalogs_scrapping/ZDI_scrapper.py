import requests
from bs4 import BeautifulSoup
import csv

# List to store the extracted records
advisories = []

for year in range(2005, 2026):
    url = f"https://www.zerodayinitiative.com/advisories/published/{year}/"
    
    print(f"Fetching data from {url}")
    
    try:
        # Fetch the page for the specific year
        response = requests.get(url)
        response.raise_for_status()

        # Parse the page content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all rows in the table (rows under 'publishedAdvisories')
        rows = soup.find_all('tr', id='publishedAdvisories')

        # Iterate over each row and extract the necessary data
        for row in rows:
            try:
                # Extract data from each <td> element
                #zdi_id = row.find_all('td')[0].text.strip()
                #zdi_can = row.find_all('td')[1].text.strip()
                affected_vendor = row.find_all('td')[2].text.strip()
                
                # Extract the CVE directly from the 4th column (CVE column)
                cve = row.find_all('td')[3].text.strip()
                
                # Extract CVSS v3.0 from the 5th column
                cvss = row.find_all('td')[4].text.strip()
                
                # Extract the Published and Updated dates from columns 6 and 7 respectively
                published = row.find_all('td')[5].text.strip()
                updated = row.find_all('td')[6].text.strip()

                # Store the extracted data in the advisories list
                #advisories.append([zdi_id, zdi_can, affected_vendor, cve, cvss, published, updated])
                advisories.append([affected_vendor, cve, cvss, published, updated])

            except Exception as e:
                print(f"Error extracting data from row: {e}")
                continue

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        continue

# Save the extracted data to a CSV file
with open('data/catalogs/ZDI_catalog.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    #writer.writerow(['ZDI ID', 'ZDI CAN', 'Affected Vendor(s)', 'CVE_ID', 'CVSS v3.0', 'Published', 'Updated'])
    writer.writerow(['Affected Vendor(s)', 'CVE_ID', 'CVSS v3.0', 'Published', 'Updated'])
    writer.writerows(advisories)


