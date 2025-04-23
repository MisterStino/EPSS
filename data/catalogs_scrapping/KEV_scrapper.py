import requests
import pandas as pd

# URL of the CISA KEV CSV
url = 'https://www.cisa.gov/sites/default/files/csv/known_exploited_vulnerabilities.csv'

# Download the CSV
response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to download the file. Status code: {response.status_code}")
    exit(1)


csv_path = 'data/catalogs/KEV_catalog.csv'
with open(csv_path, 'wb') as f:
    f.write(response.content)

df = pd.read_csv(csv_path)

# Rename 'cveID' to 'CVE_ID'
df.rename(columns={'cveID': 'CVE_ID'}, inplace=True)

# Drop unwanted columns
columns_to_remove = [
    'vulnerabilityName',
    'shortDescription',
    'requiredAction',
    'dueDate',
    'notes'
]
df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

# Save cleaned CSV
df.to_csv(csv_path, index=False)

