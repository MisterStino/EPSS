import os
import gzip
import json
import csv
import requests

# Settings
os.makedirs("nvd_data", exist_ok=True)
os.makedirs("data/catalogs", exist_ok=True)  # Create the output folder

BASE_URL = "https://nvd.nist.gov/feeds/json/cve/1.1/"
YEARS = list(range(2002, 2026))

CSV_OUTPUT_PATH = os.path.join("data", "catalogs", "CVE_catalog.csv")

# CSV_FIELDS = [
#     'CVE_ID', 'Published_Date', 'Last_Modified_Date', 'Description',
#     'CWE_ID', 'CVSSv3_Base_Score', 'CVSSv3_Severity',
#     'Attack_Vector', 'Attack_Complexity', 'Privileges_Required',
#     'User_Interaction', 'Scope',
#     'Confidentiality_Impact', 'Integrity_Impact', 'Availability_Impact',
#     'Vendors_Products'
# ]

CSV_FIELDS = [
    'CVE_ID', 'Published_Date', 'Last_Modified_Date',
    'CWE_ID', 'CVSSv3_Base_Score', 'CVSSv3_Severity',
    'Attack_Vector', 'Attack_Complexity', 'Privileges_Required',
    'User_Interaction', 'Scope',
    'Confidentiality_Impact', 'Integrity_Impact', 'Availability_Impact'
]

# CSV output
with open(CSV_OUTPUT_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
    writer.writeheader()

    for year in YEARS:
        filename = f"nvdcve-1.1-{year}.json.gz"
        filepath = os.path.join("nvd_data", filename)
        url = BASE_URL + filename

        # Download
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            r = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(r.content)

        print(f"Processing {filename}...")
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)

            for item in data['CVE_Items']:
                cve_id = item['cve']['CVE_data_meta']['ID']
                published = item.get('publishedDate', '').split('T')[0]
                modified = item.get('lastModifiedDate', '').split('T')[0]

                description = ''
                try:
                    description = item['cve']['description']['description_data'][0]['value']
                except (KeyError, IndexError):
                    pass

                cwe = ''
                try:
                    cwe_raw = item['cve']['problemtype']['problemtype_data'][0]['description'][0]['value']
                    if cwe_raw not in ("NVD-CWE-noinfo", "NVD-CWE-Other"):
                        cwe = cwe_raw
                except (KeyError, IndexError):
                    pass

                # Initialize CVSS fields
                cvss_score = severity = attack_vector = attack_complexity = ''
                privileges = user_interaction = scope = ''
                conf_impact = integ_impact = avail_impact = ''

                try:
                    cvss = item['impact']['baseMetricV3']['cvssV3']
                    cvss_score = cvss['baseScore']
                    severity = cvss['baseSeverity']
                    attack_vector = cvss['attackVector']
                    attack_complexity = cvss['attackComplexity']
                    privileges = cvss['privilegesRequired']
                    user_interaction = cvss['userInteraction']
                    scope = cvss['scope']
                    conf_impact = cvss['confidentialityImpact']
                    integ_impact = cvss['integrityImpact']
                    avail_impact = cvss['availabilityImpact']
                except KeyError:
                    pass

                # Vendor/Product info
                products = set()
                try:
                    vendor_data = item['cve']['affects']['vendor']['vendor_data']
                    for vendor in vendor_data:
                        vendor_name = vendor['vendor_name']
                        for product in vendor['product']['product_data']:
                            product_name = product['product_name']
                            products.add(f"{vendor_name}:{product_name}")
                except KeyError:
                    pass

                writer.writerow({
                    'CVE_ID': cve_id,
                    'Published_Date': published,
                    'Last_Modified_Date': modified,
                    #'Description': description,
                    'CWE_ID': cwe,
                    'CVSSv3_Base_Score': cvss_score,
                    'CVSSv3_Severity': severity,
                    'Attack_Vector': attack_vector,
                    'Attack_Complexity': attack_complexity,
                    'Privileges_Required': privileges,
                    'User_Interaction': user_interaction,
                    'Scope': scope,
                    'Confidentiality_Impact': conf_impact,
                    'Integrity_Impact': integ_impact,
                    'Availability_Impact': avail_impact,
                    #'Vendors_Products': "; ".join(products)
                })
