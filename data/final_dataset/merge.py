import pandas as pd

# Load both CSV files
cve_df = pd.read_csv('data/catalogs/CVE_catalog.csv')
exploit_db_df = pd.read_csv('data/catalogs/exploitDB_catalog.csv')
kev_df = pd.read_csv('data/catalogs/KEV_catalog.csv')
metasploit_db_df = pd.read_csv('data/catalogs/metasploit_catalog.csv')
zdi_df = pd.read_csv('data/catalogs/ZDI_catalog.csv')

# Drop duplicates based on CVE_ID
cve_df = cve_df.drop_duplicates(subset='CVE_ID')
exploit_db_df = exploit_db_df.drop_duplicates(subset='CVE_ID')
kev_df = kev_df.drop_duplicates(subset='CVE_ID')
metasploit_db_df = metasploit_db_df.drop_duplicates(subset='CVE_ID')
zdi_df = zdi_df.drop_duplicates(subset='CVE_ID')

# Prefix all columns to their respective catalogs except CVE_ID
exploit_db_df = exploit_db_df.rename(columns={
    col: f"exploitDB_{col}" for col in exploit_db_df.columns if col != "CVE_ID"
})
kev_df = kev_df.rename(columns={
    col: f"KEV_{col}" for col in kev_df.columns if col != "CVE_ID"
})
metasploit_db_df = metasploit_db_df.rename(columns={
    col: f"metasploit_{col}" for col in metasploit_db_df.columns if col != "CVE_ID"
})
zdi_df = zdi_df.rename(columns={
    col: f"ZDI_{col}" for col in zdi_df.columns if col != "CVE_ID"
})

# Merge all DataFrames one by one
merged_df = pd.merge(cve_df, exploit_db_df, on="CVE_ID", how="outer")
merged_df = pd.merge(merged_df, kev_df, on="CVE_ID", how="outer")
merged_df = pd.merge(merged_df, metasploit_db_df, on="CVE_ID", how="outer")
merged_df = pd.merge(merged_df, zdi_df, on="CVE_ID", how="outer")

# Fill missing CVSSv3_Base_Score values from ZDI catalog
merged_df['CVSSv3_Base_Score'] = merged_df['CVSSv3_Base_Score'].fillna(merged_df['ZDI_CVSS v3.0'])
merged_df.drop(columns=['ZDI_CVSS v3.0'], inplace=True)

merged_df['ZDI_Affected Vendor(s)'] = merged_df['ZDI_Affected Vendor(s)'].fillna(merged_df['KEV_vendorProject'])
merged_df.drop(columns=['KEV_vendorProject'], inplace=True)
merged_df.rename(columns={'ZDI_Affected Vendor(s)': 'Vendor'}, inplace=True)

# Save the final merged DataFrame
merged_df.to_csv('data/final_dataset/enriched_catalog.csv', index=False)



# # Inner merge to find common CVE_IDs
# common_df = pd.merge(cve_df, exploit_db_df, on='CVE_ID', how='inner')

# # Print the count of common CVE_IDs
# print(f"Number of CVE_IDs in common: {len(common_df)}")