import pandas as pd

# Load the two CSV files
cve_catalog = 'EPSS/processed_data/enriched_catalog.csv' 
mastodon = 'EPSS/processed_data/mastodon_catalog.csv' 
reddit = 'EPSS/processed_data/reddit_catalog.csv'

# Read both files into dataframes
df1 = pd.read_csv(cve_catalog)
df2 = pd.read_csv(mastodon)
df3 = pd.read_csv(reddit)

df3 = pd.read_csv('EPSS/processed_data/reddit_catalog.csv', delimiter=';')

# Merge on CVE ID columns
merged_mastodon = pd.merge(df1, df2, left_on='CVE_ID', right_on='cve_ids', how='inner')
merged_reddit = pd.merge(df1, df3, left_on='CVE_ID', right_on='CVE ID', how='inner')

# Drop the duplicate columns
merged_mastodon.drop(columns=['cve_ids'], inplace=True)
merged_reddit.drop(columns=['CVE ID'], inplace=True)

# Save the merged result to a new CSV
merged_mastodon.to_csv('EPSS/features/merged_mastodon.csv', index=False)
merged_reddit.to_csv('EPSS/features/merged_reddit.csv', index=False)

print("Merge completed. Output saved as 'EPSS/features/merged_mastodon.csv' and 'EPSS/features/merged_reddit.csv'.")