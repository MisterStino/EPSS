import pandas as pd

# Load CSVs
submission_df = pd.read_csv("reddit_data/submissions_reddit.csv", sep=";")
comments_df = pd.read_csv("reddit_data/comments_reddit.csv", sep=";")

submission_df = submission_df[['Score', 'Date', 'Post ID', 'CVE ID']]
comments_df = comments_df[['Score', 'Date', 'Post ID', 'CVE ID']]

# Combine both files (stack them)
merged = pd.concat([submission_df, comments_df], ignore_index=True)

# Remove any duplicates if needed
merged = merged.drop_duplicates()

# Save to new CSV
merged.to_csv("EPSS/processed_data/reddit_catalog.csv", sep=';', index=False)