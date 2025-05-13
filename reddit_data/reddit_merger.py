import pandas as pd

# Load CSVs
submission_df = pd.read_csv("reddit_data\submissions_reddit.csv", sep=";")
comments_df = pd.read_csv("reddit_data/comments_reddit.csv", sep=";")

# Merge on 'Post ID'
merged_df = comments_df.merge(
    submission_df,
    on="Post ID",
    how="left",
    suffixes=('_Comment', '_Submission')
)

# Reorder columns for clarity
merged_df = merged_df[[
    'Score_Submission', 'Date_Submission', 'CVE ID_Submission', 'Post ID',
    'Score_Comment', 'Date_Comment', 'Comment ID', 'CVE ID_Comment'
]]

# Save to new CSV
merged_df.to_csv("merged_reddit.csv", sep=";", index=False)

print("Merged CSV saved as 'merged_reddit.csv'")
