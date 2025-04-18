import pandas as pd

# Read the CSV file
df = pd.read_csv('github_commit_dates.csv')

# Remove time portion from commit_date (keeping only YYYY-MM-DD)
df['commit_date'] = df['commit_date'].str.split('T').str[0]

# Group by cve_id and commit_date, count occurrences
result = df.groupby(['cve_id', 'commit_date']).size().reset_index(name='commit_count')

# Save the result to a new CSV file
result.to_csv('github_commit_timestamps.csv', index=False)

# Print the result
print(result)