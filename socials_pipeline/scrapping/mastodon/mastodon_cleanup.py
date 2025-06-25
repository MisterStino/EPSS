import csv

input_file = 'data/mastodon/mastodon_raw.csv'
output_file = 'processed_data/mastodon_catalog.csv'

# Read the input file
with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    
    # Define new fieldnames (after removing and renaming columns)
    fieldnames = ['id', 'date_mastodon', 'cve_ids']
    
    rows = []
    for row in reader:
        # Remove unwanted columns and rename
        cleaned_row = {
            'id': row['id'],
            'date_mastodon': row['created_at']
        }
        
        # Handle multiple CVEs (one for each line)
        cve_list = [cve.strip() for cve in row['cve_ids'].split(',') if cve.strip()]
        for cve in cve_list:
            new_row = cleaned_row.copy()
            new_row['cve_ids'] = cve
            rows.append(new_row)

# Save file
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Processed CSV written to {output_file}")