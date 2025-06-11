import zstandard 
import os
import json
import sys
import csv
import re
from datetime import datetime
import logging.handlers

input_file = r"W:\reddit\submissions"
output_file = r"c:/Users/stijn/Desktop/EPSS_FRESH/data/reddit/raw_submissions"
output_format = "csv"
single_field = None
write_bad_lines = True
from_date = datetime.strptime("2005-01-01", "%Y-%m-%d")
to_date = datetime.strptime("2030-12-31", "%Y-%m-%d")

# Specify the field to search in and define the regex pattern for CVE IDs
field = "selftext"
cve_pattern = re.compile(r'cve-\d{4}-\d{4,7}', re.IGNORECASE)

# Sets up logging
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)
if not os.path.exists("logs"):
    os.makedirs("logs")
log_file_handler = logging.handlers.RotatingFileHandler(os.path.join("logs", "bot.log"), maxBytes=1024*1024*16, backupCount=5)
log_file_handler.setFormatter(log_formatter)
log.addHandler(log_file_handler)


def extract_post_id(url):
    """
    Extracts the post ID from a Reddit URL.
    Example URL: https://www.reddit.com/r/AskNetsec/comments/unb6n/_/c4wvfbo
    The post ID here is "unb6n".
    """
    match = re.search(r'/comments/([^/]+)/', url)
    if match:
        return match.group(1)
    return ""


def write_line_csv(writer, obj, is_submission, cve_id, post_id):
    """Writes a CSV row with an extra column for the extracted CVE ID and post ID."""
    output_list = []
    output_list.append(str(obj['score']))
    output_list.append(datetime.fromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d"))
    
    # For submissions, include the title; for comments, a brief excerpt or "body"
    if is_submission:
        output_list.append(obj.get('title', ""))
    else:
        output_list.append(obj.get('body', ""))
    
    output_list.append(f"u/{obj['author']}")
    
    # Build the permalink URL
    if 'permalink' in obj:
        permalink = f"https://www.reddit.com{obj['permalink']}"
    else:
        permalink = f"https://www.reddit.com/r/{obj['subreddit']}/comments/{obj['link_id'][3:]}/_/{obj['id']}"
    output_list.append(permalink)
    
    # Depending on whether it's a submission or comment, include selftext/url or body
    if is_submission:
        if obj.get('is_self', False):
            output_list.append(obj.get('selftext', ""))
        else:
            output_list.append(obj.get('url', ""))
    else:
        output_list.append(obj.get('body', ""))
    
    # Append the extracted CVE ID and Post ID columns
    output_list.append(cve_id)
    output_list.append(post_id)
    
    writer.writerow(output_list)


def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            # Read and decode chunk; ignore decode errors
            chunk = reader.read(2**27).decode(errors='ignore')
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip(), file_handle.tell()
            buffer = lines[-1]
        reader.close()


def process_file(input_file, output_file, output_format, field, from_date, to_date, single_field):
    output_path = f"{output_file}.{output_format}"
    is_submission = "submission" in input_file
    log.info(f"Processing: {input_file} -> {output_path}, Is submission: {is_submission}")

    writer = None
    if output_format == "csv":
        handle = open(output_path, 'w', encoding='UTF-8', newline='')
        writer = csv.writer(handle)
        # Write CSV headers with additional columns for CVE ID and Post ID
        csv_headers = [
            "Score", "Date", "Title/Body", "Author", "Permalink",
            "Content", "CVE ID", "Post ID"
        ]
        writer.writerow(csv_headers)
    else:
        log.error(f"Unsupported output format {output_format}")
        sys.exit()

    total_lines, matched_lines, bad_lines = 0, 0, 0
    for line, _ in read_lines_zst(input_file):
        total_lines += 1
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(int(obj['created_utc']))
            if not (from_date <= created <= to_date):
                continue

            if field in obj:
                field_value = obj[field]
                match = cve_pattern.search(field_value)
                if not match:
                    continue  # Skip if no CVE ID is found

                cve_id = match.group(0)  # Extract the first CVE ID found

                # Build the permalink URL as used in CSV output
                if 'permalink' in obj:
                    url = f"https://www.reddit.com{obj['permalink']}"
                else:
                    url = f"https://www.reddit.com/r/{obj['subreddit']}/comments/{obj['link_id'][3:]}/_/{obj['id']}"
                post_id = extract_post_id(url)

                matched_lines += 1
                if output_format == "csv":
                    write_line_csv(writer, obj, is_submission, cve_id, post_id)

        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
            log.warning(f"Error processing line: {err}")

    handle.close()
    log.info(f"Processing complete: {total_lines} total, {matched_lines} matched, {bad_lines} bad lines.")


if __name__ == "__main__":
    log.info(f"Searching for CVE IDs in field: {field}")
    log.info(f"Date range: {from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}")
    log.info(f"Output format: {output_format}")

    input_files = [(input_file, output_file)] if os.path.isfile(input_file) else [
        (os.path.join(input_file, file), os.path.join(output_file, os.path.splitext(file)[0]))
        for file in os.listdir(input_file) if file.endswith(".zst")
    ]

    log.info(f"Processing {len(input_files)} files.")
    for file_in, file_out in input_files:
        try:
            process_file(file_in, file_out, output_format, field, from_date, to_date, single_field)
        except Exception as err:
            import traceback
            log.warning(f"Error processing {file_in}: {err}")
            log.warning(traceback.format_exc())