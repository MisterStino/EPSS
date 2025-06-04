import zstandard
import os
import json
import csv
import re
import traceback
from datetime import datetime
import logging.handlers

# Path to input file or folder
input_file = r"W:\reddit\comments"
# Path to output file or folder
output_file = r"W:\reddit\scraped\comments_scraped"
# Output format
output_format = "csv"

# Regex to match CVE IDs (e.g., CVE-2024-12345)
cve_regex = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

# Date range filter
from_date = datetime.strptime("2005-01-01", "%Y-%m-%d")
to_date = datetime.strptime("2030-12-31", "%Y-%m-%d")

# Logging setup
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)

if not os.path.exists("logs"):
    os.makedirs("logs")

log_file_handler = logging.handlers.RotatingFileHandler(
    os.path.join("logs", "bot.log"), maxBytes=1024 * 1024 * 16, backupCount=5
)
log_file_handler.setFormatter(log_formatter)
log.addHandler(log_file_handler)


def extract_ids(obj, is_submission):
    """Extracts Post ID and Comment ID from the Reddit object."""
    if is_submission:
        return obj['id'], ""  # Post ID, empty Comment ID
    return obj['link_id'][3:], obj['id']  # Remove "t3_" prefix from link_id



def write_line_csv(writer, obj, is_submission, cve_id):
    """Writes the extracted data into the CSV file."""
    post_id, comment_id = extract_ids(obj, is_submission)
    
    # Fix the nested f-string issue by separating the logic
    default_permalink = f"/r/{obj['subreddit']}/comments/{post_id}/_/{comment_id}"
    comment_url = f"https://www.reddit.com{obj.get('permalink', default_permalink)}"
    
    writer.writerow([
        str(obj['score']),
        datetime.utcfromtimestamp(int(obj['created_utc'])).strftime("%Y-%m-%d"),
        obj.get('title', obj.get('body', '')),
        f"u/{obj['author']}",
        comment_url,
        post_id,
        comment_id,
        cve_id
    ])


def read_lines_zst(file_name):
    """Reads and decompresses .zst files."""
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = reader.read(2**27).decode(errors='ignore')
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip()
            buffer = lines[-1]
        reader.close()


def process_file(input_file, output_file, output_format, cve_regex, from_date, to_date):
    """Processes a file, filtering only relevant comments/posts containing CVE IDs."""
    output_path = f"{output_file}.{output_format}"
    is_submission = "submission" in input_file.lower()
    log.info(f"Processing {input_file} -> {output_path} (Is submission: {is_submission})")

    writer = None
    if output_format == "csv":
        handle = open(output_path, 'w', encoding='UTF-8', newline='')
        writer = csv.writer(handle)
        
        # CSV Header
        writer.writerow(["Score", "Date", "Content", "Author", "URL", "Post ID", "Comment ID", "CVE ID"])
    else:
        log.error(f"Unsupported output format {output_format}")
        return

    total_lines, matched_lines, bad_lines = 0, 0, 0
    for line in read_lines_zst(input_file):
        total_lines += 1
        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(int(obj['created_utc']))

            if from_date <= created <= to_date:
                content = obj.get('title', '') + " " + obj.get('body', '')  # Combine title & body if available
                match = cve_regex.search(content)  # Search for CVE ID

                if match:  # Only process if a CVE ID is found
                    matched_lines += 1
                    write_line_csv(writer, obj, is_submission, match.group(0))
        except (KeyError, json.JSONDecodeError) as err:
            bad_lines += 1
            log.warning(f"Error processing line: {err}")

    handle.close()
    log.info(f"Completed: Total: {total_lines}, Matched: {matched_lines}, Errors: {bad_lines}")


if __name__ == "__main__":
    input_files = []
    if os.path.isdir(input_file):
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        for file in os.listdir(input_file):
            if file.endswith(".zst"):
                input_name = os.path.splitext(os.path.basename(file))[0]
                input_files.append((os.path.join(input_file, file), os.path.join(output_file, input_name)))
    else:
        input_files.append((input_file, output_file))

    log.info(f"Processing {len(input_files)} files")
    for file_in, file_out in input_files:
        try:
            process_file(file_in, file_out, output_format, cve_regex, from_date, to_date)
        except Exception as err:
            log.warning(f"Error processing {file_in}: {err}")
            log.warning(traceback.format_exc())
