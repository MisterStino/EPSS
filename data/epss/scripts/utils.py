import os
import glob
import gzip
import shutil
import concurrent.futures
import tempfile
import re
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decompress_file(gz_file, output_folder):
    """
    Decompress a single gzip file and save the output in the output folder.
    """
    filename = os.path.basename(gz_file)
    output_filename = filename[:-3]  # Remove the '.gz' extension
    output_filepath = os.path.join(output_folder, output_filename)
    
    try:
        with gzip.open(gz_file, 'rb') as f_in:
            with open(output_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return f"Decompressed: {filename}"
    except Exception as e:
        return f"Error decompressing {filename}: {e}"

def standardize_csv_file(csv_file, output_folder):
    """
    Process a CSV file by either removing the first line (for newer versions, date >= 2022-02-04)
    or leaving the file unchanged if the file is from version 1 (date < 2022-02-04). The output file
    is written to the specified output_folder.
    """
    # Extract date from the filename using regex (expecting format YYYY-MM-DD)
    match = re.search(r'\d{4}-\d{2}-\d{2}', csv_file)
    if match:
        file_date_str = match.group()
        try:
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format in file {csv_file}.")
            return f"Error processing {os.path.basename(csv_file)}: invalid date format"
    else:
        logger.error(f"No valid date found in filename: {csv_file}.")
        return f"Error processing {os.path.basename(csv_file)}: no date found"

    # Define the cutoff for version 1 files
    version1_cutoff = datetime.strptime("2022-02-04", "%Y-%m-%d")
    
    # Read all lines from the original CSV file
    try:
        with open(csv_file, 'r', encoding='utf-8') as original:
            lines = original.readlines()
    except Exception as e:
        logger.error(f"Error reading {os.path.basename(csv_file)}: {e}")
        return f"Error reading {os.path.basename(csv_file)}: {e}"
    
    # Process according to file version:
    # For version 1 files (date < 2022-02-04), leave the file unchanged.
    # For newer files (date >= 2022-02-04), remove the first line (assumed to be metadata).
    if file_date < version1_cutoff:
        new_lines = lines
        action = "Left unchanged"
    else:
        new_lines = lines[1:]
        action = "Removed first line"

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Prepare the output filepath in the output_folder
    output_filename = os.path.basename(csv_file)
    final_output_filepath = os.path.join(output_folder, output_filename)
    
    # Write to a temporary file in the output folder and then move it to the final destination
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile('w', delete=False, dir=output_folder, encoding='utf-8') as tmp:
            temp_file_path = tmp.name
            tmp.writelines(new_lines)
        shutil.move(temp_file_path, final_output_filepath)
        logger.info(f"{action} for file: {os.path.basename(csv_file)}")
        return f"{action} for file: {os.path.basename(csv_file)}"
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(csv_file)}: {e}")
        return f"Error processing {os.path.basename(csv_file)}: {e}"
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def standardize_epss_files_concurrently(input_folder, output_folder):
    """
    Process all CSV files in the input_folder concurrently, applying either header addition 
    or metadata removal based on the file's date. The standardized files are written to the output_folder.
    """
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files to process in {input_folder}. Starting processing...")

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {
            executor.submit(standardize_csv_file, csv_file, output_folder): csv_file
            for csv_file in csv_files
        }
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            print(result)
            results.append(result)
    return results

def decompress_all_files_concurrently(raw_folder='data/epss/raw', output_folder='data/epss/uncompressed'):
    """
    Decompress all .csv.gz files in the raw_folder concurrently,
    saving the decompressed files in output_folder.
    After decompression, we for version 1 add column headers and vor version 2 remove the first line.
    """
    logger.info('Starting decompression process...')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    gz_files = glob.glob(os.path.join(raw_folder, '*.csv.gz'))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(decompress_file, gz_file, output_folder): gz_file for gz_file in gz_files}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)
            

if __name__ == "__main__":
    decompress_all_files_concurrently()
