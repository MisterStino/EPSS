import os
import glob
import gzip
import shutil
import concurrent.futures
import tempfile
from logging import getLogger
import logging      

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)

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

def decompress_all_files_concurrently(raw_folder='data/epss/raw', output_folder='data/epss/uncompressed'):
    """
    Decompress all .csv.gz files in the raw_folder concurrently,
    saving the decompressed files in output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    else:
        return f"Output folder {output_folder} already exists. skipping decompression."
    gz_files = glob.glob(os.path.join(raw_folder, '*.csv.gz'))
    
    # Using ProcessPoolExecutor for parallel decompression.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all decompression tasks concurrently.
        futures = {executor.submit(decompress_file, gz_file, output_folder): gz_file for gz_file in gz_files}
        
        # As tasks complete, print their results.
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)

    #also remove the metadata line from the csvs
    remove_first_line_from_csvs(output_folder)


def remove_first_line_from_csvs(folder):
    """
    Removes the first line from each CSV file in the given folder.
    Overwrites the existing file so that the rest of the rows remain.
    """
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    logging.info(f"Found {len(csv_files)} CSV files to process. starting removing metadata lines...")
    for csv_file in csv_files:
        logging.info(f"Processing file: {csv_file}")
        # Read and skip the first line
        with open(csv_file, 'r', encoding='utf-8') as original:
            # Use a temp file so we don't corrupt the original if something goes wrong
            with tempfile.NamedTemporaryFile('w', delete=False, dir=folder) as tmp:
                tmp_filename = tmp.name

                # skip the first line
                first_line = original.readline()

                # now copy the rest
                for line in original:
                    tmp.write(line)

        # Move temp file to overwrite the original
        # (On Windows, you must close the file before replacing it)
        shutil.move(tmp_filename, csv_file)

        print(f"Removed first line (metadata) from: {os.path.basename(csv_file)}")


if __name__ == "__main__":
    decompress_all_files_concurrently()
