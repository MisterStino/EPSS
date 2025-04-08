import os
import logging
import pandas as pd

from data.epss.scripts.src.gen_epss_ts import create_single_time_series_csv
from data.epss.scripts.src.gen_epss_ts_parq import create_big_parquet
from data.epss.scripts.utils import decompress_all_files_concurrently
from data.epss.scripts.src.get_epss_data import get_all_epss_data
from data.full_db.scripts.src.gen_db import generate_full_database
from data.full_db.scripts.src.gen_db_parquet import generate_full_database_parquet




def run_db_pipeline():
    """
    Orchestrates the entire database pipeline:
      1. Downloads/fetches raw EPS data.
      2. Processes the raw EPS data to create the EPS time series.
      3. Merges features (and any additional modules) to generate the final full dataset.
    
    The final output is saved to:
        data/full_db/processed/final_full_data.csv
    """
    # Setup logging with INFO level.
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s')
    logging.info("Starting the full DB pipeline...")

    # Step 1: Download/fetch raw EPS data.
    raw_eps_folder = os.path.join('data', 'epss', 'raw')
    error_file = "temp_error.json"
    logging.info("Fetching raw EPS data...")
    # This function should download/fetch EPS data and store it in raw_eps_folder.
    get_all_epss_data(raw_folder=raw_eps_folder, error_file=error_file)
    logging.info("Raw EPS data fetched successfully.")

    # Step 2: Process raw EPS data to create a time series.

    logging.info("decompressing all files...")
    decompress_all_files_concurrently()
    
    logging.info("EPS time series created successfully.")
    
    logging.info("Creating big parquet epss file...")
    # this is for epss specifically, sorry bad naming
    create_big_parquet()
    # Step 3: Generate the final full database by merging features.
    logging.info("Generating the final full dataset by merging features...")
    generate_full_database_parquet()
    logging.info("Final full dataset generated successfully.")

    # # Step 4: Generate base keys for the full database.
    # logging.info("Generating base keys for the full database...")
    # generate_base_keys()
    # logging.info("Base keys generated successfully.")

    # logging.info("Starting creation of small dataset...")
    # final_full_data_path = os.path.join('data', 'full_db', 'processed', 'final_full_data.csv')
    # in_range_df, out_range_df = save_cves_by_epss_range(pd.read_csv(final_full_data_path), 0.7, 1.0)
    # merged_df = sample_and_merge_cves(in_range_df, out_range_df, sample_size=5, output_dir='data/general_utils/files')

    # logging.info("generating small base keys...")
    # generate_base_keys(final_full_data_path, 'data/general_utils/files', small=True)

    # logging.info("DB pipeline completed.")

if __name__ == '__main__':
    run_db_pipeline()
