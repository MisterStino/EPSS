import os
import logging


from data.epss.scripts.src.gen_epss_ts import create_single_time_series_csv
from data.epss.scripts.src.get_epss_data import get_all_epss_data
from data.full_db.scripts.src.gen_db import generate_full_database
from data.general_utils.base_keys import generate_base_keys
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
    #get_all_epss_data(raw_folder=raw_eps_folder, error_file=error_file)
    logging.info("Raw EPS data fetched successfully.")

    # Step 2: Process raw EPS data to create a time series.
    epss_processed_folder = os.path.join('data', 'epss', 'processed')
    epss_output_filename = 'epss_processed.csv'
    logging.info("Creating EPS time series from raw data...")
    create_single_time_series_csv(
        raw_folder=raw_eps_folder,
        output_folder=epss_processed_folder,
        output_filename=epss_output_filename
    )
    logging.info("EPS time series created successfully.")

    # Step 3: Generate the final full database by merging features.
    logging.info("Generating the final full dataset by merging features...")
    generate_full_database()
    logging.info("Final full dataset generated successfully.")

    # Step 4: Generate base keys for the full database.
    logging.info("Generating base keys for the full database...")
    generate_base_keys()
    logging.info("Base keys generated successfully.")
    logging.info("DB pipeline completed.")

if __name__ == '__main__':
    run_db_pipeline()
