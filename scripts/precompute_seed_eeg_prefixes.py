# scripts/precompute_seed_prefixes.py

import os
import glob
import scipy.io as sio
import re
import json
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def precompute_seed_eeg_prefixes(eeg_data_folder: str, output_filepath: str):
    """
    Scans all SEED .mat files, extracts their dynamic EEG key prefixes (e.g., 'djc', 'xyl'),
    and saves this mapping to a JSON file.

    Args:
        eeg_data_folder (str): Path to the Preprocessed_EEG directory containing .mat files.
        output_filepath (str): Path to save the JSON file with key prefixes.
    """
    logger.info(f"Starting precomputation of EEG key prefixes for {eeg_data_folder}")
    start_time = time.time()
    
    # Get all .mat files in the Preprocessed_EEG directory, excluding 'label.mat'
    all_mat_files = sorted(glob.glob(os.path.join(eeg_data_folder, '*.mat')))
    eeg_session_files = [f for f in all_mat_files if os.path.basename(f) != 'label.mat']

    eeg_prefix_map = {}
    
    total_files = len(eeg_session_files)
    for i, filepath in enumerate(eeg_session_files):
        filename = os.path.basename(filepath)
        
        if (i + 1) % 10 == 0 or (i + 1) == total_files: # Log progress
            logger.info(f"Processing file {i + 1}/{total_files}: {filename}")

        try:
            # Use sio.whosmat() to quickly get variable names without loading data
            # It returns a list of tuples: (name, shape, dtype)
            mat_vars_info = sio.whosmat(filepath)
            
            dynamic_eeg_prefix = None
            for var_name, _, _ in mat_vars_info:
                match = re.match(r'(.+)_eeg(\d+)', var_name)
                if match:
                    dynamic_eeg_prefix = match.group(1)
                    break # Found the prefix, no need to check further
            
            if dynamic_eeg_prefix is None:
                logger.warning(f"No 'xxx_eegX' patterns found among keys in {filename}. Skipping this file.")
            else:
                eeg_prefix_map[filename] = dynamic_eeg_prefix

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue # Continue to the next file even if one fails

    # Ensure output directory exists
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the map to a JSON file
    with open(output_filepath, 'w') as f:
        json.dump(eeg_prefix_map, f, indent=4) # Use indent for human readability

    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Precomputation complete. Found prefixes for {len(eeg_prefix_map)} files. "
                f"Saved to {output_filepath}. Time taken: {duration:.2f} seconds.")

if __name__ == "__main__":
    # Define the directory where your SEED Preprocessed_EEG .mat files are located
    SEED_EEG_DATA_DIR = "D:/My Projects/ADAPT-EEG/data/raw/seed/Preprocessed_EEG"
    
    # Define the path where the precomputed prefixes JSON file will be saved
    # It's good practice to save processed metadata in a 'processed' data folder.
    OUTPUT_PROCESSING_DIR = "D:/My Projects/ADAPT-EEG/data/processed/seed/"
    OUTPUT_PREFIX_FILE = os.path.join(OUTPUT_PROCESSING_DIR, "seed_eeg_key_prefixes.json")

    precompute_seed_eeg_prefixes(SEED_EEG_DATA_DIR, OUTPUT_PREFIX_FILE)
