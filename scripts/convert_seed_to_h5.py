import os
import glob
import re
import h5py
import scipy.io as sio
import numpy as np
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Path Configurations ---
# Directory containing source .mat files
SOURCE_MAT_DIR = "/data/huangjiehang/seed/Preprocessed_EEG"
# Directory to store target .h5 files (script will create if not exists)
DEST_H5_DIR = "/data/huangjiehang/seed/seed_h5"

def get_eeg_prefix(mat_data: dict) -> str:
    """
    Dynamically detect the variable name prefix for EEG trials from loaded .mat file data.
    E.g., if variables are 'sub01_eeg1', 'sub01_eeg2', returns 'sub01'.
    """
    for var_name in mat_data.keys():
        match = re.match(r'(.+)_eeg(\d+)', var_name)
        if match:
            return match.group(1)
    raise ValueError("No variable name matching 'xxx_eegN' format found in the .mat file.")

def convert_mat_to_h5():
    """
    Main conversion function.
    Reads .mat files from SOURCE_MAT_DIR, converts each EEG trial into a separate .h5 file,
    and saves it in DEST_H5_DIR.
    Filename format: {subjectID}_{sessionDate}_trial_{trial_index}.h5
    """
    if not os.path.isdir(SOURCE_MAT_DIR):
        logging.error(f"Source directory does not exist: {SOURCE_MAT_DIR}")
        return

    os.makedirs(DEST_H5_DIR, exist_ok=True)
    logging.info(f"Output directory created: {DEST_H5_DIR}")

    mat_files = glob.glob(os.path.join(SOURCE_MAT_DIR, '*.mat'))
    # Exclude label.mat file as it contains labels, not EEG data to be converted directly.
    mat_files = [f for f in mat_files if os.path.basename(f) != 'label.mat']

    if not mat_files:
        logging.warning(f"No .mat files found in {SOURCE_MAT_DIR}.")
        return

    logging.info(f"Found {len(mat_files)} session .mat files. Starting conversion now...")

    for mat_path in tqdm(mat_files, desc="Converting MAT to H5"):
        try:
            filename = os.path.basename(mat_path)
            # Extract the base name from the filename, e.g., '1_20131027'
            base_name = filename.replace('.mat', '')

            # Load the entire .mat file (acceptable for preprocessing stage to load all at once)
            mat_data = sio.loadmat(mat_path)
            prefix = get_eeg_prefix(mat_data)

            # SEED dataset typically has 15 trials per file
            for i in range(1, 16): # Trials are 1-indexed in .mat files
                trial_key = f'{prefix}_eeg{i}'
                
                if trial_key not in mat_data:
                    logging.warning(f"Key: '{trial_key}' not found in {filename}, skipping.")
                    continue
                
                eeg_data = mat_data[trial_key].astype(np.float32)

                # Create new .h5 filename
                h5_filename = f"{base_name}_trial_{i}.h5"
                h5_filepath = os.path.join(DEST_H5_DIR, h5_filename)

                # Use h5py to save data
                with h5py.File(h5_filepath, 'w') as f:
                    # Save EEG data in a dataset named 'eeg'
                    f.create_dataset('eeg', data=eeg_data)
        
        except Exception as e:
            logging.error(f"Error processing file {mat_path}: {e}")

    logging.info("All files converted successfully! ")
    logging.info(f"Converted .h5 files saved to: {DEST_H5_DIR}")


if __name__ == '__main__':
    convert_mat_to_h5()

