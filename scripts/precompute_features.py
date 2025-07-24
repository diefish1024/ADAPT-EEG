# scripts/precompute_features.py

import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
import sys
import shutil

sys.path.append(os.getcwd()) 
from src.utils.eeg_preprocessing import apply_preprocessing_pipeline 
from src.utils.config_parser import ConfigParser

SOURCE_H5_DIR = "/data/huangjiehang/seed/seed_h5"
DEST_FEATURES_DIR = "/data/huangjiehang/seed/seed_features_filtered" 
CONFIG_PATH = "configs/experiment_seed.yaml"

def main():
    print("Starting computing features...")
    if os.path.exists(DEST_FEATURES_DIR):
        print(f"Destination directory '{DEST_FEATURES_DIR}' already exists. Clearing it...")
        shutil.rmtree(DEST_FEATURES_DIR)
        print("Directory cleared.")

    os.makedirs(DEST_FEATURES_DIR)
    print(f"Created clean destination directory: '{DEST_FEATURES_DIR}'")

    config_parser = ConfigParser(CONFIG_PATH)
    config = config_parser.config
    preprocess_config = config['dataset']['preprocess']
    sfreq = config['dataset']['sfreq']

    raw_files = glob.glob(os.path.join(SOURCE_H5_DIR, '*.h5'))
    
    for file_path in tqdm(raw_files, desc="Preprocessing files"):
        try:
            # Read raw data
            with h5py.File(file_path, 'r') as f:
                raw_eeg_data = f['eeg'][:]

            processed_segments = apply_preprocessing_pipeline(
                raw_eeg_data, 
                preprocess_config, 
                sfreq
            )

            # Get the base filename without extension
            base_filename = os.path.basename(file_path).replace('.h5', '')

            # Loop through the list and save each segment as a separate file
            for i, segment_features in enumerate(processed_segments):
                # Create a unique name for each segment
                segment_filename = f"{base_filename}_seg_{i}.h5"
                dest_path = os.path.join(DEST_FEATURES_DIR, segment_filename)

                with h5py.File(dest_path, 'w') as f_out:
                    # Store the final feature array into a dataset named 'features'
                    f_out.create_dataset('features', data=segment_features)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print("All feature pre-calculation is completed!")
    print(f"The processed features are stored in: {DEST_FEATURES_DIR}")

if __name__ == '__main__':
    main()