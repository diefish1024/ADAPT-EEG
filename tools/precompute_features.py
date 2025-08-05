# tools/precompute_features.py

import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
import sys
import shutil
import argparse

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.eeg_preprocessing import apply_preprocessing_pipeline 
from src.utils.config_parser import ConfigParser

def main(config_path: str):
    print("Starting feature pre-computation...")
    
    config_parser = ConfigParser(config_path)
    config = config_parser.config
    
    data_dir = config['dataset']['data_dir']
    preprocess_config = config['dataset']['preprocess']
    sfreq = config['dataset']['sfreq']
    
    # Define source and destination directories based on config
    source_h5_dir = os.path.join(data_dir, "seed_h5")
    dest_features_dir = os.path.join(data_dir, "seed_features_filtered")

    if os.path.exists(dest_features_dir):
        print(f"Destination directory '{dest_features_dir}' already exists. Clearing it...")
        shutil.rmtree(dest_features_dir)
        print("Directory cleared.")

    os.makedirs(dest_features_dir)
    print(f"Created clean destination directory: '{dest_features_dir}'")

    raw_files = glob.glob(os.path.join(source_h5_dir, '*.h5'))
    
    for file_path in tqdm(raw_files, desc="Preprocessing files"):
        try:
            with h5py.File(file_path, 'r') as f:
                raw_eeg_data = f['eeg'][:]

            processed_segments = apply_preprocessing_pipeline(
                raw_eeg_data, 
                preprocess_config, 
                sfreq
            )

            base_filename = os.path.basename(file_path).replace('.h5', '')
            for i, segment_features in enumerate(processed_segments):
                segment_filename = f"{base_filename}_seg_{i}.h5"
                dest_path = os.path.join(dest_features_dir, segment_filename)
                with h5py.File(dest_path, 'w') as f_out:
                    f_out.create_dataset('features', data=segment_features)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    print("All feature pre-computation is completed!")
    print(f"The processed features are stored in: {dest_features_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EEG Feature Pre-computation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the experiment configuration YAML file.")
    args = parser.parse_args()
    main(args.config)
