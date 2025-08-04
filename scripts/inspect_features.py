# scripts/inspect_features.py

import h5py
import numpy as np
import os
import random
import sys

# Add project root to path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config_parser import ConfigParser

def inspect_feature_file(config_path: str):
    """
    Loads a random pre-processed H5 file and prints its attributes for validation.
    The expected shape is dynamically calculated from the config file.
    """
    # Load configuration to get expected shape
    try:
        parser = ConfigParser(config_path)
        config = parser.config
        preprocess_config = config['dataset']['preprocess']
        sfreq = config['dataset']['sfreq']
    except Exception as e:
        print(f"Error: Could not load or parse config file at '{config_path}'.\nDetails: {e}")
        return

    # Calculate expected shape from config
    channels = config['model']['feature_extractor']['in_channels']
    
    # Calculate sequence length (number of sub-segments)
    segment_sec = preprocess_config['segment']['window_sec']
    sub_segment_sec = preprocess_config['feature_extraction']['sub_segment']['window_sec']
    sequence_length = int(segment_sec / sub_segment_sec)
    
    num_bands = len(preprocess_config['feature_extraction']['bands'])
    expected_shape = (channels, sequence_length, num_bands)
    
    # Get processed data directory from config
    processed_dir = os.path.join(config['dataset']['data_dir'], 'seed_features_filtered')

    if not os.path.isdir(processed_dir):
        print(f"Error: Directory '{processed_dir}' not found.")
        return

    all_files = [f for f in os.listdir(processed_dir) if f.endswith('.h5')]
    if not all_files:
        print(f"Error: No .h5 files found in '{processed_dir}'.")
        return

    # Select a random file for inspection
    random_file = random.choice(all_files)
    file_path = os.path.join(processed_dir, random_file)
    print(f"--- Inspecting file: {random_file} ---")

    try:
        with h5py.File(file_path, 'r') as f:
            if 'features' not in f:
                print("Error: Dataset 'features' not found in H5 file.")
                return

            features_data = f['features'][:]

            # Checkpoint 1: Shape
            print(f"\n[1] Shape Check")
            print(f"    - Actual Shape:   {features_data.shape}")
            print(f"    - Expected Shape: {expected_shape}")
            if features_data.shape == expected_shape:
                print("    - Result: Shape is correct ✅")
            else:
                print("    - Result: Shape is incorrect ❌")

            # Checkpoint 2: Data Type
            print(f"\n[2] Data Type Check")
            print(f"    - Data Type: {features_data.dtype}")
            if features_data.dtype == 'float32':
                 print("    - Result: Data type is correct (float32) ✅")
            else:
                 print("    - Result: Data type might be incorrect, expected float32 ⚠️")

            # Checkpoint 3: Normalization Range
            print(f"\n[3] Normalization Check (values should be between 0 and 1)")
            min_val, max_val = np.min(features_data), np.max(features_data)
            print(f"    - Min Value: {min_val:.4f}")
            print(f"    - Max Value: {max_val:.4f}")
            if 0 <= min_val <= 1.0001 and 0 <= max_val <= 1.0001: # Small tolerance for float precision
                print("    - Result: Data appears to be correctly normalized ✅")
            else:
                print("    - Result: Data is not within [0, 1] range. Normalization may have failed ❌")

            # Checkpoint 4: Data Integrity (NaN or Inf)
            print(f"\n[4] Data Integrity Check")
            has_nan = np.isnan(features_data).any()
            has_inf = np.isinf(features_data).any()
            print(f"    - Contains NaN: {has_nan}")
            print(f"    - Contains Inf: {has_inf}")
            if not has_nan and not has_inf:
                 print("    - Result: Data is clean (no NaN / Inf values) ✅")
            else:
                 print("    - Result: Data contains invalid values ❌")

    except Exception as e:
        print(f"\nAn error occurred while reading the file: {e}")

if __name__ == "__main__":
    # Path to the config file you used for preprocessing
    config_file_path = "configs/experiment_seed.yaml" 
    inspect_feature_file(config_file_path)