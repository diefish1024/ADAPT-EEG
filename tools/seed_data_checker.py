# tools/seed_data_checker.py

import scipy.io as sio
import os

seed_eeg_dir = "/data3/huangjiehang/seed/Preprocessed_EEG"
eeg_file_path = os.path.join(seed_eeg_dir, '13_20140603.mat') # Pick one file to inspect

try:
    eeg_mat_data = sio.loadmat(eeg_file_path)
    print(f"Keys in {os.path.basename(eeg_file_path)}:", eeg_mat_data.keys())

    if 'djc_eeg1' in eeg_mat_data:
        eeg_trial_1 = eeg_mat_data['djc_eeg1']
        print("\nShape of 'djc_eeg_1' data:", eeg_trial_1.shape)
        print("Data type of 'djc_eeg_1':", eeg_trial_1.dtype)
        print("First 5 time points of first channel from 'eeg_1':", eeg_trial_1[0, :5])
        # You might need to check if channels are rows or columns.
        # If shape is like (time_points, channels), you'll need to transpose it during loading.
        # E.g., (8500, 62) --> (62, 8500)
    else:
        print("Could not find 'djc_eeg1' key. Please inspect manually if other keys exist.")

except FileNotFoundError:
    print(f"Error: {eeg_file_path} not found.")
except Exception as e:
    print(f"Error loading {eeg_file_path}: {e}")

