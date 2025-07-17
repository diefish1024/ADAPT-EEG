# scripts/seed_label_checker.py

import scipy.io as sio
import os

seed_base_dir = "../data/raw/seed/Preprocessed_EEG"
label_file_path = os.path.join(seed_base_dir, 'label.mat')

try:
    label_data = sio.loadmat(label_file_path)
    print("Keys in label.mat:", label_data.keys())
    # Often the actual array is inside a specific key, like 'label', 'labels', or 'data'
    # Try common keys
    if 'label' in label_data:
        labels_array = label_data['label']
    elif 'labels' in label_data:
        labels_array = label_data['labels']
    elif 'data' in label_data: # Sometimes it's just 'data'
        labels_array = label_data['data']
    else:
        labels_array = None
        print("Could not find common label key in label.mat. Please inspect manually.")

    if labels_array is not None:
        print("Content of labels_array (first element if multiple):")
        print(labels_array)
        print("Shape of labels_array:", labels_array.shape)
        # Expected: (1, 15) or (15, 1) or just (15,)
        
except FileNotFoundError:
    print(f"Error: {label_file_path} not found.")
except Exception as e:
    print(f"Error loading {label_file_path}: {e}")
