# src/datasets/seed_dataset.py

import os
import glob
import scipy.io as sio
import numpy as np
import torch
import re
import json
import h5py
from typing import List, Dict, Tuple, Any, Callable, Optional

from src.datasets.base_dataset import BaseDataset
from src.utils.eeg_preprocessing import apply_preprocessing_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SEEDDataset(BaseDataset):
    """
    An optimized version of the SEED Dataset class designed for efficient loading
    from pre-processed HDF5 (.h5) files.
 
    Assumptions:
    - Each EEG trial has been pre-processed and saved as a separate .h5 file.
    - H5 file naming convention: 'SubjectID_YYYYMMDD_trial_TrialNum.h5'
      (e.g., '1_20131027_trial_1.h5').
    - H5 files are located in 'data_dir/seed/seed_h5/'.
    - Each .h5 file contains a single dataset named 'eeg' with shape (channels, time_points).
    - The 'label.mat' file is still located in the original directory like
      'data_dir/seed/Preprocessed_EEG/'.
    """
    
    def __init__(self,
                 data_dir: str, # This should be "data/seed/"
                 subject_ids: List[int],
                 session_ids: List[int], # Will map to the 3 sessions per subject
                 task_type: str = 'classification',
                 preprocess_config: Dict[str, Any] = None,
                 sfreq: float = 200.0, # Standard SEED sampling frequency
                 transform: Optional[Callable] = None,
                 eeg_key_prefix_map_path: Optional[str] = None):
                 
        super().__init__(transform=transform)

        self.h5_data_folder = os.path.abspath(os.path.join(data_dir, 'seed_h5')) 
        self.original_mat_folder = os.path.abspath(os.path.join(data_dir, 'Preprocessed_EEG'))

        self.subject_ids = subject_ids
        self.session_ids = session_ids
        self.sfreq = sfreq
        self.task_type = task_type
        self.preprocess_config = preprocess_config

        if self.task_type != 'classification':
            raise ValueError("SEED Dataset is primarily for classification tasks.")
        
        # Mapping original labels (-1, 0, 1) to 0-indexed classes (0, 1, 2)
        self.label_mapping = {-1: 0, 0: 1, 1: 2}
        
        # Load the global trial label sequence from label.mat
        self.global_trial_labels = self._load_global_labels()

        # self.data_samples will store tuples of (h5_file_path, mapped_label)
        self._load_data()
        
        logger.info(f"Initialized SEEDDataset with {len(self.data_samples)} samples.")
        logger.info(f"Subjects included: {subject_ids}, Sessions included: {session_ids}")

    def _load_data(self) -> None:
        """
        Implements the abstract _load_data method from BaseDataset.
        For SEEDDataset, this method collects metadata (file paths, keys, labels)
        for lazy loading in __getitem__.
        """
        self.data_samples = self._collect_data_samples()

    def _load_global_labels(self) -> np.ndarray:
        """
        Loads the fixed sequence of trial labels from 'label.mat'.
        This file provides the ground truth labels for the 15 trials within each session.
        """
        label_file_path = os.path.join(self.original_mat_folder, 'label.mat')
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"label.mat not found at {label_file_path}. "
                                    f"Ensure '{self.original_mat_folder}' exists and contains 'label.mat'.")
 
        try:
            mat_labels = sio.loadmat(label_file_path)
            labels_array = mat_labels['label'].squeeze() # Squeeze to remove singleton dimensions
            if labels_array.shape != (15,):
                 logger.warning(f"Expected 15 labels in label.mat, found {labels_array.shape[0]}. "
                                f"This mismatch may lead to incorrect label assignments.")
            return labels_array.astype(int)
        except Exception as e:
            logger.error(f"Error loading global labels from {label_file_path}: {e}")
            raise

    def _collect_data_samples(self) -> List[Tuple[str, int]]:
        """
        Collects data samples from the .h5 file directory based on subject and session.
        This method is simplified as it directly handles individual trial files.
        """
        all_samples: List[Tuple[str, int]] = []
        if not os.path.isdir(self.h5_data_folder):
            raise NotADirectoryError(
                f"H5 data directory not found at '{self.h5_data_folder}'. "
                f"Please ensure you have run the data conversion script "
                f"(e.g., `scripts/convert_seed_to_h5.py`) first."
            )
            
        # 1. Group and sort all session files by subject
        parsed_files = []
        all_h5_files = glob.glob(os.path.join(self.h5_data_folder, '*.h5'))
        
        # Temporarily parse filenames to extract subject and date for sorting
        for h5_path in all_h5_files:
            filename = os.path.basename(h5_path)
            match = re.match(r'(\d+)_(\d{8})_trial_(\d+)\.h5', filename)
            if match:
                sub_id, session_date, trial_num = map(int, match.groups())
                # Only consider files for subjects specified in self.subject_ids
                if sub_id in self.subject_ids:
                    parsed_files.append({'path': h5_path, 'sub': sub_id, 'date': session_date, 'trial': trial_num})
 
        # Group by subject to find all session dates for each subject
        subject_date_map: Dict[int, List[int]] = {}
        for p_file in parsed_files:
            sub = p_file['sub']
            if sub not in subject_date_map:
                subject_date_map[sub] = []
            if p_file['date'] not in subject_date_map[sub]: # Add unique dates
                subject_date_map[sub].append(p_file['date'])
 
        # 2. Iterate through filtered subjects and sessions
        for sub_id in sorted(self.subject_ids):
            if sub_id not in subject_date_map:
                # This subject might not have any H5 files in the directory
                continue
 
            # Get all session dates for this subject and sort them to determine
            # the logical session index (1, 2, 3) (SEED has 3 sessions per subject)
            sorted_dates = sorted(subject_date_map[sub_id])
            
            for sess_idx_logical, session_date in enumerate(sorted_dates, 1): # Enumerate starting from 1
                if sess_idx_logical in self.session_ids:
                    # Find all trial files belonging to this subject and session date
                    session_trial_files = [
                        p for p in parsed_files 
                        if p['sub'] == sub_id and p['date'] == session_date
                    ]
                    # Sort trials by their number to ensure consistent order (1-15)
                    session_trial_files.sort(key=lambda x: x['trial'])
 
                    for trial_file_info in session_trial_files:
                        # Trial numbers are 1-indexed (1 to 15), convert to 0-indexed for array access
                        trial_idx_in_session = trial_file_info['trial'] - 1 
                        
                        # Fetch the original label from the global sequence based on trial index
                        original_label = self.global_trial_labels[trial_idx_in_session]
                        # Map the original label to the 0-indexed target label
                        mapped_label = self.label_mapping[original_label]
                        
                        all_samples.append((trial_file_info['path'], mapped_label))
 
        if not all_samples:
            logger.warning(f"No H5 data samples found for specified subjects {self.subject_ids} "
                           f"and sessions {self.session_ids} in '{self.h5_data_folder}'. "
                           f"This might indicate incorrect subject/session IDs or missing data.")
 
        return all_samples

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently loads and pre-processes a single EEG trial.
        Now, it reads a small .h5 file for each item, which is very fast.
 
        Args:
            idx (int): Index of the sample to retrieve.
 
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the processed
                                               EEG features and its corresponding mapped label.
        """
        h5_file_path, mapped_label = self.data_samples[idx]
 
        try:
            # Read EEG data using h5py
            with h5py.File(h5_file_path, 'r') as f:
                # Load the 'eeg' dataset and ensure float32 dtype
                eeg_data = f['eeg'][:].astype(np.float32)
 
            # Validate data dimensions: Expected (channels, time_points)
            if eeg_data.ndim != 2 or eeg_data.shape[0] != 62:
                logger.warning(f"Unexpected EEG data shape in {h5_file_path}: {eeg_data.shape}. "
                               f"Expected (62 channels, time_points). Data may be malformed or corrupted.")
            
            # Apply the pre-processing pipeline defined by preprocess_config
            # This utility function `apply_preprocessing_pipeline` is assumed to handle
            # operations like bandpass filtering, downsampling, feature extraction (e.g., DE) etc.
            processed_features = apply_preprocessing_pipeline(eeg_data, self.preprocess_config, self.sfreq)
            
            # Convert NumPy arrays to PyTorch tensors
            # .copy() is used to ensure the array is contiguous in memory before converting to tensor
            features_tensor = torch.tensor(processed_features.copy(), dtype=torch.float32)
            label_tensor = torch.tensor(mapped_label, dtype=torch.long)
 
            return features_tensor, label_tensor
 
        except Exception as e:
            logger.error(f"Error loading or processing sample {idx} from {h5_file_path}: {e}")
            raise # Re-raise the exception to propagate the error