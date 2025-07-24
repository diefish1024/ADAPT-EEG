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
from collections import defaultdict

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
                 data_dir: str,
                 subject_ids: List[int],
                 session_ids: List[int], # Will map to the 3 sessions per subject
                 task_type: str = 'classification',
                 preprocess_config: Dict[str, Any] = None,
                 sfreq: float = 200.0, # Standard SEED sampling frequency
                 transform: Optional[Callable] = None):
                 
        super().__init__(transform=transform)

        self.h5_data_folder = os.path.abspath(os.path.join(data_dir, 'seed_features_filtered')) 
        self.original_mat_folder = os.path.abspath(os.path.join(data_dir, 'Preprocessed_EEG'))

        self.subject_ids = subject_ids
        self.session_ids = session_ids
        self.sfreq = sfreq
        self.task_type = task_type
        self.preprocess_config = preprocess_config
        self.sequence_length = self.preprocess_config.get('sequence_length', 1)

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
        Collects segment file paths and groups them into sequences.
        Each sample will be a list of paths representing a sequence.
        """
        if not os.path.isdir(self.h5_data_folder):
            raise NotADirectoryError(f"H5 data directory not found at '{self.h5_data_folder}'.")

        trials_data = defaultdict(list)
        all_h5_files = glob.glob(os.path.join(self.h5_data_folder, '*.h5'))
            
        parsed_files = []
        all_h5_files = glob.glob(os.path.join(self.h5_data_folder, '*.h5'))
        
        # The regex can match both '..._trial_1.h5' and '..._trial_1_seg_0.h5'
        file_pattern = re.compile(r'(\d+)_(\d{8})_trial_(\d+)_seg_(\d+)\.h5')

        for h5_path in all_h5_files:
            filename = os.path.basename(h5_path)
            match = file_pattern.match(filename)
            
            if match:
                sub_id, session_date, trial_num, seg_num = map(int, match.groups())
                
                if sub_id in self.subject_ids:
                    trial_key = (sub_id, session_date, trial_num)
                    trials_data[trial_key].append({'path': h5_path, 'seg_num': seg_num})

        final_samples = []

        # subject_id, session_date, and trial_num to assign labels.
        subject_date_map = defaultdict(list)
        for sub_id, session_date, _ in trials_data.keys():
            if session_date not in subject_date_map[sub_id]:
                subject_date_map[sub_id].append(session_date)

        for (sub_id, session_date, trial_num), segments in trials_data.items():
            if sub_id not in subject_date_map or session_date not in subject_date_map[sub_id]:
                continue
            sess_idx_logical = subject_date_map[sub_id].index(session_date) + 1

            if sess_idx_logical not in self.session_ids:
                continue

            segments.sort(key=lambda x: x['seg_num'])
            sorted_paths = [s['path'] for s in segments]

            trial_idx_in_session = trial_num - 1
            original_label = self.global_trial_labels[trial_idx_in_session]
            mapped_label = self.label_mapping[original_label]

            num_segments = len(sorted_paths)
            for i in range(0, num_segments - self.sequence_length + 1, self.sequence_length):
                sequence_paths = sorted_paths[i : i + self.sequence_length]
                final_samples.append((sequence_paths, mapped_label))
        
        if not final_samples:
            logger.warning(f"No sequences created for subjects {self.subject_ids}. Check config and data.")

        return final_samples

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a sequence of segments, splices them together, and returns a single tensor.
        """
        sequence_paths, mapped_label = self.data_samples[idx]
        
        segment_arrays = []
        try:
            for path in sequence_paths:
                with h5py.File(path, 'r') as f:
                    segment_arrays.append(f['features'][:])
            
            spliced_features = np.concatenate(segment_arrays, axis=1)
            
            features_tensor = torch.tensor(spliced_features, dtype=torch.float32)
            label_tensor = torch.tensor(mapped_label, dtype=torch.long)

            return features_tensor, label_tensor

        except Exception as e:
            logger.error(f"Error loading or splicing sequence at index {idx} from {sequence_paths}: {e}")
            raise