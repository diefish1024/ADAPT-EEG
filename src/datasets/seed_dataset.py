# src/datasets/seed_dataset.py

import os
import glob
import h5py
import numpy as np
import torch
import re
from typing import List, Dict, Tuple, Any, Callable, Optional
from collections import defaultdict
import scipy.io as sio

from src.datasets.base_dataset import BaseDataset
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SEEDDataset(BaseDataset):
    """
    Loads pre-processed data for the SEED dataset from HDF5 (.h5) files.
    - Assumes features have been pre-computed and each .h5 file is a single sample.
    - Handles both standard features (channels, features) and sequential features 
      for models like EmT (channels, sequence, features).
    """
    
    def __init__(self,
                 data_dir: str,
                 subject_ids: List[int],
                 session_ids: List[int],
                 task_config: Dict[str, Any] = None,
                 preprocess_config: Dict[str, Any] = None,
                 transform: Optional[Callable] = None):
        
        super().__init__(transform=transform)

        self.h5_data_folder = os.path.abspath(os.path.join(data_dir, 'seed_features_filtered')) 
        self.original_mat_folder = os.path.abspath(os.path.join(data_dir, 'Preprocessed_EEG'))

        self.subject_ids = subject_ids
        self.session_ids = session_ids
        self.task_type = task_config['type']
        self.preprocess_config = preprocess_config

        if self.task_type != 'classification':
            raise ValueError("SEED Dataset is primarily for classification tasks.")
        
        # Dynamically set label mapping based on the number of classes from the task config.
        num_classes = task_config.get('num_classes', 3)
        if num_classes == 2:
            # For binary classification, map negative (-1) to 0 and positive (1) to 1.
            # Neutral (0) samples will be ignored.
            self.label_mapping = {-1: 0, 1: 1}
            logger.info("Configured for BINARY classification (positive/negative).")
        elif num_classes == 3:
            # For 3-class classification, map negative (-1) to 0, neutral (0) to 1, and positive (1) to 2.
            self.label_mapping = {-1: 0, 0: 1, 1: 2}
            logger.info("Configured for 3-CLASS classification (positive/neutral/negative).")
        else:
            raise ValueError(f"Unsupported number of classes for SEED: {num_classes}. Must be 2 or 3.")

        self.global_trial_labels = self._load_global_labels()
        self._load_data()
        
        logger.info(f"Initialized SEEDDataset with {len(self.data_samples)} samples.")
        logger.info(f"Subjects: {subject_ids}, Sessions: {session_ids}")

    def _load_data(self) -> None:
        """Collects metadata (file paths and labels) for lazy loading."""
        self.data_samples = self._collect_data_samples()

    def _load_global_labels(self) -> np.ndarray:
        """Loads the fixed sequence of trial labels from 'label.mat'."""
        label_file_path = os.path.join(self.original_mat_folder, 'label.mat')
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"label.mat not found at {label_file_path}.")
        
        mat_labels = sio.loadmat(label_file_path)
        return mat_labels['label'].squeeze().astype(int)

    def _collect_data_samples(self) -> List[Tuple[str, int]]:
        """
        Collects all segment file paths. Each file is treated as a single sample.
        """
        if not os.path.isdir(self.h5_data_folder):
            raise NotADirectoryError(f"H5 data directory not found: '{self.h5_data_folder}'.")

        # This dictionary will hold all found file paths, grouped by trial
        trials_data = defaultdict(list)
        file_pattern = re.compile(r'(\d+)_(\d{8})_trial_(\d+)_seg_(\d+)\.h5')

        for h5_path in glob.glob(os.path.join(self.h5_data_folder, '*.h5')):
            match = file_pattern.match(os.path.basename(h5_path))
            if match:
                sub_id, session_date, trial_num, seg_num = map(int, match.groups())
                if sub_id in self.subject_ids:
                    # Group paths by their trial to sort them correctly later
                    trial_key = (sub_id, session_date, trial_num)
                    trials_data[trial_key].append({'path': h5_path, 'seg_num': seg_num})

        # Map session dates to session indices (1, 2, 3) for filtering
        subject_date_map = defaultdict(lambda: sorted(list(set())))
        for sub_id, session_date, _ in trials_data.keys():
            if session_date not in subject_date_map[sub_id]:
                 subject_date_map[sub_id].append(session_date)

        final_samples = []
        for (sub_id, session_date, trial_num), segments in trials_data.items():
            try:
                # Determine the session index (1, 2, or 3) from the date
                sess_idx_logical = subject_date_map[sub_id].index(session_date) + 1
            except (KeyError, ValueError):
                continue

            # Skip sessions that are not in the requested session_ids list
            if sess_idx_logical not in self.session_ids:
                continue

            # Sort segments by their segment number to ensure correct order
            segments.sort(key=lambda x: x['seg_num'])
            
            # Get the correct label for this trial
            original_label = self.global_trial_labels[trial_num - 1]
            # If the original label is not in our mapping (e.g., neutral label '0' in binary mode), skip this trial.
            if original_label not in self.label_mapping:
                continue
            
            mapped_label = self.label_mapping[original_label]

            # Each sorted path is now a single sample
            for seg_info in segments:
                final_samples.append((seg_info['path'], mapped_label))
        
        if not final_samples:
            logger.warning(f"No samples found for subjects {self.subject_ids}. Check config and data.")
        return final_samples

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads a single pre-processed H5 file as one sample.
        """
        file_path, mapped_label = self.data_samples[idx]
        
        try:
            with h5py.File(file_path, 'r') as f:
                # 'features' is the key where data is stored by precompute_features.py
                features_data = f['features'][:]
            
            features_tensor = torch.tensor(features_data, dtype=torch.float32)
            label_tensor = torch.tensor(mapped_label, dtype=torch.long)

            # The transform can be used for data augmentation
            if self.transform:
                features_tensor = self.transform(features_tensor)

            return features_tensor, label_tensor

        except Exception as e:
            logger.error(f"Error loading sample at index {idx} from {file_path}: {e}")
            # Return a dummy tensor to prevent crashing the training loop
            return torch.zeros((62, 1, 5)), torch.tensor(0, dtype=torch.long)
