# src/datasets/seed_dataset.py

import os
import glob
import scipy.io as sio
import numpy as np
import torch
import re
import json
from typing import List, Dict, Tuple, Any, Callable, Optional

from src.datasets.base_dataset import BaseDataset
from src.utils.eeg_preprocessing import apply_preprocessing_pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SEEDDataset(BaseDataset):
    """
    Dataset for the SEED EEG dataset.
    Assumes:
    - EEG data files are named 'SubjectID_YYYYMMDD.mat' (e.g., '1_20131027.mat')
      and located in 'data_dir/Preprocessed_EEG/'.
    - Each .mat file contains 15 trials with keys 'xxx_eeg1', 'xxx_eeg2', ..., 'xxx_eeg15'.
    - A 'label.mat' file located in 'data_dir/Preprocessed_EEG/'.
      contains the fixed trial labels: [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1].
    """
    
    def __init__(self,
                 data_dir: str, # This should be "data/raw/seed/"
                 subject_ids: List[int],
                 session_ids: List[int], # Will map to the 3 sessions per subject
                 task_type: str = 'classification',
                 preprocess_config: Dict[str, Any] = None,
                 sfreq: float = 200.0, # Standard SEED sampling frequency
                 transform: Optional[Callable] = None,
                 eeg_key_prefix_map_path: Optional[str] = None):
                 
        super().__init__(transform=transform)

        self.base_data_dir = data_dir
        self.eeg_data_folder = os.path.join(self.base_data_dir, 'Preprocessed_EEG')
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

        self.eeg_key_prefix_map: Dict[str, str] = {}
        if eeg_key_prefix_map_path and os.path.exists(eeg_key_prefix_map_path):
            try:
                with open(eeg_key_prefix_map_path, 'r') as f:
                    self.eeg_key_prefix_map = json.load(f)
                logger.info(f"Loaded EEG key prefixes from {eeg_key_prefix_map_path}")
            except Exception as e:
                logger.warning("SEEDDataset: Proceeding without precomputed EEG key prefixes. Dynamic detection will be used, which may be slower during dataset loading.")
                logger.error(f"Failed to load EEG key prefixes from {eeg_key_prefix_map_path}: {e}")
        else:
            logger.warning("No EEG key prefix map provided or found at specified path. Dynamic detection will be used, which may be slower during dataset loading.")

        # data_samples will store tuples: (file_path_to_eeg_mat, eeg_key_in_mat, mapped_label)
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
        Loads the fixed trial label sequence from 'label.mat'.
        Expected: [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        """
        label_file_path = os.path.join(self.eeg_data_folder, 'label.mat')
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"label.mat not found at {os.path.join(self.eeg_data_folder, 'label.mat')}")

        try:
            mat_labels = sio.loadmat(label_file_path)
            if 'label' in mat_labels:
                labels_array = mat_labels['label']
            else:
                raise KeyError(f"Could not find a label key in '{label_file_path}'. "
                               f"Available keys: {mat_labels.keys()}")
            
            # Ensure it's a 1D array of 15 elements
            if labels_array.ndim > 1:
                labels_array = labels_array.squeeze()
            if labels_array.shape != (15,):
                logger.warning(f"Expected label.mat to contain 15 labels, but found shape {labels_array.shape}. "
                               "This might lead to indexing issues if the trial count is different.")
            return labels_array.astype(int)
        except Exception as e:
            logger.error(f"Error loading global labels from {label_file_path}: {e}")
            raise e

    def _extract_subject_session(self, filename: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Extracts subject ID and session date string from filename 'SubjectID_YYYYMMDD.mat'.
        """
        match = re.match(r'(\d+)_(\d{8})\.mat', filename)
        if match:
            subject_id = int(match.group(1))
            session_date = match.group(2) # YYYYMMDD string
            return subject_id, session_date
        return None, None
    
    def _get_eeg_prefix(self, file_path: str, filename: str) -> Optional[str]:
        """
        Retrieves the EEG key prefix for a given file.
        Prioritizes the precomputed map, falls back to dynamic detection if needed.
        """
        # Try to get from precomputed map
        if filename in self.eeg_key_prefix_map:
            return self.eeg_key_prefix_map[filename]
        
        # If not in map (or map not provided), perform dynamic detection
        logger.warning(f"EEG key prefix for '{filename}' not found in precomputed map. Dynamically detecting prefix (this may be slow).")
        try:
            mat_vars_info = sio.whosmat(file_path) # Use whosmat for efficiency (still slower than lookup)
            for var_name, _, _ in mat_vars_info:
                match = re.match(r'(.+)_eeg(\d+)', var_name)
                if match:
                    # If dynamically found, store it in the map for potential future use (within this run)
                    self.eeg_key_prefix_map[filename] = match.group(1)
                    return match.group(1)
            logger.warning(f"No 'xxx_eegX' patterns found among keys in {filename} during dynamic detection.")
            return None
        except Exception as e:
            logger.error(f"Error dynamically determining prefix for {filename}: {e}")
            return None

    def _collect_data_samples(self) -> List[Tuple[str, str, int]]:
        """
        Collects all (file_path_to_eeg_mat, eeg_key_in_mat, mapped_label) tuples.
        Uses the precomputed or dynamically determined EEG key prefix for each .mat file.
        """
        all_samples = []
        
        all_mat_files = sorted(glob.glob(os.path.join(self.eeg_data_folder, '*.mat')))
        
        eeg_session_files = [f for f in all_mat_files if os.path.basename(f) != 'label.mat']
 
        subject_sessions_map: Dict[int, List[Tuple[str, str]]] = {}
        for filepath in eeg_session_files:
            filename = os.path.basename(filepath)
            sub_id, session_date_str = self._extract_subject_session(filename)
            
            if sub_id is None or session_date_str is None:
                continue # Skip warning for cleaner log where parsing fails for label.mat/readme.txt
            
            if sub_id in self.subject_ids:
                if sub_id not in subject_sessions_map:
                    subject_sessions_map[sub_id] = []
                subject_sessions_map[sub_id].append((filepath, session_date_str))
        
        for sub_id in sorted(subject_sessions_map.keys()):
            sessions_for_subject = sorted(subject_sessions_map[sub_id], key=lambda x: x[1])
            
            for sess_idx_logical, (file_path, _) in enumerate(sessions_for_subject):
                if (sess_idx_logical + 1) in self.session_ids:
                    filename = os.path.basename(file_path)
                    dynamic_eeg_prefix = self._get_eeg_prefix(file_path, filename)
 
                    if dynamic_eeg_prefix is None:
                        logger.error(f"SEEDDataset._collect_data_samples: Could not determine EEG key prefix for {filename}. Skipping this session.")
                        continue
                    
                    for trial_idx in range(15):
                        eeg_key = f'{dynamic_eeg_prefix}_eeg{trial_idx + 1}'
 
                        original_label = self.global_trial_labels[trial_idx]
                        mapped_label = self.label_mapping.get(original_label, original_label)
                        all_samples.append((file_path, eeg_key, mapped_label))
 
        if not all_samples:
            logger.warning(f"No SEED data samples found for subjects {self.subject_ids}, sessions {self.session_ids} in {self.eeg_data_folder}. Check paths and subject/session IDs.")
 
        return all_samples

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a single EEG sample (trial).
        """
        file_path, eeg_key, mapped_label = self.data_samples[idx]
 
        try:
            mat_data = sio.loadmat(file_path)
            eeg_data = mat_data[eeg_key].astype(np.float32)
 
            # The data inspection shows (62, 47001), which is (channels, time_points).
            # This is the desired format for apply_preprocessing_pipeline, so no transpose is needed.
            if eeg_data.ndim != 2 or eeg_data.shape[0] != 62: # Sanity check for expected shape
                logger.warning(f"Unexpected EEG data shape for {eeg_key} in {os.path.basename(file_path)}: {eeg_data.shape}. Expected (62, time_points).")
 
            # Apply preprocessing pipeline (filtering, feature extraction, normalization)
            processed_features = apply_preprocessing_pipeline(eeg_data, self.preprocess_config, self.sfreq)
            
            features_tensor = torch.tensor(processed_features.copy(), dtype=torch.float32)
            label_tensor = torch.tensor(mapped_label, dtype=torch.long) # Labels are long for classification
 
            return features_tensor, label_tensor
 
        except KeyError as ke:
            logger.error(f"KeyError: {eeg_key} not found in {file_path}. Error: {ke}")
            raise
        except Exception as e:
            logger.error(f"Error loading or processing sample {idx} ({file_path}, {eeg_key}): {e}")
            raise