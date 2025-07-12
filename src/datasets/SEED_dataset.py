# src/datasets/SEED_dataset.py
import torch
from torch.utils.data import Dataset
import os
import scipy.io as sio
import numpy as np
from typing import Optional, List, Dict, Tuple, Any, Callable

from .base_dataset import BaseDataset

class SEEDDataset(BaseDataset):
    """
    load and preprocess SEED_EEG data
    """
    
    def __init__(self,
                 root_dir: str,
                 subject_ids: list,
                 session_ids: list,
                 data_type: str = 'de_features', # 'processed_eeg' / 'de_features'
                 transform: Optional[Callable] = None,
                 label_mapping: Optional[dict] = None):
        """
        15 subjects * 3 experiments/subject
        subject_ids: List containing the subject indices to load (1 to 15).
        session_ids: List containing the session indices to load (1 to 3). If None, all sessions are loaded.
        transform: Transformation to apply to the data samples.
        label_mapping: Dictionary used to map original labels (-1, 0, 1) to labels used by the model (0, 1, 2).
        """
        super().__init__(transform=transform) 

        self.root_dir = root_dir
        self.subject_ids = subject_ids
        self.session_ids = session_ids if session_ids is not None else [1, 2, 3]
        self.data_type = data_type
        self.transform = transform
        self.label_mapping = label_mapping if label_mapping is not None else {-1: 0, 0: 1, 1: 2}

        self.data_samples = [] # (data, label)
        self._load_data()

    def _load_data(self):
        folder_prefix = "MD" if self.data_type == 'de_features' else "Preprocessed_EEG"
        for sub_id in self.subject_ids:
            for sess_id in self.session_ids:
                # The actual file name needs to be adjusted
                for filename in os.listdir(self.root_dir):
                    if filename.endswith('.mat'):
                        filepath = os.path.join(self.root_dir, filename)
                        data = sio.loadmat(filepath)
                        
                        raw_labels = data['labels'][0]
                        
                        for i in range(15):
                            eeg_key = f'eeg_{i+1}' # eeg_1, eeg_2, ..., eeg_15
                            if eeg_key in data:
                                eeg_data = data[eeg_key]
                                label = raw_labels[i]
                                mapped_label = self.label_mapping.get(label, label)
                                self.data_samples.append((eeg_data.astype(np.float32), mapped_label))

    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        data, label = self.data_samples[idx]
        data = torch.from_numpy(data)

        if self.transform:
            data = self.transform(data)

        return data, label