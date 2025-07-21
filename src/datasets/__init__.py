# src/datasets/__init__.py
 
from .base_dataset import BaseDataset
from .seed_dataset import SEEDDataset
from typing import Dict, Any, List, Optional
 
def get_eeg_dataset(dataset_name: str, data_dir: str, subject_ids: List[int], 
                    task_type: str, preprocess_config: Dict[str, Any], sfreq: float,
                    session_ids: Optional[List[int]] = None,) -> BaseDataset:
    """
    Factory function to get an EEG dataset instance based on its name.
    """
    if dataset_name.lower() == 'seed':
        return SEEDDataset(
            data_dir=data_dir,
            subject_ids=subject_ids,
            session_ids=session_ids if session_ids is not None else [1,2,3], # Default sessions for SEED
            task_type=task_type,
            preprocess_config=preprocess_config,
            sfreq=sfreq
        )
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Choose 'SEED'.")