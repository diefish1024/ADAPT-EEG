# src/datasets/base_dataset.py
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, List
import numpy as np

class BaseDataset(Dataset, ABC):
    """
    Abstract Base Class: Base class for all EEG datasets.

    This class defines a common interface for data loading, storage, and retrieval,
    forcing subclasses to implement specific data loading logic while providing
    generic __len__ and __getitem__ methods.
    """
    def __init__(self, transform: Optional[Callable] = None):
        """
        Initializes the BaseDataset.

        Args:
            transform: An optional transform function to be applied to data samples
                       when retrieved via __getitem__.
        """
        self.transform = transform
        self.data_samples: List[Tuple[np.ndarray, Any]] = [] # Stores (data_sample, label) tuples

    @abstractmethod
    def _load_data(self) -> None:
        """
        Abstract method: To be implemented by subclasses for loading data into self.data_samples.

        This method should handle dataset-specific file parsing, preprocessing,
        and populating self.data_samples with (data, label) pairs.
        """
        pass

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
        Retrieves a data sample and its corresponding label given an index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the data tensor and its label.
        """
        data, label = self.data_samples[idx]
        
        # Assume input data is a NumPy array; convert it to a PyTorch Tensor.
        # Ensure data type is float32, which is commonly used in deep learning models.
        data_tensor = torch.from_numpy(data).float()

        if self.transform:
            data_tensor = self.transform(data_tensor)

        return data_tensor, label

