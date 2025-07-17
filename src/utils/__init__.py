# src/utils/__init__.py

from .metrics import get_metrics_calculator
from .logger import get_logger
from .config_parser import ConfigParser
from .early_stopping import EarlyStopping
from .data_augmentations import get_eeg_augmentations
from .eeg_preprocessing import apply_preprocessing_pipeline

import random
import numpy as np
import torch

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
