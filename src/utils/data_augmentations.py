# src/utils/data_augmentations.py

import torch
import numpy as np
from typing import List

class EEGTransform:
    """
    Base class for EEG data augmentation.
    All EEG transformations should inherit this class and implement the __call__ method.
    """
    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Applies the transformation to EEG data.
        Args:
            eeg_data (torch.Tensor): Input EEG data compatible with the transform.
        Returns:
            torch.Tensor: Transformed EEG data.
        """
        return eeg_data

class AdditiveNoise(EEGTransform):
    """
    Add Gaussian noise to EEG data.
    """
    def __init__(self, noise_std: float = 0.05):
        """
        Initializes the AdditiveNoise transform.
        Args:
            noise_std (float): Standard deviation of the Gaussian noise.
        """
        self.noise_std = noise_std

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data (torch.Tensor): EEG data (channels, timepoints).
        Returns:
            torch.Tensor: EEG data after adding noise.
        """
        noise = torch.randn_like(eeg_data) * self.noise_std
        return eeg_data + noise

class ChannelDropout(EEGTransform):
    """
    Randomly discard some EEG channels and set the data of the corresponding channels to zero.
    """
    def __init__(self, p_dropout: float = 0.1):
        """
        Initializes the ChannelDropout transform.
        Args:
            p_dropout (float): The probability of each channel being dropped.
        """
        self.p_dropout = p_dropout

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data (torch.Tensor): Input EEG data (channels, timepoints).
        Returns:
            torch.Tensor: EEG data with some channels zeroed.
        """
        if eeg_data.dim() < 2:
            raise ValueError("EEG data must have at least 2 dimensions (channels, timepoints).")
        
        num_channels = eeg_data.shape[0]
        # Create a mask that randomly selects channels to discard
        mask = (torch.rand(num_channels) > self.p_dropout).float().to(eeg_data.device)
        # Expand mask to (channels, 1) for broadcast multiplication with (channels, timepoints)
        return eeg_data * mask.unsqueeze(1)

class TimeShift(EEGTransform):
    """
    Randomly shift the EEG signal along the time dimension.
    """
    def __init__(self, max_shift_rate: float = 0.1):
        """
        Initializes the TimeShift transform.
        Args:
            max_shift_rate (float): Maximum fractional shift relative to total timepoints.
        """
        self.max_shift_rate = max_shift_rate

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data (torch.Tensor): EEG data (channels, timepoints).
        Returns:
            torch.Tensor: EEG data after time shifting.
        """
        timepoints = eeg_data.shape[-1]
        max_shift_points = int(timepoints * self.max_shift_rate)
        
        if max_shift_points == 0: # Avoid shifting if shift points is 0
            return eeg_data
        
        shift = np.random.randint(-max_shift_points, max_shift_points + 1)
        
        if shift == 0:
            return eeg_data
        elif shift > 0: # Shift right, pad left with zeros
            return torch.cat((torch.zeros_like(eeg_data[..., :shift]), eeg_data[..., :-shift]), dim=-1)
        else: # Shift left, pad right with zeros
            return torch.cat((eeg_data[..., -shift:], torch.zeros_like(eeg_data[..., shift:])), dim=-1)
        
class Compose(EEGTransform):
    """
    Combine multiple EEG transformations together.
    """
    def __init__(self, transforms: List[EEGTransform]):
        """
        Initializes the Compose transform.
        Args:
            transforms (List[EEGTransform]): A list of EEGTransform objects to compose.
        """
        self.transforms = transforms

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data (torch.Tensor): Input EEG data.
        Returns:
            torch.Tensor: Transformed EEG data by applying all transforms in sequence.
        """
        for t in self.transforms:
            eeg_data = t(eeg_data)
        return eeg_data

def get_eeg_augmentations(config: dict) -> EEGTransform:
    """
    Returns a composition of EEG data augmentations based on config.
    Args:
        config (dict): Configuration dictionary for augmentations, e.g.,
                       {'additive_noise': {'noise_std': 0.05},
                        'channel_dropout': {'p_dropout': 0.1},
                        'time_shift': {'max_shift_rate': 0.05}}
    Returns:
        EEGTransform: A Compose object containing the configured augmentations,
                      or a base EEGTransform if no augmentations are specified.
    """
    transforms = []
    if config.get('additive_noise'):
        # Ensure a default is provided for get if key might be missing
        transforms.append(AdditiveNoise(config['additive_noise'].get('noise_std', 0.05)))
    if config.get('channel_dropout'):
        transforms.append(ChannelDropout(config['channel_dropout'].get('p_dropout', 0.1)))
    if config.get('time_shift'):
        transforms.append(TimeShift(config['time_shift'].get('max_shift_rate', 0.1)))
 
    if not transforms:
        return EEGTransform() # Return a no-op transform if no augmentations are configured
    
    return Compose(transforms)

