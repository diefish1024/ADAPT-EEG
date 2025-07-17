# src/utils/data_augmentations.py

import torch
import numpy as np

class EEGTransform:
    """
    Base class for EEG data augmentation.
    All EEG transformations should inherit this class and implement the __call__ method.
    """
    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        return eeg_data

class AdditiveNoise(EEGTransform):
    """
    Add Gaussian noise to EEG data.
    """
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data (torch.Tensor): (channels, timepoints)。
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
        Args:
            p_dropout (float): The probability of each channel being dropped.
        """
        self.p_dropout = p_dropout

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data (torch.Tensor): inputs EEG data, (channels, timepoints)。
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
        self.max_shift_rate = max_shift_rate

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        timepoints = eeg_data.shape[-1]
        max_shift_points = int(timepoints * self.max_shift_rate)
        
        if max_shift_points == 0: # Avoid shifting if shift points is 0
            return eeg_data
        
        shift = np.random.randint(-max_shift_points, max_shift_points + 1)
        
        if shift == 0:
            return eeg_data
        elif shift > 0: # Shift right, pad left
            return torch.cat((torch.zeros_like(eeg_data[..., :shift]), eeg_data[..., :-shift]), dim=-1)
        else: # Shift left, pad right
            return torch.cat((eeg_data[..., -shift:], torch.zeros_like(eeg_data[..., shift:])), dim=-1)
        
class Compose:
    """
    Combine multiple EEG transformations together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            eeg_data = t(eeg_data)
        return eeg_data

def get_eeg_augmentations(config: dict) -> EEGTransform:
    """
    Returns a composition of EEG data augmentations based on config.
    """
    transforms = []
    if config.get('random_noise', False):
        transforms.append(AdditiveNoise(config['noise_std']))
 
    if not transforms:
        return EEGTransform()
    
    return Compose(transforms)