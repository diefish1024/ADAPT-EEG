# src/utils/data_augmentations.py

import torch
import numpy as np

class EEGTransform:
    """
    EEG 数据增强的基类。
    所有 EEG 转换应继承此类并实现 __call__ 方法。
    """
    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement __call__ method.")

class AdditiveNoise(EEGTransform):
    """
    向 EEG 数据添加高斯噪声。
    """
    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data (torch.Tensor): 输入 EEG 数据，形状通常为 (channels, timepoints) 或 (batch, channels, timepoints)。
                                     这里假设 (channels, timepoints)。
        Returns:
            torch.Tensor: 添加噪声后的 EEG 数据。
        """
        noise = torch.randn_like(eeg_data) * self.noise_std
        return eeg_data + noise

class ChannelDropout(EEGTransform):
    """
    随机丢弃部分 EEG 通道，将对应通道的数据置零。
    """
    def __init__(self, p_dropout: float = 0.1):
        """
        Args:
            p_dropout (float): 每个通道被丢弃的概率。
        """
        self.p_dropout = p_dropout

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_data (torch.Tensor): 输入 EEG 数据，形状 (channels, timepoints)。
        Returns:
            torch.Tensor: 部分通道被置零的 EEG 数据。
        """
        if eeg_data.dim() < 2:
            raise ValueError("EEG data must have at least 2 dimensions (channels, timepoints).")
        
        num_channels = eeg_data.shape[0]
        # 创建一个掩码，随机选择要丢弃的通道
        mask = (torch.rand(num_channels) > self.p_dropout).float().to(eeg_data.device)
        # 将掩码扩展到 (channels, 1) 以便与 (channels, timepoints) 进行广播相乘
        return eeg_data * mask.unsqueeze(1)

class TimeShift(EEGTransform):
    """
    对 EEG 信号沿时间维度进行随机平移。
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
    将多个 EEG 转换组合在一起。
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, eeg_data: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            eeg_data = t(eeg_data)
        return eeg_data
