# src/utils/eeg_preprocessing.py

import numpy as np
import scipy.signal
import logging
from typing import List

logger = logging.getLogger(__name__)

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to EEG data.
    Args:
        data (np.ndarray): EEG data, expected shape (channels, time_points).
        lowcut (float): Lower cutoff frequency.
        highcut (float): Higher cutoff frequency.
        fs (float): Sampling frequency.
        order (int): Order of the filter.
    Returns:
        np.ndarray: Filtered EEG data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.filtfilt(b, a, data, axis=-1) # Apply along the time axis
    return y

def calculate_differential_entropy(eeg_data_chunk: np.ndarray) -> np.ndarray:
    """
    Calculate differential entropy (DE) as an EEG feature.
    DE is defined as 0.5 * log(2 * pi * e * sigma^2), where sigma^2 is variance.
    Assuming the data follows a Gaussian distribution for simplicity.
    
    Args:
        eeg_data_chunk (np.ndarray): EEG data for a specific channel and time segment.
                                      Expected shape (channels, time_points) for a band.
    Returns:
        np.ndarray: Differential entropy value(s) for each channel in the chunk. Shape (channels,)
    """
    if eeg_data_chunk.ndim == 1: # if a specific channel is passed as 1D
        eeg_data_chunk = eeg_data_chunk[np.newaxis, :] # Make it (1, time_points)
    elif eeg_data_chunk.ndim == 0: # single value
        return np.array([0.0]) # Handle scalar case if it occurs

    variance = np.var(eeg_data_chunk, axis=-1) # Variance across time points for each channel
    # Avoid log of zero or negative variance for numerical stability
    variance[variance <= 1e-6] = 1e-6 
    
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    return de

def minmax_normalize(data: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalization.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data) # Handle case where all values are same
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def downsample_data(data: np.ndarray, original_sfreq: float, target_sfreq: float) -> np.ndarray:
    """
    Downsamples the data to a target sampling frequency.
    """
    if original_sfreq == target_sfreq:
        return data
    num_samples_original = data.shape[-1]
    num_samples_target = int(num_samples_original * target_sfreq / original_sfreq)
    resampled_data = scipy.signal.resample(data, num_samples_target, axis=-1)
    return resampled_data

def segment_data(data: np.ndarray, window_size_samples: int, step_samples: int) -> List[np.ndarray]:
    """
    Segments the data using a sliding window.
    """
    segments = []
    total_len = data.shape[-1]
    for start in range(0, total_len - window_size_samples + 1, step_samples):
        end = start + window_size_samples
        segments.append(data[:, start:end])
    return segments

def apply_preprocessing_pipeline(eeg_data: np.ndarray, preprocess_config: dict, sfreq: float) -> List[np.ndarray]:
    """
    Applies the full preprocessing pipeline.
    This function now processes one trial and returns a list of processed segments.
    """
    if eeg_data.ndim != 2:
        raise ValueError(f"Input EEG data must be 2D, but got shape {eeg_data.shape}")

    # 1. Filtering (applied to the whole trial first)
    current_data = eeg_data
    if preprocess_config.get('filter', {}).get('enable', False):
        lowcut = preprocess_config['filter']['lowcut']
        highcut = preprocess_config['filter']['highcut']
        order = preprocess_config['filter'].get('order', 5)
        current_data = butter_bandpass_filter(current_data, lowcut, highcut, sfreq, order)
        logger.debug(f"Applied bandpass filter ({lowcut}-{highcut} Hz).")

    # 2. Downsampling (applied to the whole trial)
    current_sfreq = sfreq
    if preprocess_config.get('downsample', {}).get('enable', False):
        target_sfreq = preprocess_config['downsample']['target_sfreq']
        current_data = downsample_data(current_data, original_sfreq=sfreq, target_sfreq=target_sfreq)
        current_sfreq = target_sfreq # Update sfreq for subsequent steps
        logger.debug(f"Downsampled data to {target_sfreq} Hz.")

    # 3. Segmentation (applied to the whole trial)
    if preprocess_config.get('segment', {}).get('enable', False):
        window_sec = preprocess_config['segment']['window_sec']
        step_sec = preprocess_config['segment']['step_sec']
        window_samples = int(window_sec * current_sfreq)
        step_samples = int(step_sec * current_sfreq)
        segments = segment_data(current_data, window_samples, step_samples)
        logger.debug(f"Segmented data into {len(segments)} windows.")
    else:
        # If no segmentation, treat the whole trial as a single segment
        segments = [current_data]

    # 4. Feature Extraction and Normalization (applied to each segment)
    final_features_list = []
    for segment in segments:
        processed_segment = segment
        if preprocess_config.get('feature_extraction', {}).get('enable', False):
            # ... (feature extraction logic is the same, but now on a segment)
            # This example only shows 'raw' and 'de' as before
            feature_type = preprocess_config['feature_extraction']['type']
            if feature_type == 'de':
                 # DE feature extraction logic here
                pass # Replace with your DE extraction for a segment
            elif feature_type == 'raw':
                processed_segment = segment # Keep the raw segment
        
        if preprocess_config.get('normalize', False):
            processed_segment = minmax_normalize(processed_segment)
        
        final_features_list.append(processed_segment)

    return final_features_list