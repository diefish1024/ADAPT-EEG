# src/utils/eeg_preprocessing.py

import numpy as np
import scipy.signal
import logging

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

def apply_preprocessing_pipeline(eeg_data: np.ndarray, preprocess_config: dict, sfreq: float) -> np.ndarray:
    """
    Applies the full preprocessing pipeline (filtering, feature extraction, normalization).
    
    Args:
        eeg_data (np.ndarray): Raw EEG data. Expected (channels, time_points).
        preprocess_config (dict): Configuration dictionary for preprocessing.
        sfreq (float): Sampling frequency of the EEG data.
    
    Returns:
        np.ndarray: Processed EEG features (1D) or filtered raw data (2D).
    """
    if eeg_data.ndim != 2:
        raise ValueError(f"EEG data for preprocessing must be 2D (channels, time_points), but got shape {eeg_data.shape}")

    # 1. Filtering
    filtered_data = eeg_data # Start with original data or previous stage output
    if preprocess_config.get('filter', {}).get('enable', False):
        lowcut = preprocess_config['filter']['lowcut']
        highcut = preprocess_config['filter']['highcut']
        order = preprocess_config['filter'].get('order', 5)
        filtered_data = butter_bandpass_filter(eeg_data, lowcut, highcut, sfreq, order)
        logger.debug(f"Applied bandpass filter ({lowcut}-{highcut} Hz). New shape: {filtered_data.shape}")

    # 2. Feature Extraction (Differential Entropy or Raw)
    processed_features = filtered_data # If no specific feature extraction, use filtered data
    if preprocess_config.get('feature_extraction', {}).get('enable', False):
        feature_type = preprocess_config['feature_extraction']['type']
        if feature_type == 'de':
            de_features_list = []
            bands = preprocess_config['feature_extraction']['bands']
            
            for band_name, (low, high) in bands.items():
                # Filter for specific band first
                band_data = butter_bandpass_filter(filtered_data, low, high, sfreq, order=5)
                # Calculate DE for each channel in this band
                de_vals_per_channel = calculate_differential_entropy(band_data) # Shape (channels,)
                de_features_list.append(de_vals_per_channel)
            
            # Concatenate DE features from all bands to form a single 1D feature vector per sample
            processed_features = np.concatenate(de_features_list, axis=0) # Shape: (num_bands * num_channels,)
            logger.debug(f"Extracted DE features. Shape: {processed_features.shape}")
        elif feature_type == 'raw':
             # If 'raw' features are requested, simply pass the filtered EEG data as is (2D: channels, time_points).
             # The EEGResNet18 will handle this 2D input.
            processed_features = filtered_data
            logger.debug(f"Using raw/filtered EEG data as features. Shape: {processed_features.shape}")
        else:
            raise ValueError(f"Unsupported feature extraction type: {feature_type}. Choose 'de' or 'raw'.")
    else:
        # If feature extraction is disabled, the 'features' are the filtered (or original) EEG data.
        # This will be passed as a (channels, time_points) tensor to EEGResNet18.
        processed_features = filtered_data
        logger.debug(f"No specific feature extraction enabled. Using raw/filtered EEG. Shape: {processed_features.shape}")

    # 3. Normalization
    if preprocess_config.get('normalize', False):
        # Normalize the *final* feature representation.
        # If it's a 2D tensor (channels, time_points), normalize across the whole tensor.
        # If it's a 1D vector (features), normalize the vector.
        processed_features = minmax_normalize(processed_features)
        logger.debug(f"Applied Min-Max normalization. Shape: {processed_features.shape}")
        
    return processed_features
