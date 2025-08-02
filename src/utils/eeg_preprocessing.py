# src/utils/eeg_preprocessing.py

import numpy as np
import scipy.signal
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """Applies a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    y = scipy.signal.filtfilt(b, a, data, axis=-1)
    return y

def cheby2_bandpass_filter(data: np.ndarray, band_cut: List[float], fs: float, filt_allowance: List[float] = [0.2, 5], axis: int = -1) -> np.ndarray:
    """
    Applies a Chebyshev Type II bandpass filter. Adapted from EmT's methodology.
    """
    a_stop = 30  # stopband attenuation
    a_pass = 3   # passband attenuation
    n_freq = fs / 2  # Nyquist frequency
    f_pass = (np.array(band_cut) / n_freq).tolist()
    f_stop = [(band_cut[0] - filt_allowance[0]) / n_freq, (band_cut[1] + filt_allowance[1]) / n_freq]
    
    # Ensure stop frequencies are within valid range
    f_stop[0] = max(f_stop[0], 0.01)
    f_stop[1] = min(f_stop[1], 0.99)

    n, ws = scipy.signal.cheb2ord(f_pass, f_stop, a_pass, a_stop)
    sos = scipy.signal.cheby2(n, a_stop, ws, 'bandpass', output='sos')
    data_out = scipy.signal.sosfilt(sos, data, axis=axis)
    return data_out

def calculate_differential_entropy(eeg_data_chunk: np.ndarray) -> np.ndarray:
    """Calculates differential entropy (DE)."""
    if eeg_data_chunk.ndim == 1:
        eeg_data_chunk = eeg_data_chunk[np.newaxis, :]
    
    variance = np.var(eeg_data_chunk, axis=-1)
    variance[variance <= 1e-6] = 1e-6 
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    return de

def minmax_normalize(data: np.ndarray) -> np.ndarray:
    """Applies min-max normalization."""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def downsample_data(data: np.ndarray, original_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Downsamples data to a target sampling frequency."""
    if original_sfreq == target_sfreq:
        return data
    num_samples_original = data.shape[-1]
    num_samples_target = int(num_samples_original * target_sfreq / original_sfreq)
    return scipy.signal.resample(data, num_samples_target, axis=-1)

def segment_data(data: np.ndarray, window_size_samples: int, step_samples: int) -> List[np.ndarray]:
    """Segments data using a sliding window."""
    segments = []
    total_len = data.shape[-1]
    for start in range(0, total_len - window_size_samples + 1, step_samples):
        end = start + window_size_samples
        segments.append(data[:, start:end])
    return segments

def apply_preprocessing_pipeline(eeg_data: np.ndarray, preprocess_config: Dict[str, Any], sfreq: float) -> List[np.ndarray]:
    """
    Applies the full preprocessing pipeline for one trial, returning a list of processed segments.
    """
    if eeg_data.ndim != 2:
        raise ValueError(f"Input EEG data must be 2D, but got shape {eeg_data.shape}")

    # 1. Downsampling (Applied first to maintain consistency)
    current_data = eeg_data
    current_sfreq = sfreq
    if preprocess_config.get('downsample', {}).get('enable', False):
        target_sfreq = preprocess_config['downsample']['target_sfreq']
        current_data = downsample_data(current_data, original_sfreq=sfreq, target_sfreq=target_sfreq)
        current_sfreq = target_sfreq
        logger.debug(f"Downsampled data to {target_sfreq} Hz.")

    # 2. Segmentation (Splits trial into larger segments)
    if preprocess_config.get('segment', {}).get('enable', False):
        window_sec = preprocess_config['segment']['window_sec']
        step_sec = preprocess_config['segment']['step_sec']
        window_samples = int(window_sec * current_sfreq)
        step_samples = int(step_sec * current_sfreq)
        segments = segment_data(current_data, window_samples, step_samples)
    else:
        segments = [current_data]

    # 3. Feature Extraction (Applied to each segment)
    final_features_list = []
    feature_config = preprocess_config.get('feature_extraction', {})
    if not feature_config.get('enable', False):
        return segments # Return raw, segmented data if no feature extraction

    feature_type = feature_config.get('type', 'raw')
    
    for segment in segments:
        if feature_type == 'de_emt':
            # EmT-specific feature extraction
            bands = feature_config.get('bands', {})
            sub_segment_config = feature_config.get('sub_segment', {})
            if not bands or not sub_segment_config:
                raise ValueError("Config for 'de_emt' must include 'bands' and 'sub_segment' sections.")

            sub_window_sec = sub_segment_config['window_sec']
            sub_step_sec = sub_segment_config['step_sec']
            sub_window_samples = int(sub_window_sec * current_sfreq)
            sub_step_samples = int(sub_step_sec * current_sfreq)

            # Segment the larger segment into sub-segments (e.g., 1-second windows)
            sub_segments = segment_data(segment, sub_window_samples, sub_step_samples)
            
            all_bands_features = []
            for band_name, band_range in bands.items():
                # Filter the whole segment first for efficiency
                filtered_segment = cheby2_bandpass_filter(segment, band_range, current_sfreq)
                
                band_de_features = []
                # Now calculate DE on sub-segments of the filtered data
                sub_segments_filtered = segment_data(filtered_segment, sub_window_samples, sub_step_samples)
                for sub_seg in sub_segments_filtered:
                    de = calculate_differential_entropy(sub_seg)
                    band_de_features.append(de)
                
                # Shape: (num_sub_segments, channels)
                all_bands_features.append(np.stack(band_de_features))

            # Stack features: (num_bands, num_sub_segments, channels) -> (channels, num_sub_segments, num_bands)
            processed_segment = np.stack(all_bands_features).transpose(2, 1, 0)
        
        elif feature_type == 'de':
            # Your original DE feature extraction
            bands = feature_config.get('bands', {})
            band_features = []
            for band_name, band_range in bands.items():
                filtered_data = butter_bandpass_filter(segment, band_range[0], band_range[1], current_sfreq)
                de = calculate_differential_entropy(filtered_data)
                band_features.append(de)
            # Shape: (channels, num_bands)
            processed_segment = np.stack(band_features, axis=-1)
        
        else: # 'raw'
            processed_segment = segment

        if preprocess_config.get('normalize', False):
            processed_segment = minmax_normalize(processed_segment)
        
        final_features_list.append(processed_segment)

    return final_features_list
