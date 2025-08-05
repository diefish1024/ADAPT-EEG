# tests/test_eeg_preprocessing.py

import unittest
import numpy as np
import sys
import os

# Add project root to path to allow importing from src
# Assumes this test file is directly under 'tests/' and 'src/' is in the project root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import all relevant functions from eeg_preprocessing module
from src.utils.eeg_preprocessing import (
    calculate_differential_entropy,
    butter_bandpass_filter,
    cheby2_bandpass_filter,
    downsample_data,
    segment_data,
    minmax_normalize,
    apply_preprocessing_pipeline
)

class Test_CalculateDifferentialEntropy(unittest.TestCase):
    """Tests for the calculate_differential_entropy function."""

    def test_calculate_differential_entropy_basic(self):
        """Test DE calculation with a known, deterministic variance."""
        # Create data with a precise, known variance of 1.0.
        # The mean is 0, so variance is ((-1)^2 + 1^2) / 2 = 1.
        data = np.array([[-1, 1], [-1, 1]], dtype=np.float32)
        
        # Expected DE for variance=1 is 0.5 * log(2*pi*e)
        expected_de = 0.5 * np.log(2 * np.pi * np.e)
        
        de_values = calculate_differential_entropy(data)
        
        self.assertEqual(de_values.shape, (2,), "Output shape should be (num_channels,).")
        # With deterministic data, use a very small tolerance for precision.
        np.testing.assert_allclose(de_values, expected_de, rtol=1e-6, err_msg="DE calculation is incorrect.")

    def test_calculate_differential_entropy_zero_variance(self):
        """Test DE calculation with zero variance data (e.g., all same values)."""
        data = np.ones((2, 100)) # Zero variance across time for each channel
        de_values = calculate_differential_entropy(data)
        self.assertTrue(np.all(np.isfinite(de_values)), "DE should be finite even for zero variance.")
        # Technically, DE for zero variance is -inf, but numerical stability might lead to a large negative number
        # or a specific handled value. Assuming the function handles log(0) or small variance gracefully,
        # we check for finiteness as per original test.


class Test_BandpassFilters(unittest.TestCase):
    """Tests for butter_bandpass_filter and cheby2_bandpass_filter functions."""

    def setUp(self):
        """Set up common test data: a mixed sine wave signal."""
        self.fs = 200.0
        self.num_channels = 2
        self.num_samples = 1000
        # Create a simple sine wave mixture for testing filter frequency response indirectly
        t = np.linspace(0, 5, self.num_samples, endpoint=False)
        low_freq_signal = np.sin(2 * np.pi * 5 * t)   # 5 Hz component
        high_freq_signal = 0.5 * np.sin(2 * np.pi * 50 * t) # 50 Hz component
        self.mock_data = np.array([low_freq_signal + high_freq_signal] * self.num_channels)

    def test_butter_bandpass_filter_shape_preservation(self):
        """Test if the Butterworth filter preserves data shape."""
        filtered_data = butter_bandpass_filter(self.mock_data, lowcut=8.0, highcut=30.0, fs=self.fs)
        self.assertEqual(self.mock_data.shape, filtered_data.shape, "Output shape should match input shape.")

    def test_butter_bandpass_filter_integrity(self):
        """Test if the Butterworth filter output is finite (no NaNs or Infs)."""
        filtered_data = butter_bandpass_filter(self.mock_data, lowcut=8.0, highcut=30.0, fs=self.fs)
        self.assertTrue(np.all(np.isfinite(filtered_data)), "Output should not contain NaN or Inf values.")

    def test_cheby2_bandpass_filter_shape_preservation(self):
        """Test if the Chebyshev II filter preserves data shape."""
        filtered_data = cheby2_bandpass_filter(self.mock_data, band_cut=[8.0, 30.0], fs=self.fs)
        self.assertEqual(self.mock_data.shape, filtered_data.shape, "Output shape should match input shape.")
    
    def test_cheby2_bandpass_filter_integrity(self):
        """Test if the Chebyshev II filter output is finite (no NaNs or Infs)."""
        filtered_data = cheby2_bandpass_filter(self.mock_data, band_cut=[8.0, 30.0], fs=self.fs)
        self.assertTrue(np.all(np.isfinite(filtered_data)), "Output should not contain NaN or Inf values.")


class Test_CorePreprocessingBlocks(unittest.TestCase):
    """Tests for basic EEG preprocessing blocks: downsampling, segmentation, normalization."""

    def setUp(self):
        """Set up common test data."""
        self.fs = 200.0
        self.num_channels = 2
        self.num_samples = 1000
        self.mock_data = np.random.randn(self.num_channels, self.num_samples) # Random noise data

    def test_downsample_data_correct_samples(self):
        """Test if downsampling produces the correct number of samples."""
        target_fs = 100.0
        downsampled_data = downsample_data(self.mock_data, original_sfreq=self.fs, target_sfreq=target_fs)
        expected_samples = int(self.num_samples * target_fs / self.fs)
        self.assertEqual(downsampled_data.shape, (self.num_channels, expected_samples), "Downsampled data has incorrect sample count.")

    def test_segment_data_correct_segments_and_shape(self):
        """Test if segmentation creates the correct number and shape of segments."""
        window_samples = 200
        step_samples = 100
        segments = segment_data(self.mock_data, window_samples, step_samples)
        
        expected_num_segments = (self.num_samples - window_samples) // step_samples + 1
        self.assertEqual(len(segments), expected_num_segments, "Incorrect number of segments.")
        self.assertEqual(segments[0].shape, (self.num_channels, window_samples), "Segment shape is incorrect.")

    def test_minmax_normalize_range(self):
        """Test if min-max normalization scales data to the [0, 1] range."""
        data = np.array([[-10, 0, 10, 20]], dtype=np.float32)
        normalized_data = minmax_normalize(data)
        # Using assertAlmostEqual for floating point comparison with a default precision
        self.assertAlmostEqual(np.min(normalized_data), 0.0, msg="Normalized min should be 0.")
        self.assertAlmostEqual(np.max(normalized_data), 1.0, msg="Normalized max should be 1.")
    
    def test_minmax_normalize_zero_range_data(self):
        """Test normalization when all data points are the same (zero range)."""
        data = np.ones((2, 10)) # All values are 1
        normalized_data = minmax_normalize(data)
        self.assertTrue(np.all(normalized_data == 0), "Data with zero range should normalize to 0.")


class Test_FullPreprocessingPipeline(unittest.TestCase):
    """Tests for the integrated apply_preprocessing_pipeline function."""

    def setUp(self):
        """Set up common test data and configuration for the pipeline."""
        self.sfreq = 200.0
        self.mock_eeg_data = np.random.randn(62, int(10 * self.sfreq)) # 10 seconds of data
        self.config_de_emt = {
            'downsample': {'enable': True, 'target_sfreq': 128.0},
            'segment': {'enable': True, 'window_sec': 4.0, 'step_sec': 4.0},
            'feature_extraction': {
                'enable': True,
                'type': 'de_emt',
                'bands': {'alpha': [8, 13], 'beta': [13, 30]},
                'sub_segment': {'window_sec': 1.0, 'step_sec': 1.0}
            },
            'normalize': True
        }

    def test_apply_preprocessing_pipeline_de_emt_full_flow(self):
        """Test the full pipeline for 'de_emt' feature extraction and its output properties."""
        processed_features = apply_preprocessing_pipeline(
            self.mock_eeg_data, self.config_de_emt, self.sfreq
        )
        
        # Expected shape calculation based on the config.
        # Original 10s trial @ 200Hz -> downsampled to 128Hz: 10 * 128 = 1280 samples.
        # Segmentation: 4s window @ 128Hz = 512 samples/window. 4s step = 512 samples/step.
        # Number of segments: (1280 - 512) // 512 + 1 = 768 // 512 + 1 = 1 + 1 = 2 segments.
        num_segments = 2 
        num_channels = 62
        # Within each 4s segment, sub-segmentation into 1s sub-segments: 4 sub-segments.
        num_sub_segments = 4 
        num_bands = 2 # alpha, beta
        
        self.assertEqual(len(processed_features), num_segments, "Should produce the correct number of segments.")
        
        first_segment = processed_features[0]
        self.assertEqual(first_segment.shape, (num_channels, num_sub_segments, num_bands), "First segment's feature shape is incorrect.")
        self.assertTrue(np.all(np.isfinite(first_segment)), "Extracted features should be finite (no NaNs or Infs).")
        # Check normalization: features should be within [0, 1] range, allowing minor float deviation.
        self.assertTrue(np.amin(first_segment) >= -1e-7 and np.amax(first_segment) <= 1.0 + 1e-7, "Features should be normalized to [0, 1].")


if __name__ == '__main__':
    unittest.main()
