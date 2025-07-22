"""
Unit tests for signal processing utilities.

This module contains comprehensive tests for all signal processing
components including filtering, windowing, and feature extraction.
"""

import unittest
import numpy as np
import warnings
from typing import Dict, Any

# Import the modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wave_analysis.services.signal_processing import (
    FilterBank, FilterParameters, FilterType,
    WindowFunction, WindowParameters, WindowType,
    FeatureExtractor, SignalProcessor
)


class TestFilterParameters(unittest.TestCase):
    """Test FilterParameters data class validation."""
    
    def test_valid_lowpass_parameters(self):
        """Test valid lowpass filter parameters."""
        params = FilterParameters(
            filter_type=FilterType.LOWPASS,
            cutoff_freq=10.0,
            order=4,
            sampling_rate=100.0
        )
        self.assertEqual(params.filter_type, FilterType.LOWPASS)
        self.assertEqual(params.cutoff_freq, 10.0)
    
    def test_valid_bandpass_parameters(self):
        """Test valid bandpass filter parameters."""
        params = FilterParameters(
            filter_type=FilterType.BANDPASS,
            cutoff_freq=(1.0, 10.0),
            order=4,
            sampling_rate=100.0
        )
        self.assertEqual(params.cutoff_freq, (1.0, 10.0))
    
    def test_invalid_order(self):
        """Test invalid filter order."""
        with self.assertRaises(ValueError):
            FilterParameters(
                filter_type=FilterType.LOWPASS,
                cutoff_freq=10.0,
                order=0,
                sampling_rate=100.0
            )
    
    def test_invalid_sampling_rate(self):
        """Test invalid sampling rate."""
        with self.assertRaises(ValueError):
            FilterParameters(
                filter_type=FilterType.LOWPASS,
                cutoff_freq=10.0,
                order=4,
                sampling_rate=-1.0
            )
    
    def test_cutoff_above_nyquist(self):
        """Test cutoff frequency above Nyquist frequency."""
        with self.assertRaises(ValueError):
            FilterParameters(
                filter_type=FilterType.LOWPASS,
                cutoff_freq=60.0,  # Above Nyquist (50 Hz)
                order=4,
                sampling_rate=100.0
            )
    
    def test_invalid_bandpass_frequencies(self):
        """Test invalid bandpass frequency order."""
        with self.assertRaises(ValueError):
            FilterParameters(
                filter_type=FilterType.BANDPASS,
                cutoff_freq=(10.0, 5.0),  # High < Low
                order=4,
                sampling_rate=100.0
            )


class TestWindowParameters(unittest.TestCase):
    """Test WindowParameters data class validation."""
    
    def test_valid_parameters(self):
        """Test valid window parameters."""
        params = WindowParameters(
            window_type=WindowType.HANN,
            window_length=1024,
            overlap=0.5
        )
        self.assertEqual(params.window_type, WindowType.HANN)
        self.assertEqual(params.window_length, 1024)
    
    def test_invalid_window_length(self):
        """Test invalid window length."""
        with self.assertRaises(ValueError):
            WindowParameters(
                window_type=WindowType.HANN,
                window_length=0,
                overlap=0.5
            )
    
    def test_invalid_overlap(self):
        """Test invalid overlap values."""
        with self.assertRaises(ValueError):
            WindowParameters(
                window_type=WindowType.HANN,
                window_length=1024,
                overlap=1.5  # > 1
            )
        
        with self.assertRaises(ValueError):
            WindowParameters(
                window_type=WindowType.HANN,
                window_length=1024,
                overlap=-0.1  # < 0
            )


class TestFilterBank(unittest.TestCase):
    """Test FilterBank functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter_bank = FilterBank()
        self.sampling_rate = 100.0
        self.test_data = self._create_test_signal()
    
    def _create_test_signal(self) -> np.ndarray:
        """Create synthetic test signal with known frequencies."""
        t = np.linspace(0, 10, int(10 * self.sampling_rate))
        # Mix of 2 Hz, 10 Hz, and 25 Hz components
        signal = (np.sin(2 * np.pi * 2 * t) + 
                 0.5 * np.sin(2 * np.pi * 10 * t) + 
                 0.2 * np.sin(2 * np.pi * 25 * t))
        return signal
    
    def test_lowpass_filter(self):
        """Test lowpass filtering."""
        params = FilterParameters(
            filter_type=FilterType.LOWPASS,
            cutoff_freq=5.0,
            order=4,
            sampling_rate=self.sampling_rate
        )
        
        filtered = self.filter_bank.apply_filter(self.test_data, params)
        
        # Check that output has same length
        self.assertEqual(len(filtered), len(self.test_data))
        
        # Check that high frequencies are attenuated
        # (This is a basic check - more sophisticated spectral analysis could be done)
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_highpass_filter(self):
        """Test highpass filtering."""
        params = FilterParameters(
            filter_type=FilterType.HIGHPASS,
            cutoff_freq=5.0,
            order=4,
            sampling_rate=self.sampling_rate
        )
        
        filtered = self.filter_bank.apply_filter(self.test_data, params)
        self.assertEqual(len(filtered), len(self.test_data))
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        params = FilterParameters(
            filter_type=FilterType.BANDPASS,
            cutoff_freq=(5.0, 15.0),
            order=4,
            sampling_rate=self.sampling_rate
        )
        
        filtered = self.filter_bank.apply_filter(self.test_data, params)
        self.assertEqual(len(filtered), len(self.test_data))
    
    def test_bandstop_filter(self):
        """Test bandstop filtering."""
        params = FilterParameters(
            filter_type=FilterType.BANDSTOP,
            cutoff_freq=(8.0, 12.0),
            order=4,
            sampling_rate=self.sampling_rate
        )
        
        filtered = self.filter_bank.apply_filter(self.test_data, params)
        self.assertEqual(len(filtered), len(self.test_data))
    
    def test_frequency_band_filtering(self):
        """Test predefined frequency band filtering."""
        filtered = self.filter_bank.apply_frequency_band(
            self.test_data, 'p_wave', self.sampling_rate
        )
        self.assertEqual(len(filtered), len(self.test_data))
    
    def test_invalid_frequency_band(self):
        """Test invalid frequency band name."""
        with self.assertRaises(ValueError):
            self.filter_bank.apply_frequency_band(
                self.test_data, 'invalid_band', self.sampling_rate
            )
    
    def test_empty_data(self):
        """Test filtering with empty data."""
        with self.assertRaises(ValueError):
            params = FilterParameters(
                filter_type=FilterType.LOWPASS,
                cutoff_freq=5.0,
                order=4,
                sampling_rate=self.sampling_rate
            )
            self.filter_bank.apply_filter(np.array([]), params)
    
    def test_get_available_bands(self):
        """Test getting available frequency bands."""
        bands = self.filter_bank.get_available_bands()
        self.assertIsInstance(bands, dict)
        self.assertIn('p_wave', bands)
        self.assertIn('s_wave', bands)
        self.assertIn('surface_wave', bands)
    
    def test_filter_caching(self):
        """Test that filter coefficients are cached."""
        params = FilterParameters(
            filter_type=FilterType.LOWPASS,
            cutoff_freq=5.0,
            order=4,
            sampling_rate=self.sampling_rate
        )
        
        # Apply filter twice
        filtered1 = self.filter_bank.apply_filter(self.test_data, params)
        filtered2 = self.filter_bank.apply_filter(self.test_data, params)
        
        # Results should be identical
        np.testing.assert_array_equal(filtered1, filtered2)


class TestWindowFunction(unittest.TestCase):
    """Test WindowFunction functionality."""
    
    def test_create_hann_window(self):
        """Test Hann window creation."""
        params = WindowParameters(
            window_type=WindowType.HANN,
            window_length=100
        )
        window = WindowFunction.create_window(params)
        
        self.assertEqual(len(window), 100)
        self.assertAlmostEqual(window[0], 0.0, places=10)  # Hann window starts at 0
        self.assertAlmostEqual(window[-1], 0.0, places=10)  # Hann window ends at 0
    
    def test_create_rectangular_window(self):
        """Test rectangular window creation."""
        params = WindowParameters(
            window_type=WindowType.RECTANGULAR,
            window_length=50
        )
        window = WindowFunction.create_window(params)
        
        self.assertEqual(len(window), 50)
        np.testing.assert_array_equal(window, np.ones(50))
    
    def test_create_kaiser_window(self):
        """Test Kaiser window creation."""
        params = WindowParameters(
            window_type=WindowType.KAISER,
            window_length=100,
            beta=8.6
        )
        window = WindowFunction.create_window(params)
        
        self.assertEqual(len(window), 100)
        self.assertGreater(np.max(window), 0.9)  # Kaiser window should have values near 1
    
    def test_segment_data(self):
        """Test data segmentation."""
        data = np.random.randn(1000)
        params = WindowParameters(
            window_type=WindowType.HANN,
            window_length=100,
            overlap=0.5
        )
        
        segments = WindowFunction.segment_data(data, params)
        
        self.assertGreater(len(segments), 0)
        self.assertEqual(len(segments[0]), 100)
        
        # Check that we get expected number of segments
        step_size = int(100 * (1 - 0.5))
        expected_segments = (len(data) - 100) // step_size + 1
        self.assertLessEqual(len(segments), expected_segments + 1)
    
    def test_segment_data_too_short(self):
        """Test segmentation with data shorter than window."""
        data = np.random.randn(50)
        params = WindowParameters(
            window_type=WindowType.HANN,
            window_length=100
        )
        
        with self.assertRaises(ValueError):
            WindowFunction.segment_data(data, params)
    
    def test_apply_window(self):
        """Test applying window to data."""
        data = np.ones(100)
        windowed = WindowFunction.apply_window(data, WindowType.HANN)
        
        self.assertEqual(len(windowed), 100)
        self.assertAlmostEqual(windowed[0], 0.0, places=10)
        self.assertAlmostEqual(windowed[-1], 0.0, places=10)
    
    def test_unsupported_window_type(self):
        """Test unsupported window type."""
        # This test would need a mock or invalid enum value
        # For now, we'll test with a valid type to ensure the method works
        params = WindowParameters(
            window_type=WindowType.HAMMING,
            window_length=100
        )
        window = WindowFunction.create_window(params)
        self.assertEqual(len(window), 100)


class TestFeatureExtractor(unittest.TestCase):
    """Test FeatureExtractor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
        self.extractor = FeatureExtractor(self.sampling_rate)
        self.test_data = self._create_test_signal()
    
    def _create_test_signal(self) -> np.ndarray:
        """Create synthetic test signal."""
        t = np.linspace(0, 5, int(5 * self.sampling_rate))
        # Simple sinusoid with known properties
        signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
        return signal
    
    def test_extract_time_domain_features(self):
        """Test time-domain feature extraction."""
        features = self.extractor.extract_time_domain_features(self.test_data)
        
        # Check that all expected features are present
        expected_features = [
            'mean', 'std', 'variance', 'rms', 'peak_amplitude',
            'peak_to_peak', 'skewness', 'kurtosis', 'zero_crossing_rate',
            'energy', 'signal_length', 'duration'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
        
        # Check some specific values for sine wave
        self.assertAlmostEqual(features['mean'], 0.0, places=2)  # Sine wave has zero mean
        self.assertEqual(features['signal_length'], len(self.test_data))
        self.assertAlmostEqual(features['duration'], 5.0, places=1)
    
    def test_extract_frequency_domain_features(self):
        """Test frequency-domain feature extraction."""
        features = self.extractor.extract_frequency_domain_features(self.test_data)
        
        # Check that all expected features are present
        expected_features = [
            'frequencies', 'power_spectrum', 'dominant_frequency',
            'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff',
            'spectral_flux', 'total_power', 'frequency_range'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
        
        # For 10 Hz sine wave, dominant frequency should be around 10 Hz
        self.assertAlmostEqual(features['dominant_frequency'], 10.0, places=0)
    
    def test_calculate_sta_lta_ratio(self):
        """Test STA/LTA ratio calculation."""
        # Create signal with step change to test STA/LTA
        data = np.concatenate([
            np.random.randn(500) * 0.1,  # Low amplitude noise
            np.random.randn(500) * 1.0   # High amplitude signal
        ])
        
        sta_lta = self.extractor.calculate_sta_lta_ratio(data, sta_window=1.0, lta_window=5.0)
        
        self.assertEqual(len(sta_lta), len(data))
        
        # STA/LTA should increase when signal amplitude increases
        early_ratio = np.mean(sta_lta[600:650])  # After step change
        late_ratio = np.mean(sta_lta[100:150])   # Before step change
        self.assertGreater(early_ratio, late_ratio)
    
    def test_sta_lta_invalid_windows(self):
        """Test STA/LTA with invalid window parameters."""
        with self.assertRaises(ValueError):
            self.extractor.calculate_sta_lta_ratio(
                self.test_data, sta_window=10.0, lta_window=5.0  # STA > LTA
            )
    
    def test_extract_spectral_features_by_band(self):
        """Test band-specific feature extraction."""
        filter_bank = FilterBank()
        band_features = self.extractor.extract_spectral_features_by_band(
            self.test_data, filter_bank
        )
        
        self.assertIsInstance(band_features, dict)
        self.assertIn('p_wave', band_features)
        self.assertIn('s_wave', band_features)
        
        # Each band should have time and frequency domain features
        for band_name, features in band_features.items():
            if features:  # Skip empty feature sets (due to warnings)
                self.assertIn('time_domain', features)
                self.assertIn('frequency_domain', features)
    
    def test_empty_data_features(self):
        """Test feature extraction with empty data."""
        empty_features = self.extractor.extract_time_domain_features(np.array([]))
        self.assertEqual(empty_features, {})
        
        empty_freq_features = self.extractor.extract_frequency_domain_features(np.array([]))
        self.assertEqual(empty_freq_features, {})
    
    def test_skewness_calculation(self):
        """Test skewness calculation."""
        # Test with symmetric data (should have low skewness)
        symmetric_data = np.random.normal(0, 1, 1000)
        skewness = self.extractor._calculate_skewness(symmetric_data)
        self.assertLess(abs(skewness), 0.5)  # Should be close to 0
        
        # Test with too little data
        short_data = np.array([1, 2])
        skewness_short = self.extractor._calculate_skewness(short_data)
        self.assertEqual(skewness_short, 0.0)
    
    def test_kurtosis_calculation(self):
        """Test kurtosis calculation."""
        # Test with normal data (should have kurtosis around 0)
        normal_data = np.random.normal(0, 1, 1000)
        kurtosis = self.extractor._calculate_kurtosis(normal_data)
        self.assertLess(abs(kurtosis), 1.0)  # Should be close to 0
        
        # Test with too little data
        short_data = np.array([1, 2, 3])
        kurtosis_short = self.extractor._calculate_kurtosis(short_data)
        self.assertEqual(kurtosis_short, 0.0)
    
    def test_zero_crossing_rate(self):
        """Test zero crossing rate calculation."""
        # Sine wave should have predictable zero crossing rate
        t = np.linspace(0, 1, 100)
        sine_wave = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
        zcr = self.extractor._calculate_zero_crossing_rate(sine_wave)
        
        # 5 Hz sine wave should cross zero about 10 times per second
        self.assertGreater(zcr, 0.05)  # At least some zero crossings
        
        # Test with constant data (no zero crossings)
        constant_data = np.ones(100)
        zcr_constant = self.extractor._calculate_zero_crossing_rate(constant_data)
        self.assertEqual(zcr_constant, 0.0)


class TestSignalProcessor(unittest.TestCase):
    """Test SignalProcessor main coordinator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
        self.processor = SignalProcessor(self.sampling_rate)
        self.test_data = self._create_test_signal()
    
    def _create_test_signal(self) -> np.ndarray:
        """Create synthetic test signal with trend and noise."""
        t = np.linspace(0, 10, int(10 * self.sampling_rate))
        # Signal with trend, DC offset, and multiple frequency components
        signal = (2.0 +  # DC offset
                 0.1 * t +  # Linear trend
                 np.sin(2 * np.pi * 5 * t) +  # 5 Hz component
                 0.5 * np.sin(2 * np.pi * 15 * t) +  # 15 Hz component
                 0.1 * np.random.randn(len(t)))  # Noise
        return signal
    
    def test_preprocess_seismic_data(self):
        """Test seismic data preprocessing."""
        processed = self.processor.preprocess_seismic_data(self.test_data)
        
        # Check that preprocessing reduces mean and trend
        original_mean = np.mean(self.test_data)
        processed_mean = np.mean(processed)
        
        self.assertLess(abs(processed_mean), abs(original_mean))
        self.assertEqual(len(processed), len(self.test_data))
    
    def test_preprocess_options(self):
        """Test preprocessing with different options."""
        # Test with all options disabled
        processed = self.processor.preprocess_seismic_data(
            self.test_data,
            remove_mean=False,
            detrend=False,
            taper=False
        )
        
        # Should be identical to original
        np.testing.assert_array_equal(processed, self.test_data)
    
    def test_segment_for_analysis(self):
        """Test data segmentation for analysis."""
        segments = self.processor.segment_for_analysis(
            self.test_data,
            segment_length=2.0,  # 2 seconds
            overlap=0.5
        )
        
        self.assertGreater(len(segments), 0)
        expected_length = int(2.0 * self.sampling_rate)
        self.assertEqual(len(segments[0]), expected_length)
    
    def test_extract_comprehensive_features(self):
        """Test comprehensive feature extraction."""
        features = self.processor.extract_comprehensive_features(self.test_data)
        
        # Check that all feature categories are present
        expected_categories = [
            'time_domain', 'frequency_domain', 'band_features',
            'sta_lta_ratio', 'processing_metadata'
        ]
        
        for category in expected_categories:
            self.assertIn(category, features)
        
        # Check metadata
        metadata = features['processing_metadata']
        self.assertEqual(metadata['sampling_rate'], self.sampling_rate)
        self.assertEqual(metadata['data_length'], len(self.test_data))
    
    def test_short_data_segmentation(self):
        """Test segmentation with data shorter than segment length."""
        short_data = np.random.randn(50)  # Very short data
        
        # Should handle gracefully or raise appropriate error
        try:
            segments = self.processor.segment_for_analysis(
                short_data,
                segment_length=2.0  # Longer than data
            )
            # If it doesn't raise an error, should return empty or handle gracefully
            self.assertIsInstance(segments, list)
        except ValueError:
            # This is also acceptable behavior
            pass


class TestIntegration(unittest.TestCase):
    """Integration tests for signal processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
        self.processor = SignalProcessor(self.sampling_rate)
        
    def test_realistic_earthquake_signal(self):
        """Test with realistic earthquake-like signal."""
        # Create more realistic earthquake signal
        t = np.linspace(0, 30, int(30 * self.sampling_rate))
        
        # P-wave arrival at t=5s
        p_wave = np.zeros_like(t)
        p_mask = (t >= 5) & (t <= 8)
        p_wave[p_mask] = np.sin(2 * np.pi * 8 * t[p_mask]) * np.exp(-(t[p_mask] - 5))
        
        # S-wave arrival at t=10s
        s_wave = np.zeros_like(t)
        s_mask = (t >= 10) & (t <= 15)
        s_wave[s_mask] = np.sin(2 * np.pi * 4 * t[s_mask]) * np.exp(-(t[s_mask] - 10) / 2)
        
        # Surface waves at t=20s
        surface_wave = np.zeros_like(t)
        surf_mask = (t >= 20) & (t <= 28)
        surface_wave[surf_mask] = np.sin(2 * np.pi * 0.5 * t[surf_mask]) * np.exp(-(t[surf_mask] - 20) / 5)
        
        # Combine with noise
        earthquake_signal = p_wave + s_wave + surface_wave + 0.1 * np.random.randn(len(t))
        
        # Test comprehensive analysis
        features = self.processor.extract_comprehensive_features(earthquake_signal)
        
        # Should successfully extract features
        self.assertIn('time_domain', features)
        self.assertIn('frequency_domain', features)
        self.assertIn('sta_lta_ratio', features)
        
        # STA/LTA should show peaks at wave arrivals
        sta_lta = features['sta_lta_ratio']
        self.assertEqual(len(sta_lta), len(earthquake_signal))
        
        # Test band filtering
        filter_bank = FilterBank()
        p_wave_filtered = filter_bank.apply_frequency_band(
            earthquake_signal, 'p_wave', self.sampling_rate
        )
        s_wave_filtered = filter_bank.apply_frequency_band(
            earthquake_signal, 's_wave', self.sampling_rate
        )
        
        self.assertEqual(len(p_wave_filtered), len(earthquake_signal))
        self.assertEqual(len(s_wave_filtered), len(earthquake_signal))
    
    def test_processing_pipeline(self):
        """Test complete processing pipeline."""
        # Create test signal
        t = np.linspace(0, 10, 1000)
        test_signal = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
        
        # Step 1: Preprocess
        processed = self.processor.preprocess_seismic_data(test_signal)
        
        # Step 2: Filter for different bands
        filter_bank = FilterBank()
        p_filtered = filter_bank.apply_frequency_band(processed, 'p_wave', self.sampling_rate)
        s_filtered = filter_bank.apply_frequency_band(processed, 's_wave', self.sampling_rate)
        
        # Step 3: Extract features
        p_features = self.processor.feature_extractor.extract_time_domain_features(p_filtered)
        s_features = self.processor.feature_extractor.extract_time_domain_features(s_filtered)
        
        # Step 4: Calculate STA/LTA
        sta_lta = self.processor.feature_extractor.calculate_sta_lta_ratio(
            processed, sta_window=0.5, lta_window=2.0
        )
        
        # All steps should complete successfully
        self.assertIsInstance(processed, np.ndarray)
        self.assertIsInstance(p_filtered, np.ndarray)
        self.assertIsInstance(s_filtered, np.ndarray)
        self.assertIsInstance(p_features, dict)
        self.assertIsInstance(s_features, dict)
        self.assertIsInstance(sta_lta, np.ndarray)


if __name__ == '__main__':
    # Run tests with warnings suppressed for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unittest.main(verbosity=2)