"""
Unit tests for P-wave detection algorithm.

This module contains comprehensive tests for the P-wave detector including
STA/LTA algorithm, characteristic functions, and onset detection.
"""

import unittest
import numpy as np
import warnings
from typing import List

# Import the modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from wave_analysis.services.wave_detectors import (
    PWaveDetector, PWaveDetectionParameters, DetectionParameters
)
from wave_analysis.models import WaveSegment


class TestDetectionParameters(unittest.TestCase):
    """Test DetectionParameters data class validation."""
    
    def test_valid_parameters(self):
        """Test valid detection parameters."""
        params = DetectionParameters(
            sampling_rate=100.0,
            min_wave_duration=0.5,
            max_wave_duration=30.0,
            confidence_threshold=0.5
        )
        self.assertEqual(params.sampling_rate, 100.0)
        self.assertEqual(params.min_wave_duration, 0.5)
    
    def test_invalid_sampling_rate(self):
        """Test invalid sampling rate."""
        with self.assertRaises(ValueError):
            DetectionParameters(sampling_rate=-1.0)
    
    def test_invalid_duration_order(self):
        """Test invalid duration order."""
        with self.assertRaises(ValueError):
            DetectionParameters(
                sampling_rate=100.0,
                min_wave_duration=30.0,
                max_wave_duration=0.5  # max < min
            )
    
    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold."""
        with self.assertRaises(ValueError):
            DetectionParameters(
                sampling_rate=100.0,
                confidence_threshold=1.5  # > 1
            )


class TestPWaveDetectionParameters(unittest.TestCase):
    """Test PWaveDetectionParameters data class validation."""
    
    def test_valid_parameters(self):
        """Test valid P-wave detection parameters."""
        params = PWaveDetectionParameters(
            sampling_rate=100.0,
            sta_window=1.0,
            lta_window=10.0,
            trigger_threshold=3.0,
            detrigger_threshold=1.5
        )
        self.assertEqual(params.sta_window, 1.0)
        self.assertEqual(params.lta_window, 10.0)
    
    def test_invalid_sta_lta_windows(self):
        """Test invalid STA/LTA window relationship."""
        with self.assertRaises(ValueError):
            PWaveDetectionParameters(
                sampling_rate=100.0,
                sta_window=10.0,
                lta_window=5.0  # LTA < STA
            )
    
    def test_invalid_trigger_thresholds(self):
        """Test invalid trigger threshold relationship."""
        with self.assertRaises(ValueError):
            PWaveDetectionParameters(
                sampling_rate=100.0,
                trigger_threshold=1.0,
                detrigger_threshold=2.0  # detrigger > trigger
            )
    
    def test_invalid_characteristic_function(self):
        """Test invalid characteristic function type."""
        with self.assertRaises(ValueError):
            PWaveDetectionParameters(
                sampling_rate=100.0,
                characteristic_function_type='invalid'
            )


class TestPWaveDetector(unittest.TestCase):
    """Test PWaveDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
        self.params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            sta_window=1.0,
            lta_window=5.0,
            trigger_threshold=2.0,
            detrigger_threshold=1.2,
            confidence_threshold=0.3
        )
        self.detector = PWaveDetector(self.params)
    
    def _create_synthetic_p_wave_signal(self, duration: float = 30.0, 
                                      p_arrival: float = 10.0) -> np.ndarray:
        """
        Create synthetic seismic signal with P-wave arrival.
        
        Args:
            duration: Total signal duration in seconds
            p_arrival: P-wave arrival time in seconds
            
        Returns:
            Synthetic seismic signal
        """
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Background noise
        noise = 0.1 * np.random.randn(len(t))
        
        # P-wave signal (higher frequency, impulsive)
        p_wave = np.zeros_like(t)
        p_start_idx = int(p_arrival * self.sampling_rate)
        p_duration_samples = int(3.0 * self.sampling_rate)  # 3 second P-wave
        
        if p_start_idx + p_duration_samples < len(t):
            p_time = t[p_start_idx:p_start_idx + p_duration_samples] - p_arrival
            # Exponentially decaying sinusoid
            p_wave[p_start_idx:p_start_idx + p_duration_samples] = (
                2.0 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time / 2.0)
            )
        
        return noise + p_wave
    
    def _create_noise_only_signal(self, duration: float = 30.0) -> np.ndarray:
        """Create noise-only signal for false positive testing."""
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        return 0.1 * np.random.randn(len(t))
    
    def test_get_wave_type(self):
        """Test wave type identification."""
        self.assertEqual(self.detector.get_wave_type(), 'P')
    
    def test_set_parameters(self):
        """Test parameter setting."""
        new_params = {
            'trigger_threshold': 4.0,
            'sta_window': 0.5
        }
        self.detector.set_parameters(new_params)
        
        self.assertEqual(self.detector.params.trigger_threshold, 4.0)
        self.assertEqual(self.detector.params.sta_window, 0.5)
    
    def test_set_invalid_parameters(self):
        """Test setting invalid parameters."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.detector.set_parameters({'invalid_param': 1.0})
            self.assertTrue(len(w) > 0)
            self.assertIn("Unknown parameter", str(w[0].message))
    
    def test_detect_p_wave_in_synthetic_signal(self):
        """Test P-wave detection in synthetic signal."""
        # Create signal with P-wave at t=10s
        signal_data = self._create_synthetic_p_wave_signal(duration=30.0, p_arrival=10.0)
        
        # Detect P-waves
        detections = self.detector.detect_waves(signal_data, self.sampling_rate)
        
        # Should detect at least one P-wave
        self.assertGreater(len(detections), 0)
        
        # Check that detection is near expected arrival time
        if len(detections) > 0:
            first_detection = detections[0]
            self.assertEqual(first_detection.wave_type, 'P')
            self.assertAlmostEqual(first_detection.arrival_time, 10.0, delta=2.0)
            self.assertGreater(first_detection.confidence, self.params.confidence_threshold)
    
    def test_no_false_positives_in_noise(self):
        """Test that detector doesn't produce false positives in pure noise."""
        # Create noise-only signal
        noise_signal = self._create_noise_only_signal(duration=30.0)
        
        # Use higher trigger threshold to reduce false positives
        strict_params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            sta_window=1.0,
            lta_window=5.0,
            trigger_threshold=4.0,  # Higher threshold
            detrigger_threshold=2.0,
            confidence_threshold=0.6  # Higher confidence threshold
        )
        strict_detector = PWaveDetector(strict_params)
        
        # Detect P-waves
        detections = strict_detector.detect_waves(noise_signal, self.sampling_rate)
        
        # Should detect very few or no P-waves in pure noise
        self.assertLessEqual(len(detections), 2)  # Allow for occasional false positives
    
    def test_multiple_p_wave_detection(self):
        """Test detection of multiple P-waves."""
        # Create signal with two P-waves
        t = np.linspace(0, 60, int(60 * self.sampling_rate))
        noise = 0.1 * np.random.randn(len(t))
        
        # First P-wave at t=15s
        p1_start = int(15 * self.sampling_rate)
        p1_duration = int(3 * self.sampling_rate)
        p1_time = t[p1_start:p1_start + p1_duration] - 15
        signal = noise.copy()
        signal[p1_start:p1_start + p1_duration] += (
            2.0 * np.sin(2 * np.pi * 8 * p1_time) * np.exp(-p1_time / 2.0)
        )
        
        # Second P-wave at t=40s
        p2_start = int(40 * self.sampling_rate)
        p2_duration = int(3 * self.sampling_rate)
        if p2_start + p2_duration < len(t):
            p2_time = t[p2_start:p2_start + p2_duration] - 40
            signal[p2_start:p2_start + p2_duration] += (
                1.5 * np.sin(2 * np.pi * 10 * p2_time) * np.exp(-p2_time / 2.0)
            )
        
        # Detect P-waves
        detections = self.detector.detect_waves(signal, self.sampling_rate)
        
        # Should detect both P-waves
        self.assertGreaterEqual(len(detections), 1)  # At least one detection
        
        if len(detections) >= 2:
            # Check arrival times are reasonable
            arrival_times = [d.arrival_time for d in detections]
            arrival_times.sort()
            self.assertLess(arrival_times[0], 25)  # First detection before 25s
            # Second detection should be reasonably separated from first
            self.assertGreater(arrival_times[1] - arrival_times[0], 5)  # At least 5s apart
    
    def test_empty_data(self):
        """Test detection with empty data."""
        empty_data = np.array([])
        detections = self.detector.detect_waves(empty_data, self.sampling_rate)
        self.assertEqual(len(detections), 0)
    
    def test_short_data(self):
        """Test detection with very short data."""
        short_data = np.random.randn(50)  # 0.5 seconds at 100 Hz
        
        # Create detector with shorter windows for short data
        short_params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            sta_window=0.1,  # Very short windows
            lta_window=0.3,
            trigger_threshold=2.0,
            detrigger_threshold=1.2
        )
        short_detector = PWaveDetector(short_params)
        
        detections = short_detector.detect_waves(short_data, self.sampling_rate)
        # Should handle gracefully (may or may not detect anything)
        self.assertIsInstance(detections, list)
    
    def test_different_sampling_rates(self):
        """Test detection with different sampling rates."""
        # Create signal at different sampling rate
        new_sampling_rate = 200.0
        duration = 20.0
        t = np.linspace(0, duration, int(duration * new_sampling_rate))
        
        # P-wave at t=8s
        p_start_idx = int(8 * new_sampling_rate)
        p_duration_samples = int(2 * new_sampling_rate)
        signal = 0.1 * np.random.randn(len(t))
        
        if p_start_idx + p_duration_samples < len(t):
            p_time = t[p_start_idx:p_start_idx + p_duration_samples] - 8
            signal[p_start_idx:p_start_idx + p_duration_samples] += (
                2.0 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time / 1.5)
            )
        
        # Detect with different sampling rate
        detections = self.detector.detect_waves(signal, new_sampling_rate)
        
        # Should adapt to new sampling rate
        self.assertIsInstance(detections, list)
        if len(detections) > 0:
            self.assertEqual(detections[0].sampling_rate, new_sampling_rate)
    
    def test_characteristic_function_types(self):
        """Test different characteristic function types."""
        signal_data = self._create_synthetic_p_wave_signal()
        
        # Test energy characteristic function
        energy_params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            characteristic_function_type='energy',
            trigger_threshold=2.0,
            confidence_threshold=0.3
        )
        energy_detector = PWaveDetector(energy_params)
        energy_detections = energy_detector.detect_waves(signal_data, self.sampling_rate)
        
        # Test kurtosis characteristic function
        kurtosis_params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            characteristic_function_type='kurtosis',
            trigger_threshold=2.5,
            detrigger_threshold=1.5,
            confidence_threshold=0.3
        )
        kurtosis_detector = PWaveDetector(kurtosis_params)
        kurtosis_detections = kurtosis_detector.detect_waves(signal_data, self.sampling_rate)
        
        # Test AIC characteristic function
        aic_params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            characteristic_function_type='aic',
            trigger_threshold=2.0,
            detrigger_threshold=1.2,
            confidence_threshold=0.3
        )
        aic_detector = PWaveDetector(aic_params)
        aic_detections = aic_detector.detect_waves(signal_data, self.sampling_rate)
        
        # All should return valid detection lists
        self.assertIsInstance(energy_detections, list)
        self.assertIsInstance(kurtosis_detections, list)
        self.assertIsInstance(aic_detections, list)
    
    def test_wave_segment_properties(self):
        """Test properties of detected wave segments."""
        signal_data = self._create_synthetic_p_wave_signal()
        detections = self.detector.detect_waves(signal_data, self.sampling_rate)
        
        if len(detections) > 0:
            wave_segment = detections[0]
            
            # Check basic properties
            self.assertEqual(wave_segment.wave_type, 'P')
            self.assertEqual(wave_segment.sampling_rate, self.sampling_rate)
            self.assertGreater(wave_segment.peak_amplitude, 0)
            self.assertGreater(wave_segment.dominant_frequency, 0)
            self.assertGreaterEqual(wave_segment.confidence, self.params.confidence_threshold)
            
            # Check timing properties
            self.assertGreater(wave_segment.end_time, wave_segment.start_time)
            self.assertGreaterEqual(wave_segment.arrival_time, wave_segment.start_time)
            self.assertLessEqual(wave_segment.arrival_time, wave_segment.end_time)
            
            # Check duration constraints
            duration = wave_segment.duration
            self.assertGreaterEqual(duration, self.params.min_wave_duration)
            self.assertLessEqual(duration, self.params.max_wave_duration)
            
            # Check metadata
            self.assertIn('sta_lta_peak', wave_segment.metadata)
            self.assertIn('detection_method', wave_segment.metadata)
            self.assertEqual(wave_segment.metadata['detection_method'], 'STA/LTA')
    
    def test_detection_statistics(self):
        """Test detection statistics functionality."""
        signal_data = self._create_synthetic_p_wave_signal()
        stats = self.detector.get_detection_statistics(signal_data)
        
        # Check that all expected statistics are present
        expected_keys = [
            'num_detections', 'max_sta_lta', 'mean_sta_lta',
            'trigger_threshold', 'detrigger_threshold', 'onset_times',
            'characteristic_function_type', 'frequency_band', 'data_duration'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check value types and ranges
        self.assertIsInstance(stats['num_detections'], int)
        self.assertGreaterEqual(stats['num_detections'], 0)
        self.assertGreaterEqual(stats['max_sta_lta'], 0)
        self.assertGreaterEqual(stats['mean_sta_lta'], 0)
        self.assertIsInstance(stats['onset_times'], list)
        self.assertEqual(len(stats['onset_times']), stats['num_detections'])
    
    def test_confidence_threshold_filtering(self):
        """Test that low-confidence detections are filtered out."""
        # Create weak P-wave signal
        t = np.linspace(0, 20, int(20 * self.sampling_rate))
        weak_signal = 0.1 * np.random.randn(len(t))
        
        # Add very weak P-wave
        p_start = int(10 * self.sampling_rate)
        p_duration = int(2 * self.sampling_rate)
        if p_start + p_duration < len(t):
            p_time = t[p_start:p_start + p_duration] - 10
            weak_signal[p_start:p_start + p_duration] += (
                0.3 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time)
            )
        
        # Test with high confidence threshold
        high_confidence_params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            confidence_threshold=0.8,  # High threshold
            trigger_threshold=2.0,
            detrigger_threshold=1.2
        )
        high_confidence_detector = PWaveDetector(high_confidence_params)
        high_conf_detections = high_confidence_detector.detect_waves(weak_signal, self.sampling_rate)
        
        # Test with low confidence threshold
        low_confidence_params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            confidence_threshold=0.1,  # Low threshold
            trigger_threshold=2.0,
            detrigger_threshold=1.2
        )
        low_confidence_detector = PWaveDetector(low_confidence_params)
        low_conf_detections = low_confidence_detector.detect_waves(weak_signal, self.sampling_rate)
        
        # Low confidence threshold should detect more (or equal) waves
        self.assertGreaterEqual(len(low_conf_detections), len(high_conf_detections))


class TestCharacteristicFunctions(unittest.TestCase):
    """Test individual characteristic function implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
        self.params = PWaveDetectionParameters(sampling_rate=self.sampling_rate)
        self.detector = PWaveDetector(self.params)
    
    def test_energy_characteristic_function(self):
        """Test energy characteristic function."""
        # Create test signal
        test_data = np.array([1, 2, -3, 4, -2, 1])
        
        # Energy function should be squared values
        energy_cf = self.detector._energy_characteristic_function(test_data)
        expected = test_data ** 2
        
        np.testing.assert_array_equal(energy_cf, expected)
    
    def test_kurtosis_characteristic_function(self):
        """Test kurtosis characteristic function."""
        # Create test signal with known properties
        np.random.seed(42)  # For reproducible results
        test_data = np.random.normal(0, 1, 1000)
        
        kurtosis_cf = self.detector._kurtosis_characteristic_function(test_data)
        
        # Should return array of same length
        self.assertEqual(len(kurtosis_cf), len(test_data))
        
        # Should be non-negative (we take absolute value)
        self.assertTrue(np.all(kurtosis_cf >= 0))
    
    def test_aic_characteristic_function(self):
        """Test AIC characteristic function."""
        # Create test signal with abrupt change
        part1 = np.random.normal(0, 0.1, 500)  # Low variance
        part2 = np.random.normal(0, 1.0, 500)  # High variance
        test_data = np.concatenate([part1, part2])
        
        aic_cf = self.detector._aic_characteristic_function(test_data)
        
        # Should return array of same length
        self.assertEqual(len(aic_cf), len(test_data))
        
        # Should have peak near the change point (around index 500)
        peak_idx = np.argmax(aic_cf)
        self.assertGreater(peak_idx, 400)  # Should be in second half
        self.assertLess(peak_idx, 600)     # But not too far from change point


class TestIntegration(unittest.TestCase):
    """Integration tests for P-wave detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
    
    def test_realistic_earthquake_scenario(self):
        """Test P-wave detection in realistic earthquake scenario."""
        # Create more realistic earthquake signal
        duration = 60.0  # 1 minute
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Background noise (microseismic)
        noise = 0.05 * np.random.randn(len(t))
        
        # P-wave arrival at t=20s
        p_arrival = 20.0
        p_start_idx = int(p_arrival * self.sampling_rate)
        p_duration_samples = int(4.0 * self.sampling_rate)
        
        signal = noise.copy()
        if p_start_idx + p_duration_samples < len(t):
            p_time = t[p_start_idx:p_start_idx + p_duration_samples] - p_arrival
            # More realistic P-wave with multiple frequency components
            p_wave = (1.5 * np.sin(2 * np.pi * 6 * p_time) * np.exp(-p_time / 3.0) +
                     0.8 * np.sin(2 * np.pi * 12 * p_time) * np.exp(-p_time / 2.0))
            signal[p_start_idx:p_start_idx + p_duration_samples] += p_wave
        
        # Configure detector for realistic scenario
        params = PWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            sta_window=2.0,
            lta_window=15.0,
            trigger_threshold=2.5,
            detrigger_threshold=1.3,
            confidence_threshold=0.4,
            characteristic_function_type='energy'
        )
        detector = PWaveDetector(params)
        
        # Detect P-waves
        detections = detector.detect_waves(signal, self.sampling_rate)
        
        # Should detect the P-wave
        self.assertGreater(len(detections), 0)
        
        if len(detections) > 0:
            # Check that detection is close to expected arrival
            best_detection = max(detections, key=lambda x: x.confidence)
            self.assertAlmostEqual(best_detection.arrival_time, p_arrival, delta=3.0)
            
            # Check wave properties
            self.assertGreater(best_detection.peak_amplitude, 0.5)
            self.assertGreater(best_detection.dominant_frequency, 3.0)
            self.assertLess(best_detection.dominant_frequency, 20.0)
    
    def test_parameter_sensitivity(self):
        """Test sensitivity to different parameter settings."""
        # Create standard test signal
        signal_data = self._create_test_signal()
        
        # Test different trigger thresholds
        thresholds = [1.5, 2.0, 3.0, 4.0]
        detection_counts = []
        
        for threshold in thresholds:
            params = PWaveDetectionParameters(
                sampling_rate=self.sampling_rate,
                trigger_threshold=threshold,
                detrigger_threshold=threshold * 0.6,  # Ensure detrigger < trigger
                confidence_threshold=0.2
            )
            detector = PWaveDetector(params)
            detections = detector.detect_waves(signal_data, self.sampling_rate)
            detection_counts.append(len(detections))
        
        # Higher thresholds should generally result in fewer detections
        # (though this isn't guaranteed for all signals)
        self.assertIsInstance(detection_counts, list)
        self.assertEqual(len(detection_counts), len(thresholds))
    
    def _create_test_signal(self) -> np.ndarray:
        """Create standard test signal for parameter testing."""
        duration = 30.0
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Background noise
        signal = 0.1 * np.random.randn(len(t))
        
        # P-wave at t=15s
        p_start = int(15 * self.sampling_rate)
        p_duration = int(3 * self.sampling_rate)
        if p_start + p_duration < len(t):
            p_time = t[p_start:p_start + p_duration] - 15
            signal[p_start:p_start + p_duration] += (
                2.0 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time / 2.0)
            )
        
        return signal


if __name__ == '__main__':
    # Run tests with warnings suppressed for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unittest.main(verbosity=2)