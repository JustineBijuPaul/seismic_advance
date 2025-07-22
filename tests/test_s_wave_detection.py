"""
Unit tests for S-wave detection algorithm.

This module tests the SWaveDetector class with various scenarios including
known S-P time differences, polarization analysis, and particle motion detection.
"""

import unittest
import numpy as np
from typing import List, Dict, Any
import warnings

# Import the classes we're testing
from wave_analysis.services.wave_detectors import SWaveDetector, SWaveDetectionParameters
from wave_analysis.models import WaveSegment, ArrivalTimes


class TestSWaveDetection(unittest.TestCase):
    """Test cases for S-wave detection algorithm."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sampling_rate = 100.0  # 100 Hz sampling rate
        self.default_params = SWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            sta_window=2.0,
            lta_window=15.0,
            trigger_threshold=2.5,
            detrigger_threshold=1.3,
            frequency_band=(0.5, 10.0),
            polarization_window=1.0,
            amplitude_ratio_threshold=1.5,
            p_wave_context_window=5.0,
            min_wave_duration=0.5,
            max_wave_duration=30.0,
            confidence_threshold=0.5
        )
        self.detector = SWaveDetector(self.default_params)
    
    def test_detector_initialization(self):
        """Test proper initialization of S-wave detector."""
        self.assertEqual(self.detector.get_wave_type(), 'S')
        self.assertEqual(self.detector.params.sampling_rate, self.sampling_rate)
        
    def test_invalid_sta_lta_windows(self):
        """Test invalid STA/LTA window relationship."""
        with self.assertRaises(ValueError):
            SWaveDetectionParameters(
                sampling_rate=100.0,
                sta_window=15.0,
                lta_window=10.0  # LTA < STA
            )
    
    def test_invalid_trigger_thresholds(self):
        """Test invalid trigger threshold relationship."""
        with self.assertRaises(ValueError):
            SWaveDetectionParameters(
                sampling_rate=100.0,
                trigger_threshold=1.0,
                detrigger_threshold=2.0  # detrigger > trigger
            )
    
    def test_invalid_polarization_window(self):
        """Test invalid polarization window."""
        with self.assertRaises(ValueError):
            SWaveDetectionParameters(
                sampling_rate=100.0,
                polarization_window=-1.0
            )
    
    def test_invalid_amplitude_ratio_threshold(self):
        """Test invalid amplitude ratio threshold."""
        with self.assertRaises(ValueError):
            SWaveDetectionParameters(
                sampling_rate=100.0,
                amplitude_ratio_threshold=-1.0
            )


class TestSWaveDetector(unittest.TestCase):
    """Test SWaveDetector functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
        self.params = SWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            sta_window=2.0,
            lta_window=10.0,
            trigger_threshold=2.0,
            detrigger_threshold=1.2,
            confidence_threshold=0.3
        )
        self.detector = SWaveDetector(self.params)
    
    def _create_synthetic_s_wave_signal(self, duration: float = 40.0, 
                                      p_arrival: float = 10.0,
                                      s_arrival: float = 18.0) -> np.ndarray:
        """
        Create synthetic seismic signal with P-wave and S-wave arrivals.
        
        Args:
            duration: Total signal duration in seconds
            p_arrival: P-wave arrival time in seconds
            s_arrival: S-wave arrival time in seconds
            
        Returns:
            Synthetic seismic signal
        """
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Background noise
        noise = 0.1 * np.random.randn(len(t))
        
        # P-wave signal (higher frequency, shorter duration)
        p_wave = np.zeros_like(t)
        p_start_idx = int(p_arrival * self.sampling_rate)
        p_duration_samples = int(3.0 * self.sampling_rate)  # 3 second P-wave
        
        if p_start_idx + p_duration_samples < len(t):
            p_time = t[p_start_idx:p_start_idx + p_duration_samples] - p_arrival
            p_wave[p_start_idx:p_start_idx + p_duration_samples] = (
                1.5 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time / 2.0)
            )
        
        # S-wave signal (lower frequency, longer duration, higher amplitude)
        s_wave = np.zeros_like(t)
        s_start_idx = int(s_arrival * self.sampling_rate)
        s_duration_samples = int(6.0 * self.sampling_rate)  # 6 second S-wave
        
        if s_start_idx + s_duration_samples < len(t):
            s_time = t[s_start_idx:s_start_idx + s_duration_samples] - s_arrival
            s_wave[s_start_idx:s_start_idx + s_duration_samples] = (
                2.5 * np.sin(2 * np.pi * 4 * s_time) * np.exp(-s_time / 4.0)
            )
        
        return noise + p_wave + s_wave
    
    def _create_multi_channel_signal(self, duration: float = 40.0,
                                   p_arrival: float = 10.0,
                                   s_arrival: float = 18.0) -> np.ndarray:
        """Create multi-channel synthetic signal for polarization testing."""
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        n_channels = 3
        
        # Initialize multi-channel data
        multi_channel_data = np.zeros((n_channels, len(t)))
        
        # Add noise to all channels
        for i in range(n_channels):
            multi_channel_data[i] = 0.1 * np.random.randn(len(t))
        
        # Add P-wave (more linear polarization)
        p_start_idx = int(p_arrival * self.sampling_rate)
        p_duration_samples = int(3.0 * self.sampling_rate)
        
        if p_start_idx + p_duration_samples < len(t):
            p_time = t[p_start_idx:p_start_idx + p_duration_samples] - p_arrival
            p_signal = 1.5 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time / 2.0)
            
            # P-wave primarily on vertical component (channel 0)
            multi_channel_data[0, p_start_idx:p_start_idx + p_duration_samples] += p_signal
            # Some energy on horizontal components
            multi_channel_data[1, p_start_idx:p_start_idx + p_duration_samples] += 0.3 * p_signal
            multi_channel_data[2, p_start_idx:p_start_idx + p_duration_samples] += 0.2 * p_signal
        
        # Add S-wave (more complex polarization)
        s_start_idx = int(s_arrival * self.sampling_rate)
        s_duration_samples = int(6.0 * self.sampling_rate)
        
        if s_start_idx + s_duration_samples < len(t):
            s_time = t[s_start_idx:s_start_idx + s_duration_samples] - s_arrival
            s_signal_base = 2.5 * np.sin(2 * np.pi * 4 * s_time) * np.exp(-s_time / 4.0)
            
            # S-wave with different phases on different components (shear motion)
            multi_channel_data[0, s_start_idx:s_start_idx + s_duration_samples] += 0.5 * s_signal_base
            multi_channel_data[1, s_start_idx:s_start_idx + s_duration_samples] += s_signal_base
            multi_channel_data[2, s_start_idx:s_start_idx + s_duration_samples] += (
                0.8 * np.sin(2 * np.pi * 4 * s_time + np.pi/4) * np.exp(-s_time / 4.0)
            )
        
        return multi_channel_data
    
    def test_get_wave_type(self):
        """Test wave type identification."""
        self.assertEqual(self.detector.get_wave_type(), 'S')
    
    def test_set_parameters(self):
        """Test parameter setting."""
        new_params = {
            'trigger_threshold': 3.0,
            'sta_window': 1.5
        }
        self.detector.set_parameters(new_params)
        
        self.assertEqual(self.detector.params.trigger_threshold, 3.0)
        self.assertEqual(self.detector.params.sta_window, 1.5)
    
    def test_set_invalid_parameters(self):
        """Test setting invalid parameters."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.detector.set_parameters({'invalid_param': 1.0})
            self.assertTrue(len(w) > 0)
            self.assertIn("Unknown parameter", str(w[0].message))
    
    def test_detect_s_wave_single_channel(self):
        """Test S-wave detection in single-channel signal."""
        # Create signal with P-wave and S-wave
        signal_data = self._create_synthetic_s_wave_signal(duration=40.0, p_arrival=10.0, s_arrival=18.0)
        
        # Provide P-wave context
        metadata = {
            'p_wave_arrivals': [10.0]
        }
        
        # Detect S-waves
        detections = self.detector.detect_waves(signal_data, self.sampling_rate, metadata)
        
        # Should detect at least one S-wave
        self.assertGreater(len(detections), 0)
        
        # Check that detection is near expected arrival time
        if len(detections) > 0:
            first_detection = detections[0]
            self.assertEqual(first_detection.wave_type, 'S')
            # S-wave should arrive after P-wave (10s) but timing may vary due to detection algorithm
            self.assertGreater(first_detection.arrival_time, 12.0)  # After P-wave + exclusion
            self.assertLess(first_detection.arrival_time, 25.0)     # Within reasonable range
            self.assertGreater(first_detection.confidence, self.params.confidence_threshold)
    
    def test_detect_s_wave_multi_channel(self):
        """Test S-wave detection in multi-channel signal."""
        # Create multi-channel signal
        multi_channel_data = self._create_multi_channel_signal(duration=40.0, p_arrival=10.0, s_arrival=18.0)
        
        # Provide P-wave context
        metadata = {
            'p_wave_arrivals': [10.0]
        }
        
        # Detect S-waves
        detections = self.detector.detect_waves(multi_channel_data, self.sampling_rate, metadata)
        
        # Should detect S-wave using polarization analysis
        self.assertGreater(len(detections), 0)
        
        if len(detections) > 0:
            first_detection = detections[0]
            self.assertEqual(first_detection.wave_type, 'S')
            # S-wave should arrive after P-wave (10s) but timing may vary due to detection algorithm
            self.assertGreater(first_detection.arrival_time, 12.0)  # After P-wave + exclusion
            self.assertLess(first_detection.arrival_time, 25.0)     # Within reasonable range
            
            # Check that multi-channel detection method is recorded
            self.assertIn('multi_channel', first_detection.metadata['detection_method'])
    
    def test_empty_data(self):
        """Test detection with empty data."""
        empty_data = np.array([])
        detections = self.detector.detect_waves(empty_data, self.sampling_rate)
        self.assertEqual(len(detections), 0)
    
    def test_short_data(self):
        """Test detection with very short data."""
        short_data = np.random.randn(100)  # 1 second at 100 Hz
        
        # Create detector with shorter windows for short data
        short_params = SWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            sta_window=0.2,
            lta_window=0.8,
            trigger_threshold=2.0,
            detrigger_threshold=1.2
        )
        short_detector = SWaveDetector(short_params)
        
        detections = short_detector.detect_waves(short_data, self.sampling_rate)
        # Should handle gracefully
        self.assertIsInstance(detections, list)
    
    def test_different_sampling_rates(self):
        """Test detection with different sampling rates."""
        # Create signal at different sampling rate
        new_sampling_rate = 200.0
        duration = 30.0
        t = np.linspace(0, duration, int(duration * new_sampling_rate))
        
        # S-wave at t=15s
        s_start_idx = int(15 * new_sampling_rate)
        s_duration_samples = int(4 * new_sampling_rate)
        signal = 0.1 * np.random.randn(len(t))
        
        if s_start_idx + s_duration_samples < len(t):
            s_time = t[s_start_idx:s_start_idx + s_duration_samples] - 15
            signal[s_start_idx:s_start_idx + s_duration_samples] += (
                2.0 * np.sin(2 * np.pi * 4 * s_time) * np.exp(-s_time / 3.0)
            )
        
        # Detect with different sampling rate
        detections = self.detector.detect_waves(signal, new_sampling_rate)
        
        # Should adapt to new sampling rate
        self.assertIsInstance(detections, list)
        if len(detections) > 0:
            self.assertEqual(detections[0].sampling_rate, new_sampling_rate)
    
    def test_p_wave_context_extraction(self):
        """Test extraction of P-wave context from metadata."""
        # Test different metadata formats
        metadata_formats = [
            {'p_wave_arrivals': [5.0, 15.0]},
            {'p_waves': [{'arrival_time': 5.0}, {'arrival_time': 15.0}]},
            {}  # No P-wave context
        ]
        
        for metadata in metadata_formats:
            p_times = self.detector._extract_p_wave_times(metadata)
            
            if 'p_wave_arrivals' in metadata:
                self.assertEqual(p_times, [5.0, 15.0])
            elif 'p_waves' in metadata:
                self.assertEqual(p_times, [5.0, 15.0])
            else:
                self.assertEqual(p_times, [])
    
    def test_wave_segment_properties(self):
        """Test properties of detected S-wave segments."""
        signal_data = self._create_synthetic_s_wave_signal()
        metadata = {'p_wave_arrivals': [10.0]}
        
        detections = self.detector.detect_waves(signal_data, self.sampling_rate, metadata)
        
        if len(detections) > 0:
            wave_segment = detections[0]
            
            # Check basic properties
            self.assertEqual(wave_segment.wave_type, 'S')
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
            self.assertIn('S-wave', wave_segment.metadata['detection_method'])
    
    def test_detection_statistics(self):
        """Test detection statistics functionality."""
        signal_data = self._create_synthetic_s_wave_signal()
        metadata = {'p_wave_arrivals': [10.0]}
        
        stats = self.detector.get_detection_statistics(signal_data, metadata)
        
        # Check that all expected statistics are present
        expected_keys = [
            'num_detections', 'trigger_threshold', 'detrigger_threshold',
            'onset_times', 'frequency_band', 'data_duration', 'detection_method'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check P-wave context statistics
        self.assertIn('p_wave_context', stats)
        self.assertIn('num_p_waves', stats)
        self.assertEqual(stats['num_p_waves'], 1)
        
        # Check S-P time differences if S-waves detected
        if stats['num_detections'] > 0:
            # S-P time differences should be present (may be empty list if no valid pairs)
            self.assertIn('sp_time_differences', stats)
            if stats['sp_time_differences']:  # Only check mean if there are valid differences
                self.assertIn('mean_sp_time', stats)
    
    def test_s_wave_without_p_wave_context(self):
        """Test S-wave detection without P-wave context."""
        signal_data = self._create_synthetic_s_wave_signal()
        
        # No P-wave context provided
        detections = self.detector.detect_waves(signal_data, self.sampling_rate)
        
        # Should still be able to detect S-waves (though may be less accurate)
        self.assertIsInstance(detections, list)
        # May or may not detect depending on signal characteristics
    
    def test_confidence_threshold_filtering(self):
        """Test that low-confidence detections are filtered out."""
        # Create weak S-wave signal
        t = np.linspace(0, 30, int(30 * self.sampling_rate))
        weak_signal = 0.1 * np.random.randn(len(t))
        
        # Add very weak S-wave
        s_start = int(15 * self.sampling_rate)
        s_duration = int(3 * self.sampling_rate)
        if s_start + s_duration < len(t):
            s_time = t[s_start:s_start + s_duration] - 15
            weak_signal[s_start:s_start + s_duration] += (
                0.4 * np.sin(2 * np.pi * 4 * s_time) * np.exp(-s_time / 2)
            )
        
        # Test with high confidence threshold
        high_confidence_params = SWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            confidence_threshold=0.8,  # High threshold
            trigger_threshold=2.0,
            detrigger_threshold=1.2
        )
        high_confidence_detector = SWaveDetector(high_confidence_params)
        high_conf_detections = high_confidence_detector.detect_waves(weak_signal, self.sampling_rate)
        
        # Test with low confidence threshold
        low_confidence_params = SWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            confidence_threshold=0.1,  # Low threshold
            trigger_threshold=2.0,
            detrigger_threshold=1.2
        )
        low_confidence_detector = SWaveDetector(low_confidence_params)
        low_conf_detections = low_confidence_detector.detect_waves(weak_signal, self.sampling_rate)
        
        # Low confidence threshold should detect more (or equal) waves
        self.assertGreaterEqual(len(low_conf_detections), len(high_conf_detections))


class TestPolarizationAnalysis(unittest.TestCase):
    """Test polarization analysis components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
        self.params = SWaveDetectionParameters(sampling_rate=self.sampling_rate)
        self.detector = SWaveDetector(self.params)
    
    def test_polarization_attributes_calculation(self):
        """Test polarization attributes calculation."""
        # Create test multi-channel data
        n_channels = 3
        n_samples = 1000
        
        # Create data with known polarization characteristics
        t = np.linspace(0, 10, n_samples)
        multi_channel_data = np.zeros((n_channels, n_samples))
        
        # Linear polarization (P-wave like)
        linear_signal = np.sin(2 * np.pi * 5 * t)
        multi_channel_data[0, :] = linear_signal
        multi_channel_data[1, :] = 0.3 * linear_signal  # Some coupling
        multi_channel_data[2, :] = 0.1 * linear_signal
        
        polarization = self.detector._calculate_polarization_attributes(multi_channel_data)
        
        # Should return array of same length as input
        self.assertEqual(len(polarization), n_samples)
        
        # Should be non-negative
        self.assertTrue(np.all(polarization >= 0))
    
    def test_particle_motion_calculation(self):
        """Test particle motion calculation."""
        # Create test multi-channel data
        n_channels = 3
        n_samples = 1000
        
        # Create data with complex particle motion (S-wave like)
        t = np.linspace(0, 10, n_samples)
        multi_channel_data = np.zeros((n_channels, n_samples))
        
        # Complex motion with phase differences
        multi_channel_data[0, :] = np.sin(2 * np.pi * 3 * t)
        multi_channel_data[1, :] = np.sin(2 * np.pi * 3 * t + np.pi/4)
        multi_channel_data[2, :] = np.sin(2 * np.pi * 3 * t + np.pi/2)
        
        particle_motion = self.detector._calculate_particle_motion(multi_channel_data)
        
        # Should return array of same length as input
        self.assertEqual(len(particle_motion), n_samples)
        
        # Should be non-negative
        self.assertTrue(np.all(particle_motion >= 0))
    
    def test_s_wave_characteristics_combination(self):
        """Test combination of S-wave characteristics."""
        # Create test polarization and particle motion arrays
        n_samples = 1000
        polarization = np.random.rand(n_samples) * 10  # High polarization values
        particle_motion = np.random.rand(n_samples) * 2  # Complex motion values
        
        combined = self.detector._combine_s_wave_characteristics(polarization, particle_motion)
        
        # Should return array of same length
        self.assertEqual(len(combined), n_samples)
        
        # Should be normalized (approximately between 0 and 1)
        self.assertTrue(np.all(combined >= 0))
        self.assertTrue(np.all(combined <= 1.1))  # Allow small numerical errors


class TestSWaveBoundaryDetection(unittest.TestCase):
    """Test S-wave boundary detection methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
        self.params = SWaveDetectionParameters(sampling_rate=self.sampling_rate)
        self.detector = SWaveDetector(self.params)
    
    def test_s_wave_boundary_determination(self):
        """Test S-wave boundary determination."""
        # Create test data with known S-wave
        duration = 30.0
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        data = 0.1 * np.random.randn(len(t))
        
        # Add S-wave at specific location
        s_start = int(10 * self.sampling_rate)
        s_duration = int(5 * self.sampling_rate)
        s_time = t[s_start:s_start + s_duration] - 10
        data[s_start:s_start + s_duration] += (
            2.0 * np.sin(2 * np.pi * 4 * s_time) * np.exp(-s_time / 3.0)
        )
        
        # Create mock STA/LTA with peak at S-wave onset
        sta_lta = np.ones(len(data))
        onset_idx = s_start + int(0.5 * self.sampling_rate)  # Slightly after start
        sta_lta[onset_idx] = 5.0  # High value at onset
        
        # Test boundary determination
        start_idx, end_idx = self.detector._determine_s_wave_boundaries(data, onset_idx, sta_lta)
        
        # Boundaries should be reasonable
        self.assertLess(start_idx, onset_idx)
        self.assertGreater(end_idx, onset_idx)
        self.assertGreater(end_idx - start_idx, self.params.min_wave_duration * self.sampling_rate)
        self.assertLess(end_idx - start_idx, self.params.max_wave_duration * self.sampling_rate)
    
    def test_s_wave_onset_refinement(self):
        """Test S-wave onset time refinement."""
        # Create test STA/LTA with gradual increase
        n_samples = 1000
        sta_lta = np.ones(n_samples)
        
        # Create gradual increase typical of S-wave
        peak_idx = 500
        for i in range(peak_idx - 100, peak_idx + 1):
            if i >= 0:
                progress = (i - (peak_idx - 100)) / 100
                sta_lta[i] = 1.0 + 4.0 * progress  # Gradual increase to 5.0
        
        refined_onset = self.detector._refine_s_wave_onset(sta_lta, peak_idx, [])
        
        # Refined onset should be before the peak
        self.assertLess(refined_onset, peak_idx)
        self.assertGreaterEqual(refined_onset, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for S-wave detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0
    
    def test_realistic_earthquake_scenario(self):
        """Test S-wave detection in realistic earthquake scenario."""
        # Create more realistic earthquake signal
        duration = 60.0  # 1 minute
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Background noise
        noise = 0.05 * np.random.randn(len(t))
        
        # P-wave arrival at t=15s
        p_arrival = 15.0
        p_start_idx = int(p_arrival * self.sampling_rate)
        p_duration_samples = int(3.0 * self.sampling_rate)
        
        signal = noise.copy()
        if p_start_idx + p_duration_samples < len(t):
            p_time = t[p_start_idx:p_start_idx + p_duration_samples] - p_arrival
            p_wave = 1.2 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time / 2.5)
            signal[p_start_idx:p_start_idx + p_duration_samples] += p_wave
        
        # S-wave arrival at t=27s (realistic S-P time)
        s_arrival = 27.0
        s_start_idx = int(s_arrival * self.sampling_rate)
        s_duration_samples = int(8.0 * self.sampling_rate)
        
        if s_start_idx + s_duration_samples < len(t):
            s_time = t[s_start_idx:s_start_idx + s_duration_samples] - s_arrival
            s_wave = 2.0 * np.sin(2 * np.pi * 4 * s_time) * np.exp(-s_time / 5.0)
            signal[s_start_idx:s_start_idx + s_duration_samples] += s_wave
        
        # Configure detector for realistic scenario
        params = SWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            sta_window=3.0,
            lta_window=20.0,
            trigger_threshold=2.0,
            detrigger_threshold=1.3,
            confidence_threshold=0.4
        )
        detector = SWaveDetector(params)
        
        # Provide P-wave context
        metadata = {'p_wave_arrivals': [p_arrival]}
        
        # Detect S-waves
        detections = detector.detect_waves(signal, self.sampling_rate, metadata)
        
        # Should detect the S-wave
        self.assertGreater(len(detections), 0)
        
        if len(detections) > 0:
            # Check that detection is close to expected arrival
            best_detection = max(detections, key=lambda x: x.confidence)
            self.assertAlmostEqual(best_detection.arrival_time, s_arrival, delta=4.0)
            
            # Check wave properties
            self.assertGreater(best_detection.peak_amplitude, 0.5)
            self.assertGreater(best_detection.dominant_frequency, 1.0)
            self.assertLess(best_detection.dominant_frequency, 15.0)
            
            # Check S-P time
            sp_time = best_detection.arrival_time - p_arrival
            self.assertGreater(sp_time, 5.0)  # Reasonable S-P time
            self.assertLess(sp_time, 20.0)
    
    def test_multi_channel_vs_single_channel(self):
        """Test comparison between multi-channel and single-channel detection."""
        # Create single-channel signal
        single_channel = self._create_test_signal()
        
        # Create multi-channel version
        multi_channel = np.zeros((3, len(single_channel)))
        multi_channel[0] = single_channel
        multi_channel[1] = single_channel * 0.8 + 0.1 * np.random.randn(len(single_channel))
        multi_channel[2] = single_channel * 0.6 + 0.1 * np.random.randn(len(single_channel))
        
        # Test both detection methods
        params = SWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            confidence_threshold=0.3
        )
        detector = SWaveDetector(params)
        
        metadata = {'p_wave_arrivals': [10.0]}
        
        single_detections = detector.detect_waves(single_channel, self.sampling_rate, metadata)
        multi_detections = detector.detect_waves(multi_channel, self.sampling_rate, metadata)
        
        # Both should return valid detection lists
        self.assertIsInstance(single_detections, list)
        self.assertIsInstance(multi_detections, list)
        
        # Check detection method is recorded correctly
        if single_detections:
            self.assertIn('single_channel', single_detections[0].metadata['detection_method'])
        if multi_detections:
            self.assertIn('multi_channel', multi_detections[0].metadata['detection_method'])
    
    def _create_test_signal(self) -> np.ndarray:
        """Create standard test signal."""
        duration = 40.0
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Background noise
        signal = 0.1 * np.random.randn(len(t))
        
        # P-wave at t=10s
        p_start = int(10 * self.sampling_rate)
        p_duration = int(3 * self.sampling_rate)
        if p_start + p_duration < len(t):
            p_time = t[p_start:p_start + p_duration] - 10
            signal[p_start:p_start + p_duration] += (
                1.5 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time / 2.0)
            )
        
        # S-wave at t=18s
        s_start = int(18 * self.sampling_rate)
        s_duration = int(6 * self.sampling_rate)
        if s_start + s_duration < len(t):
            s_time = t[s_start:s_start + s_duration] - 18
            signal[s_start:s_start + s_duration] += (
                2.0 * np.sin(2 * np.pi * 4 * s_time) * np.exp(-s_time / 4.0)
            )
        
        return signal


if __name__ == '__main__':
    # Run tests with warnings suppressed for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unittest.main(verbosity=2)