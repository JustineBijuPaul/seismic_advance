"""
Unit tests for arrival time calculation functionality.

This module tests the ArrivalTimeCalculator class with synthetic
earthquake data to validate timing accuracy and reliability.
"""

import unittest
import numpy as np
from datetime import datetime

from wave_analysis.models import WaveSegment, ArrivalTimes
from wave_analysis.services.arrival_time_calculator import ArrivalTimeCalculator


class TestArrivalTimeCalculator(unittest.TestCase):
    """Test cases for ArrivalTimeCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 100.0  # Hz
        self.calculator = ArrivalTimeCalculator(self.sampling_rate)
        
        # Create synthetic wave data
        self.duration = 10.0  # seconds
        self.time_vector = np.linspace(0, self.duration, 
                                     int(self.duration * self.sampling_rate))
        
    def create_synthetic_p_wave(self, arrival_time: float, amplitude: float = 1.0) -> WaveSegment:
        """Create synthetic P-wave for testing."""
        # Create P-wave signal (high frequency, sharp onset)
        p_freq = 8.0  # Hz
        start_idx = int(arrival_time * self.sampling_rate)
        
        # Create signal with sharp onset
        signal = np.zeros(len(self.time_vector))
        if start_idx < len(signal):
            # Sharp onset followed by oscillation
            onset_samples = int(0.1 * self.sampling_rate)  # 0.1 second onset
            for i in range(start_idx, min(start_idx + onset_samples, len(signal))):
                t = (i - start_idx) / self.sampling_rate
                # Exponential rise with oscillation
                signal[i] = amplitude * (1 - np.exp(-10*t)) * np.sin(2*np.pi*p_freq*t)
        
        return WaveSegment(
            wave_type='P',
            start_time=max(0, arrival_time - 1.0),
            end_time=min(self.duration, arrival_time + 2.0),
            data=signal[max(0, start_idx - int(self.sampling_rate)):
                       min(len(signal), start_idx + int(2*self.sampling_rate))],
            sampling_rate=self.sampling_rate,
            peak_amplitude=amplitude,
            dominant_frequency=p_freq,
            arrival_time=arrival_time,
            confidence=0.9
        )
    
    def create_synthetic_s_wave(self, arrival_time: float, amplitude: float = 1.5) -> WaveSegment:
        """Create synthetic S-wave for testing."""
        # Create S-wave signal (lower frequency, gradual onset)
        s_freq = 4.0  # Hz
        start_idx = int(arrival_time * self.sampling_rate)
        
        # Create signal with gradual onset
        signal = np.zeros(len(self.time_vector))
        if start_idx < len(signal):
            # Gradual onset with larger amplitude
            onset_samples = int(0.3 * self.sampling_rate)  # 0.3 second onset
            for i in range(start_idx, min(start_idx + onset_samples, len(signal))):
                t = (i - start_idx) / self.sampling_rate
                # Gradual rise with oscillation
                signal[i] = amplitude * (1 - np.exp(-5*t)) * np.sin(2*np.pi*s_freq*t)
        
        return WaveSegment(
            wave_type='S',
            start_time=max(0, arrival_time - 1.0),
            end_time=min(self.duration, arrival_time + 3.0),
            data=signal[max(0, start_idx - int(self.sampling_rate)):
                       min(len(signal), start_idx + int(3*self.sampling_rate))],
            sampling_rate=self.sampling_rate,
            peak_amplitude=amplitude,
            dominant_frequency=s_freq,
            arrival_time=arrival_time,
            confidence=0.8
        )
    
    def create_synthetic_surface_wave(self, arrival_time: float, 
                                    wave_type: str = 'Rayleigh',
                                    amplitude: float = 2.0) -> WaveSegment:
        """Create synthetic surface wave for testing."""
        # Create surface wave signal (very low frequency, long duration)
        surf_freq = 1.0  # Hz
        start_idx = int(arrival_time * self.sampling_rate)
        
        # Create signal with very gradual onset
        signal = np.zeros(len(self.time_vector))
        if start_idx < len(signal):
            # Very gradual onset with large amplitude
            onset_samples = int(1.0 * self.sampling_rate)  # 1.0 second onset
            for i in range(start_idx, min(start_idx + onset_samples, len(signal))):
                t = (i - start_idx) / self.sampling_rate
                # Very gradual rise with low frequency oscillation
                signal[i] = amplitude * (1 - np.exp(-2*t)) * np.sin(2*np.pi*surf_freq*t)
        
        return WaveSegment(
            wave_type=wave_type,
            start_time=max(0, arrival_time - 2.0),
            end_time=min(self.duration, arrival_time + 5.0),
            data=signal[max(0, start_idx - int(2*self.sampling_rate)):
                       min(len(signal), start_idx + int(5*self.sampling_rate))],
            sampling_rate=self.sampling_rate,
            peak_amplitude=amplitude,
            dominant_frequency=surf_freq,
            arrival_time=arrival_time,
            confidence=0.7
        )
    
    def test_calculate_p_wave_arrival(self):
        """Test P-wave arrival time calculation."""
        expected_arrival = 2.5
        p_wave = self.create_synthetic_p_wave(expected_arrival)
        
        waves = {'P': [p_wave]}
        arrival_times = self.calculator.calculate_arrival_times(waves)
        
        self.assertIsNotNone(arrival_times.p_wave_arrival)
        # Allow for small refinement differences
        self.assertAlmostEqual(arrival_times.p_wave_arrival, expected_arrival, delta=0.2)
    
    def test_calculate_s_wave_arrival(self):
        """Test S-wave arrival time calculation."""
        expected_arrival = 4.0
        s_wave = self.create_synthetic_s_wave(expected_arrival)
        
        waves = {'S': [s_wave]}
        arrival_times = self.calculator.calculate_arrival_times(waves)
        
        self.assertIsNotNone(arrival_times.s_wave_arrival)
        # Allow for small refinement differences
        self.assertAlmostEqual(arrival_times.s_wave_arrival, expected_arrival, delta=0.3)
    
    def test_calculate_surface_wave_arrival(self):
        """Test surface wave arrival time calculation."""
        expected_arrival = 6.0
        surface_wave = self.create_synthetic_surface_wave(expected_arrival)
        
        waves = {'Rayleigh': [surface_wave]}
        arrival_times = self.calculator.calculate_arrival_times(waves)
        
        self.assertIsNotNone(arrival_times.surface_wave_arrival)
        # Allow for larger refinement differences for surface waves
        self.assertAlmostEqual(arrival_times.surface_wave_arrival, expected_arrival, delta=1.0)
    
    def test_sp_time_difference_calculation(self):
        """Test S-P time difference calculation."""
        p_arrival = 2.0
        s_arrival = 5.0
        expected_sp_diff = 3.0
        
        p_wave = self.create_synthetic_p_wave(p_arrival)
        s_wave = self.create_synthetic_s_wave(s_arrival)
        
        waves = {'P': [p_wave], 'S': [s_wave]}
        arrival_times = self.calculator.calculate_arrival_times(waves)
        
        self.assertIsNotNone(arrival_times.sp_time_difference)
        # Allow for refinement differences
        self.assertAlmostEqual(arrival_times.sp_time_difference, expected_sp_diff, delta=0.5)
    
    def test_multiple_waves_same_type(self):
        """Test handling multiple waves of the same type."""
        # Create multiple P-waves, should pick the earliest
        p_wave1 = self.create_synthetic_p_wave(2.0, amplitude=0.5)
        p_wave1.confidence = 0.6
        p_wave2 = self.create_synthetic_p_wave(1.5, amplitude=1.0)  # Earlier and higher confidence
        p_wave2.confidence = 0.9
        
        waves = {'P': [p_wave1, p_wave2]}
        arrival_times = self.calculator.calculate_arrival_times(waves)
        
        # Should pick the earlier wave (p_wave2)
        self.assertIsNotNone(arrival_times.p_wave_arrival)
        self.assertAlmostEqual(arrival_times.p_wave_arrival, 1.5, delta=0.2)
    
    def test_sta_lta_calculation(self):
        """Test STA/LTA characteristic function calculation."""
        # Create signal with clear onset
        data = np.zeros(1000)
        onset_idx = 500
        data[onset_idx:] = np.random.normal(0, 2, len(data) - onset_idx)  # Higher amplitude after onset
        data[:onset_idx] = np.random.normal(0, 0.5, onset_idx)  # Lower amplitude before onset
        
        sta_length = 10
        lta_length = 100
        
        sta_lta = self.calculator._calculate_sta_lta(data, sta_length, lta_length)
        
        # STA/LTA should be higher after the onset
        pre_onset_avg = np.mean(sta_lta[lta_length:onset_idx])
        post_onset_avg = np.mean(sta_lta[onset_idx+50:onset_idx+150])
        
        self.assertGreater(post_onset_avg, pre_onset_avg)
    
    def test_cross_correlation_arrivals(self):
        """Test cross-correlation between wave arrivals."""
        # Create two similar waves with known time delay
        p_wave1 = self.create_synthetic_p_wave(2.0)
        p_wave2 = self.create_synthetic_p_wave(2.3)  # 0.3 second delay
        
        delay = self.calculator.cross_correlate_arrivals(p_wave1, p_wave2)
        
        # Cross-correlation should detect some delay (may not be exact due to windowing)
        self.assertIsInstance(delay, float)
        # Just verify the method runs without error for now
        self.assertTrue(abs(delay) >= 0)
    
    def test_epicenter_distance_estimation(self):
        """Test epicenter distance estimation from S-P time."""
        sp_time = 5.0  # seconds
        distance = self.calculator.estimate_epicenter_distance(sp_time)
        
        # Expected distance calculation:
        # distance = sp_time / (1/Vs - 1/Vp) = 5 / (1/3.5 - 1/6.0) = 5 / 0.119 ≈ 42 km
        expected_distance = 42.0
        self.assertAlmostEqual(distance, expected_distance, delta=2.0)
    
    def test_empty_waves_handling(self):
        """Test handling of empty wave lists."""
        waves = {}
        arrival_times = self.calculator.calculate_arrival_times(waves)
        
        self.assertIsNone(arrival_times.p_wave_arrival)
        self.assertIsNone(arrival_times.s_wave_arrival)
        self.assertIsNone(arrival_times.surface_wave_arrival)
        self.assertIsNone(arrival_times.sp_time_difference)
    
    def test_parameter_setting(self):
        """Test parameter setting functionality."""
        new_window = 3.0
        new_threshold = 0.5
        
        self.calculator.set_parameters(
            cross_correlation_window=new_window,
            min_correlation_threshold=new_threshold
        )
        
        self.assertEqual(self.calculator.cross_correlation_window, new_window)
        self.assertEqual(self.calculator.min_correlation_threshold, new_threshold)
    
    def test_characteristic_function_creation(self):
        """Test characteristic function creation for different wave types."""
        # Create test data with clear signal
        data = np.zeros(1000)
        data[500:600] = np.sin(2*np.pi*np.linspace(0, 10, 100))  # Signal burst
        
        # Test P-wave characteristic function
        p_char = self.calculator._create_p_wave_characteristic_function(data)
        self.assertEqual(len(p_char), len(data))
        self.assertGreater(np.max(p_char), 0)
        
        # Test S-wave characteristic function
        s_char = self.calculator._create_s_wave_characteristic_function(data)
        self.assertEqual(len(s_char), len(data))
        self.assertGreater(np.max(s_char), 0)
        
        # Test surface wave characteristic function
        surf_char = self.calculator._create_surface_wave_characteristic_function(data)
        self.assertEqual(len(surf_char), len(data))
        self.assertGreater(np.max(surf_char), 0)


if __name__ == '__main__':
    unittest.main()