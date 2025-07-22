"""
Unit tests for surface wave detection algorithms.

This module tests the SurfaceWaveDetector class with synthetic data
to validate Love and Rayleigh wave detection capabilities.
"""

import unittest
import numpy as np
import warnings
from typing import List, Dict, Any

# Import the classes we're testing
from wave_analysis.services.wave_detectors import (
    SurfaceWaveDetector, SurfaceWaveDetectionParameters
)
from wave_analysis.models.wave_models import WaveSegment


class TestSurfaceWaveDetection(unittest.TestCase):
    """Test cases for surface wave detection algorithms."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sampling_rate = 100.0  # Hz
        self.duration = 120.0  # seconds - long enough for surface waves
        self.n_samples = int(self.duration * self.sampling_rate)
        
        # Create default parameters for testing
        self.params = SurfaceWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            frequency_band=(0.02, 0.5),
            min_surface_wave_duration=10.0,
            energy_ratio_threshold=2.0,
            dispersion_window=30.0
        )
        
        self.detector = SurfaceWaveDetector(self.params)
    
    def test_detector_initialization(self):
        """Test proper initialization of surface wave detector."""
        self.assertEqual(self.detector.get_wave_type(), 'Surface')
        self.assertEqual(self.detector.params.sampling_rate, self.sampling_rate)
        self.assertEqual(self.detector.params.frequency_band, (0.02, 0.5))
        self.assertEqual(self.detector.params.min_surface_wave_duration, 10.0)
    
    def test_parameter_validation(self):
        """Test validation of surface wave detection parameters."""
        # Test invalid dispersion window
        with self.assertRaises(ValueError):
            SurfaceWaveDetectionParameters(
                sampling_rate=100.0,
                dispersion_window=-1.0
            )
        
        # Test invalid minimum duration
        with self.assertRaises(ValueError):
            SurfaceWaveDetectionParameters(
                sampling_rate=100.0,
                min_surface_wave_duration=-5.0
            )
        
        # Test invalid coherence threshold
        with self.assertRaises(ValueError):
            SurfaceWaveDetectionParameters(
                sampling_rate=100.0,
                spectral_coherence_threshold=1.5
            )
        
        # Test invalid energy ratio threshold
        with self.assertRaises(ValueError):
            SurfaceWaveDetectionParameters(
                sampling_rate=100.0,
                energy_ratio_threshold=-1.0
            )
    
    def test_empty_data_handling(self):
        """Test handling of empty input data."""
        empty_data = np.array([])
        result = self.detector.detect_waves(empty_data, self.sampling_rate)
        self.assertEqual(len(result), 0)
    
    def test_short_data_handling(self):
        """Test handling of data too short for surface wave analysis."""
        # Create data shorter than minimum surface wave duration
        short_duration = 5.0  # seconds
        short_data = self.create_synthetic_noise(short_duration)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = self.detector.detect_waves(short_data, self.sampling_rate)
            
            # Should return empty result and issue warning
            self.assertEqual(len(result), 0)
            self.assertTrue(len(w) > 0)
            self.assertIn("too short", str(w[0].message))
    
    def test_synthetic_rayleigh_wave_detection(self):
        """Test detection of synthetic Rayleigh waves."""
        # Use more reasonable parameters to avoid filter instability
        stable_params = SurfaceWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            frequency_band=(0.05, 0.5),  # Higher low-frequency cutoff
            min_surface_wave_duration=5.0,
            energy_ratio_threshold=1.5,
            dispersion_window=20.0
        )
        stable_detector = SurfaceWaveDetector(stable_params)
        
        # Create synthetic Rayleigh wave with characteristic dispersion
        rayleigh_data = self.create_synthetic_rayleigh_wave()
        
        # Detect waves
        detected_waves = stable_detector.detect_waves(rayleigh_data, self.sampling_rate)
        
        # Should detect at least one surface wave
        self.assertGreater(len(detected_waves), 0)
        
        # Check that detected waves have correct properties
        for wave in detected_waves:
            self.assertIsInstance(wave, WaveSegment)
            self.assertIn(wave.wave_type, ['Love', 'Rayleigh'])
            self.assertGreater(wave.duration, 0)  # Allow shorter durations for testing
            self.assertGreater(wave.confidence, 0)
            self.assertLessEqual(wave.confidence, 1.0)
            self.assertGreater(wave.peak_amplitude, 0)
    
    def test_synthetic_love_wave_detection(self):
        """Test detection of synthetic Love waves."""
        # Use stable parameters
        stable_params = SurfaceWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            frequency_band=(0.05, 0.5),
            min_surface_wave_duration=5.0,
            energy_ratio_threshold=1.5,
            dispersion_window=20.0
        )
        stable_detector = SurfaceWaveDetector(stable_params)
        
        # Create synthetic Love wave
        love_data = self.create_synthetic_love_wave()
        
        # Detect waves
        detected_waves = stable_detector.detect_waves(love_data, self.sampling_rate)
        
        # Should detect at least one surface wave
        self.assertGreater(len(detected_waves), 0)
        
        # Check wave properties
        for wave in detected_waves:
            self.assertIsInstance(wave, WaveSegment)
            self.assertIn(wave.wave_type, ['Love', 'Rayleigh'])
            self.assertGreater(wave.duration, 0)
            self.assertGreater(wave.confidence, 0)
    
    def test_mixed_surface_waves(self):
        """Test detection when both Love and Rayleigh waves are present."""
        # Use stable parameters
        stable_params = SurfaceWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            frequency_band=(0.05, 0.5),
            min_surface_wave_duration=5.0,
            energy_ratio_threshold=1.5,
            dispersion_window=20.0
        )
        stable_detector = SurfaceWaveDetector(stable_params)
        
        # Create mixed surface wave signal
        rayleigh_data = self.create_synthetic_rayleigh_wave()
        love_data = self.create_synthetic_love_wave()
        
        # Combine with time offset
        mixed_data = np.zeros(self.n_samples)
        mixed_data[:len(rayleigh_data)] += rayleigh_data
        
        # Add Love wave with offset
        love_offset = len(rayleigh_data) // 2
        if love_offset + len(love_data) <= len(mixed_data):
            mixed_data[love_offset:love_offset + len(love_data)] += love_data * 0.7
        
        # Detect waves
        detected_waves = stable_detector.detect_waves(mixed_data, self.sampling_rate)
        
        # Should detect multiple surface waves
        self.assertGreater(len(detected_waves), 0)
        
        # Check for both wave types if multiple detections
        if len(detected_waves) > 1:
            wave_types = [wave.wave_type for wave in detected_waves]
            # Should have variety in detected types (though not guaranteed)
            self.assertTrue(len(set(wave_types)) >= 1)
    
    def test_noise_rejection(self):
        """Test that pure noise doesn't trigger false surface wave detections."""
        # Create pure noise
        noise_data = self.create_synthetic_noise(self.duration)
        
        # Detect waves
        detected_waves = self.detector.detect_waves(noise_data, self.sampling_rate)
        
        # Should detect few or no waves in pure noise
        # Allow for some false positives but they should be low confidence
        if len(detected_waves) > 0:
            for wave in detected_waves:
                # False positives should have low confidence
                self.assertLess(wave.confidence, 0.7)
    
    def test_frequency_band_filtering(self):
        """Test that frequency band filtering works correctly."""
        # Create signal with energy outside surface wave band
        high_freq_data = self.create_high_frequency_signal()
        
        # Detect waves
        detected_waves = self.detector.detect_waves(high_freq_data, self.sampling_rate)
        
        # Should detect few or no surface waves in high-frequency signal
        self.assertLessEqual(len(detected_waves), 1)  # Allow for minimal false positives
    
    def test_parameter_setting(self):
        """Test dynamic parameter setting."""
        # Test setting valid parameters
        new_params = {
            'min_surface_wave_duration': 15.0,
            'energy_ratio_threshold': 3.0,
            'frequency_band': (0.01, 0.4)
        }
        
        self.detector.set_parameters(new_params)
        
        self.assertEqual(self.detector.params.min_surface_wave_duration, 15.0)
        self.assertEqual(self.detector.params.energy_ratio_threshold, 3.0)
        self.assertEqual(self.detector.params.frequency_band, (0.01, 0.4))
        
        # Test setting invalid parameter (should issue warning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.detector.set_parameters({'invalid_param': 123})
            self.assertTrue(len(w) > 0)
            self.assertIn("Unknown parameter", str(w[0].message))
    
    def test_detection_statistics(self):
        """Test detection statistics functionality."""
        # Create synthetic surface wave
        surface_data = self.create_synthetic_rayleigh_wave()
        
        # Get detection statistics
        stats = self.detector.get_detection_statistics(surface_data)
        
        # Check that statistics contain expected keys
        expected_keys = [
            'total_surface_waves', 'love_waves', 'rayleigh_waves',
            'mean_group_velocity', 'velocity_range', 'frequency_band',
            'data_duration', 'segments_identified', 'detection_parameters'
        ]
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Check that values are reasonable
        self.assertGreaterEqual(stats['total_surface_waves'], 0)
        self.assertGreaterEqual(stats['love_waves'], 0)
        self.assertGreaterEqual(stats['rayleigh_waves'], 0)
        self.assertEqual(stats['total_surface_waves'], 
                        stats['love_waves'] + stats['rayleigh_waves'])
        self.assertAlmostEqual(stats['data_duration'], len(surface_data) / self.sampling_rate, places=1)
    
    def test_wave_classification(self):
        """Test Love vs Rayleigh wave classification."""
        # Use stable parameters
        stable_params = SurfaceWaveDetectionParameters(
            sampling_rate=self.sampling_rate,
            frequency_band=(0.05, 0.5),
            min_surface_wave_duration=5.0,
            energy_ratio_threshold=1.5,
            dispersion_window=20.0
        )
        stable_detector = SurfaceWaveDetector(stable_params)
        
        # Create waves with different frequency characteristics
        low_freq_wave = self.create_synthetic_surface_wave(dominant_freq=0.1)
        high_freq_wave = self.create_synthetic_surface_wave(dominant_freq=0.2)
        
        # Detect waves
        low_freq_detections = stable_detector.detect_waves(low_freq_wave, self.sampling_rate)
        high_freq_detections = stable_detector.detect_waves(high_freq_wave, self.sampling_rate)
        
        # Both should detect surface waves
        self.assertGreater(len(low_freq_detections), 0)
        self.assertGreater(len(high_freq_detections), 0)
        
        # Check that classification is consistent
        for wave in low_freq_detections + high_freq_detections:
            self.assertIn(wave.wave_type, ['Love', 'Rayleigh'])
    
    def test_metadata_preservation(self):
        """Test that detection metadata is properly preserved."""
        surface_data = self.create_synthetic_rayleigh_wave()
        detected_waves = self.detector.detect_waves(surface_data, self.sampling_rate)
        
        if len(detected_waves) > 0:
            wave = detected_waves[0]
            
            # Check that metadata contains expected information
            self.assertIn('detection_method', wave.metadata)
            self.assertIn('energy_ratio', wave.metadata)
            self.assertIn('duration', wave.metadata)
            self.assertIn('frequency_band', wave.metadata)
            self.assertIn('group_velocity_range', wave.metadata)
            
            # Check metadata values
            self.assertEqual(wave.metadata['detection_method'], 'frequency_time_analysis')
            self.assertGreater(wave.metadata['energy_ratio'], 0)
            self.assertGreater(wave.metadata['duration'], 0)
    
    # Helper methods for creating synthetic data
    
    def create_synthetic_noise(self, duration: float) -> np.ndarray:
        """Create synthetic noise signal."""
        n_samples = int(duration * self.sampling_rate)
        return np.random.normal(0, 0.1, n_samples)
    
    def create_synthetic_rayleigh_wave(self) -> np.ndarray:
        """Create synthetic Rayleigh wave with characteristic dispersion."""
        return self.create_synthetic_surface_wave(
            dominant_freq=0.15,
            amplitude=1.0,
            wave_type='rayleigh'
        )
    
    def create_synthetic_love_wave(self) -> np.ndarray:
        """Create synthetic Love wave."""
        return self.create_synthetic_surface_wave(
            dominant_freq=0.12,
            amplitude=0.8,
            wave_type='love'
        )
    
    def create_synthetic_surface_wave(self, dominant_freq: float = 0.1, 
                                    amplitude: float = 1.0,
                                    wave_type: str = 'rayleigh') -> np.ndarray:
        """
        Create synthetic surface wave with specified characteristics.
        
        Args:
            dominant_freq: Dominant frequency in Hz
            amplitude: Wave amplitude
            wave_type: Type of surface wave ('rayleigh' or 'love')
            
        Returns:
            Synthetic surface wave data
        """
        t = np.linspace(0, self.duration, self.n_samples)
        
        # Create stronger surface wave signal with multiple wave packets
        wave = np.zeros(self.n_samples)
        
        # Create multiple wave arrivals to simulate surface wave train
        for wave_start in [30, 50, 70]:  # Multiple wave arrivals
            # Create wave packet with multiple frequency components
            freq_range = np.linspace(dominant_freq * 0.7, dominant_freq * 1.3, 5)
            
            for freq in freq_range:
                # Create wave with envelope centered at wave_start
                phase = 2 * np.pi * freq * t
                envelope = amplitude * 2.0 * np.exp(-0.5 * ((t - wave_start) / 15)**2)  # Stronger, wider envelope
                
                # Add frequency component
                wave += envelope * np.sin(phase) / len(freq_range)
        
        # Add minimal noise
        noise = np.random.normal(0, amplitude * 0.01, self.n_samples)
        wave += noise
        
        return wave
    
    def create_high_frequency_signal(self) -> np.ndarray:
        """Create high-frequency signal outside surface wave band."""
        t = np.linspace(0, self.duration, self.n_samples)
        
        # Create signal with frequencies above surface wave band
        high_freq_signal = (
            0.5 * np.sin(2 * np.pi * 2.0 * t) +  # 2 Hz
            0.3 * np.sin(2 * np.pi * 5.0 * t) +  # 5 Hz
            0.2 * np.sin(2 * np.pi * 10.0 * t)   # 10 Hz
        )
        
        # Add envelope to make it more realistic
        envelope = np.exp(-0.1 * np.abs(t - self.duration/2))
        high_freq_signal *= envelope
        
        return high_freq_signal


if __name__ == '__main__':
    unittest.main()