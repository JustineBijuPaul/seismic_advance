"""
Unit tests for TestDataManager class.

This module contains comprehensive tests for the test data management system,
validating synthetic earthquake generation, reference data loading, noise generation,
and data quality validation.
"""

import unittest
import numpy as np
import tempfile
import os
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from tests.test_data_manager import (
    TestDataManager, SyntheticEarthquakeParams, ReferenceEarthquake,
    NoiseProfile
)


class TestSyntheticEarthquakeParams(unittest.TestCase):
    """Test SyntheticEarthquakeParams validation."""
    
    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        params = SyntheticEarthquakeParams(
            magnitude=5.5,
            distance=100.0,
            depth=10.0,
            duration=60.0
        )
        self.assertEqual(params.magnitude, 5.5)
        self.assertEqual(params.distance, 100.0)
        self.assertEqual(params.depth, 10.0)
        self.assertEqual(params.duration, 60.0)
        self.assertEqual(params.sampling_rate, 100.0)  # Default value
    
    def test_invalid_magnitude(self):
        """Test validation of magnitude parameter."""
        with self.assertRaises(ValueError):
            SyntheticEarthquakeParams(
                magnitude=-1.0,  # Invalid
                distance=100.0,
                depth=10.0,
                duration=60.0
            )
        
        with self.assertRaises(ValueError):
            SyntheticEarthquakeParams(
                magnitude=15.0,  # Invalid
                distance=100.0,
                depth=10.0,
                duration=60.0
            )
    
    def test_invalid_distance(self):
        """Test validation of distance parameter."""
        with self.assertRaises(ValueError):
            SyntheticEarthquakeParams(
                magnitude=5.0,
                distance=-10.0,  # Invalid
                depth=10.0,
                duration=60.0
            )
    
    def test_invalid_depth(self):
        """Test validation of depth parameter."""
        with self.assertRaises(ValueError):
            SyntheticEarthquakeParams(
                magnitude=5.0,
                distance=100.0,
                depth=-5.0,  # Invalid
                duration=60.0
            )
    
    def test_invalid_duration(self):
        """Test validation of duration parameter."""
        with self.assertRaises(ValueError):
            SyntheticEarthquakeParams(
                magnitude=5.0,
                distance=100.0,
                depth=10.0,
                duration=-10.0  # Invalid
            )


class TestTestDataManager(unittest.TestCase):
    """Test TestDataManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TestDataManager(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.manager.cleanup_cache()
    
    def test_initialization(self):
        """Test TestDataManager initialization."""
        self.assertTrue(os.path.exists(self.manager.cache_dir))
        self.assertEqual(len(self.manager.reference_earthquakes), 0)
        self.assertEqual(len(self.manager.synthetic_data_cache), 0)
    
    def test_create_synthetic_earthquake(self):
        """Test synthetic earthquake generation."""
        params = SyntheticEarthquakeParams(
            magnitude=6.0,
            distance=50.0,
            depth=15.0,
            duration=30.0,
            sampling_rate=100.0
        )
        
        data = self.manager.create_synthetic_earthquake(params)
        
        # Validate output
        expected_samples = int(params.duration * params.sampling_rate)
        self.assertEqual(len(data), expected_samples)
        self.assertIsInstance(data, np.ndarray)
        
        # Check that data contains signal (not all zeros)
        self.assertGreater(np.max(np.abs(data)), 0)
        
        # Check that data is cached
        self.assertEqual(len(self.manager.synthetic_data_cache), 1)
    
    def test_synthetic_earthquake_caching(self):
        """Test that synthetic earthquakes are properly cached."""
        params = SyntheticEarthquakeParams(
            magnitude=5.0,
            distance=100.0,
            depth=10.0,
            duration=20.0
        )
        
        # Generate data twice
        data1 = self.manager.create_synthetic_earthquake(params)
        data2 = self.manager.create_synthetic_earthquake(params)
        
        # Should be identical (cached)
        np.testing.assert_array_equal(data1, data2)
        self.assertEqual(len(self.manager.synthetic_data_cache), 1)
    
    def test_p_wave_generation(self):
        """Test P-wave component generation."""
        time_array = np.linspace(0, 10, 1000)
        arrival_time = 2.0
        magnitude = 5.5
        distance = 75.0
        
        p_wave = self.manager._generate_p_wave(time_array, arrival_time, magnitude, distance)
        
        # Check that P-wave starts at arrival time
        pre_arrival = p_wave[time_array < arrival_time]
        self.assertTrue(np.allclose(pre_arrival, 0, atol=1e-10))
        
        # Check that there's signal after arrival
        post_arrival = p_wave[time_array >= arrival_time]
        self.assertGreater(np.max(np.abs(post_arrival)), 0)
    
    def test_s_wave_generation(self):
        """Test S-wave component generation."""
        time_array = np.linspace(0, 15, 1500)
        arrival_time = 5.0
        magnitude = 6.0
        distance = 100.0
        
        s_wave = self.manager._generate_s_wave(time_array, arrival_time, magnitude, distance)
        
        # Check that S-wave starts at arrival time
        pre_arrival = s_wave[time_array < arrival_time]
        self.assertTrue(np.allclose(pre_arrival, 0, atol=1e-10))
        
        # Check that there's signal after arrival
        post_arrival = s_wave[time_array >= arrival_time]
        self.assertGreater(np.max(np.abs(post_arrival)), 0)
    
    def test_surface_wave_generation(self):
        """Test surface wave component generation."""
        time_array = np.linspace(0, 30, 3000)
        arrival_time = 10.0
        magnitude = 6.5
        distance = 200.0
        
        surface_wave = self.manager._generate_surface_wave(
            time_array, arrival_time, magnitude, distance
        )
        
        # Check that surface wave starts at arrival time
        pre_arrival = surface_wave[time_array < arrival_time]
        self.assertTrue(np.allclose(pre_arrival, 0, atol=1e-10))
        
        # Check that there's signal after arrival
        post_arrival = surface_wave[time_array >= arrival_time]
        self.assertGreater(np.max(np.abs(post_arrival)), 0)
    
    def test_noise_generation(self):
        """Test various noise generation methods."""
        num_samples = 1000
        amplitude = 0.5
        
        # Test white noise
        white_noise = self.manager._generate_noise(num_samples, amplitude, 'white')
        self.assertEqual(len(white_noise), num_samples)
        self.assertAlmostEqual(np.std(white_noise), amplitude, places=1)
        
        # Test pink noise
        pink_noise = self.manager._generate_noise(num_samples, amplitude, 'pink')
        self.assertEqual(len(pink_noise), num_samples)
        
        # Test seismic noise
        seismic_noise = self.manager._generate_noise(num_samples, amplitude, 'seismic')
        self.assertEqual(len(seismic_noise), num_samples)
    
    def test_generate_noise_samples(self):
        """Test noise sample generation with profiles."""
        profiles = [
            NoiseProfile(
                noise_type='white',
                amplitude=0.1,
                frequency_range=(0, 50),
                duration=5.0
            ),
            NoiseProfile(
                noise_type='pink',
                amplitude=0.2,
                frequency_range=(1, 20),
                duration=10.0
            )
        ]
        
        noise_samples = self.manager.generate_noise_samples(profiles)
        
        self.assertEqual(len(noise_samples), 2)
        self.assertIn('white_0.1', noise_samples)
        self.assertIn('pink_0.2', noise_samples)
        
        # Check sample lengths
        self.assertEqual(len(noise_samples['white_0.1']), 500)  # 5s * 100Hz
        self.assertEqual(len(noise_samples['pink_0.2']), 1000)  # 10s * 100Hz
    
    @patch('requests.get')
    def test_load_usgs_earthquakes_success(self, mock_get):
        """Test successful loading of USGS earthquake data."""
        # Mock USGS API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'features': [
                {
                    'id': 'test_eq_1',
                    'properties': {
                        'mag': 6.5,
                        'time': 1640995200000,  # 2022-01-01 00:00:00 UTC
                        'place': 'Test Location',
                        'type': 'earthquake',
                        'status': 'reviewed',
                        'tsunami': 0,
                        'url': 'https://test.url'
                    },
                    'geometry': {
                        'coordinates': [-120.0, 35.0, 10.0]
                    }
                }
            ]
        }
        mock_get.return_value = mock_response
        
        earthquakes = self.manager.load_reference_earthquakes('usgs', limit=1)
        
        self.assertEqual(len(earthquakes), 1)
        eq = earthquakes[0]
        self.assertEqual(eq.event_id, 'test_eq_1')
        self.assertEqual(eq.magnitude, 6.5)
        self.assertEqual(eq.location, (35.0, -120.0))  # lat, lon
        self.assertEqual(eq.depth, 10.0)
        self.assertEqual(eq.metadata['place'], 'Test Location')
    
    @patch('requests.get')
    def test_load_usgs_earthquakes_failure(self, mock_get):
        """Test handling of USGS API failure."""
        mock_get.side_effect = Exception("API Error")
        
        earthquakes = self.manager.load_reference_earthquakes('usgs', limit=1)
        
        self.assertEqual(len(earthquakes), 0)
    
    def test_load_local_reference_data(self):
        """Test loading reference data from local file."""
        # Create test reference data file
        reference_data = [
            {
                'event_id': 'local_eq_1',
                'magnitude': 7.0,
                'location': [40.0, -125.0],
                'depth': 20.0,
                'origin_time': '2022-01-01T12:00:00',
                'data_url': 'https://local.test',
                'metadata': {'source': 'local'}
            }
        ]
        
        reference_file = os.path.join(self.manager.cache_dir, 'reference_earthquakes.json')
        with open(reference_file, 'w') as f:
            json.dump(reference_data, f)
        
        earthquakes = self.manager.load_reference_earthquakes('local')
        
        self.assertEqual(len(earthquakes), 1)
        eq = earthquakes[0]
        self.assertEqual(eq.event_id, 'local_eq_1')
        self.assertEqual(eq.magnitude, 7.0)
        self.assertEqual(eq.location, (40.0, -125.0))
        self.assertEqual(eq.depth, 20.0)
    
    def test_create_multi_channel_data(self):
        """Test multi-channel data generation."""
        params = SyntheticEarthquakeParams(
            magnitude=5.5,
            distance=80.0,
            depth=12.0,
            duration=25.0
        )
        
        channels = 3
        multi_data = self.manager.create_multi_channel_data(channels, params)
        
        # Check dimensions
        expected_samples = int(params.duration * params.sampling_rate)
        self.assertEqual(multi_data.shape, (channels, expected_samples))
        
        # Check that channels are different (due to random variations)
        self.assertFalse(np.array_equal(multi_data[0], multi_data[1]))
        self.assertFalse(np.array_equal(multi_data[1], multi_data[2]))
    
    def test_validate_test_data_quality(self):
        """Test data quality validation."""
        params = SyntheticEarthquakeParams(
            magnitude=6.0,
            distance=100.0,
            depth=15.0,
            duration=30.0
        )
        
        data = self.manager.create_synthetic_earthquake(params)
        validation = self.manager.validate_test_data_quality(data, params)
        
        # Check validation structure
        expected_keys = [
            'data_length_valid', 'amplitude_range_valid', 'frequency_content_valid',
            'noise_level_valid', 'wave_arrivals_detected', 'quality_score', 'warnings'
        ]
        for key in expected_keys:
            self.assertIn(key, validation)
        
        # Check that quality score is reasonable
        self.assertGreaterEqual(validation['quality_score'], 0.0)
        self.assertLessEqual(validation['quality_score'], 1.0)
        
        # For good synthetic data, most checks should pass
        self.assertGreaterEqual(validation['quality_score'], 0.6)
    
    def test_save_and_load_test_dataset(self):
        """Test saving and loading test datasets."""
        test_data = {
            'earthquake_data': np.random.random(1000),
            'parameters': {
                'magnitude': 5.5,
                'distance': 100.0
            },
            'metadata': {
                'created': datetime.now().isoformat()
            }
        }
        
        filename = 'test_dataset.json'
        
        # Save dataset
        self.manager.save_test_dataset(filename, test_data)
        
        # Check file exists
        filepath = os.path.join(self.manager.cache_dir, filename)
        self.assertTrue(os.path.exists(filepath))
        
        # Load dataset
        loaded_data = self.manager.load_test_dataset(filename)
        
        self.assertIsNotNone(loaded_data)
        self.assertIn('earthquake_data', loaded_data)
        self.assertIn('parameters', loaded_data)
        self.assertIn('metadata', loaded_data)
        
        # Check that numpy array was properly restored
        np.testing.assert_array_equal(
            test_data['earthquake_data'],
            loaded_data['earthquake_data']
        )
    
    def test_load_nonexistent_dataset(self):
        """Test loading non-existent dataset."""
        result = self.manager.load_test_dataset('nonexistent.json')
        self.assertIsNone(result)
    
    def test_power_spectrum_computation(self):
        """Test power spectrum computation."""
        # Create test signal with known frequency
        sampling_rate = 100.0
        duration = 2.0
        test_freq = 10.0
        
        t = np.linspace(0, duration, int(duration * sampling_rate))
        signal = np.sin(2 * np.pi * test_freq * t)
        
        freqs, psd = self.manager._compute_power_spectrum(signal, sampling_rate)
        
        # Find peak frequency
        peak_idx = np.argmax(psd)
        peak_freq = freqs[peak_idx]
        
        # Should be close to test frequency
        self.assertAlmostEqual(peak_freq, test_freq, places=0)
    
    def test_cache_key_generation(self):
        """Test cache key generation for earthquake parameters."""
        params = SyntheticEarthquakeParams(
            magnitude=5.5,
            distance=100.0,
            depth=15.0,
            duration=30.0,
            noise_level=0.1
        )
        
        key = self.manager._create_cache_key(params)
        
        self.assertIsInstance(key, str)
        self.assertIn('5.5', key)
        self.assertIn('100.0', key)
        self.assertIn('15.0', key)
        self.assertIn('30.0', key)
        self.assertIn('0.1', key)


class TestNoiseProfile(unittest.TestCase):
    """Test NoiseProfile data class."""
    
    def test_noise_profile_creation(self):
        """Test NoiseProfile creation."""
        profile = NoiseProfile(
            noise_type='white',
            amplitude=0.5,
            frequency_range=(1.0, 20.0),
            duration=10.0
        )
        
        self.assertEqual(profile.noise_type, 'white')
        self.assertEqual(profile.amplitude, 0.5)
        self.assertEqual(profile.frequency_range, (1.0, 20.0))
        self.assertEqual(profile.duration, 10.0)
        self.assertEqual(profile.sampling_rate, 100.0)  # Default


class TestReferenceEarthquake(unittest.TestCase):
    """Test ReferenceEarthquake data class."""
    
    def test_reference_earthquake_creation(self):
        """Test ReferenceEarthquake creation."""
        earthquake = ReferenceEarthquake(
            event_id='test_eq',
            magnitude=6.5,
            location=(35.0, -120.0),
            depth=10.0,
            origin_time=datetime(2022, 1, 1, 12, 0, 0)
        )
        
        self.assertEqual(earthquake.event_id, 'test_eq')
        self.assertEqual(earthquake.magnitude, 6.5)
        self.assertEqual(earthquake.location, (35.0, -120.0))
        self.assertEqual(earthquake.depth, 10.0)
        self.assertEqual(earthquake.origin_time, datetime(2022, 1, 1, 12, 0, 0))


if __name__ == '__main__':
    unittest.main()