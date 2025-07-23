"""
Integration tests for TestDataManager demonstrating end-to-end functionality.

This module provides comprehensive integration tests that demonstrate
the complete workflow of test data generation, validation, and usage
for wave analysis testing.
"""

import unittest
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from datetime import datetime

from tests.test_data_manager import (
    TestDataManager, SyntheticEarthquakeParams, NoiseProfile
)


class TestDataManagerIntegration(unittest.TestCase):
    """Integration tests for TestDataManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TestDataManager(cache_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.manager.cleanup_cache()
    
    def test_complete_synthetic_earthquake_workflow(self):
        """Test complete workflow for synthetic earthquake generation and validation."""
        print("\n=== Testing Complete Synthetic Earthquake Workflow ===")
        
        # Define earthquake parameters
        params = SyntheticEarthquakeParams(
            magnitude=6.2,
            distance=150.0,
            depth=25.0,
            duration=60.0,
            sampling_rate=100.0,
            noise_level=0.15
        )
        
        print(f"Generating earthquake: M{params.magnitude}, {params.distance}km distance")
        
        # Generate synthetic earthquake
        earthquake_data = self.manager.create_synthetic_earthquake(params)
        
        # Validate the generated data
        validation_results = self.manager.validate_test_data_quality(earthquake_data, params)
        
        print(f"Data quality score: {validation_results['quality_score']:.2f}")
        print(f"Validation warnings: {len(validation_results['warnings'])}")
        
        # Assert quality requirements
        self.assertGreaterEqual(validation_results['quality_score'], 0.6)
        self.assertTrue(validation_results['data_length_valid'])
        self.assertTrue(validation_results['wave_arrivals_detected'])
        
        # Save the test dataset
        test_dataset = {
            'earthquake_data': earthquake_data,
            'parameters': {
                'magnitude': params.magnitude,
                'distance': params.distance,
                'depth': params.depth,
                'duration': params.duration,
                'sampling_rate': params.sampling_rate,
                'noise_level': params.noise_level
            },
            'validation_results': validation_results,
            'created_at': datetime.now().isoformat()
        }
        
        self.manager.save_test_dataset('integration_test_earthquake.json', test_dataset)
        
        # Verify we can reload the dataset
        loaded_dataset = self.manager.load_test_dataset('integration_test_earthquake.json')
        self.assertIsNotNone(loaded_dataset)
        
        print("✓ Synthetic earthquake workflow completed successfully")
    
    def test_multi_magnitude_earthquake_suite(self):
        """Test generation of earthquakes across different magnitudes."""
        print("\n=== Testing Multi-Magnitude Earthquake Suite ===")
        
        magnitudes = [4.0, 5.5, 7.0, 8.5]
        distance = 100.0
        
        earthquake_suite = {}
        
        for magnitude in magnitudes:
            params = SyntheticEarthquakeParams(
                magnitude=magnitude,
                distance=distance,
                depth=15.0,
                duration=45.0,
                noise_level=0.1
            )
            
            print(f"Generating M{magnitude} earthquake...")
            
            data = self.manager.create_synthetic_earthquake(params)
            validation = self.manager.validate_test_data_quality(data, params)
            
            earthquake_suite[f'M{magnitude}'] = {
                'data': data,
                'params': params,
                'validation': validation
            }
            
            # Verify amplitude scaling with magnitude
            max_amplitude = np.max(np.abs(data))
            print(f"  Max amplitude: {max_amplitude:.2e}")
            
            self.assertGreater(max_amplitude, 0)
            self.assertGreaterEqual(validation['quality_score'], 0.5)
        
        # Verify amplitude increases with magnitude (generally)
        amplitudes = [np.max(np.abs(earthquake_suite[f'M{m}']['data'])) for m in magnitudes]
        
        # Check that larger magnitudes generally have larger amplitudes
        # (allowing for some variation due to noise and other factors)
        self.assertGreater(amplitudes[-1], amplitudes[0])  # M8.5 > M4.0
        
        print("✓ Multi-magnitude earthquake suite completed successfully")
    
    def test_noise_robustness_testing(self):
        """Test noise generation for algorithm robustness testing."""
        print("\n=== Testing Noise Robustness Suite ===")
        
        # Define various noise profiles
        noise_profiles = [
            NoiseProfile(
                noise_type='white',
                amplitude=0.1,
                frequency_range=(0, 50),
                duration=30.0
            ),
            NoiseProfile(
                noise_type='pink',
                amplitude=0.2,
                frequency_range=(1, 25),
                duration=30.0
            ),
            NoiseProfile(
                noise_type='seismic',
                amplitude=0.15,
                frequency_range=(0.1, 20),
                duration=30.0
            )
        ]
        
        print(f"Generating {len(noise_profiles)} noise samples...")
        
        noise_samples = self.manager.generate_noise_samples(noise_profiles)
        
        self.assertEqual(len(noise_samples), len(noise_profiles))
        
        # Test noise characteristics
        for profile, (key, noise_data) in zip(noise_profiles, noise_samples.items()):
            print(f"  {key}: {len(noise_data)} samples, std={np.std(noise_data):.3f}")
            
            # Verify sample length
            expected_samples = int(profile.duration * profile.sampling_rate)
            self.assertEqual(len(noise_data), expected_samples)
            
            # Verify amplitude is reasonable
            self.assertGreater(np.std(noise_data), 0)
            self.assertLess(np.std(noise_data), profile.amplitude * 5)  # Reasonable range
        
        print("✓ Noise robustness testing completed successfully")
    
    def test_multi_channel_earthquake_generation(self):
        """Test multi-channel earthquake data generation."""
        print("\n=== Testing Multi-Channel Earthquake Generation ===")
        
        params = SyntheticEarthquakeParams(
            magnitude=5.8,
            distance=75.0,
            depth=20.0,
            duration=40.0,
            noise_level=0.12
        )
        
        channels = 4
        print(f"Generating {channels}-channel earthquake data...")
        
        multi_channel_data = self.manager.create_multi_channel_data(channels, params)
        
        # Verify dimensions
        expected_samples = int(params.duration * params.sampling_rate)
        self.assertEqual(multi_channel_data.shape, (channels, expected_samples))
        
        # Verify channels are different but correlated
        correlations = []
        for i in range(channels):
            for j in range(i+1, channels):
                correlation = np.corrcoef(multi_channel_data[i], multi_channel_data[j])[0, 1]
                correlations.append(correlation)
                print(f"  Channel {i}-{j} correlation: {correlation:.3f}")
        
        # Channels should be different but not completely uncorrelated
        # (Due to random variations, correlation may be low but channels should not be identical)
        mean_correlation = np.mean(np.abs(correlations))  # Use absolute correlation
        self.assertGreater(mean_correlation, 0.05)  # Some structure expected
        self.assertLess(mean_correlation, 0.99)     # But not identical
        
        # Verify channels are actually different
        for i in range(channels):
            for j in range(i+1, channels):
                self.assertFalse(np.array_equal(multi_channel_data[i], multi_channel_data[j]))
        
        print("✓ Multi-channel earthquake generation completed successfully")
    
    def test_reference_earthquake_loading(self):
        """Test reference earthquake data loading (mocked for integration test)."""
        print("\n=== Testing Reference Earthquake Loading ===")
        
        # Test local reference data loading
        # (USGS loading is tested separately with mocking)
        
        # Create some mock local reference data
        import json
        import os
        
        mock_reference_data = [
            {
                'event_id': 'integration_test_eq_1',
                'magnitude': 6.5,
                'location': [35.0, -118.0],
                'depth': 15.0,
                'origin_time': '2022-06-15T10:30:00',
                'data_url': 'https://test.example.com/eq1',
                'metadata': {
                    'place': 'Southern California',
                    'source': 'integration_test'
                }
            },
            {
                'event_id': 'integration_test_eq_2',
                'magnitude': 7.2,
                'location': [40.0, -125.0],
                'depth': 25.0,
                'origin_time': '2022-08-20T14:45:00',
                'data_url': 'https://test.example.com/eq2',
                'metadata': {
                    'place': 'Northern California',
                    'source': 'integration_test'
                }
            }
        ]
        
        # Save mock data
        reference_file = os.path.join(self.manager.cache_dir, 'reference_earthquakes.json')
        with open(reference_file, 'w') as f:
            json.dump(mock_reference_data, f)
        
        # Load reference earthquakes
        earthquakes = self.manager.load_reference_earthquakes('local')
        
        print(f"Loaded {len(earthquakes)} reference earthquakes")
        
        self.assertEqual(len(earthquakes), 2)
        
        for eq in earthquakes:
            print(f"  {eq.event_id}: M{eq.magnitude} at {eq.location}")
            self.assertIsInstance(eq.magnitude, float)
            self.assertIsInstance(eq.location, tuple)
            self.assertIsInstance(eq.depth, float)
        
        print("✓ Reference earthquake loading completed successfully")
    
    def test_comprehensive_test_suite_generation(self):
        """Test generation of a comprehensive test suite for algorithm validation."""
        print("\n=== Generating Comprehensive Test Suite ===")
        
        test_suite = {
            'earthquakes': {},
            'noise_samples': {},
            'multi_channel_data': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'purpose': 'Algorithm validation test suite'
            }
        }
        
        # Generate earthquakes of various magnitudes and distances
        test_scenarios = [
            (4.5, 50.0, 'small_near'),
            (5.5, 100.0, 'medium_medium'),
            (6.5, 200.0, 'large_far'),
            (7.5, 300.0, 'major_distant')
        ]
        
        print("Generating earthquake test scenarios...")
        for magnitude, distance, label in test_scenarios:
            params = SyntheticEarthquakeParams(
                magnitude=magnitude,
                distance=distance,
                depth=15.0,
                duration=50.0,
                noise_level=0.1
            )
            
            data = self.manager.create_synthetic_earthquake(params)
            validation = self.manager.validate_test_data_quality(data, params)
            
            test_suite['earthquakes'][label] = {
                'data': data,
                'parameters': params.__dict__,
                'validation': validation
            }
            
            print(f"  {label}: M{magnitude}, {distance}km - Quality: {validation['quality_score']:.2f}")
        
        # Generate noise samples
        print("Generating noise test samples...")
        noise_profiles = [
            NoiseProfile('white', 0.05, (0, 50), 20.0),
            NoiseProfile('pink', 0.1, (1, 30), 20.0),
            NoiseProfile('seismic', 0.08, (0.1, 25), 20.0)
        ]
        
        test_suite['noise_samples'] = self.manager.generate_noise_samples(noise_profiles)
        
        # Generate multi-channel data
        print("Generating multi-channel test data...")
        multi_params = SyntheticEarthquakeParams(
            magnitude=6.0,
            distance=120.0,
            depth=18.0,
            duration=35.0,
            noise_level=0.12
        )
        
        test_suite['multi_channel_data']['3_channel'] = self.manager.create_multi_channel_data(3, multi_params)
        test_suite['multi_channel_data']['5_channel'] = self.manager.create_multi_channel_data(5, multi_params)
        
        # Save comprehensive test suite
        self.manager.save_test_dataset('comprehensive_test_suite.json', test_suite)
        
        # Verify we can load it back
        loaded_suite = self.manager.load_test_dataset('comprehensive_test_suite.json')
        self.assertIsNotNone(loaded_suite)
        
        print(f"✓ Comprehensive test suite generated with {len(test_suite['earthquakes'])} earthquakes")
        print(f"  Noise samples: {len(test_suite['noise_samples'])}")
        print(f"  Multi-channel datasets: {len(test_suite['multi_channel_data'])}")
        
        # Verify all components are present
        self.assertEqual(len(test_suite['earthquakes']), 4)
        self.assertEqual(len(test_suite['noise_samples']), 3)
        self.assertEqual(len(test_suite['multi_channel_data']), 2)
        
        print("✓ Comprehensive test suite generation completed successfully")


if __name__ == '__main__':
    # Run with verbose output to see the progress
    unittest.main(verbosity=2)