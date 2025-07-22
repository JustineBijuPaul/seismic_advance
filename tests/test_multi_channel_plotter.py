"""
Unit tests for MultiChannelPlotter class.

Tests the multi-channel visualization functionality with synthetic multi-channel data.
"""

import unittest
import numpy as np
import json

from wave_analysis.models import WaveSegment
from wave_analysis.services.multi_channel_plotter import MultiChannelPlotter, ChannelData


class TestMultiChannelPlotter(unittest.TestCase):
    """Test cases for MultiChannelPlotter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plotter = MultiChannelPlotter()
        
        # Create synthetic multi-channel data
        self.sampling_rate = 100.0  # 100 Hz
        self.duration = 10.0  # 10 seconds
        self.samples = int(self.duration * self.sampling_rate)
        
        # Create three channels with different characteristics
        self.channel1 = self._create_channel_data('CH1', orientation='Z')  # Vertical
        self.channel2 = self._create_channel_data('CH2', orientation='N')  # North
        self.channel3 = self._create_channel_data('CH3', orientation='E')  # East
        
        # Create wave segments for testing
        self.p_wave = WaveSegment(
            wave_type='P',
            start_time=2.0,
            end_time=4.0,
            data=np.random.random(200),
            sampling_rate=self.sampling_rate,
            peak_amplitude=0.8,
            dominant_frequency=8.0,
            arrival_time=2.5,
            confidence=0.9
        )
        
        self.s_wave = WaveSegment(
            wave_type='S',
            start_time=5.0,
            end_time=7.0,
            data=np.random.random(200),
            sampling_rate=self.sampling_rate,
            peak_amplitude=0.6,
            dominant_frequency=4.0,
            arrival_time=5.5,
            confidence=0.85
        )
    
    def _create_channel_data(self, channel_id: str, orientation: str) -> ChannelData:
        """Create synthetic channel data."""
        # Create base signal
        t = np.linspace(0, self.duration, self.samples)
        
        # Add different frequency components based on orientation
        if orientation == 'Z':  # Vertical - more P-wave content
            signal = (0.5 * np.sin(2 * np.pi * 8 * t) * np.exp(-0.5 * (t - 2.5)**2) +  # P-wave
                     0.3 * np.sin(2 * np.pi * 4 * t) * np.exp(-0.3 * (t - 5.5)**2))    # S-wave
        elif orientation == 'N':  # North - more S-wave content
            signal = (0.2 * np.sin(2 * np.pi * 8 * t) * np.exp(-0.5 * (t - 2.5)**2) +  # P-wave
                     0.6 * np.sin(2 * np.pi * 4 * t) * np.exp(-0.3 * (t - 5.5)**2))    # S-wave
        else:  # East - similar to North but phase shifted
            signal = (0.2 * np.sin(2 * np.pi * 8 * t + np.pi/4) * np.exp(-0.5 * (t - 2.5)**2) +
                     0.6 * np.sin(2 * np.pi * 4 * t + np.pi/2) * np.exp(-0.3 * (t - 5.5)**2))
        
        # Add noise
        noise = 0.1 * np.random.normal(0, 1, self.samples)
        signal += noise
        
        # Create location data
        location = {
            'lat': 40.0 + np.random.uniform(-0.1, 0.1),
            'lon': -120.0 + np.random.uniform(-0.1, 0.1),
            'elevation': 1000.0 + np.random.uniform(-100, 100)
        }
        
        return ChannelData(
            channel_id=channel_id,
            data=signal,
            sampling_rate=self.sampling_rate,
            location=location,
            orientation=orientation
        )
    
    def test_channel_data_creation(self):
        """Test ChannelData class functionality."""
        channel = self.channel1
        
        self.assertEqual(channel.channel_id, 'CH1')
        self.assertEqual(channel.sampling_rate, self.sampling_rate)
        self.assertEqual(channel.orientation, 'Z')
        self.assertEqual(len(channel.data), self.samples)
        self.assertAlmostEqual(channel.duration, self.duration, places=1)
        self.assertIn('lat', channel.location)
        self.assertIn('lon', channel.location)
    
    def test_create_multi_channel_plot(self):
        """Test creating multi-channel time series plot."""
        channels = [self.channel1, self.channel2, self.channel3]
        plot_data = self.plotter.create_multi_channel_plot(channels)
        
        # Check basic structure
        self.assertEqual(plot_data['type'], 'multi_channel_time_series')
        self.assertIn('datasets', plot_data)
        self.assertIn('layout', plot_data)
        
        # Check that all channels are represented
        self.assertEqual(len(plot_data['datasets']), 3)
        
        # Check dataset properties
        for i, dataset in enumerate(plot_data['datasets']):
            self.assertEqual(dataset['channel_id'], channels[i].channel_id)
            self.assertIn('data', dataset)
            self.assertGreater(len(dataset['data']), 0)
            
            # Check data format
            data_point = dataset['data'][0]
            self.assertIn('x', data_point)  # Time
            self.assertIn('y', data_point)  # Amplitude
    
    def test_create_multi_channel_plot_empty_input(self):
        """Test creating multi-channel plot with empty input."""
        plot_data = self.plotter.create_multi_channel_plot([])
        
        self.assertEqual(plot_data['type'], 'empty')
        self.assertIn('message', plot_data)
        self.assertEqual(len(plot_data['datasets']), 0)
    
    def test_create_cross_correlation_plot(self):
        """Test creating cross-correlation visualization."""
        channels = [self.channel1, self.channel2]
        plot_data = self.plotter.create_cross_correlation_plot(channels)
        
        # Check basic structure
        self.assertEqual(plot_data['type'], 'cross_correlation')
        self.assertIn('datasets', plot_data)
        self.assertIn('correlation_summary', plot_data)
        
        # Should have one dataset for the channel pair
        self.assertEqual(len(plot_data['datasets']), 1)
        
        # Check correlation summary
        summary = plot_data['correlation_summary']
        self.assertIn('max_correlation', summary)
        self.assertIn('mean_correlation', summary)
        self.assertIn('pair_count', summary)
        self.assertEqual(summary['pair_count'], 1)
    
    def test_create_cross_correlation_plot_insufficient_channels(self):
        """Test cross-correlation with insufficient channels."""
        plot_data = self.plotter.create_cross_correlation_plot([self.channel1])
        
        self.assertEqual(plot_data['type'], 'empty')
        self.assertIn('Need at least 2 channels', plot_data['message'])
    
    def test_create_coherence_plot(self):
        """Test creating coherence analysis plots."""
        channels = [self.channel1, self.channel2, self.channel3]
        plot_data = self.plotter.create_coherence_plot(channels)
        
        # Check basic structure
        self.assertEqual(plot_data['type'], 'coherence_analysis')
        self.assertIn('datasets', plot_data)
        
        # Should have datasets for all channel pairs (3 choose 2 = 3)
        self.assertEqual(len(plot_data['datasets']), 3)
        
        # Check dataset properties
        for dataset in plot_data['datasets']:
            self.assertIn('p_band_coherence', dataset)
            self.assertIn('s_band_coherence', dataset)
            self.assertIn('surface_band_coherence', dataset)
            self.assertGreater(len(dataset['data']), 0)
    
    def test_create_coherence_plot_insufficient_channels(self):
        """Test coherence analysis with insufficient channels."""
        plot_data = self.plotter.create_coherence_plot([self.channel1])
        
        self.assertEqual(plot_data['type'], 'empty')
        self.assertIn('Need at least 2 channels', plot_data['message'])
    
    def test_create_correlation_matrix_plot(self):
        """Test creating correlation matrix heatmap."""
        channels = [self.channel1, self.channel2, self.channel3]
        plot_data = self.plotter.create_correlation_matrix_plot(channels)
        
        # Check basic structure
        self.assertEqual(plot_data['type'], 'correlation_matrix')
        self.assertIn('data', plot_data)
        
        # Check matrix dimensions
        matrix_data = plot_data['data']
        self.assertEqual(len(matrix_data['z']), 3)  # 3x3 matrix
        self.assertEqual(len(matrix_data['z'][0]), 3)
        self.assertEqual(len(matrix_data['x']), 3)  # Channel labels
        self.assertEqual(len(matrix_data['y']), 3)
        
        # Check diagonal elements are 1.0 (self-correlation)
        for i in range(3):
            self.assertAlmostEqual(matrix_data['z'][i][i], 1.0, places=5)
    
    def test_create_channel_comparison_plot(self):
        """Test creating channel comparison plot with wave arrivals."""
        channels = [self.channel1, self.channel2]
        wave_segments = [self.p_wave, self.s_wave]
        
        plot_data = self.plotter.create_channel_comparison_plot(channels, wave_segments)
        
        # Check basic structure
        self.assertEqual(plot_data['type'], 'channel_wave_comparison')
        self.assertIn('datasets', plot_data)
        self.assertIn('annotations', plot_data)
        
        # Should have datasets for all channels
        self.assertEqual(len(plot_data['datasets']), 2)
        
        # Should have annotations for wave arrivals
        self.assertEqual(len(plot_data['annotations']), 2)  # P and S waves
    
    def test_calculate_cross_correlation(self):
        """Test cross-correlation calculation between channels."""
        correlation_data = self.plotter._calculate_cross_correlation(self.channel1, self.channel2)
        
        # Check required fields
        self.assertEqual(correlation_data['channel1_id'], 'CH1')
        self.assertEqual(correlation_data['channel2_id'], 'CH2')
        self.assertIn('lag_times', correlation_data)
        self.assertIn('correlation', correlation_data)
        self.assertIn('peak_lag', correlation_data)
        self.assertIn('peak_correlation', correlation_data)
        
        # Check data types and ranges
        self.assertIsInstance(correlation_data['peak_lag'], (int, float))
        self.assertIsInstance(correlation_data['peak_correlation'], (int, float))
        self.assertEqual(len(correlation_data['lag_times']), len(correlation_data['correlation']))
    
    def test_calculate_cross_correlation_different_sampling_rates(self):
        """Test cross-correlation with different sampling rates raises error."""
        # Create channel with different sampling rate
        different_rate_channel = ChannelData(
            channel_id='CH_DIFF',
            data=np.random.random(500),
            sampling_rate=50.0,  # Different rate
            orientation='Z'
        )
        
        with self.assertRaises(ValueError) as context:
            self.plotter._calculate_cross_correlation(self.channel1, different_rate_channel)
        
        self.assertIn('same sampling rate', str(context.exception))
    
    def test_calculate_coherence(self):
        """Test coherence calculation between channels."""
        coherence_data = self.plotter._calculate_coherence(self.channel1, self.channel2)
        
        # Check required fields
        self.assertEqual(coherence_data['channel1_id'], 'CH1')
        self.assertEqual(coherence_data['channel2_id'], 'CH2')
        self.assertIn('frequencies', coherence_data)
        self.assertIn('coherence', coherence_data)
        self.assertIn('p_band_coherence', coherence_data)
        self.assertIn('s_band_coherence', coherence_data)
        self.assertIn('surface_band_coherence', coherence_data)
        
        # Check coherence values are in valid range [0, 1]
        coherence_values = coherence_data['coherence']
        self.assertTrue(np.all(coherence_values >= 0))
        self.assertTrue(np.all(coherence_values <= 1))
        
        # Check band coherence values
        self.assertGreaterEqual(coherence_data['p_band_coherence'], 0)
        self.assertLessEqual(coherence_data['p_band_coherence'], 1)
    
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        channels = [self.channel1, self.channel2, self.channel3]
        correlation_matrix = self.plotter._calculate_correlation_matrix(channels)
        
        # Check matrix properties
        self.assertEqual(correlation_matrix.shape, (3, 3))
        
        # Check diagonal elements are 1.0
        for i in range(3):
            self.assertAlmostEqual(correlation_matrix[i, i], 1.0, places=5)
        
        # Check matrix is symmetric
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(correlation_matrix[i, j], correlation_matrix[j, i], places=5)
        
        # Check correlation values are in valid range [-1, 1]
        self.assertTrue(np.all(correlation_matrix >= -1))
        self.assertTrue(np.all(correlation_matrix <= 1))
    
    def test_export_correlation_data(self):
        """Test exporting correlation analysis data."""
        channels = [self.channel1, self.channel2]
        exported_data = self.plotter.export_correlation_data(channels, 'json')
        
        # Parse JSON to verify structure
        data = json.loads(exported_data)
        
        # Check required fields
        self.assertIn('CH1_vs_CH2', data)
        self.assertIn('correlation_matrix', data)
        self.assertIn('channel_labels', data)
        
        # Check pair data
        pair_data = data['CH1_vs_CH2']
        self.assertIn('peak_correlation', pair_data)
        self.assertIn('peak_lag', pair_data)
        self.assertIn('lag_times', pair_data)
        self.assertIn('correlation_values', pair_data)
        
        # Check matrix data
        self.assertEqual(len(data['correlation_matrix']), 2)
        self.assertEqual(len(data['correlation_matrix'][0]), 2)
        self.assertEqual(data['channel_labels'], ['CH1', 'CH2'])
    
    def test_export_correlation_data_insufficient_channels(self):
        """Test exporting correlation data with insufficient channels."""
        with self.assertRaises(ValueError) as context:
            self.plotter.export_correlation_data([self.channel1], 'json')
        
        self.assertIn('Need at least 2 channels', str(context.exception))
    
    def test_export_coherence_data(self):
        """Test exporting coherence analysis data."""
        channels = [self.channel1, self.channel2]
        exported_data = self.plotter.export_coherence_data(channels, 'json')
        
        # Parse JSON to verify structure
        data = json.loads(exported_data)
        
        # Check required fields
        self.assertIn('CH1_vs_CH2', data)
        
        # Check pair data
        pair_data = data['CH1_vs_CH2']
        self.assertIn('frequencies', pair_data)
        self.assertIn('coherence_values', pair_data)
        self.assertIn('p_band_coherence', pair_data)
        self.assertIn('s_band_coherence', pair_data)
        self.assertIn('surface_band_coherence', pair_data)
        
        # Check data types
        self.assertIsInstance(pair_data['frequencies'], list)
        self.assertIsInstance(pair_data['coherence_values'], list)
        self.assertEqual(len(pair_data['frequencies']), len(pair_data['coherence_values']))
    
    def test_configuration_methods(self):
        """Test configuration setter methods."""
        # Test correlation config
        new_corr_config = {'max_lag_seconds': 5.0, 'normalize': False}
        self.plotter.set_correlation_config(new_corr_config)
        self.assertEqual(self.plotter.correlation_config['max_lag_seconds'], 5.0)
        self.assertEqual(self.plotter.correlation_config['normalize'], False)
        
        # Test coherence config
        new_coh_config = {'nperseg': 512, 'window': 'blackman'}
        self.plotter.set_coherence_config(new_coh_config)
        self.assertEqual(self.plotter.coherence_config['nperseg'], 512)
        self.assertEqual(self.plotter.coherence_config['window'], 'blackman')
        
        # Test color scheme
        new_colors = ['#FF0000', '#00FF00', '#0000FF']
        self.plotter.set_color_scheme(new_colors)
        self.assertEqual(self.plotter.default_colors, new_colors)
    
    def test_wave_color_mapping(self):
        """Test wave type color mapping."""
        self.assertEqual(self.plotter._get_wave_color('P'), '#FF6B6B')
        self.assertEqual(self.plotter._get_wave_color('S'), '#4ECDC4')
        self.assertEqual(self.plotter._get_wave_color('Love'), '#45B7D1')
        self.assertEqual(self.plotter._get_wave_color('Rayleigh'), '#96CEB4')
        self.assertEqual(self.plotter._get_wave_color('Unknown'), '#333333')
    
    def test_synthetic_multi_channel_data_properties(self):
        """Test properties of synthetic multi-channel data."""
        # Test that different orientations have different characteristics
        channels = [self.channel1, self.channel2, self.channel3]
        
        # Calculate RMS for each channel
        rms_values = [np.sqrt(np.mean(ch.data**2)) for ch in channels]
        
        # Channels should have different energy levels due to different orientations
        self.assertNotEqual(rms_values[0], rms_values[1])
        self.assertNotEqual(rms_values[1], rms_values[2])
        
        # All channels should have reasonable signal levels
        for rms in rms_values:
            self.assertGreater(rms, 0.1)  # Not too small
            self.assertLess(rms, 2.0)     # Not too large


if __name__ == '__main__':
    unittest.main()