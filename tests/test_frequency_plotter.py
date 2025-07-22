"""
Unit tests for FrequencyPlotter class.

Tests the frequency-domain visualization functionality for different wave types
with synthetic wave data.
"""

import unittest
import numpy as np
from datetime import datetime
import json

from wave_analysis.models import WaveSegment, FrequencyData
from wave_analysis.services.frequency_plotter import FrequencyPlotter


class TestFrequencyPlotter(unittest.TestCase):
    """Test cases for FrequencyPlotter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plotter = FrequencyPlotter()
        
        # Create synthetic wave data
        self.sampling_rate = 100.0  # 100 Hz
        self.duration = 5.0  # 5 seconds
        
        # Create synthetic P-wave with known frequency content
        p_wave_data = self._create_synthetic_p_wave()
        self.p_wave = WaveSegment(
            wave_type='P',
            start_time=1.0,
            end_time=3.0,
            data=p_wave_data,
            sampling_rate=self.sampling_rate,
            peak_amplitude=np.max(np.abs(p_wave_data)),
            dominant_frequency=8.0,
            arrival_time=1.5,
            confidence=0.9
        )
        
        # Create synthetic S-wave with different frequency content
        s_wave_data = self._create_synthetic_s_wave()
        self.s_wave = WaveSegment(
            wave_type='S',
            start_time=3.5,
            end_time=5.5,
            data=s_wave_data,
            sampling_rate=self.sampling_rate,
            peak_amplitude=np.max(np.abs(s_wave_data)),
            dominant_frequency=4.0,
            arrival_time=4.0,
            confidence=0.85
        )
        
        # Create synthetic surface wave
        surface_wave_data = self._create_synthetic_surface_wave()
        self.surface_wave = WaveSegment(
            wave_type='Love',
            start_time=6.0,
            end_time=8.0,
            data=surface_wave_data,
            sampling_rate=self.sampling_rate,
            peak_amplitude=np.max(np.abs(surface_wave_data)),
            dominant_frequency=2.0,
            arrival_time=6.5,
            confidence=0.8
        )
    
    def _create_synthetic_p_wave(self) -> np.ndarray:
        """Create synthetic P-wave with known frequency content."""
        duration = 2.0
        samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # High-frequency signal (8 Hz dominant)
        frequency = 8.0
        envelope = np.exp(-2 * t)
        wave = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add some harmonics
        wave += 0.3 * envelope * np.sin(2 * np.pi * frequency * 2 * t)
        
        # Add noise
        noise = 0.1 * np.random.normal(0, 1, samples)
        return wave + noise
    
    def _create_synthetic_s_wave(self) -> np.ndarray:
        """Create synthetic S-wave with known frequency content."""
        duration = 2.0
        samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # Medium-frequency signal (4 Hz dominant)
        frequency = 4.0
        envelope = np.exp(-1 * t)
        wave = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add some harmonics
        wave += 0.4 * envelope * np.sin(2 * np.pi * frequency * 1.5 * t)
        
        # Add noise
        noise = 0.1 * np.random.normal(0, 1, samples)
        return wave + noise
    
    def _create_synthetic_surface_wave(self) -> np.ndarray:
        """Create synthetic surface wave with known frequency content."""
        duration = 2.0
        samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # Low-frequency signal (2 Hz dominant)
        frequency = 2.0
        envelope = np.exp(-0.5 * t)
        wave = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add some harmonics
        wave += 0.2 * envelope * np.sin(2 * np.pi * frequency * 0.5 * t)
        
        # Add noise
        noise = 0.05 * np.random.normal(0, 1, samples)
        return wave + noise
    
    def test_create_frequency_plot_single_wave(self):
        """Test creating frequency plot for single wave segment."""
        plot_data = self.plotter.create_frequency_plot([self.p_wave])
        
        # Check basic structure
        self.assertEqual(plot_data['type'], 'frequency_spectrum')
        self.assertIn('datasets', plot_data)
        self.assertIn('layout', plot_data)
        self.assertIn('config', plot_data)
        
        # Check dataset
        self.assertEqual(len(plot_data['datasets']), 1)
        dataset = plot_data['datasets'][0]
        self.assertEqual(dataset['label'], 'P Wave Spectrum')
        self.assertEqual(dataset['wave_type'], 'P')
        
        # Check data points
        self.assertGreater(len(dataset['data']), 0)
        self.assertIn('x', dataset['data'][0])  # Frequency
        self.assertIn('y', dataset['data'][0])  # Power
    
    def test_create_frequency_plot_multiple_waves(self):
        """Test creating frequency plot for multiple wave segments."""
        waves = [self.p_wave, self.s_wave, self.surface_wave]
        plot_data = self.plotter.create_frequency_plot(waves)
        
        # Check that all wave types are represented
        self.assertEqual(len(plot_data['datasets']), 3)
        
        wave_types = [dataset['wave_type'] for dataset in plot_data['datasets']]
        self.assertIn('P', wave_types)
        self.assertIn('S', wave_types)
        self.assertIn('Love', wave_types)
        
        # Check frequency markers
        self.assertIn('annotations', plot_data)
        self.assertEqual(len(plot_data['annotations']), 3)
    
    def test_create_frequency_plot_empty_input(self):
        """Test creating frequency plot with empty wave list."""
        plot_data = self.plotter.create_frequency_plot([])
        
        self.assertEqual(plot_data['type'], 'empty')
        self.assertIn('message', plot_data)
        self.assertEqual(len(plot_data['datasets']), 0)
    
    def test_create_spectrogram_plot(self):
        """Test creating spectrogram plot."""
        plot_data = self.plotter.create_spectrogram_plot([self.p_wave])
        
        self.assertEqual(plot_data['type'], 'spectrogram')
        self.assertIn('data', plot_data)
        self.assertIn('layout', plot_data)
        
        # Check spectrogram data structure
        spec_data = plot_data['data']
        self.assertIn('z', spec_data)  # Power spectral density matrix
        self.assertIn('x', spec_data)  # Time axis
        self.assertIn('y', spec_data)  # Frequency axis
        self.assertEqual(spec_data['type'], 'heatmap')
        
        # Check dimensions
        self.assertGreater(len(spec_data['x']), 0)
        self.assertGreater(len(spec_data['y']), 0)
        self.assertGreater(len(spec_data['z']), 0)
    
    def test_create_frequency_comparison_plot(self):
        """Test creating frequency comparison plot."""
        waves = [self.p_wave, self.s_wave, self.surface_wave]
        plot_data = self.plotter.create_frequency_comparison_plot(waves)
        
        self.assertEqual(plot_data['type'], 'frequency_comparison')
        self.assertEqual(len(plot_data['datasets']), 3)
        
        # Check that all datasets are configured for comparison
        for dataset in plot_data['datasets']:
            self.assertFalse(dataset['fill'])
            self.assertEqual(dataset['borderWidth'], 2)
        
        # Check frequency band annotations
        self.assertIn('annotations', plot_data)
        self.assertGreater(len(plot_data['annotations']), 0)
    
    def test_create_power_spectral_density_plot(self):
        """Test creating power spectral density plot."""
        waves = [self.p_wave, self.s_wave]
        plot_data = self.plotter.create_power_spectral_density_plot(waves)
        
        self.assertEqual(plot_data['type'], 'power_spectral_density')
        self.assertEqual(len(plot_data['datasets']), 2)
        
        # Check PSD-specific configuration
        layout = plot_data['layout']
        self.assertEqual(layout['scales']['y']['title']['text'], 'Power Spectral Density (dB/Hz)')
    
    def test_calculate_spectrogram(self):
        """Test spectrogram calculation."""
        frequencies, times, Sxx = self.plotter._calculate_spectrogram(self.p_wave)
        
        # Check output dimensions
        self.assertGreater(len(frequencies), 0)
        self.assertGreater(len(times), 0)
        self.assertEqual(Sxx.shape[0], len(frequencies))
        self.assertEqual(Sxx.shape[1], len(times))
        
        # Check frequency range
        self.assertGreaterEqual(frequencies[0], 0)
        self.assertLessEqual(frequencies[-1], self.sampling_rate / 2)
        
        # Check time adjustment
        self.assertGreaterEqual(times[0], self.p_wave.start_time)
    
    def test_calculate_frequency_spectrum(self):
        """Test frequency spectrum calculation."""
        frequencies, psd = self.plotter._calculate_frequency_spectrum(self.p_wave)
        
        # Check output dimensions
        self.assertEqual(len(frequencies), len(psd))
        self.assertGreater(len(frequencies), 0)
        
        # Check frequency range
        self.assertGreaterEqual(frequencies[0], 0)
        self.assertLessEqual(frequencies[-1], self.sampling_rate / 2)
        
        # Check that PSD values are positive
        self.assertTrue(np.all(psd >= 0))
        
        # Check that dominant frequency is near expected value
        peak_idx = np.argmax(psd)
        peak_frequency = frequencies[peak_idx]
        self.assertAlmostEqual(peak_frequency, self.p_wave.dominant_frequency, delta=2.0)
    
    def test_calculate_power_spectral_density(self):
        """Test power spectral density calculation using Welch's method."""
        frequencies, psd = self.plotter._calculate_power_spectral_density(self.p_wave)
        
        # Check output dimensions
        self.assertEqual(len(frequencies), len(psd))
        self.assertGreater(len(frequencies), 0)
        
        # Check that PSD values are positive
        self.assertTrue(np.all(psd >= 0))
        
        # Check frequency resolution
        freq_resolution = frequencies[1] - frequencies[0]
        self.assertGreater(freq_resolution, 0)
    
    def test_group_waves_by_type(self):
        """Test grouping waves by type."""
        waves = [self.p_wave, self.s_wave, self.surface_wave]
        grouped = self.plotter._group_waves_by_type(waves)
        
        self.assertIn('P', grouped)
        self.assertIn('S', grouped)
        self.assertIn('Love', grouped)
        
        self.assertEqual(len(grouped['P']), 1)
        self.assertEqual(len(grouped['S']), 1)
        self.assertEqual(len(grouped['Love']), 1)
    
    def test_create_frequency_spectrum_dataset(self):
        """Test creating frequency spectrum dataset."""
        dataset = self.plotter._create_frequency_spectrum_dataset('P', [self.p_wave])
        
        self.assertEqual(dataset['label'], 'P Wave Spectrum')
        self.assertEqual(dataset['wave_type'], 'P')
        self.assertEqual(dataset['wave_count'], 1)
        self.assertIn('borderColor', dataset)
        self.assertIn('data', dataset)
        self.assertGreater(len(dataset['data']), 0)
        
        # Check data format
        data_point = dataset['data'][0]
        self.assertIn('x', data_point)  # Frequency
        self.assertIn('y', data_point)  # Power
        self.assertIsInstance(data_point['x'], (int, float))
        self.assertIsInstance(data_point['y'], (int, float))
    
    def test_create_psd_dataset(self):
        """Test creating PSD dataset."""
        dataset = self.plotter._create_psd_dataset('P', [self.p_wave])
        
        self.assertEqual(dataset['label'], 'P Wave PSD')
        self.assertEqual(dataset['wave_type'], 'P')
        self.assertFalse(dataset['fill'])  # PSD plots typically not filled
        
        # Check that data is in dB scale
        data_values = [point['y'] for point in dataset['data']]
        # dB values can be negative, so just check they're reasonable
        self.assertTrue(all(isinstance(val, (int, float)) for val in data_values))
    
    def test_create_frequency_markers(self):
        """Test creating frequency markers."""
        waves = [self.p_wave, self.s_wave]
        markers = self.plotter._create_frequency_markers(waves)
        
        self.assertEqual(len(markers), 2)
        
        # Check P-wave marker
        p_marker = markers[0]
        self.assertEqual(p_marker['value'], self.p_wave.dominant_frequency)
        self.assertEqual(p_marker['type'], 'line')
        self.assertEqual(p_marker['mode'], 'vertical')
        self.assertIn('P', p_marker['label']['content'])
    
    def test_create_spectrogram_markers(self):
        """Test creating spectrogram markers."""
        times = np.linspace(0, 10, 100)  # 10 second time range
        markers = self.plotter._create_spectrogram_markers([self.p_wave], times)
        
        self.assertEqual(len(markers), 1)
        marker = markers[0]
        self.assertEqual(marker['type'], 'line')
        self.assertEqual(marker['value'], self.p_wave.arrival_time)
        self.assertEqual(marker['borderColor'], 'white')
    
    def test_create_frequency_band_annotations(self):
        """Test creating frequency band annotations."""
        annotations = self.plotter._create_frequency_band_annotations()
        
        self.assertEqual(len(annotations), 3)  # P, S, Surface wave bands
        
        # Check that bands have expected properties
        for annotation in annotations:
            self.assertEqual(annotation['type'], 'box')
            self.assertIn('xMin', annotation)
            self.assertIn('xMax', annotation)
            self.assertIn('backgroundColor', annotation)
            self.assertIn('band', annotation['label']['content'])
    
    def test_frequency_layout_config(self):
        """Test frequency layout configuration."""
        layout = self.plotter._create_frequency_layout_config()
        
        self.assertIn('responsive', layout)
        self.assertIn('scales', layout)
        
        # Check logarithmic frequency axis
        x_scale = layout['scales']['x']
        self.assertEqual(x_scale['type'], 'logarithmic')
        self.assertEqual(x_scale['title']['text'], 'Frequency (Hz)')
        
        # Check zoom and pan configuration
        zoom_config = layout['plugins']['zoom']
        self.assertTrue(zoom_config['zoom']['wheel']['enabled'])
        self.assertTrue(zoom_config['pan']['enabled'])
        self.assertEqual(zoom_config['zoom']['mode'], 'xy')
    
    def test_spectrogram_layout_config(self):
        """Test spectrogram layout configuration."""
        layout = self.plotter._create_spectrogram_layout_config(self.p_wave)
        
        self.assertIn('title', layout)
        self.assertIn('P Wave', layout['title'])
        
        # Check axes configuration
        self.assertEqual(layout['xaxis']['title'], 'Time (seconds)')
        self.assertEqual(layout['yaxis']['title'], 'Frequency (Hz)')
        
        # Check time range matches wave
        self.assertEqual(layout['xaxis']['range'][0], self.p_wave.start_time)
        self.assertEqual(layout['xaxis']['range'][1], self.p_wave.end_time)
    
    def test_psd_layout_config(self):
        """Test PSD layout configuration."""
        layout = self.plotter._create_psd_layout_config()
        
        # Check PSD-specific y-axis label
        y_scale = layout['scales']['y']
        self.assertEqual(y_scale['title']['text'], 'Power Spectral Density (dB/Hz)')
        
        # Check logarithmic frequency axis
        x_scale = layout['scales']['x']
        self.assertEqual(x_scale['type'], 'logarithmic')
    
    def test_set_spectrogram_config(self):
        """Test setting custom spectrogram configuration."""
        custom_config = {
            'window': 'hamming',
            'nperseg': 512,
            'log_scale': False
        }
        
        self.plotter.set_spectrogram_config(custom_config)
        
        self.assertEqual(self.plotter.spectrogram_config['window'], 'hamming')
        self.assertEqual(self.plotter.spectrogram_config['nperseg'], 512)
        self.assertFalse(self.plotter.spectrogram_config['log_scale'])
        # Other settings should be preserved
        self.assertIn('noverlap', self.plotter.spectrogram_config)
    
    def test_set_spectrum_config(self):
        """Test setting custom spectrum configuration."""
        custom_config = {
            'window': 'blackman',
            'detrend': 'constant'
        }
        
        self.plotter.set_spectrum_config(custom_config)
        
        self.assertEqual(self.plotter.spectrum_config['window'], 'blackman')
        self.assertEqual(self.plotter.spectrum_config['detrend'], 'constant')
        # Other settings should be preserved
        self.assertIn('scaling', self.plotter.spectrum_config)
    
    def test_set_color_scheme(self):
        """Test setting custom color scheme."""
        custom_colors = {
            'P': '#FF0000',
            'S': '#00FF00'
        }
        
        self.plotter.set_color_scheme(custom_colors)
        
        self.assertEqual(self.plotter.default_colors['P'], '#FF0000')
        self.assertEqual(self.plotter.default_colors['S'], '#00FF00')
        # Original colors should be preserved
        self.assertIn('Love', self.plotter.default_colors)
    
    def test_export_frequency_data(self):
        """Test exporting frequency data."""
        waves = [self.p_wave, self.s_wave]
        json_export = self.plotter.export_frequency_data(waves, 'json')
        
        # Should be valid JSON
        parsed = json.loads(json_export)
        
        # Check structure
        self.assertGreater(len(parsed), 0)
        
        # Check that each wave has frequency data
        for key, data in parsed.items():
            self.assertIn('frequencies', data)
            self.assertIn('power_spectral_density', data)
            self.assertIn('dominant_frequency', data)
            self.assertIn('wave_type', data)
            
            # Check data types
            self.assertIsInstance(data['frequencies'], list)
            self.assertIsInstance(data['power_spectral_density'], list)
            self.assertIsInstance(data['dominant_frequency'], (int, float))
            self.assertIsInstance(data['wave_type'], str)
    
    def test_export_frequency_data_invalid_format(self):
        """Test exporting frequency data with invalid format."""
        with self.assertRaises(ValueError):
            self.plotter.export_frequency_data([self.p_wave], 'invalid_format')
    
    def test_time_series_plot_delegation(self):
        """Test that time series plot method delegates properly."""
        result = self.plotter.create_time_series_plot([self.p_wave])
        
        self.assertEqual(result['type'], 'time_series')
        self.assertIn('message', result)
        self.assertIn('TimeSeriesPlotter', result['message'])


if __name__ == '__main__':
    unittest.main()