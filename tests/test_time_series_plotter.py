"""
Unit tests for TimeSeriesPlotter class.

Tests the time-series visualization functionality for different wave types
with synthetic wave data.
"""

import unittest
import numpy as np
from datetime import datetime
import json

from wave_analysis.models import WaveSegment, WaveAnalysisResult
from wave_analysis.services.time_series_plotter import TimeSeriesPlotter


class TestTimeSeriesPlotter(unittest.TestCase):
    """Test cases for TimeSeriesPlotter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plotter = TimeSeriesPlotter()
        
        # Create synthetic wave data
        self.sampling_rate = 100.0  # 100 Hz
        self.duration = 10.0  # 10 seconds
        self.time_points = np.linspace(0, self.duration, int(self.duration * self.sampling_rate))
        
        # Create synthetic P-wave
        p_wave_data = self._create_synthetic_p_wave()
        self.p_wave = WaveSegment(
            wave_type='P',
            start_time=2.0,
            end_time=4.0,
            data=p_wave_data,
            sampling_rate=self.sampling_rate,
            peak_amplitude=np.max(np.abs(p_wave_data)),
            dominant_frequency=8.0,
            arrival_time=2.5,
            confidence=0.9
        )
        
        # Create synthetic S-wave
        s_wave_data = self._create_synthetic_s_wave()
        self.s_wave = WaveSegment(
            wave_type='S',
            start_time=5.0,
            end_time=7.0,
            data=s_wave_data,
            sampling_rate=self.sampling_rate,
            peak_amplitude=np.max(np.abs(s_wave_data)),
            dominant_frequency=4.0,
            arrival_time=5.5,
            confidence=0.85
        )
        
        # Create synthetic surface wave
        surface_wave_data = self._create_synthetic_surface_wave()
        self.surface_wave = WaveSegment(
            wave_type='Love',
            start_time=8.0,
            end_time=10.0,
            data=surface_wave_data,
            sampling_rate=self.sampling_rate,
            peak_amplitude=np.max(np.abs(surface_wave_data)),
            dominant_frequency=2.0,
            arrival_time=8.5,
            confidence=0.8
        )
        
        # Create complete wave analysis result
        original_data = self._create_synthetic_earthquake()
        self.wave_result = WaveAnalysisResult(
            original_data=original_data,
            sampling_rate=self.sampling_rate,
            p_waves=[self.p_wave],
            s_waves=[self.s_wave],
            surface_waves=[self.surface_wave]
        )
    
    def _create_synthetic_p_wave(self) -> np.ndarray:
        """Create synthetic P-wave data."""
        duration = 2.0
        samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # High-frequency onset with exponential decay
        frequency = 8.0
        envelope = np.exp(-3 * t)
        wave = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        noise = 0.1 * np.random.normal(0, 1, samples)
        return wave + noise
    
    def _create_synthetic_s_wave(self) -> np.ndarray:
        """Create synthetic S-wave data."""
        duration = 2.0
        samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # Lower frequency with longer duration
        frequency = 4.0
        envelope = np.exp(-1.5 * t)
        wave = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        noise = 0.1 * np.random.normal(0, 1, samples)
        return wave + noise
    
    def _create_synthetic_surface_wave(self) -> np.ndarray:
        """Create synthetic surface wave data."""
        duration = 2.0
        samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, samples)
        
        # Very low frequency with long duration
        frequency = 2.0
        envelope = np.exp(-0.5 * t)
        wave = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        noise = 0.05 * np.random.normal(0, 1, samples)
        return wave + noise
    
    def _create_synthetic_earthquake(self) -> np.ndarray:
        """Create synthetic complete earthquake signal."""
        samples = int(self.duration * self.sampling_rate)
        signal = np.zeros(samples)
        
        # Add P-wave
        p_start = int(2.0 * self.sampling_rate)
        p_end = int(4.0 * self.sampling_rate)
        signal[p_start:p_end] += self._create_synthetic_p_wave()
        
        # Add S-wave
        s_start = int(5.0 * self.sampling_rate)
        s_end = int(7.0 * self.sampling_rate)
        signal[s_start:s_end] += self._create_synthetic_s_wave()
        
        # Add surface wave
        surf_start = int(8.0 * self.sampling_rate)
        surf_end = int(10.0 * self.sampling_rate)
        signal[surf_start:surf_end] += self._create_synthetic_surface_wave()
        
        # Add background noise
        noise = 0.05 * np.random.normal(0, 1, samples)
        return signal + noise
    
    def test_create_time_series_plot_single_wave(self):
        """Test creating time series plot for single wave segment."""
        plot_data = self.plotter.create_time_series_plot([self.p_wave])
        
        # Check basic structure
        self.assertEqual(plot_data['type'], 'time_series')
        self.assertIn('datasets', plot_data)
        self.assertIn('layout', plot_data)
        self.assertIn('config', plot_data)
        
        # Check dataset
        self.assertEqual(len(plot_data['datasets']), 1)
        dataset = plot_data['datasets'][0]
        self.assertEqual(dataset['label'], 'P-waves')
        self.assertEqual(dataset['wave_type'], 'P')
        self.assertEqual(dataset['wave_count'], 1)
        
        # Check data points
        self.assertGreater(len(dataset['data']), 0)
        self.assertIn('x', dataset['data'][0])
        self.assertIn('y', dataset['data'][0])
    
    def test_create_time_series_plot_multiple_waves(self):
        """Test creating time series plot for multiple wave segments."""
        waves = [self.p_wave, self.s_wave, self.surface_wave]
        plot_data = self.plotter.create_time_series_plot(waves)
        
        # Check that all wave types are represented
        self.assertEqual(len(plot_data['datasets']), 3)
        
        wave_types = [dataset['wave_type'] for dataset in plot_data['datasets']]
        self.assertIn('P', wave_types)
        self.assertIn('S', wave_types)
        self.assertIn('Love', wave_types)
        
        # Check arrival markers
        self.assertIn('annotations', plot_data)
        self.assertEqual(len(plot_data['annotations']), 3)
    
    def test_create_time_series_plot_empty_input(self):
        """Test creating plot with empty wave list."""
        plot_data = self.plotter.create_time_series_plot([])
        
        self.assertEqual(plot_data['type'], 'empty')
        self.assertIn('message', plot_data)
        self.assertEqual(len(plot_data['datasets']), 0)
    
    def test_create_p_wave_plot(self):
        """Test creating specialized P-wave plot."""
        plot_data = self.plotter.create_p_wave_plot([self.p_wave])
        
        self.assertEqual(plot_data['type'], 'p_wave_analysis')
        self.assertEqual(plot_data['title'], 'P-Wave Analysis')
        self.assertEqual(len(plot_data['datasets']), 1)
        
        # Check for onset markers
        self.assertIn('annotations', plot_data)
        self.assertGreater(len(plot_data['annotations']), 0)
    
    def test_create_s_wave_plot(self):
        """Test creating specialized S-wave plot."""
        plot_data = self.plotter.create_s_wave_plot([self.s_wave])
        
        self.assertEqual(plot_data['type'], 's_wave_analysis')
        self.assertEqual(plot_data['title'], 'S-Wave Analysis')
        self.assertEqual(len(plot_data['datasets']), 1)
    
    def test_create_surface_wave_plot(self):
        """Test creating specialized surface wave plot."""
        plot_data = self.plotter.create_surface_wave_plot([self.surface_wave])
        
        self.assertEqual(plot_data['type'], 'surface_wave_analysis')
        self.assertEqual(plot_data['title'], 'Surface Wave Analysis')
        self.assertEqual(len(plot_data['datasets']), 1)
    
    def test_create_multi_wave_comparison(self):
        """Test creating multi-wave comparison plot."""
        plot_data = self.plotter.create_multi_wave_comparison(self.wave_result)
        
        self.assertEqual(plot_data['type'], 'multi_wave_comparison')
        self.assertEqual(plot_data['title'], 'Complete Wave Analysis')
        
        # Should have original waveform plus all wave types
        self.assertGreaterEqual(len(plot_data['datasets']), 4)  # Original + P + S + Surface
        
        # Check for phase annotations
        self.assertIn('annotations', plot_data)
        self.assertGreater(len(plot_data['annotations']), 0)
    
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
    
    def test_create_wave_dataset(self):
        """Test creating dataset for specific wave type."""
        dataset = self.plotter._create_wave_dataset('P', [self.p_wave])
        
        self.assertEqual(dataset['label'], 'P-waves')
        self.assertEqual(dataset['wave_type'], 'P')
        self.assertEqual(dataset['wave_count'], 1)
        self.assertIn('borderColor', dataset)
        self.assertIn('data', dataset)
        self.assertGreater(len(dataset['data']), 0)
    
    def test_create_arrival_markers(self):
        """Test creating arrival time markers."""
        waves = [self.p_wave, self.s_wave]
        markers = self.plotter._create_arrival_markers(waves)
        
        self.assertEqual(len(markers), 2)
        
        # Check P-wave marker
        p_marker = markers[0]
        self.assertEqual(p_marker['value'], self.p_wave.arrival_time)
        self.assertEqual(p_marker['type'], 'line')
        self.assertEqual(p_marker['mode'], 'vertical')
    
    def test_create_onset_markers(self):
        """Test creating onset markers."""
        markers = self.plotter._create_onset_markers([self.p_wave])
        
        self.assertEqual(len(markers), 1)
        marker = markers[0]
        self.assertEqual(marker['type'], 'point')
        self.assertEqual(marker['xValue'], self.p_wave.arrival_time)
        self.assertEqual(marker['yValue'], self.p_wave.peak_amplitude)
    
    def test_layout_config(self):
        """Test layout configuration creation."""
        layout = self.plotter._create_layout_config()
        
        self.assertIn('responsive', layout)
        self.assertIn('scales', layout)
        self.assertIn('plugins', layout)
        
        # Check axes configuration
        self.assertIn('x', layout['scales'])
        self.assertIn('y', layout['scales'])
        self.assertIn('y1', layout['scales'])  # Secondary axis
        
        # Check zoom and pan configuration
        zoom_config = layout['plugins']['zoom']
        self.assertTrue(zoom_config['zoom']['wheel']['enabled'])
        self.assertTrue(zoom_config['pan']['enabled'])
    
    def test_plot_config(self):
        """Test plot configuration creation."""
        config = self.plotter._create_plot_config()
        
        self.assertIn('responsive', config)
        self.assertIn('interaction', config)
        self.assertIn('elements', config)
        
        # Check interaction settings
        self.assertEqual(config['interaction']['mode'], 'index')
        self.assertFalse(config['interaction']['intersect'])
    
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
    
    def test_set_plot_config(self):
        """Test setting custom plot configuration."""
        custom_config = {
            'line_width': 3,
            'grid': False
        }
        
        self.plotter.set_plot_config(custom_config)
        
        self.assertEqual(self.plotter.plot_config['line_width'], 3)
        self.assertFalse(self.plotter.plot_config['grid'])
        # Other settings should be preserved
        self.assertIn('marker_size', self.plotter.plot_config)
    
    def test_export_plot_data_json(self):
        """Test exporting plot data as JSON."""
        plot_data = self.plotter.create_time_series_plot([self.p_wave])
        json_export = self.plotter.export_plot_data(plot_data, 'json')
        
        # Should be valid JSON
        parsed = json.loads(json_export)
        self.assertEqual(parsed['type'], 'time_series')
        self.assertIn('datasets', parsed)
    
    def test_export_plot_data_invalid_format(self):
        """Test exporting plot data with invalid format."""
        plot_data = self.plotter.create_time_series_plot([self.p_wave])
        
        with self.assertRaises(ValueError):
            self.plotter.export_plot_data(plot_data, 'invalid_format')
    
    def test_wave_segment_validation(self):
        """Test that wave segments are properly validated."""
        # Test with valid wave segment
        plot_data = self.plotter.create_time_series_plot([self.p_wave])
        self.assertGreater(len(plot_data['datasets']), 0)
        
        # Test with multiple wave segments of same type
        p_wave2 = WaveSegment(
            wave_type='P',
            start_time=3.0,
            end_time=4.5,
            data=self._create_synthetic_p_wave(),
            sampling_rate=self.sampling_rate,
            peak_amplitude=0.5,
            dominant_frequency=7.0,
            arrival_time=3.2,
            confidence=0.8
        )
        
        plot_data = self.plotter.create_time_series_plot([self.p_wave, p_wave2])
        
        # Should combine both P-waves into single dataset
        p_datasets = [d for d in plot_data['datasets'] if d['wave_type'] == 'P']
        self.assertEqual(len(p_datasets), 1)
        self.assertEqual(p_datasets[0]['wave_count'], 2)
    
    def test_characteristic_function_trace(self):
        """Test creating characteristic function trace."""
        # Add characteristic function to P-wave metadata
        char_func = np.random.random(200)  # Synthetic characteristic function
        self.p_wave.metadata['characteristic_function'] = char_func
        
        trace = self.plotter._create_characteristic_function_trace([self.p_wave])
        
        self.assertIsNotNone(trace)
        self.assertEqual(trace['label'], 'Characteristic Function')
        self.assertEqual(trace['yAxisID'], 'y1')  # Secondary y-axis
        self.assertEqual(len(trace['data']), len(char_func))
    
    def test_polarization_trace(self):
        """Test creating polarization analysis traces."""
        # Add polarization components to S-wave metadata
        h_component = np.random.random(200)
        v_component = np.random.random(200)
        self.s_wave.metadata['horizontal_component'] = h_component
        self.s_wave.metadata['vertical_component'] = v_component
        
        traces = self.plotter._create_polarization_trace([self.s_wave])
        
        self.assertEqual(len(traces), 2)  # Horizontal and vertical components
        self.assertEqual(traces[0]['label'], 'Horizontal Component')
        self.assertEqual(traces[1]['label'], 'Vertical Component')


if __name__ == '__main__':
    unittest.main()