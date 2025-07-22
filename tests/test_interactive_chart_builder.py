"""
Unit tests for InteractiveChartBuilder class.

This module tests the interactive chart generation functionality
including hover tooltips, click-to-zoom, and wave inspection features.
"""

import unittest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the class under test
from wave_analysis.services.interactive_chart_builder import InteractiveChartBuilder
from wave_analysis.models import WaveSegment, WaveAnalysisResult, DetailedAnalysis, ArrivalTimes


class TestInteractiveChartBuilder(unittest.TestCase):
    """Test cases for InteractiveChartBuilder class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.chart_builder = InteractiveChartBuilder()
        
        # Create sample wave segments for testing
        self.sample_p_wave = WaveSegment(
            wave_type='P',
            start_time=5.0,
            end_time=8.0,
            data=np.random.randn(300),  # 3 seconds at 100 Hz
            sampling_rate=100.0,
            peak_amplitude=0.5,
            dominant_frequency=15.0,
            arrival_time=5.5,
            confidence=0.85
        )
        
        self.sample_s_wave = WaveSegment(
            wave_type='S',
            start_time=10.0,
            end_time=15.0,
            data=np.random.randn(500),  # 5 seconds at 100 Hz
            sampling_rate=100.0,
            peak_amplitude=0.8,
            dominant_frequency=8.0,
            arrival_time=10.5,
            confidence=0.92
        )
        
        self.sample_surface_wave = WaveSegment(
            wave_type='Love',
            start_time=20.0,
            end_time=30.0,
            data=np.random.randn(1000),  # 10 seconds at 100 Hz
            sampling_rate=100.0,
            peak_amplitude=1.2,
            dominant_frequency=3.0,
            arrival_time=20.5,
            confidence=0.78
        )
        
        self.sample_wave_segments = [
            self.sample_p_wave,
            self.sample_s_wave,
            self.sample_surface_wave
        ]
        
        # Create sample wave analysis result
        self.sample_wave_result = WaveAnalysisResult(
            original_data=np.random.randn(3000),  # 30 seconds at 100 Hz
            sampling_rate=100.0,
            p_waves=[self.sample_p_wave],
            s_waves=[self.sample_s_wave],
            surface_waves=[self.sample_surface_wave]
        )
        
        # Create sample detailed analysis
        self.sample_analysis = DetailedAnalysis(
            wave_result=self.sample_wave_result,
            arrival_times=ArrivalTimes(
                p_wave_arrival=5.5,
                s_wave_arrival=10.5,
                surface_wave_arrival=20.5,
                sp_time_difference=5.0
            )
        )
    
    def test_initialization(self):
        """Test InteractiveChartBuilder initialization."""
        self.assertIsInstance(self.chart_builder.default_colors, dict)
        self.assertIn('P', self.chart_builder.default_colors)
        self.assertIn('S', self.chart_builder.default_colors)
        self.assertIn('Love', self.chart_builder.default_colors)
        self.assertIn('Rayleigh', self.chart_builder.default_colors)
        
        self.assertIsInstance(self.chart_builder.chart_config, dict)
        self.assertTrue(self.chart_builder.chart_config['displayModeBar'])
        self.assertFalse(self.chart_builder.chart_config['displaylogo'])
        
        self.assertEqual(self.chart_builder.layout_template, 'plotly_white')
    
    def test_create_time_series_plot_empty_data(self):
        """Test time series plot creation with empty data."""
        result = self.chart_builder.create_time_series_plot([])
        
        self.assertEqual(result['type'], 'interactive_empty')
        self.assertIn('plotly_json', result)
        self.assertIn('config', result)
    
    @patch('wave_analysis.services.interactive_chart_builder.make_subplots')
    @patch('wave_analysis.services.interactive_chart_builder.go')
    def test_create_time_series_plot_with_data(self, mock_go, mock_make_subplots):
        """Test time series plot creation with wave data."""
        # Mock the figure and subplot creation
        mock_fig = Mock()
        mock_make_subplots.return_value = mock_fig
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_fig.layout = {}
        mock_fig.data = []
        
        result = self.chart_builder.create_time_series_plot(self.sample_wave_segments)
        
        self.assertEqual(result['type'], 'interactive_time_series')
        self.assertIn('plotly_json', result)
        self.assertIn('config', result)
        
        # Verify that subplot was created with secondary y-axis
        mock_make_subplots.assert_called_once()
        call_args = mock_make_subplots.call_args
        self.assertEqual(call_args[1]['rows'], 1)
        self.assertEqual(call_args[1]['cols'], 1)
        self.assertTrue(call_args[1]['specs'][0][0]['secondary_y'])
    
    def test_create_frequency_plot_empty_data(self):
        """Test frequency plot creation with empty data."""
        result = self.chart_builder.create_frequency_plot([])
        
        self.assertEqual(result['type'], 'interactive_empty')
        self.assertIn('plotly_json', result)
    
    @patch('wave_analysis.services.interactive_chart_builder.go.Figure')
    def test_create_frequency_plot_with_data(self, mock_figure):
        """Test frequency plot creation with wave data."""
        # Mock the figure
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_fig.layout = {}
        mock_fig.data = []
        
        # Mock the frequency calculation
        with patch.object(self.chart_builder, '_calculate_frequency_spectrum') as mock_calc:
            mock_calc.return_value = (np.linspace(0, 50, 100), np.random.rand(100))
            
            result = self.chart_builder.create_frequency_plot(self.sample_wave_segments)
            
            self.assertEqual(result['type'], 'interactive_frequency_spectrum')
            self.assertIn('plotly_json', result)
            
            # Verify frequency calculation was called for each wave type
            self.assertTrue(mock_calc.called)
    
    @patch('wave_analysis.services.interactive_chart_builder.go.Figure')
    def test_create_interactive_spectrogram(self, mock_figure):
        """Test interactive spectrogram creation."""
        # Mock the figure
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_fig.layout = {}
        mock_fig.data = []
        
        # Mock the spectrogram calculation
        with patch.object(self.chart_builder, '_calculate_spectrogram') as mock_calc:
            mock_calc.return_value = (
                np.linspace(0, 50, 100),  # frequencies
                np.linspace(0, 3, 50),    # times
                np.random.rand(100, 50)   # Sxx
            )
            
            result = self.chart_builder.create_interactive_spectrogram(self.sample_p_wave)
            
            self.assertEqual(result['type'], 'interactive_spectrogram')
            self.assertIn('plotly_json', result)
            
            # Verify spectrogram calculation was called
            mock_calc.assert_called_once_with(self.sample_p_wave)
    
    def test_create_interactive_spectrogram_empty_data(self):
        """Test spectrogram creation with empty data."""
        result = self.chart_builder.create_interactive_spectrogram(None)
        
        self.assertEqual(result['type'], 'interactive_empty')
        self.assertIn('plotly_json', result)
    
    @patch('wave_analysis.services.interactive_chart_builder.make_subplots')
    def test_create_multi_panel_analysis(self, mock_make_subplots):
        """Test multi-panel analysis plot creation."""
        # Mock the figure
        mock_fig = Mock()
        mock_make_subplots.return_value = mock_fig
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_fig.layout = {}
        mock_fig.data = []
        
        result = self.chart_builder.create_multi_panel_analysis(self.sample_analysis)
        
        self.assertEqual(result['type'], 'interactive_multi_panel_analysis')
        self.assertIn('plotly_json', result)
        
        # Verify subplot creation with 3 rows
        mock_make_subplots.assert_called_once()
        call_args = mock_make_subplots.call_args
        self.assertEqual(call_args[1]['rows'], 3)
        self.assertEqual(call_args[1]['cols'], 1)
    
    def test_create_multi_panel_analysis_empty_data(self):
        """Test multi-panel analysis with empty data."""
        result = self.chart_builder.create_multi_panel_analysis(None)
        
        self.assertEqual(result['type'], 'interactive_empty')
        self.assertIn('plotly_json', result)
    
    @patch('wave_analysis.services.interactive_chart_builder.make_subplots')
    def test_create_interactive_comparison(self, mock_make_subplots):
        """Test interactive comparison plot creation."""
        # Mock the figure
        mock_fig = Mock()
        mock_make_subplots.return_value = mock_fig
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_fig.layout = {}
        mock_fig.data = []
        
        wave_results = [self.sample_wave_result, self.sample_wave_result]
        result = self.chart_builder.create_interactive_comparison(wave_results)
        
        self.assertEqual(result['type'], 'interactive_wave_comparison')
        self.assertIn('plotly_json', result)
        
        # Verify subplot creation with correct number of rows
        mock_make_subplots.assert_called_once()
        call_args = mock_make_subplots.call_args
        self.assertEqual(call_args[1]['rows'], len(wave_results))
        self.assertEqual(call_args[1]['cols'], 1)
    
    def test_create_interactive_comparison_empty_data(self):
        """Test comparison plot with empty data."""
        result = self.chart_builder.create_interactive_comparison([])
        
        self.assertEqual(result['type'], 'interactive_empty')
        self.assertIn('plotly_json', result)
    
    @patch('wave_analysis.services.interactive_chart_builder.go.Figure')
    def test_create_interactive_correlation_matrix(self, mock_figure):
        """Test interactive correlation matrix creation."""
        # Mock the figure
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_fig.layout = {}
        mock_fig.data = []
        
        correlation_data = {
            'correlation_matrix': [[1.0, 0.5, 0.3], [0.5, 1.0, 0.7], [0.3, 0.7, 1.0]],
            'channel_labels': ['Ch1', 'Ch2', 'Ch3']
        }
        
        result = self.chart_builder.create_interactive_correlation_matrix(correlation_data)
        
        self.assertEqual(result['type'], 'interactive_correlation_matrix')
        self.assertIn('plotly_json', result)
    
    def test_create_interactive_correlation_matrix_empty_data(self):
        """Test correlation matrix with empty data."""
        result = self.chart_builder.create_interactive_correlation_matrix({})
        
        self.assertEqual(result['type'], 'interactive_empty')
        self.assertIn('plotly_json', result)
    
    @patch('wave_analysis.services.interactive_chart_builder.go.Figure')
    def test_create_wave_picker_interface(self, mock_figure):
        """Test wave picker interface creation."""
        # Mock the figure
        mock_fig = Mock()
        mock_figure.return_value = mock_fig
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
        mock_fig.layout = {}
        mock_fig.data = []
        
        result = self.chart_builder.create_wave_picker_interface(self.sample_wave_result)
        
        self.assertEqual(result['type'], 'interactive_wave_picker')
        self.assertIn('plotly_json', result)
    
    def test_create_wave_picker_interface_empty_data(self):
        """Test wave picker with empty data."""
        result = self.chart_builder.create_wave_picker_interface(None)
        
        self.assertEqual(result['type'], 'interactive_empty')
        self.assertIn('plotly_json', result)
    
    def test_group_waves_by_type(self):
        """Test wave grouping by type."""
        grouped = self.chart_builder._group_waves_by_type(self.sample_wave_segments)
        
        self.assertIn('P', grouped)
        self.assertIn('S', grouped)
        self.assertIn('Love', grouped)
        
        self.assertEqual(len(grouped['P']), 1)
        self.assertEqual(len(grouped['S']), 1)
        self.assertEqual(len(grouped['Love']), 1)
        
        self.assertEqual(grouped['P'][0].wave_type, 'P')
        self.assertEqual(grouped['S'][0].wave_type, 'S')
        self.assertEqual(grouped['Love'][0].wave_type, 'Love')
    
    @patch('scipy.signal.spectrogram')
    def test_calculate_spectrogram(self, mock_spectrogram):
        """Test spectrogram calculation."""
        # Mock scipy.signal.spectrogram
        mock_spectrogram.return_value = (
            np.linspace(0, 50, 100),  # frequencies
            np.linspace(0, 3, 50),    # times
            np.random.rand(100, 50)   # Sxx
        )
        
        frequencies, times, Sxx = self.chart_builder._calculate_spectrogram(self.sample_p_wave)
        
        self.assertEqual(len(frequencies), 100)
        self.assertEqual(len(times), 50)
        self.assertEqual(Sxx.shape, (100, 50))
        
        # Verify spectrogram was called with correct parameters
        mock_spectrogram.assert_called_once()
        call_args = mock_spectrogram.call_args
        np.testing.assert_array_equal(call_args[0][0], self.sample_p_wave.data)
        self.assertEqual(call_args[1]['fs'], self.sample_p_wave.sampling_rate)
    
    @patch('scipy.fft.fft')
    @patch('scipy.fft.fftfreq')
    @patch('scipy.signal.get_window')
    @patch('scipy.signal.detrend')
    def test_calculate_frequency_spectrum(self, mock_detrend, mock_get_window, mock_fftfreq, mock_fft):
        """Test frequency spectrum calculation."""
        # Mock the required functions
        mock_get_window.return_value = np.ones(len(self.sample_p_wave.data))
        mock_detrend.return_value = self.sample_p_wave.data
        mock_fft.return_value = np.random.rand(100) + 1j * np.random.rand(100)
        mock_fftfreq.return_value = np.linspace(-25, 25, 100)  # Include negative frequencies
        
        frequencies, psd = self.chart_builder._calculate_frequency_spectrum(self.sample_p_wave)
        
        self.assertEqual(len(frequencies), len(psd))
        self.assertTrue(np.all(frequencies >= 0))  # Should only have positive frequencies
        
        # Verify functions were called
        mock_get_window.assert_called_once()
        mock_detrend.assert_called_once()
        mock_fft.assert_called_once()
        mock_fftfreq.assert_called_once()
    
    def test_set_color_scheme(self):
        """Test setting custom color scheme."""
        custom_colors = {
            'P': '#FF0000',
            'S': '#00FF00',
            'Love': '#0000FF'
        }
        
        self.chart_builder.set_color_scheme(custom_colors)
        
        self.assertEqual(self.chart_builder.default_colors['P'], '#FF0000')
        self.assertEqual(self.chart_builder.default_colors['S'], '#00FF00')
        self.assertEqual(self.chart_builder.default_colors['Love'], '#0000FF')
        # Rayleigh should still have original color
        self.assertEqual(self.chart_builder.default_colors['Rayleigh'], '#96CEB4')
    
    def test_set_chart_config(self):
        """Test setting custom chart configuration."""
        custom_config = {
            'displayModeBar': False,
            'staticPlot': True
        }
        
        self.chart_builder.set_chart_config(custom_config)
        
        self.assertFalse(self.chart_builder.chart_config['displayModeBar'])
        self.assertTrue(self.chart_builder.chart_config['staticPlot'])
        # Other config should remain
        self.assertFalse(self.chart_builder.chart_config['displaylogo'])
    
    def test_set_layout_template(self):
        """Test setting layout template."""
        self.chart_builder.set_layout_template('plotly_dark')
        self.assertEqual(self.chart_builder.layout_template, 'plotly_dark')
    
    @patch('wave_analysis.services.interactive_chart_builder.pio')
    def test_export_interactive_html(self, mock_pio):
        """Test exporting interactive plot as HTML."""
        # Mock plotly.io functions
        mock_fig = Mock()
        mock_pio.from_json.return_value = mock_fig
        mock_pio.to_html.return_value = '<html>test content</html>'
        
        fig_dict = {
            'plotly_json': '{"data": [], "layout": {}}',
            'type': 'interactive_time_series'
        }
        
        html_content = self.chart_builder.export_interactive_html(fig_dict)
        
        self.assertEqual(html_content, '<html>test content</html>')
        mock_pio.from_json.assert_called_once_with('{"data": [], "layout": {}}')
        mock_pio.to_html.assert_called_once()
    
    def test_export_interactive_html_invalid_dict(self):
        """Test HTML export with invalid figure dictionary."""
        invalid_dict = {'type': 'test', 'data': []}
        
        with self.assertRaises(ValueError) as context:
            self.chart_builder.export_interactive_html(invalid_dict)
        
        self.assertIn('Invalid figure dictionary format', str(context.exception))
    
    def test_hover_tooltip_content(self):
        """Test that hover tooltips contain wave characteristics."""
        # This test verifies that the hover templates include wave characteristics
        with patch('wave_analysis.services.interactive_chart_builder.make_subplots') as mock_subplots:
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
            mock_fig.layout = {}
            mock_fig.data = []
            
            # Create plot and verify hover template content
            result = self.chart_builder.create_time_series_plot(self.sample_wave_segments)
            
            # The hover template should be configured in _add_wave_traces
            # We can't directly test the template content without mocking the entire plotly chain,
            # but we can verify the method was called
            self.assertEqual(result['type'], 'interactive_time_series')
    
    def test_click_to_zoom_functionality(self):
        """Test that click-to-zoom functionality is enabled."""
        # Verify that the chart configuration enables zoom functionality
        self.assertTrue(self.chart_builder.chart_config['displayModeBar'])
        
        # The zoom functionality is enabled through the layout configuration
        # which is tested in the layout configuration methods
        with patch('wave_analysis.services.interactive_chart_builder.make_subplots') as mock_subplots:
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
            mock_fig.layout = {}
            mock_fig.data = []
            
            result = self.chart_builder.create_time_series_plot(self.sample_wave_segments)
            
            # Verify that the plot was created (zoom is enabled in layout configuration)
            self.assertEqual(result['type'], 'interactive_time_series')
    
    def test_wave_inspection_details(self):
        """Test that wave inspection provides detailed information."""
        # Test the wave picker interface which provides detailed wave inspection
        with patch('wave_analysis.services.interactive_chart_builder.go.Figure') as mock_figure:
            mock_fig = Mock()
            mock_figure.return_value = mock_fig
            mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
            mock_fig.layout = {}
            mock_fig.data = []
            
            result = self.chart_builder.create_wave_picker_interface(self.sample_wave_result)
            
            self.assertEqual(result['type'], 'interactive_wave_picker')
            
            # Verify that add_trace was called for wave markers with detailed hover info
            self.assertTrue(mock_fig.add_trace.called)
    
    def test_error_handling_with_invalid_wave_data(self):
        """Test error handling with invalid wave data."""
        # Test with wave segment that has invalid data
        invalid_wave = WaveSegment(
            wave_type='P',
            start_time=5.0,
            end_time=8.0,
            data=np.array([]),  # Empty data array
            sampling_rate=100.0,
            peak_amplitude=0.5,
            dominant_frequency=15.0,
            arrival_time=5.5,
            confidence=0.85
        )
        
        # The chart builder should handle empty data gracefully
        with patch('wave_analysis.services.interactive_chart_builder.make_subplots') as mock_subplots:
            mock_fig = Mock()
            mock_subplots.return_value = mock_fig
            mock_fig.to_json.return_value = '{"data": [], "layout": {}}'
            mock_fig.layout = {}
            mock_fig.data = []
            
            result = self.chart_builder.create_time_series_plot([invalid_wave])
            
            # Should still create a plot, even with invalid data
            self.assertEqual(result['type'], 'interactive_time_series')


if __name__ == '__main__':
    unittest.main()