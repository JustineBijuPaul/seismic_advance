"""
Integration tests for WaveVisualizer class.

This module tests the complete visualization workflow including
integration between time-series, frequency, and multi-channel components.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the classes under test
from wave_analysis.services.wave_visualizer import WaveVisualizer
from wave_analysis.services.multi_channel_plotter import ChannelData
from wave_analysis.models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, 
    ArrivalTimes, MagnitudeEstimate, QualityMetrics, FrequencyData
)


class TestWaveVisualizerIntegration(unittest.TestCase):
    """Integration test cases for WaveVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.visualizer = WaveVisualizer(interactive=True)
        self.static_visualizer = WaveVisualizer(interactive=False)
        
        # Create comprehensive test data
        self.sample_p_wave = WaveSegment(
            wave_type='P',
            start_time=5.0,
            end_time=8.0,
            data=np.random.randn(300),
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
            data=np.random.randn(500),
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
            data=np.random.randn(1000),
            sampling_rate=100.0,
            peak_amplitude=1.2,
            dominant_frequency=3.0,
            arrival_time=20.5,
            confidence=0.78
        )
        
        # Create wave analysis result
        self.sample_wave_result = WaveAnalysisResult(
            original_data=np.random.randn(3000),
            sampling_rate=100.0,
            p_waves=[self.sample_p_wave],
            s_waves=[self.sample_s_wave],
            surface_waves=[self.sample_surface_wave]
        )
        
        # Create quality metrics
        self.sample_quality_metrics = QualityMetrics(
            signal_to_noise_ratio=15.5,
            detection_confidence=0.87,
            analysis_quality_score=0.92,
            data_completeness=0.98,
            processing_warnings=['Minor noise detected']
        )
        
        # Create magnitude estimates
        self.sample_magnitude_estimates = [
            MagnitudeEstimate(
                method='ML',
                magnitude=4.2,
                confidence=0.85,
                wave_type_used='P'
            ),
            MagnitudeEstimate(
                method='Mb',
                magnitude=4.1,
                confidence=0.78,
                wave_type_used='P'
            )
        ]
        
        # Create frequency data
        self.sample_frequency_data = {
            'P': FrequencyData(
                frequencies=np.linspace(0, 50, 100),
                power_spectrum=np.random.rand(100),
                dominant_frequency=15.0,
                frequency_range=(5.0, 25.0),
                spectral_centroid=12.5,
                bandwidth=8.0
            )
        }
        
        # Create detailed analysis
        self.sample_analysis = DetailedAnalysis(
            wave_result=self.sample_wave_result,
            arrival_times=ArrivalTimes(
                p_wave_arrival=5.5,
                s_wave_arrival=10.5,
                surface_wave_arrival=20.5,
                sp_time_difference=5.0
            ),
            magnitude_estimates=self.sample_magnitude_estimates,
            epicenter_distance=45.2,
            frequency_analysis=self.sample_frequency_data,
            quality_metrics=self.sample_quality_metrics
        )
        
        # Create channel data for multi-channel tests
        self.sample_channels = [
            ChannelData('CH1', np.random.randn(1000), 100.0, {'lat': 40.0, 'lon': -120.0}, 'N'),
            ChannelData('CH2', np.random.randn(1000), 100.0, {'lat': 40.0, 'lon': -120.0}, 'E'),
            ChannelData('CH3', np.random.randn(1000), 100.0, {'lat': 40.0, 'lon': -120.0}, 'Z')
        ]
    
    def test_initialization(self):
        """Test WaveVisualizer initialization."""
        # Test interactive mode
        self.assertTrue(self.visualizer.interactive)
        self.assertIsNotNone(self.visualizer.interactive_builder)
        self.assertIsNotNone(self.visualizer.time_series_plotter)
        self.assertIsNotNone(self.visualizer.frequency_plotter)
        self.assertIsNotNone(self.visualizer.multi_channel_plotter)
        
        # Test static mode
        self.assertFalse(self.static_visualizer.interactive)
        self.assertIsNone(self.static_visualizer.interactive_builder)
        self.assertIsNotNone(self.static_visualizer.time_series_plotter)
    
    def test_create_comprehensive_analysis_plot(self):
        """Test comprehensive analysis plot creation."""
        result = self.visualizer.create_comprehensive_analysis_plot(self.sample_analysis)
        
        self.assertIn('type', result)
        self.assertTrue(result['type'].startswith('interactive_'))
        
        # Test with empty analysis
        empty_result = self.visualizer.create_comprehensive_analysis_plot(None)
        self.assertEqual(empty_result['type'], 'empty_visualization')
    
    def test_create_wave_separation_plot(self):
        """Test wave separation plot creation."""
        result = self.visualizer.create_wave_separation_plot(self.sample_wave_result)
        
        self.assertIn('type', result)
        self.assertTrue(result['type'].startswith('interactive_'))
        
        # Test static version
        static_result = self.static_visualizer.create_wave_separation_plot(self.sample_wave_result)
        self.assertIn('type', static_result)
    
    def test_create_frequency_analysis_plot(self):
        """Test frequency analysis plot creation."""
        wave_segments = [self.sample_p_wave, self.sample_s_wave, self.sample_surface_wave]
        result = self.visualizer.create_frequency_analysis_plot(wave_segments)
        
        self.assertIn('type', result)
        self.assertTrue(result['type'].startswith('interactive_'))
        
        # Test with empty data
        empty_result = self.visualizer.create_frequency_analysis_plot([])
        self.assertEqual(empty_result['type'], 'empty_visualization')
    
    def test_create_multi_channel_analysis(self):
        """Test multi-channel analysis creation."""
        result = self.visualizer.create_multi_channel_analysis(self.sample_channels)
        
        self.assertIn('type', result)
        
        # Test with wave segments
        wave_segments = [self.sample_p_wave, self.sample_s_wave]
        result_with_waves = self.visualizer.create_multi_channel_analysis(
            self.sample_channels, wave_segments
        )
        self.assertIn('type', result_with_waves)
        
        # Test with empty channels
        empty_result = self.visualizer.create_multi_channel_analysis([])
        self.assertEqual(empty_result['type'], 'empty_visualization')
    
    def test_create_correlation_analysis(self):
        """Test correlation analysis creation."""
        result = self.visualizer.create_correlation_analysis(self.sample_channels)
        
        self.assertIn('type', result)
        
        # Test with insufficient channels
        insufficient_result = self.visualizer.create_correlation_analysis([self.sample_channels[0]])
        self.assertEqual(insufficient_result['type'], 'empty_visualization')
    
    def test_create_wave_picker_interface(self):
        """Test wave picker interface creation."""
        result = self.visualizer.create_wave_picker_interface(self.sample_wave_result)
        
        self.assertIn('type', result)
        
        # Test static version
        static_result = self.static_visualizer.create_wave_picker_interface(self.sample_wave_result)
        self.assertIn('type', static_result)
        
        # Test with empty data
        empty_result = self.visualizer.create_wave_picker_interface(None)
        self.assertEqual(empty_result['type'], 'empty_visualization')
    
    def test_create_spectrogram_analysis(self):
        """Test spectrogram analysis creation."""
        result = self.visualizer.create_spectrogram_analysis(self.sample_p_wave)
        
        self.assertIn('type', result)
        
        # Test with empty data
        empty_result = self.visualizer.create_spectrogram_analysis(None)
        self.assertEqual(empty_result['type'], 'empty_visualization')
    
    def test_create_quality_metrics_plot(self):
        """Test quality metrics plot creation."""
        result = self.visualizer.create_quality_metrics_plot(self.sample_analysis)
        
        self.assertEqual(result['type'], 'quality_metrics')
        self.assertIn('data', result)
        self.assertIn('warnings', result)
        
        # Test with no quality metrics
        analysis_no_quality = DetailedAnalysis(
            wave_result=self.sample_wave_result,
            arrival_times=self.sample_analysis.arrival_times
        )
        empty_result = self.visualizer.create_quality_metrics_plot(analysis_no_quality)
        self.assertEqual(empty_result['type'], 'empty_visualization')
    
    def test_create_magnitude_comparison_plot(self):
        """Test magnitude comparison plot creation."""
        analyses = [self.sample_analysis, self.sample_analysis]  # Duplicate for comparison
        result = self.visualizer.create_magnitude_comparison_plot(analyses)
        
        self.assertEqual(result['type'], 'magnitude_comparison')
        self.assertIn('data', result)
        self.assertIsInstance(result['data'], list)
        
        # Test with empty analyses
        empty_result = self.visualizer.create_magnitude_comparison_plot([])
        self.assertEqual(empty_result['type'], 'empty_visualization')
    
    def test_create_arrival_time_analysis(self):
        """Test arrival time analysis creation."""
        result = self.visualizer.create_arrival_time_analysis(self.sample_analysis)
        
        self.assertEqual(result['type'], 'arrival_times')
        self.assertIn('data', result)
        self.assertIn('epicenter_distance', result)
        
        # Test with no arrival times
        analysis_no_arrivals = DetailedAnalysis(
            wave_result=self.sample_wave_result,
            arrival_times=ArrivalTimes()  # Empty arrival times
        )
        empty_result = self.visualizer.create_arrival_time_analysis(analysis_no_arrivals)
        self.assertEqual(empty_result['type'], 'empty_visualization')
    
    def test_generate_analysis_report(self):
        """Test comprehensive analysis report generation."""
        report = self.visualizer.generate_analysis_report(self.sample_analysis)
        
        self.assertEqual(report['type'], 'comprehensive_report')
        self.assertIn('timestamp', report)
        self.assertIn('analysis_id', report)
        self.assertIn('visualizations', report)
        self.assertIn('summary', report)
        
        # Check that all expected visualizations are present
        visualizations = report['visualizations']
        self.assertIn('comprehensive', visualizations)
        self.assertIn('wave_separation', visualizations)
        self.assertIn('frequency_analysis', visualizations)
        self.assertIn('arrival_times', visualizations)
        self.assertIn('quality_metrics', visualizations)
        
        # Check summary content
        summary = report['summary']
        self.assertIn('total_waves_detected', summary)
        self.assertIn('wave_types_detected', summary)
        self.assertIn('best_magnitude', summary)
        self.assertIn('epicenter_distance_km', summary)
        self.assertIn('overall_quality', summary)
    
    def test_set_visualization_settings(self):
        """Test visualization settings update."""
        custom_settings = {
            'show_arrival_markers': False,
            'color_by_wave_type': False,
            'color_scheme': {
                'P': '#FF0000',
                'S': '#00FF00'
            }
        }
        
        self.visualizer.set_visualization_settings(custom_settings)
        
        # Check that settings were updated
        self.assertFalse(self.visualizer.default_settings['show_arrival_markers'])
        self.assertFalse(self.visualizer.default_settings['color_by_wave_type'])
        self.assertIn('color_scheme', self.visualizer.default_settings)
    
    def test_export_visualization(self):
        """Test visualization export functionality."""
        plot_data = self.visualizer.create_wave_separation_plot(self.sample_wave_result)
        
        # Test JSON export
        json_export = self.visualizer.export_visualization(plot_data, 'json')
        self.assertIsInstance(json_export, str)
        
        # Test HTML export (mocked)
        with patch.object(self.visualizer.interactive_builder, 'export_interactive_html') as mock_export:
            mock_export.return_value = '<html>test</html>'
            html_export = self.visualizer.export_visualization(plot_data, 'html')
            self.assertEqual(html_export, '<html>test</html>')
            mock_export.assert_called_once()
        
        # Test unsupported format
        with self.assertRaises(ValueError):
            self.visualizer.export_visualization(plot_data, 'unsupported')
    
    def test_integration_workflow(self):
        """Test complete integration workflow."""
        # Step 1: Create comprehensive analysis
        comprehensive_plot = self.visualizer.create_comprehensive_analysis_plot(self.sample_analysis)
        self.assertIn('type', comprehensive_plot)
        
        # Step 2: Create individual component plots
        wave_separation = self.visualizer.create_wave_separation_plot(self.sample_wave_result)
        frequency_analysis = self.visualizer.create_frequency_analysis_plot([self.sample_p_wave])
        spectrogram = self.visualizer.create_spectrogram_analysis(self.sample_p_wave)
        
        # Step 3: Create multi-channel analysis
        multi_channel = self.visualizer.create_multi_channel_analysis(self.sample_channels)
        correlation = self.visualizer.create_correlation_analysis(self.sample_channels)
        
        # Step 4: Generate complete report
        report = self.visualizer.generate_analysis_report(self.sample_analysis)
        
        # Verify all components were created successfully
        self.assertIn('type', wave_separation)
        self.assertIn('type', frequency_analysis)
        self.assertIn('type', spectrogram)
        self.assertIn('type', multi_channel)
        self.assertIn('type', correlation)
        self.assertEqual(report['type'], 'comprehensive_report')
        
        # Verify report contains all visualizations
        self.assertGreaterEqual(len(report['visualizations']), 5)
    
    def test_error_handling_and_graceful_degradation(self):
        """Test error handling and graceful degradation."""
        # Test with corrupted data
        corrupted_wave = WaveSegment(
            wave_type='P',
            start_time=5.0,
            end_time=8.0,
            data=np.array([]),  # Empty data
            sampling_rate=100.0,
            peak_amplitude=0.5,
            dominant_frequency=15.0,
            arrival_time=5.5,
            confidence=0.85
        )
        
        # Should handle gracefully
        result = self.visualizer.create_frequency_analysis_plot([corrupted_wave])
        self.assertIn('type', result)
        
        # Test with None inputs
        none_result = self.visualizer.create_comprehensive_analysis_plot(None)
        self.assertEqual(none_result['type'], 'empty_visualization')
        
        # Test with incomplete analysis
        incomplete_analysis = DetailedAnalysis(
            wave_result=WaveAnalysisResult(
                original_data=np.random.randn(100),
                sampling_rate=100.0
            ),
            arrival_times=ArrivalTimes()
        )
        
        incomplete_result = self.visualizer.create_comprehensive_analysis_plot(incomplete_analysis)
        self.assertIn('type', incomplete_result)
    
    def test_get_supported_formats(self):
        """Test getting supported export formats."""
        formats = self.visualizer.get_supported_formats()
        self.assertIn('json', formats)
        self.assertIn('html', formats)  # Interactive mode
        
        static_formats = self.static_visualizer.get_supported_formats()
        self.assertIn('json', static_formats)
        self.assertNotIn('html', static_formats)  # Static mode
    
    def test_get_available_plot_types(self):
        """Test getting available plot types."""
        plot_types = self.visualizer.get_available_plot_types()
        
        expected_types = [
            'comprehensive_analysis',
            'wave_separation',
            'frequency_analysis',
            'multi_channel_analysis',
            'correlation_analysis',
            'wave_picker',
            'spectrogram',
            'quality_metrics',
            'magnitude_comparison',
            'arrival_time_analysis'
        ]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, plot_types)
    
    def test_static_vs_interactive_mode_differences(self):
        """Test differences between static and interactive modes."""
        # Create same plot in both modes
        interactive_result = self.visualizer.create_wave_separation_plot(self.sample_wave_result)
        static_result = self.static_visualizer.create_wave_separation_plot(self.sample_wave_result)
        
        # Interactive should have different type prefix
        self.assertTrue(interactive_result['type'].startswith('interactive_'))
        self.assertFalse(static_result['type'].startswith('interactive_'))
        
        # Interactive should have plotly_json
        if 'plotly_json' in interactive_result:
            self.assertIsInstance(interactive_result['plotly_json'], str)
        
        # Static should use traditional plotting format
        self.assertIn('type', static_result)


if __name__ == '__main__':
    unittest.main()