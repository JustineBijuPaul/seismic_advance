"""
Main wave visualization coordinator service.

This module provides the WaveVisualizer class that integrates time-series,
frequency, and multi-channel plotting components with automatic plot
generation for analysis results.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
from datetime import datetime

from ..models import WaveSegment, WaveAnalysisResult, DetailedAnalysis
from ..interfaces import WaveVisualizerInterface
from .time_series_plotter import TimeSeriesPlotter
from .frequency_plotter import FrequencyPlotter
from .multi_channel_plotter import MultiChannelPlotter, ChannelData
from .interactive_chart_builder import InteractiveChartBuilder


class WaveVisualizer:
    """
    Main visualization coordinator that integrates all plotting components.
    
    This class serves as the primary interface for creating comprehensive
    wave analysis visualizations by coordinating between different specialized
    plotting services.
    """
    
    def __init__(self, interactive: bool = True):
        """
        Initialize the wave visualizer.
        
        Args:
            interactive: Whether to use interactive plots (Plotly) or static plots
        """
        self.interactive = interactive
        
        # Initialize plotting components
        self.time_series_plotter = TimeSeriesPlotter()
        self.frequency_plotter = FrequencyPlotter()
        self.multi_channel_plotter = MultiChannelPlotter()
        
        if interactive:
            self.interactive_builder = InteractiveChartBuilder()
        else:
            self.interactive_builder = None
        
        # Default visualization settings
        self.default_settings = {
            'show_arrival_markers': True,
            'show_confidence_bands': True,
            'color_by_wave_type': True,
            'include_frequency_analysis': True,
            'include_quality_metrics': True,
            'auto_scale': True
        }
    
    def create_comprehensive_analysis_plot(self, analysis: DetailedAnalysis) -> Dict[str, Any]:
        """
        Create comprehensive visualization for complete wave analysis.
        
        Args:
            analysis: Complete detailed analysis result
            
        Returns:
            Comprehensive plot data with multiple panels
        """
        if not analysis or not analysis.wave_result:
            return self._create_empty_visualization('No analysis data available')
        
        if self.interactive:
            return self.interactive_builder.create_multi_panel_analysis(analysis)
        else:
            return self._create_static_comprehensive_plot(analysis)
    
    def create_wave_separation_plot(self, wave_result: WaveAnalysisResult) -> Dict[str, Any]:
        """
        Create visualization showing separated wave types.
        
        Args:
            wave_result: Wave analysis result with separated waves
            
        Returns:
            Wave separation plot data
        """
        if not wave_result:
            return self._create_empty_visualization('No wave data available')
        
        all_waves = wave_result.p_waves + wave_result.s_waves + wave_result.surface_waves
        
        if self.interactive:
            return self.interactive_builder.create_time_series_plot(all_waves)
        else:
            return self.time_series_plotter.create_multi_wave_comparison(wave_result)
    
    def create_frequency_analysis_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create frequency domain analysis visualization.
        
        Args:
            wave_segments: List of wave segments to analyze
            
        Returns:
            Frequency analysis plot data
        """
        if not wave_segments:
            return self._create_empty_visualization('No wave data for frequency analysis')
        
        if self.interactive:
            return self.interactive_builder.create_frequency_plot(wave_segments)
        else:
            return self.frequency_plotter.create_frequency_comparison_plot(wave_segments)
    
    def create_multi_channel_analysis(self, channels: List[ChannelData], 
                                    wave_segments: Optional[List[WaveSegment]] = None) -> Dict[str, Any]:
        """
        Create multi-channel analysis visualization.
        
        Args:
            channels: List of channel data
            wave_segments: Optional wave segments for arrival markers
            
        Returns:
            Multi-channel analysis plot data
        """
        if not channels:
            return self._create_empty_visualization('No channel data available')
        
        if wave_segments:
            return self.multi_channel_plotter.create_channel_comparison_plot(channels, wave_segments)
        else:
            return self.multi_channel_plotter.create_multi_channel_plot(channels)
    
    def create_correlation_analysis(self, channels: List[ChannelData]) -> Dict[str, Any]:
        """
        Create correlation analysis visualization.
        
        Args:
            channels: List of channel data for correlation analysis
            
        Returns:
            Correlation analysis plot data
        """
        if len(channels) < 2:
            return self._create_empty_visualization('Need at least 2 channels for correlation analysis')
        
        if self.interactive:
            # Calculate correlation data
            correlation_data = self._calculate_correlation_matrix(channels)
            return self.interactive_builder.create_interactive_correlation_matrix(correlation_data)
        else:
            return self.multi_channel_plotter.create_correlation_matrix_plot(channels)
    
    def create_wave_picker_interface(self, wave_result: WaveAnalysisResult) -> Dict[str, Any]:
        """
        Create interactive wave picker interface.
        
        Args:
            wave_result: Wave analysis result for picking
            
        Returns:
            Wave picker interface plot data
        """
        if not wave_result:
            return self._create_empty_visualization('No wave data for picking')
        
        if self.interactive:
            return self.interactive_builder.create_wave_picker_interface(wave_result)
        else:
            # For static plots, just show the time series with markers
            all_waves = wave_result.p_waves + wave_result.s_waves + wave_result.surface_waves
            return self.time_series_plotter.create_time_series_plot(all_waves)
    
    def create_spectrogram_analysis(self, wave_segment: WaveSegment) -> Dict[str, Any]:
        """
        Create spectrogram analysis visualization.
        
        Args:
            wave_segment: Wave segment for spectrogram analysis
            
        Returns:
            Spectrogram plot data
        """
        if not wave_segment:
            return self._create_empty_visualization('No wave data for spectrogram')
        
        if self.interactive:
            return self.interactive_builder.create_interactive_spectrogram(wave_segment)
        else:
            return self.frequency_plotter.create_spectrogram_plot([wave_segment])
    
    def create_quality_metrics_plot(self, analysis: DetailedAnalysis) -> Dict[str, Any]:
        """
        Create quality metrics visualization.
        
        Args:
            analysis: Analysis with quality metrics
            
        Returns:
            Quality metrics plot data
        """
        if not analysis or not analysis.quality_metrics:
            return self._create_empty_visualization('No quality metrics available')
        
        return self._create_quality_metrics_visualization(analysis.quality_metrics)
    
    def create_magnitude_comparison_plot(self, analyses: List[DetailedAnalysis]) -> Dict[str, Any]:
        """
        Create magnitude comparison visualization for multiple analyses.
        
        Args:
            analyses: List of detailed analyses with magnitude estimates
            
        Returns:
            Magnitude comparison plot data
        """
        if not analyses:
            return self._create_empty_visualization('No analyses for magnitude comparison')
        
        return self._create_magnitude_comparison_visualization(analyses)
    
    def create_arrival_time_analysis(self, analysis: DetailedAnalysis) -> Dict[str, Any]:
        """
        Create arrival time analysis visualization.
        
        Args:
            analysis: Analysis with arrival time data
            
        Returns:
            Arrival time analysis plot data
        """
        if not analysis or not analysis.arrival_times:
            return self._create_empty_visualization('No arrival time data available')
        
        # Check if arrival times have any actual data
        arrival_times = analysis.arrival_times
        has_data = (arrival_times.p_wave_arrival is not None or 
                   arrival_times.s_wave_arrival is not None or 
                   arrival_times.surface_wave_arrival is not None)
        
        if not has_data:
            return self._create_empty_visualization('No arrival time data available')
        
        return self._create_arrival_time_visualization(analysis)
    
    def generate_analysis_report(self, analysis: DetailedAnalysis) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with all visualizations.
        
        Args:
            analysis: Complete detailed analysis
            
        Returns:
            Complete analysis report with multiple visualizations
        """
        if not analysis:
            return self._create_empty_visualization('No analysis data for report')
        
        report = {
            'type': 'comprehensive_report',
            'timestamp': datetime.now().isoformat(),
            'analysis_id': id(analysis),
            'visualizations': {}
        }
        
        # Add comprehensive analysis plot
        report['visualizations']['comprehensive'] = self.create_comprehensive_analysis_plot(analysis)
        
        # Add wave separation plot
        report['visualizations']['wave_separation'] = self.create_wave_separation_plot(analysis.wave_result)
        
        # Add frequency analysis
        all_waves = (analysis.wave_result.p_waves + 
                    analysis.wave_result.s_waves + 
                    analysis.wave_result.surface_waves)
        if all_waves:
            report['visualizations']['frequency_analysis'] = self.create_frequency_analysis_plot(all_waves)
        
        # Add arrival time analysis
        if analysis.arrival_times:
            report['visualizations']['arrival_times'] = self.create_arrival_time_analysis(analysis)
        
        # Add quality metrics
        if analysis.quality_metrics:
            report['visualizations']['quality_metrics'] = self.create_quality_metrics_plot(analysis)
        
        # Add summary statistics
        report['summary'] = self._generate_analysis_summary(analysis)
        
        return report
    
    def set_visualization_settings(self, settings: Dict[str, Any]) -> None:
        """
        Update visualization settings.
        
        Args:
            settings: Dictionary of setting name-value pairs
        """
        self.default_settings.update(settings)
        
        # Apply settings to plotting components
        if 'color_scheme' in settings:
            color_scheme = settings['color_scheme']
            self.time_series_plotter.set_color_scheme(color_scheme)
            self.frequency_plotter.set_color_scheme(color_scheme)
            if self.interactive_builder:
                self.interactive_builder.set_color_scheme(color_scheme)
    
    def export_visualization(self, plot_data: Dict[str, Any], 
                           format_type: str = 'html', 
                           filename: Optional[str] = None) -> Union[str, bytes]:
        """
        Export visualization in specified format.
        
        Args:
            plot_data: Plot data from any create_* method
            format_type: Export format ('html', 'png', 'pdf', 'json')
            filename: Optional filename
            
        Returns:
            Exported data as string or bytes
        """
        if format_type == 'html' and self.interactive_builder:
            return self.interactive_builder.export_interactive_html(plot_data, filename)
        elif format_type == 'json':
            import json
            return json.dumps(plot_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _create_static_comprehensive_plot(self, analysis: DetailedAnalysis) -> Dict[str, Any]:
        """Create static comprehensive plot using traditional plotters."""
        # This would create a multi-panel static plot
        # For now, return the time series plot
        all_waves = (analysis.wave_result.p_waves + 
                    analysis.wave_result.s_waves + 
                    analysis.wave_result.surface_waves)
        return self.time_series_plotter.create_multi_wave_comparison(analysis.wave_result)
    
    def _calculate_correlation_matrix(self, channels: List[ChannelData]) -> Dict[str, Any]:
        """Calculate correlation matrix for channels."""
        n_channels = len(channels)
        correlation_matrix = np.zeros((n_channels, n_channels))
        
        # Make all data the same length
        min_length = min(len(ch.data) for ch in channels)
        
        for i in range(n_channels):
            for j in range(n_channels):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    data1 = channels[i].data[:min_length]
                    data2 = channels[j].data[:min_length]
                    
                    # Calculate Pearson correlation
                    correlation_matrix[i, j] = np.corrcoef(data1, data2)[0, 1]
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'channel_labels': [ch.channel_id for ch in channels]
        }
    
    def _create_quality_metrics_visualization(self, quality_metrics) -> Dict[str, Any]:
        """Create quality metrics visualization."""
        # Create a simple bar chart of quality metrics
        metrics = {
            'SNR': quality_metrics.signal_to_noise_ratio,
            'Detection Confidence': quality_metrics.detection_confidence,
            'Analysis Quality': quality_metrics.analysis_quality_score,
            'Data Completeness': quality_metrics.data_completeness
        }
        
        return {
            'type': 'quality_metrics',
            'data': metrics,
            'warnings': quality_metrics.processing_warnings
        }
    
    def _create_magnitude_comparison_visualization(self, analyses: List[DetailedAnalysis]) -> Dict[str, Any]:
        """Create magnitude comparison visualization."""
        magnitude_data = []
        
        for i, analysis in enumerate(analyses):
            for mag_est in analysis.magnitude_estimates:
                magnitude_data.append({
                    'analysis_id': i,
                    'method': mag_est.method,
                    'magnitude': mag_est.magnitude,
                    'confidence': mag_est.confidence,
                    'wave_type': mag_est.wave_type_used
                })
        
        return {
            'type': 'magnitude_comparison',
            'data': magnitude_data
        }
    
    def _create_arrival_time_visualization(self, analysis: DetailedAnalysis) -> Dict[str, Any]:
        """Create arrival time visualization."""
        arrival_data = {}
        
        if analysis.arrival_times.p_wave_arrival:
            arrival_data['P-wave'] = analysis.arrival_times.p_wave_arrival
        
        if analysis.arrival_times.s_wave_arrival:
            arrival_data['S-wave'] = analysis.arrival_times.s_wave_arrival
        
        if analysis.arrival_times.surface_wave_arrival:
            arrival_data['Surface wave'] = analysis.arrival_times.surface_wave_arrival
        
        if analysis.arrival_times.sp_time_difference:
            arrival_data['S-P difference'] = analysis.arrival_times.sp_time_difference
        
        return {
            'type': 'arrival_times',
            'data': arrival_data,
            'epicenter_distance': analysis.epicenter_distance
        }
    
    def _generate_analysis_summary(self, analysis: DetailedAnalysis) -> Dict[str, Any]:
        """Generate summary statistics for analysis."""
        summary = {
            'total_waves_detected': analysis.wave_result.total_waves_detected,
            'wave_types_detected': analysis.wave_result.wave_types_detected,
            'processing_timestamp': analysis.analysis_timestamp.isoformat(),
            'has_complete_analysis': analysis.has_complete_analysis
        }
        
        if analysis.best_magnitude_estimate:
            summary['best_magnitude'] = {
                'value': analysis.best_magnitude_estimate.magnitude,
                'method': analysis.best_magnitude_estimate.method,
                'confidence': analysis.best_magnitude_estimate.confidence
            }
        
        if analysis.epicenter_distance:
            summary['epicenter_distance_km'] = analysis.epicenter_distance
        
        if analysis.quality_metrics:
            summary['overall_quality'] = analysis.quality_metrics.analysis_quality_score
        
        return summary
    
    def _create_empty_visualization(self, message: str) -> Dict[str, Any]:
        """Create empty visualization with message."""
        return {
            'type': 'empty_visualization',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        formats = ['json']
        if self.interactive_builder:
            formats.extend(['html', 'png'])
        return formats
    
    def get_available_plot_types(self) -> List[str]:
        """Get list of available plot types."""
        return [
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