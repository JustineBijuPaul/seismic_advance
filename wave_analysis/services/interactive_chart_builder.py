"""
Interactive chart builder service for wave visualization.

This module provides comprehensive interactive visualization capabilities
using Plotly for web interactivity, including hover tooltips with wave
characteristics and click-to-zoom functionality.
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy import signal
from scipy.fft import fft, fftfreq

from ..models import WaveSegment, WaveAnalysisResult, DetailedAnalysis
from ..interfaces import WaveVisualizerInterface


class InteractiveChartBuilder(WaveVisualizerInterface):
    """
    Creates interactive visualizations using Plotly for web interactivity.
    
    Supports hover tooltips with wave characteristics, click-to-zoom functionality,
    and detailed wave inspection capabilities.
    """
    
    def __init__(self):
        """Initialize the interactive chart builder."""
        self.default_colors = {
            'P': '#FF6B6B',      # Red for P-waves
            'S': '#4ECDC4',      # Teal for S-waves  
            'Love': '#45B7D1',   # Blue for Love waves
            'Rayleigh': '#96CEB4' # Green for Rayleigh waves
        }
        self.chart_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'wave_analysis',
                'height': 600,
                'width': 1000,
                'scale': 2
            }
        }
        self.layout_template = 'plotly_white'
    
    def create_time_series_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create interactive time series visualization for wave segments.
        
        Args:
            wave_segments: List of wave segments to visualize
            
        Returns:
            Interactive plot data dictionary for rendering with Plotly
        """
        if not wave_segments:
            return self._create_empty_plot('No wave data available')
        
        # Create figure with secondary y-axis for characteristic functions
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]],
            subplot_titles=("Seismic Wave Time Series Analysis",)
        )
        
        # Group waves by type
        waves_by_type = self._group_waves_by_type(wave_segments)
        
        # Add traces for each wave type
        for wave_type, waves in waves_by_type.items():
            self._add_wave_traces(fig, wave_type, waves)
        
        # Add arrival time markers
        self._add_arrival_markers(fig, wave_segments)
        
        # Configure layout
        self._configure_time_series_layout(fig, wave_segments)
        
        return self._fig_to_dict(fig, 'time_series')
    
    def create_frequency_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create interactive frequency domain visualization for wave segments.
        
        Args:
            wave_segments: List of wave segments to analyze
            
        Returns:
            Interactive frequency plot data dictionary
        """
        if not wave_segments:
            return self._create_empty_plot('No wave data available for frequency analysis')
        
        fig = go.Figure()
        
        # Group waves by type
        waves_by_type = self._group_waves_by_type(wave_segments)
        
        # Add frequency spectrum for each wave type
        for wave_type, waves in waves_by_type.items():
            self._add_frequency_traces(fig, wave_type, waves)
        
        # Add dominant frequency markers
        self._add_frequency_markers(fig, wave_segments)
        
        # Configure layout
        self._configure_frequency_layout(fig)
        
        return self._fig_to_dict(fig, 'frequency_spectrum')
    
    def create_interactive_spectrogram(self, wave_segment: WaveSegment) -> Dict[str, Any]:
        """
        Create interactive spectrogram visualization for time-frequency analysis.
        
        Args:
            wave_segment: Wave segment to analyze
            
        Returns:
            Interactive spectrogram plot data
        """
        if not wave_segment:
            return self._create_empty_plot('No wave data available for spectrogram')
        
        # Calculate spectrogram
        frequencies, times, Sxx = self._calculate_spectrogram(wave_segment)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=10 * np.log10(Sxx + 1e-10),  # Convert to dB
            x=times + wave_segment.start_time,  # Adjust time axis
            y=frequencies,
            colorscale='Viridis',
            colorbar=dict(title="Power (dB)"),
            hovertemplate=(
                "Time: %{x:.2f} s<br>"
                "Frequency: %{y:.2f} Hz<br>"
                "Power: %{z:.1f} dB<br>"
                "<extra></extra>"
            )
        ))
        
        # Configure layout
        fig.update_layout(
            title=f'Interactive Spectrogram - {wave_segment.wave_type} Wave',
            xaxis_title='Time (seconds)',
            yaxis_title='Frequency (Hz)',
            template=self.layout_template,
            hovermode='closest'
        )
        
        return self._fig_to_dict(fig, 'spectrogram')
    
    def create_multi_panel_analysis(self, analysis: DetailedAnalysis) -> Dict[str, Any]:
        """
        Create multi-panel interactive visualization with time series, frequency, and analysis.
        
        Args:
            analysis: Complete detailed analysis result
            
        Returns:
            Multi-panel interactive plot data
        """
        if not analysis or not analysis.wave_result:
            return self._create_empty_plot('No analysis data available')
        
        # Create subplots with 3 rows
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Time Series Analysis',
                'Frequency Spectrum',
                'Wave Characteristics'
            ),
            specs=[
                [{"secondary_y": True}],  # Time series with secondary y-axis
                [{"secondary_y": False}], # Frequency plot
                [{"secondary_y": False}]  # Characteristics plot
            ],
            vertical_spacing=0.08
        )
        
        # Add time series data
        self._add_multi_panel_time_series(fig, analysis.wave_result, row=1)
        
        # Add frequency analysis
        self._add_multi_panel_frequency(fig, analysis, row=2)
        
        # Add wave characteristics
        self._add_multi_panel_characteristics(fig, analysis, row=3)
        
        # Configure layout
        self._configure_multi_panel_layout(fig, analysis)
        
        return self._fig_to_dict(fig, 'multi_panel_analysis')
    
    def create_interactive_correlation_matrix(self, correlation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create interactive correlation matrix heatmap.
        
        Args:
            correlation_data: Correlation matrix data with labels
            
        Returns:
            Interactive correlation matrix plot
        """
        if 'correlation_matrix' not in correlation_data:
            return self._create_empty_plot('No correlation matrix data available')
        
        matrix = np.array(correlation_data['correlation_matrix'])
        labels = correlation_data.get('channel_labels', [f'Ch{i}' for i in range(len(matrix))])
        
        # Create heatmap with custom hover template
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Correlation Coefficient"),
            hovertemplate=(
                "Channel 1: %{y}<br>"
                "Channel 2: %{x}<br>"
                "Correlation: %{z:.3f}<br>"
                "<extra></extra>"
            )
        ))
        
        # Add correlation values as text annotations
        annotations = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                annotations.append(
                    dict(
                        x=labels[j],
                        y=labels[i],
                        text=f"{matrix[i][j]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(matrix[i][j]) > 0.5 else "black")
                    )
                )
        
        fig.update_layout(
            title='Interactive Channel Correlation Matrix',
            annotations=annotations,
            template=self.layout_template,
            width=600,
            height=600
        )
        
        return self._fig_to_dict(fig, 'correlation_matrix')
    
    def create_wave_picker_interface(self, wave_result: WaveAnalysisResult) -> Dict[str, Any]:
        """
        Create interactive wave picker interface for manual wave identification.
        
        Args:
            wave_result: Wave analysis result with detected waves
            
        Returns:
            Interactive wave picker plot data
        """
        if not wave_result:
            return self._create_empty_plot('No wave data available for picking')
        
        fig = go.Figure()
        
        # Add original waveform
        duration = len(wave_result.original_data) / wave_result.sampling_rate
        time_points = np.linspace(0, duration, len(wave_result.original_data))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=wave_result.original_data,
            mode='lines',
            name='Original Waveform',
            line=dict(color='#333333', width=1),
            hovertemplate=(
                "Time: %{x:.3f} s<br>"
                "Amplitude: %{y:.2e}<br>"
                "<extra></extra>"
            )
        ))
        
        # Add detected wave markers
        all_waves = wave_result.p_waves + wave_result.s_waves + wave_result.surface_waves
        for wave in all_waves:
            fig.add_trace(go.Scatter(
                x=[wave.arrival_time],
                y=[wave.peak_amplitude],
                mode='markers',
                name=f'{wave.wave_type} Wave',
                marker=dict(
                    color=self.default_colors.get(wave.wave_type, '#333333'),
                    size=10,
                    symbol='diamond'
                ),
                hovertemplate=(
                    f"Wave Type: {wave.wave_type}<br>"
                    f"Arrival Time: {wave.arrival_time:.3f} s<br>"
                    f"Peak Amplitude: {wave.peak_amplitude:.2e}<br>"
                    f"Dominant Frequency: {wave.dominant_frequency:.2f} Hz<br>"
                    f"Confidence: {wave.confidence:.2f}<br>"
                    "<extra></extra>"
                ),
                showlegend=False
            ))
        
        # Configure layout for wave picking
        fig.update_layout(
            title='Interactive Wave Picker',
            xaxis_title='Time (seconds)',
            yaxis_title='Amplitude',
            template=self.layout_template,
            hovermode='closest',
            dragmode='zoom',
            selectdirection='h'  # 'h' for horizontal selection
        )
        
        return self._fig_to_dict(fig, 'wave_picker')
    
    def _add_multi_panel_time_series(self, fig: make_subplots, wave_result: WaveAnalysisResult, row: int) -> None:
        """Add time series data to multi-panel plot."""
        # Add original waveform
        duration = len(wave_result.original_data) / wave_result.sampling_rate
        time_points = np.linspace(0, duration, len(wave_result.original_data))
        
        fig.add_trace(go.Scatter(
            x=time_points,
            y=wave_result.original_data,
            mode='lines',
            name='Original',
            line=dict(color='#CCCCCC', width=1),
            hovertemplate="Time: %{x:.3f} s<br>Amplitude: %{y:.2e}<br><extra></extra>"
        ), row=row, col=1)
        
        # Add detected waves
        all_waves = wave_result.p_waves + wave_result.s_waves + wave_result.surface_waves
        for wave in all_waves:
            time_points = np.linspace(wave.start_time, wave.end_time, len(wave.data))
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=wave.data,
                mode='lines',
                name=f'{wave.wave_type} Wave',
                line=dict(
                    color=self.default_colors.get(wave.wave_type, '#333333'),
                    width=2
                ),
                hovertemplate=(
                    f"Wave: {wave.wave_type}<br>"
                    "Time: %{x:.3f} s<br>"
                    "Amplitude: %{y:.2e}<br>"
                    f"Confidence: {wave.confidence:.2f}<br>"
                    "<extra></extra>"
                )
            ), row=row, col=1)
    
    def _add_multi_panel_frequency(self, fig: make_subplots, analysis: DetailedAnalysis, row: int) -> None:
        """Add frequency analysis to multi-panel plot."""
        for wave_type, freq_data in analysis.frequency_analysis.items():
            fig.add_trace(go.Scatter(
                x=freq_data.frequencies,
                y=freq_data.power_spectrum,
                mode='lines',
                name=f'{wave_type} Spectrum',
                line=dict(
                    color=self.default_colors.get(wave_type, '#333333'),
                    width=2
                ),
                hovertemplate=(
                    f"Wave: {wave_type}<br>"
                    "Frequency: %{x:.2f} Hz<br>"
                    "Power: %{y:.2e}<br>"
                    "<extra></extra>"
                )
            ), row=row, col=1)
    
    def _add_multi_panel_characteristics(self, fig: make_subplots, analysis: DetailedAnalysis, row: int) -> None:
        """Add wave characteristics to multi-panel plot."""
        # Create bar chart of wave characteristics
        wave_types = []
        arrival_times = []
        
        if analysis.arrival_times.p_wave_arrival:
            wave_types.append('P-wave')
            arrival_times.append(analysis.arrival_times.p_wave_arrival)
        
        if analysis.arrival_times.s_wave_arrival:
            wave_types.append('S-wave')
            arrival_times.append(analysis.arrival_times.s_wave_arrival)
        
        if analysis.arrival_times.surface_wave_arrival:
            wave_types.append('Surface')
            arrival_times.append(analysis.arrival_times.surface_wave_arrival)
        
        if wave_types:
            fig.add_trace(go.Bar(
                x=wave_types,
                y=arrival_times,
                name='Arrival Times',
                marker_color=[self.default_colors.get(wt.split('-')[0], '#333333') for wt in wave_types],
                hovertemplate=(
                    "Wave Type: %{x}<br>"
                    "Arrival Time: %{y:.3f} s<br>"
                    "<extra></extra>"
                )
            ), row=row, col=1)
    
    def _configure_multi_panel_layout(self, fig: make_subplots, analysis: DetailedAnalysis) -> None:
        """Configure layout for multi-panel plots."""
        fig.update_layout(
            title='Comprehensive Wave Analysis Dashboard',
            template=self.layout_template,
            height=900,
            hovermode='x unified'
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_xaxes(title_text="Wave Type", row=3, col=1)
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)
        fig.update_yaxes(title_text="Arrival Time (s)", row=3, col=1)
    
    def _group_waves_by_type(self, wave_segments: List[WaveSegment]) -> Dict[str, List[WaveSegment]]:
        """Group wave segments by their type."""
        waves_by_type = {}
        for wave in wave_segments:
            if wave.wave_type not in waves_by_type:
                waves_by_type[wave.wave_type] = []
            waves_by_type[wave.wave_type].append(wave)
        return waves_by_type
    
    def _add_wave_traces(self, fig: go.Figure, wave_type: str, waves: List[WaveSegment]) -> None:
        """Add wave traces to the figure."""
        # Combine all wave segments of the same type
        combined_data = []
        combined_times = []
        wave_info = []
        
        for wave in waves:
            duration = wave.end_time - wave.start_time
            time_points = np.linspace(wave.start_time, wave.end_time, len(wave.data))
            
            combined_times.extend(time_points.tolist())
            combined_data.extend(wave.data.tolist())
            
            # Store wave info for hover tooltips
            for i, (t, d) in enumerate(zip(time_points, wave.data)):
                wave_info.append({
                    'time': t,
                    'amplitude': d,
                    'wave_type': wave.wave_type,
                    'arrival_time': wave.arrival_time,
                    'dominant_freq': wave.dominant_frequency,
                    'confidence': wave.confidence,
                    'peak_amplitude': wave.peak_amplitude
                })
        
        if combined_data:
            # Create custom hover template with wave characteristics
            hover_template = (
                "Time: %{x:.3f} s<br>"
                "Amplitude: %{y:.2e}<br>"
                f"Wave Type: {wave_type}<br>"
                "Arrival: %{customdata[0]:.3f} s<br>"
                "Dom. Freq: %{customdata[1]:.2f} Hz<br>"
                "Confidence: %{customdata[2]:.2f}<br>"
                "<extra></extra>"
            )
            
            # Prepare custom data for hover
            customdata = [[info['arrival_time'], info['dominant_freq'], info['confidence']] 
                         for info in wave_info]
            
            fig.add_trace(go.Scatter(
                x=combined_times,
                y=combined_data,
                mode='lines',
                name=f'{wave_type}-waves',
                line=dict(
                    color=self.default_colors.get(wave_type, '#333333'),
                    width=2
                ),
                customdata=customdata,
                hovertemplate=hover_template
            ))
    
    def _add_arrival_markers(self, fig: go.Figure, wave_segments: List[WaveSegment]) -> None:
        """Add arrival time markers to the figure."""
        for wave in wave_segments:
            fig.add_vline(
                x=wave.arrival_time,
                line_dash="dash",
                line_color=self.default_colors.get(wave.wave_type, '#333333'),
                annotation_text=f"{wave.wave_type} arrival",
                annotation_position="top"
            )
    
    def _add_frequency_traces(self, fig: go.Figure, wave_type: str, waves: List[WaveSegment]) -> None:
        """Add frequency spectrum traces to the figure."""
        # Use the wave with highest confidence
        primary_wave = max(waves, key=lambda w: w.confidence)
        
        # Calculate frequency spectrum
        frequencies, psd = self._calculate_frequency_spectrum(primary_wave)
        
        # Create hover template with wave characteristics
        hover_template = (
            "Frequency: %{x:.2f} Hz<br>"
            "Power: %{y:.2e}<br>"
            f"Wave Type: {wave_type}<br>"
            f"Arrival Time: {primary_wave.arrival_time:.3f} s<br>"
            f"Confidence: {primary_wave.confidence:.2f}<br>"
            "<extra></extra>"
        )
        
        fig.add_trace(go.Scatter(
            x=frequencies,
            y=psd,
            mode='lines',
            name=f'{wave_type} Spectrum',
            line=dict(
                color=self.default_colors.get(wave_type, '#333333'),
                width=2
            ),
            fill='tonexty' if wave_type != 'P' else 'tozeroy',
            fillcolor=self._hex_to_rgba(self.default_colors.get(wave_type, '#333333'), 0.2),
            hovertemplate=hover_template
        ))
    
    def _add_frequency_markers(self, fig: go.Figure, wave_segments: List[WaveSegment]) -> None:
        """Add dominant frequency markers to the figure."""
        for wave in wave_segments:
            fig.add_vline(
                x=wave.dominant_frequency,
                line_dash="dash",
                line_color=self.default_colors.get(wave.wave_type, '#333333'),
                annotation_text=f"{wave.wave_type}: {wave.dominant_frequency:.1f} Hz",
                annotation_position="top"
            )
    
    def _calculate_spectrogram(self, wave: WaveSegment) -> tuple:
        """Calculate spectrogram for a wave segment."""
        # Adjust parameters based on data length
        data_length = len(wave.data)
        nperseg = min(256, data_length // 4)
        noverlap = nperseg // 2
        nfft = max(nperseg, 256)
        
        frequencies, times, Sxx = signal.spectrogram(
            wave.data,
            fs=wave.sampling_rate,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            scaling='density'
        )
        
        return frequencies, times, Sxx
    
    def _calculate_frequency_spectrum(self, wave: WaveSegment) -> tuple:
        """Calculate frequency spectrum for a wave segment."""
        # Handle empty data
        if len(wave.data) == 0:
            return np.array([0, 1]), np.array([0, 0])
        
        # Ensure minimum data length
        if len(wave.data) < 4:
            # Pad with zeros if too short
            padded_data = np.pad(wave.data, (0, 4 - len(wave.data)), 'constant')
        else:
            padded_data = wave.data
        
        # Apply window function
        window = signal.get_window('hann', len(padded_data))
        windowed_data = padded_data * window
        
        # Detrend only if data length > 1
        if len(windowed_data) > 1:
            windowed_data = signal.detrend(windowed_data, type='linear')
        
        # Calculate FFT
        fft_data = fft(windowed_data)
        frequencies = fftfreq(len(windowed_data), 1/wave.sampling_rate)
        
        # Take positive frequencies only
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        fft_data = fft_data[positive_freq_idx]
        
        # Calculate power spectral density
        psd = np.abs(fft_data)**2 / (wave.sampling_rate * len(windowed_data))
        
        return frequencies, psd
    
    def _configure_time_series_layout(self, fig: go.Figure, wave_segments: List[WaveSegment]) -> None:
        """Configure layout for time series plots."""
        fig.update_layout(
            title='Interactive Seismic Wave Analysis',
            xaxis_title='Time (seconds)',
            yaxis_title='Amplitude',
            template=self.layout_template,
            hovermode='x unified',
            dragmode='zoom'
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1s", step="second", stepmode="backward"),
                        dict(count=5, label="5s", step="second", stepmode="backward"),
                        dict(count=10, label="10s", step="second", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="linear"
            )
        )
    
    def _configure_frequency_layout(self, fig: go.Figure) -> None:
        """Configure layout for frequency plots."""
        fig.update_layout(
            title='Interactive Frequency Spectrum Analysis',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Power Spectral Density',
            xaxis_type='log',
            template=self.layout_template,
            hovermode='x unified'
        )
    
    def _fig_to_dict(self, fig: go.Figure, plot_type: str) -> Dict[str, Any]:
        """Convert Plotly figure to dictionary format."""
        return {
            'type': f'interactive_{plot_type}',
            'plotly_json': fig.to_json(),
            'config': self.chart_config,
            'layout': fig.layout,
            'data': fig.data
        }
    
    def _create_empty_plot(self, message: str = 'No data available') -> Dict[str, Any]:
        """Create empty plot with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            template=self.layout_template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return self._fig_to_dict(fig, 'empty')
    
    def _hex_to_rgba(self, hex_color: str, alpha: float = 1.0) -> str:
        """Convert hex color to rgba format."""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return f'rgba(51, 51, 51, {alpha})'  # Default gray
        
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return f'rgba({r}, {g}, {b}, {alpha})'
        except ValueError:
            return f'rgba(51, 51, 51, {alpha})'  # Default gray
    
    def set_color_scheme(self, colors: Dict[str, str]) -> None:
        """Set custom color scheme for wave types."""
        self.default_colors.update(colors)
    
    def set_chart_config(self, config: Dict[str, Any]) -> None:
        """Set custom chart configuration."""
        self.chart_config.update(config)
    
    def set_layout_template(self, template: str) -> None:
        """Set Plotly layout template."""
        self.layout_template = template
    
    def export_interactive_html(self, fig_dict: Dict[str, Any], filename: str = 'wave_analysis.html') -> str:
        """
        Export interactive plot as standalone HTML file.
        
        Args:
            fig_dict: Plot dictionary from any create_* method
            filename: Output filename
            
        Returns:
            HTML content as string
        """
        if 'plotly_json' in fig_dict:
            fig = pio.from_json(fig_dict['plotly_json'])
            html_content = pio.to_html(fig, config=self.chart_config, include_plotlyjs=True)
            return html_content
        else:
            raise ValueError("Invalid figure dictionary format")