"""
Multi-channel plotting service for wave visualization.

This module provides comprehensive multi-channel visualization capabilities
including cross-correlation and coherence analysis for seismic data.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import json

from ..models import WaveSegment
from ..interfaces import WaveVisualizerInterface


class ChannelData:
    """
    Represents data from a single seismic channel.
    """
    
    def __init__(self, channel_id: str, data: np.ndarray, sampling_rate: float, 
                 location: Optional[Dict[str, float]] = None, orientation: Optional[str] = None):
        """
        Initialize channel data.
        
        Args:
            channel_id: Unique identifier for the channel
            data: Time series data
            sampling_rate: Sampling rate in Hz
            location: Optional location coordinates (lat, lon, elevation)
            orientation: Optional orientation (N, E, Z for North, East, Vertical)
        """
        self.channel_id = channel_id
        self.data = data
        self.sampling_rate = sampling_rate
        self.location = location or {}
        self.orientation = orientation
        self.duration = len(data) / sampling_rate
    
    def __repr__(self):
        return f"ChannelData(id={self.channel_id}, samples={len(self.data)}, rate={self.sampling_rate}Hz)"


class MultiChannelPlotter(WaveVisualizerInterface):
    """
    Creates multi-channel visualizations for seismic wave data.
    
    Supports cross-correlation visualization between channels and
    coherence analysis plots for channel relationships.
    """
    
    def __init__(self):
        """Initialize the multi-channel plotter."""
        self.default_colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#96CEB4',  # Green
            '#FFEAA7',  # Yellow
            '#DDA0DD',  # Plum
            '#98D8C8',  # Mint
            '#F7DC6F'   # Light Yellow
        ]
        self.correlation_config = {
            'max_lag_seconds': 10.0,
            'normalize': True,
            'detrend': True
        }
        self.coherence_config = {
            'nperseg': 256,
            'noverlap': 128,
            'window': 'hann'
        }
    
    def create_time_series_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create time series visualization (delegated to TimeSeriesPlotter).
        
        This method is part of the interface but delegates to TimeSeriesPlotter.
        """
        return {'type': 'time_series', 'message': 'Use TimeSeriesPlotter for time series analysis'}
    
    def create_frequency_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create frequency domain visualization (delegated to FrequencyPlotter).
        
        This method is part of the interface but delegates to FrequencyPlotter.
        """
        return {'type': 'frequency', 'message': 'Use FrequencyPlotter for frequency analysis'}
    
    def create_multi_channel_plot(self, channels: List[ChannelData]) -> Dict[str, Any]:
        """
        Create multi-channel time series plot.
        
        Args:
            channels: List of channel data to visualize
            
        Returns:
            Multi-channel plot data dictionary for rendering
        """
        if not channels:
            return self._create_empty_plot('No channel data available')
        
        plot_data = {
            'type': 'multi_channel_time_series',
            'title': 'Multi-Channel Seismic Data',
            'datasets': [],
            'layout': self._create_multi_channel_layout_config(len(channels)),
            'config': self._create_plot_config()
        }
        
        # Create dataset for each channel
        for i, channel in enumerate(channels):
            dataset = self._create_channel_dataset(channel, i)
            plot_data['datasets'].append(dataset)
        
        return plot_data
    
    def create_cross_correlation_plot(self, channels: List[ChannelData]) -> Dict[str, Any]:
        """
        Create cross-correlation visualization between channels.
        
        Args:
            channels: List of channel data for correlation analysis
            
        Returns:
            Cross-correlation plot data dictionary
        """
        if len(channels) < 2:
            return self._create_empty_plot('Need at least 2 channels for cross-correlation')
        
        plot_data = {
            'type': 'cross_correlation',
            'title': 'Cross-Correlation Analysis',
            'datasets': [],
            'layout': self._create_correlation_layout_config(),
            'config': self._create_plot_config()
        }
        
        # Calculate cross-correlations between all channel pairs
        correlation_pairs = []
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                correlation_data = self._calculate_cross_correlation(channels[i], channels[j])
                correlation_pairs.append(correlation_data)
        
        # Create datasets for correlation plots
        for i, corr_data in enumerate(correlation_pairs):
            dataset = self._create_correlation_dataset(corr_data, i)
            plot_data['datasets'].append(dataset)
        
        # Add correlation summary
        plot_data['correlation_summary'] = self._create_correlation_summary(correlation_pairs)
        
        return plot_data
    
    def create_coherence_plot(self, channels: List[ChannelData]) -> Dict[str, Any]:
        """
        Create coherence analysis plots for channel relationships.
        
        Args:
            channels: List of channel data for coherence analysis
            
        Returns:
            Coherence plot data dictionary
        """
        if len(channels) < 2:
            return self._create_empty_plot('Need at least 2 channels for coherence analysis')
        
        plot_data = {
            'type': 'coherence_analysis',
            'title': 'Channel Coherence Analysis',
            'datasets': [],
            'layout': self._create_coherence_layout_config(),
            'config': self._create_plot_config()
        }
        
        # Calculate coherence between all channel pairs
        coherence_pairs = []
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                coherence_data = self._calculate_coherence(channels[i], channels[j])
                coherence_pairs.append(coherence_data)
        
        # Create datasets for coherence plots
        for i, coh_data in enumerate(coherence_pairs):
            dataset = self._create_coherence_dataset(coh_data, i)
            plot_data['datasets'].append(dataset)
        
        return plot_data
    
    def create_correlation_matrix_plot(self, channels: List[ChannelData]) -> Dict[str, Any]:
        """
        Create correlation matrix heatmap for all channel pairs.
        
        Args:
            channels: List of channel data
            
        Returns:
            Correlation matrix plot data
        """
        if len(channels) < 2:
            return self._create_empty_plot('Need at least 2 channels for correlation matrix')
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(channels)
        channel_labels = [ch.channel_id for ch in channels]
        
        plot_data = {
            'type': 'correlation_matrix',
            'title': 'Channel Correlation Matrix',
            'data': {
                'z': correlation_matrix.tolist(),
                'x': channel_labels,
                'y': channel_labels,
                'type': 'heatmap',
                'colorscale': 'RdBu',
                'zmid': 0,
                'showscale': True,
                'colorbar': {
                    'title': 'Correlation Coefficient'
                }
            },
            'layout': self._create_matrix_layout_config(channel_labels),
            'config': self._create_plot_config()
        }
        
        return plot_data
    
    def create_channel_comparison_plot(self, channels: List[ChannelData], 
                                     wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create comparison plot showing wave arrivals across multiple channels.
        
        Args:
            channels: List of channel data
            wave_segments: List of wave segments with arrival times
            
        Returns:
            Channel comparison plot data
        """
        if not channels or not wave_segments:
            return self._create_empty_plot('Need both channel data and wave segments')
        
        plot_data = {
            'type': 'channel_wave_comparison',
            'title': 'Wave Arrivals Across Channels',
            'datasets': [],
            'layout': self._create_comparison_layout_config(len(channels)),
            'config': self._create_plot_config()
        }
        
        # Create dataset for each channel
        for i, channel in enumerate(channels):
            dataset = self._create_channel_dataset(channel, i)
            plot_data['datasets'].append(dataset)
        
        # Add wave arrival markers
        arrival_markers = self._create_multi_channel_arrival_markers(wave_segments, len(channels))
        if arrival_markers:
            plot_data['annotations'] = arrival_markers
        
        return plot_data
    
    def _calculate_cross_correlation(self, channel1: ChannelData, channel2: ChannelData) -> Dict[str, Any]:
        """Calculate cross-correlation between two channels."""
        # Ensure both channels have the same sampling rate
        if channel1.sampling_rate != channel2.sampling_rate:
            raise ValueError("Channels must have the same sampling rate for correlation")
        
        # Make data the same length (truncate to shorter)
        min_length = min(len(channel1.data), len(channel2.data))
        data1 = channel1.data[:min_length]
        data2 = channel2.data[:min_length]
        
        # Detrend if configured
        if self.correlation_config['detrend']:
            data1 = signal.detrend(data1)
            data2 = signal.detrend(data2)
        
        # Calculate cross-correlation
        correlation = signal.correlate(data1, data2, mode='full')
        
        # Normalize if configured
        if self.correlation_config['normalize']:
            correlation = correlation / (np.std(data1) * np.std(data2) * len(data1))
        
        # Create lag array
        max_lag_samples = int(self.correlation_config['max_lag_seconds'] * channel1.sampling_rate)
        lags = signal.correlation_lags(len(data1), len(data2), mode='full')
        
        # Limit to specified lag range
        center = len(lags) // 2
        start_idx = max(0, center - max_lag_samples)
        end_idx = min(len(lags), center + max_lag_samples + 1)
        
        limited_lags = lags[start_idx:end_idx]
        limited_correlation = correlation[start_idx:end_idx]
        
        # Convert lags to time
        lag_times = limited_lags / channel1.sampling_rate
        
        # Find peak correlation
        peak_idx = np.argmax(np.abs(limited_correlation))
        peak_lag = lag_times[peak_idx]
        peak_correlation = limited_correlation[peak_idx]
        
        return {
            'channel1_id': channel1.channel_id,
            'channel2_id': channel2.channel_id,
            'lag_times': lag_times,
            'correlation': limited_correlation,
            'peak_lag': peak_lag,
            'peak_correlation': peak_correlation,
            'pair_label': f'{channel1.channel_id} vs {channel2.channel_id}'
        }
    
    def _calculate_coherence(self, channel1: ChannelData, channel2: ChannelData) -> Dict[str, Any]:
        """Calculate coherence between two channels."""
        # Ensure both channels have the same sampling rate
        if channel1.sampling_rate != channel2.sampling_rate:
            raise ValueError("Channels must have the same sampling rate for coherence")
        
        # Make data the same length (truncate to shorter)
        min_length = min(len(channel1.data), len(channel2.data))
        data1 = channel1.data[:min_length]
        data2 = channel2.data[:min_length]
        
        # Calculate coherence
        frequencies, coherence = signal.coherence(
            data1, data2,
            fs=channel1.sampling_rate,
            window=self.coherence_config['window'],
            nperseg=self.coherence_config['nperseg'],
            noverlap=self.coherence_config['noverlap']
        )
        
        # Calculate mean coherence in different frequency bands
        p_band_coherence = np.mean(coherence[(frequencies >= 5) & (frequencies <= 20)])
        s_band_coherence = np.mean(coherence[(frequencies >= 2) & (frequencies <= 10)])
        surface_band_coherence = np.mean(coherence[(frequencies >= 0.1) & (frequencies <= 5)])
        
        return {
            'channel1_id': channel1.channel_id,
            'channel2_id': channel2.channel_id,
            'frequencies': frequencies,
            'coherence': coherence,
            'p_band_coherence': p_band_coherence,
            's_band_coherence': s_band_coherence,
            'surface_band_coherence': surface_band_coherence,
            'pair_label': f'{channel1.channel_id} vs {channel2.channel_id}'
        }
    
    def _calculate_correlation_matrix(self, channels: List[ChannelData]) -> np.ndarray:
        """Calculate correlation matrix for all channel pairs."""
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
                    
                    # Detrend data
                    if self.correlation_config['detrend']:
                        data1 = signal.detrend(data1)
                        data2 = signal.detrend(data2)
                    
                    # Calculate Pearson correlation
                    corr_coeff, _ = pearsonr(data1, data2)
                    correlation_matrix[i, j] = corr_coeff
        
        return correlation_matrix
    
    def _create_channel_dataset(self, channel: ChannelData, color_index: int) -> Dict[str, Any]:
        """Create dataset for a single channel."""
        # Create time array
        time_points = np.linspace(0, channel.duration, len(channel.data))
        
        # Get color for this channel
        color = self.default_colors[color_index % len(self.default_colors)]
        
        dataset = {
            'label': f'Channel {channel.channel_id}',
            'data': [{'x': t, 'y': d} for t, d in zip(time_points, channel.data)],
            'borderColor': color,
            'backgroundColor': color + '20',  # Add transparency
            'borderWidth': 1,
            'pointRadius': 0,
            'fill': False,
            'tension': 0.1,
            'channel_id': channel.channel_id,
            'yAxisID': f'y{color_index + 1}' if color_index > 0 else 'y'  # Separate y-axes
        }
        
        return dataset
    
    def _create_correlation_dataset(self, correlation_data: Dict[str, Any], color_index: int) -> Dict[str, Any]:
        """Create dataset for cross-correlation plot."""
        color = self.default_colors[color_index % len(self.default_colors)]
        
        dataset = {
            'label': correlation_data['pair_label'],
            'data': [{'x': t, 'y': c} for t, c in zip(correlation_data['lag_times'], 
                                                      correlation_data['correlation'])],
            'borderColor': color,
            'backgroundColor': color + '20',
            'borderWidth': 2,
            'pointRadius': 0,
            'fill': False,
            'tension': 0.1,
            'peak_lag': correlation_data['peak_lag'],
            'peak_correlation': correlation_data['peak_correlation']
        }
        
        return dataset
    
    def _create_coherence_dataset(self, coherence_data: Dict[str, Any], color_index: int) -> Dict[str, Any]:
        """Create dataset for coherence plot."""
        color = self.default_colors[color_index % len(self.default_colors)]
        
        dataset = {
            'label': coherence_data['pair_label'],
            'data': [{'x': f, 'y': c} for f, c in zip(coherence_data['frequencies'], 
                                                      coherence_data['coherence'])],
            'borderColor': color,
            'backgroundColor': color + '20',
            'borderWidth': 2,
            'pointRadius': 0,
            'fill': True,
            'tension': 0.1,
            'p_band_coherence': coherence_data['p_band_coherence'],
            's_band_coherence': coherence_data['s_band_coherence'],
            'surface_band_coherence': coherence_data['surface_band_coherence']
        }
        
        return dataset
    
    def _create_correlation_summary(self, correlation_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary statistics for correlation analysis."""
        peak_correlations = [pair['peak_correlation'] for pair in correlation_pairs]
        peak_lags = [pair['peak_lag'] for pair in correlation_pairs]
        
        return {
            'max_correlation': float(np.max(np.abs(peak_correlations))),
            'mean_correlation': float(np.mean(np.abs(peak_correlations))),
            'std_correlation': float(np.std(peak_correlations)),
            'max_lag': float(np.max(np.abs(peak_lags))),
            'mean_lag': float(np.mean(np.abs(peak_lags))),
            'pair_count': len(correlation_pairs)
        }
    
    def _create_multi_channel_arrival_markers(self, wave_segments: List[WaveSegment], 
                                            n_channels: int) -> List[Dict[str, Any]]:
        """Create arrival time markers for multi-channel plots."""
        markers = []
        
        for wave in wave_segments:
            # Create marker that spans all channels
            marker = {
                'type': 'line',
                'mode': 'vertical',
                'scaleID': 'x',
                'value': wave.arrival_time,
                'borderColor': self._get_wave_color(wave.wave_type),
                'borderWidth': 2,
                'borderDash': [5, 5],
                'label': {
                    'content': f'{wave.wave_type} arrival',
                    'enabled': True,
                    'position': 'top'
                }
            }
            markers.append(marker)
        
        return markers
    
    def _get_wave_color(self, wave_type: str) -> str:
        """Get color for wave type."""
        wave_colors = {
            'P': '#FF6B6B',
            'S': '#4ECDC4',
            'Love': '#45B7D1',
            'Rayleigh': '#96CEB4'
        }
        return wave_colors.get(wave_type, '#333333')
    
    def _create_multi_channel_layout_config(self, n_channels: int) -> Dict[str, Any]:
        """Create layout configuration for multi-channel plots."""
        layout = {
            'responsive': True,
            'maintainAspectRatio': False,
            'scales': {
                'x': {
                    'type': 'linear',
                    'position': 'bottom',
                    'title': {
                        'display': True,
                        'text': 'Time (seconds)'
                    },
                    'grid': {
                        'display': True
                    }
                }
            },
            'plugins': {
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'tooltip': {
                    'mode': 'index',
                    'intersect': False
                },
                'zoom': {
                    'zoom': {
                        'wheel': {
                            'enabled': True
                        },
                        'pinch': {
                            'enabled': True
                        },
                        'mode': 'x'
                    },
                    'pan': {
                        'enabled': True,
                        'mode': 'x'
                    }
                }
            }
        }
        
        # Add separate y-axes for each channel
        for i in range(n_channels):
            y_axis_id = 'y' if i == 0 else f'y{i + 1}'
            layout['scales'][y_axis_id] = {
                'type': 'linear',
                'position': 'left' if i == 0 else 'right',
                'title': {
                    'display': True,
                    'text': f'Channel {i + 1} Amplitude'
                },
                'grid': {
                    'drawOnChartArea': i == 0  # Only show grid for first channel
                }
            }
        
        return layout
    
    def _create_correlation_layout_config(self) -> Dict[str, Any]:
        """Create layout configuration for correlation plots."""
        return {
            'responsive': True,
            'maintainAspectRatio': False,
            'scales': {
                'x': {
                    'type': 'linear',
                    'position': 'bottom',
                    'title': {
                        'display': True,
                        'text': 'Lag Time (seconds)'
                    },
                    'grid': {
                        'display': True
                    }
                },
                'y': {
                    'type': 'linear',
                    'title': {
                        'display': True,
                        'text': 'Cross-Correlation'
                    },
                    'grid': {
                        'display': True
                    }
                }
            },
            'plugins': {
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'tooltip': {
                    'mode': 'index',
                    'intersect': False
                }
            }
        }
    
    def _create_coherence_layout_config(self) -> Dict[str, Any]:
        """Create layout configuration for coherence plots."""
        return {
            'responsive': True,
            'maintainAspectRatio': False,
            'scales': {
                'x': {
                    'type': 'logarithmic',
                    'position': 'bottom',
                    'title': {
                        'display': True,
                        'text': 'Frequency (Hz)'
                    },
                    'min': 0.1,
                    'max': 50,
                    'grid': {
                        'display': True
                    }
                },
                'y': {
                    'type': 'linear',
                    'title': {
                        'display': True,
                        'text': 'Coherence'
                    },
                    'min': 0,
                    'max': 1,
                    'grid': {
                        'display': True
                    }
                }
            },
            'plugins': {
                'legend': {
                    'display': True,
                    'position': 'top'
                },
                'tooltip': {
                    'mode': 'index',
                    'intersect': False
                }
            }
        }
    
    def _create_matrix_layout_config(self, channel_labels: List[str]) -> Dict[str, Any]:
        """Create layout configuration for correlation matrix."""
        return {
            'title': 'Channel Correlation Matrix',
            'xaxis': {
                'title': 'Channels',
                'tickvals': list(range(len(channel_labels))),
                'ticktext': channel_labels
            },
            'yaxis': {
                'title': 'Channels',
                'tickvals': list(range(len(channel_labels))),
                'ticktext': channel_labels
            },
            'width': 600,
            'height': 600
        }
    
    def _create_comparison_layout_config(self, n_channels: int) -> Dict[str, Any]:
        """Create layout configuration for channel comparison plots."""
        return self._create_multi_channel_layout_config(n_channels)
    
    def _create_plot_config(self) -> Dict[str, Any]:
        """Create plot configuration."""
        return {
            'responsive': True,
            'interaction': {
                'mode': 'index',
                'intersect': False
            },
            'elements': {
                'point': {
                    'radius': 1
                },
                'line': {
                    'borderWidth': 1
                }
            }
        }
    
    def _create_empty_plot(self, message: str = 'No multi-channel data available') -> Dict[str, Any]:
        """Create empty plot with message."""
        return {
            'type': 'empty',
            'message': message,
            'datasets': [],
            'layout': self._create_multi_channel_layout_config(1),
            'config': self._create_plot_config()
        }
    
    def set_correlation_config(self, config: Dict[str, Any]) -> None:
        """Set custom correlation configuration."""
        self.correlation_config.update(config)
    
    def set_coherence_config(self, config: Dict[str, Any]) -> None:
        """Set custom coherence configuration."""
        self.coherence_config.update(config)
    
    def set_color_scheme(self, colors: List[str]) -> None:
        """Set custom color scheme for channels."""
        self.default_colors = colors
    
    def export_correlation_data(self, channels: List[ChannelData], format_type: str = 'json') -> str:
        """Export correlation analysis data in specified format."""
        if len(channels) < 2:
            raise ValueError("Need at least 2 channels for correlation export")
        
        correlation_data = {}
        
        # Calculate all pairwise correlations
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                corr_result = self._calculate_cross_correlation(channels[i], channels[j])
                pair_key = f'{channels[i].channel_id}_vs_{channels[j].channel_id}'
                correlation_data[pair_key] = {
                    'peak_correlation': float(corr_result['peak_correlation']),
                    'peak_lag': float(corr_result['peak_lag']),
                    'lag_times': corr_result['lag_times'].tolist(),
                    'correlation_values': corr_result['correlation'].tolist()
                }
        
        # Add correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(channels)
        correlation_data['correlation_matrix'] = correlation_matrix.tolist()
        correlation_data['channel_labels'] = [ch.channel_id for ch in channels]
        
        if format_type == 'json':
            return json.dumps(correlation_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def export_coherence_data(self, channels: List[ChannelData], format_type: str = 'json') -> str:
        """Export coherence analysis data in specified format."""
        if len(channels) < 2:
            raise ValueError("Need at least 2 channels for coherence export")
        
        coherence_data = {}
        
        # Calculate all pairwise coherences
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                coh_result = self._calculate_coherence(channels[i], channels[j])
                pair_key = f'{channels[i].channel_id}_vs_{channels[j].channel_id}'
                coherence_data[pair_key] = {
                    'frequencies': coh_result['frequencies'].tolist(),
                    'coherence_values': coh_result['coherence'].tolist(),
                    'p_band_coherence': float(coh_result['p_band_coherence']),
                    's_band_coherence': float(coh_result['s_band_coherence']),
                    'surface_band_coherence': float(coh_result['surface_band_coherence'])
                }
        
        if format_type == 'json':
            return json.dumps(coherence_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")