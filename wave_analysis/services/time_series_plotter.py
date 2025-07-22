"""
Time series plotting service for wave visualization.

This module provides comprehensive time-series visualization capabilities
for different wave types with interactive features.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import json

from ..models import WaveSegment, WaveAnalysisResult
from ..interfaces import WaveVisualizerInterface


class TimeSeriesPlotter(WaveVisualizerInterface):
    """
    Creates time-series visualizations for seismic wave data.
    
    Supports separate plotting for P-waves, S-waves, and surface waves
    with interactive zoom and pan capabilities.
    """
    
    def __init__(self):
        """Initialize the time series plotter."""
        self.default_colors = {
            'P': '#FF6B6B',      # Red for P-waves
            'S': '#4ECDC4',      # Teal for S-waves  
            'Love': '#45B7D1',   # Blue for Love waves
            'Rayleigh': '#96CEB4' # Green for Rayleigh waves
        }
        self.plot_config = {
            'line_width': 2,
            'marker_size': 4,
            'grid': True,
            'interactive': True,
            'zoom_enabled': True,
            'pan_enabled': True
        }
    
    def create_time_series_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create time series visualization for wave segments.
        
        Args:
            wave_segments: List of wave segments to visualize
            
        Returns:
            Plot data dictionary for rendering with Chart.js or Plotly
        """
        if not wave_segments:
            return self._create_empty_plot()
        
        # Group waves by type
        waves_by_type = self._group_waves_by_type(wave_segments)
        
        # Create plot data for each wave type
        plot_data = {
            'type': 'time_series',
            'title': 'Seismic Wave Time Series Analysis',
            'datasets': [],
            'layout': self._create_layout_config(),
            'config': self._create_plot_config()
        }
        
        for wave_type, waves in waves_by_type.items():
            dataset = self._create_wave_dataset(wave_type, waves)
            plot_data['datasets'].append(dataset)
        
        # Add arrival time markers
        arrival_markers = self._create_arrival_markers(wave_segments)
        if arrival_markers:
            plot_data['annotations'] = arrival_markers
        
        return plot_data
    
    def create_frequency_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create frequency domain visualization (implemented in FrequencyPlotter).
        
        This method is part of the interface but delegates to FrequencyPlotter.
        """
        # This will be implemented in the FrequencyPlotter class
        return {'type': 'frequency', 'message': 'Use FrequencyPlotter for frequency analysis'}
    
    def create_p_wave_plot(self, p_waves: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create specialized plot for P-waves only.
        
        Args:
            p_waves: List of P-wave segments
            
        Returns:
            Plot data optimized for P-wave characteristics
        """
        if not p_waves:
            return self._create_empty_plot('No P-waves detected')
        
        plot_data = {
            'type': 'p_wave_analysis',
            'title': 'P-Wave Analysis',
            'datasets': [],
            'layout': self._create_layout_config(),
            'config': self._create_plot_config()
        }
        
        # Create main P-wave trace
        p_wave_data = self._create_wave_dataset('P', p_waves)
        plot_data['datasets'].append(p_wave_data)
        
        # Add onset markers
        onset_markers = self._create_onset_markers(p_waves)
        if onset_markers:
            plot_data['annotations'] = onset_markers
        
        # Add characteristic function if available
        char_func_data = self._create_characteristic_function_trace(p_waves)
        if char_func_data:
            plot_data['datasets'].append(char_func_data)
        
        return plot_data
    
    def create_s_wave_plot(self, s_waves: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create specialized plot for S-waves only.
        
        Args:
            s_waves: List of S-wave segments
            
        Returns:
            Plot data optimized for S-wave characteristics
        """
        if not s_waves:
            return self._create_empty_plot('No S-waves detected')
        
        plot_data = {
            'type': 's_wave_analysis',
            'title': 'S-Wave Analysis',
            'datasets': [],
            'layout': self._create_layout_config(),
            'config': self._create_plot_config()
        }
        
        # Create main S-wave trace
        s_wave_data = self._create_wave_dataset('S', s_waves)
        plot_data['datasets'].append(s_wave_data)
        
        # Add polarization analysis if available
        polarization_data = self._create_polarization_trace(s_waves)
        if polarization_data:
            plot_data['datasets'].extend(polarization_data)
        
        return plot_data
    
    def create_surface_wave_plot(self, surface_waves: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create specialized plot for surface waves.
        
        Args:
            surface_waves: List of surface wave segments
            
        Returns:
            Plot data optimized for surface wave characteristics
        """
        if not surface_waves:
            return self._create_empty_plot('No surface waves detected')
        
        plot_data = {
            'type': 'surface_wave_analysis',
            'title': 'Surface Wave Analysis',
            'datasets': [],
            'layout': self._create_layout_config(),
            'config': self._create_plot_config()
        }
        
        # Separate Love and Rayleigh waves
        love_waves = [w for w in surface_waves if w.wave_type == 'Love']
        rayleigh_waves = [w for w in surface_waves if w.wave_type == 'Rayleigh']
        
        if love_waves:
            love_data = self._create_wave_dataset('Love', love_waves)
            plot_data['datasets'].append(love_data)
        
        if rayleigh_waves:
            rayleigh_data = self._create_wave_dataset('Rayleigh', rayleigh_waves)
            plot_data['datasets'].append(rayleigh_data)
        
        # Add group velocity analysis if available
        velocity_data = self._create_group_velocity_trace(surface_waves)
        if velocity_data:
            plot_data['datasets'].append(velocity_data)
        
        return plot_data
    
    def create_multi_wave_comparison(self, wave_result: WaveAnalysisResult) -> Dict[str, Any]:
        """
        Create comparison plot showing all wave types together.
        
        Args:
            wave_result: Complete wave analysis result
            
        Returns:
            Plot data showing all wave types in context
        """
        plot_data = {
            'type': 'multi_wave_comparison',
            'title': 'Complete Wave Analysis',
            'datasets': [],
            'layout': self._create_layout_config(),
            'config': self._create_plot_config()
        }
        
        # Add original waveform as background
        original_trace = self._create_original_waveform_trace(wave_result)
        plot_data['datasets'].append(original_trace)
        
        # Add each wave type
        all_waves = wave_result.p_waves + wave_result.s_waves + wave_result.surface_waves
        waves_by_type = self._group_waves_by_type(all_waves)
        
        for wave_type, waves in waves_by_type.items():
            dataset = self._create_wave_dataset(wave_type, waves)
            plot_data['datasets'].append(dataset)
        
        # Add phase arrival annotations
        phase_annotations = self._create_phase_annotations(wave_result)
        if phase_annotations:
            plot_data['annotations'] = phase_annotations
        
        return plot_data
    
    def _group_waves_by_type(self, wave_segments: List[WaveSegment]) -> Dict[str, List[WaveSegment]]:
        """Group wave segments by their type."""
        waves_by_type = {}
        for wave in wave_segments:
            if wave.wave_type not in waves_by_type:
                waves_by_type[wave.wave_type] = []
            waves_by_type[wave.wave_type].append(wave)
        return waves_by_type
    
    def _create_wave_dataset(self, wave_type: str, waves: List[WaveSegment]) -> Dict[str, Any]:
        """Create dataset for a specific wave type."""
        # Combine all wave segments of the same type
        combined_data = []
        combined_times = []
        
        for wave in waves:
            # Create time array for this wave segment
            duration = wave.end_time - wave.start_time
            time_points = np.linspace(wave.start_time, wave.end_time, len(wave.data))
            
            combined_times.extend(time_points.tolist())
            combined_data.extend(wave.data.tolist())
        
        dataset = {
            'label': f'{wave_type}-waves',
            'data': [{'x': t, 'y': d} for t, d in zip(combined_times, combined_data)],
            'borderColor': self.default_colors.get(wave_type, '#333333'),
            'backgroundColor': self.default_colors.get(wave_type, '#333333') + '20',  # Add transparency
            'borderWidth': self.plot_config['line_width'],
            'pointRadius': 0,  # No points for time series
            'fill': False,
            'tension': 0.1,
            'wave_type': wave_type,
            'wave_count': len(waves)
        }
        
        return dataset
    
    def _create_arrival_markers(self, wave_segments: List[WaveSegment]) -> List[Dict[str, Any]]:
        """Create arrival time markers for wave segments."""
        markers = []
        
        for wave in wave_segments:
            marker = {
                'type': 'line',
                'mode': 'vertical',
                'scaleID': 'x',
                'value': wave.arrival_time,
                'borderColor': self.default_colors.get(wave.wave_type, '#333333'),
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
    
    def _create_onset_markers(self, waves: List[WaveSegment]) -> List[Dict[str, Any]]:
        """Create onset markers for wave detection."""
        markers = []
        
        for i, wave in enumerate(waves):
            marker = {
                'type': 'point',
                'xValue': wave.arrival_time,
                'yValue': wave.peak_amplitude,
                'backgroundColor': self.default_colors.get(wave.wave_type, '#333333'),
                'borderColor': '#FFFFFF',
                'borderWidth': 2,
                'radius': 6,
                'label': {
                    'content': f'Onset {i+1}',
                    'enabled': True,
                    'position': 'top'
                }
            }
            markers.append(marker)
        
        return markers
    
    def _create_characteristic_function_trace(self, p_waves: List[WaveSegment]) -> Optional[Dict[str, Any]]:
        """Create characteristic function trace for P-wave detection."""
        if not p_waves or 'characteristic_function' not in p_waves[0].metadata:
            return None
        
        # This would be populated by the P-wave detector
        char_func = p_waves[0].metadata.get('characteristic_function', [])
        if len(char_func) == 0:
            return None
        
        # Create time array
        start_time = p_waves[0].start_time
        end_time = p_waves[0].end_time
        time_points = np.linspace(start_time, end_time, len(char_func))
        
        return {
            'label': 'Characteristic Function',
            'data': [{'x': t, 'y': cf} for t, cf in zip(time_points, char_func)],
            'borderColor': '#FFA500',
            'backgroundColor': '#FFA50020',
            'borderWidth': 1,
            'pointRadius': 0,
            'fill': True,
            'yAxisID': 'y1'  # Secondary y-axis
        }
    
    def _create_polarization_trace(self, s_waves: List[WaveSegment]) -> List[Dict[str, Any]]:
        """Create polarization analysis traces for S-waves."""
        traces = []
        
        for wave in s_waves:
            if 'horizontal_component' in wave.metadata and 'vertical_component' in wave.metadata:
                h_component = wave.metadata['horizontal_component']
                v_component = wave.metadata['vertical_component']
                
                duration = wave.end_time - wave.start_time
                time_points = np.linspace(wave.start_time, wave.end_time, len(h_component))
                
                # Horizontal component
                traces.append({
                    'label': 'Horizontal Component',
                    'data': [{'x': t, 'y': h} for t, h in zip(time_points, h_component)],
                    'borderColor': '#FF9999',
                    'borderWidth': 1,
                    'pointRadius': 0,
                    'fill': False
                })
                
                # Vertical component
                traces.append({
                    'label': 'Vertical Component',
                    'data': [{'x': t, 'y': v} for t, v in zip(time_points, v_component)],
                    'borderColor': '#99FF99',
                    'borderWidth': 1,
                    'pointRadius': 0,
                    'fill': False
                })
        
        return traces
    
    def _create_group_velocity_trace(self, surface_waves: List[WaveSegment]) -> Optional[Dict[str, Any]]:
        """Create group velocity analysis trace for surface waves."""
        # This would be populated by surface wave analysis
        # For now, return None as it requires advanced dispersion analysis
        return None
    
    def _create_original_waveform_trace(self, wave_result: WaveAnalysisResult) -> Dict[str, Any]:
        """Create trace for the original waveform."""
        duration = len(wave_result.original_data) / wave_result.sampling_rate
        time_points = np.linspace(0, duration, len(wave_result.original_data))
        
        return {
            'label': 'Original Waveform',
            'data': [{'x': t, 'y': d} for t, d in zip(time_points, wave_result.original_data)],
            'borderColor': '#CCCCCC',
            'backgroundColor': '#CCCCCC10',
            'borderWidth': 1,
            'pointRadius': 0,
            'fill': False,
            'order': 10  # Draw behind other traces
        }
    
    def _create_phase_annotations(self, wave_result: WaveAnalysisResult) -> List[Dict[str, Any]]:
        """Create phase arrival annotations for complete analysis."""
        annotations = []
        
        # P-wave arrivals
        for wave in wave_result.p_waves:
            annotations.append({
                'type': 'line',
                'mode': 'vertical',
                'scaleID': 'x',
                'value': wave.arrival_time,
                'borderColor': self.default_colors['P'],
                'borderWidth': 2,
                'label': {
                    'content': 'P',
                    'enabled': True,
                    'position': 'top'
                }
            })
        
        # S-wave arrivals
        for wave in wave_result.s_waves:
            annotations.append({
                'type': 'line',
                'mode': 'vertical',
                'scaleID': 'x',
                'value': wave.arrival_time,
                'borderColor': self.default_colors['S'],
                'borderWidth': 2,
                'label': {
                    'content': 'S',
                    'enabled': True,
                    'position': 'top'
                }
            })
        
        # Surface wave arrivals
        for wave in wave_result.surface_waves:
            annotations.append({
                'type': 'line',
                'mode': 'vertical',
                'scaleID': 'x',
                'value': wave.arrival_time,
                'borderColor': self.default_colors[wave.wave_type],
                'borderWidth': 2,
                'label': {
                    'content': wave.wave_type[0],  # 'L' or 'R'
                    'enabled': True,
                    'position': 'top'
                }
            })
        
        return annotations
    
    def _create_layout_config(self) -> Dict[str, Any]:
        """Create layout configuration for plots."""
        return {
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
                        'display': self.plot_config['grid']
                    }
                },
                'y': {
                    'type': 'linear',
                    'title': {
                        'display': True,
                        'text': 'Amplitude'
                    },
                    'grid': {
                        'display': self.plot_config['grid']
                    }
                },
                'y1': {  # Secondary y-axis for characteristic functions
                    'type': 'linear',
                    'position': 'right',
                    'title': {
                        'display': True,
                        'text': 'Characteristic Function'
                    },
                    'grid': {
                        'drawOnChartArea': False
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
                            'enabled': self.plot_config['zoom_enabled']
                        },
                        'pinch': {
                            'enabled': self.plot_config['zoom_enabled']
                        },
                        'mode': 'x'
                    },
                    'pan': {
                        'enabled': self.plot_config['pan_enabled'],
                        'mode': 'x'
                    }
                }
            }
        }
    
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
                    'radius': self.plot_config['marker_size']
                },
                'line': {
                    'borderWidth': self.plot_config['line_width']
                }
            }
        }
    
    def _create_empty_plot(self, message: str = 'No wave data available') -> Dict[str, Any]:
        """Create empty plot with message."""
        return {
            'type': 'empty',
            'message': message,
            'datasets': [],
            'layout': self._create_layout_config(),
            'config': self._create_plot_config()
        }
    
    def set_color_scheme(self, colors: Dict[str, str]) -> None:
        """Set custom color scheme for wave types."""
        self.default_colors.update(colors)
    
    def set_plot_config(self, config: Dict[str, Any]) -> None:
        """Set custom plot configuration."""
        self.plot_config.update(config)
    
    def export_plot_data(self, plot_data: Dict[str, Any], format_type: str = 'json') -> str:
        """Export plot data in specified format."""
        if format_type == 'json':
            return json.dumps(plot_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")