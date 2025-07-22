"""
Frequency domain plotting service for wave visualization.

This module provides comprehensive frequency-domain visualization capabilities
including spectrograms and frequency spectrum plots for different wave types.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import json

from ..models import WaveSegment, FrequencyData
from ..interfaces import WaveVisualizerInterface


class FrequencyPlotter(WaveVisualizerInterface):
    """
    Creates frequency-domain visualizations for seismic wave data.
    
    Supports spectrogram plotting for time-frequency analysis and
    frequency spectrum plots for individual wave components.
    """
    
    def __init__(self):
        """Initialize the frequency plotter."""
        self.default_colors = {
            'P': '#FF6B6B',      # Red for P-waves
            'S': '#4ECDC4',      # Teal for S-waves  
            'Love': '#45B7D1',   # Blue for Love waves
            'Rayleigh': '#96CEB4' # Green for Rayleigh waves
        }
        self.spectrogram_config = {
            'window': 'hann',
            'nperseg': 256,
            'noverlap': 128,
            'nfft': 512,
            'colormap': 'viridis',
            'log_scale': True
        }
        self.spectrum_config = {
            'window': 'hann',
            'detrend': 'linear',
            'scaling': 'density'
        }
    
    def create_time_series_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create time series visualization (delegated to TimeSeriesPlotter).
        
        This method is part of the interface but delegates to TimeSeriesPlotter.
        """
        return {'type': 'time_series', 'message': 'Use TimeSeriesPlotter for time series analysis'}
    
    def create_frequency_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create frequency domain visualization for wave segments.
        
        Args:
            wave_segments: List of wave segments to analyze
            
        Returns:
            Frequency plot data dictionary for rendering
        """
        if not wave_segments:
            return self._create_empty_plot('No wave data available for frequency analysis')
        
        # Group waves by type
        waves_by_type = self._group_waves_by_type(wave_segments)
        
        # Create frequency spectrum plot
        plot_data = {
            'type': 'frequency_spectrum',
            'title': 'Frequency Spectrum Analysis',
            'datasets': [],
            'layout': self._create_frequency_layout_config(),
            'config': self._create_plot_config()
        }
        
        for wave_type, waves in waves_by_type.items():
            spectrum_data = self._create_frequency_spectrum_dataset(wave_type, waves)
            plot_data['datasets'].append(spectrum_data)
        
        # Add dominant frequency markers
        freq_markers = self._create_frequency_markers(wave_segments)
        if freq_markers:
            plot_data['annotations'] = freq_markers
        
        return plot_data
    
    def create_spectrogram_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create spectrogram visualization for time-frequency analysis.
        
        Args:
            wave_segments: List of wave segments to analyze
            
        Returns:
            Spectrogram plot data dictionary for rendering
        """
        if not wave_segments:
            return self._create_empty_plot('No wave data available for spectrogram')
        
        # For spectrogram, we typically use the longest or most significant wave
        primary_wave = max(wave_segments, key=lambda w: w.duration)
        
        # Calculate spectrogram
        frequencies, times, Sxx = self._calculate_spectrogram(primary_wave)
        
        plot_data = {
            'type': 'spectrogram',
            'title': f'Spectrogram - {primary_wave.wave_type} Wave',
            'data': {
                'z': Sxx.tolist(),  # Power spectral density
                'x': times.tolist(),  # Time axis
                'y': frequencies.tolist(),  # Frequency axis
                'type': 'heatmap',
                'colorscale': self.spectrogram_config['colormap'],
                'showscale': True,
                'colorbar': {
                    'title': 'Power Spectral Density (dB)'
                }
            },
            'layout': self._create_spectrogram_layout_config(primary_wave),
            'config': self._create_plot_config()
        }
        
        # Add wave arrival markers
        arrival_markers = self._create_spectrogram_markers(wave_segments, times)
        if arrival_markers:
            plot_data['annotations'] = arrival_markers
        
        return plot_data
    
    def create_frequency_comparison_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create frequency comparison plot showing all wave types together.
        
        Args:
            wave_segments: List of wave segments to compare
            
        Returns:
            Frequency comparison plot data
        """
        if not wave_segments:
            return self._create_empty_plot('No wave data available')
        
        waves_by_type = self._group_waves_by_type(wave_segments)
        
        plot_data = {
            'type': 'frequency_comparison',
            'title': 'Wave Type Frequency Comparison',
            'datasets': [],
            'layout': self._create_frequency_layout_config(),
            'config': self._create_plot_config()
        }
        
        # Create overlaid frequency spectra
        for wave_type, waves in waves_by_type.items():
            if waves:
                spectrum_data = self._create_frequency_spectrum_dataset(wave_type, waves)
                spectrum_data['fill'] = False
                spectrum_data['borderWidth'] = 2
                plot_data['datasets'].append(spectrum_data)
        
        # Add frequency band annotations
        band_annotations = self._create_frequency_band_annotations()
        if band_annotations:
            plot_data['annotations'] = band_annotations
        
        return plot_data
    
    def create_power_spectral_density_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create power spectral density plot for wave segments.
        
        Args:
            wave_segments: List of wave segments to analyze
            
        Returns:
            PSD plot data dictionary
        """
        if not wave_segments:
            return self._create_empty_plot('No wave data available')
        
        plot_data = {
            'type': 'power_spectral_density',
            'title': 'Power Spectral Density Analysis',
            'datasets': [],
            'layout': self._create_psd_layout_config(),
            'config': self._create_plot_config()
        }
        
        waves_by_type = self._group_waves_by_type(wave_segments)
        
        for wave_type, waves in waves_by_type.items():
            if waves:
                psd_data = self._create_psd_dataset(wave_type, waves)
                plot_data['datasets'].append(psd_data)
        
        return plot_data
    
    def _calculate_spectrogram(self, wave: WaveSegment) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate spectrogram for a wave segment."""
        frequencies, times, Sxx = signal.spectrogram(
            wave.data,
            fs=wave.sampling_rate,
            window=self.spectrogram_config['window'],
            nperseg=self.spectrogram_config['nperseg'],
            noverlap=self.spectrogram_config['noverlap'],
            nfft=self.spectrogram_config['nfft'],
            scaling='density'
        )
        
        # Convert to dB scale if configured
        if self.spectrogram_config['log_scale']:
            Sxx = 10 * np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
        
        # Adjust time axis to match wave timing
        times = times + wave.start_time
        
        return frequencies, times, Sxx
    
    def _calculate_frequency_spectrum(self, wave: WaveSegment) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate frequency spectrum for a wave segment."""
        # Apply window function
        if self.spectrum_config['window']:
            window = signal.get_window(self.spectrum_config['window'], len(wave.data))
            windowed_data = wave.data * window
        else:
            windowed_data = wave.data
        
        # Detrend if configured
        if self.spectrum_config['detrend']:
            windowed_data = signal.detrend(windowed_data, type=self.spectrum_config['detrend'])
        
        # Calculate FFT
        fft_data = fft(windowed_data)
        frequencies = fftfreq(len(windowed_data), 1/wave.sampling_rate)
        
        # Take positive frequencies only
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        fft_data = fft_data[positive_freq_idx]
        
        # Calculate power spectral density
        if self.spectrum_config['scaling'] == 'density':
            psd = np.abs(fft_data)**2 / (wave.sampling_rate * len(windowed_data))
        else:
            psd = np.abs(fft_data)**2
        
        return frequencies, psd
    
    def _calculate_power_spectral_density(self, wave: WaveSegment) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate power spectral density using Welch's method."""
        frequencies, psd = signal.welch(
            wave.data,
            fs=wave.sampling_rate,
            window=self.spectrum_config['window'],
            nperseg=min(len(wave.data)//4, 256),
            scaling=self.spectrum_config['scaling']
        )
        
        return frequencies, psd
    
    def _group_waves_by_type(self, wave_segments: List[WaveSegment]) -> Dict[str, List[WaveSegment]]:
        """Group wave segments by their type."""
        waves_by_type = {}
        for wave in wave_segments:
            if wave.wave_type not in waves_by_type:
                waves_by_type[wave.wave_type] = []
            waves_by_type[wave.wave_type].append(wave)
        return waves_by_type
    
    def _create_frequency_spectrum_dataset(self, wave_type: str, waves: List[WaveSegment]) -> Dict[str, Any]:
        """Create frequency spectrum dataset for a wave type."""
        # Combine spectra from all waves of the same type
        combined_frequencies = []
        combined_psd = []
        
        for wave in waves:
            frequencies, psd = self._calculate_frequency_spectrum(wave)
            
            # Interpolate to common frequency grid if needed
            if not combined_frequencies:
                combined_frequencies = frequencies
                combined_psd = psd
            else:
                # Average with existing spectrum
                combined_psd = (combined_psd + np.interp(combined_frequencies, frequencies, psd)) / 2
        
        dataset = {
            'label': f'{wave_type} Wave Spectrum',
            'data': [{'x': f, 'y': p} for f, p in zip(combined_frequencies, combined_psd)],
            'borderColor': self.default_colors.get(wave_type, '#333333'),
            'backgroundColor': self.default_colors.get(wave_type, '#333333') + '20',
            'borderWidth': 2,
            'pointRadius': 0,
            'fill': True,
            'tension': 0.1,
            'wave_type': wave_type,
            'wave_count': len(waves)
        }
        
        return dataset
    
    def _create_psd_dataset(self, wave_type: str, waves: List[WaveSegment]) -> Dict[str, Any]:
        """Create power spectral density dataset for a wave type."""
        # Use the wave with highest confidence or longest duration
        primary_wave = max(waves, key=lambda w: w.confidence * w.duration)
        
        frequencies, psd = self._calculate_power_spectral_density(primary_wave)
        
        # Convert to dB
        psd_db = 10 * np.log10(psd + 1e-10)
        
        dataset = {
            'label': f'{wave_type} Wave PSD',
            'data': [{'x': f, 'y': p} for f, p in zip(frequencies, psd_db)],
            'borderColor': self.default_colors.get(wave_type, '#333333'),
            'backgroundColor': self.default_colors.get(wave_type, '#333333') + '20',
            'borderWidth': 2,
            'pointRadius': 0,
            'fill': False,
            'tension': 0.1,
            'wave_type': wave_type
        }
        
        return dataset
    
    def _create_frequency_markers(self, wave_segments: List[WaveSegment]) -> List[Dict[str, Any]]:
        """Create dominant frequency markers."""
        markers = []
        
        for wave in wave_segments:
            marker = {
                'type': 'line',
                'mode': 'vertical',
                'scaleID': 'x',
                'value': wave.dominant_frequency,
                'borderColor': self.default_colors.get(wave.wave_type, '#333333'),
                'borderWidth': 2,
                'borderDash': [5, 5],
                'label': {
                    'content': f'{wave.wave_type}: {wave.dominant_frequency:.1f} Hz',
                    'enabled': True,
                    'position': 'top'
                }
            }
            markers.append(marker)
        
        return markers
    
    def _create_spectrogram_markers(self, wave_segments: List[WaveSegment], times: np.ndarray) -> List[Dict[str, Any]]:
        """Create arrival time markers for spectrogram."""
        markers = []
        
        for wave in wave_segments:
            if wave.arrival_time >= times[0] and wave.arrival_time <= times[-1]:
                marker = {
                    'type': 'line',
                    'mode': 'vertical',
                    'scaleID': 'x',
                    'value': wave.arrival_time,
                    'borderColor': 'white',
                    'borderWidth': 2,
                    'label': {
                        'content': f'{wave.wave_type} arrival',
                        'enabled': True,
                        'position': 'top',
                        'color': 'white'
                    }
                }
                markers.append(marker)
        
        return markers
    
    def _create_frequency_band_annotations(self) -> List[Dict[str, Any]]:
        """Create frequency band annotations for seismic waves."""
        bands = [
            {'name': 'P-wave band', 'min': 5, 'max': 20, 'color': '#FF6B6B40'},
            {'name': 'S-wave band', 'min': 2, 'max': 10, 'color': '#4ECDC440'},
            {'name': 'Surface wave band', 'min': 0.1, 'max': 5, 'color': '#45B7D140'}
        ]
        
        annotations = []
        for band in bands:
            annotation = {
                'type': 'box',
                'xMin': band['min'],
                'xMax': band['max'],
                'backgroundColor': band['color'],
                'borderColor': band['color'][:-2],  # Remove transparency
                'borderWidth': 1,
                'label': {
                    'content': band['name'],
                    'enabled': True,
                    'position': 'center'
                }
            }
            annotations.append(annotation)
        
        return annotations
    
    def _create_frequency_layout_config(self) -> Dict[str, Any]:
        """Create layout configuration for frequency plots."""
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
                        'text': 'Power Spectral Density'
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
                        'mode': 'xy'
                    },
                    'pan': {
                        'enabled': True,
                        'mode': 'xy'
                    }
                }
            }
        }
    
    def _create_spectrogram_layout_config(self, wave: WaveSegment) -> Dict[str, Any]:
        """Create layout configuration for spectrogram plots."""
        return {
            'title': f'Spectrogram - {wave.wave_type} Wave',
            'xaxis': {
                'title': 'Time (seconds)',
                'range': [wave.start_time, wave.end_time]
            },
            'yaxis': {
                'title': 'Frequency (Hz)',
                'range': [0, min(wave.sampling_rate/2, 25)]  # Nyquist frequency or 25 Hz
            },
            'coloraxis': {
                'colorbar': {
                    'title': 'Power (dB)'
                }
            }
        }
    
    def _create_psd_layout_config(self) -> Dict[str, Any]:
        """Create layout configuration for PSD plots."""
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
                        'text': 'Power Spectral Density (dB/Hz)'
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
                    'radius': 2
                },
                'line': {
                    'borderWidth': 2
                }
            }
        }
    
    def _create_empty_plot(self, message: str = 'No frequency data available') -> Dict[str, Any]:
        """Create empty plot with message."""
        return {
            'type': 'empty',
            'message': message,
            'datasets': [],
            'layout': self._create_frequency_layout_config(),
            'config': self._create_plot_config()
        }
    
    def set_spectrogram_config(self, config: Dict[str, Any]) -> None:
        """Set custom spectrogram configuration."""
        self.spectrogram_config.update(config)
    
    def set_spectrum_config(self, config: Dict[str, Any]) -> None:
        """Set custom spectrum configuration."""
        self.spectrum_config.update(config)
    
    def set_color_scheme(self, colors: Dict[str, str]) -> None:
        """Set custom color scheme for wave types."""
        self.default_colors.update(colors)
    
    def export_frequency_data(self, wave_segments: List[WaveSegment], format_type: str = 'json') -> str:
        """Export frequency analysis data in specified format."""
        frequency_data = {}
        
        for wave in wave_segments:
            frequencies, psd = self._calculate_frequency_spectrum(wave)
            frequency_data[f'{wave.wave_type}_{wave.arrival_time}'] = {
                'frequencies': frequencies.tolist(),
                'power_spectral_density': psd.tolist(),
                'dominant_frequency': wave.dominant_frequency,
                'wave_type': wave.wave_type
            }
        
        if format_type == 'json':
            return json.dumps(frequency_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")