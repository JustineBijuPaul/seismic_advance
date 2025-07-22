"""
Frequency Analyzer for spectral analysis of wave components.

This module implements sophisticated algorithms for analyzing the frequency
characteristics of different wave types using FFT-based methods and
spectral analysis techniques.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import signal
from scipy.fft import fft, fftfreq
import logging

from ..models import WaveSegment, FrequencyData
from ..interfaces import WaveAnalyzerInterface


logger = logging.getLogger(__name__)


class FrequencyAnalyzer:
    """
    Analyzer for frequency domain characteristics of seismic waves.
    
    This class implements various spectral analysis methods for determining
    frequency characteristics of P-waves, S-waves, and surface waves.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize the frequency analyzer.
        
        Args:
            sampling_rate: Sampling rate of the seismic data in Hz
        """
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # Default parameters for frequency analysis
        self.window_type = 'hann'
        self.nperseg = 1024  # Segment length for spectrograms
        self.noverlap = 512  # Overlap for spectrograms
        self.frequency_bands = {
            'P': (1.0, 15.0),      # P-wave typical frequency range
            'S': (0.5, 10.0),      # S-wave typical frequency range
            'Love': (0.02, 0.5),   # Love wave frequency range
            'Rayleigh': (0.02, 0.5) # Rayleigh wave frequency range
        }
        
    def analyze_wave_frequencies(self, waves: Dict[str, List[WaveSegment]]) -> Dict[str, FrequencyData]:
        """
        Analyze frequency characteristics for all wave types.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            
        Returns:
            Dictionary mapping wave types to their frequency analysis
        """
        frequency_results = {}
        
        for wave_type, wave_segments in waves.items():
            if wave_segments:
                # Analyze the most prominent wave of each type
                best_wave = max(wave_segments, key=lambda w: w.peak_amplitude)
                freq_data = self.analyze_single_wave_frequency(best_wave)
                frequency_results[wave_type] = freq_data
                
        return frequency_results
    
    def analyze_single_wave_frequency(self, wave: WaveSegment) -> FrequencyData:
        """
        Analyze frequency characteristics of a single wave segment.
        
        Args:
            wave: Wave segment to analyze
            
        Returns:
            FrequencyData object with spectral analysis results
        """
        # Ensure we have enough data for meaningful analysis
        if len(wave.data) < 64:
            self.logger.warning(f"Wave segment too short for reliable frequency analysis: {len(wave.data)} samples")
        
        # Calculate power spectral density
        frequencies, power_spectrum = self._calculate_power_spectrum(wave.data)
        
        # Find dominant frequency
        dominant_frequency = self._find_dominant_frequency(frequencies, power_spectrum, wave.wave_type)
        
        # Calculate frequency range with significant energy
        frequency_range = self._calculate_frequency_range(frequencies, power_spectrum)
        
        # Calculate spectral centroid
        spectral_centroid = self._calculate_spectral_centroid(frequencies, power_spectrum)
        
        # Calculate spectral bandwidth
        bandwidth = self._calculate_spectral_bandwidth(frequencies, power_spectrum, spectral_centroid)
        
        return FrequencyData(
            frequencies=frequencies,
            power_spectrum=power_spectrum,
            dominant_frequency=dominant_frequency,
            frequency_range=frequency_range,
            spectral_centroid=spectral_centroid,
            bandwidth=bandwidth
        )
    
    def _calculate_power_spectrum(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate power spectral density using Welch's method.
        
        Args:
            data: Time series data
            
        Returns:
            Tuple of (frequencies, power_spectrum)
        """
        # Use Welch's method for robust PSD estimation
        nperseg = min(self.nperseg, len(data) // 4)  # Ensure reasonable segment size
        noverlap = nperseg // 2
        
        if nperseg < 64:
            # For very short segments, use simple FFT
            frequencies = fftfreq(len(data), 1/self.sampling_rate)[:len(data)//2]
            fft_data = fft(data)
            power_spectrum = np.abs(fft_data[:len(data)//2])**2
        else:
            frequencies, power_spectrum = signal.welch(
                data, 
                fs=self.sampling_rate,
                window=self.window_type,
                nperseg=nperseg,
                noverlap=noverlap
            )
        
        return frequencies, power_spectrum
    
    def _find_dominant_frequency(self, frequencies: np.ndarray, 
                               power_spectrum: np.ndarray, 
                               wave_type: str) -> float:
        """
        Find the dominant frequency within the expected range for the wave type.
        
        Args:
            frequencies: Frequency array
            power_spectrum: Power spectral density
            wave_type: Type of wave ('P', 'S', 'Love', 'Rayleigh')
            
        Returns:
            Dominant frequency in Hz
        """
        # Get expected frequency range for this wave type
        freq_range = self.frequency_bands.get(wave_type, (0.1, 20.0))
        
        # Find indices within the expected frequency range
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        
        if not np.any(freq_mask):
            # If no frequencies in expected range, use full spectrum
            freq_mask = frequencies > 0
        
        # Find peak within the range
        masked_spectrum = power_spectrum.copy()
        masked_spectrum[~freq_mask] = 0
        
        peak_idx = np.argmax(masked_spectrum)
        dominant_freq = frequencies[peak_idx]
        
        self.logger.debug(f"Dominant frequency for {wave_type}-wave: {dominant_freq:.2f} Hz")
        
        return dominant_freq
    
    def _calculate_frequency_range(self, frequencies: np.ndarray, 
                                 power_spectrum: np.ndarray) -> Tuple[float, float]:
        """
        Calculate frequency range containing significant energy.
        
        Args:
            frequencies: Frequency array
            power_spectrum: Power spectral density
            
        Returns:
            Tuple of (min_freq, max_freq) containing 90% of energy
        """
        # Calculate cumulative energy
        total_energy = np.sum(power_spectrum)
        cumulative_energy = np.cumsum(power_spectrum)
        
        # Find frequencies containing 5% to 95% of energy
        energy_5_percent = 0.05 * total_energy
        energy_95_percent = 0.95 * total_energy
        
        min_idx = np.argmax(cumulative_energy >= energy_5_percent)
        max_idx = np.argmax(cumulative_energy >= energy_95_percent)
        
        min_freq = frequencies[min_idx]
        max_freq = frequencies[max_idx]
        
        return (min_freq, max_freq)
    
    def _calculate_spectral_centroid(self, frequencies: np.ndarray, 
                                   power_spectrum: np.ndarray) -> float:
        """
        Calculate spectral centroid (center of mass of spectrum).
        
        Args:
            frequencies: Frequency array
            power_spectrum: Power spectral density
            
        Returns:
            Spectral centroid frequency in Hz
        """
        # Avoid division by zero
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0
        
        # Calculate weighted average frequency
        centroid = np.sum(frequencies * power_spectrum) / total_power
        
        return centroid
    
    def _calculate_spectral_bandwidth(self, frequencies: np.ndarray, 
                                    power_spectrum: np.ndarray,
                                    centroid: float) -> float:
        """
        Calculate spectral bandwidth around the centroid.
        
        Args:
            frequencies: Frequency array
            power_spectrum: Power spectral density
            centroid: Spectral centroid frequency
            
        Returns:
            Spectral bandwidth in Hz
        """
        # Avoid division by zero
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0
        
        # Calculate weighted standard deviation
        variance = np.sum(((frequencies - centroid) ** 2) * power_spectrum) / total_power
        bandwidth = np.sqrt(variance)
        
        return bandwidth
    
    def create_spectrogram(self, wave: WaveSegment) -> Dict[str, np.ndarray]:
        """
        Create time-frequency spectrogram for a wave segment.
        
        Args:
            wave: Wave segment to analyze
            
        Returns:
            Dictionary with 'times', 'frequencies', and 'spectrogram' arrays
        """
        # Calculate spectrogram
        nperseg = min(self.nperseg, len(wave.data) // 4)
        noverlap = nperseg // 2
        
        if nperseg < 64:
            # For short segments, create a simple time-frequency representation
            frequencies = fftfreq(len(wave.data), 1/self.sampling_rate)[:len(wave.data)//2]
            times = np.linspace(0, wave.duration, len(wave.data))
            
            # Create a simple spectrogram by windowing
            window_size = len(wave.data) // 10
            if window_size < 32:
                window_size = len(wave.data)
            
            n_windows = len(wave.data) // window_size
            spectrogram = np.zeros((len(frequencies), n_windows))
            
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = min(start_idx + window_size, len(wave.data))
                window_data = wave.data[start_idx:end_idx]
                
                if len(window_data) > 0:
                    fft_data = fft(window_data)
                    spectrogram[:, i] = np.abs(fft_data[:len(frequencies)])**2
            
            times = np.linspace(0, wave.duration, n_windows)
        else:
            frequencies, times, spectrogram = signal.spectrogram(
                wave.data,
                fs=self.sampling_rate,
                window=self.window_type,
                nperseg=nperseg,
                noverlap=noverlap
            )
        
        return {
            'times': times,
            'frequencies': frequencies,
            'spectrogram': spectrogram
        }
    
    def analyze_frequency_evolution(self, wave: WaveSegment) -> Dict[str, np.ndarray]:
        """
        Analyze how frequency content evolves over time in a wave.
        
        Args:
            wave: Wave segment to analyze
            
        Returns:
            Dictionary with time evolution of frequency characteristics
        """
        spectrogram_data = self.create_spectrogram(wave)
        times = spectrogram_data['times']
        frequencies = spectrogram_data['frequencies']
        spectrogram = spectrogram_data['spectrogram']
        
        # Calculate dominant frequency over time
        dominant_freq_evolution = np.zeros(len(times))
        spectral_centroid_evolution = np.zeros(len(times))
        
        for i in range(len(times)):
            spectrum_slice = spectrogram[:, i]
            
            # Dominant frequency at this time
            peak_idx = np.argmax(spectrum_slice)
            dominant_freq_evolution[i] = frequencies[peak_idx]
            
            # Spectral centroid at this time
            total_power = np.sum(spectrum_slice)
            if total_power > 0:
                spectral_centroid_evolution[i] = np.sum(frequencies * spectrum_slice) / total_power
            else:
                spectral_centroid_evolution[i] = 0
        
        return {
            'times': times,
            'dominant_frequency': dominant_freq_evolution,
            'spectral_centroid': spectral_centroid_evolution
        }
    
    def compare_wave_frequencies(self, wave1: WaveSegment, wave2: WaveSegment) -> Dict[str, float]:
        """
        Compare frequency characteristics between two waves.
        
        Args:
            wave1: First wave segment
            wave2: Second wave segment
            
        Returns:
            Dictionary with comparison metrics
        """
        freq_data1 = self.analyze_single_wave_frequency(wave1)
        freq_data2 = self.analyze_single_wave_frequency(wave2)
        
        # Calculate frequency differences
        dominant_freq_diff = abs(freq_data1.dominant_frequency - freq_data2.dominant_frequency)
        centroid_diff = abs(freq_data1.spectral_centroid - freq_data2.spectral_centroid)
        bandwidth_diff = abs(freq_data1.bandwidth - freq_data2.bandwidth)
        
        # Calculate spectral correlation
        # Interpolate to common frequency grid for comparison
        common_freqs = np.linspace(
            max(freq_data1.frequencies[0], freq_data2.frequencies[0]),
            min(freq_data1.frequencies[-1], freq_data2.frequencies[-1]),
            min(len(freq_data1.frequencies), len(freq_data2.frequencies))
        )
        
        spectrum1_interp = np.interp(common_freqs, freq_data1.frequencies, freq_data1.power_spectrum)
        spectrum2_interp = np.interp(common_freqs, freq_data2.frequencies, freq_data2.power_spectrum)
        
        # Normalize spectra
        spectrum1_norm = spectrum1_interp / (np.sum(spectrum1_interp) + 1e-10)
        spectrum2_norm = spectrum2_interp / (np.sum(spectrum2_interp) + 1e-10)
        
        # Calculate correlation
        spectral_correlation = np.corrcoef(spectrum1_norm, spectrum2_norm)[0, 1]
        if np.isnan(spectral_correlation):
            spectral_correlation = 0.0
        
        return {
            'dominant_frequency_difference': dominant_freq_diff,
            'spectral_centroid_difference': centroid_diff,
            'bandwidth_difference': bandwidth_diff,
            'spectral_correlation': spectral_correlation
        }
    
    def set_parameters(self, **kwargs):
        """
        Set analysis parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'window_type' in kwargs:
            self.window_type = kwargs['window_type']
        if 'nperseg' in kwargs:
            self.nperseg = kwargs['nperseg']
        if 'noverlap' in kwargs:
            self.noverlap = kwargs['noverlap']
        if 'frequency_bands' in kwargs:
            self.frequency_bands.update(kwargs['frequency_bands'])