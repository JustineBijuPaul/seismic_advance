"""
Signal Processing Utilities for Wave Analysis

This module provides comprehensive signal processing capabilities for seismic
wave analysis including filtering, windowing, segmentation, and feature extraction.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings


class FilterType(Enum):
    """Enumeration of available filter types."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


class WindowType(Enum):
    """Enumeration of available window functions."""
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    KAISER = "kaiser"
    TUKEY = "tukey"
    RECTANGULAR = "rectangular"


@dataclass
class FilterParameters:
    """Parameters for digital filtering operations."""
    filter_type: FilterType
    cutoff_freq: Union[float, Tuple[float, float]]  # Single freq or (low, high)
    order: int = 4
    sampling_rate: float = 100.0
    
    def __post_init__(self):
        """Validate filter parameters."""
        if self.order <= 0:
            raise ValueError("Filter order must be positive")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        
        # Validate cutoff frequencies
        nyquist = self.sampling_rate / 2
        if self.filter_type in [FilterType.LOWPASS, FilterType.HIGHPASS]:
            if not isinstance(self.cutoff_freq, (int, float)):
                raise ValueError(f"{self.filter_type.value} requires single cutoff frequency")
            if self.cutoff_freq >= nyquist:
                raise ValueError(f"Cutoff frequency {self.cutoff_freq} must be less than Nyquist frequency {nyquist}")
        else:  # BANDPASS or BANDSTOP
            if not isinstance(self.cutoff_freq, (tuple, list)) or len(self.cutoff_freq) != 2:
                raise ValueError(f"{self.filter_type.value} requires two cutoff frequencies")
            low, high = self.cutoff_freq
            if low >= high:
                raise ValueError("Low cutoff must be less than high cutoff")
            if high >= nyquist:
                raise ValueError(f"High cutoff frequency {high} must be less than Nyquist frequency {nyquist}")


@dataclass
class WindowParameters:
    """Parameters for windowing operations."""
    window_type: WindowType
    window_length: int
    overlap: float = 0.5  # Overlap fraction (0-1)
    beta: float = 8.6  # Kaiser window parameter
    alpha: float = 0.25  # Tukey window parameter
    
    def __post_init__(self):
        """Validate window parameters."""
        if self.window_length <= 0:
            raise ValueError("Window length must be positive")
        if not 0 <= self.overlap < 1:
            raise ValueError("Overlap must be between 0 and 1 (exclusive)")
        if self.beta <= 0:
            raise ValueError("Kaiser beta parameter must be positive")
        if not 0 <= self.alpha <= 1:
            raise ValueError("Tukey alpha parameter must be between 0 and 1")


class FilterBank:
    """
    Digital filter bank for seismic signal processing.
    
    Provides various filtering operations optimized for earthquake
    wave analysis with different frequency bands.
    """
    
    # Standard frequency bands for seismic analysis
    FREQUENCY_BANDS = {
        'teleseismic': (0.02, 2.0),      # Long-period teleseismic waves
        'regional': (0.5, 10.0),         # Regional earthquake waves  
        'local': (1.0, 30.0),            # Local earthquake waves
        'high_freq': (10.0, 50.0),       # High-frequency analysis
        'p_wave': (1.0, 15.0),           # Typical P-wave band
        's_wave': (0.5, 10.0),           # Typical S-wave band
        'surface_wave': (0.02, 0.5),     # Surface wave band
        'microseismic': (0.1, 1.0),      # Microseismic noise band
    }
    
    def __init__(self):
        """Initialize the filter bank."""
        self._filter_cache = {}  # Cache for computed filter coefficients
    
    def apply_filter(self, data: np.ndarray, params: FilterParameters) -> np.ndarray:
        """
        Apply digital filter to seismic data.
        
        Args:
            data: Input seismic time series
            params: Filter parameters
            
        Returns:
            Filtered seismic data
            
        Raises:
            ValueError: If input data is invalid
        """
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        
        # Create cache key for filter coefficients
        cache_key = (params.filter_type, params.cutoff_freq, params.order, params.sampling_rate)
        
        # Get or compute filter coefficients
        if cache_key not in self._filter_cache:
            self._filter_cache[cache_key] = self._design_filter(params)
        
        b, a = self._filter_cache[cache_key]
        
        # Apply filter with zero-phase filtering to avoid phase distortion
        try:
            filtered_data = signal.filtfilt(b, a, data)
        except ValueError as e:
            # Handle edge cases like data too short for filter
            warnings.warn(f"Filter application failed: {e}. Returning original data.")
            return data.copy()
        
        return filtered_data
    
    def _design_filter(self, params: FilterParameters) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design digital filter coefficients.
        
        Args:
            params: Filter parameters
            
        Returns:
            Tuple of (numerator, denominator) coefficients
        """
        nyquist = params.sampling_rate / 2
        
        if params.filter_type == FilterType.LOWPASS:
            b, a = signal.butter(params.order, params.cutoff_freq / nyquist, btype='low')
        elif params.filter_type == FilterType.HIGHPASS:
            b, a = signal.butter(params.order, params.cutoff_freq / nyquist, btype='high')
        elif params.filter_type == FilterType.BANDPASS:
            low, high = params.cutoff_freq
            b, a = signal.butter(params.order, [low / nyquist, high / nyquist], btype='band')
        elif params.filter_type == FilterType.BANDSTOP:
            low, high = params.cutoff_freq
            b, a = signal.butter(params.order, [low / nyquist, high / nyquist], btype='bandstop')
        else:
            raise ValueError(f"Unsupported filter type: {params.filter_type}")
        
        return b, a
    
    def apply_frequency_band(self, data: np.ndarray, band_name: str, 
                           sampling_rate: float, order: int = 4) -> np.ndarray:
        """
        Apply predefined frequency band filter.
        
        Args:
            data: Input seismic data
            band_name: Name of frequency band from FREQUENCY_BANDS
            sampling_rate: Sampling rate in Hz
            order: Filter order
            
        Returns:
            Filtered data for specified frequency band
            
        Raises:
            ValueError: If band_name is not recognized
        """
        if band_name not in self.FREQUENCY_BANDS:
            raise ValueError(f"Unknown frequency band: {band_name}. "
                           f"Available bands: {list(self.FREQUENCY_BANDS.keys())}")
        
        low_freq, high_freq = self.FREQUENCY_BANDS[band_name]
        params = FilterParameters(
            filter_type=FilterType.BANDPASS,
            cutoff_freq=(low_freq, high_freq),
            order=order,
            sampling_rate=sampling_rate
        )
        
        return self.apply_filter(data, params)
    
    def get_available_bands(self) -> Dict[str, Tuple[float, float]]:
        """Get dictionary of available frequency bands."""
        return self.FREQUENCY_BANDS.copy()


class WindowFunction:
    """
    Windowing functions for time series segmentation and analysis.
    
    Provides various window functions optimized for seismic signal
    processing and spectral analysis.
    """
    
    @staticmethod
    def create_window(params: WindowParameters) -> np.ndarray:
        """
        Create window function array.
        
        Args:
            params: Window parameters
            
        Returns:
            Window function array
            
        Raises:
            ValueError: If window type is not supported
        """
        if params.window_type == WindowType.HANN:
            return signal.windows.hann(params.window_length)
        elif params.window_type == WindowType.HAMMING:
            return signal.windows.hamming(params.window_length)
        elif params.window_type == WindowType.BLACKMAN:
            return signal.windows.blackman(params.window_length)
        elif params.window_type == WindowType.KAISER:
            return signal.windows.kaiser(params.window_length, params.beta)
        elif params.window_type == WindowType.TUKEY:
            return signal.windows.tukey(params.window_length, params.alpha)
        elif params.window_type == WindowType.RECTANGULAR:
            return np.ones(params.window_length)
        else:
            raise ValueError(f"Unsupported window type: {params.window_type}")
    
    @staticmethod
    def segment_data(data: np.ndarray, params: WindowParameters) -> List[np.ndarray]:
        """
        Segment time series data using overlapping windows.
        
        Args:
            data: Input time series data
            params: Window parameters
            
        Returns:
            List of windowed data segments
            
        Raises:
            ValueError: If data is too short for windowing
        """
        if len(data) < params.window_length:
            raise ValueError(f"Data length {len(data)} is shorter than window length {params.window_length}")
        
        # Calculate step size based on overlap
        step_size = int(params.window_length * (1 - params.overlap))
        if step_size <= 0:
            step_size = 1
        
        # Create window function
        window = WindowFunction.create_window(params)
        
        # Generate segments
        segments = []
        start = 0
        while start + params.window_length <= len(data):
            segment = data[start:start + params.window_length] * window
            segments.append(segment)
            start += step_size
        
        return segments
    
    @staticmethod
    def apply_window(data: np.ndarray, window_type: WindowType, **kwargs) -> np.ndarray:
        """
        Apply window function to data array.
        
        Args:
            data: Input data array
            window_type: Type of window to apply
            **kwargs: Additional window parameters
            
        Returns:
            Windowed data array
        """
        params = WindowParameters(
            window_type=window_type,
            window_length=len(data),
            **kwargs
        )
        window = WindowFunction.create_window(params)
        return data * window


class FeatureExtractor:
    """
    Feature extraction utilities for seismic wave analysis.
    
    Provides comprehensive feature extraction capabilities including
    time-domain, frequency-domain, and time-frequency features.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
    
    def extract_time_domain_features(self, data: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain features from seismic data.
        
        Args:
            data: Input seismic time series
            
        Returns:
            Dictionary of time-domain features
        """
        if len(data) == 0:
            return {}
        
        features = {
            'mean': np.mean(data),
            'std': np.std(data),
            'variance': np.var(data),
            'rms': np.sqrt(np.mean(data**2)),
            'peak_amplitude': np.max(np.abs(data)),
            'peak_to_peak': np.ptp(data),
            'skewness': self._calculate_skewness(data),
            'kurtosis': self._calculate_kurtosis(data),
            'zero_crossing_rate': self._calculate_zero_crossing_rate(data),
            'energy': np.sum(data**2),
            'signal_length': len(data),
            'duration': len(data) / self.sampling_rate
        }
        
        return features
    
    def extract_frequency_domain_features(self, data: np.ndarray, 
                                        window_type: WindowType = WindowType.HANN) -> Dict[str, Any]:
        """
        Extract frequency-domain features from seismic data.
        
        Args:
            data: Input seismic time series
            window_type: Window function for spectral analysis
            
        Returns:
            Dictionary of frequency-domain features
        """
        if len(data) == 0:
            return {}
        
        # Apply window function
        windowed_data = WindowFunction.apply_window(data, window_type)
        
        # Compute FFT
        fft_data = fft(windowed_data)
        frequencies = fftfreq(len(data), 1/self.sampling_rate)
        
        # Use only positive frequencies
        positive_freq_mask = frequencies >= 0
        frequencies = frequencies[positive_freq_mask]
        power_spectrum = np.abs(fft_data[positive_freq_mask])**2
        
        # Normalize power spectrum
        power_spectrum = power_spectrum / np.sum(power_spectrum)
        
        features = {
            'frequencies': frequencies,
            'power_spectrum': power_spectrum,
            'dominant_frequency': frequencies[np.argmax(power_spectrum)],
            'spectral_centroid': np.sum(frequencies * power_spectrum),
            'spectral_bandwidth': self._calculate_spectral_bandwidth(frequencies, power_spectrum),
            'spectral_rolloff': self._calculate_spectral_rolloff(frequencies, power_spectrum),
            'spectral_flux': self._calculate_spectral_flux(power_spectrum),
            'total_power': np.sum(power_spectrum),
            'frequency_range': (frequencies[0], frequencies[-1])
        }
        
        return features
    
    def extract_spectral_features_by_band(self, data: np.ndarray, 
                                        filter_bank: FilterBank) -> Dict[str, Dict[str, Any]]:
        """
        Extract spectral features for different frequency bands.
        
        Args:
            data: Input seismic data
            filter_bank: FilterBank instance for band filtering
            
        Returns:
            Dictionary mapping band names to their spectral features
        """
        band_features = {}
        
        for band_name in filter_bank.get_available_bands():
            try:
                # Filter data for this band
                filtered_data = filter_bank.apply_frequency_band(
                    data, band_name, self.sampling_rate
                )
                
                # Extract features for filtered data
                time_features = self.extract_time_domain_features(filtered_data)
                freq_features = self.extract_frequency_domain_features(filtered_data)
                
                # Combine features
                band_features[band_name] = {
                    'time_domain': time_features,
                    'frequency_domain': freq_features
                }
                
            except Exception as e:
                warnings.warn(f"Feature extraction failed for band {band_name}: {e}")
                band_features[band_name] = {}
        
        return band_features
    
    def calculate_sta_lta_ratio(self, data: np.ndarray, sta_window: float, 
                              lta_window: float) -> np.ndarray:
        """
        Calculate Short-Term Average / Long-Term Average ratio.
        
        This is a fundamental algorithm for earthquake detection and
        P-wave picking in seismology.
        
        Args:
            data: Input seismic data
            sta_window: Short-term average window length in seconds
            lta_window: Long-term average window length in seconds
            
        Returns:
            STA/LTA ratio time series
            
        Raises:
            ValueError: If window parameters are invalid
        """
        if sta_window >= lta_window:
            raise ValueError("STA window must be shorter than LTA window")
        if lta_window * self.sampling_rate >= len(data):
            raise ValueError("LTA window is too long for data length")
        
        # Convert time windows to sample counts
        sta_samples = int(sta_window * self.sampling_rate)
        lta_samples = int(lta_window * self.sampling_rate)
        
        # Calculate squared data for energy computation
        data_squared = data**2
        
        # Initialize output array
        sta_lta = np.zeros(len(data))
        
        # Calculate STA/LTA ratio for each sample
        for i in range(lta_samples, len(data)):
            # Long-term average (energy)
            lta = np.mean(data_squared[i-lta_samples:i])
            
            # Short-term average (energy)
            sta_start = max(0, i-sta_samples)
            sta = np.mean(data_squared[sta_start:i])
            
            # Calculate ratio, avoiding division by zero
            if lta > 0:
                sta_lta[i] = sta / lta
            else:
                sta_lta[i] = 0
        
        return sta_lta
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std)**3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data distribution."""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std)**4) - 3  # Excess kurtosis
    
    def _calculate_zero_crossing_rate(self, data: np.ndarray) -> float:
        """Calculate zero crossing rate."""
        if len(data) < 2:
            return 0.0
        
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        return zero_crossings / (len(data) - 1)
    
    def _calculate_spectral_bandwidth(self, frequencies: np.ndarray, 
                                    power_spectrum: np.ndarray) -> float:
        """Calculate spectral bandwidth."""
        centroid = np.sum(frequencies * power_spectrum)
        return np.sqrt(np.sum(((frequencies - centroid)**2) * power_spectrum))
    
    def _calculate_spectral_rolloff(self, frequencies: np.ndarray, 
                                  power_spectrum: np.ndarray, 
                                  rolloff_threshold: float = 0.85) -> float:
        """Calculate spectral rolloff frequency."""
        cumulative_power = np.cumsum(power_spectrum)
        total_power = cumulative_power[-1]
        rolloff_index = np.where(cumulative_power >= rolloff_threshold * total_power)[0]
        
        if len(rolloff_index) > 0:
            return frequencies[rolloff_index[0]]
        else:
            return frequencies[-1]
    
    def _calculate_spectral_flux(self, power_spectrum: np.ndarray) -> float:
        """Calculate spectral flux (measure of spectral change)."""
        if len(power_spectrum) < 2:
            return 0.0
        
        # For single spectrum, return normalized variance as flux measure
        return np.var(power_spectrum) / (np.mean(power_spectrum) + 1e-10)


class SignalProcessor:
    """
    Main signal processing coordinator class.
    
    Provides high-level interface for all signal processing operations
    needed for seismic wave analysis.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize signal processor.
        
        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.filter_bank = FilterBank()
        self.feature_extractor = FeatureExtractor(sampling_rate)
    
    def preprocess_seismic_data(self, data: np.ndarray, 
                              remove_mean: bool = True,
                              detrend: bool = True,
                              taper: bool = True,
                              taper_fraction: float = 0.05) -> np.ndarray:
        """
        Apply standard preprocessing to seismic data.
        
        Args:
            data: Raw seismic time series
            remove_mean: Whether to remove DC component
            detrend: Whether to remove linear trend
            taper: Whether to apply cosine taper
            taper_fraction: Fraction of data to taper at each end
            
        Returns:
            Preprocessed seismic data
        """
        processed_data = data.copy()
        
        # Remove mean (DC component)
        if remove_mean:
            processed_data = processed_data - np.mean(processed_data)
        
        # Remove linear trend
        if detrend:
            processed_data = signal.detrend(processed_data, type='linear')
        
        # Apply cosine taper to reduce edge effects
        if taper and len(processed_data) > 10:
            taper_samples = int(len(processed_data) * taper_fraction)
            if taper_samples > 0:
                taper_window = signal.windows.tukey(len(processed_data), 
                                                  alpha=2*taper_fraction)
                processed_data = processed_data * taper_window
        
        return processed_data
    
    def segment_for_analysis(self, data: np.ndarray, 
                           segment_length: float,
                           overlap: float = 0.5,
                           window_type: WindowType = WindowType.HANN) -> List[np.ndarray]:
        """
        Segment data for analysis with specified parameters.
        
        Args:
            data: Input seismic data
            segment_length: Length of each segment in seconds
            overlap: Overlap fraction between segments
            window_type: Window function to apply
            
        Returns:
            List of segmented and windowed data arrays
        """
        window_samples = int(segment_length * self.sampling_rate)
        
        params = WindowParameters(
            window_type=window_type,
            window_length=window_samples,
            overlap=overlap
        )
        
        return WindowFunction.segment_data(data, params)
    
    def extract_comprehensive_features(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive feature set for wave analysis.
        
        Args:
            data: Input seismic data
            
        Returns:
            Dictionary containing all extracted features
        """
        # Preprocess data
        processed_data = self.preprocess_seismic_data(data)
        
        # Extract basic features
        time_features = self.feature_extractor.extract_time_domain_features(processed_data)
        freq_features = self.feature_extractor.extract_frequency_domain_features(processed_data)
        
        # Extract band-specific features
        band_features = self.feature_extractor.extract_spectral_features_by_band(
            processed_data, self.filter_bank
        )
        
        # Calculate STA/LTA for P-wave detection (adjust windows for data length)
        data_duration = len(processed_data) / self.sampling_rate
        lta_window = min(10.0, data_duration * 0.3)  # Use 30% of data for LTA
        sta_window = min(1.0, lta_window * 0.1)      # STA is 10% of LTA
        
        if lta_window > sta_window and lta_window * self.sampling_rate < len(processed_data):
            sta_lta = self.feature_extractor.calculate_sta_lta_ratio(
                processed_data, sta_window=sta_window, lta_window=lta_window
            )
        else:
            # Data too short for meaningful STA/LTA
            sta_lta = np.zeros(len(processed_data))
        
        return {
            'time_domain': time_features,
            'frequency_domain': freq_features,
            'band_features': band_features,
            'sta_lta_ratio': sta_lta,
            'processing_metadata': {
                'sampling_rate': self.sampling_rate,
                'data_length': len(data),
                'processed_length': len(processed_data)
            }
        }