"""
Wave Detection Algorithms

This module implements specialized algorithms for detecting different types
of seismic waves including P-waves, S-waves, and surface waves.
"""

import numpy as np
from scipy import signal
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

from ..interfaces import WaveDetectorInterface
from ..models import WaveSegment
from .signal_processing import (
    SignalProcessor, FilterBank, FilterParameters, FilterType,
    FeatureExtractor, WindowFunction, WindowType, WindowParameters
)


@dataclass
class DetectionParameters:
    """Base parameters for wave detection algorithms."""
    sampling_rate: float
    min_wave_duration: float = 0.5  # Minimum wave duration in seconds
    max_wave_duration: float = 30.0  # Maximum wave duration in seconds
    confidence_threshold: float = 0.5  # Minimum confidence for detection
    
    def __post_init__(self):
        """Validate detection parameters."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if self.min_wave_duration >= self.max_wave_duration:
            raise ValueError("Min duration must be less than max duration")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")


@dataclass
class PWaveDetectionParameters(DetectionParameters):
    """Parameters specific to P-wave detection."""
    sta_window: float = 1.0  # Short-term average window in seconds
    lta_window: float = 10.0  # Long-term average window in seconds
    trigger_threshold: float = 3.0  # STA/LTA trigger threshold
    detrigger_threshold: float = 1.5  # STA/LTA detrigger threshold
    frequency_band: Tuple[float, float] = (1.0, 15.0)  # P-wave frequency band
    characteristic_function_type: str = 'energy'  # 'energy', 'kurtosis', or 'aic'
    
    def __post_init__(self):
        """Validate P-wave detection parameters."""
        super().__post_init__()
        if self.sta_window >= self.lta_window:
            raise ValueError("STA window must be shorter than LTA window")
        if self.trigger_threshold <= self.detrigger_threshold:
            raise ValueError("Trigger threshold must be greater than detrigger threshold")
        if self.characteristic_function_type not in ['energy', 'kurtosis', 'aic']:
            raise ValueError("Invalid characteristic function type")


class PWaveDetector(WaveDetectorInterface):
    """
    P-wave detection algorithm using STA/LTA and characteristic functions.
    
    This detector implements the classic Short-Term Average / Long-Term Average
    algorithm combined with characteristic functions for robust P-wave onset detection.
    """
    
    def __init__(self, parameters: PWaveDetectionParameters):
        """Initialize P-wave detector."""
        self.params = parameters
        self.signal_processor = SignalProcessor(parameters.sampling_rate)
        self.filter_bank = FilterBank()
        self.feature_extractor = FeatureExtractor(parameters.sampling_rate)
        
    def detect_waves(self, data: np.ndarray, sampling_rate: float, 
                    metadata: Optional[Dict[str, Any]] = None) -> List[WaveSegment]:
        """Detect P-waves in seismic data."""
        if len(data) == 0:
            return []
        
        # Update sampling rate if different from initialization
        if sampling_rate != self.params.sampling_rate:
            self.params.sampling_rate = sampling_rate
            self.signal_processor = SignalProcessor(sampling_rate)
            self.feature_extractor = FeatureExtractor(sampling_rate)
        
        # Preprocess the data
        processed_data = self.signal_processor.preprocess_seismic_data(data)
        
        # Apply P-wave frequency band filter
        filtered_data = self._apply_p_wave_filter(processed_data)
        
        # Calculate characteristic function
        char_function = self._calculate_characteristic_function(filtered_data)
        
        # Calculate STA/LTA ratio
        sta_lta = self._calculate_sta_lta(char_function)
        
        # Detect P-wave onsets using trigger/detrigger logic
        onset_times = self._detect_onsets(sta_lta)
        
        # Create wave segments from detected onsets
        wave_segments = self._create_p_wave_segments(
            processed_data, filtered_data, onset_times, sta_lta
        )
        
        return wave_segments
    
    def get_wave_type(self) -> str:
        """Get the type of waves this detector identifies."""
        return 'P'
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set detection parameters."""
        for key, value in parameters.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}")
    
    def _apply_p_wave_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply P-wave frequency band filter."""
        filter_params = FilterParameters(
            filter_type=FilterType.BANDPASS,
            cutoff_freq=self.params.frequency_band,
            order=4,
            sampling_rate=self.params.sampling_rate
        )
        return self.filter_bank.apply_filter(data, filter_params)
    
    def _calculate_characteristic_function(self, data: np.ndarray) -> np.ndarray:
        """Calculate characteristic function for P-wave detection."""
        if self.params.characteristic_function_type == 'energy':
            return self._energy_characteristic_function(data)
        elif self.params.characteristic_function_type == 'kurtosis':
            return self._kurtosis_characteristic_function(data)
        elif self.params.characteristic_function_type == 'aic':
            return self._aic_characteristic_function(data)
        else:
            raise ValueError(f"Unknown characteristic function: {self.params.characteristic_function_type}")
    
    def _energy_characteristic_function(self, data: np.ndarray) -> np.ndarray:
        """Calculate energy-based characteristic function."""
        return data ** 2
    
    def _kurtosis_characteristic_function(self, data: np.ndarray, 
                                       window_length: int = None) -> np.ndarray:
        """Calculate kurtosis-based characteristic function."""
        if window_length is None:
            window_length = int(0.5 * self.params.sampling_rate)
        
        char_function = np.zeros(len(data))
        
        for i in range(window_length, len(data)):
            window_data = data[i-window_length:i]
            if len(window_data) > 3:
                mean = np.mean(window_data)
                std = np.std(window_data)
                if std > 0:
                    normalized = (window_data - mean) / std
                    char_function[i] = np.mean(normalized ** 4) - 3
        
        return np.abs(char_function)
    
    def _aic_characteristic_function(self, data: np.ndarray, 
                                   window_length: int = None) -> np.ndarray:
        """Calculate AIC characteristic function."""
        if window_length is None:
            window_length = int(1.0 * self.params.sampling_rate)
        
        char_function = np.zeros(len(data))
        
        for i in range(window_length, len(data) - window_length):
            left_window = data[i-window_length:i]
            right_window = data[i:i+window_length]
            
            var_left = np.var(left_window) if len(left_window) > 1 else 1e-10
            var_right = np.var(right_window) if len(right_window) > 1 else 1e-10
            var_total = np.var(data[i-window_length:i+window_length])
            
            if var_total > 0:
                aic = len(left_window) * np.log(var_left) + len(right_window) * np.log(var_right)
                aic_total = (len(left_window) + len(right_window)) * np.log(var_total)
                char_function[i] = aic_total - aic
        
        return char_function
    
    def _calculate_sta_lta(self, char_function: np.ndarray) -> np.ndarray:
        """Calculate STA/LTA ratio for the characteristic function."""
        return self.feature_extractor.calculate_sta_lta_ratio(
            char_function, 
            self.params.sta_window, 
            self.params.lta_window
        )
    
    def _detect_onsets(self, sta_lta: np.ndarray) -> List[int]:
        """Detect P-wave onsets using trigger/detrigger logic."""
        onsets = []
        triggered = False
        trigger_start = 0
        
        for i, ratio in enumerate(sta_lta):
            if not triggered and ratio > self.params.trigger_threshold:
                triggered = True
                trigger_start = i
            elif triggered and ratio < self.params.detrigger_threshold:
                if i > trigger_start:
                    trigger_segment = sta_lta[trigger_start:i]
                    max_idx = np.argmax(trigger_segment)
                    onset_idx = trigger_start + max_idx
                    onsets.append(onset_idx)
                
                triggered = False
        
        return onsets
    
    def _create_p_wave_segments(self, original_data: np.ndarray, 
                              filtered_data: np.ndarray,
                              onset_times: List[int], 
                              sta_lta: np.ndarray) -> List[WaveSegment]:
        """Create WaveSegment objects from detected onsets."""
        wave_segments = []
        
        for onset_idx in onset_times:
            # Simple boundary determination
            min_samples = int(self.params.min_wave_duration * self.params.sampling_rate)
            max_samples = int(self.params.max_wave_duration * self.params.sampling_rate)
            
            start_idx = max(0, onset_idx - min_samples // 2)
            end_idx = min(len(original_data), onset_idx + max_samples)
            
            if end_idx <= start_idx:
                continue
            
            wave_data = original_data[start_idx:end_idx]
            peak_amplitude = np.max(np.abs(wave_data))
            
            freq_features = self.feature_extractor.extract_frequency_domain_features(wave_data)
            dominant_frequency = freq_features.get('dominant_frequency', 0.0)
            
            confidence = min(1.0, sta_lta[onset_idx] / (self.params.trigger_threshold * 2))
            
            if confidence >= self.params.confidence_threshold:
                wave_segment = WaveSegment(
                    wave_type='P',
                    start_time=start_idx / self.params.sampling_rate,
                    end_time=end_idx / self.params.sampling_rate,
                    data=wave_data,
                    sampling_rate=self.params.sampling_rate,
                    peak_amplitude=peak_amplitude,
                    dominant_frequency=dominant_frequency,
                    arrival_time=onset_idx / self.params.sampling_rate,
                    confidence=confidence,
                    metadata={
                        'sta_lta_peak': sta_lta[onset_idx],
                        'detection_method': 'STA/LTA',
                        'characteristic_function': self.params.characteristic_function_type,
                        'frequency_band': self.params.frequency_band
                    }
                )
                wave_segments.append(wave_segment)
        
        return wave_segments
    
    def get_detection_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """Get detection statistics and diagnostic information."""
        processed_data = self.signal_processor.preprocess_seismic_data(data)
        filtered_data = self._apply_p_wave_filter(processed_data)
        
        char_function = self._calculate_characteristic_function(filtered_data)
        sta_lta = self._calculate_sta_lta(char_function)
        
        onsets = self._detect_onsets(sta_lta)
        
        return {
            'num_detections': len(onsets),
            'max_sta_lta': np.max(sta_lta) if len(sta_lta) > 0 else 0,
            'mean_sta_lta': np.mean(sta_lta) if len(sta_lta) > 0 else 0,
            'trigger_threshold': self.params.trigger_threshold,
            'detrigger_threshold': self.params.detrigger_threshold,
            'onset_times': [onset / self.params.sampling_rate for onset in onsets],
            'characteristic_function_type': self.params.characteristic_function_type,
            'frequency_band': self.params.frequency_band,
            'data_duration': len(data) / self.params.sampling_rate
        }


@dataclass
class SWaveDetectionParameters(DetectionParameters):
    """Parameters specific to S-wave detection."""
    sta_window: float = 2.0  # Short-term average window in seconds
    lta_window: float = 15.0  # Long-term average window in seconds
    trigger_threshold: float = 2.5  # STA/LTA trigger threshold
    detrigger_threshold: float = 1.3  # STA/LTA detrigger threshold
    frequency_band: Tuple[float, float] = (0.5, 10.0)  # S-wave frequency band
    polarization_window: float = 1.0  # Window for polarization analysis in seconds
    amplitude_ratio_threshold: float = 1.5  # S/P amplitude ratio threshold
    p_wave_context_window: float = 5.0  # Window to look for P-wave before S-wave
    
    def __post_init__(self):
        """Validate S-wave detection parameters."""
        super().__post_init__()
        if self.sta_window >= self.lta_window:
            raise ValueError("STA window must be shorter than LTA window")
        if self.trigger_threshold <= self.detrigger_threshold:
            raise ValueError("Trigger threshold must be greater than detrigger threshold")
        if self.polarization_window <= 0:
            raise ValueError("Polarization window must be positive")
        if self.amplitude_ratio_threshold <= 0:
            raise ValueError("Amplitude ratio threshold must be positive")


class SWaveDetector(WaveDetectorInterface):
    """
    S-wave detection algorithm using polarization analysis and particle motion.
    
    This detector implements S-wave detection using:
    1. Polarization analysis to identify shear wave motion
    2. Particle motion analysis for S-wave characteristics
    3. Amplitude ratio calculations relative to P-waves
    4. STA/LTA triggering adapted for S-wave characteristics
    """
    
    def __init__(self, parameters: SWaveDetectionParameters):
        """
        Initialize S-wave detector.
        
        Args:
            parameters: S-wave detection parameters
        """
        self.params = parameters
        self.signal_processor = SignalProcessor(parameters.sampling_rate)
        self.filter_bank = FilterBank()
        self.feature_extractor = FeatureExtractor(parameters.sampling_rate)
        
    def detect_waves(self, data: np.ndarray, sampling_rate: float, 
                    metadata: Optional[Dict[str, Any]] = None) -> List[WaveSegment]:
        """
        Detect S-waves in seismic data.
        
        Args:
            data: Raw seismic time series data (can be multi-channel)
            sampling_rate: Sampling rate in Hz
            metadata: Optional metadata (may include P-wave detections)
            
        Returns:
            List of detected S-wave segments
        """
        if len(data) == 0:
            return []
        
        # Update sampling rate if different from initialization
        if sampling_rate != self.params.sampling_rate:
            self.params.sampling_rate = sampling_rate
            self.signal_processor = SignalProcessor(sampling_rate)
            self.feature_extractor = FeatureExtractor(sampling_rate)
        
        # Handle multi-channel data (assume 3-component if 2D)
        if data.ndim == 1:
            # Single channel - use modified approach
            return self._detect_single_channel(data, metadata)
        else:
            # Multi-channel - use full polarization analysis
            return self._detect_multi_channel(data, metadata)
    
    def get_wave_type(self) -> str:
        """Get the type of waves this detector identifies."""
        return 'S'
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set detection parameters.
        
        Args:
            parameters: Dictionary of parameter name-value pairs
        """
        for key, value in parameters.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}") 
   
    def _detect_single_channel(self, data: np.ndarray, 
                             metadata: Optional[Dict[str, Any]] = None) -> List[WaveSegment]:
        """
        Detect S-waves in single-channel data.
        
        Uses amplitude-based detection with P-wave context.
        """
        # Preprocess the data
        processed_data = self.signal_processor.preprocess_seismic_data(data)
        
        # Apply S-wave frequency band filter
        filtered_data = self._apply_s_wave_filter(processed_data)
        
        # Calculate characteristic function (energy-based for single channel)
        char_function = filtered_data ** 2
        
        # Calculate STA/LTA ratio
        sta_lta = self._calculate_sta_lta(char_function)
        
        # Get P-wave context if available
        p_wave_times = self._extract_p_wave_times(metadata)
        
        # Detect S-wave onsets using trigger/detrigger logic with P-wave context
        onset_times = self._detect_onsets_with_context(sta_lta, p_wave_times)
        
        # Create wave segments from detected onsets
        wave_segments = self._create_wave_segments(
            processed_data, filtered_data, onset_times, sta_lta, 'single_channel'
        )
        
        return wave_segments
    
    def _detect_multi_channel(self, data: np.ndarray, 
                            metadata: Optional[Dict[str, Any]] = None) -> List[WaveSegment]:
        """
        Detect S-waves in multi-channel data using polarization analysis.
        
        Args:
            data: Multi-channel seismic data (channels x samples)
            metadata: Optional metadata
            
        Returns:
            List of detected S-wave segments
        """
        if data.shape[0] < 2:
            # Not enough channels for polarization analysis
            return self._detect_single_channel(data[0], metadata)
        
        # Preprocess each channel
        processed_channels = []
        filtered_channels = []
        
        for i in range(data.shape[0]):
            processed = self.signal_processor.preprocess_seismic_data(data[i])
            filtered = self._apply_s_wave_filter(processed)
            processed_channels.append(processed)
            filtered_channels.append(filtered)
        
        processed_data = np.array(processed_channels)
        filtered_data = np.array(filtered_channels)
        
        # Calculate polarization attributes
        polarization_attrs = self._calculate_polarization_attributes(filtered_data)
        
        # Calculate particle motion characteristics
        particle_motion = self._calculate_particle_motion(filtered_data)
        
        # Combine polarization and particle motion for characteristic function
        char_function = self._combine_s_wave_characteristics(
            polarization_attrs, particle_motion
        )
        
        # Calculate STA/LTA ratio
        sta_lta = self._calculate_sta_lta(char_function)
        
        # Get P-wave context if available
        p_wave_times = self._extract_p_wave_times(metadata)
        
        # Detect S-wave onsets
        onset_times = self._detect_onsets_with_context(sta_lta, p_wave_times)
        
        # Create wave segments (use first channel as representative)
        wave_segments = self._create_wave_segments(
            processed_data[0], filtered_data[0], onset_times, sta_lta, 'multi_channel',
            additional_data={'polarization': polarization_attrs, 'particle_motion': particle_motion}
        )
        
        return wave_segments
    
    def _apply_s_wave_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply S-wave frequency band filter."""
        filter_params = FilterParameters(
            filter_type=FilterType.BANDPASS,
            cutoff_freq=self.params.frequency_band,
            order=4,
            sampling_rate=self.params.sampling_rate
        )
        return self.filter_bank.apply_filter(data, filter_params)
    
    def _calculate_sta_lta(self, char_function: np.ndarray) -> np.ndarray:
        """Calculate STA/LTA ratio for the characteristic function."""
        try:
            return self.feature_extractor.calculate_sta_lta_ratio(
                char_function, 
                self.params.sta_window, 
                self.params.lta_window
            )
        except ValueError:
            # Handle case where data is too short
            return np.zeros(len(char_function))
    
    def _calculate_polarization_attributes(self, multi_channel_data: np.ndarray) -> np.ndarray:
        """
        Calculate polarization attributes for multi-channel data.
        
        Args:
            multi_channel_data: Array of shape (channels, samples)
            
        Returns:
            Polarization attribute time series
        """
        n_channels, n_samples = multi_channel_data.shape
        window_samples = int(self.params.polarization_window * self.params.sampling_rate)
        
        # Initialize polarization attributes
        polarization = np.zeros(n_samples)
        
        # Calculate covariance matrix in sliding windows
        for i in range(window_samples, n_samples):
            start_idx = i - window_samples
            window_data = multi_channel_data[:, start_idx:i]
            
            # Calculate covariance matrix
            cov_matrix = np.cov(window_data)
            
            # Calculate eigenvalues
            eigenvals = np.linalg.eigvals(cov_matrix)
            eigenvals = np.sort(eigenvals)[::-1]  # Sort in descending order
            
            # Polarization measure: ratio of largest to smallest eigenvalue
            if len(eigenvals) >= 2 and eigenvals[-1] > 1e-10:
                polarization[i] = eigenvals[0] / eigenvals[-1]
            else:
                polarization[i] = 1.0
        
        return polarization
    
    def _calculate_particle_motion(self, multi_channel_data: np.ndarray) -> np.ndarray:
        """
        Calculate particle motion characteristics.
        
        S-waves typically show more complex particle motion compared to P-waves.
        """
        n_channels, n_samples = multi_channel_data.shape
        window_samples = int(self.params.polarization_window * self.params.sampling_rate)
        
        particle_motion = np.zeros(n_samples)
        
        for i in range(window_samples, n_samples):
            start_idx = i - window_samples
            window_data = multi_channel_data[:, start_idx:i]
            
            if n_channels >= 2:
                # Calculate the complexity of particle motion
                # Use the variance of the angle between consecutive motion vectors
                if window_data.shape[1] > 1:
                    # Calculate motion vectors
                    motion_vectors = np.diff(window_data, axis=1)
                    
                    if motion_vectors.shape[1] > 1:
                        # Calculate angles between consecutive vectors
                        angles = []
                        for j in range(motion_vectors.shape[1] - 1):
                            v1 = motion_vectors[:, j]
                            v2 = motion_vectors[:, j + 1]
                            
                            # Calculate angle between vectors
                            dot_product = np.dot(v1, v2)
                            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
                            
                            if norms > 1e-10:
                                cos_angle = np.clip(dot_product / norms, -1, 1)
                                angle = np.arccos(cos_angle)
                                angles.append(angle)
                        
                        if angles:
                            # Particle motion complexity: variance of angles
                            particle_motion[i] = np.var(angles)
        
        return particle_motion
    
    def _combine_s_wave_characteristics(self, polarization: np.ndarray, 
                                     particle_motion: np.ndarray) -> np.ndarray:
        """
        Combine polarization and particle motion into S-wave characteristic function.
        
        Args:
            polarization: Polarization attribute time series
            particle_motion: Particle motion complexity time series
            
        Returns:
            Combined characteristic function for S-wave detection
        """
        # Normalize both attributes to [0, 1] range
        if np.max(polarization) > np.min(polarization):
            norm_polarization = (polarization - np.min(polarization)) / (np.max(polarization) - np.min(polarization))
        else:
            norm_polarization = np.zeros_like(polarization)
        
        if np.max(particle_motion) > np.min(particle_motion):
            norm_particle_motion = (particle_motion - np.min(particle_motion)) / (np.max(particle_motion) - np.min(particle_motion))
        else:
            norm_particle_motion = np.zeros_like(particle_motion)
        
        # Combine with weights (polarization is more important for S-wave detection)
        combined = 0.7 * norm_polarization + 0.3 * norm_particle_motion
        
        return combined    

    def _extract_p_wave_times(self, metadata: Optional[Dict[str, Any]]) -> List[float]:
        """
        Extract P-wave arrival times from metadata.
        
        Args:
            metadata: Metadata that may contain P-wave information
            
        Returns:
            List of P-wave arrival times in seconds
        """
        p_wave_times = []
        
        if metadata is not None:
            # Look for P-wave times in various possible formats
            if 'p_wave_arrivals' in metadata:
                p_wave_times = metadata['p_wave_arrivals']
            elif 'p_waves' in metadata:
                # Extract from WaveSegment objects or dictionaries
                p_waves = metadata['p_waves']
                if isinstance(p_waves, list):
                    for wave in p_waves:
                        if hasattr(wave, 'arrival_time'):
                            p_wave_times.append(wave.arrival_time)
                        elif isinstance(wave, dict) and 'arrival_time' in wave:
                            p_wave_times.append(wave['arrival_time'])
            elif 'arrival_times' in metadata:
                arrivals = metadata['arrival_times']
                if isinstance(arrivals, dict) and 'p_wave_arrival' in arrivals:
                    if arrivals['p_wave_arrival'] is not None:
                        p_wave_times = [arrivals['p_wave_arrival']]
        
        return p_wave_times
    
    def _detect_onsets_with_context(self, sta_lta: np.ndarray, 
                                  p_wave_times: List[float]) -> List[int]:
        """
        Detect S-wave onsets using trigger/detrigger logic with P-wave context.
        
        Args:
            sta_lta: STA/LTA ratio time series
            p_wave_times: List of P-wave arrival times in seconds
            
        Returns:
            List of S-wave onset sample indices
        """
        onsets = []
        triggered = False
        trigger_start = 0
        
        # Convert P-wave times to sample indices
        p_wave_samples = [int(t * self.params.sampling_rate) for t in p_wave_times]
        
        for i, ratio in enumerate(sta_lta):
            # Check if we're in a valid S-wave detection window
            valid_window = self._is_valid_s_wave_window(i, p_wave_samples)
            
            if not triggered and ratio > self.params.trigger_threshold and valid_window:
                # Trigger detected in valid window
                triggered = True
                trigger_start = i
            elif triggered and ratio < self.params.detrigger_threshold:
                # Detrigger detected - find the actual onset within the triggered period
                if i > trigger_start:
                    # Look for the maximum STA/LTA ratio in the triggered period
                    trigger_segment = sta_lta[trigger_start:i]
                    max_idx = np.argmax(trigger_segment)
                    onset_idx = trigger_start + max_idx
                    
                    # Refine onset time and validate
                    refined_onset = self._refine_s_wave_onset(sta_lta, onset_idx, p_wave_samples)
                    if refined_onset is not None:
                        onsets.append(refined_onset)
                
                triggered = False
        
        return onsets
    
    def _is_valid_s_wave_window(self, sample_idx: int, p_wave_samples: List[int]) -> bool:
        """
        Check if the current sample is in a valid window for S-wave detection.
        
        S-waves should arrive after P-waves, so we look for S-waves in a window
        following each P-wave arrival.
        
        Args:
            sample_idx: Current sample index
            p_wave_samples: List of P-wave arrival sample indices
            
        Returns:
            True if sample is in valid S-wave detection window
        """
        if not p_wave_samples:
            # No P-wave context - allow detection anywhere
            return True
        
        context_window_samples = int(self.params.p_wave_context_window * self.params.sampling_rate)
        
        for p_sample in p_wave_samples:
            # S-wave should arrive after P-wave but within reasonable time
            if p_sample < sample_idx <= p_sample + context_window_samples:
                return True
        
        return False
    
    def _refine_s_wave_onset(self, sta_lta: np.ndarray, peak_idx: int, 
                           p_wave_samples: List[int]) -> Optional[int]:
        """
        Refine S-wave onset time and validate against P-wave context.
        
        Args:
            sta_lta: STA/LTA ratio time series
            peak_idx: Index of STA/LTA peak
            p_wave_samples: List of P-wave sample indices
            
        Returns:
            Refined onset index or None if invalid
        """
        search_window = int(2.0 * self.params.sampling_rate)  # 2 second search window
        start_idx = max(0, peak_idx - search_window)
        search_segment = sta_lta[start_idx:peak_idx]
        
        if len(search_segment) < 2:
            return peak_idx
        
        # Calculate gradient to find steepest rise
        gradient = np.gradient(search_segment)
        
        # Find the point with steepest positive gradient
        max_gradient_idx = np.argmax(gradient)
        refined_onset = start_idx + max_gradient_idx
        
        # Validate timing relative to P-waves
        if self._validate_s_wave_timing(refined_onset, p_wave_samples):
            return refined_onset
        else:
            return None
    
    def _validate_s_wave_timing(self, s_onset_idx: int, p_wave_samples: List[int]) -> bool:
        """
        Validate S-wave timing relative to P-wave arrivals.
        
        Args:
            s_onset_idx: S-wave onset sample index
            p_wave_samples: List of P-wave sample indices
            
        Returns:
            True if timing is valid for S-wave
        """
        if not p_wave_samples:
            return True  # No P-wave context to validate against
        
        s_time = s_onset_idx / self.params.sampling_rate
        
        for p_sample in p_wave_samples:
            p_time = p_sample / self.params.sampling_rate
            sp_time_diff = s_time - p_time
            
            # S-P time difference should be positive and reasonable
            # Typical S-P times range from 1-30 seconds for local earthquakes
            if 0.5 <= sp_time_diff <= 30.0:
                return True
        
        return False
    
    def _create_wave_segments(self, original_data: np.ndarray, 
                            filtered_data: np.ndarray,
                            onset_times: List[int], 
                            sta_lta: np.ndarray,
                            detection_method: str,
                            additional_data: Optional[Dict[str, Any]] = None) -> List[WaveSegment]:
        """
        Create WaveSegment objects from detected S-wave onsets.
        
        Args:
            original_data: Original seismic data
            filtered_data: Filtered seismic data
            onset_times: List of onset sample indices
            sta_lta: STA/LTA ratio time series
            detection_method: Method used for detection
            additional_data: Additional analysis data
            
        Returns:
            List of S-wave segments
        """
        wave_segments = []
        
        for onset_idx in onset_times:
            # Determine wave segment boundaries
            start_idx, end_idx = self._determine_s_wave_boundaries(
                filtered_data, onset_idx, sta_lta
            )
            
            if end_idx <= start_idx:
                continue
            
            # Extract wave data
            wave_data = original_data[start_idx:end_idx]
            
            # Calculate wave characteristics
            peak_amplitude = np.max(np.abs(wave_data))
            
            # Extract frequency features
            freq_features = self.feature_extractor.extract_frequency_domain_features(wave_data)
            dominant_frequency = freq_features.get('dominant_frequency', 0.0)
            
            # Calculate confidence based on STA/LTA peak and amplitude ratio
            base_confidence = min(1.0, sta_lta[onset_idx] / (self.params.trigger_threshold * 2))
            
            # Adjust confidence based on amplitude ratio if we have context
            amplitude_confidence = self._calculate_amplitude_ratio_confidence(
                wave_data, additional_data
            )
            
            # Combined confidence
            confidence = (base_confidence + amplitude_confidence) / 2
            
            # Only include waves that meet minimum confidence
            if confidence >= self.params.confidence_threshold:
                # Prepare metadata
                metadata = {
                    'sta_lta_peak': sta_lta[onset_idx],
                    'detection_method': f'S-wave {detection_method}',
                    'frequency_band': self.params.frequency_band,
                    'amplitude_ratio_confidence': amplitude_confidence
                }
                
                # Add polarization data if available
                if additional_data:
                    if 'polarization' in additional_data:
                        polarization_at_onset = additional_data['polarization'][onset_idx] if onset_idx < len(additional_data['polarization']) else 0
                        metadata['polarization_value'] = polarization_at_onset
                    
                    if 'particle_motion' in additional_data:
                        particle_motion_at_onset = additional_data['particle_motion'][onset_idx] if onset_idx < len(additional_data['particle_motion']) else 0
                        metadata['particle_motion_complexity'] = particle_motion_at_onset
                
                wave_segment = WaveSegment(
                    wave_type='S',
                    start_time=start_idx / self.params.sampling_rate,
                    end_time=end_idx / self.params.sampling_rate,
                    data=wave_data,
                    sampling_rate=self.params.sampling_rate,
                    peak_amplitude=peak_amplitude,
                    dominant_frequency=dominant_frequency,
                    arrival_time=onset_idx / self.params.sampling_rate,
                    confidence=confidence,
                    metadata=metadata
                )
                wave_segments.append(wave_segment)
        
        return wave_segments
    
    def _calculate_amplitude_ratio_confidence(self, s_wave_data: np.ndarray, 
                                            additional_data: Optional[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on S-wave to P-wave amplitude ratio.
        
        Args:
            s_wave_data: S-wave data segment
            additional_data: Additional analysis data
            
        Returns:
            Confidence score based on amplitude ratio
        """
        # Default confidence if no P-wave context
        if not additional_data or 'p_wave_amplitude' not in additional_data:
            return 0.5
        
        s_amplitude = np.max(np.abs(s_wave_data))
        p_amplitude = additional_data['p_wave_amplitude']
        
        if p_amplitude > 0:
            amplitude_ratio = s_amplitude / p_amplitude
            
            # S-waves typically have larger amplitudes than P-waves
            if amplitude_ratio >= self.params.amplitude_ratio_threshold:
                # Higher ratio gives higher confidence, but cap at 1.0
                confidence = min(1.0, amplitude_ratio / (self.params.amplitude_ratio_threshold * 2))
            else:
                # Lower ratio gives lower confidence
                confidence = amplitude_ratio / self.params.amplitude_ratio_threshold
        else:
            confidence = 0.5
        
        return confidence
    
    def _determine_s_wave_boundaries(self, data: np.ndarray, onset_idx: int, 
                                   sta_lta: np.ndarray) -> Tuple[int, int]:
        """
        Determine the start and end boundaries of an S-wave segment.
        
        Args:
            data: Filtered seismic data
            onset_idx: S-wave onset index
            sta_lta: STA/LTA ratio time series
            
        Returns:
            Tuple of (start_index, end_index)
        """
        # Minimum and maximum wave durations in samples
        min_samples = int(self.params.min_wave_duration * self.params.sampling_rate)
        max_samples = int(self.params.max_wave_duration * self.params.sampling_rate)
        
        # Start boundary: look backward from onset for low STA/LTA
        start_idx = onset_idx
        search_start = max(0, onset_idx - max_samples // 2)
        
        for i in range(onset_idx, search_start, -1):
            if i < len(sta_lta) and sta_lta[i] < self.params.detrigger_threshold:
                start_idx = i
                break
        
        # End boundary: S-waves typically last longer than P-waves
        # Look for energy decay and STA/LTA drop
        end_idx = min(len(data), onset_idx + max_samples)
        energy_threshold = 0.05 * np.max(data[onset_idx:onset_idx + min_samples] ** 2)
        
        for i in range(onset_idx + min_samples, min(len(data), onset_idx + max_samples)):
            # Check STA/LTA criterion
            if i < len(sta_lta) and sta_lta[i] < self.params.detrigger_threshold:
                # Also check energy decay (more lenient for S-waves)
                current_energy = np.mean(data[i-min_samples//2:i] ** 2)
                if current_energy < energy_threshold:
                    end_idx = i
                    break
        
        return start_idx, end_idx
    
    def get_detection_statistics(self, data: np.ndarray, 
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get detection statistics and diagnostic information for S-wave detection.
        
        Args:
            data: Seismic data to analyze
            metadata: Optional metadata with P-wave context
            
        Returns:
            Dictionary of detection statistics
        """
        # Handle multi-channel vs single-channel data
        if data.ndim == 1:
            processed_data = self.signal_processor.preprocess_seismic_data(data)
            filtered_data = self._apply_s_wave_filter(processed_data)
            char_function = filtered_data ** 2
            detection_method = 'single_channel'
        else:
            # Multi-channel analysis
            processed_channels = []
            for i in range(data.shape[0]):
                processed = self.signal_processor.preprocess_seismic_data(data[i])
                processed_channels.append(processed)
            
            processed_data = np.array(processed_channels)
            filtered_data = np.array([self._apply_s_wave_filter(ch) for ch in processed_data])
            
            # Calculate polarization and particle motion
            polarization = self._calculate_polarization_attributes(filtered_data)
            particle_motion = self._calculate_particle_motion(filtered_data)
            char_function = self._combine_s_wave_characteristics(polarization, particle_motion)
            detection_method = 'multi_channel'
            
            # Use first channel as representative
            processed_data = processed_data[0]
            filtered_data = filtered_data[0]
        
        # Calculate STA/LTA
        sta_lta = self._calculate_sta_lta(char_function)
        
        # Get P-wave context
        p_wave_times = self._extract_p_wave_times(metadata)
        
        # Detect onsets
        onsets = self._detect_onsets_with_context(sta_lta, p_wave_times)
        
        stats = {
            'num_detections': len(onsets),
            'max_sta_lta': np.max(sta_lta) if len(sta_lta) > 0 else 0,
            'mean_sta_lta': np.mean(sta_lta) if len(sta_lta) > 0 else 0,
            'trigger_threshold': self.params.trigger_threshold,
            'detrigger_threshold': self.params.detrigger_threshold,
            'onset_times': [onset / self.params.sampling_rate for onset in onsets],
            'frequency_band': self.params.frequency_band,
            'detection_method': detection_method,
            'data_duration': len(processed_data) / self.params.sampling_rate,
            'p_wave_context': p_wave_times,
            'amplitude_ratio_threshold': self.params.amplitude_ratio_threshold
        }
        
        # Calculate S-P time differences if both P and S waves detected
        if p_wave_times and onsets:
            sp_time_differences = []
            for s_onset in onsets:
                s_time = s_onset / self.params.sampling_rate
                for p_time in p_wave_times:
                    sp_diff = s_time - p_time
                    if 0.5 <= sp_diff <= 30.0:  # Reasonable S-P time range
                        sp_time_differences.append(sp_diff)
            
            stats['sp_time_differences'] = sp_time_differences
            stats['num_p_waves'] = len(p_wave_times)
            if sp_time_differences:
                stats['mean_sp_time'] = np.mean(sp_time_differences)
        else:
            stats['sp_time_differences'] = []
            stats['num_p_waves'] = len(p_wave_times)
        
        return stats


@dataclass
class SurfaceWaveDetectionParameters(DetectionParameters):
    """Parameters specific to surface wave detection."""
    frequency_band: Tuple[float, float] = (0.02, 0.5)  # Surface wave frequency band
    love_wave_band: Tuple[float, float] = (0.02, 0.3)  # Love wave specific band
    rayleigh_wave_band: Tuple[float, float] = (0.02, 0.5)  # Rayleigh wave specific band
    group_velocity_range: Tuple[float, float] = (2.5, 4.5)  # Expected group velocity km/s
    phase_velocity_range: Tuple[float, float] = (3.0, 5.0)  # Expected phase velocity km/s
    dispersion_window: float = 30.0  # Window for dispersion analysis in seconds
    min_surface_wave_duration: float = 10.0  # Minimum duration for surface waves
    spectral_coherence_threshold: float = 0.6  # Coherence threshold for wave identification
    energy_ratio_threshold: float = 2.0  # Surface wave to body wave energy ratio
    
    def __post_init__(self):
        """Validate surface wave detection parameters."""
        super().__post_init__()
        if self.dispersion_window <= 0:
            raise ValueError("Dispersion window must be positive")
        if self.min_surface_wave_duration <= 0:
            raise ValueError("Minimum surface wave duration must be positive")
        if not 0 <= self.spectral_coherence_threshold <= 1:
            raise ValueError("Spectral coherence threshold must be between 0 and 1")
        if self.energy_ratio_threshold <= 0:
            raise ValueError("Energy ratio threshold must be positive")


class SurfaceWaveDetector(WaveDetectorInterface):
    """
    Surface wave detection algorithm for Love and Rayleigh wave identification.
    
    This detector implements surface wave detection using:
    1. Frequency-time analysis for long-period wave identification
    2. Group velocity calculations for surface wave characterization
    3. Dispersion analysis to distinguish Love and Rayleigh waves
    4. Energy ratio analysis to separate surface waves from body waves
    """
    
    def __init__(self, parameters: SurfaceWaveDetectionParameters):
        """
        Initialize surface wave detector.
        
        Args:
            parameters: Surface wave detection parameters
        """
        self.params = parameters
        self.signal_processor = SignalProcessor(parameters.sampling_rate)
        self.filter_bank = FilterBank()
        self.feature_extractor = FeatureExtractor(parameters.sampling_rate)
        
    def detect_waves(self, data: np.ndarray, sampling_rate: float, 
                    metadata: Optional[Dict[str, Any]] = None) -> List[WaveSegment]:
        """
        Detect surface waves in seismic data.
        
        Args:
            data: Raw seismic time series data
            sampling_rate: Sampling rate in Hz
            metadata: Optional metadata (may include P-wave and S-wave detections)
            
        Returns:
            List of detected surface wave segments (Love and Rayleigh)
        """
        if len(data) == 0:
            return []
        
        # Update sampling rate if different from initialization
        if sampling_rate != self.params.sampling_rate:
            self.params.sampling_rate = sampling_rate
            self.signal_processor = SignalProcessor(sampling_rate)
            self.feature_extractor = FeatureExtractor(sampling_rate)
        
        # Check if data is long enough for surface wave analysis
        data_duration = len(data) / sampling_rate
        if data_duration < self.params.min_surface_wave_duration:
            warnings.warn(f"Data duration {data_duration:.1f}s is too short for surface wave detection "
                         f"(minimum: {self.params.min_surface_wave_duration}s)")
            return []
        
        # Preprocess the data
        processed_data = self.signal_processor.preprocess_seismic_data(data)
        
        # Apply surface wave frequency band filter
        filtered_data = self._apply_surface_wave_filter(processed_data)
        
        # Perform frequency-time analysis
        time_freq_analysis = self._perform_frequency_time_analysis(filtered_data)
        
        # Calculate group velocity characteristics
        group_velocity_data = self._calculate_group_velocity_characteristics(
            filtered_data, time_freq_analysis
        )
        
        # Identify surface wave segments using energy and dispersion criteria
        surface_wave_segments = self._identify_surface_wave_segments(
            processed_data, filtered_data, time_freq_analysis, group_velocity_data, metadata
        )
        
        # Separate Love and Rayleigh waves
        classified_segments = self._classify_surface_waves(surface_wave_segments, filtered_data)
        
        return classified_segments
    
    def get_wave_type(self) -> str:
        """Get the type of waves this detector identifies."""
        return 'Surface'  # Will be refined to 'Love' or 'Rayleigh' in classification
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set detection parameters.
        
        Args:
            parameters: Dictionary of parameter name-value pairs
        """
        for key, value in parameters.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                warnings.warn(f"Unknown parameter: {key}")
    
    def _apply_surface_wave_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply surface wave frequency band filter."""
        filter_params = FilterParameters(
            filter_type=FilterType.BANDPASS,
            cutoff_freq=self.params.frequency_band,
            order=4,
            sampling_rate=self.params.sampling_rate
        )
        return self.filter_bank.apply_filter(data, filter_params)
    
    def _perform_frequency_time_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Perform frequency-time analysis using spectrogram.
        
        Args:
            data: Filtered seismic data
            
        Returns:
            Dictionary containing spectrogram and related analysis
        """
        # Calculate spectrogram parameters
        window_length = int(self.params.dispersion_window * self.params.sampling_rate)
        window_length = min(window_length, len(data) // 4)  # Ensure reasonable window size
        overlap = int(window_length * 0.75)  # 75% overlap for good time resolution
        
        if window_length < 64:  # Minimum window size for meaningful analysis
            window_length = min(64, len(data) // 2)
            overlap = window_length // 2
        
        try:
            # Compute spectrogram
            frequencies, times, spectrogram = signal.spectrogram(
                data, 
                fs=self.params.sampling_rate,
                window='hann',
                nperseg=window_length,
                noverlap=overlap,
                scaling='density'
            )
            
            # Focus on surface wave frequency range
            freq_mask = (frequencies >= self.params.frequency_band[0]) & \
                       (frequencies <= self.params.frequency_band[1])
            
            surface_frequencies = frequencies[freq_mask]
            surface_spectrogram = spectrogram[freq_mask, :]
            
            # Calculate spectral energy in surface wave band
            surface_energy = np.sum(surface_spectrogram, axis=0)
            
            return {
                'frequencies': surface_frequencies,
                'times': times,
                'spectrogram': surface_spectrogram,
                'surface_energy': surface_energy,
                'total_energy': np.sum(spectrogram, axis=0),
                'frequency_resolution': frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0,
                'time_resolution': times[1] - times[0] if len(times) > 1 else 0
            }
            
        except Exception as e:
            warnings.warn(f"Spectrogram calculation failed: {e}")
            return {
                'frequencies': np.array([]),
                'times': np.array([]),
                'spectrogram': np.array([]),
                'surface_energy': np.array([]),
                'total_energy': np.array([]),
                'frequency_resolution': 0,
                'time_resolution': 0
            }
    
    def _calculate_group_velocity_characteristics(self, data: np.ndarray, 
                                                time_freq_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate group velocity characteristics for surface wave identification.
        
        Args:
            data: Filtered seismic data
            time_freq_analysis: Results from frequency-time analysis
            
        Returns:
            Dictionary containing group velocity analysis results
        """
        if len(time_freq_analysis['times']) == 0:
            return {'group_velocities': np.array([]), 'dispersion_curve': np.array([])}
        
        frequencies = time_freq_analysis['frequencies']
        times = time_freq_analysis['times']
        spectrogram = time_freq_analysis['spectrogram']
        
        # Initialize group velocity arrays
        group_velocities = np.zeros(len(frequencies))
        dispersion_curve = np.zeros((len(frequencies), len(times)))
        
        # Calculate group velocity for each frequency
        for i, freq in enumerate(frequencies):
            if freq > 0:
                # Extract amplitude envelope for this frequency
                freq_amplitude = spectrogram[i, :]
                
                # Find peak energy arrival time
                if np.max(freq_amplitude) > 0:
                    peak_time_idx = np.argmax(freq_amplitude)
                    peak_time = times[peak_time_idx]
                    
                    # Estimate group velocity (simplified approach)
                    # In real implementation, would need epicentral distance
                    # For now, use characteristic values for surface waves
                    if self.params.group_velocity_range[0] <= 3.5 <= self.params.group_velocity_range[1]:
                        group_velocities[i] = 3.5  # Typical surface wave group velocity
                    else:
                        group_velocities[i] = np.mean(self.params.group_velocity_range)
                    
                    # Store dispersion information
                    dispersion_curve[i, :] = freq_amplitude
        
        return {
            'group_velocities': group_velocities,
            'dispersion_curve': dispersion_curve,
            'mean_group_velocity': np.mean(group_velocities[group_velocities > 0]) if np.any(group_velocities > 0) else 0,
            'velocity_range': (np.min(group_velocities[group_velocities > 0]) if np.any(group_velocities > 0) else 0,
                              np.max(group_velocities[group_velocities > 0]) if np.any(group_velocities > 0) else 0)
        }
    
    def _identify_surface_wave_segments(self, original_data: np.ndarray, 
                                      filtered_data: np.ndarray,
                                      time_freq_analysis: Dict[str, Any],
                                      group_velocity_data: Dict[str, Any],
                                      metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Identify surface wave segments using energy and dispersion criteria.
        
        Args:
            original_data: Original seismic data
            filtered_data: Filtered seismic data
            time_freq_analysis: Frequency-time analysis results
            group_velocity_data: Group velocity analysis results
            metadata: Optional metadata with body wave information
            
        Returns:
            List of potential surface wave segments with characteristics
        """
        if len(time_freq_analysis['times']) == 0:
            return []
        
        surface_energy = time_freq_analysis['surface_energy']
        total_energy = time_freq_analysis['total_energy']
        times = time_freq_analysis['times']
        
        # Calculate energy ratio (surface wave energy / total energy)
        energy_ratio = np.divide(surface_energy, total_energy, 
                               out=np.zeros_like(surface_energy), 
                               where=total_energy != 0)
        
        # Find segments where surface wave energy is dominant
        dominant_mask = energy_ratio > (1.0 / self.params.energy_ratio_threshold)
        
        if not np.any(dominant_mask):
            return []
        
        # Find continuous segments of surface wave dominance
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_dominant in enumerate(dominant_mask):
            if is_dominant and not in_segment:
                # Start of new segment
                in_segment = True
                segment_start = i
            elif not is_dominant and in_segment:
                # End of current segment
                in_segment = False
                segment_end = i
                
                # Check if segment is long enough
                segment_duration = times[segment_end] - times[segment_start]
                if segment_duration >= self.params.min_surface_wave_duration:
                    segments.append({
                        'start_time': times[segment_start],
                        'end_time': times[segment_end],
                        'start_idx': segment_start,
                        'end_idx': segment_end,
                        'duration': segment_duration,
                        'mean_energy_ratio': np.mean(energy_ratio[segment_start:segment_end]),
                        'peak_energy_ratio': np.max(energy_ratio[segment_start:segment_end])
                    })
        
        # Handle case where segment extends to end of data
        if in_segment:
            segment_end = len(dominant_mask) - 1
            segment_duration = times[segment_end] - times[segment_start]
            if segment_duration >= self.params.min_surface_wave_duration:
                segments.append({
                    'start_time': times[segment_start],
                    'end_time': times[segment_end],
                    'start_idx': segment_start,
                    'end_idx': segment_end,
                    'duration': segment_duration,
                    'mean_energy_ratio': np.mean(energy_ratio[segment_start:segment_end]),
                    'peak_energy_ratio': np.max(energy_ratio[segment_start:segment_end])
                })
        
        return segments
    
    def _classify_surface_waves(self, segments: List[Dict[str, Any]], 
                              filtered_data: np.ndarray) -> List[WaveSegment]:
        """
        Classify surface wave segments into Love and Rayleigh waves.
        
        Args:
            segments: List of identified surface wave segments
            filtered_data: Filtered seismic data
            
        Returns:
            List of classified WaveSegment objects
        """
        classified_waves = []
        
        for segment in segments:
            # Convert time indices to sample indices
            start_sample = int(segment['start_time'] * self.params.sampling_rate)
            end_sample = int(segment['end_time'] * self.params.sampling_rate)
            
            # Ensure indices are within data bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(filtered_data), end_sample)
            
            if end_sample <= start_sample:
                continue
            
            # Extract segment data
            segment_data = filtered_data[start_sample:end_sample]
            
            # Classify as Love or Rayleigh based on frequency characteristics
            wave_type = self._determine_surface_wave_type(segment_data)
            
            # Calculate wave characteristics
            peak_amplitude = np.max(np.abs(segment_data))
            
            # Extract frequency features
            freq_features = self.feature_extractor.extract_frequency_domain_features(segment_data)
            dominant_frequency = freq_features.get('dominant_frequency', 0.0)
            
            # Calculate confidence based on energy ratio and duration
            confidence = min(1.0, segment['mean_energy_ratio'] * 
                           (segment['duration'] / self.params.min_surface_wave_duration))
            
            # Create WaveSegment
            wave_segment = WaveSegment(
                wave_type=wave_type,
                start_time=segment['start_time'],
                end_time=segment['end_time'],
                data=segment_data,
                sampling_rate=self.params.sampling_rate,
                peak_amplitude=peak_amplitude,
                dominant_frequency=dominant_frequency,
                arrival_time=segment['start_time'],  # Surface waves have gradual onset
                confidence=confidence,
                metadata={
                    'detection_method': 'frequency_time_analysis',
                    'energy_ratio': segment['mean_energy_ratio'],
                    'peak_energy_ratio': segment['peak_energy_ratio'],
                    'duration': segment['duration'],
                    'frequency_band': self.params.frequency_band,
                    'group_velocity_range': self.params.group_velocity_range
                }
            )
            
            classified_waves.append(wave_segment)
        
        return classified_waves
    
    def _determine_surface_wave_type(self, segment_data: np.ndarray) -> str:
        """
        Determine whether surface wave segment is Love or Rayleigh wave.
        
        Args:
            segment_data: Surface wave segment data
            
        Returns:
            Wave type string ('Love' or 'Rayleigh')
        """
        # Extract frequency characteristics
        freq_features = self.feature_extractor.extract_frequency_domain_features(segment_data)
        dominant_freq = freq_features.get('dominant_frequency', 0.0)
        
        # Simple classification based on frequency characteristics
        # Love waves typically have slightly higher frequencies than Rayleigh waves
        # This is a simplified approach - real classification would need multi-component data
        
        love_band_center = np.mean(self.params.love_wave_band)
        rayleigh_band_center = np.mean(self.params.rayleigh_wave_band)
        
        # Calculate distance to each wave type's characteristic frequency
        love_distance = abs(dominant_freq - love_band_center)
        rayleigh_distance = abs(dominant_freq - rayleigh_band_center)
        
        if love_distance < rayleigh_distance:
            return 'Love'
        else:
            return 'Rayleigh'
    
    def get_detection_statistics(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Get detection statistics and diagnostic information.
        
        Args:
            data: Input seismic data
            
        Returns:
            Dictionary containing detection statistics
        """
        processed_data = self.signal_processor.preprocess_seismic_data(data)
        filtered_data = self._apply_surface_wave_filter(processed_data)
        
        time_freq_analysis = self._perform_frequency_time_analysis(filtered_data)
        group_velocity_data = self._calculate_group_velocity_characteristics(
            filtered_data, time_freq_analysis
        )
        
        segments = self._identify_surface_wave_segments(
            processed_data, filtered_data, time_freq_analysis, group_velocity_data
        )
        
        classified_waves = self._classify_surface_waves(segments, filtered_data)
        
        # Count wave types
        love_waves = [w for w in classified_waves if w.wave_type == 'Love']
        rayleigh_waves = [w for w in classified_waves if w.wave_type == 'Rayleigh']
        
        return {
            'total_surface_waves': len(classified_waves),
            'love_waves': len(love_waves),
            'rayleigh_waves': len(rayleigh_waves),
            'mean_group_velocity': group_velocity_data.get('mean_group_velocity', 0),
            'velocity_range': group_velocity_data.get('velocity_range', (0, 0)),
            'frequency_band': self.params.frequency_band,
            'data_duration': len(data) / self.params.sampling_rate,
            'segments_identified': len(segments),
            'mean_segment_duration': np.mean([s['duration'] for s in segments]) if segments else 0,
            'detection_parameters': {
                'min_duration': self.params.min_surface_wave_duration,
                'energy_ratio_threshold': self.params.energy_ratio_threshold,
                'group_velocity_range': self.params.group_velocity_range
            }
        }