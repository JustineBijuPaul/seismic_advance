"""
Core data models for wave analysis.

This module defines the fundamental data structures used throughout
the wave analysis system for representing waves, analysis results,
and related metadata.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime


@dataclass
class WaveSegment:
    """
    Represents a segment of seismic wave data.
    
    This is the fundamental unit for representing detected waves
    of any type (P, S, Love, Rayleigh) with their characteristics.
    """
    wave_type: str  # 'P', 'S', 'Love', 'Rayleigh'
    start_time: float  # Start time in seconds from beginning of record
    end_time: float    # End time in seconds from beginning of record
    data: np.ndarray   # Raw wave data segment
    sampling_rate: float  # Sampling rate in Hz
    peak_amplitude: float  # Maximum amplitude in the segment
    dominant_frequency: float  # Dominant frequency in Hz
    arrival_time: float  # Precise arrival time in seconds
    confidence: float = 1.0  # Detection confidence score (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __post_init__(self):
        """Validate wave segment data after initialization."""
        if self.end_time <= self.start_time:
            raise ValueError("End time must be greater than start time")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.wave_type not in ['P', 'S', 'Love', 'Rayleigh']:
            raise ValueError(f"Invalid wave type: {self.wave_type}")
    
    @property
    def duration(self) -> float:
        """Get the duration of the wave segment in seconds."""
        return self.end_time - self.start_time
    
    @property
    def sample_count(self) -> int:
        """Get the number of samples in the wave segment."""
        return len(self.data)


@dataclass
class ArrivalTimes:
    """
    Represents arrival times for different wave types.
    """
    p_wave_arrival: Optional[float] = None  # P-wave arrival time in seconds
    s_wave_arrival: Optional[float] = None  # S-wave arrival time in seconds
    surface_wave_arrival: Optional[float] = None  # Surface wave arrival time
    sp_time_difference: Optional[float] = None  # S-P time difference
    
    def __post_init__(self):
        """Calculate S-P time difference if both arrivals are available."""
        if (self.p_wave_arrival is not None and 
            self.s_wave_arrival is not None and 
            self.sp_time_difference is None):
            self.sp_time_difference = self.s_wave_arrival - self.p_wave_arrival


@dataclass
class MagnitudeEstimate:
    """
    Represents an earthquake magnitude estimate.
    """
    method: str  # 'ML', 'Mb', 'Ms', 'Mw'
    magnitude: float  # Estimated magnitude value
    confidence: float  # Confidence in the estimate (0-1)
    wave_type_used: str  # Wave type used for calculation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate magnitude estimate data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.method not in ['ML', 'Mb', 'Ms', 'Mw']:
            raise ValueError(f"Invalid magnitude method: {self.method}")


@dataclass
class FrequencyData:
    """
    Represents frequency domain analysis results.
    """
    frequencies: np.ndarray  # Frequency bins in Hz
    power_spectrum: np.ndarray  # Power spectral density
    dominant_frequency: float  # Peak frequency in Hz
    frequency_range: tuple  # (min_freq, max_freq) with significant energy
    spectral_centroid: float  # Spectral centroid frequency
    bandwidth: float  # Spectral bandwidth
    
    def __post_init__(self):
        """Validate frequency data."""
        if len(self.frequencies) != len(self.power_spectrum):
            raise ValueError("Frequencies and power spectrum must have same length")


@dataclass
class QualityMetrics:
    """
    Represents quality metrics for wave analysis results.
    """
    signal_to_noise_ratio: float  # Overall SNR
    detection_confidence: float  # Average detection confidence
    analysis_quality_score: float  # Overall analysis quality (0-1)
    data_completeness: float  # Fraction of expected data present (0-1)
    processing_warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate quality metrics."""
        if not 0 <= self.detection_confidence <= 1:
            raise ValueError("Detection confidence must be between 0 and 1")
        if not 0 <= self.analysis_quality_score <= 1:
            raise ValueError("Analysis quality score must be between 0 and 1")
        if not 0 <= self.data_completeness <= 1:
            raise ValueError("Data completeness must be between 0 and 1")


@dataclass
class WaveAnalysisResult:
    """
    Represents the result of wave separation analysis.
    
    This is the primary result structure containing all detected
    wave types and associated metadata.
    """
    original_data: np.ndarray  # Original seismic time series
    sampling_rate: float  # Sampling rate in Hz
    p_waves: List[WaveSegment] = field(default_factory=list)  # Detected P-waves
    s_waves: List[WaveSegment] = field(default_factory=list)  # Detected S-waves
    surface_waves: List[WaveSegment] = field(default_factory=list)  # Surface waves
    metadata: Dict[str, Any] = field(default_factory=dict)  # Analysis metadata
    processing_timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate wave analysis result data."""
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
        if len(self.original_data) == 0:
            raise ValueError("Original data cannot be empty")
    
    @property
    def total_waves_detected(self) -> int:
        """Get total number of waves detected across all types."""
        return len(self.p_waves) + len(self.s_waves) + len(self.surface_waves)
    
    @property
    def wave_types_detected(self) -> List[str]:
        """Get list of wave types that were detected."""
        types = []
        if self.p_waves:
            types.append('P')
        if self.s_waves:
            types.append('S')
        if self.surface_waves:
            types.extend([wave.wave_type for wave in self.surface_waves])
        return list(set(types))
    
    def get_waves_by_type(self, wave_type: str) -> List[WaveSegment]:
        """Get all wave segments of a specific type."""
        if wave_type == 'P':
            return self.p_waves
        elif wave_type == 'S':
            return self.s_waves
        elif wave_type in ['Love', 'Rayleigh']:
            return [wave for wave in self.surface_waves if wave.wave_type == wave_type]
        else:
            raise ValueError(f"Unknown wave type: {wave_type}")


@dataclass
class DetailedAnalysis:
    """
    Represents comprehensive analysis results including timing,
    frequency, magnitude, and quality information.
    """
    wave_result: WaveAnalysisResult  # Original wave separation result
    arrival_times: ArrivalTimes  # Calculated arrival times
    magnitude_estimates: List[MagnitudeEstimate] = field(default_factory=list)
    epicenter_distance: Optional[float] = None  # Distance to epicenter in km
    frequency_analysis: Dict[str, FrequencyData] = field(default_factory=dict)
    quality_metrics: Optional[QualityMetrics] = None
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def best_magnitude_estimate(self) -> Optional[MagnitudeEstimate]:
        """Get the magnitude estimate with highest confidence."""
        if not self.magnitude_estimates:
            return None
        return max(self.magnitude_estimates, key=lambda x: x.confidence)
    
    @property
    def has_complete_analysis(self) -> bool:
        """Check if analysis contains all expected components."""
        return (
            self.arrival_times.p_wave_arrival is not None and
            self.arrival_times.s_wave_arrival is not None and
            len(self.magnitude_estimates) > 0 and
            self.quality_metrics is not None
        )
    
    def get_frequency_data(self, wave_type: str) -> Optional[FrequencyData]:
        """Get frequency analysis data for a specific wave type."""
        return self.frequency_analysis.get(wave_type)
    
    def add_magnitude_estimate(self, estimate: MagnitudeEstimate) -> None:
        """Add a magnitude estimate to the analysis."""
        self.magnitude_estimates.append(estimate)
    
    def set_frequency_analysis(self, wave_type: str, freq_data: FrequencyData) -> None:
        """Set frequency analysis data for a wave type."""
        self.frequency_analysis[wave_type] = freq_data