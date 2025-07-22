"""
Base interfaces for wave analysis components.

This module defines the core interfaces that all wave analysis components
must implement to ensure consistent behavior across the system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

from .models import WaveSegment, WaveAnalysisResult, DetailedAnalysis


class WaveDetectorInterface(ABC):
    """
    Base interface for wave detection algorithms.
    
    All wave detectors (P-wave, S-wave, surface wave) must implement
    this interface to ensure consistent detection behavior.
    """
    
    @abstractmethod
    def detect_waves(self, data: np.ndarray, sampling_rate: float, 
                    metadata: Optional[Dict[str, Any]] = None) -> List[WaveSegment]:
        """
        Detect waves in seismic data.
        
        Args:
            data: Raw seismic time series data
            sampling_rate: Sampling rate in Hz
            metadata: Optional metadata about the data
            
        Returns:
            List of detected wave segments
        """
        pass
    
    @abstractmethod
    def get_wave_type(self) -> str:
        """
        Get the type of waves this detector identifies.
        
        Returns:
            Wave type string ('P', 'S', 'Love', 'Rayleigh')
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set detection parameters.
        
        Args:
            parameters: Dictionary of parameter name-value pairs
        """
        pass


class WaveAnalyzerInterface(ABC):
    """
    Base interface for wave analysis algorithms.
    
    All wave analyzers must implement this interface to ensure
    consistent analysis behavior across different wave types.
    """
    
    @abstractmethod
    def analyze_waves(self, wave_result: WaveAnalysisResult) -> DetailedAnalysis:
        """
        Perform detailed analysis of separated waves.
        
        Args:
            wave_result: Result from wave separation containing all wave types
            
        Returns:
            Detailed analysis with timing, frequency, and magnitude information
        """
        pass
    
    @abstractmethod
    def calculate_arrival_times(self, waves: Dict[str, List[WaveSegment]]) -> Dict[str, float]:
        """
        Calculate precise arrival times for different wave types.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            
        Returns:
            Dictionary mapping wave types to arrival times
        """
        pass
    
    @abstractmethod
    def estimate_magnitude(self, waves: Dict[str, List[WaveSegment]]) -> List[Dict[str, Any]]:
        """
        Estimate earthquake magnitude using wave characteristics.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            
        Returns:
            List of magnitude estimates using different methods
        """
        pass


class WaveVisualizerInterface(ABC):
    """
    Base interface for wave visualization components.
    """
    
    @abstractmethod
    def create_time_series_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create time series visualization for wave segments.
        
        Args:
            wave_segments: List of wave segments to visualize
            
        Returns:
            Plot data dictionary for rendering
        """
        pass
    
    @abstractmethod
    def create_frequency_plot(self, wave_segments: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create frequency domain visualization for wave segments.
        
        Args:
            wave_segments: List of wave segments to analyze
            
        Returns:
            Frequency plot data dictionary for rendering
        """
        pass


class WaveExporterInterface(ABC):
    """
    Base interface for wave data export components.
    """
    
    @abstractmethod
    def export_waves(self, waves: Dict[str, List[WaveSegment]], 
                    format_type: str) -> bytes:
        """
        Export wave data in specified format.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            format_type: Export format ('mseed', 'sac', 'csv')
            
        Returns:
            Exported data as bytes
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported export formats.
        
        Returns:
            List of supported format strings
        """
        pass