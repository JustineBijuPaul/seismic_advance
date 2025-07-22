"""
Arrival Time Calculator for precise wave timing analysis.

This module implements sophisticated algorithms for calculating precise
arrival times of different wave types using cross-correlation and
other advanced timing techniques.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import signal
from scipy.optimize import minimize_scalar
import logging

from ..models import WaveSegment, ArrivalTimes
from ..interfaces import WaveAnalyzerInterface


logger = logging.getLogger(__name__)


class ArrivalTimeCalculator:
    """
    Calculator for precise arrival times using cross-correlation and other methods.
    
    This class implements various algorithms for refining wave arrival times
    beyond the initial detection, including cross-correlation methods and
    characteristic function analysis.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize the arrival time calculator.
        
        Args:
            sampling_rate: Sampling rate of the seismic data in Hz
        """
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # Default parameters for arrival time calculation
        self.cross_correlation_window = 2.0  # seconds
        self.refinement_window = 0.5  # seconds around initial pick
        self.min_correlation_threshold = 0.3
        
    def calculate_arrival_times(self, waves: Dict[str, List[WaveSegment]]) -> ArrivalTimes:
        """
        Calculate precise arrival times for all wave types.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            
        Returns:
            ArrivalTimes object with calculated arrival times
        """
        arrival_times = ArrivalTimes()
        
        # Calculate P-wave arrival time
        if 'P' in waves and waves['P']:
            p_arrival = self._calculate_p_wave_arrival(waves['P'])
            arrival_times.p_wave_arrival = p_arrival
            
        # Calculate S-wave arrival time
        if 'S' in waves and waves['S']:
            s_arrival = self._calculate_s_wave_arrival(waves['S'])
            arrival_times.s_wave_arrival = s_arrival
            
        # Calculate surface wave arrival time
        surface_waves = []
        if 'Love' in waves:
            surface_waves.extend(waves['Love'])
        if 'Rayleigh' in waves:
            surface_waves.extend(waves['Rayleigh'])
            
        if surface_waves:
            surface_arrival = self._calculate_surface_wave_arrival(surface_waves)
            arrival_times.surface_wave_arrival = surface_arrival
            
        # Calculate S-P time difference
        if arrival_times.p_wave_arrival and arrival_times.s_wave_arrival:
            arrival_times.sp_time_difference = (
                arrival_times.s_wave_arrival - arrival_times.p_wave_arrival
            )
            
        return arrival_times
    
    def _calculate_p_wave_arrival(self, p_waves: List[WaveSegment]) -> Optional[float]:
        """
        Calculate precise P-wave arrival time using cross-correlation.
        
        Args:
            p_waves: List of detected P-wave segments
            
        Returns:
            Refined P-wave arrival time in seconds
        """
        if not p_waves:
            return None
            
        # Use the earliest P-wave with highest confidence
        best_p_wave = min(p_waves, key=lambda w: (w.arrival_time, -w.confidence))
        
        # Refine arrival time using characteristic function
        refined_time = self._refine_arrival_with_characteristic_function(
            best_p_wave, 'P'
        )
        
        self.logger.debug(f"P-wave arrival refined from {best_p_wave.arrival_time:.3f} "
                         f"to {refined_time:.3f} seconds")
        
        return refined_time
    
    def _calculate_s_wave_arrival(self, s_waves: List[WaveSegment]) -> Optional[float]:
        """
        Calculate precise S-wave arrival time using polarization analysis.
        
        Args:
            s_waves: List of detected S-wave segments
            
        Returns:
            Refined S-wave arrival time in seconds
        """
        if not s_waves:
            return None
            
        # Use the earliest S-wave with highest confidence
        best_s_wave = min(s_waves, key=lambda w: (w.arrival_time, -w.confidence))
        
        # Refine arrival time using particle motion analysis
        refined_time = self._refine_arrival_with_particle_motion(best_s_wave)
        
        self.logger.debug(f"S-wave arrival refined from {best_s_wave.arrival_time:.3f} "
                         f"to {refined_time:.3f} seconds")
        
        return refined_time
    
    def _calculate_surface_wave_arrival(self, surface_waves: List[WaveSegment]) -> Optional[float]:
        """
        Calculate surface wave arrival time.
        
        Args:
            surface_waves: List of detected surface wave segments
            
        Returns:
            Surface wave arrival time in seconds
        """
        if not surface_waves:
            return None
            
        # Use the earliest surface wave
        earliest_surface = min(surface_waves, key=lambda w: w.arrival_time)
        
        # Surface waves typically don't need as much refinement
        # but we can still apply basic characteristic function analysis
        refined_time = self._refine_arrival_with_characteristic_function(
            earliest_surface, 'Surface'
        )
        
        return refined_time
    
    def _refine_arrival_with_characteristic_function(self, 
                                                   wave: WaveSegment, 
                                                   wave_type: str) -> float:
        """
        Refine arrival time using characteristic function analysis.
        
        Args:
            wave: Wave segment to refine
            wave_type: Type of wave ('P', 'S', 'Surface')
            
        Returns:
            Refined arrival time
        """
        # Create characteristic function based on wave type
        if wave_type == 'P':
            char_func = self._create_p_wave_characteristic_function(wave.data)
        elif wave_type == 'S':
            char_func = self._create_s_wave_characteristic_function(wave.data)
        else:
            char_func = self._create_surface_wave_characteristic_function(wave.data)
        
        # Find peak in characteristic function
        peak_idx = np.argmax(char_func)
        
        # Convert to time relative to wave start
        peak_time_offset = peak_idx / self.sampling_rate
        
        # Return absolute arrival time
        return wave.start_time + peak_time_offset
    
    def _refine_arrival_with_particle_motion(self, wave: WaveSegment) -> float:
        """
        Refine S-wave arrival using particle motion analysis.
        
        Args:
            wave: S-wave segment to refine
            
        Returns:
            Refined arrival time
        """
        # For S-waves, we look for changes in particle motion
        # This is a simplified implementation - in practice would use
        # multi-component data
        
        # Calculate instantaneous amplitude
        analytic_signal = signal.hilbert(wave.data)
        instantaneous_amplitude = np.abs(analytic_signal)
        
        # Find significant increase in amplitude
        amplitude_gradient = np.gradient(instantaneous_amplitude)
        
        # Find the steepest positive gradient (arrival)
        max_gradient_idx = np.argmax(amplitude_gradient)
        
        # Convert to absolute time
        arrival_offset = max_gradient_idx / self.sampling_rate
        return wave.start_time + arrival_offset
    
    def _create_p_wave_characteristic_function(self, data: np.ndarray) -> np.ndarray:
        """
        Create characteristic function for P-wave detection.
        
        Args:
            data: Seismic data segment
            
        Returns:
            Characteristic function values
        """
        # STA/LTA characteristic function for P-waves
        sta_length = int(0.1 * self.sampling_rate)  # 0.1 second
        lta_length = int(1.0 * self.sampling_rate)  # 1.0 second
        
        return self._calculate_sta_lta(data, sta_length, lta_length)
    
    def _create_s_wave_characteristic_function(self, data: np.ndarray) -> np.ndarray:
        """
        Create characteristic function for S-wave detection.
        
        Args:
            data: Seismic data segment
            
        Returns:
            Characteristic function values
        """
        # Energy-based characteristic function for S-waves
        window_length = int(0.2 * self.sampling_rate)  # 0.2 second window
        
        # Calculate energy in sliding window
        energy = np.zeros(len(data))
        for i in range(window_length, len(data)):
            window_data = data[i-window_length:i]
            energy[i] = np.sum(window_data**2)
            
        return energy
    
    def _create_surface_wave_characteristic_function(self, data: np.ndarray) -> np.ndarray:
        """
        Create characteristic function for surface wave detection.
        
        Args:
            data: Seismic data segment
            
        Returns:
            Characteristic function values
        """
        # Envelope-based characteristic function for surface waves
        analytic_signal = signal.hilbert(data)
        envelope = np.abs(analytic_signal)
        
        # Smooth the envelope
        window_length = int(0.5 * self.sampling_rate)  # 0.5 second smoothing
        if window_length > 0:
            envelope = signal.savgol_filter(envelope, 
                                          min(window_length, len(envelope)//2*2-1), 
                                          3)
        
        return envelope
    
    def _calculate_sta_lta(self, data: np.ndarray, sta_length: int, lta_length: int) -> np.ndarray:
        """
        Calculate STA/LTA characteristic function.
        
        Args:
            data: Input seismic data
            sta_length: Short-term average window length in samples
            lta_length: Long-term average window length in samples
            
        Returns:
            STA/LTA ratio values
        """
        # Ensure minimum window sizes
        sta_length = max(sta_length, 1)
        lta_length = max(lta_length, sta_length + 1)
        
        # Calculate squared data for energy
        data_squared = data**2
        
        # Initialize output
        sta_lta = np.zeros(len(data))
        
        # Calculate STA/LTA
        for i in range(lta_length, len(data)):
            # Long-term average
            lta = np.mean(data_squared[i-lta_length:i-sta_length])
            
            # Short-term average
            sta = np.mean(data_squared[i-sta_length:i])
            
            # Avoid division by zero
            if lta > 0:
                sta_lta[i] = sta / lta
            else:
                sta_lta[i] = 0
                
        return sta_lta
    
    def cross_correlate_arrivals(self, wave1: WaveSegment, wave2: WaveSegment) -> float:
        """
        Calculate cross-correlation between two wave arrivals for timing refinement.
        
        Args:
            wave1: First wave segment
            wave2: Second wave segment
            
        Returns:
            Time delay between arrivals in seconds
        """
        # Extract data around arrival times
        window_samples = int(self.cross_correlation_window * self.sampling_rate)
        
        # Get data segments
        data1 = wave1.data[:window_samples] if len(wave1.data) >= window_samples else wave1.data
        data2 = wave2.data[:window_samples] if len(wave2.data) >= window_samples else wave2.data
        
        # Normalize data
        data1 = (data1 - np.mean(data1)) / (np.std(data1) + 1e-10)
        data2 = (data2 - np.mean(data2)) / (np.std(data2) + 1e-10)
        
        # Calculate cross-correlation
        correlation = signal.correlate(data1, data2, mode='full')
        
        # Find peak correlation
        peak_idx = np.argmax(np.abs(correlation))
        
        # Convert to time delay
        delay_samples = peak_idx - (len(data2) - 1)
        delay_time = delay_samples / self.sampling_rate
        
        return delay_time
    
    def estimate_epicenter_distance(self, sp_time_difference: float) -> float:
        """
        Estimate epicenter distance using S-P time difference.
        
        Args:
            sp_time_difference: S-P time difference in seconds
            
        Returns:
            Estimated epicenter distance in kilometers
        """
        # Typical P-wave velocity: 6.0 km/s
        # Typical S-wave velocity: 3.5 km/s
        p_velocity = 6.0  # km/s
        s_velocity = 3.5  # km/s
        
        # Distance = (S-P time) / (1/Vs - 1/Vp)
        velocity_factor = (1.0 / s_velocity) - (1.0 / p_velocity)
        distance = sp_time_difference / velocity_factor
        
        return max(0, distance)  # Ensure non-negative distance
    
    def set_parameters(self, **kwargs):
        """
        Set calculation parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'cross_correlation_window' in kwargs:
            self.cross_correlation_window = kwargs['cross_correlation_window']
        if 'refinement_window' in kwargs:
            self.refinement_window = kwargs['refinement_window']
        if 'min_correlation_threshold' in kwargs:
            self.min_correlation_threshold = kwargs['min_correlation_threshold']