"""
Main Wave Analyzer for comprehensive earthquake wave analysis.

This module provides the main WaveAnalyzer class that coordinates
all wave analysis components including arrival time calculation,
frequency analysis, and magnitude estimation.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

from ..models import WaveAnalysisResult, DetailedAnalysis, ArrivalTimes, QualityMetrics
from ..interfaces import WaveAnalyzerInterface
from .arrival_time_calculator import ArrivalTimeCalculator
from .frequency_analyzer import FrequencyAnalyzer
from .magnitude_estimator import MagnitudeEstimator


logger = logging.getLogger(__name__)


class WaveAnalyzer(WaveAnalyzerInterface):
    """
    Main analyzer for comprehensive wave analysis.
    
    This class coordinates all analysis components to provide complete
    earthquake wave analysis including timing, frequency characteristics,
    and magnitude estimation.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize the wave analyzer.
        
        Args:
            sampling_rate: Sampling rate of the seismic data in Hz
        """
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # Initialize analysis components
        self.arrival_calculator = ArrivalTimeCalculator(sampling_rate)
        self.frequency_analyzer = FrequencyAnalyzer(sampling_rate)
        self.magnitude_estimator = MagnitudeEstimator(sampling_rate)
        
    def analyze_waves(self, wave_result: WaveAnalysisResult) -> DetailedAnalysis:
        """
        Perform comprehensive analysis of separated waves.
        
        Args:
            wave_result: Result from wave separation containing all wave types
            
        Returns:
            DetailedAnalysis with complete wave analysis results
        """
        self.logger.info("Starting comprehensive wave analysis")
        
        # Organize waves by type
        waves_by_type = self._organize_waves_by_type(wave_result)
        
        # Calculate arrival times
        arrival_times = self.calculate_arrival_times(waves_by_type)
        
        # Perform frequency analysis
        frequency_analysis = self.frequency_analyzer.analyze_wave_frequencies(waves_by_type)
        
        # Estimate magnitude
        epicenter_distance = None
        if arrival_times.sp_time_difference:
            epicenter_distance = self.arrival_calculator.estimate_epicenter_distance(
                arrival_times.sp_time_difference
            )
        
        magnitude_estimates = self.estimate_magnitude(waves_by_type, epicenter_distance)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(wave_result, waves_by_type)
        
        # Create detailed analysis result
        detailed_analysis = DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates,
            epicenter_distance=epicenter_distance,
            frequency_analysis=frequency_analysis,
            quality_metrics=quality_metrics
        )
        
        self.logger.info(f"Wave analysis completed. Found {len(magnitude_estimates)} magnitude estimates")
        
        return detailed_analysis
    
    def calculate_arrival_times(self, waves: Dict[str, List]) -> ArrivalTimes:
        """
        Calculate precise arrival times for different wave types.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            
        Returns:
            ArrivalTimes object with calculated arrival times
        """
        return self.arrival_calculator.calculate_arrival_times(waves)
    
    def estimate_magnitude(self, waves: Dict[str, List], 
                         epicenter_distance: Optional[float] = None) -> List:
        """
        Estimate earthquake magnitude using wave characteristics.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            epicenter_distance: Distance to epicenter in km (if known)
            
        Returns:
            List of magnitude estimates using different methods
        """
        return self.magnitude_estimator.estimate_all_magnitudes(waves, epicenter_distance)
    
    def _organize_waves_by_type(self, wave_result: WaveAnalysisResult) -> Dict[str, List]:
        """
        Organize wave segments by type for analysis.
        
        Args:
            wave_result: Wave separation result
            
        Returns:
            Dictionary mapping wave types to their segments
        """
        waves_by_type = {}
        
        if wave_result.p_waves:
            waves_by_type['P'] = wave_result.p_waves
            
        if wave_result.s_waves:
            waves_by_type['S'] = wave_result.s_waves
            
        # Separate surface waves by type
        love_waves = [w for w in wave_result.surface_waves if w.wave_type == 'Love']
        rayleigh_waves = [w for w in wave_result.surface_waves if w.wave_type == 'Rayleigh']
        
        if love_waves:
            waves_by_type['Love'] = love_waves
        if rayleigh_waves:
            waves_by_type['Rayleigh'] = rayleigh_waves
            
        return waves_by_type
    
    def _calculate_quality_metrics(self, wave_result: WaveAnalysisResult, 
                                 waves_by_type: Dict[str, List]) -> QualityMetrics:
        """
        Calculate quality metrics for the analysis.
        
        Args:
            wave_result: Original wave separation result
            waves_by_type: Organized waves by type
            
        Returns:
            QualityMetrics object
        """
        # Calculate overall signal-to-noise ratio
        signal_power = np.mean(wave_result.original_data**2)
        
        # Estimate noise from first 10% of data (assuming it's pre-event)
        noise_length = max(1, len(wave_result.original_data) // 10)
        noise_data = wave_result.original_data[:noise_length]
        noise_power = np.mean(noise_data**2)
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 50.0  # High SNR if no noise detected
        
        # Calculate average detection confidence
        all_waves = []
        for wave_list in waves_by_type.values():
            all_waves.extend(wave_list)
        
        if all_waves:
            avg_confidence = np.mean([w.confidence for w in all_waves])
        else:
            avg_confidence = 0.0
        
        # Calculate analysis quality score
        quality_factors = []
        
        # Factor 1: Number of wave types detected
        wave_type_factor = len(waves_by_type) / 4.0  # Max 4 types (P, S, Love, Rayleigh)
        quality_factors.append(wave_type_factor)
        
        # Factor 2: SNR quality
        snr_factor = min(1.0, max(0.0, (snr - 10) / 20))  # Good SNR is 10-30 dB
        quality_factors.append(snr_factor)
        
        # Factor 3: Detection confidence
        quality_factors.append(avg_confidence)
        
        # Factor 4: Data completeness (assume complete for now)
        data_completeness = 1.0
        quality_factors.append(data_completeness)
        
        analysis_quality_score = np.mean(quality_factors)
        
        # Collect processing warnings
        warnings = []
        if snr < 10:
            warnings.append("Low signal-to-noise ratio detected")
        if avg_confidence < 0.5:
            warnings.append("Low average detection confidence")
        if len(waves_by_type) < 2:
            warnings.append("Limited wave types detected")
        
        return QualityMetrics(
            signal_to_noise_ratio=snr,
            detection_confidence=avg_confidence,
            analysis_quality_score=analysis_quality_score,
            data_completeness=data_completeness,
            processing_warnings=warnings
        )
    
    def analyze_single_wave_type(self, wave_type: str, wave_segments: List) -> Dict:
        """
        Perform detailed analysis of a single wave type.
        
        Args:
            wave_type: Type of wave to analyze
            wave_segments: List of wave segments of this type
            
        Returns:
            Dictionary with analysis results for this wave type
        """
        if not wave_segments:
            return {}
        
        # Get the best wave segment (highest amplitude * confidence)
        best_wave = max(wave_segments, key=lambda w: w.peak_amplitude * w.confidence)
        
        # Frequency analysis
        freq_data = self.frequency_analyzer.analyze_single_wave_frequency(best_wave)
        
        # Arrival time analysis
        arrival_time = best_wave.arrival_time
        
        # Basic characteristics
        characteristics = {
            'wave_type': wave_type,
            'arrival_time': arrival_time,
            'peak_amplitude': best_wave.peak_amplitude,
            'duration': best_wave.duration,
            'dominant_frequency': freq_data.dominant_frequency,
            'frequency_range': freq_data.frequency_range,
            'spectral_centroid': freq_data.spectral_centroid,
            'bandwidth': freq_data.bandwidth,
            'confidence': best_wave.confidence,
            'segment_count': len(wave_segments)
        }
        
        return characteristics
    
    def compare_wave_characteristics(self, analysis1: DetailedAnalysis, 
                                   analysis2: DetailedAnalysis) -> Dict:
        """
        Compare characteristics between two wave analyses.
        
        Args:
            analysis1: First analysis result
            analysis2: Second analysis result
            
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        # Compare arrival times
        if (analysis1.arrival_times.p_wave_arrival and 
            analysis2.arrival_times.p_wave_arrival):
            p_time_diff = abs(analysis1.arrival_times.p_wave_arrival - 
                            analysis2.arrival_times.p_wave_arrival)
            comparison['p_wave_time_difference'] = p_time_diff
        
        if (analysis1.arrival_times.s_wave_arrival and 
            analysis2.arrival_times.s_wave_arrival):
            s_time_diff = abs(analysis1.arrival_times.s_wave_arrival - 
                            analysis2.arrival_times.s_wave_arrival)
            comparison['s_wave_time_difference'] = s_time_diff
        
        # Compare magnitudes
        if analysis1.magnitude_estimates and analysis2.magnitude_estimates:
            mag1 = analysis1.best_magnitude_estimate
            mag2 = analysis2.best_magnitude_estimate
            
            if mag1 and mag2:
                mag_diff = abs(mag1.magnitude - mag2.magnitude)
                comparison['magnitude_difference'] = mag_diff
        
        # Compare quality metrics
        if analysis1.quality_metrics and analysis2.quality_metrics:
            snr_diff = abs(analysis1.quality_metrics.signal_to_noise_ratio - 
                          analysis2.quality_metrics.signal_to_noise_ratio)
            comparison['snr_difference'] = snr_diff
        
        return comparison
    
    def set_parameters(self, **kwargs):
        """
        Set analysis parameters for all components.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        # Pass parameters to individual components
        if 'arrival_time_params' in kwargs:
            self.arrival_calculator.set_parameters(**kwargs['arrival_time_params'])
        
        if 'frequency_params' in kwargs:
            self.frequency_analyzer.set_parameters(**kwargs['frequency_params'])
        
        if 'magnitude_params' in kwargs:
            self.magnitude_estimator.set_parameters(**kwargs['magnitude_params'])