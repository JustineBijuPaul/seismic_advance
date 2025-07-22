"""
Quality Metrics Calculator for wave analysis result validation.

This module implements comprehensive quality assessment for wave analysis results,
including signal-to-noise ratio calculations, confidence scoring, and validation
of wave detection results.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from scipy import signal, stats

from ..models import WaveSegment, WaveAnalysisResult, DetailedAnalysis, QualityMetrics
from ..interfaces import WaveAnalyzerInterface


logger = logging.getLogger(__name__)


class QualityMetricsCalculator:
    """
    Calculator for comprehensive quality metrics of wave analysis results.
    
    This class implements various quality assessment methods including
    signal-to-noise ratio calculations, detection confidence scoring,
    and overall analysis quality validation.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize the quality metrics calculator.
        
        Args:
            sampling_rate: Sampling rate of the seismic data in Hz
        """
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # Quality assessment parameters
        self.noise_window_fraction = 0.1  # Use first 10% of data for noise estimation
        self.min_snr_threshold = 3.0  # Minimum acceptable SNR in dB
        self.min_confidence_threshold = 0.3  # Minimum detection confidence
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        }
        
    def calculate_quality_metrics(self, wave_result: WaveAnalysisResult, 
                                detailed_analysis: Optional[DetailedAnalysis] = None) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics for wave analysis results.
        
        Args:
            wave_result: Wave separation result to assess
            detailed_analysis: Optional detailed analysis for additional metrics
            
        Returns:
            QualityMetrics object with comprehensive quality assessment
        """
        self.logger.info("Calculating quality metrics for wave analysis")
        
        # Calculate signal-to-noise ratio
        snr = self._calculate_signal_to_noise_ratio(wave_result.original_data)
        
        # Calculate detection confidence
        detection_confidence = self._calculate_detection_confidence(wave_result)
        
        # Calculate data completeness
        data_completeness = self._calculate_data_completeness(wave_result)
        
        # Calculate overall analysis quality score
        analysis_quality_score = self._calculate_analysis_quality_score(
            wave_result, snr, detection_confidence, data_completeness, detailed_analysis
        )
        
        # Generate processing warnings
        warnings = self._generate_quality_warnings(
            wave_result, snr, detection_confidence, data_completeness
        )
        
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=snr,
            detection_confidence=detection_confidence,
            analysis_quality_score=analysis_quality_score,
            data_completeness=data_completeness,
            processing_warnings=warnings
        )
        
        self.logger.info(f"Quality assessment complete. Overall score: {analysis_quality_score:.3f}")
        
        return quality_metrics
    
    def _calculate_signal_to_noise_ratio(self, data: np.ndarray) -> float:
        """
        Calculate signal-to-noise ratio of the seismic data.
        
        Args:
            data: Raw seismic time series data
            
        Returns:
            Signal-to-noise ratio in dB
        """
        # Use first portion of data as noise estimate (pre-event)
        noise_length = max(1, int(len(data) * self.noise_window_fraction))
        noise_data = data[:noise_length]
        
        # Calculate noise power
        noise_power = np.mean(noise_data**2)
        
        # Calculate signal power (entire data)
        signal_power = np.mean(data**2)
        
        # Calculate SNR in dB
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 50.0  # High SNR if no noise detected
        
        return snr_db
    
    def _calculate_detection_confidence(self, wave_result: WaveAnalysisResult) -> float:
        """
        Calculate average detection confidence across all detected waves.
        
        Args:
            wave_result: Wave separation result
            
        Returns:
            Average detection confidence (0-1)
        """
        all_waves = []
        all_waves.extend(wave_result.p_waves)
        all_waves.extend(wave_result.s_waves)
        all_waves.extend(wave_result.surface_waves)
        
        if not all_waves:
            return 0.0
        
        # Calculate weighted average confidence (weighted by amplitude)
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for wave in all_waves:
            weight = wave.peak_amplitude
            total_weighted_confidence += wave.confidence * weight
            total_weight += weight
        
        if total_weight > 0:
            avg_confidence = total_weighted_confidence / total_weight
        else:
            avg_confidence = np.mean([wave.confidence for wave in all_waves])
        
        return avg_confidence
    
    def _calculate_data_completeness(self, wave_result: WaveAnalysisResult) -> float:
        """
        Calculate data completeness metric.
        
        Args:
            wave_result: Wave separation result
            
        Returns:
            Data completeness score (0-1)
        """
        # Check for data gaps, spikes, or other quality issues
        data = wave_result.original_data
        
        # Check for non-finite values
        finite_fraction = np.sum(np.isfinite(data)) / len(data)
        
        # Check for data saturation (clipping)
        data_range = np.max(data) - np.min(data)
        if data_range > 0:
            # Look for potential clipping (many samples at min/max values)
            max_val = np.max(data)
            min_val = np.min(data)
            
            max_count = np.sum(np.abs(data - max_val) < 0.01 * data_range)
            min_count = np.sum(np.abs(data - min_val) < 0.01 * data_range)
            
            clipping_fraction = (max_count + min_count) / len(data)
            saturation_factor = max(0, 1 - clipping_fraction * 10)  # Penalize clipping
        else:
            saturation_factor = 0.0  # Constant data is problematic
        
        # Check for data gaps (consecutive zeros or constant values)
        diff_data = np.diff(data)
        zero_diff_fraction = np.sum(np.abs(diff_data) < 1e-10) / len(diff_data)
        gap_factor = max(0, 1 - zero_diff_fraction * 5)  # Penalize gaps
        
        # Combine factors
        completeness = finite_fraction * saturation_factor * gap_factor
        
        return completeness
    
    def _calculate_analysis_quality_score(self, wave_result: WaveAnalysisResult,
                                        snr: float, detection_confidence: float,
                                        data_completeness: float,
                                        detailed_analysis: Optional[DetailedAnalysis] = None) -> float:
        """
        Calculate overall analysis quality score.
        
        Args:
            wave_result: Wave separation result
            snr: Signal-to-noise ratio in dB
            detection_confidence: Average detection confidence
            data_completeness: Data completeness score
            detailed_analysis: Optional detailed analysis for additional metrics
            
        Returns:
            Overall quality score (0-1)
        """
        quality_factors = []
        
        # Factor 1: SNR quality (0-1)
        snr_factor = self._normalize_snr_to_quality(snr)
        quality_factors.append(snr_factor)
        
        # Factor 2: Detection confidence (0-1)
        quality_factors.append(detection_confidence)
        
        # Factor 3: Data completeness (0-1)
        quality_factors.append(data_completeness)
        
        # Factor 4: Wave type diversity (0-1)
        wave_types_detected = len(wave_result.wave_types_detected)
        max_wave_types = 4  # P, S, Love, Rayleigh
        diversity_factor = wave_types_detected / max_wave_types
        quality_factors.append(diversity_factor)
        
        # Factor 5: Wave count quality (0-1)
        total_waves = wave_result.total_waves_detected
        if total_waves > 0:
            # Prefer moderate number of detections (not too few, not too many)
            optimal_wave_count = 5
            wave_count_factor = min(1.0, total_waves / optimal_wave_count)
            if total_waves > optimal_wave_count * 2:
                # Penalize excessive detections (might be noise)
                wave_count_factor *= 0.5
        else:
            wave_count_factor = 0.0
        quality_factors.append(wave_count_factor)
        
        # Factor 6: Timing consistency (if detailed analysis available)
        if detailed_analysis and detailed_analysis.arrival_times:
            timing_factor = self._assess_timing_consistency(detailed_analysis.arrival_times)
            quality_factors.append(timing_factor)
        
        # Factor 7: Magnitude consistency (if detailed analysis available)
        if detailed_analysis and detailed_analysis.magnitude_estimates:
            magnitude_factor = self._assess_magnitude_consistency(detailed_analysis.magnitude_estimates)
            quality_factors.append(magnitude_factor)
        
        # Calculate weighted average
        overall_quality = np.mean(quality_factors)
        
        return overall_quality
    
    def _normalize_snr_to_quality(self, snr_db: float) -> float:
        """
        Normalize SNR to quality score (0-1).
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Quality score based on SNR
        """
        # SNR quality mapping
        if snr_db >= 20:
            return 1.0  # Excellent
        elif snr_db >= 10:
            return 0.8  # Good
        elif snr_db >= 5:
            return 0.6  # Fair
        elif snr_db >= 0:
            return 0.4  # Poor
        else:
            return 0.2  # Very poor
    
    def _assess_timing_consistency(self, arrival_times) -> float:
        """
        Assess consistency of arrival time calculations.
        
        Args:
            arrival_times: ArrivalTimes object
            
        Returns:
            Timing consistency score (0-1)
        """
        consistency_score = 1.0
        
        # Check P-S time difference consistency
        if (arrival_times.p_wave_arrival is not None and 
            arrival_times.s_wave_arrival is not None):
            
            sp_diff = arrival_times.s_wave_arrival - arrival_times.p_wave_arrival
            
            # S-P time should be positive and reasonable (0.1 to 100 seconds)
            if sp_diff <= 0:
                consistency_score *= 0.5  # Negative S-P time is problematic
            elif sp_diff < 0.1 or sp_diff > 100:
                consistency_score *= 0.7  # Unrealistic S-P time
        
        # Check surface wave timing
        if (arrival_times.surface_wave_arrival is not None and
            arrival_times.s_wave_arrival is not None):
            
            surface_delay = arrival_times.surface_wave_arrival - arrival_times.s_wave_arrival
            
            # Surface waves should arrive after S-waves
            if surface_delay < 0:
                consistency_score *= 0.6
        
        return consistency_score
    
    def _assess_magnitude_consistency(self, magnitude_estimates: List) -> float:
        """
        Assess consistency of magnitude estimates.
        
        Args:
            magnitude_estimates: List of MagnitudeEstimate objects
            
        Returns:
            Magnitude consistency score (0-1)
        """
        if len(magnitude_estimates) < 2:
            return 1.0  # Can't assess consistency with fewer than 2 estimates
        
        # Calculate standard deviation of magnitude estimates
        magnitudes = [est.magnitude for est in magnitude_estimates]
        mag_std = np.std(magnitudes)
        
        # Good consistency: std < 0.5, Fair: std < 1.0, Poor: std >= 1.0
        if mag_std < 0.5:
            return 1.0
        elif mag_std < 1.0:
            return 0.7
        else:
            return 0.4
    
    def _generate_quality_warnings(self, wave_result: WaveAnalysisResult,
                                 snr: float, detection_confidence: float,
                                 data_completeness: float) -> List[str]:
        """
        Generate quality warnings based on analysis results.
        
        Args:
            wave_result: Wave separation result
            snr: Signal-to-noise ratio
            detection_confidence: Average detection confidence
            data_completeness: Data completeness score
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        # SNR warnings
        if snr < self.min_snr_threshold:
            warnings.append(f"Low signal-to-noise ratio: {snr:.1f} dB (minimum recommended: {self.min_snr_threshold} dB)")
        
        # Detection confidence warnings
        if detection_confidence < self.min_confidence_threshold:
            warnings.append(f"Low average detection confidence: {detection_confidence:.2f} (minimum recommended: {self.min_confidence_threshold})")
        
        # Data completeness warnings
        if data_completeness < 0.9:
            warnings.append(f"Data completeness issues detected: {data_completeness:.2f}")
        
        # Wave detection warnings
        if wave_result.total_waves_detected == 0:
            warnings.append("No waves detected in the data")
        elif wave_result.total_waves_detected > 20:
            warnings.append(f"Unusually high number of wave detections: {wave_result.total_waves_detected} (possible noise)")
        
        # Wave type warnings
        if len(wave_result.wave_types_detected) < 2:
            warnings.append("Limited wave types detected - analysis may be incomplete")
        
        # P-wave specific warnings
        if not wave_result.p_waves:
            warnings.append("No P-waves detected - primary wave analysis unavailable")
        
        # S-wave specific warnings
        if not wave_result.s_waves:
            warnings.append("No S-waves detected - secondary wave analysis unavailable")
        
        return warnings
    
    def validate_wave_detection_results(self, wave_result: WaveAnalysisResult) -> Dict[str, bool]:
        """
        Validate wave detection results for consistency and quality.
        
        Args:
            wave_result: Wave separation result to validate
            
        Returns:
            Dictionary with validation results for different aspects
        """
        validation_results = {}
        
        # Validate P-wave detections
        validation_results['p_waves_valid'] = self._validate_p_waves(wave_result.p_waves)
        
        # Validate S-wave detections
        validation_results['s_waves_valid'] = self._validate_s_waves(wave_result.s_waves)
        
        # Validate surface wave detections
        validation_results['surface_waves_valid'] = self._validate_surface_waves(wave_result.surface_waves)
        
        # Validate timing relationships
        validation_results['timing_valid'] = self._validate_wave_timing(wave_result)
        
        # Validate amplitude relationships
        validation_results['amplitudes_valid'] = self._validate_wave_amplitudes(wave_result)
        
        # Overall validation
        validation_results['overall_valid'] = all(validation_results.values())
        
        return validation_results
    
    def _validate_p_waves(self, p_waves: List[WaveSegment]) -> bool:
        """Validate P-wave detections."""
        if not p_waves:
            return True  # No P-waves is acceptable
        
        for wave in p_waves:
            # Check frequency range (P-waves typically 1-15 Hz)
            if wave.dominant_frequency < 0.5 or wave.dominant_frequency > 20:
                return False
            
            # Check duration (P-waves typically short, < 10 seconds)
            if wave.duration > 15:
                return False
            
            # Check confidence
            if wave.confidence < 0.1:
                return False
        
        return True
    
    def _validate_s_waves(self, s_waves: List[WaveSegment]) -> bool:
        """Validate S-wave detections."""
        if not s_waves:
            return True  # No S-waves is acceptable
        
        for wave in s_waves:
            # Check frequency range (S-waves typically 0.5-10 Hz)
            if wave.dominant_frequency < 0.2 or wave.dominant_frequency > 15:
                return False
            
            # Check duration (S-waves typically longer than P-waves)
            if wave.duration > 30:
                return False
            
            # Check confidence
            if wave.confidence < 0.1:
                return False
        
        return True
    
    def _validate_surface_waves(self, surface_waves: List[WaveSegment]) -> bool:
        """Validate surface wave detections."""
        if not surface_waves:
            return True  # No surface waves is acceptable
        
        for wave in surface_waves:
            # Check frequency range (Surface waves typically 0.02-0.5 Hz)
            if wave.dominant_frequency < 0.01 or wave.dominant_frequency > 2:
                return False
            
            # Check duration (Surface waves typically long duration)
            if wave.duration < 5:
                return False
            
            # Check wave type
            if wave.wave_type not in ['Love', 'Rayleigh']:
                return False
            
            # Check confidence
            if wave.confidence < 0.1:
                return False
        
        return True
    
    def _validate_wave_timing(self, wave_result: WaveAnalysisResult) -> bool:
        """Validate timing relationships between wave types."""
        # Get earliest arrival times for each wave type
        p_arrival = None
        s_arrival = None
        surface_arrival = None
        
        if wave_result.p_waves:
            p_arrival = min(wave.arrival_time for wave in wave_result.p_waves)
        
        if wave_result.s_waves:
            s_arrival = min(wave.arrival_time for wave in wave_result.s_waves)
        
        if wave_result.surface_waves:
            surface_arrival = min(wave.arrival_time for wave in wave_result.surface_waves)
        
        # Check timing order: P < S < Surface
        if p_arrival is not None and s_arrival is not None:
            if s_arrival <= p_arrival:
                return False  # S-waves should arrive after P-waves
        
        if s_arrival is not None and surface_arrival is not None:
            if surface_arrival < s_arrival:
                return False  # Surface waves should arrive after S-waves
        
        if p_arrival is not None and surface_arrival is not None:
            if surface_arrival <= p_arrival:
                return False  # Surface waves should arrive after P-waves
        
        return True
    
    def _validate_wave_amplitudes(self, wave_result: WaveAnalysisResult) -> bool:
        """Validate amplitude relationships between wave types."""
        # This is a simplified validation - in practice, amplitude relationships
        # depend on many factors including distance, magnitude, and local geology
        
        # Get maximum amplitudes for each wave type
        p_max_amp = 0
        s_max_amp = 0
        surface_max_amp = 0
        
        if wave_result.p_waves:
            p_max_amp = max(wave.peak_amplitude for wave in wave_result.p_waves)
        
        if wave_result.s_waves:
            s_max_amp = max(wave.peak_amplitude for wave in wave_result.s_waves)
        
        if wave_result.surface_waves:
            surface_max_amp = max(wave.peak_amplitude for wave in wave_result.surface_waves)
        
        # Basic amplitude checks (these are very loose constraints)
        # S-waves are often larger than P-waves
        if p_max_amp > 0 and s_max_amp > 0:
            if p_max_amp > s_max_amp * 5:  # P-waves shouldn't be much larger than S-waves
                return False
        
        # Surface waves are often the largest
        if surface_max_amp > 0 and (p_max_amp > 0 or s_max_amp > 0):
            max_body_wave = max(p_max_amp, s_max_amp)
            if max_body_wave > surface_max_amp * 10:  # Body waves shouldn't be much larger than surface waves
                return False
        
        return True
    
    def assess_data_quality_for_analysis(self, data: np.ndarray) -> Dict[str, float]:
        """
        Assess raw data quality before analysis.
        
        Args:
            data: Raw seismic time series data
            
        Returns:
            Dictionary with data quality metrics
        """
        quality_metrics = {}
        
        # Basic statistics
        quality_metrics['mean'] = np.mean(data)
        quality_metrics['std'] = np.std(data)
        quality_metrics['min'] = np.min(data)
        quality_metrics['max'] = np.max(data)
        quality_metrics['range'] = quality_metrics['max'] - quality_metrics['min']
        
        # Data completeness
        quality_metrics['finite_fraction'] = np.sum(np.isfinite(data)) / len(data)
        
        # Dynamic range
        if quality_metrics['std'] > 0:
            quality_metrics['dynamic_range'] = quality_metrics['range'] / quality_metrics['std']
        else:
            quality_metrics['dynamic_range'] = 0
        
        # Signal variability
        diff_data = np.diff(data)
        quality_metrics['variability'] = np.std(diff_data) / (quality_metrics['std'] + 1e-10)
        
        # Potential clipping detection
        threshold = 0.95 * quality_metrics['range']
        clipped_samples = np.sum((np.abs(data - quality_metrics['mean']) > threshold))
        quality_metrics['clipping_fraction'] = clipped_samples / len(data)
        
        return quality_metrics
    
    def set_parameters(self, **kwargs):
        """
        Set quality assessment parameters.
        
        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'noise_window_fraction' in kwargs:
            self.noise_window_fraction = kwargs['noise_window_fraction']
        if 'min_snr_threshold' in kwargs:
            self.min_snr_threshold = kwargs['min_snr_threshold']
        if 'min_confidence_threshold' in kwargs:
            self.min_confidence_threshold = kwargs['min_confidence_threshold']
        if 'quality_thresholds' in kwargs:
            self.quality_thresholds.update(kwargs['quality_thresholds'])