"""
Wave Separation Engine

This module implements the main orchestrator for separating seismic waves
into P-waves, S-waves, and surface waves using integrated detection algorithms.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings
import logging
from datetime import datetime

from ..interfaces import WaveDetectorInterface
from ..models import WaveSegment, WaveAnalysisResult, QualityMetrics
from .wave_detectors import (
    PWaveDetector, PWaveDetectionParameters,
    SWaveDetector, SWaveDetectionParameters,
    SurfaceWaveDetector, SurfaceWaveDetectionParameters
)
from .signal_processing import SignalProcessor


@dataclass
class WaveSeparationParameters:
    """Parameters for wave separation engine."""
    sampling_rate: float
    
    # P-wave detection parameters
    p_wave_params: Optional[PWaveDetectionParameters] = None
    
    # S-wave detection parameters  
    s_wave_params: Optional[SWaveDetectionParameters] = None
    
    # Surface wave detection parameters
    surface_wave_params: Optional[SurfaceWaveDetectionParameters] = None
    
    # Quality control parameters
    min_snr: float = 2.0  # Minimum signal-to-noise ratio
    min_detection_confidence: float = 0.3  # Minimum detection confidence
    max_processing_time: float = 300.0  # Maximum processing time in seconds
    
    # Validation parameters
    validate_wave_separation: bool = True
    remove_overlapping_detections: bool = True
    overlap_threshold: float = 0.5  # Maximum allowed overlap fraction
    
    def __post_init__(self):
        """Initialize default parameters if not provided."""
        if self.p_wave_params is None:
            self.p_wave_params = PWaveDetectionParameters(sampling_rate=self.sampling_rate)
        
        if self.s_wave_params is None:
            self.s_wave_params = SWaveDetectionParameters(sampling_rate=self.sampling_rate)
        
        if self.surface_wave_params is None:
            self.surface_wave_params = SurfaceWaveDetectionParameters(sampling_rate=self.sampling_rate)


@dataclass
class WaveSeparationResult:
    """Result of wave separation process with quality metrics."""
    wave_analysis_result: WaveAnalysisResult
    quality_metrics: QualityMetrics
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class WaveSeparationEngine:
    """
    Main orchestrator for separating seismic waves into P, S, and surface waves.
    
    This class integrates multiple wave detection algorithms and provides
    quality control and validation for comprehensive wave separation.
    """
    
    def __init__(self, parameters: WaveSeparationParameters):
        """
        Initialize wave separation engine.
        
        Args:
            parameters: Wave separation parameters
        """
        self.params = parameters
        self.logger = logging.getLogger(__name__)
        
        # Initialize wave detectors
        self.p_wave_detector = PWaveDetector(parameters.p_wave_params)
        self.s_wave_detector = SWaveDetector(parameters.s_wave_params)
        self.surface_wave_detector = SurfaceWaveDetector(parameters.surface_wave_params)
        
        # Initialize signal processor
        self.signal_processor = SignalProcessor(parameters.sampling_rate)
        
        # Processing statistics
        self.processing_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0
        }
    
    def separate_waves(self, seismic_data: np.ndarray, 
                      metadata: Optional[Dict[str, Any]] = None) -> WaveSeparationResult:
        """
        Separate seismic waves into P, S, and surface wave components.
        
        Args:
            seismic_data: Raw seismic time series data
            metadata: Optional metadata about the seismic data
            
        Returns:
            WaveSeparationResult containing separated waves and quality metrics
        """
        start_time = datetime.now()
        processing_metadata = {
            'start_time': start_time,
            'data_length': len(seismic_data),
            'sampling_rate': self.params.sampling_rate
        }
        warnings_list = []
        errors_list = []
        
        try:
            # Validate input data
            self._validate_input_data(seismic_data)
            
            # Preprocess data
            processed_data = self._preprocess_data(seismic_data)
            processing_metadata['preprocessing_applied'] = True
            
            # Calculate data quality metrics
            data_quality = self._assess_data_quality(processed_data)
            processing_metadata['data_quality'] = data_quality
            
            # Check if data quality is sufficient for analysis
            if data_quality['snr'] < self.params.min_snr:
                warnings_list.append(f"Low SNR detected: {data_quality['snr']:.2f}")
            
            # Detect P-waves
            self.logger.info("Starting P-wave detection")
            p_waves = self._detect_p_waves(processed_data, metadata)
            processing_metadata['p_waves_detected'] = len(p_waves)
            
            # Detect S-waves (with P-wave context)
            self.logger.info("Starting S-wave detection")
            s_wave_metadata = self._create_s_wave_metadata(metadata, p_waves)
            s_waves = self._detect_s_waves(processed_data, s_wave_metadata)
            processing_metadata['s_waves_detected'] = len(s_waves)
            
            # Detect surface waves
            self.logger.info("Starting surface wave detection")
            surface_wave_metadata = self._create_surface_wave_metadata(metadata, p_waves, s_waves)
            surface_waves = self._detect_surface_waves(processed_data, surface_wave_metadata)
            processing_metadata['surface_waves_detected'] = len(surface_waves)
            
            # Apply quality control and validation
            if self.params.validate_wave_separation:
                p_waves, s_waves, surface_waves = self._validate_wave_separation(
                    p_waves, s_waves, surface_waves
                )
                processing_metadata['validation_applied'] = True
            
            # Remove overlapping detections if requested
            if self.params.remove_overlapping_detections:
                p_waves, s_waves, surface_waves = self._remove_overlapping_detections(
                    p_waves, s_waves, surface_waves
                )
                processing_metadata['overlap_removal_applied'] = True
            
            # Create wave analysis result
            wave_result = WaveAnalysisResult(
                original_data=seismic_data,
                sampling_rate=self.params.sampling_rate,
                p_waves=p_waves,
                s_waves=s_waves,
                surface_waves=surface_waves,
                metadata=metadata or {}
            )
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                wave_result, data_quality, processing_metadata
            )
            
            # Update processing metadata
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            processing_metadata.update({
                'end_time': end_time,
                'processing_time_seconds': processing_time,
                'success': True
            })
            
            # Update statistics
            self._update_processing_stats(processing_time, success=True)
            
            self.logger.info(f"Wave separation completed successfully in {processing_time:.2f} seconds")
            
            return WaveSeparationResult(
                wave_analysis_result=wave_result,
                quality_metrics=quality_metrics,
                processing_metadata=processing_metadata,
                warnings=warnings_list,
                errors=errors_list
            )
            
        except Exception as e:
            # Handle processing errors
            error_msg = f"Wave separation failed: {str(e)}"
            self.logger.error(error_msg)
            errors_list.append(error_msg)
            
            # Create minimal result for error case
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            processing_metadata.update({
                'end_time': end_time,
                'processing_time_seconds': processing_time,
                'success': False,
                'error': str(e)
            })
            
            # Update statistics
            self._update_processing_stats(processing_time, success=False)
            
            # Create empty wave result
            wave_result = WaveAnalysisResult(
                original_data=seismic_data,
                sampling_rate=self.params.sampling_rate,
                metadata=metadata or {}
            )
            
            # Create minimal quality metrics
            quality_metrics = QualityMetrics(
                signal_to_noise_ratio=0.0,
                detection_confidence=0.0,
                analysis_quality_score=0.0,
                data_completeness=0.0,
                processing_warnings=warnings_list + [error_msg]
            )
            
            return WaveSeparationResult(
                wave_analysis_result=wave_result,
                quality_metrics=quality_metrics,
                processing_metadata=processing_metadata,
                warnings=warnings_list,
                errors=errors_list
            )
    
    def detect_p_waves(self, data: np.ndarray) -> List[WaveSegment]:
        """
        Detect P-waves in seismic data.
        
        Args:
            data: Raw seismic time series data
            
        Returns:
            List of detected P-wave segments
        """
        processed_data = self._preprocess_data(data)
        return self.p_wave_detector.detect_waves(processed_data, self.params.sampling_rate)
    
    def detect_s_waves(self, data: np.ndarray, p_waves: Optional[List[WaveSegment]] = None) -> List[WaveSegment]:
        """
        Detect S-waves in seismic data.
        
        Args:
            data: Raw seismic time series data
            p_waves: Optional P-wave detections for context
            
        Returns:
            List of detected S-wave segments
        """
        processed_data = self._preprocess_data(data)
        metadata = self._create_s_wave_metadata(None, p_waves or [])
        return self.s_wave_detector.detect_waves(processed_data, self.params.sampling_rate, metadata)
    
    def detect_surface_waves(self, data: np.ndarray) -> List[WaveSegment]:
        """
        Detect surface waves in seismic data.
        
        Args:
            data: Raw seismic time series data
            
        Returns:
            List of detected surface wave segments
        """
        processed_data = self._preprocess_data(data)
        return self.surface_wave_detector.detect_waves(processed_data, self.params.sampling_rate)
    
    def _validate_input_data(self, data: np.ndarray) -> None:
        """
        Validate input seismic data.
        
        Args:
            data: Input seismic data
            
        Raises:
            ValueError: If data is invalid
        """
        if len(data) == 0:
            raise ValueError("Input data cannot be empty")
        
        if not np.isfinite(data).all():
            raise ValueError("Input data contains non-finite values")
        
        # Check minimum data length (at least 10 seconds)
        min_samples = int(10 * self.params.sampling_rate)
        if len(data) < min_samples:
            raise ValueError(f"Input data too short: {len(data)} samples, minimum {min_samples}")
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess seismic data for wave detection.
        
        Args:
            data: Raw seismic data
            
        Returns:
            Preprocessed seismic data
        """
        return self.signal_processor.preprocess_seismic_data(data)
    
    def _assess_data_quality(self, data: np.ndarray) -> Dict[str, float]:
        """
        Assess the quality of seismic data.
        
        Args:
            data: Preprocessed seismic data
            
        Returns:
            Dictionary containing quality metrics
        """
        # Calculate signal-to-noise ratio (simple estimate)
        signal_power = np.var(data)
        
        # Estimate noise from first and last 10% of data
        noise_start = data[:len(data)//10]
        noise_end = data[-len(data)//10:]
        noise_power = (np.var(noise_start) + np.var(noise_end)) / 2
        
        snr = signal_power / max(noise_power, 1e-10)
        
        # Calculate data completeness (fraction of non-zero samples)
        completeness = np.count_nonzero(data) / len(data)
        
        # Calculate dynamic range
        dynamic_range = np.max(np.abs(data)) / (np.mean(np.abs(data)) + 1e-10)
        
        return {
            'snr': snr,
            'completeness': completeness,
            'dynamic_range': dynamic_range,
            'max_amplitude': np.max(np.abs(data)),
            'rms_amplitude': np.sqrt(np.mean(data**2))
        }
    
    def _detect_p_waves(self, data: np.ndarray, metadata: Optional[Dict[str, Any]]) -> List[WaveSegment]:
        """Detect P-waves with error handling."""
        try:
            return self.p_wave_detector.detect_waves(data, self.params.sampling_rate, metadata)
        except Exception as e:
            self.logger.warning(f"P-wave detection failed: {e}")
            return []
    
    def _detect_s_waves(self, data: np.ndarray, metadata: Optional[Dict[str, Any]]) -> List[WaveSegment]:
        """Detect S-waves with error handling."""
        try:
            return self.s_wave_detector.detect_waves(data, self.params.sampling_rate, metadata)
        except Exception as e:
            self.logger.warning(f"S-wave detection failed: {e}")
            return []
    
    def _detect_surface_waves(self, data: np.ndarray, metadata: Optional[Dict[str, Any]]) -> List[WaveSegment]:
        """Detect surface waves with error handling."""
        try:
            return self.surface_wave_detector.detect_waves(data, self.params.sampling_rate, metadata)
        except Exception as e:
            self.logger.warning(f"Surface wave detection failed: {e}")
            return []
    
    def _create_s_wave_metadata(self, original_metadata: Optional[Dict[str, Any]], 
                              p_waves: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create metadata for S-wave detection including P-wave context.
        
        Args:
            original_metadata: Original metadata
            p_waves: Detected P-wave segments
            
        Returns:
            Metadata dictionary for S-wave detection
        """
        metadata = original_metadata.copy() if original_metadata else {}
        
        # Add P-wave arrival times
        if p_waves:
            metadata['p_wave_arrivals'] = [wave.arrival_time for wave in p_waves]
            metadata['p_waves'] = p_waves
        
        return metadata
    
    def _create_surface_wave_metadata(self, original_metadata: Optional[Dict[str, Any]],
                                    p_waves: List[WaveSegment], 
                                    s_waves: List[WaveSegment]) -> Dict[str, Any]:
        """
        Create metadata for surface wave detection.
        
        Args:
            original_metadata: Original metadata
            p_waves: Detected P-wave segments
            s_waves: Detected S-wave segments
            
        Returns:
            Metadata dictionary for surface wave detection
        """
        metadata = original_metadata.copy() if original_metadata else {}
        
        # Add body wave context
        if p_waves:
            metadata['p_waves'] = p_waves
        if s_waves:
            metadata['s_waves'] = s_waves
        
        return metadata
    
    def _validate_wave_separation(self, p_waves: List[WaveSegment], 
                                s_waves: List[WaveSegment],
                                surface_waves: List[WaveSegment]) -> Tuple[List[WaveSegment], List[WaveSegment], List[WaveSegment]]:
        """
        Validate wave separation results and filter low-quality detections.
        
        Args:
            p_waves: Detected P-wave segments
            s_waves: Detected S-wave segments  
            surface_waves: Detected surface wave segments
            
        Returns:
            Tuple of validated wave segments
        """
        # Filter by confidence threshold
        validated_p_waves = [w for w in p_waves if w.confidence >= self.params.min_detection_confidence]
        validated_s_waves = [w for w in s_waves if w.confidence >= self.params.min_detection_confidence]
        validated_surface_waves = [w for w in surface_waves if w.confidence >= self.params.min_detection_confidence]
        
        # Additional validation logic can be added here
        # For example: check for reasonable S-P time differences, wave ordering, etc.
        
        return validated_p_waves, validated_s_waves, validated_surface_waves
    
    def _remove_overlapping_detections(self, p_waves: List[WaveSegment],
                                     s_waves: List[WaveSegment],
                                     surface_waves: List[WaveSegment]) -> Tuple[List[WaveSegment], List[WaveSegment], List[WaveSegment]]:
        """
        Remove overlapping wave detections based on confidence scores.
        
        Args:
            p_waves: P-wave segments
            s_waves: S-wave segments
            surface_waves: Surface wave segments
            
        Returns:
            Tuple of non-overlapping wave segments
        """
        all_waves = p_waves + s_waves + surface_waves
        if not all_waves:
            return p_waves, s_waves, surface_waves
        
        # Sort by start time
        all_waves.sort(key=lambda w: w.start_time)
        
        # Remove overlaps
        non_overlapping = []
        for wave in all_waves:
            overlaps = False
            for existing in non_overlapping:
                if self._calculate_overlap(wave, existing) > self.params.overlap_threshold:
                    # Keep the wave with higher confidence
                    if wave.confidence > existing.confidence:
                        non_overlapping.remove(existing)
                        non_overlapping.append(wave)
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(wave)
        
        # Separate back into wave types
        filtered_p_waves = [w for w in non_overlapping if w.wave_type == 'P']
        filtered_s_waves = [w for w in non_overlapping if w.wave_type == 'S']
        filtered_surface_waves = [w for w in non_overlapping if w.wave_type in ['Love', 'Rayleigh']]
        
        return filtered_p_waves, filtered_s_waves, filtered_surface_waves
    
    def _calculate_overlap(self, wave1: WaveSegment, wave2: WaveSegment) -> float:
        """
        Calculate overlap fraction between two wave segments.
        
        Args:
            wave1: First wave segment
            wave2: Second wave segment
            
        Returns:
            Overlap fraction (0-1)
        """
        start_overlap = max(wave1.start_time, wave2.start_time)
        end_overlap = min(wave1.end_time, wave2.end_time)
        
        if end_overlap <= start_overlap:
            return 0.0
        
        overlap_duration = end_overlap - start_overlap
        min_duration = min(wave1.duration, wave2.duration)
        
        return overlap_duration / min_duration if min_duration > 0 else 0.0
    
    def _calculate_quality_metrics(self, wave_result: WaveAnalysisResult,
                                 data_quality: Dict[str, float],
                                 processing_metadata: Dict[str, Any]) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics for wave separation results.
        
        Args:
            wave_result: Wave analysis result
            data_quality: Data quality assessment
            processing_metadata: Processing metadata
            
        Returns:
            Quality metrics object
        """
        # Calculate average detection confidence
        all_waves = wave_result.p_waves + wave_result.s_waves + wave_result.surface_waves
        if all_waves:
            avg_confidence = np.mean([w.confidence for w in all_waves])
        else:
            avg_confidence = 0.0
        
        # Calculate analysis quality score based on multiple factors
        quality_factors = []
        
        # Factor 1: Data quality
        quality_factors.append(min(1.0, data_quality['snr'] / 10.0))  # Normalize SNR
        
        # Factor 2: Detection confidence
        quality_factors.append(avg_confidence)
        
        # Factor 3: Data completeness
        quality_factors.append(data_quality['completeness'])
        
        # Factor 4: Number of detections (more detections can indicate better analysis)
        detection_factor = min(1.0, len(all_waves) / 5.0)  # Normalize to 5 detections
        quality_factors.append(detection_factor)
        
        analysis_quality_score = np.mean(quality_factors)
        
        # Collect processing warnings
        warnings = []
        if data_quality['snr'] < self.params.min_snr:
            warnings.append(f"Low signal-to-noise ratio: {data_quality['snr']:.2f}")
        if avg_confidence < 0.5:
            warnings.append(f"Low average detection confidence: {avg_confidence:.2f}")
        if data_quality['completeness'] < 0.9:
            warnings.append(f"Incomplete data: {data_quality['completeness']:.2f}")
        
        return QualityMetrics(
            signal_to_noise_ratio=data_quality['snr'],
            detection_confidence=avg_confidence,
            analysis_quality_score=analysis_quality_score,
            data_completeness=data_quality['completeness'],
            processing_warnings=warnings
        )
    
    def _update_processing_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics."""
        self.processing_stats['total_analyses'] += 1
        
        if success:
            self.processing_stats['successful_analyses'] += 1
        else:
            self.processing_stats['failed_analyses'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_analyses']
        current_avg = self.processing_stats['average_processing_time']
        self.processing_stats['average_processing_time'] = (
            (current_avg * (total - 1) + processing_time) / total
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return self.processing_stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'average_processing_time': 0.0
        }
    
    def update_parameters(self, new_params: WaveSeparationParameters) -> None:
        """
        Update wave separation parameters.
        
        Args:
            new_params: New wave separation parameters
        """
        self.params = new_params
        
        # Update detector parameters
        self.p_wave_detector = PWaveDetector(new_params.p_wave_params)
        self.s_wave_detector = SWaveDetector(new_params.s_wave_params)
        self.surface_wave_detector = SurfaceWaveDetector(new_params.surface_wave_params)
        
        # Update signal processor
        self.signal_processor = SignalProcessor(new_params.sampling_rate)