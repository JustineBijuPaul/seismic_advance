"""
Streaming data analysis for real-time wave detection.

This module provides continuous analysis capabilities for streaming seismic data
with buffer management and sliding window processing for real-time monitoring.
"""

import numpy as np
import threading
import time
from collections import deque
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from ..models.wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, QualityMetrics
)
from ..interfaces import WaveDetectorInterface, WaveAnalyzerInterface


@dataclass
class StreamingBuffer:
    """
    Circular buffer for streaming seismic data with overlap management.
    """
    buffer_size: int  # Maximum buffer size in samples
    overlap_size: int  # Overlap between windows in samples
    sampling_rate: float  # Sampling rate in Hz
    data: deque = field(default_factory=deque)  # Circular buffer for data
    timestamps: deque = field(default_factory=deque)  # Timestamps for each sample
    buffer_lock: threading.Lock = field(default_factory=threading.Lock)
    
    def __post_init__(self):
        """Initialize buffer with validation."""
        if self.buffer_size <= 0:
            raise ValueError("Buffer size must be positive")
        if self.overlap_size >= self.buffer_size:
            raise ValueError("Overlap size must be less than buffer size")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")
    
    def add_data(self, new_data: np.ndarray, timestamp: datetime = None) -> None:
        """
        Add new data to the streaming buffer.
        
        Args:
            new_data: New seismic data samples
            timestamp: Timestamp for the first sample (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.buffer_lock:
            # Add new samples to buffer
            for i, sample in enumerate(new_data):
                sample_time = timestamp + timedelta(seconds=i / self.sampling_rate)
                
                self.data.append(sample)
                self.timestamps.append(sample_time)
                
                # Remove old data if buffer is full
                if len(self.data) > self.buffer_size:
                    self.data.popleft()
                    self.timestamps.popleft()
    
    def get_analysis_window(self) -> Tuple[np.ndarray, List[datetime]]:
        """
        Get current analysis window with overlap handling.
        
        Returns:
            Tuple of (data_array, timestamp_list) for analysis
        """
        with self.buffer_lock:
            if len(self.data) < self.buffer_size:
                # Not enough data for full window
                return np.array(list(self.data)), list(self.timestamps)
            
            # Return full buffer for analysis
            return np.array(list(self.data)), list(self.timestamps)
    
    def get_new_data_window(self) -> Tuple[np.ndarray, List[datetime]]:
        """
        Get only the new data since last overlap for processing.
        
        Returns:
            Tuple of (new_data_array, timestamp_list) for new samples
        """
        with self.buffer_lock:
            if len(self.data) < self.overlap_size:
                return np.array([]), []
            
            # Get data beyond the overlap region
            new_data_size = len(self.data) - self.overlap_size
            if new_data_size <= 0:
                return np.array([]), []
            
            new_data = list(self.data)[-new_data_size:]
            new_timestamps = list(self.timestamps)[-new_data_size:]
            
            return np.array(new_data), new_timestamps
    
    @property
    def is_full(self) -> bool:
        """Check if buffer has enough data for analysis."""
        with self.buffer_lock:
            return len(self.data) >= self.buffer_size
    
    @property
    def current_size(self) -> int:
        """Get current number of samples in buffer."""
        with self.buffer_lock:
            return len(self.data)


@dataclass
class StreamingAnalysisResult:
    """
    Result from streaming analysis containing detected events and metadata.
    """
    analysis_timestamp: datetime
    window_start_time: datetime
    window_end_time: datetime
    wave_result: Optional[WaveAnalysisResult] = None
    detailed_analysis: Optional[DetailedAnalysis] = None
    processing_time: float = 0.0  # Processing time in seconds
    buffer_status: Dict[str, Any] = field(default_factory=dict)
    alerts_triggered: List[str] = field(default_factory=list)


class StreamingAnalyzer:
    """
    Real-time streaming analyzer for continuous seismic wave detection.
    
    This class manages continuous analysis of streaming seismic data using
    sliding window processing with configurable buffer management.
    """
    
    def __init__(self, 
                 wave_detectors: Dict[str, WaveDetectorInterface],
                 wave_analyzer: WaveAnalyzerInterface,
                 buffer_size_seconds: float = 60.0,
                 overlap_seconds: float = 10.0,
                 sampling_rate: float = 100.0,
                 analysis_interval: float = 5.0,
                 min_detection_threshold: float = 0.5):
        """
        Initialize streaming analyzer.
        
        Args:
            wave_detectors: Dictionary of wave detectors by type
            wave_analyzer: Wave analyzer for detailed analysis
            buffer_size_seconds: Buffer size in seconds
            overlap_seconds: Overlap between analysis windows in seconds
            sampling_rate: Expected sampling rate in Hz
            analysis_interval: Time between analyses in seconds
            min_detection_threshold: Minimum confidence for wave detection
        """
        self.wave_detectors = wave_detectors
        self.wave_analyzer = wave_analyzer
        self.sampling_rate = sampling_rate
        self.analysis_interval = analysis_interval
        self.min_detection_threshold = min_detection_threshold
        
        # Calculate buffer sizes in samples
        buffer_samples = int(buffer_size_seconds * sampling_rate)
        overlap_samples = int(overlap_seconds * sampling_rate)
        
        # Initialize streaming buffer
        self.buffer = StreamingBuffer(
            buffer_size=buffer_samples,
            overlap_size=overlap_samples,
            sampling_rate=sampling_rate
        )
        
        # Analysis state
        self.is_running = False
        self.analysis_thread = None
        self.last_analysis_time = None
        self.analysis_results = deque(maxlen=100)  # Keep last 100 results
        
        # Event callbacks
        self.event_callbacks: List[Callable[[StreamingAnalysisResult], None]] = []
        self.alert_callbacks: List[Callable[[str, StreamingAnalysisResult], None]] = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.processing_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'average_processing_time': 0.0,
            'last_error': None
        }
    
    def add_event_callback(self, callback: Callable[[StreamingAnalysisResult], None]) -> None:
        """Add callback for analysis events."""
        self.event_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, StreamingAnalysisResult], None]) -> None:
        """Add callback for alert events."""
        self.alert_callbacks.append(callback)
    
    def start_streaming(self) -> None:
        """Start continuous streaming analysis."""
        if self.is_running:
            self.logger.warning("Streaming analysis already running")
            return
        
        self.is_running = True
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.analysis_thread.start()
        self.logger.info("Started streaming analysis")
    
    def stop_streaming(self) -> None:
        """Stop continuous streaming analysis."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5.0)
        self.logger.info("Stopped streaming analysis")
    
    def add_data(self, data: np.ndarray, timestamp: datetime = None) -> None:
        """
        Add new seismic data to the streaming buffer.
        
        Args:
            data: New seismic data samples
            timestamp: Timestamp for the first sample
        """
        self.buffer.add_data(data, timestamp)
    
    def _analysis_loop(self) -> None:
        """Main analysis loop running in separate thread."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check if it's time for analysis
                if (self.last_analysis_time is None or 
                    current_time - self.last_analysis_time >= self.analysis_interval):
                    
                    if self.buffer.is_full:
                        self._perform_analysis()
                        self.last_analysis_time = current_time
                    
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                self.processing_stats['last_error'] = str(e)
                time.sleep(1.0)  # Wait before retrying
    
    def _perform_analysis(self) -> None:
        """Perform wave analysis on current buffer data."""
        start_time = time.time()
        
        try:
            # Get analysis window
            data, timestamps = self.buffer.get_analysis_window()
            
            if len(data) == 0:
                return
            
            # Create analysis result
            result = StreamingAnalysisResult(
                analysis_timestamp=datetime.now(),
                window_start_time=timestamps[0] if timestamps else datetime.now(),
                window_end_time=timestamps[-1] if timestamps else datetime.now()
            )
            
            # Detect waves using all detectors
            all_waves = []
            p_waves = []
            s_waves = []
            surface_waves = []
            
            for detector_type, detector in self.wave_detectors.items():
                try:
                    detected_waves = detector.detect_waves(data, self.sampling_rate)
                    
                    # Filter by confidence threshold
                    filtered_waves = [
                        wave for wave in detected_waves 
                        if wave.confidence >= self.min_detection_threshold
                    ]
                    
                    # Categorize waves
                    for wave in filtered_waves:
                        if wave.wave_type == 'P':
                            p_waves.append(wave)
                        elif wave.wave_type == 'S':
                            s_waves.append(wave)
                        elif wave.wave_type in ['Love', 'Rayleigh']:
                            surface_waves.append(wave)
                    
                    all_waves.extend(filtered_waves)
                    
                except Exception as e:
                    self.logger.error(f"Error in {detector_type} detector: {e}")
            
            # Create wave analysis result if waves detected
            if all_waves:
                wave_result = WaveAnalysisResult(
                    original_data=data,
                    sampling_rate=self.sampling_rate,
                    p_waves=p_waves,
                    s_waves=s_waves,
                    surface_waves=surface_waves,
                    metadata={
                        'streaming_analysis': True,
                        'window_start': result.window_start_time.isoformat(),
                        'window_end': result.window_end_time.isoformat(),
                        'total_detections': len(all_waves)
                    }
                )
                
                result.wave_result = wave_result
                
                # Perform detailed analysis if significant waves detected
                if len(all_waves) >= 2:  # Need multiple waves for detailed analysis
                    try:
                        detailed_analysis = self.wave_analyzer.analyze_waves(wave_result)
                        result.detailed_analysis = detailed_analysis
                        
                        # Check for alerts
                        alerts = self._check_alerts(detailed_analysis)
                        result.alerts_triggered = alerts
                        
                        # Trigger alert callbacks
                        for alert in alerts:
                            for callback in self.alert_callbacks:
                                try:
                                    callback(alert, result)
                                except Exception as e:
                                    self.logger.error(f"Error in alert callback: {e}")
                    
                    except Exception as e:
                        self.logger.error(f"Error in detailed analysis: {e}")
            
            # Update processing statistics and result
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self._update_processing_stats(processing_time, success=True)
            
            # Add buffer status
            result.buffer_status = {
                'buffer_size': self.buffer.current_size,
                'buffer_full': self.buffer.is_full,
                'sampling_rate': self.sampling_rate
            }
            
            # Store result
            self.analysis_results.append(result)
            
            # Trigger event callbacks
            for callback in self.event_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
            
            self.logger.debug(f"Streaming analysis completed in {processing_time:.3f}s, "
                            f"detected {len(all_waves)} waves")
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Error in streaming analysis: {e}")
            self._update_processing_stats(processing_time, success=False)
            
            # Still create a result with error information
            result = StreamingAnalysisResult(
                analysis_timestamp=datetime.now(),
                window_start_time=datetime.now(),
                window_end_time=datetime.now(),
                processing_time=processing_time
            )
            result.buffer_status = {
                'buffer_size': self.buffer.current_size,
                'buffer_full': self.buffer.is_full,
                'sampling_rate': self.sampling_rate,
                'error': str(e)
            }
            self.analysis_results.append(result)
    
    def _check_alerts(self, analysis: DetailedAnalysis) -> List[str]:
        """
        Check analysis results for alert conditions.
        
        Args:
            analysis: Detailed analysis results
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        # Check for significant magnitude
        best_magnitude = analysis.best_magnitude_estimate
        if best_magnitude and best_magnitude.magnitude >= 4.0:
            alerts.append(f"Significant earthquake detected: M{best_magnitude.magnitude:.1f}")
        
        # Check for multiple wave types (indicates strong event)
        wave_types = analysis.wave_result.wave_types_detected
        if len(wave_types) >= 3:
            alerts.append(f"Complex earthquake event: {', '.join(wave_types)} waves detected")
        
        # Check for high confidence detections
        if analysis.quality_metrics and analysis.quality_metrics.detection_confidence >= 0.9:
            alerts.append("High confidence earthquake detection")
        
        return alerts
    
    def _update_processing_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics."""
        self.processing_stats['total_analyses'] += 1
        
        if success:
            self.processing_stats['successful_analyses'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['average_processing_time']
        total_analyses = self.processing_stats['total_analyses']
        
        self.processing_stats['average_processing_time'] = (
            (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        )
    
    def get_recent_results(self, count: int = 10) -> List[StreamingAnalysisResult]:
        """
        Get recent analysis results.
        
        Args:
            count: Number of recent results to return
            
        Returns:
            List of recent analysis results
        """
        return list(self.analysis_results)[-count:]
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats = self.processing_stats.copy()
        stats['success_rate'] = (
            stats['successful_analyses'] / max(stats['total_analyses'], 1)
        )
        stats['is_running'] = self.is_running
        stats['buffer_status'] = {
            'current_size': self.buffer.current_size,
            'is_full': self.buffer.is_full,
            'buffer_capacity': self.buffer.buffer_size
        }
        return stats
    
    def force_analysis(self) -> Optional[StreamingAnalysisResult]:
        """
        Force immediate analysis of current buffer data.
        
        Returns:
            Analysis result if successful, None otherwise
        """
        if not self.buffer.is_full:
            self.logger.warning("Buffer not full, analysis may be incomplete")
        
        try:
            self._perform_analysis()
            return self.analysis_results[-1] if self.analysis_results else None
        except Exception as e:
            self.logger.error(f"Error in forced analysis: {e}")
            return None
    
    def clear_buffer(self) -> None:
        """Clear the streaming buffer."""
        with self.buffer.buffer_lock:
            self.buffer.data.clear()
            self.buffer.timestamps.clear()
        self.logger.info("Streaming buffer cleared")
    
    def set_analysis_parameters(self, **kwargs) -> None:
        """
        Update analysis parameters.
        
        Args:
            **kwargs: Parameter updates (analysis_interval, min_detection_threshold, etc.)
        """
        if 'analysis_interval' in kwargs:
            self.analysis_interval = kwargs['analysis_interval']
        
        if 'min_detection_threshold' in kwargs:
            self.min_detection_threshold = kwargs['min_detection_threshold']
        
        # Update detector parameters if provided
        for detector_name, detector in self.wave_detectors.items():
            detector_params = kwargs.get(f'{detector_name}_params')
            if detector_params:
                detector.set_parameters(detector_params)
        
        self.logger.info(f"Updated analysis parameters: {kwargs}")