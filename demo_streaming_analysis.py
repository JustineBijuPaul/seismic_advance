#!/usr/bin/env python3
"""
Demonstration of streaming wave analysis capabilities.

This script shows how to use the StreamingAnalyzer for real-time
seismic data processing and wave detection.
"""

import numpy as np
import time
import logging
from datetime import datetime

from wave_analysis import StreamingAnalyzer, StreamingAnalysisResult
from wave_analysis.interfaces import WaveDetectorInterface, WaveAnalyzerInterface
from wave_analysis.models.wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, 
    ArrivalTimes, MagnitudeEstimate, QualityMetrics
)


class DemoWaveDetector(WaveDetectorInterface):
    """Demo wave detector for demonstration purposes."""
    
    def __init__(self, wave_type: str):
        self.wave_type = wave_type
        self.parameters = {}
    
    def detect_waves(self, data: np.ndarray, sampling_rate: float, metadata=None):
        """Simple wave detection based on amplitude thresholds."""
        waves = []
        
        # Simple threshold-based detection
        threshold = np.std(data) * 2.5
        peaks = np.where(np.abs(data) > threshold)[0]
        
        if len(peaks) > 10:  # Need sufficient peaks for a wave
            # Group consecutive peaks into wave segments
            segments = []
            current_segment = [peaks[0]]
            
            for peak in peaks[1:]:
                if peak - current_segment[-1] < sampling_rate * 2:  # Within 2 seconds
                    current_segment.append(peak)
                else:
                    if len(current_segment) > 5:  # Minimum segment size
                        segments.append(current_segment)
                    current_segment = [peak]
            
            if len(current_segment) > 5:
                segments.append(current_segment)
            
            # Create wave segments
            for i, segment in enumerate(segments[:3]):  # Max 3 waves per analysis
                start_idx = segment[0]
                end_idx = segment[-1]
                
                wave_data = data[start_idx:end_idx+1]
                
                wave = WaveSegment(
                    wave_type=self.wave_type,
                    start_time=start_idx / sampling_rate,
                    end_time=end_idx / sampling_rate,
                    data=wave_data,
                    sampling_rate=sampling_rate,
                    peak_amplitude=np.max(np.abs(wave_data)),
                    dominant_frequency=self._estimate_frequency(wave_data, sampling_rate),
                    arrival_time=start_idx / sampling_rate,
                    confidence=min(0.9, len(segment) / 20.0)  # Confidence based on segment size
                )
                waves.append(wave)
        
        return waves
    
    def _estimate_frequency(self, data: np.ndarray, sampling_rate: float) -> float:
        """Estimate dominant frequency using FFT."""
        if len(data) < 4:
            return 1.0
        
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
        
        # Find peak frequency (excluding DC component)
        magnitude = np.abs(fft[1:len(fft)//2])
        if len(magnitude) > 0:
            peak_idx = np.argmax(magnitude)
            return abs(freqs[peak_idx + 1])
        return 1.0
    
    def get_wave_type(self) -> str:
        return self.wave_type
    
    def set_parameters(self, parameters: dict) -> None:
        self.parameters.update(parameters)


class DemoWaveAnalyzer(WaveAnalyzerInterface):
    """Demo wave analyzer for demonstration purposes."""
    
    def analyze_waves(self, wave_result: WaveAnalysisResult) -> DetailedAnalysis:
        """Perform basic wave analysis."""
        # Calculate arrival times
        p_arrival = None
        s_arrival = None
        surface_arrival = None
        
        if wave_result.p_waves:
            p_arrival = min(wave.arrival_time for wave in wave_result.p_waves)
        
        if wave_result.s_waves:
            s_arrival = min(wave.arrival_time for wave in wave_result.s_waves)
        
        if wave_result.surface_waves:
            surface_arrival = min(wave.arrival_time for wave in wave_result.surface_waves)
        
        arrival_times = ArrivalTimes(
            p_wave_arrival=p_arrival,
            s_wave_arrival=s_arrival,
            surface_wave_arrival=surface_arrival
        )
        
        # Estimate magnitude based on peak amplitudes
        magnitude_estimates = []
        all_waves = wave_result.p_waves + wave_result.s_waves + wave_result.surface_waves
        
        if all_waves:
            max_amplitude = max(wave.peak_amplitude for wave in all_waves)
            # Simple magnitude estimation: log scale based on amplitude
            estimated_magnitude = np.log10(max_amplitude * 1000) + 1.0
            estimated_magnitude = max(1.0, min(8.0, estimated_magnitude))  # Clamp to reasonable range
            
            magnitude_estimates.append(MagnitudeEstimate(
                method='ML',
                magnitude=estimated_magnitude,
                confidence=0.7,
                wave_type_used='P' if wave_result.p_waves else 'S'
            ))
        
        # Calculate quality metrics
        total_waves = len(all_waves)
        avg_confidence = np.mean([wave.confidence for wave in all_waves]) if all_waves else 0.0
        
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=10.0,  # Simplified
            detection_confidence=avg_confidence,
            analysis_quality_score=min(1.0, total_waves / 3.0),  # Better with more wave types
            data_completeness=1.0
        )
        
        return DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates,
            quality_metrics=quality_metrics
        )
    
    def calculate_arrival_times(self, waves: dict) -> dict:
        """Calculate arrival times for wave types."""
        arrival_times = {}
        for wave_type, wave_list in waves.items():
            if wave_list:
                arrival_times[wave_type] = min(wave.arrival_time for wave in wave_list)
        return arrival_times
    
    def estimate_magnitude(self, waves: dict) -> list:
        """Estimate magnitude from wave characteristics."""
        all_waves = []
        for wave_list in waves.values():
            all_waves.extend(wave_list)
        
        if not all_waves:
            return []
        
        max_amplitude = max(wave.peak_amplitude for wave in all_waves)
        magnitude = np.log10(max_amplitude * 1000) + 1.0
        magnitude = max(1.0, min(8.0, magnitude))
        
        return [{'method': 'ML', 'magnitude': magnitude, 'confidence': 0.7}]


def create_synthetic_earthquake_signal(duration: float, sampling_rate: float, 
                                     magnitude: float = 4.0) -> np.ndarray:
    """Create synthetic earthquake signal for demonstration."""
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Background noise
    noise = np.random.randn(n_samples) * 0.1
    
    # P-wave (high frequency, early arrival)
    p_start = duration * 0.2
    p_mask = t >= p_start
    p_wave = np.where(
        p_mask,
        np.sin(2 * np.pi * 8 * (t - p_start)) * np.exp(-(t - p_start) * 2) * magnitude * 0.5,
        0
    )
    
    # S-wave (medium frequency, later arrival)
    s_start = duration * 0.4
    s_mask = t >= s_start
    s_wave = np.where(
        s_mask,
        np.sin(2 * np.pi * 4 * (t - s_start)) * np.exp(-(t - s_start) * 1.5) * magnitude * 0.8,
        0
    )
    
    # Surface wave (low frequency, latest arrival)
    surf_start = duration * 0.7
    surf_mask = t >= surf_start
    surf_wave = np.where(
        surf_mask,
        np.sin(2 * np.pi * 1.5 * (t - surf_start)) * np.exp(-(t - surf_start) * 1) * magnitude * 1.2,
        0
    )
    
    return noise + p_wave + s_wave + surf_wave


def main():
    """Main demonstration function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=== Streaming Wave Analysis Demonstration ===\n")
    
    # Create demo detectors and analyzer
    detectors = {
        'P': DemoWaveDetector('P'),
        'S': DemoWaveDetector('S'),
        'Surface': DemoWaveDetector('Love')
    }
    
    analyzer = DemoWaveAnalyzer()
    
    # Create streaming analyzer
    streaming_analyzer = StreamingAnalyzer(
        wave_detectors=detectors,
        wave_analyzer=analyzer,
        buffer_size_seconds=30.0,
        overlap_seconds=5.0,
        sampling_rate=100.0,
        analysis_interval=3.0,
        min_detection_threshold=0.3
    )
    
    # Set up event callbacks
    def on_analysis_result(result: StreamingAnalysisResult):
        print(f"\n--- Analysis Result at {result.analysis_timestamp.strftime('%H:%M:%S')} ---")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        if result.wave_result:
            total_waves = result.wave_result.total_waves_detected
            wave_types = result.wave_result.wave_types_detected
            print(f"Detected {total_waves} waves: {', '.join(wave_types)}")
            
            if result.detailed_analysis:
                analysis = result.detailed_analysis
                
                # Show arrival times
                if analysis.arrival_times.p_wave_arrival:
                    print(f"P-wave arrival: {analysis.arrival_times.p_wave_arrival:.2f}s")
                if analysis.arrival_times.s_wave_arrival:
                    print(f"S-wave arrival: {analysis.arrival_times.s_wave_arrival:.2f}s")
                if analysis.arrival_times.sp_time_difference:
                    print(f"S-P time: {analysis.arrival_times.sp_time_difference:.2f}s")
                
                # Show magnitude estimates
                if analysis.magnitude_estimates:
                    best_mag = analysis.best_magnitude_estimate
                    print(f"Estimated magnitude: {best_mag.magnitude:.1f} ({best_mag.method})")
                
                # Show quality metrics
                if analysis.quality_metrics:
                    quality = analysis.quality_metrics
                    print(f"Detection confidence: {quality.detection_confidence:.2f}")
                    print(f"Analysis quality: {quality.analysis_quality_score:.2f}")
        else:
            print("No significant waves detected")
        
        if result.alerts_triggered:
            print(f"ALERTS: {', '.join(result.alerts_triggered)}")
    
    def on_alert(alert: str, result: StreamingAnalysisResult):
        print(f"\n🚨 ALERT: {alert}")
    
    streaming_analyzer.add_event_callback(on_analysis_result)
    streaming_analyzer.add_alert_callback(on_alert)
    
    print("Starting streaming analysis...")
    print("Simulating real-time seismic data with synthetic earthquakes...\n")
    
    # Start streaming
    streaming_analyzer.start_streaming()
    
    try:
        # Simulate different earthquake scenarios
        scenarios = [
            {"magnitude": 3.2, "duration": 20, "description": "Small local earthquake"},
            {"magnitude": 4.5, "duration": 25, "description": "Moderate regional earthquake"},
            {"magnitude": 5.8, "duration": 30, "description": "Strong distant earthquake"},
        ]
        
        for i, scenario in enumerate(scenarios):
            print(f"Scenario {i+1}: {scenario['description']} (M{scenario['magnitude']})")
            
            # Generate synthetic earthquake data
            earthquake_data = create_synthetic_earthquake_signal(
                duration=scenario['duration'],
                sampling_rate=100.0,
                magnitude=scenario['magnitude']
            )
            
            # Stream data in chunks to simulate real-time acquisition
            chunk_size = 200  # 2 seconds of data at 100 Hz
            for j in range(0, len(earthquake_data), chunk_size):
                chunk = earthquake_data[j:j+chunk_size]
                streaming_analyzer.add_data(chunk)
                time.sleep(0.2)  # Simulate real-time delay
            
            print(f"Completed streaming scenario {i+1}")
            time.sleep(2.0)  # Wait between scenarios
        
        # Wait for final analysis
        print("\nWaiting for final analysis...")
        time.sleep(5.0)
        
    finally:
        streaming_analyzer.stop_streaming()
    
    # Show final statistics
    print("\n=== Final Statistics ===")
    stats = streaming_analyzer.get_processing_stats()
    print(f"Total analyses performed: {stats['total_analyses']}")
    print(f"Successful analyses: {stats['successful_analyses']}")
    print(f"Success rate: {stats['success_rate']:.1%}")
    print(f"Average processing time: {stats['average_processing_time']:.3f}s")
    
    # Show recent results summary
    recent_results = streaming_analyzer.get_recent_results(count=5)
    print(f"\nRecent results ({len(recent_results)} shown):")
    for i, result in enumerate(recent_results[-3:], 1):  # Show last 3
        waves_detected = result.wave_result.total_waves_detected if result.wave_result else 0
        print(f"  {i}. {result.analysis_timestamp.strftime('%H:%M:%S')} - "
              f"{waves_detected} waves, {result.processing_time:.3f}s")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()