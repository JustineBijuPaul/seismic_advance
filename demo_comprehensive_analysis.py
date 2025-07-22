#!/usr/bin/env python3
"""
Demonstration of the comprehensive wave analysis service.

This script demonstrates how to use the WaveAnalyzer and QualityMetricsCalculator
to perform complete earthquake wave analysis with quality assessment.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from wave_analysis.services.wave_analyzer import WaveAnalyzer
from wave_analysis.services.quality_metrics import QualityMetricsCalculator
from wave_analysis.models import WaveSegment, WaveAnalysisResult


def create_synthetic_earthquake_data(duration=120.0, sampling_rate=100.0):
    """Create synthetic earthquake data for demonstration."""
    t = np.linspace(0, duration, int(duration * sampling_rate))
    signal = np.zeros_like(t)
    
    # Background noise
    noise = 0.05 * np.random.normal(0, 1, len(signal))
    
    # P-wave arrival at 15 seconds
    p_start = 15.0
    p_duration = 3.0
    p_mask = (t >= p_start) & (t <= p_start + p_duration)
    p_envelope = np.exp(-2 * (t[p_mask] - p_start))
    signal[p_mask] += 1.5 * np.sin(2 * np.pi * 10 * (t[p_mask] - p_start)) * p_envelope
    
    # S-wave arrival at 28 seconds
    s_start = 28.0
    s_duration = 8.0
    s_mask = (t >= s_start) & (t <= s_start + s_duration)
    s_envelope = np.exp(-1 * (t[s_mask] - s_start))
    signal[s_mask] += 2.5 * np.sin(2 * np.pi * 5 * (t[s_mask] - s_start)) * s_envelope
    
    # Love wave arrival at 45 seconds
    love_start = 45.0
    love_duration = 25.0
    love_mask = (t >= love_start) & (t <= love_start + love_duration)
    love_envelope = np.exp(-0.15 * (t[love_mask] - love_start))
    signal[love_mask] += 3.0 * np.sin(2 * np.pi * 0.8 * (t[love_mask] - love_start)) * love_envelope
    
    # Rayleigh wave arrival at 50 seconds
    rayleigh_start = 50.0
    rayleigh_duration = 30.0
    rayleigh_mask = (t >= rayleigh_start) & (t <= rayleigh_start + rayleigh_duration)
    rayleigh_envelope = np.exp(-0.12 * (t[rayleigh_mask] - rayleigh_start))
    signal[rayleigh_mask] += 2.8 * np.sin(2 * np.pi * 0.6 * (t[rayleigh_mask] - rayleigh_start)) * rayleigh_envelope
    
    # Add noise
    signal += noise
    
    return signal, t, sampling_rate


def create_wave_analysis_result(data, sampling_rate):
    """Create a WaveAnalysisResult from synthetic data."""
    # Create realistic wave segments based on the synthetic data
    p_waves = [
        WaveSegment(
            wave_type='P',
            start_time=15.0,
            end_time=18.0,
            data=data[1500:1800],
            sampling_rate=sampling_rate,
            peak_amplitude=1.5,
            dominant_frequency=10.0,
            arrival_time=15.2,
            confidence=0.85
        )
    ]
    
    s_waves = [
        WaveSegment(
            wave_type='S',
            start_time=28.0,
            end_time=36.0,
            data=data[2800:3600],
            sampling_rate=sampling_rate,
            peak_amplitude=2.5,
            dominant_frequency=5.0,
            arrival_time=28.3,
            confidence=0.78
        )
    ]
    
    surface_waves = [
        WaveSegment(
            wave_type='Love',
            start_time=45.0,
            end_time=70.0,
            data=data[4500:7000],
            sampling_rate=sampling_rate,
            peak_amplitude=3.0,
            dominant_frequency=0.8,
            arrival_time=45.5,
            confidence=0.82
        ),
        WaveSegment(
            wave_type='Rayleigh',
            start_time=50.0,
            end_time=80.0,
            data=data[5000:8000],
            sampling_rate=sampling_rate,
            peak_amplitude=2.8,
            dominant_frequency=0.6,
            arrival_time=50.2,
            confidence=0.79
        )
    ]
    
    return WaveAnalysisResult(
        original_data=data,
        sampling_rate=sampling_rate,
        p_waves=p_waves,
        s_waves=s_waves,
        surface_waves=surface_waves,
        metadata={
            'station': 'DEMO_STATION',
            'location': {'lat': 40.0, 'lon': -120.0},
            'event_time': datetime.now().isoformat()
        }
    )


def demonstrate_comprehensive_analysis():
    """Demonstrate the comprehensive wave analysis workflow."""
    print("=== Comprehensive Wave Analysis Demonstration ===\n")
    
    # Create synthetic earthquake data
    print("1. Creating synthetic earthquake data...")
    data, time_axis, sampling_rate = create_synthetic_earthquake_data()
    wave_result = create_wave_analysis_result(data, sampling_rate)
    
    print(f"   - Data duration: {len(data)/sampling_rate:.1f} seconds")
    print(f"   - Sampling rate: {sampling_rate} Hz")
    print(f"   - Detected waves: {wave_result.total_waves_detected}")
    print(f"   - Wave types: {', '.join(wave_result.wave_types_detected)}")
    
    # Initialize analyzers
    print("\n2. Initializing analysis components...")
    wave_analyzer = WaveAnalyzer(sampling_rate)
    quality_calculator = QualityMetricsCalculator(sampling_rate)
    
    # Perform comprehensive wave analysis
    print("\n3. Performing comprehensive wave analysis...")
    detailed_analysis = wave_analyzer.analyze_waves(wave_result)
    
    # Display arrival time results
    print("\n   Arrival Time Analysis:")
    arrival_times = detailed_analysis.arrival_times
    if arrival_times.p_wave_arrival:
        print(f"   - P-wave arrival: {arrival_times.p_wave_arrival:.2f} seconds")
    if arrival_times.s_wave_arrival:
        print(f"   - S-wave arrival: {arrival_times.s_wave_arrival:.2f} seconds")
    if arrival_times.surface_wave_arrival:
        print(f"   - Surface wave arrival: {arrival_times.surface_wave_arrival:.2f} seconds")
    if arrival_times.sp_time_difference:
        print(f"   - S-P time difference: {arrival_times.sp_time_difference:.2f} seconds")
    
    # Display magnitude estimates
    print("\n   Magnitude Estimates:")
    for mag_est in detailed_analysis.magnitude_estimates:
        print(f"   - {mag_est.method}: {mag_est.magnitude:.2f} (confidence: {mag_est.confidence:.2f})")
    
    # Display epicenter distance
    if detailed_analysis.epicenter_distance:
        print(f"\n   Epicenter Distance: {detailed_analysis.epicenter_distance:.1f} km")
    
    # Display frequency analysis
    print("\n   Frequency Analysis:")
    for wave_type, freq_data in detailed_analysis.frequency_analysis.items():
        print(f"   - {wave_type}-wave dominant frequency: {freq_data.dominant_frequency:.2f} Hz")
        print(f"     Spectral centroid: {freq_data.spectral_centroid:.2f} Hz")
        print(f"     Bandwidth: {freq_data.bandwidth:.2f} Hz")
    
    # Perform quality assessment
    print("\n4. Performing quality assessment...")
    quality_metrics = quality_calculator.calculate_quality_metrics(wave_result, detailed_analysis)
    
    print(f"\n   Quality Metrics:")
    print(f"   - Signal-to-noise ratio: {quality_metrics.signal_to_noise_ratio:.1f} dB")
    print(f"   - Detection confidence: {quality_metrics.detection_confidence:.2f}")
    print(f"   - Analysis quality score: {quality_metrics.analysis_quality_score:.2f}")
    print(f"   - Data completeness: {quality_metrics.data_completeness:.2f}")
    
    if quality_metrics.processing_warnings:
        print(f"\n   Quality Warnings:")
        for warning in quality_metrics.processing_warnings:
            print(f"   - {warning}")
    else:
        print(f"\n   No quality warnings - excellent data quality!")
    
    # Validate wave detection results
    print("\n5. Validating wave detection results...")
    validation_results = quality_calculator.validate_wave_detection_results(wave_result)
    
    print(f"\n   Validation Results:")
    for validation_type, is_valid in validation_results.items():
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"   - {validation_type.replace('_', ' ').title()}: {status}")
    
    # Individual wave type analysis
    print("\n6. Individual wave type analysis...")
    for wave_type in ['P', 'S', 'Love']:
        if wave_type == 'P' and wave_result.p_waves:
            characteristics = wave_analyzer.analyze_single_wave_type(wave_type, wave_result.p_waves)
        elif wave_type == 'S' and wave_result.s_waves:
            characteristics = wave_analyzer.analyze_single_wave_type(wave_type, wave_result.s_waves)
        elif wave_type == 'Love':
            love_waves = [w for w in wave_result.surface_waves if w.wave_type == 'Love']
            if love_waves:
                characteristics = wave_analyzer.analyze_single_wave_type(wave_type, love_waves)
            else:
                continue
        else:
            continue
        
        print(f"\n   {wave_type}-wave Characteristics:")
        print(f"   - Arrival time: {characteristics['arrival_time']:.2f} seconds")
        print(f"   - Peak amplitude: {characteristics['peak_amplitude']:.2f}")
        print(f"   - Duration: {characteristics['duration']:.2f} seconds")
        print(f"   - Dominant frequency: {characteristics['dominant_frequency']:.2f} Hz")
        print(f"   - Confidence: {characteristics['confidence']:.2f}")
    
    # Summary
    print(f"\n=== Analysis Summary ===")
    print(f"Overall Analysis Quality: {quality_metrics.analysis_quality_score:.2f}/1.0")
    if quality_metrics.analysis_quality_score >= 0.8:
        print("Excellent quality analysis - results are highly reliable")
    elif quality_metrics.analysis_quality_score >= 0.6:
        print("Good quality analysis - results are reliable")
    elif quality_metrics.analysis_quality_score >= 0.4:
        print("Fair quality analysis - results should be used with caution")
    else:
        print("Poor quality analysis - results may not be reliable")
    
    print(f"\nBest magnitude estimate: {detailed_analysis.best_magnitude_estimate.magnitude:.2f} ({detailed_analysis.best_magnitude_estimate.method})")
    print(f"Analysis completeness: {'Complete' if detailed_analysis.has_complete_analysis else 'Partial'}")
    
    return detailed_analysis, quality_metrics


if __name__ == "__main__":
    try:
        detailed_analysis, quality_metrics = demonstrate_comprehensive_analysis()
        print(f"\n✓ Demonstration completed successfully!")
    except Exception as e:
        print(f"\n✗ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()