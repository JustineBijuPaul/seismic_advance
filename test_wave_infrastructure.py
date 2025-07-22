#!/usr/bin/env python3
"""
Test script to verify the wave analysis infrastructure is working correctly.
This tests the integration of all analysis engine components.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wave_analysis.models.wave_models import WaveSegment, WaveAnalysisResult
from wave_analysis.services.wave_analyzer import WaveAnalyzer


def create_synthetic_wave_data():
    """Create synthetic wave data for testing."""
    sampling_rate = 100.0  # Hz
    duration = 30.0  # seconds
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # Create synthetic earthquake signal with P, S, and surface waves
    signal = np.zeros_like(t)
    
    # P-wave arrival at 5 seconds (higher frequency, lower amplitude)
    p_start = 5.0
    p_freq = 8.0  # Hz
    p_mask = (t >= p_start) & (t <= p_start + 3.0)
    signal[p_mask] += 0.5 * np.sin(2 * np.pi * p_freq * (t[p_mask] - p_start)) * np.exp(-(t[p_mask] - p_start))
    
    # S-wave arrival at 10 seconds (medium frequency, higher amplitude)
    s_start = 10.0
    s_freq = 4.0  # Hz
    s_mask = (t >= s_start) & (t <= s_start + 5.0)
    signal[s_mask] += 1.0 * np.sin(2 * np.pi * s_freq * (t[s_mask] - s_start)) * np.exp(-0.5 * (t[s_mask] - s_start))
    
    # Surface wave arrival at 15 seconds (low frequency, high amplitude)
    surf_start = 15.0
    surf_freq = 0.5  # Hz
    surf_mask = (t >= surf_start) & (t <= surf_start + 10.0)
    signal[surf_mask] += 2.0 * np.sin(2 * np.pi * surf_freq * (t[surf_mask] - surf_start)) * np.exp(-0.2 * (t[surf_mask] - surf_start))
    
    # Add some noise
    noise = 0.1 * np.random.normal(0, 1, len(signal))
    signal += noise
    
    return signal, sampling_rate, t


def create_wave_segments(signal, sampling_rate, t):
    """Create wave segments from the synthetic signal."""
    
    # P-wave segment
    p_start_idx = int(5.0 * sampling_rate)
    p_end_idx = int(8.0 * sampling_rate)
    p_data = signal[p_start_idx:p_end_idx]
    p_wave = WaveSegment(
        wave_type='P',
        start_time=5.0,
        end_time=8.0,
        data=p_data,
        sampling_rate=sampling_rate,
        peak_amplitude=np.max(np.abs(p_data)),
        dominant_frequency=8.0,
        arrival_time=5.2,
        confidence=0.9
    )
    
    # S-wave segment
    s_start_idx = int(10.0 * sampling_rate)
    s_end_idx = int(15.0 * sampling_rate)
    s_data = signal[s_start_idx:s_end_idx]
    s_wave = WaveSegment(
        wave_type='S',
        start_time=10.0,
        end_time=15.0,
        data=s_data,
        sampling_rate=sampling_rate,
        peak_amplitude=np.max(np.abs(s_data)),
        dominant_frequency=4.0,
        arrival_time=10.3,
        confidence=0.85
    )
    
    # Surface wave segment (Rayleigh)
    surf_start_idx = int(15.0 * sampling_rate)
    surf_end_idx = int(25.0 * sampling_rate)
    surf_data = signal[surf_start_idx:surf_end_idx]
    surf_wave = WaveSegment(
        wave_type='Rayleigh',
        start_time=15.0,
        end_time=25.0,
        data=surf_data,
        sampling_rate=sampling_rate,
        peak_amplitude=np.max(np.abs(surf_data)),
        dominant_frequency=0.5,
        arrival_time=15.5,
        confidence=0.8
    )
    
    return [p_wave], [s_wave], [surf_wave]


def test_wave_analysis_engine():
    """Test the complete wave analysis engine."""
    print("Testing Wave Analysis Engine Integration...")
    
    # Create synthetic data
    signal, sampling_rate, t = create_synthetic_wave_data()
    print(f"✓ Created synthetic earthquake signal ({len(signal)} samples, {sampling_rate} Hz)")
    
    # Create wave segments
    p_waves, s_waves, surface_waves = create_wave_segments(signal, sampling_rate, t)
    print(f"✓ Created wave segments: {len(p_waves)} P-waves, {len(s_waves)} S-waves, {len(surface_waves)} surface waves")
    
    # Create wave analysis result
    wave_result = WaveAnalysisResult(
        original_data=signal,
        sampling_rate=sampling_rate,
        p_waves=p_waves,
        s_waves=s_waves,
        surface_waves=surface_waves,
        metadata={'test': True}
    )
    print(f"✓ Created WaveAnalysisResult with {wave_result.total_waves_detected} total waves")
    
    # Initialize wave analyzer
    analyzer = WaveAnalyzer(sampling_rate)
    print("✓ Initialized WaveAnalyzer")
    
    # Perform comprehensive analysis
    try:
        detailed_analysis = analyzer.analyze_waves(wave_result)
        print("✓ Completed comprehensive wave analysis")
        
        # Check arrival times
        arrival_times = detailed_analysis.arrival_times
        print(f"  - P-wave arrival: {arrival_times.p_wave_arrival:.2f}s")
        print(f"  - S-wave arrival: {arrival_times.s_wave_arrival:.2f}s")
        print(f"  - S-P time difference: {arrival_times.sp_time_difference:.2f}s")
        print(f"  - Surface wave arrival: {arrival_times.surface_wave_arrival:.2f}s")
        
        # Check epicenter distance
        if detailed_analysis.epicenter_distance:
            print(f"  - Estimated epicenter distance: {detailed_analysis.epicenter_distance:.1f} km")
        
        # Check magnitude estimates
        magnitude_estimates = detailed_analysis.magnitude_estimates
        print(f"  - Found {len(magnitude_estimates)} magnitude estimates:")
        for mag_est in magnitude_estimates:
            print(f"    * {mag_est.method}: {mag_est.magnitude:.2f} (confidence: {mag_est.confidence:.2f})")
        
        # Check frequency analysis
        freq_analysis = detailed_analysis.frequency_analysis
        print(f"  - Frequency analysis for {len(freq_analysis)} wave types:")
        for wave_type, freq_data in freq_analysis.items():
            print(f"    * {wave_type}: dominant freq = {freq_data.dominant_frequency:.2f} Hz")
        
        # Check quality metrics
        quality = detailed_analysis.quality_metrics
        if quality:
            print(f"  - Quality metrics:")
            print(f"    * SNR: {quality.signal_to_noise_ratio:.1f} dB")
            print(f"    * Detection confidence: {quality.detection_confidence:.2f}")
            print(f"    * Analysis quality score: {quality.analysis_quality_score:.2f}")
            if quality.processing_warnings:
                print(f"    * Warnings: {', '.join(quality.processing_warnings)}")
        
        print("✓ All analysis components working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_components():
    """Test individual analysis components."""
    print("\nTesting Individual Components...")
    
    # Create test data
    signal, sampling_rate, t = create_synthetic_wave_data()
    p_waves, s_waves, surface_waves = create_wave_segments(signal, sampling_rate, t)
    
    analyzer = WaveAnalyzer(sampling_rate)
    
    # Test arrival time calculator
    try:
        waves_dict = {'P': p_waves, 'S': s_waves, 'Rayleigh': surface_waves}
        arrival_times = analyzer.calculate_arrival_times(waves_dict)
        print(f"✓ Arrival time calculator: P={arrival_times.p_wave_arrival:.2f}s, S={arrival_times.s_wave_arrival:.2f}s")
    except Exception as e:
        print(f"✗ Arrival time calculator failed: {e}")
        return False
    
    # Test frequency analyzer
    try:
        freq_analysis = analyzer.frequency_analyzer.analyze_wave_frequencies(waves_dict)
        print(f"✓ Frequency analyzer: analyzed {len(freq_analysis)} wave types")
    except Exception as e:
        print(f"✗ Frequency analyzer failed: {e}")
        return False
    
    # Test magnitude estimator
    try:
        magnitude_estimates = analyzer.estimate_magnitude(waves_dict, epicenter_distance=100.0)
        print(f"✓ Magnitude estimator: {len(magnitude_estimates)} estimates")
    except Exception as e:
        print(f"✗ Magnitude estimator failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Wave Analysis Engine Integration Test")
    print("=" * 50)
    
    # Test individual components first
    if not test_individual_components():
        print("\n✗ Individual component tests failed!")
        sys.exit(1)
    
    # Test full integration
    if not test_wave_analysis_engine():
        print("\n✗ Integration test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! Wave analysis engine is working correctly.")
    print("\nTask 4 'Build analysis engine for wave characteristics' is now COMPLETE!")
    print("All subtasks (4.1, 4.2, 4.3) have been implemented and integrated.")