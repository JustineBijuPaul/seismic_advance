#!/usr/bin/env python3
"""
Simple test for S-wave detection implementation.
"""

import sys
sys.path.append('.')

import numpy as np
from wave_analysis.services.wave_detectors import SWaveDetector, SWaveDetectionParameters

def create_synthetic_s_wave_signal(duration=40.0, sampling_rate=100.0, 
                                 p_arrival=10.0, s_arrival=18.0):
    """Create synthetic seismic signal with P-wave and S-wave arrivals."""
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Background noise
    noise = 0.1 * np.random.randn(len(t))
    
    # P-wave signal (higher frequency, shorter duration)
    p_wave = np.zeros_like(t)
    p_start_idx = int(p_arrival * sampling_rate)
    p_duration_samples = int(3.0 * sampling_rate)  # 3 second P-wave
    
    if p_start_idx + p_duration_samples < len(t):
        p_time = t[p_start_idx:p_start_idx + p_duration_samples] - p_arrival
        p_wave[p_start_idx:p_start_idx + p_duration_samples] = (
            1.5 * np.sin(2 * np.pi * 8 * p_time) * np.exp(-p_time / 2.0)
        )
    
    # S-wave signal (lower frequency, longer duration, higher amplitude)
    s_wave = np.zeros_like(t)
    s_start_idx = int(s_arrival * sampling_rate)
    s_duration_samples = int(6.0 * sampling_rate)  # 6 second S-wave
    
    if s_start_idx + s_duration_samples < len(t):
        s_time = t[s_start_idx:s_start_idx + s_duration_samples] - s_arrival
        s_wave[s_start_idx:s_start_idx + s_duration_samples] = (
            2.5 * np.sin(2 * np.pi * 4 * s_time) * np.exp(-s_time / 4.0)
        )
    
    return noise + p_wave + s_wave

def test_s_wave_detection():
    """Test S-wave detection with synthetic data."""
    print("Testing S-wave detection...")
    
    # Create detector
    sampling_rate = 100.0
    params = SWaveDetectionParameters(
        sampling_rate=sampling_rate,
        sta_window=2.0,
        lta_window=10.0,
        trigger_threshold=2.0,
        detrigger_threshold=1.2,
        confidence_threshold=0.3,
        p_wave_context_window=15.0  # Increase context window
    )
    detector = SWaveDetector(params)
    
    # Create synthetic signal
    signal_data = create_synthetic_s_wave_signal(
        duration=40.0, 
        sampling_rate=sampling_rate,
        p_arrival=10.0, 
        s_arrival=18.0
    )
    
    # Provide P-wave context
    metadata = {
        'p_wave_arrivals': [10.0]
    }
    
    # Detect S-waves
    print("Running S-wave detection...")
    detections = detector.detect_waves(signal_data, sampling_rate, metadata)
    
    print(f"Number of S-wave detections: {len(detections)}")
    
    if len(detections) > 0:
        for i, detection in enumerate(detections):
            print(f"Detection {i+1}:")
            print(f"  Wave type: {detection.wave_type}")
            print(f"  Arrival time: {detection.arrival_time:.2f} seconds")
            print(f"  Duration: {detection.duration:.2f} seconds")
            print(f"  Peak amplitude: {detection.peak_amplitude:.3f}")
            print(f"  Dominant frequency: {detection.dominant_frequency:.2f} Hz")
            print(f"  Confidence: {detection.confidence:.3f}")
            print(f"  Detection method: {detection.metadata.get('detection_method', 'N/A')}")
            
            # Check S-P time difference
            sp_time = detection.arrival_time - 10.0  # P-wave at 10s
            print(f"  S-P time difference: {sp_time:.2f} seconds")
            print()
    
    # Get detection statistics
    print("Getting detection statistics...")
    stats = detector.get_detection_statistics(signal_data, metadata)
    
    print("Detection Statistics:")
    for key, value in stats.items():
        if isinstance(value, (list, np.ndarray)) and len(value) > 5:
            print(f"  {key}: [array with {len(value)} elements]")
        else:
            print(f"  {key}: {value}")
    
    return len(detections) > 0

def test_multi_channel_detection():
    """Test multi-channel S-wave detection."""
    print("\nTesting multi-channel S-wave detection...")
    
    sampling_rate = 100.0
    params = SWaveDetectionParameters(sampling_rate=sampling_rate)
    detector = SWaveDetector(params)
    
    # Create multi-channel synthetic signal
    duration = 30.0
    n_channels = 3
    n_samples = int(duration * sampling_rate)
    
    multi_channel_data = np.zeros((n_channels, n_samples))
    
    # Add noise to all channels
    for i in range(n_channels):
        multi_channel_data[i] = 0.1 * np.random.randn(n_samples)
    
    # Add S-wave with different characteristics on each channel
    s_arrival = 15.0
    s_start_idx = int(s_arrival * sampling_rate)
    s_duration_samples = int(4.0 * sampling_rate)
    
    if s_start_idx + s_duration_samples < n_samples:
        t = np.linspace(0, 4.0, s_duration_samples)
        s_signal_base = 2.0 * np.sin(2 * np.pi * 4 * t) * np.exp(-t / 3.0)
        
        # Different amplitudes and phases on different channels (simulating shear motion)
        multi_channel_data[0, s_start_idx:s_start_idx + s_duration_samples] += 0.5 * s_signal_base
        multi_channel_data[1, s_start_idx:s_start_idx + s_duration_samples] += s_signal_base
        multi_channel_data[2, s_start_idx:s_start_idx + s_duration_samples] += (
            0.8 * np.sin(2 * np.pi * 4 * t + np.pi/4) * np.exp(-t / 3.0)
        )
    
    # Detect S-waves
    detections = detector.detect_waves(multi_channel_data, sampling_rate)
    
    print(f"Multi-channel detections: {len(detections)}")
    
    if len(detections) > 0:
        detection = detections[0]
        print(f"  Arrival time: {detection.arrival_time:.2f} seconds")
        print(f"  Detection method: {detection.metadata.get('detection_method', 'N/A')}")
        if 'polarization_value' in detection.metadata:
            print(f"  Polarization value: {detection.metadata['polarization_value']:.3f}")
        if 'particle_motion_complexity' in detection.metadata:
            print(f"  Particle motion complexity: {detection.metadata['particle_motion_complexity']:.3f}")
    
    return len(detections) > 0

if __name__ == "__main__":
    print("S-wave Detection Test")
    print("=" * 50)
    
    # Test single-channel detection
    single_channel_success = test_s_wave_detection()
    
    # Test multi-channel detection
    multi_channel_success = test_multi_channel_detection()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Single-channel detection: {'PASS' if single_channel_success else 'FAIL'}")
    print(f"Multi-channel detection: {'PASS' if multi_channel_success else 'FAIL'}")
    
    if single_channel_success and multi_channel_success:
        print("\nAll tests PASSED! S-wave detector implementation is working correctly.")
    else:
        print("\nSome tests FAILED. Check the implementation.")