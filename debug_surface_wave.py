"""
Debug script for surface wave detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from wave_analysis.services.wave_detectors import SurfaceWaveDetector, SurfaceWaveDetectionParameters

def create_synthetic_surface_wave(sampling_rate=100.0, duration=120.0, dominant_freq=0.1):
    """Create synthetic surface wave for testing."""
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Create stronger surface wave signal
    wave = np.zeros(n_samples)
    
    # Create multiple wave packets to simulate surface wave train
    for wave_start in [30, 50, 70]:  # Multiple wave arrivals
        # Create wave packet with multiple frequency components
        freq_range = np.linspace(dominant_freq * 0.7, dominant_freq * 1.3, 5)
        
        for freq in freq_range:
            # Create wave with envelope centered at wave_start
            phase = 2 * np.pi * freq * t
            envelope = 2.0 * np.exp(-0.5 * ((t - wave_start) / 15)**2)  # Stronger, wider envelope
            
            # Add frequency component
            wave += envelope * np.sin(phase) / len(freq_range)
    
    # Add minimal noise
    noise = np.random.normal(0, 0.01, n_samples)
    wave += noise
    
    return wave

def main():
    # Create detector with more reasonable parameters
    params = SurfaceWaveDetectionParameters(
        sampling_rate=100.0,
        frequency_band=(0.05, 0.5),     # Higher low-frequency cutoff to avoid filter issues
        min_surface_wave_duration=5.0,  # Reduced minimum duration
        energy_ratio_threshold=1.5,     # More sensitive threshold
        dispersion_window=20.0           # Smaller window for better resolution
    )
    
    detector = SurfaceWaveDetector(params)
    
    # Create synthetic surface wave
    surface_wave = create_synthetic_surface_wave()
    
    print(f"Created surface wave with {len(surface_wave)} samples")
    print(f"Duration: {len(surface_wave) / 100.0} seconds")
    print(f"Max amplitude: {np.max(np.abs(surface_wave))}")
    
    # Get detection statistics
    stats = detector.get_detection_statistics(surface_wave)
    print("\nDetection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Debug the detection process step by step
    print("\nDebugging detection process...")
    
    # Step 1: Preprocessing
    processed_data = detector.signal_processor.preprocess_seismic_data(surface_wave)
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed data max amplitude: {np.max(np.abs(processed_data))}")
    
    # Step 2: Filtering
    filtered_data = detector._apply_surface_wave_filter(processed_data)
    print(f"Filtered data max amplitude: {np.max(np.abs(filtered_data))}")
    
    # Step 3: Frequency-time analysis
    time_freq_analysis = detector._perform_frequency_time_analysis(filtered_data)
    print(f"Frequency-time analysis keys: {time_freq_analysis.keys()}")
    print(f"Number of frequency bins: {len(time_freq_analysis['frequencies'])}")
    print(f"Number of time bins: {len(time_freq_analysis['times'])}")
    print(f"Surface energy shape: {time_freq_analysis['surface_energy'].shape}")
    print(f"Surface energy max: {np.max(time_freq_analysis['surface_energy'])}")
    print(f"Total energy max: {np.max(time_freq_analysis['total_energy'])}")
    
    # Step 4: Check energy ratio
    surface_energy = time_freq_analysis['surface_energy']
    total_energy = time_freq_analysis['total_energy']
    energy_ratio = np.divide(surface_energy, total_energy, 
                           out=np.zeros_like(surface_energy), 
                           where=total_energy != 0)
    print(f"Energy ratio max: {np.max(energy_ratio)}")
    print(f"Energy ratio mean: {np.mean(energy_ratio)}")
    print(f"Threshold: {1.0 / params.energy_ratio_threshold}")
    print(f"Points above threshold: {np.sum(energy_ratio > (1.0 / params.energy_ratio_threshold))}")
    
    # Try detection
    detected_waves = detector.detect_waves(surface_wave, 100.0)
    print(f"\nDetected {len(detected_waves)} surface waves")
    
    for i, wave in enumerate(detected_waves):
        print(f"Wave {i+1}:")
        print(f"  Type: {wave.wave_type}")
        print(f"  Duration: {wave.duration:.2f}s")
        print(f"  Confidence: {wave.confidence:.3f}")
        print(f"  Peak amplitude: {wave.peak_amplitude:.3f}")
        print(f"  Dominant frequency: {wave.dominant_frequency:.3f} Hz")

if __name__ == "__main__":
    main()