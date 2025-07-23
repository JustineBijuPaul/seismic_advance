"""
Demonstration script for TestDataManager functionality.

This script demonstrates how to use the TestDataManager for generating
synthetic earthquake data, noise samples, and validating test data quality
for wave analysis algorithm testing.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from tests.test_data_manager import (
    TestDataManager, SyntheticEarthquakeParams, NoiseProfile
)


def main():
    """Demonstrate TestDataManager functionality."""
    print("=== TestDataManager Demonstration ===\n")
    
    # Initialize test data manager
    manager = TestDataManager()
    print(f"Initialized TestDataManager with cache directory: {manager.cache_dir}")
    
    # 1. Generate synthetic earthquake data
    print("\n1. Generating Synthetic Earthquake Data")
    print("-" * 40)
    
    params = SyntheticEarthquakeParams(
        magnitude=6.0,
        distance=100.0,
        depth=15.0,
        duration=60.0,
        sampling_rate=100.0,
        noise_level=0.1
    )
    
    print(f"Earthquake parameters:")
    print(f"  Magnitude: {params.magnitude}")
    print(f"  Distance: {params.distance} km")
    print(f"  Depth: {params.depth} km")
    print(f"  Duration: {params.duration} seconds")
    print(f"  Sampling rate: {params.sampling_rate} Hz")
    print(f"  Noise level: {params.noise_level}")
    
    earthquake_data = manager.create_synthetic_earthquake(params)
    print(f"\nGenerated {len(earthquake_data)} samples of synthetic earthquake data")
    
    # 2. Validate data quality
    print("\n2. Validating Data Quality")
    print("-" * 30)
    
    validation = manager.validate_test_data_quality(earthquake_data, params)
    print(f"Quality score: {validation['quality_score']:.2f}")
    print(f"Data length valid: {validation['data_length_valid']}")
    print(f"Amplitude range valid: {validation['amplitude_range_valid']}")
    print(f"Frequency content valid: {validation['frequency_content_valid']}")
    print(f"Noise level valid: {validation['noise_level_valid']}")
    print(f"Wave arrivals detected: {validation['wave_arrivals_detected']}")
    
    if validation['warnings']:
        print(f"Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # 3. Generate noise samples
    print("\n3. Generating Noise Samples")
    print("-" * 30)
    
    noise_profiles = [
        NoiseProfile(
            noise_type='white',
            amplitude=0.1,
            frequency_range=(0, 50),
            duration=30.0
        ),
        NoiseProfile(
            noise_type='seismic',
            amplitude=0.15,
            frequency_range=(0.1, 20),
            duration=30.0
        )
    ]
    
    noise_samples = manager.generate_noise_samples(noise_profiles)
    print(f"Generated {len(noise_samples)} noise samples:")
    for name, data in noise_samples.items():
        print(f"  {name}: {len(data)} samples, std={np.std(data):.3f}")
    
    # 4. Generate multi-channel data
    print("\n4. Generating Multi-Channel Data")
    print("-" * 35)
    
    channels = 3
    multi_channel_data = manager.create_multi_channel_data(channels, params)
    print(f"Generated {channels}-channel data with shape: {multi_channel_data.shape}")
    
    # Calculate correlations between channels
    correlations = []
    for i in range(channels):
        for j in range(i+1, channels):
            corr = np.corrcoef(multi_channel_data[i], multi_channel_data[j])[0, 1]
            correlations.append(corr)
            print(f"  Channel {i}-{j} correlation: {corr:.3f}")
    
    # 5. Create visualization
    print("\n5. Creating Visualizations")
    print("-" * 28)
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('TestDataManager Demonstration', fontsize=16)
        
        # Plot earthquake data
        time_axis = np.linspace(0, params.duration, len(earthquake_data))
        axes[0, 0].plot(time_axis, earthquake_data)
        axes[0, 0].set_title('Synthetic Earthquake Data')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True)
        
        # Plot frequency spectrum
        freqs = np.fft.fftfreq(len(earthquake_data), 1/params.sampling_rate)
        fft_data = np.fft.fft(earthquake_data)
        psd = np.abs(fft_data) ** 2
        
        # Plot only positive frequencies
        pos_freqs = freqs[:len(freqs)//2]
        pos_psd = psd[:len(psd)//2]
        
        axes[0, 1].loglog(pos_freqs[1:], pos_psd[1:])  # Skip DC component
        axes[0, 1].set_title('Frequency Spectrum')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power Spectral Density')
        axes[0, 1].grid(True)
        
        # Plot noise samples
        noise_time = np.linspace(0, 30, len(list(noise_samples.values())[0]))
        for i, (name, data) in enumerate(noise_samples.items()):
            axes[1, 0].plot(noise_time, data, label=name, alpha=0.7)
        axes[1, 0].set_title('Noise Samples')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Amplitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot multi-channel data (first 10 seconds)
        plot_samples = int(10 * params.sampling_rate)
        plot_time = np.linspace(0, 10, plot_samples)
        
        for i in range(channels):
            axes[1, 1].plot(plot_time, multi_channel_data[i, :plot_samples], 
                           label=f'Channel {i}', alpha=0.7)
        axes[1, 1].set_title('Multi-Channel Data (First 10s)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Amplitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = 'test_data_manager_demo.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved as: {plot_filename}")
        
        # Show plot if running interactively
        if os.environ.get('DISPLAY') or os.name == 'nt':  # Unix with display or Windows
            plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # 6. Save test dataset
    print("\n6. Saving Test Dataset")
    print("-" * 25)
    
    test_dataset = {
        'earthquake_data': earthquake_data,
        'parameters': params.__dict__,
        'validation_results': validation,
        'noise_samples': noise_samples,
        'multi_channel_data': multi_channel_data,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'purpose': 'TestDataManager demonstration',
            'quality_score': validation['quality_score']
        }
    }
    
    dataset_filename = 'demo_test_dataset.json'
    manager.save_test_dataset(dataset_filename, test_dataset)
    print(f"Test dataset saved as: {dataset_filename}")
    
    # Verify we can load it back
    loaded_dataset = manager.load_test_dataset(dataset_filename)
    if loaded_dataset:
        print("✓ Dataset successfully loaded and verified")
    else:
        print("✗ Failed to load dataset")
    
    # 7. Generate test suite for different magnitudes
    print("\n7. Generating Multi-Magnitude Test Suite")
    print("-" * 42)
    
    magnitudes = [4.0, 5.5, 7.0]
    test_suite = {}
    
    for mag in magnitudes:
        test_params = SyntheticEarthquakeParams(
            magnitude=mag,
            distance=100.0,
            depth=15.0,
            duration=30.0,
            noise_level=0.1
        )
        
        data = manager.create_synthetic_earthquake(test_params)
        validation = manager.validate_test_data_quality(data, test_params)
        
        test_suite[f'M{mag}'] = {
            'data': data,
            'parameters': test_params.__dict__,
            'validation': validation
        }
        
        max_amplitude = np.max(np.abs(data))
        print(f"  M{mag}: {len(data)} samples, max amplitude: {max_amplitude:.2e}, quality: {validation['quality_score']:.2f}")
    
    # Save multi-magnitude test suite
    suite_filename = 'multi_magnitude_test_suite.json'
    manager.save_test_dataset(suite_filename, test_suite)
    print(f"Multi-magnitude test suite saved as: {suite_filename}")
    
    # 8. Summary
    print("\n8. Summary")
    print("-" * 12)
    print(f"✓ Generated synthetic earthquake data with quality score: {validation['quality_score']:.2f}")
    print(f"✓ Created {len(noise_samples)} different noise samples")
    print(f"✓ Generated {channels}-channel multi-channel data")
    print(f"✓ Created test suite with {len(test_suite)} different magnitude scenarios")
    print(f"✓ All data saved to cache directory: {manager.cache_dir}")
    
    # Cleanup
    print(f"\nTo clean up test data, run: manager.cleanup_cache()")
    print("Or manually delete the cache directory when done.")
    
    return manager


if __name__ == '__main__':
    # Run the demonstration
    test_manager = main()
    
    # Keep the manager object available for interactive use
    print(f"\nTestDataManager instance available as 'test_manager'")
    print("You can continue to use it for additional testing:")
    print("  - test_manager.create_synthetic_earthquake(params)")
    print("  - test_manager.generate_noise_samples(profiles)")
    print("  - test_manager.validate_test_data_quality(data, params)")