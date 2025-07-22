#!/usr/bin/env python3
"""
Demonstration of integrated visualization components.

This script shows how all three visualization components work together
to provide comprehensive wave analysis visualization capabilities.
"""

import numpy as np
from wave_analysis import (
    WaveSegment, 
    WaveAnalysisResult,
    TimeSeriesPlotter, 
    FrequencyPlotter, 
    MultiChannelPlotter, 
    ChannelData
)


def create_synthetic_earthquake_data():
    """Create synthetic earthquake data for demonstration."""
    sampling_rate = 100.0  # 100 Hz
    duration = 20.0  # 20 seconds
    samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, samples)
    
    # Create P-wave (arrives at t=5s)
    p_wave_signal = 0.8 * np.sin(2 * np.pi * 8 * t) * np.exp(-0.5 * (t - 5)**2)
    
    # Create S-wave (arrives at t=10s)
    s_wave_signal = 0.6 * np.sin(2 * np.pi * 4 * t) * np.exp(-0.3 * (t - 10)**2)
    
    # Create surface wave (arrives at t=15s)
    surface_wave_signal = 0.4 * np.sin(2 * np.pi * 1 * t) * np.exp(-0.1 * (t - 15)**2)
    
    # Combine signals with noise
    noise = 0.1 * np.random.normal(0, 1, samples)
    combined_signal = p_wave_signal + s_wave_signal + surface_wave_signal + noise
    
    # Create wave segments
    p_wave = WaveSegment(
        wave_type='P',
        start_time=4.0,
        end_time=6.0,
        data=combined_signal[400:600],
        sampling_rate=sampling_rate,
        peak_amplitude=0.8,
        dominant_frequency=8.0,
        arrival_time=5.0,
        confidence=0.9
    )
    
    s_wave = WaveSegment(
        wave_type='S',
        start_time=9.0,
        end_time=11.0,
        data=combined_signal[900:1100],
        sampling_rate=sampling_rate,
        peak_amplitude=0.6,
        dominant_frequency=4.0,
        arrival_time=10.0,
        confidence=0.85
    )
    
    surface_wave = WaveSegment(
        wave_type='Rayleigh',
        start_time=14.0,
        end_time=16.0,
        data=combined_signal[1400:1600],
        sampling_rate=sampling_rate,
        peak_amplitude=0.4,
        dominant_frequency=1.0,
        arrival_time=15.0,
        confidence=0.8
    )
    
    # Create WaveAnalysisResult object
    wave_result = WaveAnalysisResult(
        original_data=combined_signal,
        sampling_rate=sampling_rate,
        p_waves=[p_wave],
        s_waves=[s_wave],
        surface_waves=[surface_wave],
        metadata={'duration': duration, 'samples': samples}
    )
    
    return [p_wave, s_wave, surface_wave], wave_result, combined_signal, sampling_rate


def create_multi_channel_data():
    """Create synthetic multi-channel data."""
    sampling_rate = 100.0
    duration = 20.0
    samples = int(duration * sampling_rate)
    
    # Create three channels with different orientations
    channels = []
    orientations = ['Z', 'N', 'E']  # Vertical, North, East
    
    for i, orientation in enumerate(orientations):
        t = np.linspace(0, duration, samples)
        
        # Different amplitude patterns for different orientations
        if orientation == 'Z':  # Vertical - more P-wave
            signal = (0.8 * np.sin(2 * np.pi * 8 * t) * np.exp(-0.5 * (t - 5)**2) +
                     0.3 * np.sin(2 * np.pi * 4 * t) * np.exp(-0.3 * (t - 10)**2))
        elif orientation == 'N':  # North - more S-wave
            signal = (0.3 * np.sin(2 * np.pi * 8 * t) * np.exp(-0.5 * (t - 5)**2) +
                     0.7 * np.sin(2 * np.pi * 4 * t) * np.exp(-0.3 * (t - 10)**2))
        else:  # East - similar to North but phase shifted
            signal = (0.3 * np.sin(2 * np.pi * 8 * t + np.pi/4) * np.exp(-0.5 * (t - 5)**2) +
                     0.7 * np.sin(2 * np.pi * 4 * t + np.pi/2) * np.exp(-0.3 * (t - 10)**2))
        
        # Add noise
        noise = 0.1 * np.random.normal(0, 1, samples)
        signal += noise
        
        # Create channel data
        channel = ChannelData(
            channel_id=f'CH{i+1}_{orientation}',
            data=signal,
            sampling_rate=sampling_rate,
            location={'lat': 40.0 + i*0.01, 'lon': -120.0 + i*0.01, 'elevation': 1000.0},
            orientation=orientation
        )
        channels.append(channel)
    
    return channels


def demonstrate_time_series_visualization():
    """Demonstrate time series visualization capabilities."""
    print("=== Time Series Visualization Demo ===")
    
    wave_segments, wave_result, _, _ = create_synthetic_earthquake_data()
    plotter = TimeSeriesPlotter()
    
    # Create time series plot
    plot_data = plotter.create_time_series_plot(wave_segments)
    print(f"Created time series plot with {len(plot_data['datasets'])} datasets")
    print(f"Plot type: {plot_data['type']}")
    
    # Create multi-wave comparison
    comparison_plot = plotter.create_multi_wave_comparison(wave_result)
    print(f"Created comparison plot with {len(comparison_plot['datasets'])} datasets")
    
    # Export plot data
    exported_data = plotter.export_plot_data(wave_segments, 'json')
    print(f"Exported plot data: {len(exported_data)} characters")
    
    print("Time series visualization demo completed successfully!\n")


def demonstrate_frequency_visualization():
    """Demonstrate frequency domain visualization capabilities."""
    print("=== Frequency Domain Visualization Demo ===")
    
    wave_segments, _, _, _ = create_synthetic_earthquake_data()
    plotter = FrequencyPlotter()
    
    # Create frequency spectrum plot
    freq_plot = plotter.create_frequency_plot(wave_segments)
    print(f"Created frequency plot with {len(freq_plot['datasets'])} datasets")
    print(f"Plot type: {freq_plot['type']}")
    
    # Create spectrogram plot
    spectrogram_plot = plotter.create_spectrogram_plot(wave_segments)
    spectrogram_data_count = 1 if 'data' in spectrogram_plot else len(spectrogram_plot.get('datasets', []))
    print(f"Created spectrogram plot with {spectrogram_data_count} data element(s)")
    
    # Create power spectral density plot
    psd_plot = plotter.create_power_spectral_density_plot(wave_segments)
    print(f"Created PSD plot with {len(psd_plot['datasets'])} datasets")
    
    # Export frequency data
    exported_data = plotter.export_frequency_data(wave_segments, 'json')
    print(f"Exported frequency data: {len(exported_data)} characters")
    
    print("Frequency domain visualization demo completed successfully!\n")


def demonstrate_multi_channel_visualization():
    """Demonstrate multi-channel visualization capabilities."""
    print("=== Multi-Channel Visualization Demo ===")
    
    channels = create_multi_channel_data()
    wave_segments, _, _, _ = create_synthetic_earthquake_data()
    plotter = MultiChannelPlotter()
    
    # Create multi-channel time series plot
    multi_plot = plotter.create_multi_channel_plot(channels)
    print(f"Created multi-channel plot with {len(multi_plot['datasets'])} datasets")
    print(f"Plot type: {multi_plot['type']}")
    
    # Create cross-correlation plot
    corr_plot = plotter.create_cross_correlation_plot(channels)
    print(f"Created correlation plot with {len(corr_plot['datasets'])} datasets")
    print(f"Correlation summary: {corr_plot['correlation_summary']['pair_count']} pairs")
    
    # Create coherence plot
    coherence_plot = plotter.create_coherence_plot(channels)
    print(f"Created coherence plot with {len(coherence_plot['datasets'])} datasets")
    
    # Create correlation matrix
    matrix_plot = plotter.create_correlation_matrix_plot(channels)
    print(f"Created correlation matrix: {len(matrix_plot['data']['x'])}x{len(matrix_plot['data']['y'])}")
    
    # Create channel comparison with wave arrivals
    comparison_plot = plotter.create_channel_comparison_plot(channels, wave_segments)
    print(f"Created channel comparison with {len(comparison_plot['annotations'])} wave markers")
    
    # Export correlation data
    corr_data = plotter.export_correlation_data(channels, 'json')
    print(f"Exported correlation data: {len(corr_data)} characters")
    
    # Export coherence data
    coh_data = plotter.export_coherence_data(channels, 'json')
    print(f"Exported coherence data: {len(coh_data)} characters")
    
    print("Multi-channel visualization demo completed successfully!\n")


def demonstrate_integrated_workflow():
    """Demonstrate integrated visualization workflow."""
    print("=== Integrated Visualization Workflow Demo ===")
    
    # Create data
    wave_segments, wave_result, combined_signal, sampling_rate = create_synthetic_earthquake_data()
    channels = create_multi_channel_data()
    
    # Initialize all plotters
    time_plotter = TimeSeriesPlotter()
    freq_plotter = FrequencyPlotter()
    multi_plotter = MultiChannelPlotter()
    
    print("Initialized all visualization components")
    
    # Create comprehensive visualization suite
    visualizations = {
        'time_series': time_plotter.create_time_series_plot(wave_segments),
        'frequency_spectrum': freq_plotter.create_frequency_plot(wave_segments),
        'spectrogram': freq_plotter.create_spectrogram_plot(wave_segments),
        'multi_channel': multi_plotter.create_multi_channel_plot(channels),
        'cross_correlation': multi_plotter.create_cross_correlation_plot(channels),
        'coherence': multi_plotter.create_coherence_plot(channels),
        'correlation_matrix': multi_plotter.create_correlation_matrix_plot(channels)
    }
    
    print(f"Created {len(visualizations)} different visualization types:")
    for viz_type, viz_data in visualizations.items():
        dataset_count = len(viz_data.get('datasets', []))
        print(f"  - {viz_type}: {dataset_count} datasets")
    
    print("Integrated visualization workflow completed successfully!\n")


def main():
    """Main demonstration function."""
    print("Earthquake Wave Analysis - Visualization Components Demo")
    print("=" * 60)
    
    try:
        # Run individual component demos
        demonstrate_time_series_visualization()
        demonstrate_frequency_visualization()
        demonstrate_multi_channel_visualization()
        
        # Run integrated workflow demo
        demonstrate_integrated_workflow()
        
        print("=" * 60)
        print("All visualization components are working correctly!")
        print("The earthquake wave analysis system now has comprehensive")
        print("visualization capabilities including:")
        print("  ✓ Time series plotting for individual wave types")
        print("  ✓ Frequency domain analysis and spectrograms")
        print("  ✓ Multi-channel correlation and coherence analysis")
        print("  ✓ Interactive plotting with zoom and pan capabilities")
        print("  ✓ Export capabilities for all visualization types")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        raise


if __name__ == '__main__':
    main()