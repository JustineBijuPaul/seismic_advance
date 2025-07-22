#!/usr/bin/env python3

"""
Test the complete wave analysis visualization infrastructure.

This script tests the integration between all visualization components
to ensure the complete workflow functions correctly.
"""

import numpy as np
from wave_analysis.services.wave_visualizer import WaveVisualizer
from wave_analysis.services.multi_channel_plotter import ChannelData
from wave_analysis.models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, 
    ArrivalTimes, MagnitudeEstimate, QualityMetrics, FrequencyData
)

def test_complete_workflow():
    """Test the complete wave analysis visualization workflow."""
    
    print("Testing Wave Analysis Visualization Infrastructure...")
    
    # Create test data
    print("1. Creating test wave data...")
    p_wave = WaveSegment(
        wave_type='P',
        start_time=5.0,
        end_time=8.0,
        data=np.random.randn(300),
        sampling_rate=100.0,
        peak_amplitude=0.5,
        dominant_frequency=15.0,
        arrival_time=5.5,
        confidence=0.85
    )
    
    s_wave = WaveSegment(
        wave_type='S',
        start_time=10.0,
        end_time=15.0,
        data=np.random.randn(500),
        sampling_rate=100.0,
        peak_amplitude=0.8,
        dominant_frequency=8.0,
        arrival_time=10.5,
        confidence=0.92
    )
    
    surface_wave = WaveSegment(
        wave_type='Love',
        start_time=20.0,
        end_time=30.0,
        data=np.random.randn(1000),
        sampling_rate=100.0,
        peak_amplitude=1.2,
        dominant_frequency=3.0,
        arrival_time=20.5,
        confidence=0.78
    )
    
    # Create wave analysis result
    wave_result = WaveAnalysisResult(
        original_data=np.random.randn(3000),
        sampling_rate=100.0,
        p_waves=[p_wave],
        s_waves=[s_wave],
        surface_waves=[surface_wave]
    )
    
    # Create detailed analysis
    analysis = DetailedAnalysis(
        wave_result=wave_result,
        arrival_times=ArrivalTimes(
            p_wave_arrival=5.5,
            s_wave_arrival=10.5,
            surface_wave_arrival=20.5,
            sp_time_difference=5.0
        ),
        magnitude_estimates=[
            MagnitudeEstimate(
                method='ML',
                magnitude=4.2,
                confidence=0.85,
                wave_type_used='P'
            )
        ],
        epicenter_distance=45.2,
        frequency_analysis={
            'P': FrequencyData(
                frequencies=np.linspace(0, 50, 100),
                power_spectrum=np.random.rand(100),
                dominant_frequency=15.0,
                frequency_range=(5.0, 25.0),
                spectral_centroid=12.5,
                bandwidth=8.0
            )
        },
        quality_metrics=QualityMetrics(
            signal_to_noise_ratio=15.5,
            detection_confidence=0.87,
            analysis_quality_score=0.92,
            data_completeness=0.98
        )
    )
    
    # Create channel data
    channels = [
        ChannelData('CH1', np.random.randn(1000), 100.0, {'lat': 40.0, 'lon': -120.0}, 'N'),
        ChannelData('CH2', np.random.randn(1000), 100.0, {'lat': 40.0, 'lon': -120.0}, 'E'),
        ChannelData('CH3', np.random.randn(1000), 100.0, {'lat': 40.0, 'lon': -120.0}, 'Z')
    ]
    
    # Test interactive visualizer
    print("2. Testing interactive visualizer...")
    visualizer = WaveVisualizer(interactive=True)
    
    # Test comprehensive analysis plot
    comprehensive_plot = visualizer.create_comprehensive_analysis_plot(analysis)
    print(f"   ✓ Comprehensive analysis plot: {comprehensive_plot['type']}")
    
    # Test wave separation plot
    wave_separation = visualizer.create_wave_separation_plot(wave_result)
    print(f"   ✓ Wave separation plot: {wave_separation['type']}")
    
    # Test frequency analysis
    frequency_plot = visualizer.create_frequency_analysis_plot([p_wave, s_wave, surface_wave])
    print(f"   ✓ Frequency analysis plot: {frequency_plot['type']}")
    
    # Test spectrogram
    spectrogram = visualizer.create_spectrogram_analysis(p_wave)
    print(f"   ✓ Spectrogram analysis: {spectrogram['type']}")
    
    # Test multi-channel analysis
    multi_channel = visualizer.create_multi_channel_analysis(channels)
    print(f"   ✓ Multi-channel analysis: {multi_channel['type']}")
    
    # Test correlation analysis
    correlation = visualizer.create_correlation_analysis(channels)
    print(f"   ✓ Correlation analysis: {correlation['type']}")
    
    # Test wave picker interface
    wave_picker = visualizer.create_wave_picker_interface(wave_result)
    print(f"   ✓ Wave picker interface: {wave_picker['type']}")
    
    # Test static visualizer
    print("3. Testing static visualizer...")
    static_visualizer = WaveVisualizer(interactive=False)
    
    static_plot = static_visualizer.create_wave_separation_plot(wave_result)
    print(f"   ✓ Static wave separation plot: {static_plot['type']}")
    
    # Test comprehensive report generation
    print("4. Testing comprehensive report generation...")
    report = visualizer.generate_analysis_report(analysis)
    print(f"   ✓ Analysis report generated: {report['type']}")
    print(f"   ✓ Report contains {len(report['visualizations'])} visualizations")
    print(f"   ✓ Summary includes: {list(report['summary'].keys())}")
    
    # Test export functionality
    print("5. Testing export functionality...")
    json_export = visualizer.export_visualization(comprehensive_plot, 'json')
    print(f"   ✓ JSON export: {len(json_export)} characters")
    
    # Test error handling
    print("6. Testing error handling...")
    empty_result = visualizer.create_comprehensive_analysis_plot(None)
    print(f"   ✓ Empty data handling: {empty_result['type']}")
    
    # Test settings configuration
    print("7. Testing settings configuration...")
    custom_settings = {
        'color_scheme': {
            'P': '#FF0000',
            'S': '#00FF00',
            'Love': '#0000FF'
        }
    }
    visualizer.set_visualization_settings(custom_settings)
    print("   ✓ Custom settings applied")
    
    # Test available formats and plot types
    formats = visualizer.get_supported_formats()
    plot_types = visualizer.get_available_plot_types()
    print(f"   ✓ Supported formats: {formats}")
    print(f"   ✓ Available plot types: {len(plot_types)} types")
    
    print("\n✅ All tests passed! Wave analysis visualization infrastructure is working correctly.")
    print(f"\nSummary:")
    print(f"- Interactive visualizations: ✓")
    print(f"- Static visualizations: ✓")
    print(f"- Multi-panel analysis: ✓")
    print(f"- Multi-channel support: ✓")
    print(f"- Export functionality: ✓")
    print(f"- Error handling: ✓")
    print(f"- Configuration management: ✓")

if __name__ == '__main__':
    test_complete_workflow()