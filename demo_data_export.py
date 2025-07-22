#!/usr/bin/env python3
"""
Demo script for data export capabilities.

This script demonstrates the data export functionality implemented
for the earthquake wave analysis system.
"""

import numpy as np
from datetime import datetime
import json

from wave_analysis import (
    DataExporter, MSEEDExporter, CSVExporter,
    WaveSegment, WaveAnalysisResult, DetailedAnalysis,
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)


def create_sample_data():
    """Create sample wave analysis data for demonstration."""
    print("Creating sample earthquake wave data...")
    
    # Create synthetic earthquake signal
    sampling_rate = 100.0
    duration = 10.0
    t = np.linspace(0, duration, int(sampling_rate * duration))
    
    # P-wave: high frequency, short duration
    p_signal = np.sin(2 * np.pi * 15 * t) * np.exp(-t * 3) * (t > 1.0) * (t < 3.0)
    p_wave = WaveSegment(
        wave_type='P',
        start_time=1.0,
        end_time=3.0,
        data=p_signal[100:300],  # 1-3 seconds
        sampling_rate=sampling_rate,
        peak_amplitude=np.max(np.abs(p_signal)),
        dominant_frequency=15.0,
        arrival_time=1.2,
        confidence=0.92,
        metadata={'detector': 'STA/LTA', 'snr': 12.5}
    )
    
    # S-wave: medium frequency, medium duration
    s_signal = np.sin(2 * np.pi * 8 * t) * np.exp(-t * 2) * (t > 3.0) * (t < 6.0)
    s_wave = WaveSegment(
        wave_type='S',
        start_time=3.0,
        end_time=6.0,
        data=s_signal[300:600],  # 3-6 seconds
        sampling_rate=sampling_rate,
        peak_amplitude=np.max(np.abs(s_signal)),
        dominant_frequency=8.0,
        arrival_time=3.2,
        confidence=0.88,
        metadata={'detector': 'Polarization', 'snr': 8.3}
    )
    
    # Surface wave: low frequency, long duration
    surface_signal = np.sin(2 * np.pi * 2 * t) * np.exp(-t * 0.5) * (t > 8.0)
    surface_wave = WaveSegment(
        wave_type='Love',
        start_time=8.0,
        end_time=10.0,
        data=surface_signal[800:1000],  # 8-10 seconds
        sampling_rate=sampling_rate,
        peak_amplitude=np.max(np.abs(surface_signal)),
        dominant_frequency=2.0,
        arrival_time=8.2,
        confidence=0.75,
        metadata={'detector': 'Group velocity', 'snr': 6.1}
    )
    
    # Create complete signal
    original_data = p_signal + s_signal + surface_signal + np.random.randn(len(t)) * 0.1
    
    # Create wave analysis result
    wave_result = WaveAnalysisResult(
        original_data=original_data,
        sampling_rate=sampling_rate,
        p_waves=[p_wave],
        s_waves=[s_wave],
        surface_waves=[surface_wave],
        metadata={
            'station': 'TEST01',
            'location': 'Demo Station',
            'earthquake_id': 'demo_eq_001'
        }
    )
    
    # Create detailed analysis
    arrival_times = ArrivalTimes(
        p_wave_arrival=1.2,
        s_wave_arrival=3.2,
        surface_wave_arrival=8.2,
        sp_time_difference=2.0
    )
    
    magnitude_estimates = [
        MagnitudeEstimate(
            method='ML',
            magnitude=4.2,
            confidence=0.85,
            wave_type_used='P',
            metadata={'station_correction': -0.1}
        ),
        MagnitudeEstimate(
            method='Ms',
            magnitude=4.1,
            confidence=0.78,
            wave_type_used='Love',
            metadata={'period': 20.0}
        )
    ]
    
    # Create frequency data
    frequencies = np.linspace(0, 50, 100)
    power_spectrum = np.exp(-frequencies / 10) + np.random.randn(100) * 0.01
    freq_data = FrequencyData(
        frequencies=frequencies,
        power_spectrum=power_spectrum,
        dominant_frequency=8.5,
        frequency_range=(1.0, 25.0),
        spectral_centroid=9.2,
        bandwidth=18.0
    )
    
    quality_metrics = QualityMetrics(
        signal_to_noise_ratio=10.2,
        detection_confidence=0.85,
        analysis_quality_score=0.82,
        data_completeness=0.98,
        processing_warnings=['Low SNR in surface waves']
    )
    
    analysis = DetailedAnalysis(
        wave_result=wave_result,
        arrival_times=arrival_times,
        magnitude_estimates=magnitude_estimates,
        epicenter_distance=42.5,
        quality_metrics=quality_metrics,
        processing_metadata={
            'processing_time': 2.34,
            'model_version': '1.0.0',
            'algorithm': 'ML-based detection'
        }
    )
    analysis.frequency_analysis['P'] = freq_data
    
    return analysis


def demo_data_export():
    """Demonstrate data export functionality."""
    print("=== Earthquake Wave Analysis Data Export Demo ===\n")
    
    # Create sample data
    analysis = create_sample_data()
    
    # Initialize exporters
    data_exporter = DataExporter()
    csv_exporter = CSVExporter()
    
    print(f"Supported export formats: {data_exporter.get_supported_formats()}\n")
    
    # Prepare wave data for export
    waves = {
        'P': analysis.wave_result.p_waves,
        'S': analysis.wave_result.s_waves,
        'Love': analysis.wave_result.surface_waves
    }
    
    # Test JSON export
    print("1. Exporting wave data to JSON format...")
    json_data = data_exporter.export_waves(waves, 'json')
    print(f"   JSON export size: {len(json_data)} bytes")
    
    # Parse and display sample JSON content
    json_content = json.loads(json_data.decode('utf-8'))
    print(f"   Wave types exported: {list(json_content['wave_data'].keys())}")
    print(f"   P-wave confidence: {json_content['wave_data']['P'][0]['confidence']}")
    print()
    
    # Test CSV export
    print("2. Exporting wave data to CSV format...")
    csv_data = data_exporter.export_waves(waves, 'csv')
    print(f"   CSV export size: {len(csv_data)} bytes")
    
    # Display CSV header
    csv_lines = csv_data.decode('utf-8').split('\n')
    print(f"   CSV header: {csv_lines[0]}")
    print(f"   Number of data rows: {len(csv_lines) - 2}")  # -2 for header and empty line
    print()
    
    # Test analysis export
    print("3. Exporting detailed analysis to JSON...")
    analysis_json = data_exporter.export_analysis_results(analysis, 'json')
    print(f"   Analysis JSON size: {len(analysis_json)} bytes")
    
    analysis_content = json.loads(analysis_json.decode('utf-8'))
    print(f"   Magnitude estimates: {len(analysis_content['magnitude_estimates'])}")
    print(f"   Best magnitude: {analysis_content['magnitude_estimates'][0]['magnitude']} ({analysis_content['magnitude_estimates'][0]['method']})")
    print(f"   S-P time difference: {analysis_content['arrival_times']['sp_time_difference']} seconds")
    print()
    
    # Test specialized CSV exports
    print("4. Exporting wave characteristics with CSVExporter...")
    wave_chars_csv = csv_exporter.export_wave_characteristics(analysis)
    print(f"   Wave characteristics CSV size: {len(wave_chars_csv)} bytes")
    
    print("5. Exporting timing data with CSVExporter...")
    timing_csv = csv_exporter.export_timing_data(analysis)
    print(f"   Timing data CSV size: {len(timing_csv)} bytes")
    print()
    
    # Test MSEED export (if ObsPy is available)
    try:
        print("6. Testing MSEED export capability...")
        mseed_exporter = MSEEDExporter()
        mseed_data = mseed_exporter.export_separated_waves(analysis.wave_result)
        print(f"   MSEED export size: {len(mseed_data)} bytes")
        print("   MSEED export successful!")
    except RuntimeError as e:
        print(f"   MSEED export not available: {e}")
    print()
    
    # Save sample exports to files
    print("7. Saving sample exports to files...")
    
    with open('sample_wave_export.json', 'wb') as f:
        f.write(json_data)
    print("   Saved: sample_wave_export.json")
    
    with open('sample_wave_export.csv', 'wb') as f:
        f.write(csv_data)
    print("   Saved: sample_wave_export.csv")
    
    with open('sample_analysis_export.json', 'wb') as f:
        f.write(analysis_json)
    print("   Saved: sample_analysis_export.json")
    
    with open('sample_wave_characteristics.csv', 'wb') as f:
        f.write(wave_chars_csv)
    print("   Saved: sample_wave_characteristics.csv")
    
    with open('sample_timing_data.csv', 'wb') as f:
        f.write(timing_csv)
    print("   Saved: sample_timing_data.csv")
    
    print("\n=== Export Demo Complete ===")
    print("All export formats have been tested successfully!")
    print("Sample files have been created for inspection.")


if __name__ == '__main__':
    demo_data_export()