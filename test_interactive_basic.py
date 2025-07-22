#!/usr/bin/env python3

from wave_analysis.services.interactive_chart_builder import InteractiveChartBuilder
import numpy as np
from wave_analysis.models import WaveSegment

def test_basic_functionality():
    """Test basic InteractiveChartBuilder functionality."""
    
    # Create a simple test
    builder = InteractiveChartBuilder()
    print('InteractiveChartBuilder created successfully')

    # Test basic functionality
    wave = WaveSegment(
        wave_type='P',
        start_time=1.0,
        end_time=2.0,
        data=np.random.randn(100),
        sampling_rate=100.0,
        peak_amplitude=0.5,
        dominant_frequency=10.0,
        arrival_time=1.5,
        confidence=0.8
    )

    result = builder.create_time_series_plot([wave])
    print(f'Time series plot created: {result["type"]}')

    result = builder.create_frequency_plot([wave])
    print(f'Frequency plot created: {result["type"]}')

    result = builder.create_interactive_spectrogram(wave)
    print(f'Spectrogram created: {result["type"]}')

    print('All basic tests passed!')

if __name__ == '__main__':
    test_basic_functionality()