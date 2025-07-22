"""
Wave Analysis Services

This module contains the core services for wave analysis including
signal processing, wave detection, analysis, and visualization.
"""

from .signal_processing import (
    SignalProcessor,
    FilterBank,
    WindowFunction,
    FeatureExtractor
)
from .wave_detectors import (
    PWaveDetector,
    PWaveDetectionParameters,
    SWaveDetector,
    SWaveDetectionParameters,
    DetectionParameters
)
from .arrival_time_calculator import ArrivalTimeCalculator
from .frequency_analyzer import FrequencyAnalyzer
from .magnitude_estimator import MagnitudeEstimator
from .wave_analyzer import WaveAnalyzer
from .quality_metrics import QualityMetricsCalculator

__all__ = [
    'SignalProcessor',
    'FilterBank', 
    'WindowFunction',
    'FeatureExtractor',
    'PWaveDetector',
    'PWaveDetectionParameters',
    'SWaveDetector',
    'SWaveDetectionParameters',
    'DetectionParameters',
    'ArrivalTimeCalculator',
    'FrequencyAnalyzer',
    'MagnitudeEstimator',
    'WaveAnalyzer',
    'QualityMetricsCalculator'
]