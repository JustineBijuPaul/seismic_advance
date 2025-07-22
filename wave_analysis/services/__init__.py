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
from .time_series_plotter import TimeSeriesPlotter
from .frequency_plotter import FrequencyPlotter
from .multi_channel_plotter import MultiChannelPlotter, ChannelData
from .data_exporter import DataExporter, MSEEDExporter, CSVExporter
from .pdf_report_generator import PDFReportGenerator, PDFReportError

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
    'QualityMetricsCalculator',
    'TimeSeriesPlotter',
    'FrequencyPlotter',
    'MultiChannelPlotter',
    'ChannelData',
    'DataExporter',
    'MSEEDExporter',
    'CSVExporter',
    'PDFReportGenerator',
    'PDFReportError'
]