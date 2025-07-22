"""
Data models for wave analysis components.
"""

from .wave_models import (
    WaveSegment,
    WaveAnalysisResult,
    DetailedAnalysis,
    ArrivalTimes,
    MagnitudeEstimate,
    FrequencyData,
    QualityMetrics
)

__all__ = [
    'WaveSegment',
    'WaveAnalysisResult',
    'DetailedAnalysis',
    'ArrivalTimes',
    'MagnitudeEstimate',
    'FrequencyData',
    'QualityMetrics'
]