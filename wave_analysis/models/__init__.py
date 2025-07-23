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

from .database_models import (
    WaveAnalysisRepository,
    create_wave_analysis_repository
)

__all__ = [
    'WaveSegment',
    'WaveAnalysisResult',
    'DetailedAnalysis',
    'ArrivalTimes',
    'MagnitudeEstimate',
    'FrequencyData',
    'QualityMetrics',
    'WaveAnalysisRepository',
    'create_wave_analysis_repository'
]