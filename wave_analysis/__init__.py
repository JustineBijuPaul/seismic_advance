"""
Wave Analysis Module

This module provides comprehensive earthquake wave analysis capabilities including
P-wave, S-wave, and surface wave detection, separation, and analysis.
"""

__version__ = "1.0.0"
__author__ = "Earthquake Analysis System"

from .models import (
    WaveSegment, 
    WaveAnalysisResult, 
    DetailedAnalysis,
    ArrivalTimes,
    MagnitudeEstimate,
    FrequencyData,
    QualityMetrics
)
from .interfaces import (
    WaveDetectorInterface, 
    WaveAnalyzerInterface,
    WaveVisualizerInterface,
    WaveExporterInterface
)
from .services import (
    TimeSeriesPlotter,
    FrequencyPlotter,
    MultiChannelPlotter,
    ChannelData,
    DataExporter,
    MSEEDExporter,
    CSVExporter,
    StreamingAnalyzer,
    StreamingBuffer,
    StreamingAnalysisResult,
    AlertSystem,
    AlertEvent,
    AlertThreshold,
    AlertSeverity,
    AlertType,
    AlertHandlerInterface,
    WebSocketAlertHandler,
    LogAlertHandler,
    AnalysisCacheManager,
    CacheDecorator,
    CacheWarmingService,
    WarmingStrategy,
    PerformanceProfiler,
    MemoryMonitor,
    PerformanceMetrics,
    profile_wave_operation,
    global_profiler,
    PerformanceBenchmarkSuite,
    SyntheticDataGenerator,
    BenchmarkResult,
    ScalabilityTestResult,
    PerformanceOptimizer,
    ParallelProcessor,
    AlgorithmOptimizer,
    CacheOptimizer,
    OptimizationResult
)

__all__ = [
    'WaveSegment',
    'WaveAnalysisResult', 
    'DetailedAnalysis',
    'ArrivalTimes',
    'MagnitudeEstimate',
    'FrequencyData',
    'QualityMetrics',
    'WaveDetectorInterface',
    'WaveAnalyzerInterface',
    'WaveVisualizerInterface',
    'WaveExporterInterface',
    'TimeSeriesPlotter',
    'FrequencyPlotter',
    'MultiChannelPlotter',
    'ChannelData',
    'DataExporter',
    'MSEEDExporter',
    'CSVExporter',
    'StreamingAnalyzer',
    'StreamingBuffer',
    'StreamingAnalysisResult',
    'AlertSystem',
    'AlertEvent',
    'AlertThreshold',
    'AlertSeverity',
    'AlertType',
    'AlertHandlerInterface',
    'WebSocketAlertHandler',
    'LogAlertHandler',
    'AnalysisCacheManager',
    'CacheDecorator',
    'CacheWarmingService',
    'WarmingStrategy',
    'PerformanceProfiler',
    'MemoryMonitor',
    'PerformanceMetrics',
    'profile_wave_operation',
    'global_profiler',
    'PerformanceBenchmarkSuite',
    'SyntheticDataGenerator',
    'BenchmarkResult',
    'ScalabilityTestResult',
    'PerformanceOptimizer',
    'ParallelProcessor',
    'AlgorithmOptimizer',
    'CacheOptimizer',
    'OptimizationResult'
]