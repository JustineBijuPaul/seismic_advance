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
    SurfaceWaveDetector,
    SurfaceWaveDetectionParameters,
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
from .wave_separation_engine import WaveSeparationEngine, WaveSeparationParameters, WaveSeparationResult
from .streaming_analyzer import StreamingAnalyzer, StreamingBuffer, StreamingAnalysisResult
from .alert_system import (
    AlertSystem,
    AlertEvent,
    AlertThreshold,
    AlertSeverity,
    AlertType,
    AlertHandlerInterface,
    WebSocketAlertHandler,
    LogAlertHandler
)
from .analysis_cache import AnalysisCacheManager, CacheDecorator
from .cache_warming import CacheWarmingService, WarmingStrategy
from .performance_profiler import (
    PerformanceProfiler, MemoryMonitor, PerformanceMetrics,
    profile_wave_operation, global_profiler
)
from .performance_benchmarks import (
    PerformanceBenchmarkSuite, SyntheticDataGenerator,
    BenchmarkResult, ScalabilityTestResult
)
from .performance_optimizer import (
    PerformanceOptimizer, ParallelProcessor, AlgorithmOptimizer,
    CacheOptimizer, OptimizationResult
)
from .wave_pattern_library import (
    WavePatternLibrary,
    WavePattern,
    WavePatternType,
    PatternCategory,
    PatternComparison
)
from .pattern_comparison import PatternComparisonService, AnalysisComparison

__all__ = [
    'SignalProcessor',
    'FilterBank', 
    'WindowFunction',
    'FeatureExtractor',
    'PWaveDetector',
    'PWaveDetectionParameters',
    'SWaveDetector',
    'SWaveDetectionParameters',
    'SurfaceWaveDetector',
    'SurfaceWaveDetectionParameters',
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
    'PDFReportError',
    'WaveSeparationEngine',
    'WaveSeparationParameters',
    'WaveSeparationResult',
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
    'OptimizationResult',
    'WavePatternLibrary',
    'WavePattern',
    'WavePatternType',
    'PatternCategory',
    'PatternComparison',
    'PatternComparisonService',
    'AnalysisComparison'
]