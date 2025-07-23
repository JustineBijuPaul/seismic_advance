"""
Example demonstrating the usage of the wave analysis caching system.

This script shows how to integrate and use the caching system with
wave analysis operations for improved performance.
"""

import os
import sys
import numpy as np
from datetime import datetime
import logging

# Add the parent directory to the path to import wave_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_analysis.services.analysis_cache import AnalysisCacheManager
from wave_analysis.services.cache_warming import CacheWarmingService, WarmingStrategy
from wave_analysis.services.cached_wave_analyzer import CachedWaveAnalyzer
from wave_analysis.models.wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis,
    ArrivalTimes, MagnitudeEstimate, QualityMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample seismic data for demonstration."""
    # Generate synthetic seismic data
    duration = 60  # seconds
    sampling_rate = 100  # Hz
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    # Create a synthetic earthquake signal
    # P-wave arrival at 10 seconds
    p_wave_start = 10
    p_wave_signal = np.exp(-(t - p_wave_start)**2 / 0.5) * np.sin(2 * np.pi * 8 * (t - p_wave_start))
    p_wave_signal[t < p_wave_start] = 0
    
    # S-wave arrival at 18 seconds
    s_wave_start = 18
    s_wave_signal = np.exp(-(t - s_wave_start)**2 / 2) * np.sin(2 * np.pi * 4 * (t - s_wave_start))
    s_wave_signal[t < s_wave_start] = 0
    
    # Surface waves at 25 seconds
    surface_wave_start = 25
    surface_wave_signal = np.exp(-(t - surface_wave_start)**2 / 5) * np.sin(2 * np.pi * 1 * (t - surface_wave_start))
    surface_wave_signal[t < surface_wave_start] = 0
    
    # Combine signals with noise
    noise = np.random.normal(0, 0.1, len(t))
    seismic_data = p_wave_signal + s_wave_signal + surface_wave_signal + noise
    
    return seismic_data, sampling_rate


def create_sample_analysis_result(seismic_data, sampling_rate):
    """Create a sample analysis result for demonstration."""
    # Create wave segments
    p_wave = WaveSegment(
        wave_type='P',
        start_time=10.0,
        end_time=15.0,
        data=seismic_data[1000:1500],  # 10-15 seconds at 100 Hz
        sampling_rate=sampling_rate,
        peak_amplitude=0.8,
        dominant_frequency=8.0,
        arrival_time=10.2,
        confidence=0.9
    )
    
    s_wave = WaveSegment(
        wave_type='S',
        start_time=18.0,
        end_time=23.0,
        data=seismic_data[1800:2300],  # 18-23 seconds at 100 Hz
        sampling_rate=sampling_rate,
        peak_amplitude=1.2,
        dominant_frequency=4.0,
        arrival_time=18.5,
        confidence=0.85
    )
    
    surface_wave = WaveSegment(
        wave_type='Rayleigh',
        start_time=25.0,
        end_time=35.0,
        data=seismic_data[2500:3500],  # 25-35 seconds at 100 Hz
        sampling_rate=sampling_rate,
        peak_amplitude=1.5,
        dominant_frequency=1.0,
        arrival_time=25.3,
        confidence=0.8
    )
    
    # Create wave analysis result
    wave_result = WaveAnalysisResult(
        original_data=seismic_data,
        sampling_rate=sampling_rate,
        p_waves=[p_wave],
        s_waves=[s_wave],
        surface_waves=[surface_wave]
    )
    
    # Create detailed analysis
    arrival_times = ArrivalTimes(
        p_wave_arrival=10.2,
        s_wave_arrival=18.5,
        surface_wave_arrival=25.3,
        sp_time_difference=8.3
    )
    
    magnitude_estimate = MagnitudeEstimate(
        method='ML',
        magnitude=4.8,
        confidence=0.85,
        wave_type_used='P'
    )
    
    quality_metrics = QualityMetrics(
        signal_to_noise_ratio=15.2,
        detection_confidence=0.85,
        analysis_quality_score=0.9,
        data_completeness=1.0
    )
    
    detailed_analysis = DetailedAnalysis(
        wave_result=wave_result,
        arrival_times=arrival_times,
        magnitude_estimates=[magnitude_estimate],
        quality_metrics=quality_metrics
    )
    
    return detailed_analysis


def demonstrate_basic_caching():
    """Demonstrate basic caching operations."""
    print("\n=== Basic Caching Demonstration ===")
    
    # Create mock database (in real usage, this would be your MongoDB instance)
    from unittest.mock import Mock
    mock_db = Mock()
    mock_collection = Mock()
    mock_db.analysis_cache = mock_collection
    
    # Mock collection methods
    mock_collection.create_index = Mock()
    mock_collection.replace_one = Mock()
    mock_collection.find_one = Mock(return_value=None)
    mock_collection.delete_many = Mock()
    mock_collection.aggregate = Mock(return_value=[])
    mock_collection.update_one = Mock()
    
    # Create cache manager
    cache_manager = AnalysisCacheManager(
        mongodb=mock_db,
        redis_client=None,  # No Redis for this example
        default_ttl_hours=24,
        max_memory_cache_size=100
    )
    
    # Create sample data
    seismic_data, sampling_rate = create_sample_data()
    detailed_analysis = create_sample_analysis_result(seismic_data, sampling_rate)
    
    file_id = "example_earthquake_001"
    operation = "detailed_analysis"
    
    print(f"Caching analysis result for file: {file_id}")
    
    # Cache the analysis result
    start_time = datetime.now()
    cache_key = cache_manager.cache_analysis_result(
        operation, file_id, detailed_analysis
    )
    cache_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Cached in {cache_time:.4f} seconds with key: {cache_key[:16]}...")
    
    # Retrieve from cache
    start_time = datetime.now()
    cached_result = cache_manager.get_cached_result(operation, file_id)
    retrieve_time = (datetime.now() - start_time).total_seconds()
    
    print(f"Retrieved from cache in {retrieve_time:.6f} seconds")
    print(f"Cache hit: {cached_result is not None}")
    
    if cached_result:
        print(f"P-wave arrival time: {cached_result.arrival_times.p_wave_arrival}s")
        print(f"Magnitude estimate: {cached_result.magnitude_estimates[0].magnitude}")
        print(f"Quality score: {cached_result.quality_metrics.analysis_quality_score}")
    
    # Get cache statistics
    stats = cache_manager.get_cache_statistics()
    print(f"\nCache Statistics:")
    print(f"  Hit rate: {stats['performance']['hit_rate']:.2%}")
    print(f"  Total hits: {stats['performance']['total_hits']}")
    print(f"  Memory cache entries: {stats['storage']['memory_cache']['entries']}")


def demonstrate_cache_with_parameters():
    """Demonstrate caching with different parameters."""
    print("\n=== Parameter-based Caching Demonstration ===")
    
    # Create mock database
    from unittest.mock import Mock
    mock_db = Mock()
    mock_collection = Mock()
    mock_db.analysis_cache = mock_collection
    mock_collection.create_index = Mock()
    mock_collection.replace_one = Mock()
    mock_collection.find_one = Mock(return_value=None)
    mock_collection.delete_many = Mock()
    mock_collection.aggregate = Mock(return_value=[])
    mock_collection.update_one = Mock()
    
    cache_manager = AnalysisCacheManager(mongodb=mock_db)
    
    # Create sample data
    seismic_data, sampling_rate = create_sample_data()
    detailed_analysis = create_sample_analysis_result(seismic_data, sampling_rate)
    
    file_id = "example_earthquake_002"
    operation = "wave_separation"
    
    # Cache with different parameters
    parameters_sets = [
        {'threshold': 0.5, 'window_size': 100, 'filter_type': 'bandpass'},
        {'threshold': 0.7, 'window_size': 100, 'filter_type': 'bandpass'},
        {'threshold': 0.5, 'window_size': 200, 'filter_type': 'bandpass'},
        {'threshold': 0.5, 'window_size': 100, 'filter_type': 'highpass'},
    ]
    
    print(f"Caching results with different parameters for file: {file_id}")
    
    for i, params in enumerate(parameters_sets):
        cache_key = cache_manager.cache_analysis_result(
            operation, file_id, detailed_analysis, params
        )
        print(f"  Cached with params {i+1}: {params}")
    
    # Test retrieval with matching parameters
    print("\nTesting parameter-specific retrieval:")
    
    for i, params in enumerate(parameters_sets):
        result = cache_manager.get_cached_result(operation, file_id, params)
        print(f"  Params {i+1} - Cache hit: {result is not None}")
    
    # Test with non-matching parameters
    different_params = {'threshold': 0.9, 'window_size': 50, 'filter_type': 'lowpass'}
    result = cache_manager.get_cached_result(operation, file_id, different_params)
    print(f"  Different params - Cache hit: {result is not None}")


def demonstrate_cache_invalidation():
    """Demonstrate cache invalidation strategies."""
    print("\n=== Cache Invalidation Demonstration ===")
    
    # Create mock database
    from unittest.mock import Mock
    mock_db = Mock()
    mock_collection = Mock()
    mock_db.analysis_cache = mock_collection
    mock_collection.create_index = Mock()
    mock_collection.replace_one = Mock()
    mock_collection.find_one = Mock(return_value=None)
    mock_collection.delete_many = Mock(return_value=Mock(deleted_count=2))
    mock_collection.aggregate = Mock(return_value=[])
    mock_collection.update_one = Mock()
    mock_collection.find = Mock(return_value=[
        {'cache_key': 'key1'}, {'cache_key': 'key2'}
    ])
    
    cache_manager = AnalysisCacheManager(mongodb=mock_db)
    
    # Create sample data
    seismic_data, sampling_rate = create_sample_data()
    detailed_analysis = create_sample_analysis_result(seismic_data, sampling_rate)
    
    # Cache multiple results
    files_and_operations = [
        ("earthquake_001", "wave_separation"),
        ("earthquake_001", "detailed_analysis"),
        ("earthquake_002", "wave_separation"),
        ("earthquake_003", "detailed_analysis"),
    ]
    
    print("Caching multiple analysis results...")
    for file_id, operation in files_and_operations:
        cache_manager.cache_analysis_result(operation, file_id, detailed_analysis)
        print(f"  Cached {operation} for {file_id}")
    
    # Test selective invalidation
    print("\nTesting selective invalidation:")
    
    # Invalidate by file_id
    invalidated_count = cache_manager.invalidate_cache(file_id="earthquake_001")
    print(f"  Invalidated {invalidated_count} entries for earthquake_001")
    
    # Invalidate by operation
    invalidated_count = cache_manager.invalidate_cache(operation="wave_separation")
    print(f"  Invalidated {invalidated_count} entries for wave_separation operation")


def demonstrate_cache_warming():
    """Demonstrate cache warming strategies."""
    print("\n=== Cache Warming Demonstration ===")
    
    # Create mock database
    from unittest.mock import Mock
    mock_db = Mock()
    mock_collection = Mock()
    mock_db.analysis_cache = mock_collection
    mock_collection.create_index = Mock()
    mock_collection.replace_one = Mock()
    mock_collection.find_one = Mock(return_value=None)
    mock_collection.delete_many = Mock()
    mock_collection.aggregate = Mock(return_value=[
        {'_id': f'file_{i}', 'total_accesses': 10 - i} for i in range(5)
    ])
    mock_collection.update_one = Mock()
    
    # Mock GridFS and wave analyses collections
    mock_db.fs = Mock()
    mock_db.fs.files = Mock()
    mock_db.fs.files.aggregate = Mock(return_value=[
        {'_id': f'recent_file_{i}'} for i in range(10)
    ])
    
    mock_db.wave_analyses = Mock()
    mock_db.wave_analyses.aggregate = Mock(return_value=[
        {'file_id': f'quality_file_{i}'} for i in range(8)
    ])
    
    cache_manager = AnalysisCacheManager(mongodb=mock_db)
    
    # Create warming service
    warming_service = CacheWarmingService(
        cache_manager=cache_manager,
        db=mock_db
    )
    
    # Mock the warm_cache method
    def mock_warm_cache(functions, file_ids, max_concurrent=5):
        return {
            'operations_attempted': len(file_ids) * len(functions),
            'operations_successful': len(file_ids) * len(functions) - 1,
            'operations_failed': 1,
            'total_time': 2.5
        }
    
    cache_manager.warm_cache = mock_warm_cache
    
    print("Executing cache warming strategies...")
    
    # Test individual strategies
    strategies_to_test = ["recent_files", "high_quality_analyses", "frequently_accessed"]
    
    for strategy_name in strategies_to_test:
        strategy = WarmingStrategy(
            name=strategy_name,
            description=f"Test {strategy_name} strategy",
            priority=1,
            max_files=5
        )
        
        result = warming_service.execute_warming_strategy(strategy)
        print(f"  {strategy_name}: {result['status']}, files warmed: {result.get('files_warmed', 0)}")
    
    # Test comprehensive warming
    print("\nExecuting comprehensive warming...")
    overall_result = warming_service.execute_all_strategies()
    print(f"  Total strategies executed: {overall_result['strategies_executed']}")
    print(f"  Total files warmed: {overall_result['total_files_warmed']}")
    print(f"  Duration: {overall_result['duration_seconds']:.2f} seconds")


def demonstrate_performance_benefits():
    """Demonstrate the performance benefits of caching."""
    print("\n=== Performance Benefits Demonstration ===")
    
    # Create mock database
    from unittest.mock import Mock
    mock_db = Mock()
    mock_collection = Mock()
    mock_db.analysis_cache = mock_collection
    mock_collection.create_index = Mock()
    mock_collection.replace_one = Mock()
    mock_collection.find_one = Mock(return_value=None)
    mock_collection.delete_many = Mock()
    mock_collection.aggregate = Mock(return_value=[])
    mock_collection.update_one = Mock()
    
    cache_manager = AnalysisCacheManager(mongodb=mock_db)
    
    # Create sample data
    seismic_data, sampling_rate = create_sample_data()
    detailed_analysis = create_sample_analysis_result(seismic_data, sampling_rate)
    
    file_id = "performance_test_file"
    operation = "detailed_analysis"
    
    # Simulate expensive computation
    def expensive_analysis():
        """Simulate an expensive analysis operation."""
        import time
        time.sleep(0.1)  # Simulate 100ms computation
        return detailed_analysis
    
    print("Comparing performance with and without caching...")
    
    # First run - no cache (expensive)
    start_time = datetime.now()
    result1 = expensive_analysis()
    first_run_time = (datetime.now() - start_time).total_seconds()
    
    # Cache the result
    cache_manager.cache_analysis_result(operation, file_id, result1)
    
    # Second run - from cache (fast)
    start_time = datetime.now()
    result2 = cache_manager.get_cached_result(operation, file_id)
    second_run_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  First run (no cache): {first_run_time:.4f} seconds")
    print(f"  Second run (cached): {second_run_time:.6f} seconds")
    
    if second_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"  Speedup: {speedup:.1f}x")
    else:
        print(f"  Speedup: >1000x (cache access was too fast to measure)")
    
    print(f"  Time saved: {(first_run_time - second_run_time) * 1000:.1f}ms")
    
    # Simulate multiple cache hits
    print("\nSimulating multiple cache accesses...")
    cache_times = []
    for i in range(10):
        start_time = datetime.now()
        result = cache_manager.get_cached_result(operation, file_id)
        cache_time = (datetime.now() - start_time).total_seconds()
        cache_times.append(cache_time)
    
    avg_cache_time = sum(cache_times) / len(cache_times)
    print(f"  Average cache access time: {avg_cache_time:.6f} seconds")
    
    if avg_cache_time > 0:
        consistent_speedup = first_run_time / avg_cache_time
        print(f"  Consistent speedup: {consistent_speedup:.1f}x")
    else:
        print(f"  Consistent speedup: >1000x (cache access was too fast to measure)")


def main():
    """Main function to run all demonstrations."""
    print("Wave Analysis Caching System Demonstration")
    print("=" * 50)
    
    try:
        demonstrate_basic_caching()
        demonstrate_cache_with_parameters()
        demonstrate_cache_invalidation()
        demonstrate_cache_warming()
        demonstrate_performance_benefits()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nKey Benefits of the Caching System:")
        print("  • Significant performance improvements (10x+ speedup)")
        print("  • Parameter-aware caching for different analysis configurations")
        print("  • Intelligent cache invalidation strategies")
        print("  • Proactive cache warming for common scenarios")
        print("  • Multi-tier storage (memory, MongoDB, Redis)")
        print("  • Comprehensive monitoring and statistics")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()