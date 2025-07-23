"""
Performance optimization tests.

This module tests the performance optimization capabilities including
parallel processing, algorithm optimization, and cache optimization.
"""

import pytest
import numpy as np
import time
import multiprocessing as mp
from unittest.mock import Mock, patch, MagicMock

from wave_analysis.services.performance_optimizer import (
    PerformanceOptimizer, ParallelProcessor, AlgorithmOptimizer,
    CacheOptimizer, OptimizationResult
)
from wave_analysis.services.analysis_cache import AnalysisCacheManager


class TestParallelProcessor:
    """Test parallel processing functionality."""
    
    def test_parallel_processor_initialization(self):
        """Test parallel processor initialization."""
        processor = ParallelProcessor(max_workers=4)
        assert processor.max_workers == 4
        
        # Test default initialization
        default_processor = ParallelProcessor()
        assert default_processor.max_workers == mp.cpu_count()
    
    def test_parallel_wave_detection(self):
        """Test parallel wave detection processing."""
        processor = ParallelProcessor(max_workers=2)
        
        # Create test signals
        test_signals = [
            np.random.random(1000),
            np.random.random(1000),
            np.random.random(1000)
        ]
        
        # Mock detector function
        def mock_detector(signal):
            time.sleep(0.01)  # Simulate processing time
            return [{'type': 'P', 'amplitude': np.max(signal)}]
        
        # Test thread-based processing
        results = processor.parallel_wave_detection(test_signals, mock_detector, use_processes=False)
        
        assert len(results) == 3
        for result in results:
            assert len(result) == 1
            assert result[0]['type'] == 'P'
            assert 'amplitude' in result[0]
    
    def test_parallel_frequency_analysis(self):
        """Test parallel frequency analysis."""
        processor = ParallelProcessor(max_workers=2)
        
        # Create test wave segments
        wave_segments = [
            np.random.random(500),
            np.random.random(500),
            np.random.random(500),
            np.random.random(500)
        ]
        
        # Mock analysis function
        def mock_analysis(segment):
            return {'dominant_freq': np.random.uniform(1, 20)}
        
        results = processor.parallel_frequency_analysis(wave_segments, mock_analysis, chunk_size=2)
        
        assert len(results) == 4
        for result in results:
            assert 'dominant_freq' in result
            assert 1 <= result['dominant_freq'] <= 20
    
    def test_parallel_feature_extraction(self):
        """Test parallel feature extraction."""
        processor = ParallelProcessor(max_workers=2)
        
        # Create test data windows
        data_windows = [
            np.random.random(100),
            np.random.random(100)
        ]
        
        # Mock feature extractors
        def extract_mean(data):
            return np.mean(data)
        
        def extract_std(data):
            return np.std(data)
        
        def failing_extractor(data):
            raise ValueError("Extraction failed")
        
        extractors = [extract_mean, extract_std, failing_extractor]
        
        results = processor.parallel_feature_extraction(data_windows, extractors)
        
        assert len(results) == 2
        for result in results:
            assert 'extract_mean' in result
            assert 'extract_std' in result
            assert 'failing_extractor' in result
            assert result['failing_extractor'] is None  # Should handle failures gracefully


class TestAlgorithmOptimizer:
    """Test algorithm optimization functionality."""
    
    def test_algorithm_optimizer_initialization(self):
        """Test algorithm optimizer initialization."""
        optimizer = AlgorithmOptimizer()
        
        assert optimizer.optimization_history == []
        assert hasattr(optimizer, 'profiler')
    
    def test_adaptive_processing_strategy(self):
        """Test adaptive processing strategy selection."""
        optimizer = AlgorithmOptimizer()
        
        # Test with large data
        large_data_characteristics = {
            'size_mb': 100,
            'noise_level': 0.2,
            'sampling_rate': 100.0
        }
        
        strategy = optimizer.adaptive_processing_strategy(large_data_characteristics)
        
        assert strategy['use_parallel_processing'] is True
        assert strategy['memory_optimization'] is True
        assert strategy['chunk_size'] == 2048
        
        # Test with small, noisy data
        small_noisy_characteristics = {
            'size_mb': 5,
            'noise_level': 0.4,
            'sampling_rate': 50.0
        }
        
        strategy = optimizer.adaptive_processing_strategy(small_noisy_characteristics)
        
        assert 'filter_parameters' in strategy
        assert strategy['filter_parameters']['filter_type'] == 'bandpass'
        assert strategy['detection_thresholds']['p_wave_threshold'] == 3.0
    
    def test_benchmark_algorithm_variants(self):
        """Test algorithm variant benchmarking."""
        optimizer = AlgorithmOptimizer()
        
        # Create test algorithms with more significant time differences
        def fast_algorithm(data):
            # Very fast algorithm
            return [{'type': 'P', 'confidence': 0.8}]
        
        def slow_algorithm(data):
            # Slower algorithm with actual computation
            result = np.sum(data ** 2)  # Some computation
            time.sleep(0.005)  # Small delay
            return [{'type': 'P', 'confidence': 0.9, 'computation': result}]
        
        def failing_algorithm(data):
            raise RuntimeError("Algorithm failed")
        
        algorithms = {
            'fast_algo': fast_algorithm,
            'slow_algo': slow_algorithm,
            'failing_algo': failing_algorithm
        }
        
        test_data = [np.random.random(100), np.random.random(100)]
        
        results = optimizer.benchmark_algorithm_variants(algorithms, test_data)
        
        assert 'fast_algo' in results
        assert 'slow_algo' in results
        assert 'failing_algo' in results
        
        # Verify success rates
        assert results['fast_algo']['success_rate'] == 1.0
        assert results['slow_algo']['success_rate'] == 1.0
        assert results['failing_algo']['success_rate'] == 0.0
        
        # Verify failing algorithm metrics
        assert results['failing_algo']['mean_execution_time'] == float('inf')
        
        # Both successful algorithms should have reasonable execution times
        assert results['fast_algo']['mean_execution_time'] > 0
        assert results['slow_algo']['mean_execution_time'] > 0
        
        # Throughput should be calculated correctly
        assert results['fast_algo']['throughput'] > 0
        assert results['slow_algo']['throughput'] > 0


class TestCacheOptimizer:
    """Test cache optimization functionality."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager for testing."""
        cache_manager = Mock(spec=AnalysisCacheManager)
        cache_manager.get_cache_statistics.return_value = {
            'hit_rate': 0.6,
            'memory_usage_mb': 200,
            'total_requests': 1000,
            'operation_stats': {
                'p_wave_detection': {'hit_rate': 0.2},
                'frequency_analysis': {'hit_rate': 0.95},
                's_wave_detection': {'hit_rate': 0.7}
            }
        }
        return cache_manager
    
    def test_cache_optimizer_initialization(self, mock_cache_manager):
        """Test cache optimizer initialization."""
        optimizer = CacheOptimizer(mock_cache_manager)
        
        assert optimizer.cache_manager == mock_cache_manager
        assert optimizer.access_patterns == {}
        assert optimizer.cache_hit_rates == {}
    
    def test_analyze_cache_performance(self, mock_cache_manager):
        """Test cache performance analysis."""
        optimizer = CacheOptimizer(mock_cache_manager)
        
        analysis = optimizer.analyze_cache_performance()
        
        assert analysis['overall_hit_rate'] == 0.6
        assert analysis['memory_usage_mb'] == 200
        assert analysis['total_requests'] == 1000
        assert 'recommendations' in analysis
        
        # Should have recommendations for low and high hit rates
        recommendations = analysis['recommendations']
        low_hit_rate_recs = [r for r in recommendations if r['type'] == 'low_hit_rate']
        high_hit_rate_recs = [r for r in recommendations if r['type'] == 'high_hit_rate']
        
        assert len(low_hit_rate_recs) > 0  # p_wave_detection has 0.2 hit rate
        assert len(high_hit_rate_recs) > 0  # frequency_analysis has 0.95 hit rate
    
    def test_optimize_cache_configuration(self, mock_cache_manager):
        """Test cache configuration optimization."""
        optimizer = CacheOptimizer(mock_cache_manager)
        
        config = optimizer.optimize_cache_configuration()
        
        assert 'cache_size_mb' in config
        assert 'ttl_seconds' in config
        assert 'eviction_policy' in config
        assert 'pre_warming_enabled' in config
        assert 'compression_enabled' in config
        
        # Should enable pre-warming due to low hit rate operations
        assert config['pre_warming_enabled'] is True
    
    def test_high_memory_usage_optimization(self):
        """Test optimization for high memory usage scenarios."""
        mock_cache_manager = Mock(spec=AnalysisCacheManager)
        mock_cache_manager.get_cache_statistics.return_value = {
            'hit_rate': 0.8,
            'memory_usage_mb': 1500,  # High memory usage
            'total_requests': 1000,
            'operation_stats': {}
        }
        
        optimizer = CacheOptimizer(mock_cache_manager)
        config = optimizer.optimize_cache_configuration()
        
        # Should reduce cache size and enable compression
        assert config['cache_size_mb'] <= 256
        assert config['compression_enabled'] is True


class TestPerformanceOptimizer:
    """Test main performance optimizer functionality."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create mock cache manager for testing."""
        cache_manager = Mock(spec=AnalysisCacheManager)
        cache_manager.get_cache_statistics.return_value = {
            'hit_rate': 0.7,
            'memory_usage_mb': 300,
            'total_requests': 500,
            'operation_stats': {}
        }
        return cache_manager
    
    def test_performance_optimizer_initialization(self, mock_cache_manager):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizer(mock_cache_manager)
        
        assert isinstance(optimizer.parallel_processor, ParallelProcessor)
        assert isinstance(optimizer.algorithm_optimizer, AlgorithmOptimizer)
        assert isinstance(optimizer.cache_optimizer, CacheOptimizer)
        assert hasattr(optimizer, 'profiler')
        assert optimizer.optimization_results == []
    
    def test_optimize_wave_analysis_pipeline(self, mock_cache_manager):
        """Test complete pipeline optimization."""
        optimizer = PerformanceOptimizer(mock_cache_manager)
        
        # Create test pipeline configuration
        pipeline_config = {
            'detection_parameters': {
                'p_wave_threshold': 2.0,
                's_wave_threshold': 1.8
            },
            'processing_mode': 'sequential'
        }
        
        # Create test data
        test_data = [
            np.random.random(1000),
            np.random.random(1000),
            np.random.random(1000)
        ]
        
        results = optimizer.optimize_wave_analysis_pipeline(pipeline_config, test_data)
        
        assert 'original_config' in results
        assert 'optimized_config' in results
        assert 'performance_improvements' in results
        assert 'optimization_steps' in results
        
        # Should include parallel processing optimization
        assert 'parallel_processing' in results['optimization_steps']
        
        # Should have performance improvements
        improvements = results['performance_improvements']
        assert 'speedup_factor' in improvements
        assert 'performance_improvement_percent' in improvements
    
    def test_get_optimization_recommendations(self, mock_cache_manager):
        """Test optimization recommendations generation."""
        optimizer = PerformanceOptimizer(mock_cache_manager)
        
        # Test with various system metrics scenarios
        high_cpu_metrics = {
            'cpu_usage_percent': 95,
            'memory_usage_mb': 500,
            'average_processing_time': 10
        }
        
        recommendations = optimizer.get_optimization_recommendations(high_cpu_metrics)
        
        cpu_recs = [r for r in recommendations if r['type'] == 'cpu_overutilization']
        assert len(cpu_recs) > 0
        assert cpu_recs[0]['priority'] == 'high'
        
        # Test with low CPU usage
        low_cpu_metrics = {
            'cpu_usage_percent': 20,
            'memory_usage_mb': 100,
            'average_processing_time': 5
        }
        
        recommendations = optimizer.get_optimization_recommendations(low_cpu_metrics)
        
        cpu_recs = [r for r in recommendations if r['type'] == 'cpu_underutilization']
        assert len(cpu_recs) > 0
        assert cpu_recs[0]['priority'] == 'medium'
        
        # Test with high memory usage
        high_memory_metrics = {
            'cpu_usage_percent': 50,
            'memory_usage_mb': 2500,  # High memory
            'average_processing_time': 5
        }
        
        recommendations = optimizer.get_optimization_recommendations(high_memory_metrics)
        
        memory_recs = [r for r in recommendations if r['type'] == 'high_memory_usage']
        assert len(memory_recs) > 0
        assert memory_recs[0]['priority'] == 'high'
        
        # Test with slow processing
        slow_processing_metrics = {
            'cpu_usage_percent': 50,
            'memory_usage_mb': 500,
            'average_processing_time': 45  # Slow processing
        }
        
        recommendations = optimizer.get_optimization_recommendations(slow_processing_metrics)
        
        processing_recs = [r for r in recommendations if r['type'] == 'slow_processing']
        assert len(processing_recs) > 0
        assert processing_recs[0]['priority'] == 'high'


class TestPerformanceIntegration:
    """Integration tests for performance optimization components."""
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create performance optimizer
        optimizer = PerformanceOptimizer()
        
        # Create realistic test scenario
        pipeline_config = {
            'detection_parameters': {
                'p_wave_threshold': 2.5,
                's_wave_threshold': 2.0,
                'surface_wave_threshold': 1.5
            },
            'processing_mode': 'sequential',
            'use_caching': True
        }
        
        # Generate test data of various sizes
        test_data = [
            np.random.random(500),   # Small
            np.random.random(2000),  # Medium
            np.random.random(5000),  # Large
        ]
        
        # Run optimization
        results = optimizer.optimize_wave_analysis_pipeline(pipeline_config, test_data)
        
        # Verify optimization results
        assert results['optimized_config'] != results['original_config']
        assert len(results['optimization_steps']) > 0
        
        # Should show performance improvements
        improvements = results['performance_improvements']
        assert improvements['speedup_factor'] > 0
        
        # Get system recommendations
        system_metrics = {
            'cpu_usage_percent': 60,
            'memory_usage_mb': 800,
            'average_processing_time': 15
        }
        
        recommendations = optimizer.get_optimization_recommendations(system_metrics)
        assert isinstance(recommendations, list)
    
    def test_parallel_processing_with_real_workload(self):
        """Test parallel processing with realistic workload."""
        processor = ParallelProcessor(max_workers=2)
        
        # Create computationally intensive task
        def intensive_analysis(signal):
            # Simulate complex analysis
            result = np.fft.fft(signal)
            features = {
                'peak_frequency': np.argmax(np.abs(result)),
                'spectral_centroid': np.sum(np.abs(result) * np.arange(len(result))) / np.sum(np.abs(result)),
                'spectral_rolloff': np.percentile(np.abs(result), 85)
            }
            return features
        
        # Create test signals
        signals = [np.random.random(2048) for _ in range(6)]
        
        # Time sequential processing
        start_time = time.time()
        sequential_results = [intensive_analysis(signal) for signal in signals]
        sequential_time = time.time() - start_time
        
        # Time parallel processing
        start_time = time.time()
        parallel_results = processor.parallel_frequency_analysis(signals, intensive_analysis)
        parallel_time = time.time() - start_time
        
        # Verify results are equivalent
        assert len(sequential_results) == len(parallel_results)
        
        # Parallel should be faster (though this may vary on single-core systems)
        print(f"Sequential time: {sequential_time:.3f}s, Parallel time: {parallel_time:.3f}s")
        
        # At minimum, parallel processing shouldn't be significantly slower
        assert parallel_time < sequential_time * 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])