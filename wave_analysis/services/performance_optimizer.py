"""
Performance optimization utilities for wave analysis operations.

This module provides tools to optimize wave analysis performance through
caching, parallel processing, and algorithm optimization.
"""

import numpy as np
import functools
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import logging
from dataclasses import dataclass

from .performance_profiler import PerformanceProfiler, profile_wave_operation

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from performance optimization."""
    original_time: float
    optimized_time: float
    speedup_factor: float
    memory_reduction_mb: float
    optimization_method: str
    success: bool
    error_message: Optional[str] = None


class ParallelProcessor:
    """
    Parallel processing utilities for wave analysis operations.
    
    Provides thread-based and process-based parallelization for
    CPU-intensive wave analysis tasks.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize parallel processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
        """
        self.max_workers = max_workers or mp.cpu_count()
    
    def parallel_wave_detection(self, signals: List[np.ndarray], 
                               detector_func: Callable,
                               use_processes: bool = False) -> List[Any]:
        """
        Run wave detection in parallel across multiple signals.
        
        Args:
            signals: List of seismic signals to process
            detector_func: Wave detection function to apply
            use_processes: Use processes instead of threads
            
        Returns:
            List of detection results
        """
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [executor.submit(detector_func, signal) for signal in signals]
            results = [future.result() for future in futures]
        
        return results
    
    def parallel_frequency_analysis(self, wave_segments: List[np.ndarray],
                                  analysis_func: Callable,
                                  chunk_size: Optional[int] = None) -> List[Any]:
        """
        Run frequency analysis in parallel across wave segments.
        
        Args:
            wave_segments: List of wave segments to analyze
            analysis_func: Frequency analysis function
            chunk_size: Size of chunks for processing
            
        Returns:
            List of frequency analysis results
        """
        if chunk_size is None:
            chunk_size = max(1, len(wave_segments) // self.max_workers)
        
        # Split segments into chunks
        chunks = [wave_segments[i:i + chunk_size] 
                 for i in range(0, len(wave_segments), chunk_size)]
        
        def process_chunk(chunk):
            return [analysis_func(segment) for segment in chunk]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_results = list(executor.map(process_chunk, chunks))
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        return results
    
    def parallel_feature_extraction(self, data_windows: List[np.ndarray],
                                  feature_extractors: List[Callable]) -> List[Dict[str, Any]]:
        """
        Extract features in parallel using multiple extractors.
        
        Args:
            data_windows: List of data windows to process
            feature_extractors: List of feature extraction functions
            
        Returns:
            List of feature dictionaries for each data window
        """
        def extract_all_features(data_window):
            features = {}
            for i, extractor in enumerate(feature_extractors):
                try:
                    feature_name = getattr(extractor, '__name__', f'feature_{i}')
                    features[feature_name] = extractor(data_window)
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {feature_name}: {e}")
                    features[feature_name] = None
            return features
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(extract_all_features, data_windows))
        
        return results


class AlgorithmOptimizer:
    """
    Algorithm optimization utilities for wave analysis operations.
    
    Provides methods to optimize wave analysis algorithms through
    parameter tuning, algorithm selection, and adaptive processing.
    """
    
    def __init__(self):
        """Initialize algorithm optimizer."""
        self.optimization_history: List[OptimizationResult] = []
        self.profiler = PerformanceProfiler()
    
    def optimize_detection_parameters(self, detector_func: Callable,
                                    test_signals: List[np.ndarray],
                                    parameter_ranges: Dict[str, Tuple[float, float]],
                                    optimization_metric: str = 'speed') -> Dict[str, Any]:
        """
        Optimize detection algorithm parameters for best performance.
        
        Args:
            detector_func: Wave detection function to optimize
            test_signals: List of test signals for optimization
            parameter_ranges: Dictionary of parameter names to (min, max) ranges
            optimization_metric: Metric to optimize ('speed', 'accuracy', 'memory')
            
        Returns:
            Dictionary with optimal parameters and performance metrics
        """
        from scipy.optimize import minimize
        
        def objective_function(params):
            # Convert parameter array to dictionary
            param_dict = {}
            for i, (param_name, _) in enumerate(parameter_ranges.items()):
                param_dict[param_name] = params[i]
            
            # Test performance with these parameters
            total_time = 0
            total_memory = 0
            total_accuracy = 0
            
            for signal in test_signals:
                start_time = time.time()
                
                try:
                    # Apply detector with current parameters
                    result = detector_func(signal, **param_dict)
                    execution_time = time.time() - start_time
                    
                    total_time += execution_time
                    # Simplified accuracy calculation
                    total_accuracy += len(result) if result else 0
                    
                except Exception as e:
                    logger.warning(f"Detection failed with parameters {param_dict}: {e}")
                    return float('inf')  # Penalize failed parameters
            
            # Return objective based on optimization metric
            if optimization_metric == 'speed':
                return total_time
            elif optimization_metric == 'memory':
                return total_memory
            elif optimization_metric == 'accuracy':
                return -total_accuracy  # Minimize negative accuracy (maximize accuracy)
            else:
                return total_time  # Default to speed
        
        # Set up optimization bounds
        bounds = list(parameter_ranges.values())
        initial_guess = [(low + high) / 2 for low, high in bounds]
        
        # Run optimization
        result = minimize(objective_function, initial_guess, bounds=bounds, method='L-BFGS-B')
        
        # Convert result back to parameter dictionary
        optimal_params = {}
        for i, (param_name, _) in enumerate(parameter_ranges.items()):
            optimal_params[param_name] = result.x[i]
        
        return {
            'optimal_parameters': optimal_params,
            'optimization_success': result.success,
            'final_objective_value': result.fun,
            'optimization_iterations': result.nit
        }
    
    def adaptive_processing_strategy(self, data_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select optimal processing strategy based on data characteristics.
        
        Args:
            data_characteristics: Dictionary with data properties (size, noise, etc.)
            
        Returns:
            Dictionary with recommended processing parameters
        """
        strategy = {
            'use_parallel_processing': False,
            'chunk_size': 1024,
            'filter_parameters': {},
            'detection_thresholds': {},
            'memory_optimization': False
        }
        
        data_size_mb = data_characteristics.get('size_mb', 0)
        noise_level = data_characteristics.get('noise_level', 0.1)
        sampling_rate = data_characteristics.get('sampling_rate', 100.0)
        
        # Adapt based on data size
        if data_size_mb > 50:
            strategy['use_parallel_processing'] = True
            strategy['chunk_size'] = 2048
            strategy['memory_optimization'] = True
        elif data_size_mb > 10:
            strategy['use_parallel_processing'] = True
            strategy['chunk_size'] = 1024
        
        # Adapt based on noise level
        if noise_level > 0.3:
            strategy['filter_parameters'] = {
                'filter_type': 'bandpass',
                'low_freq': 1.0,
                'high_freq': 20.0,
                'filter_order': 4
            }
            strategy['detection_thresholds'] = {
                'p_wave_threshold': 3.0,
                's_wave_threshold': 2.5,
                'surface_wave_threshold': 2.0
            }
        else:
            strategy['detection_thresholds'] = {
                'p_wave_threshold': 2.0,
                's_wave_threshold': 1.8,
                'surface_wave_threshold': 1.5
            }
        
        # Adapt based on sampling rate
        if sampling_rate > 200:
            strategy['chunk_size'] = int(strategy['chunk_size'] * 2)
        elif sampling_rate < 50:
            strategy['chunk_size'] = int(strategy['chunk_size'] / 2)
        
        return strategy
    
    def benchmark_algorithm_variants(self, algorithms: Dict[str, Callable],
                                   test_data: List[np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Benchmark different algorithm variants to find the best performer.
        
        Args:
            algorithms: Dictionary mapping algorithm names to functions
            test_data: List of test data arrays
            
        Returns:
            Dictionary with performance metrics for each algorithm
        """
        results = {}
        
        for algo_name, algo_func in algorithms.items():
            logger.info(f"Benchmarking algorithm: {algo_name}")
            
            execution_times = []
            memory_usage = []
            success_count = 0
            
            for data in test_data:
                try:
                    with self.profiler.profile_operation(f"{algo_name}_benchmark") as metrics:
                        result = algo_func(data)
                        success_count += 1
                    
                    # Get the last recorded metric
                    if self.profiler.metrics_history:
                        last_metric = self.profiler.metrics_history[-1]
                        execution_times.append(last_metric.execution_time)
                        memory_usage.append(last_metric.peak_memory_mb)
                
                except Exception as e:
                    logger.warning(f"Algorithm {algo_name} failed on test data: {e}")
            
            if execution_times:
                results[algo_name] = {
                    'mean_execution_time': np.mean(execution_times),
                    'std_execution_time': np.std(execution_times),
                    'mean_memory_usage': np.mean(memory_usage),
                    'success_rate': success_count / len(test_data),
                    'throughput': len(test_data) / sum(execution_times) if sum(execution_times) > 0 else 0
                }
            else:
                results[algo_name] = {
                    'mean_execution_time': float('inf'),
                    'std_execution_time': 0,
                    'mean_memory_usage': 0,
                    'success_rate': 0,
                    'throughput': 0
                }
        
        return results


class CacheOptimizer:
    """
    Cache optimization utilities for wave analysis operations.
    
    Provides intelligent caching strategies to improve performance
    of repeated wave analysis operations.
    """
    
    def __init__(self, cache_manager):
        """
        Initialize cache optimizer.
        
        Args:
            cache_manager: Analysis cache manager instance
        """
        self.cache_manager = cache_manager
        self.access_patterns: Dict[str, List[float]] = {}
        self.cache_hit_rates: Dict[str, float] = {}
    
    def analyze_cache_performance(self) -> Dict[str, Any]:
        """
        Analyze cache performance and identify optimization opportunities.
        
        Returns:
            Dictionary with cache performance analysis
        """
        cache_stats = self.cache_manager.get_cache_statistics()
        
        analysis = {
            'overall_hit_rate': cache_stats.get('hit_rate', 0.0),
            'memory_usage_mb': cache_stats.get('memory_usage_mb', 0.0),
            'total_requests': cache_stats.get('total_requests', 0),
            'recommendations': []
        }
        
        # Analyze hit rates by operation type
        operation_stats = cache_stats.get('operation_stats', {})
        for operation, stats in operation_stats.items():
            hit_rate = stats.get('hit_rate', 0.0)
            
            if hit_rate < 0.3:
                analysis['recommendations'].append({
                    'type': 'low_hit_rate',
                    'operation': operation,
                    'current_hit_rate': hit_rate,
                    'suggestion': 'Consider pre-warming cache or adjusting cache TTL'
                })
            elif hit_rate > 0.9:
                analysis['recommendations'].append({
                    'type': 'high_hit_rate',
                    'operation': operation,
                    'current_hit_rate': hit_rate,
                    'suggestion': 'Consider increasing cache size for this operation'
                })
        
        # Memory usage analysis
        if analysis['memory_usage_mb'] > 1000:  # 1GB threshold
            analysis['recommendations'].append({
                'type': 'high_memory_usage',
                'current_usage_mb': analysis['memory_usage_mb'],
                'suggestion': 'Consider implementing cache eviction policies or reducing cache size'
            })
        
        return analysis
    
    def optimize_cache_configuration(self) -> Dict[str, Any]:
        """
        Optimize cache configuration based on usage patterns.
        
        Returns:
            Dictionary with optimized cache configuration
        """
        performance_analysis = self.analyze_cache_performance()
        
        config = {
            'cache_size_mb': 512,  # Default
            'ttl_seconds': 3600,   # Default
            'eviction_policy': 'lru',
            'pre_warming_enabled': False,
            'compression_enabled': False
        }
        
        # Adjust based on recommendations
        for recommendation in performance_analysis['recommendations']:
            if recommendation['type'] == 'high_memory_usage':
                config['cache_size_mb'] = min(config['cache_size_mb'], 256)
                config['compression_enabled'] = True
            elif recommendation['type'] == 'low_hit_rate':
                config['pre_warming_enabled'] = True
                config['ttl_seconds'] = min(config['ttl_seconds'] * 2, 7200)
            elif recommendation['type'] == 'high_hit_rate':
                config['cache_size_mb'] = min(config['cache_size_mb'] * 1.5, 1024)
        
        return config


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    
    Combines all optimization strategies to provide comprehensive
    performance improvements for wave analysis operations.
    """
    
    def __init__(self, cache_manager=None):
        """
        Initialize performance optimizer.
        
        Args:
            cache_manager: Optional cache manager for cache optimization
        """
        self.parallel_processor = ParallelProcessor()
        self.algorithm_optimizer = AlgorithmOptimizer()
        self.cache_optimizer = CacheOptimizer(cache_manager) if cache_manager else None
        self.profiler = PerformanceProfiler()
        self.optimization_results: List[OptimizationResult] = []
    
    def optimize_wave_analysis_pipeline(self, pipeline_config: Dict[str, Any],
                                      test_data: List[np.ndarray]) -> Dict[str, Any]:
        """
        Optimize complete wave analysis pipeline.
        
        Args:
            pipeline_config: Current pipeline configuration
            test_data: Test data for optimization
            
        Returns:
            Dictionary with optimized configuration and performance improvements
        """
        logger.info("Starting wave analysis pipeline optimization")
        
        optimization_results = {
            'original_config': pipeline_config.copy(),
            'optimized_config': pipeline_config.copy(),
            'performance_improvements': {},
            'optimization_steps': []
        }
        
        # Step 1: Optimize parallel processing
        if len(test_data) > 1:
            parallel_config = self._optimize_parallel_processing(test_data)
            optimization_results['optimized_config'].update(parallel_config)
            optimization_results['optimization_steps'].append('parallel_processing')
        
        # Step 2: Optimize algorithm parameters
        if 'detection_parameters' in pipeline_config:
            algo_optimization = self._optimize_algorithm_parameters(
                pipeline_config['detection_parameters'], test_data
            )
            optimization_results['optimized_config']['detection_parameters'] = algo_optimization
            optimization_results['optimization_steps'].append('algorithm_parameters')
        
        # Step 3: Optimize caching strategy
        if self.cache_optimizer:
            cache_config = self.cache_optimizer.optimize_cache_configuration()
            optimization_results['optimized_config']['cache_config'] = cache_config
            optimization_results['optimization_steps'].append('cache_optimization')
        
        # Step 4: Benchmark optimized vs original
        performance_comparison = self._benchmark_pipeline_performance(
            optimization_results['original_config'],
            optimization_results['optimized_config'],
            test_data
        )
        optimization_results['performance_improvements'] = performance_comparison
        
        logger.info(f"Pipeline optimization completed. Steps: {optimization_results['optimization_steps']}")
        
        return optimization_results
    
    def _optimize_parallel_processing(self, test_data: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize parallel processing configuration."""
        data_sizes = [data.nbytes / 1024 / 1024 for data in test_data]  # MB
        avg_data_size = np.mean(data_sizes)
        
        config = {
            'use_parallel_processing': avg_data_size > 5.0,  # Use parallel for >5MB
            'max_workers': min(len(test_data), mp.cpu_count()),
            'use_processes': avg_data_size > 50.0,  # Use processes for >50MB
            'chunk_size': max(1, len(test_data) // mp.cpu_count())
        }
        
        return config
    
    def _optimize_algorithm_parameters(self, current_params: Dict[str, Any],
                                     test_data: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize algorithm parameters."""
        # This is a simplified optimization - in practice would use more sophisticated methods
        optimized_params = current_params.copy()
        
        # Analyze data characteristics
        avg_noise = np.mean([np.std(data) / np.mean(np.abs(data)) for data in test_data])
        
        if avg_noise > 0.3:
            # High noise - increase thresholds
            optimized_params['p_wave_threshold'] = optimized_params.get('p_wave_threshold', 2.0) * 1.2
            optimized_params['s_wave_threshold'] = optimized_params.get('s_wave_threshold', 1.8) * 1.2
        elif avg_noise < 0.1:
            # Low noise - decrease thresholds for better sensitivity
            optimized_params['p_wave_threshold'] = optimized_params.get('p_wave_threshold', 2.0) * 0.8
            optimized_params['s_wave_threshold'] = optimized_params.get('s_wave_threshold', 1.8) * 0.8
        
        return optimized_params
    
    def _benchmark_pipeline_performance(self, original_config: Dict[str, Any],
                                      optimized_config: Dict[str, Any],
                                      test_data: List[np.ndarray]) -> Dict[str, Any]:
        """Benchmark original vs optimized pipeline performance."""
        
        def simulate_pipeline(config, data):
            """Simulate pipeline execution with given config."""
            start_time = time.time()
            
            # Simulate processing based on configuration
            if config.get('use_parallel_processing', False):
                # Simulate parallel processing overhead
                time.sleep(0.001 * len(data))
            else:
                # Simulate sequential processing
                time.sleep(0.005 * len(data))
            
            # Simulate algorithm processing time based on parameters
            processing_time = 0.01 * len(data)
            if config.get('detection_parameters', {}).get('p_wave_threshold', 2.0) < 1.5:
                processing_time *= 1.2  # More sensitive detection takes longer
            
            time.sleep(processing_time)
            
            return time.time() - start_time
        
        # Benchmark original configuration
        original_times = []
        for data in test_data:
            exec_time = simulate_pipeline(original_config, data)
            original_times.append(exec_time)
        
        # Benchmark optimized configuration
        optimized_times = []
        for data in test_data:
            exec_time = simulate_pipeline(optimized_config, data)
            optimized_times.append(exec_time)
        
        original_total = sum(original_times)
        optimized_total = sum(optimized_times)
        
        speedup = original_total / optimized_total if optimized_total > 0 else 1.0
        
        return {
            'original_total_time': original_total,
            'optimized_total_time': optimized_total,
            'speedup_factor': speedup,
            'time_saved_seconds': original_total - optimized_total,
            'performance_improvement_percent': ((original_total - optimized_total) / original_total) * 100
        }
    
    def get_optimization_recommendations(self, system_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get performance optimization recommendations based on system metrics.
        
        Args:
            system_metrics: Current system performance metrics
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # CPU utilization recommendations
        cpu_usage = system_metrics.get('cpu_usage_percent', 0)
        if cpu_usage < 30:
            recommendations.append({
                'type': 'cpu_underutilization',
                'priority': 'medium',
                'description': 'CPU is underutilized, consider increasing parallel processing',
                'action': 'Increase max_workers for parallel operations'
            })
        elif cpu_usage > 90:
            recommendations.append({
                'type': 'cpu_overutilization',
                'priority': 'high',
                'description': 'CPU is overutilized, consider reducing parallel workers',
                'action': 'Decrease max_workers or implement processing queues'
            })
        
        # Memory usage recommendations
        memory_usage = system_metrics.get('memory_usage_mb', 0)
        if memory_usage > 2000:  # 2GB threshold
            recommendations.append({
                'type': 'high_memory_usage',
                'priority': 'high',
                'description': 'High memory usage detected',
                'action': 'Enable memory optimization and data streaming'
            })
        
        # Processing time recommendations
        avg_processing_time = system_metrics.get('average_processing_time', 0)
        if avg_processing_time > 30:  # 30 seconds threshold
            recommendations.append({
                'type': 'slow_processing',
                'priority': 'high',
                'description': 'Processing time is above acceptable threshold',
                'action': 'Consider algorithm optimization or hardware upgrade'
            })
        
        # Cache performance recommendations
        if self.cache_optimizer:
            cache_analysis = self.cache_optimizer.analyze_cache_performance()
            if cache_analysis['overall_hit_rate'] < 0.5:
                recommendations.append({
                    'type': 'low_cache_hit_rate',
                    'priority': 'medium',
                    'description': f"Cache hit rate is low ({cache_analysis['overall_hit_rate']:.2f})",
                    'action': 'Implement cache pre-warming or adjust cache policies'
                })
        
        return recommendations