"""
Performance monitoring and optimization tests.

This module tests the performance profiling, memory monitoring,
and benchmarking capabilities of the wave analysis system.
"""

import pytest
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from wave_analysis.services.performance_profiler import (
    PerformanceProfiler, MemoryMonitor, PerformanceMetrics,
    profile_wave_operation, global_profiler
)
from wave_analysis.services.performance_benchmarks import (
    PerformanceBenchmarkSuite, SyntheticDataGenerator,
    BenchmarkResult, ScalabilityTestResult
)


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(sampling_interval=0.05)
        
        assert monitor.sampling_interval == 0.05
        assert monitor.snapshots == []
        assert not monitor.monitoring
        assert monitor.monitor_thread is None
    
    def test_memory_monitoring_lifecycle(self):
        """Test memory monitoring start/stop lifecycle."""
        monitor = MemoryMonitor(sampling_interval=0.01)
        
        # Start monitoring
        monitor.start_monitoring()
        assert monitor.monitoring
        assert monitor.monitor_thread is not None
        
        # Let it collect some samples
        time.sleep(0.05)
        
        # Stop monitoring
        monitor.stop_monitoring()
        assert not monitor.monitoring
        
        # Should have collected some snapshots
        assert len(monitor.snapshots) > 0
    
    def test_peak_memory_calculation(self):
        """Test peak memory usage calculation."""
        monitor = MemoryMonitor(sampling_interval=0.01)
        
        # Start monitoring and allocate memory
        monitor.start_monitoring()
        
        # Allocate some memory to create variation
        large_array = np.random.random((1000, 1000))
        time.sleep(0.02)
        
        # Allocate more memory
        larger_array = np.random.random((2000, 1000))
        time.sleep(0.02)
        
        monitor.stop_monitoring()
        
        # Check peak memory
        peak_memory = monitor.get_peak_memory()
        average_memory = monitor.get_average_memory()
        
        assert peak_memory > 0
        assert average_memory > 0
        assert peak_memory >= average_memory
        
        # Clean up
        del large_array, larger_array
    
    def test_memory_trend_analysis(self):
        """Test memory usage trend analysis."""
        monitor = MemoryMonitor(sampling_interval=0.01)
        
        monitor.start_monitoring()
        time.sleep(0.05)
        monitor.stop_monitoring()
        
        trend_data = monitor.get_memory_trend()
        
        assert 'trend' in trend_data
        assert 'volatility' in trend_data
        assert 'min_memory' in trend_data
        assert 'max_memory' in trend_data
        
        assert trend_data['max_memory'] >= trend_data['min_memory']


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()
        
        assert profiler.metrics_history == []
        assert isinstance(profiler.memory_monitor, MemoryMonitor)
        assert profiler.active_operations == {}
    
    def test_profile_operation_context_manager(self):
        """Test operation profiling context manager."""
        profiler = PerformanceProfiler()
        
        with profiler.profile_operation("test_operation", data_size_mb=1.0) as metrics:
            # Simulate some work
            time.sleep(0.01)
            test_array = np.random.random((100, 100))
            metrics['test_metric'] = 42
        
        # Check that metrics were recorded
        assert len(profiler.metrics_history) == 1
        
        recorded_metric = profiler.metrics_history[0]
        assert recorded_metric.operation_name == "test_operation"
        assert recorded_metric.execution_time > 0
        assert recorded_metric.data_size_mb == 1.0
        assert recorded_metric.additional_metrics['test_metric'] == 42
    
    def test_profile_function_decorator(self):
        """Test function profiling decorator."""
        profiler = PerformanceProfiler()
        
        @profiler.profile_function("test_function", data_size_mb=2.0)
        def test_function(x, y):
            time.sleep(0.01)
            return x + y
        
        result = test_function(5, 3)
        
        assert result == 8
        assert len(profiler.metrics_history) == 1
        
        recorded_metric = profiler.metrics_history[0]
        assert recorded_metric.operation_name == "test_function"
        assert recorded_metric.execution_time > 0
        assert recorded_metric.data_size_mb == 2.0
    
    def test_operation_statistics(self):
        """Test operation statistics calculation."""
        profiler = PerformanceProfiler()
        
        # Record multiple operations
        for i in range(5):
            with profiler.profile_operation("repeated_operation", data_size_mb=1.0):
                time.sleep(0.01 * (i + 1))  # Variable execution time
        
        stats = profiler.get_operation_stats("repeated_operation")
        
        assert stats['operation_name'] == "repeated_operation"
        assert stats['total_executions'] == 5
        assert 'execution_time' in stats
        assert 'memory_usage' in stats
        
        # Check statistical measures
        exec_stats = stats['execution_time']
        assert 'mean' in exec_stats
        assert 'median' in exec_stats
        assert 'std' in exec_stats
        assert 'min' in exec_stats
        assert 'max' in exec_stats
    
    def test_performance_summary(self):
        """Test comprehensive performance summary."""
        profiler = PerformanceProfiler()
        
        # Record different operations
        with profiler.profile_operation("operation_a", data_size_mb=1.0):
            time.sleep(0.01)
        
        with profiler.profile_operation("operation_b", data_size_mb=2.0):
            time.sleep(0.02)
        
        summary = profiler.get_performance_summary()
        
        assert summary['total_operations'] == 2
        assert summary['unique_operations'] == 2
        assert 'operations' in summary
        assert 'overall' in summary
        
        assert 'operation_a' in summary['operations']
        assert 'operation_b' in summary['operations']
    
    def test_global_profiler_decorator(self):
        """Test global profiler decorator."""
        @profile_wave_operation("global_test")
        def test_global_function():
            time.sleep(0.01)
            return "test_result"
        
        result = test_global_function()
        
        assert result == "test_result"
        # Check that global profiler recorded the operation
        assert len(global_profiler.metrics_history) > 0


class TestSyntheticDataGenerator:
    """Test synthetic data generation for benchmarking."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = SyntheticDataGenerator(sampling_rate=50.0)
        assert generator.sampling_rate == 50.0
    
    def test_p_wave_generation(self):
        """Test P-wave signal generation."""
        generator = SyntheticDataGenerator()
        
        p_wave = generator.generate_p_wave(duration=2.0, amplitude=1.0, frequency=10.0)
        
        assert len(p_wave) == 200  # 2 seconds at 100 Hz
        assert np.max(np.abs(p_wave)) > 0
        
        # Check frequency content (simplified)
        fft = np.fft.fft(p_wave)
        freqs = np.fft.fftfreq(len(p_wave), 1/100.0)
        dominant_freq_idx = np.argmax(np.abs(fft[:len(fft)//2]))
        dominant_freq = abs(freqs[dominant_freq_idx])
        
        # Should be close to target frequency (within reasonable tolerance)
        assert abs(dominant_freq - 10.0) < 5.0
    
    def test_s_wave_generation(self):
        """Test S-wave signal generation."""
        generator = SyntheticDataGenerator()
        
        s_wave = generator.generate_s_wave(duration=3.0, amplitude=1.5, frequency=5.0)
        
        assert len(s_wave) == 300  # 3 seconds at 100 Hz
        assert np.max(np.abs(s_wave)) > 0
        
        # S-wave should generally have larger amplitude than P-wave
        p_wave = generator.generate_p_wave(duration=3.0, amplitude=1.0, frequency=10.0)
        assert np.max(np.abs(s_wave)) > np.max(np.abs(p_wave)) * 0.8
    
    def test_surface_wave_generation(self):
        """Test surface wave signal generation."""
        generator = SyntheticDataGenerator()
        
        surface_wave = generator.generate_surface_wave(duration=5.0, amplitude=2.0, frequency=1.0)
        
        assert len(surface_wave) == 500  # 5 seconds at 100 Hz
        assert np.max(np.abs(surface_wave)) > 0
    
    def test_complete_earthquake_generation(self):
        """Test complete earthquake signal generation."""
        generator = SyntheticDataGenerator()
        
        signal, ground_truth = generator.generate_complete_earthquake(
            total_duration=30.0,
            p_arrival=5.0,
            s_arrival=10.0,
            surface_arrival=20.0
        )
        
        assert len(signal) == 3000  # 30 seconds at 100 Hz
        assert np.max(np.abs(signal)) > 0
        
        # Check ground truth
        assert ground_truth['p_arrival'] == 5.0
        assert ground_truth['s_arrival'] == 10.0
        assert ground_truth['surface_arrival'] == 20.0
        assert ground_truth['sp_time'] == 5.0


class TestPerformanceBenchmarkSuite:
    """Test performance benchmarking functionality."""
    
    def test_benchmark_suite_initialization(self):
        """Test benchmark suite initialization."""
        suite = PerformanceBenchmarkSuite()
        
        assert isinstance(suite.profiler, PerformanceProfiler)
        assert isinstance(suite.data_generator, SyntheticDataGenerator)
        assert suite.benchmark_results == []
    
    def test_wave_detection_speed_benchmark(self):
        """Test wave detection speed benchmarking."""
        suite = PerformanceBenchmarkSuite()
        
        # Test with small data sizes for speed
        results = suite.benchmark_wave_detection_speed(data_sizes_mb=[0.1, 0.2])
        
        assert len(results) == 2
        
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.benchmark_name == "wave_detection_speed"
            assert result.execution_time > 0
            assert result.throughput_mbps >= 0
            assert 0 <= result.accuracy_score <= 1
    
    def test_analysis_accuracy_benchmark(self):
        """Test analysis accuracy benchmarking."""
        suite = PerformanceBenchmarkSuite()
        
        # Test with limited parameter combinations for speed
        results = suite.benchmark_analysis_accuracy(
            noise_levels=[0.1, 0.2],
            signal_strengths=[1.0, 2.0]
        )
        
        assert len(results) == 4  # 2 noise levels × 2 signal strengths
        
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.benchmark_name == "analysis_accuracy"
            assert 0 <= result.accuracy_score <= 1
    
    def test_scalability_benchmark(self):
        """Test scalability benchmarking."""
        suite = PerformanceBenchmarkSuite()
        
        # Test with small concurrent operations for speed
        results = suite.benchmark_scalability(
            concurrent_operations=[1, 2],
            operation_duration=1.0
        )
        
        assert len(results) == 2
        
        for result in results:
            assert isinstance(result, ScalabilityTestResult)
            assert result.concurrent_operations > 0
            assert result.total_execution_time > 0
            assert result.success_count >= 0
            assert result.failure_count >= 0
    
    @patch('wave_analysis.services.performance_benchmarks.logger')
    def test_comprehensive_benchmark(self, mock_logger):
        """Test comprehensive benchmark suite."""
        suite = PerformanceBenchmarkSuite()
        
        # Mock the individual benchmark methods to speed up testing
        suite.benchmark_wave_detection_speed = Mock(return_value=[
            BenchmarkResult("test", {}, 1.0, 10.0, 100.0, 0.9, 1.0)
        ])
        suite.benchmark_analysis_accuracy = Mock(return_value=[
            BenchmarkResult("test", {}, 1.0, 0.0, 0.0, 0.8, 1.0)
        ])
        suite.benchmark_scalability = Mock(return_value=[
            ScalabilityTestResult(1, 1.0, 1.0, 1.0, 100.0, 1, 0, 0.0)
        ])
        
        results = suite.run_comprehensive_benchmark()
        
        assert 'timestamp' in results
        assert 'speed_benchmarks' in results
        assert 'accuracy_benchmarks' in results
        assert 'scalability_benchmarks' in results
        assert 'summary' in results
        
        # Verify methods were called
        suite.benchmark_wave_detection_speed.assert_called_once()
        suite.benchmark_analysis_accuracy.assert_called_once()
        suite.benchmark_scalability.assert_called_once()


class TestPerformanceIntegration:
    """Integration tests for performance monitoring components."""
    
    def test_profiler_with_real_computation(self):
        """Test profiler with actual computation workload."""
        profiler = PerformanceProfiler()
        
        @profiler.profile_function("matrix_multiplication")
        def matrix_multiply():
            a = np.random.random((500, 500))
            b = np.random.random((500, 500))
            return np.dot(a, b)
        
        result = matrix_multiply()
        
        assert result.shape == (500, 500)
        assert len(profiler.metrics_history) == 1
        
        metric = profiler.metrics_history[0]
        assert metric.execution_time > 0
        assert metric.peak_memory_mb > 0
    
    def test_memory_monitoring_with_allocation(self):
        """Test memory monitoring with actual memory allocation."""
        monitor = MemoryMonitor(sampling_interval=0.01)
        
        monitor.start_monitoring()
        
        # Allocate memory in steps
        arrays = []
        for i in range(5):
            arrays.append(np.random.random((1000, 100)))
            time.sleep(0.02)
        
        monitor.stop_monitoring()
        
        peak_memory = monitor.get_peak_memory()
        trend_data = monitor.get_memory_trend()
        
        assert peak_memory > 0
        assert len(monitor.snapshots) > 0
        assert trend_data['max_memory'] > trend_data['min_memory']
        
        # Clean up
        del arrays
    
    def test_benchmark_with_synthetic_data(self):
        """Test benchmark with synthetic earthquake data."""
        suite = PerformanceBenchmarkSuite()
        
        # Generate test earthquake
        signal, ground_truth = suite.data_generator.generate_complete_earthquake(10.0)
        
        # Benchmark detection
        start_time = time.time()
        detected_waves = suite._simulate_wave_detection(signal, 100.0)
        execution_time = time.time() - start_time
        
        assert execution_time > 0
        assert len(detected_waves) >= 0
        
        # Calculate accuracy
        accuracy = suite._calculate_detection_accuracy(detected_waves, ground_truth)
        assert 0 <= accuracy <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])