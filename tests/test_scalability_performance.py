"""
Scalability and load testing for wave analysis system.

This module tests system performance under various load conditions
and concurrent access patterns to ensure scalability.
"""

import pytest
import numpy as np
import time
import threading
import concurrent.futures
from unittest.mock import Mock, patch
import psutil
import gc

from wave_analysis.services.performance_profiler import PerformanceProfiler, MemoryMonitor
from wave_analysis.services.performance_benchmarks import PerformanceBenchmarkSuite, SyntheticDataGenerator
from wave_analysis.services.performance_optimizer import PerformanceOptimizer, ParallelProcessor


class TestScalabilityUnderLoad:
    """Test system scalability under various load conditions."""
    
    def test_concurrent_analysis_requests(self):
        """Test system performance with concurrent analysis requests."""
        profiler = PerformanceProfiler()
        
        def simulate_wave_analysis(analysis_id):
            """Simulate a wave analysis operation."""
            with profiler.profile_operation(f"concurrent_analysis_{analysis_id}", data_size_mb=5.0):
                # Simulate analysis work
                data = np.random.random(50000)  # ~5MB of data
                
                # Simulate P-wave detection
                p_wave_result = np.convolve(data, np.ones(100)/100, mode='valid')
                
                # Simulate S-wave detection
                s_wave_result = np.gradient(data)
                
                # Simulate frequency analysis
                fft_result = np.fft.fft(data[:1024])
                
                # Simulate some processing delay
                time.sleep(0.01)
                
                return {
                    'p_waves': len(p_wave_result),
                    's_waves': len(s_wave_result),
                    'dominant_freq': np.argmax(np.abs(fft_result))
                }
        
        # Test with increasing concurrent requests
        concurrent_levels = [1, 2, 4, 8]
        results = {}
        
        for num_concurrent in concurrent_levels:
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(simulate_wave_analysis, i) for i in range(num_concurrent)]
                analysis_results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            total_time = time.time() - start_time
            
            results[num_concurrent] = {
                'total_time': total_time,
                'avg_time_per_request': total_time / num_concurrent,
                'requests_per_second': num_concurrent / total_time,
                'successful_requests': len(analysis_results)
            }
        
        # Verify all requests completed successfully
        for num_concurrent in concurrent_levels:
            assert results[num_concurrent]['successful_requests'] == num_concurrent
        
        # Performance should scale reasonably (not degrade exponentially)
        single_request_time = results[1]['total_time']
        max_concurrent_time = results[max(concurrent_levels)]['total_time']
        
        # With perfect scaling, max concurrent should take similar time to single request
        # Allow for some overhead but shouldn't be more than 3x slower
        assert max_concurrent_time < single_request_time * 3.0
        
        print("Concurrent Analysis Performance:")
        for num_concurrent, metrics in results.items():
            print(f"  {num_concurrent} concurrent: {metrics['total_time']:.3f}s total, "
                  f"{metrics['requests_per_second']:.2f} req/s")
    
    def test_memory_usage_under_load(self):
        """Test memory usage patterns under sustained load."""
        memory_monitor = MemoryMonitor(sampling_interval=0.05)
        
        def memory_intensive_operation():
            """Simulate memory-intensive wave analysis."""
            # Allocate large arrays to simulate seismic data processing
            large_data = np.random.random((10000, 100))  # ~80MB
            
            # Simulate processing operations
            processed_data = np.fft.fft2(large_data)
            filtered_data = np.convolve(large_data.flatten(), np.ones(100)/100, mode='valid')
            
            # Simulate analysis results
            result = {
                'data_size': large_data.nbytes,
                'processed_size': processed_data.nbytes,
                'filtered_size': filtered_data.nbytes
            }
            
            # Clean up explicitly
            del large_data, processed_data, filtered_data
            gc.collect()
            
            return result
        
        # Start memory monitoring
        memory_monitor.start_monitoring()
        
        # Run multiple operations to test memory management
        results = []
        for i in range(5):
            result = memory_intensive_operation()
            results.append(result)
            time.sleep(0.1)  # Allow memory monitoring to capture changes
        
        memory_monitor.stop_monitoring()
        
        # Analyze memory usage patterns
        peak_memory = memory_monitor.get_peak_memory()
        average_memory = memory_monitor.get_average_memory()
        memory_trend = memory_monitor.get_memory_trend()
        
        assert peak_memory > 0
        assert average_memory > 0
        assert len(memory_monitor.snapshots) > 0
        
        # Memory should not continuously increase (no major leaks)
        # Allow for some growth but trend should not be excessive
        assert abs(memory_trend['trend']) < 10.0  # Less than 10MB/sample trend
        
        print(f"Memory Usage Analysis:")
        print(f"  Peak Memory: {peak_memory:.1f}MB")
        print(f"  Average Memory: {average_memory:.1f}MB")
        print(f"  Memory Trend: {memory_trend['trend']:.3f}MB/sample")
        print(f"  Memory Volatility: {memory_trend['volatility']:.1f}MB")
    
    def test_throughput_scaling(self):
        """Test throughput scaling with different data sizes."""
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # Test with progressively larger data sizes
        data_sizes_mb = [1, 5, 10, 25, 50]
        throughput_results = []
        
        for size_mb in data_sizes_mb:
            # Generate test data
            samples_per_mb = (1024 * 1024) / 4  # 4 bytes per float32
            duration = (size_mb * samples_per_mb) / 100.0  # 100 Hz sampling
            
            signal, _ = benchmark_suite.data_generator.generate_complete_earthquake(duration)
            
            # Measure processing throughput
            start_time = time.time()
            
            # Simulate wave analysis processing
            detected_waves = benchmark_suite._simulate_wave_detection(signal, 100.0)
            
            processing_time = time.time() - start_time
            throughput = size_mb / processing_time if processing_time > 0 else 0
            
            throughput_results.append({
                'data_size_mb': size_mb,
                'processing_time': processing_time,
                'throughput_mbps': throughput,
                'detected_waves': len(detected_waves)
            })
        
        # Analyze throughput scaling
        throughputs = [r['throughput_mbps'] for r in throughput_results]
        
        # Throughput should remain relatively stable or improve with larger data
        # (due to better amortization of fixed costs)
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)
        
        # Throughput shouldn't degrade by more than 50%
        assert min_throughput > max_throughput * 0.5
        
        print("Throughput Scaling Results:")
        for result in throughput_results:
            print(f"  {result['data_size_mb']:2.0f}MB: {result['throughput_mbps']:6.2f}MB/s "
                  f"({result['processing_time']:.3f}s, {result['detected_waves']} waves)")
    
    def test_resource_utilization_efficiency(self):
        """Test efficient utilization of system resources."""
        optimizer = PerformanceOptimizer()
        
        # Monitor system resources during optimization
        initial_cpu_percent = psutil.cpu_percent(interval=1)
        initial_memory = psutil.virtual_memory()
        
        # Create test workload
        pipeline_config = {
            'detection_parameters': {
                'p_wave_threshold': 2.0,
                's_wave_threshold': 1.8
            },
            'use_parallel_processing': True
        }
        
        # Generate test data of various sizes
        test_data = []
        for size in [1000, 5000, 10000]:  # Different signal lengths
            signal = np.random.random(size)
            test_data.append(signal)
        
        # Run optimization and monitor resources
        start_time = time.time()
        optimization_results = optimizer.optimize_wave_analysis_pipeline(pipeline_config, test_data)
        optimization_time = time.time() - start_time
        
        final_cpu_percent = psutil.cpu_percent(interval=1)
        final_memory = psutil.virtual_memory()
        
        # Analyze resource utilization
        cpu_usage_during_optimization = max(initial_cpu_percent, final_cpu_percent)
        memory_increase_mb = (final_memory.used - initial_memory.used) / 1024 / 1024
        
        # Verify optimization completed successfully
        assert optimization_results['performance_improvements']['speedup_factor'] > 0
        assert len(optimization_results['optimization_steps']) > 0
        
        # Resource usage should be reasonable
        assert cpu_usage_during_optimization < 95  # Should not max out CPU
        assert abs(memory_increase_mb) < 500  # Should not use excessive memory
        
        print(f"Resource Utilization During Optimization:")
        print(f"  Optimization Time: {optimization_time:.3f}s")
        print(f"  CPU Usage: {cpu_usage_during_optimization:.1f}%")
        print(f"  Memory Change: {memory_increase_mb:+.1f}MB")
        print(f"  Speedup Factor: {optimization_results['performance_improvements']['speedup_factor']:.2f}x")


class TestLoadStressTesting:
    """Stress testing under extreme load conditions."""
    
    def test_high_frequency_requests(self):
        """Test system behavior under high-frequency request patterns."""
        profiler = PerformanceProfiler()
        request_count = 50
        request_interval = 0.01  # 100 requests per second
        
        def rapid_analysis_request(request_id):
            """Simulate rapid analysis request."""
            with profiler.profile_operation(f"rapid_request_{request_id}", data_size_mb=0.1):
                # Small, fast analysis
                data = np.random.random(1000)
                result = np.mean(data)
                time.sleep(0.001)  # Minimal processing time
                return result
        
        # Submit requests rapidly
        start_time = time.time()
        results = []
        
        for i in range(request_count):
            result = rapid_analysis_request(i)
            results.append(result)
            
            if i < request_count - 1:  # Don't sleep after last request
                time.sleep(request_interval)
        
        total_time = time.time() - start_time
        actual_request_rate = request_count / total_time
        
        # Verify all requests completed
        assert len(results) == request_count
        
        # System should handle high-frequency requests reasonably well
        expected_min_rate = 50  # At least 50 requests per second
        assert actual_request_rate > expected_min_rate
        
        print(f"High-Frequency Request Test:")
        print(f"  Completed {request_count} requests in {total_time:.3f}s")
        print(f"  Actual rate: {actual_request_rate:.1f} requests/second")
    
    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        memory_monitor = MemoryMonitor(sampling_interval=0.02)
        
        def memory_pressure_operation(operation_id):
            """Create memory pressure through large allocations."""
            try:
                # Allocate progressively larger arrays
                arrays = []
                for i in range(10):
                    size = (operation_id + 1) * 1000 * (i + 1)
                    array = np.random.random(size)
                    arrays.append(array)
                
                # Simulate some processing
                result = sum(np.sum(arr) for arr in arrays)
                
                # Clean up
                del arrays
                gc.collect()
                
                return result
                
            except MemoryError:
                # Handle memory pressure gracefully
                gc.collect()
                return None
        
        memory_monitor.start_monitoring()
        
        # Run operations that create memory pressure
        results = []
        for i in range(5):
            result = memory_pressure_operation(i)
            results.append(result)
            time.sleep(0.05)
        
        memory_monitor.stop_monitoring()
        
        # Analyze memory behavior under pressure
        peak_memory = memory_monitor.get_peak_memory()
        memory_trend = memory_monitor.get_memory_trend()
        
        # System should handle memory pressure without crashing
        successful_operations = sum(1 for r in results if r is not None)
        assert successful_operations > 0  # At least some operations should succeed
        
        # Memory should be managed reasonably
        assert peak_memory > 0
        assert len(memory_monitor.snapshots) > 0
        
        print(f"Memory Pressure Test:")
        print(f"  Successful operations: {successful_operations}/5")
        print(f"  Peak memory usage: {peak_memory:.1f}MB")
        print(f"  Memory volatility: {memory_trend['volatility']:.1f}MB")
    
    def test_sustained_load_stability(self):
        """Test system stability under sustained load."""
        profiler = PerformanceProfiler()
        duration_seconds = 10  # Run for 10 seconds
        operation_interval = 0.1  # 10 operations per second
        
        def sustained_operation(op_id):
            """Simulate sustained analysis operation."""
            with profiler.profile_operation(f"sustained_op_{op_id}", data_size_mb=1.0):
                # Moderate complexity operation
                data = np.random.random(10000)
                
                # Simulate wave detection
                filtered = np.convolve(data, np.ones(50)/50, mode='valid')
                peaks = np.where(np.abs(filtered) > np.std(filtered) * 2)[0]
                
                # Simulate frequency analysis
                fft_result = np.fft.fft(data[:1024])
                dominant_freq = np.argmax(np.abs(fft_result))
                
                return {
                    'peaks_detected': len(peaks),
                    'dominant_frequency': dominant_freq
                }
        
        # Run sustained load
        start_time = time.time()
        operation_count = 0
        results = []
        errors = []
        
        while time.time() - start_time < duration_seconds:
            try:
                result = sustained_operation(operation_count)
                results.append(result)
                operation_count += 1
                
                time.sleep(operation_interval)
                
            except Exception as e:
                errors.append(str(e))
        
        total_time = time.time() - start_time
        
        # Analyze sustained load performance
        success_rate = len(results) / operation_count if operation_count > 0 else 0
        operations_per_second = operation_count / total_time
        
        # Get performance statistics
        performance_summary = profiler.get_performance_summary()
        
        # System should maintain stability under sustained load
        assert success_rate > 0.95  # At least 95% success rate
        assert len(errors) < operation_count * 0.05  # Less than 5% errors
        assert operations_per_second > 5  # At least 5 operations per second
        
        print(f"Sustained Load Test ({duration_seconds}s):")
        print(f"  Total operations: {operation_count}")
        print(f"  Success rate: {success_rate:.3f}")
        print(f"  Operations per second: {operations_per_second:.1f}")
        print(f"  Error count: {len(errors)}")
        
        if performance_summary['total_operations'] > 0:
            avg_time = performance_summary['overall']['average_execution_time']
            print(f"  Average operation time: {avg_time:.3f}s")


class TestPerformanceRegression:
    """Test for performance regressions and baseline maintenance."""
    
    def test_baseline_performance_metrics(self):
        """Establish and verify baseline performance metrics."""
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # Define baseline expectations
        baseline_expectations = {
            'small_data_throughput_mbps': 10.0,  # Minimum 10 MB/s for small data
            'medium_data_throughput_mbps': 5.0,  # Minimum 5 MB/s for medium data
            'detection_accuracy': 0.7,           # Minimum 70% detection accuracy
            'max_memory_usage_mb': 500,          # Maximum 500MB memory usage
            'max_processing_time_s': 5.0         # Maximum 5s processing time
        }
        
        # Test small data processing
        small_results = benchmark_suite.benchmark_wave_detection_speed([1.0])
        small_throughput = small_results[0].throughput_mbps
        
        # Test medium data processing
        medium_results = benchmark_suite.benchmark_wave_detection_speed([10.0])
        medium_throughput = medium_results[0].throughput_mbps
        
        # Test accuracy
        accuracy_results = benchmark_suite.benchmark_analysis_accuracy(
            noise_levels=[0.1], signal_strengths=[1.0]
        )
        avg_accuracy = np.mean([r.accuracy_score for r in accuracy_results])
        
        # Verify against baselines
        assert small_throughput >= baseline_expectations['small_data_throughput_mbps'], \
            f"Small data throughput {small_throughput:.2f} below baseline {baseline_expectations['small_data_throughput_mbps']}"
        
        assert medium_throughput >= baseline_expectations['medium_data_throughput_mbps'], \
            f"Medium data throughput {medium_throughput:.2f} below baseline {baseline_expectations['medium_data_throughput_mbps']}"
        
        assert avg_accuracy >= baseline_expectations['detection_accuracy'], \
            f"Detection accuracy {avg_accuracy:.3f} below baseline {baseline_expectations['detection_accuracy']}"
        
        print(f"Baseline Performance Verification:")
        print(f"  Small data throughput: {small_throughput:.2f} MB/s (baseline: {baseline_expectations['small_data_throughput_mbps']})")
        print(f"  Medium data throughput: {medium_throughput:.2f} MB/s (baseline: {baseline_expectations['medium_data_throughput_mbps']})")
        print(f"  Detection accuracy: {avg_accuracy:.3f} (baseline: {baseline_expectations['detection_accuracy']})")
    
    def test_performance_consistency(self):
        """Test performance consistency across multiple runs."""
        profiler = PerformanceProfiler()
        
        def consistent_operation():
            """Operation that should have consistent performance."""
            with profiler.profile_operation("consistency_test", data_size_mb=2.0):
                data = np.random.random(20000)
                result = np.fft.fft(data)
                peak_freq = np.argmax(np.abs(result))
                return peak_freq
        
        # Run operation multiple times
        num_runs = 10
        execution_times = []
        
        for i in range(num_runs):
            start_time = time.time()
            result = consistent_operation()
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
        
        # Analyze consistency
        mean_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        coefficient_of_variation = std_time / mean_time if mean_time > 0 else 0
        
        # Performance should be reasonably consistent
        # Coefficient of variation should be less than 0.3 (30%)
        assert coefficient_of_variation < 0.3, \
            f"Performance inconsistent: CV={coefficient_of_variation:.3f}"
        
        print(f"Performance Consistency Test ({num_runs} runs):")
        print(f"  Mean execution time: {mean_time:.3f}s")
        print(f"  Standard deviation: {std_time:.3f}s")
        print(f"  Coefficient of variation: {coefficient_of_variation:.3f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])