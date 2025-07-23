"""
Performance monitoring demonstration for wave analysis system.

This script demonstrates the performance profiling, memory monitoring,
and benchmarking capabilities integrated with the wave analysis system.
"""

import numpy as np
import time
import logging
from pathlib import Path

from wave_analysis import (
    PerformanceProfiler, MemoryMonitor, profile_wave_operation,
    PerformanceBenchmarkSuite, SyntheticDataGenerator,
    global_profiler
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_performance_profiling():
    """Demonstrate performance profiling capabilities."""
    logger.info("=== Performance Profiling Demonstration ===")
    
    profiler = PerformanceProfiler()
    
    # Example 1: Using context manager
    logger.info("1. Profiling with context manager")
    with profiler.profile_operation("signal_processing", data_size_mb=5.0) as metrics:
        # Simulate signal processing
        signal = np.random.random(500000)  # ~5MB of data
        
        # Apply filtering
        filtered_signal = np.convolve(signal, np.ones(10)/10, mode='same')
        
        # Calculate FFT
        fft_result = np.fft.fft(filtered_signal)
        
        # Store additional metrics
        metrics['signal_length'] = len(signal)
        metrics['fft_peaks'] = len(np.where(np.abs(fft_result) > np.mean(np.abs(fft_result)) * 2)[0])
    
    # Example 2: Using decorator
    logger.info("2. Profiling with decorator")
    
    @profiler.profile_function("wave_detection_simulation", data_size_mb=2.0)
    def simulate_wave_detection(data):
        """Simulate wave detection processing."""
        # Simulate P-wave detection
        p_threshold = np.std(data) * 3
        p_candidates = np.where(np.abs(data) > p_threshold)[0]
        
        # Simulate S-wave detection
        s_threshold = np.std(data) * 2
        s_candidates = np.where(np.abs(data) > s_threshold)[0]
        
        # Simulate feature extraction
        features = {
            'p_wave_count': len(p_candidates),
            's_wave_count': len(s_candidates),
            'max_amplitude': np.max(np.abs(data)),
            'rms_amplitude': np.sqrt(np.mean(data**2))
        }
        
        return features
    
    # Generate test data and run detection
    test_data = np.random.random(200000)  # ~2MB
    features = simulate_wave_detection(test_data)
    logger.info(f"Detection results: {features}")
    
    # Display profiling results
    logger.info("3. Profiling results summary")
    summary = profiler.get_performance_summary()
    
    logger.info(f"Total operations: {summary['total_operations']}")
    logger.info(f"Unique operations: {summary['unique_operations']}")
    logger.info(f"Average execution time: {summary['overall']['average_execution_time']:.3f}s")
    logger.info(f"Peak memory usage: {summary['overall']['peak_memory_usage']:.1f}MB")
    
    # Detailed operation statistics
    for op_name, op_stats in summary['operations'].items():
        logger.info(f"\nOperation: {op_name}")
        logger.info(f"  Executions: {op_stats['total_executions']}")
        logger.info(f"  Avg time: {op_stats['execution_time']['mean']:.3f}s")
        logger.info(f"  Avg memory: {op_stats['memory_usage']['mean']:.1f}MB")


def demonstrate_memory_monitoring():
    """Demonstrate memory monitoring capabilities."""
    logger.info("\n=== Memory Monitoring Demonstration ===")
    
    monitor = MemoryMonitor(sampling_interval=0.05)
    
    logger.info("Starting memory monitoring...")
    monitor.start_monitoring()
    
    # Simulate memory-intensive operations
    arrays = []
    
    logger.info("Phase 1: Gradual memory allocation")
    for i in range(5):
        # Allocate progressively larger arrays
        size = (i + 1) * 100000
        array = np.random.random(size)
        arrays.append(array)
        time.sleep(0.2)
        logger.info(f"  Allocated array {i+1}: {size} elements")
    
    logger.info("Phase 2: Peak memory usage")
    # Create a large temporary array
    large_array = np.random.random((2000, 2000))
    time.sleep(0.3)
    del large_array
    
    logger.info("Phase 3: Memory cleanup")
    # Clean up arrays one by one
    for i in range(len(arrays)):
        del arrays[i]
        time.sleep(0.1)
    
    monitor.stop_monitoring()
    
    # Analyze memory usage
    peak_memory = monitor.get_peak_memory()
    avg_memory = monitor.get_average_memory()
    trend_data = monitor.get_memory_trend()
    
    logger.info(f"Memory analysis results:")
    logger.info(f"  Peak memory: {peak_memory:.1f}MB")
    logger.info(f"  Average memory: {avg_memory:.1f}MB")
    logger.info(f"  Memory trend: {trend_data['trend']:.3f}MB/sample")
    logger.info(f"  Memory volatility: {trend_data['volatility']:.3f}MB")
    logger.info(f"  Samples collected: {len(monitor.snapshots)}")


def demonstrate_global_profiler():
    """Demonstrate global profiler usage."""
    logger.info("\n=== Global Profiler Demonstration ===")
    
    @profile_wave_operation("global_matrix_operation")
    def matrix_operations():
        """Perform matrix operations for profiling."""
        # Create matrices
        a = np.random.random((1000, 1000))
        b = np.random.random((1000, 1000))
        
        # Matrix multiplication
        c = np.dot(a, b)
        
        # Eigenvalue calculation
        eigenvals = np.linalg.eigvals(c[:100, :100])  # Subset for speed
        
        return len(eigenvals)
    
    @profile_wave_operation("global_signal_analysis")
    def signal_analysis():
        """Perform signal analysis for profiling."""
        # Generate synthetic earthquake signal
        duration = 30.0
        sampling_rate = 100.0
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Create composite signal
        p_wave = 0.5 * np.exp(-t * 2) * np.sin(2 * np.pi * 10 * t)
        s_wave = 1.0 * np.exp(-t * 1) * np.sin(2 * np.pi * 5 * t)
        noise = 0.1 * np.random.normal(0, 1, len(t))
        
        signal = p_wave + s_wave + noise
        
        # Analyze signal
        fft_result = np.fft.fft(signal)
        power_spectrum = np.abs(fft_result)**2
        
        return np.argmax(power_spectrum[:len(power_spectrum)//2])
    
    # Execute profiled functions
    logger.info("Executing matrix operations...")
    eigenval_count = matrix_operations()
    logger.info(f"Computed {eigenval_count} eigenvalues")
    
    logger.info("Executing signal analysis...")
    dominant_freq_bin = signal_analysis()
    logger.info(f"Dominant frequency bin: {dominant_freq_bin}")
    
    # Show global profiler results
    global_summary = global_profiler.get_performance_summary()
    logger.info(f"Global profiler tracked {global_summary['total_operations']} operations")


def demonstrate_benchmarking():
    """Demonstrate benchmarking capabilities."""
    logger.info("\n=== Benchmarking Demonstration ===")
    
    benchmark_suite = PerformanceBenchmarkSuite()
    
    # Test synthetic data generation
    logger.info("1. Testing synthetic data generation")
    generator = SyntheticDataGenerator()
    
    # Generate different wave types
    p_wave = generator.generate_p_wave(duration=5.0)
    s_wave = generator.generate_s_wave(duration=5.0)
    surface_wave = generator.generate_surface_wave(duration=10.0)
    
    logger.info(f"Generated P-wave: {len(p_wave)} samples, max amplitude: {np.max(np.abs(p_wave)):.3f}")
    logger.info(f"Generated S-wave: {len(s_wave)} samples, max amplitude: {np.max(np.abs(s_wave)):.3f}")
    logger.info(f"Generated surface wave: {len(surface_wave)} samples, max amplitude: {np.max(np.abs(surface_wave)):.3f}")
    
    # Generate complete earthquake
    earthquake_signal, ground_truth = generator.generate_complete_earthquake(duration=20.0)
    logger.info(f"Generated complete earthquake: {len(earthquake_signal)} samples")
    logger.info(f"Ground truth arrivals: P={ground_truth['p_arrival']}s, S={ground_truth['s_arrival']}s")
    
    # Run speed benchmarks (limited for demo)
    logger.info("2. Running speed benchmarks")
    speed_results = benchmark_suite.benchmark_wave_detection_speed(data_sizes_mb=[0.5, 1.0])
    
    for result in speed_results:
        logger.info(f"  Data size: {result.test_parameters['data_size_mb']}MB")
        logger.info(f"  Execution time: {result.execution_time:.3f}s")
        logger.info(f"  Throughput: {result.throughput_mbps:.2f}MB/s")
        logger.info(f"  Accuracy: {result.accuracy_score:.3f}")
    
    # Run accuracy benchmarks (limited for demo)
    logger.info("3. Running accuracy benchmarks")
    accuracy_results = benchmark_suite.benchmark_analysis_accuracy(
        noise_levels=[0.1, 0.3],
        signal_strengths=[1.0, 2.0]
    )
    
    for result in accuracy_results:
        params = result.test_parameters
        logger.info(f"  Noise: {params['noise_level']}, Signal: {params['signal_strength']}")
        logger.info(f"  Accuracy: {result.accuracy_score:.3f}")
        logger.info(f"  Success rate: {result.success_rate:.3f}")
    
    # Run scalability test (limited for demo)
    logger.info("4. Running scalability test")
    scalability_results = benchmark_suite.benchmark_scalability(
        concurrent_operations=[1, 2],
        operation_duration=2.0
    )
    
    for result in scalability_results:
        logger.info(f"  Concurrent ops: {result.concurrent_operations}")
        logger.info(f"  Total time: {result.total_execution_time:.3f}s")
        logger.info(f"  Avg response time: {result.average_response_time:.3f}s")
        logger.info(f"  Throughput: {result.throughput_ops_per_second:.2f} ops/s")
        logger.info(f"  Success rate: {(result.success_count / (result.success_count + result.failure_count)):.3f}")


def export_performance_data():
    """Export performance monitoring data."""
    logger.info("\n=== Exporting Performance Data ===")
    
    # Export global profiler metrics
    try:
        global_profiler.export_metrics("performance_metrics.json")
        logger.info("Exported performance metrics to performance_metrics.json")
    except Exception as e:
        logger.error(f"Failed to export performance metrics: {e}")
    
    # Export benchmark results
    try:
        benchmark_suite = PerformanceBenchmarkSuite()
        benchmark_suite.export_benchmark_results("benchmark_results.json")
        logger.info("Exported benchmark results to benchmark_results.json")
    except Exception as e:
        logger.error(f"Failed to export benchmark results: {e}")


def main():
    """Main demonstration function."""
    logger.info("Starting Performance Monitoring Demonstration")
    logger.info("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_performance_profiling()
        demonstrate_memory_monitoring()
        demonstrate_global_profiler()
        demonstrate_benchmarking()
        export_performance_data()
        
        logger.info("\n" + "=" * 60)
        logger.info("Performance monitoring demonstration completed successfully!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()