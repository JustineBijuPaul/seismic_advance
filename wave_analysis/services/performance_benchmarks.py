"""
Performance benchmarking system for wave analysis operations.

This module provides comprehensive benchmarking capabilities to measure
analysis speed, accuracy, and system scalability under various conditions.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import concurrent.futures
import threading
from pathlib import Path
import json
import logging

from .performance_profiler import PerformanceProfiler, PerformanceMetrics
from ..models import WaveSegment, WaveAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark test results."""
    benchmark_name: str
    test_parameters: Dict[str, Any]
    execution_time: float
    throughput_mbps: float
    memory_peak_mb: float
    accuracy_score: float
    success_rate: float
    error_messages: List[str] = field(default_factory=list)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalabilityTestResult:
    """Results from scalability testing."""
    concurrent_operations: int
    total_execution_time: float
    average_response_time: float
    throughput_ops_per_second: float
    memory_usage_mb: float
    success_count: int
    failure_count: int
    error_rate: float


class SyntheticDataGenerator:
    """
    Generates synthetic seismic data for benchmarking.
    
    Creates realistic earthquake signals with known characteristics
    for testing wave detection accuracy and performance.
    """
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize synthetic data generator.
        
        Args:
            sampling_rate: Sampling rate for generated signals in Hz
        """
        self.sampling_rate = sampling_rate
    
    def generate_p_wave(self, duration: float, amplitude: float = 1.0, 
                       frequency: float = 10.0, noise_level: float = 0.1) -> np.ndarray:
        """
        Generate synthetic P-wave signal.
        
        Args:
            duration: Signal duration in seconds
            amplitude: Peak amplitude
            frequency: Dominant frequency in Hz
            noise_level: Noise level as fraction of signal amplitude
            
        Returns:
            Synthetic P-wave time series
        """
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # P-wave characteristics: sharp onset, high frequency
        envelope = np.exp(-t * 2.0)  # Exponential decay
        signal = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add noise
        noise = noise_level * amplitude * np.random.normal(0, 1, len(signal))
        
        return signal + noise
    
    def generate_s_wave(self, duration: float, amplitude: float = 1.5, 
                       frequency: float = 5.0, noise_level: float = 0.1) -> np.ndarray:
        """
        Generate synthetic S-wave signal.
        
        Args:
            duration: Signal duration in seconds
            amplitude: Peak amplitude
            frequency: Dominant frequency in Hz
            noise_level: Noise level as fraction of signal amplitude
            
        Returns:
            Synthetic S-wave time series
        """
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # S-wave characteristics: larger amplitude, lower frequency
        envelope = np.exp(-t * 1.0)  # Slower decay than P-wave
        signal = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add noise
        noise = noise_level * amplitude * np.random.normal(0, 1, len(signal))
        
        return signal + noise
    
    def generate_surface_wave(self, duration: float, amplitude: float = 2.0, 
                             frequency: float = 1.0, noise_level: float = 0.1) -> np.ndarray:
        """
        Generate synthetic surface wave signal.
        
        Args:
            duration: Signal duration in seconds
            amplitude: Peak amplitude
            frequency: Dominant frequency in Hz
            noise_level: Noise level as fraction of signal amplitude
            
        Returns:
            Synthetic surface wave time series
        """
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Surface wave characteristics: long duration, low frequency
        envelope = np.exp(-t * 0.2)  # Very slow decay
        signal = amplitude * envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add dispersive characteristics
        dispersive_component = 0.5 * amplitude * envelope * np.sin(2 * np.pi * frequency * 0.8 * t)
        signal += dispersive_component
        
        # Add noise
        noise = noise_level * amplitude * np.random.normal(0, 1, len(signal))
        
        return signal + noise
    
    def generate_complete_earthquake(self, total_duration: float = 60.0,
                                   p_arrival: float = 5.0, s_arrival: float = 15.0,
                                   surface_arrival: float = 30.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate complete synthetic earthquake with all wave types.
        
        Args:
            total_duration: Total signal duration in seconds
            p_arrival: P-wave arrival time in seconds
            s_arrival: S-wave arrival time in seconds
            surface_arrival: Surface wave arrival time in seconds
            
        Returns:
            Tuple of (complete signal, ground truth arrival times)
        """
        signal = np.zeros(int(total_duration * self.sampling_rate))
        
        # Add P-wave
        p_duration = 10.0
        p_start_idx = int(p_arrival * self.sampling_rate)
        p_end_idx = int((p_arrival + p_duration) * self.sampling_rate)
        if p_end_idx <= len(signal):
            p_wave = self.generate_p_wave(p_duration, amplitude=0.5)
            signal[p_start_idx:p_start_idx + len(p_wave)] += p_wave
        
        # Add S-wave
        s_duration = 15.0
        s_start_idx = int(s_arrival * self.sampling_rate)
        s_end_idx = int((s_arrival + s_duration) * self.sampling_rate)
        if s_end_idx <= len(signal):
            s_wave = self.generate_s_wave(s_duration, amplitude=1.0)
            signal[s_start_idx:s_start_idx + len(s_wave)] += s_wave
        
        # Add surface wave
        surface_duration = 25.0
        surface_start_idx = int(surface_arrival * self.sampling_rate)
        surface_end_idx = int((surface_arrival + surface_duration) * self.sampling_rate)
        if surface_end_idx <= len(signal):
            surface_wave = self.generate_surface_wave(surface_duration, amplitude=1.5)
            signal[surface_start_idx:surface_start_idx + len(surface_wave)] += surface_wave
        
        # Add background noise
        background_noise = 0.05 * np.random.normal(0, 1, len(signal))
        signal += background_noise
        
        ground_truth = {
            'p_arrival': p_arrival,
            's_arrival': s_arrival,
            'surface_arrival': surface_arrival,
            'sp_time': s_arrival - p_arrival
        }
        
        return signal, ground_truth


class PerformanceBenchmarkSuite:
    """
    Comprehensive benchmark suite for wave analysis performance testing.
    
    Tests analysis speed, accuracy, memory usage, and scalability
    under various conditions and data sizes.
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.profiler = PerformanceProfiler()
        self.data_generator = SyntheticDataGenerator()
        self.benchmark_results: List[BenchmarkResult] = []
    
    def benchmark_wave_detection_speed(self, data_sizes_mb: List[float] = None) -> List[BenchmarkResult]:
        """
        Benchmark wave detection speed across different data sizes.
        
        Args:
            data_sizes_mb: List of data sizes to test in MB
            
        Returns:
            List of benchmark results
        """
        if data_sizes_mb is None:
            data_sizes_mb = [1.0, 5.0, 10.0, 25.0, 50.0]
        
        results = []
        
        for size_mb in data_sizes_mb:
            # Calculate signal duration for target size
            # Assuming 4 bytes per sample, 100 Hz sampling rate
            samples_per_mb = (1024 * 1024) / 4
            duration = (size_mb * samples_per_mb) / 100.0
            
            # Generate test data
            test_signal, ground_truth = self.data_generator.generate_complete_earthquake(duration)
            
            # Benchmark detection
            start_time = time.time()
            
            with self.profiler.profile_operation(f"wave_detection_{size_mb}MB", size_mb) as metrics:
                # Simulate wave detection (replace with actual detector calls)
                detected_waves = self._simulate_wave_detection(test_signal, 100.0)
                metrics['detected_waves'] = len(detected_waves)
                metrics['ground_truth'] = ground_truth
            
            execution_time = time.time() - start_time
            throughput = size_mb / execution_time if execution_time > 0 else 0.0
            
            # Calculate accuracy (simplified)
            accuracy = self._calculate_detection_accuracy(detected_waves, ground_truth)
            
            result = BenchmarkResult(
                benchmark_name="wave_detection_speed",
                test_parameters={"data_size_mb": size_mb, "duration_s": duration},
                execution_time=execution_time,
                throughput_mbps=throughput,
                memory_peak_mb=self.profiler.memory_monitor.get_peak_memory(),
                accuracy_score=accuracy,
                success_rate=1.0,
                detailed_metrics={"detected_waves": len(detected_waves)}
            )
            
            results.append(result)
            self.benchmark_results.append(result)
            
            logger.info(f"Wave detection benchmark: {size_mb}MB in {execution_time:.3f}s "
                       f"({throughput:.2f}MB/s, accuracy: {accuracy:.3f})")
        
        return results
    
    def benchmark_analysis_accuracy(self, noise_levels: List[float] = None,
                                  signal_strengths: List[float] = None) -> List[BenchmarkResult]:
        """
        Benchmark wave analysis accuracy under different conditions.
        
        Args:
            noise_levels: List of noise levels to test
            signal_strengths: List of signal strengths to test
            
        Returns:
            List of benchmark results
        """
        if noise_levels is None:
            noise_levels = [0.05, 0.1, 0.2, 0.5]
        if signal_strengths is None:
            signal_strengths = [0.5, 1.0, 2.0, 5.0]
        
        results = []
        
        for noise_level in noise_levels:
            for signal_strength in signal_strengths:
                # Generate test data with specific noise and signal characteristics
                test_signal, ground_truth = self._generate_test_signal_with_conditions(
                    noise_level, signal_strength
                )
                
                start_time = time.time()
                
                # Perform analysis
                detected_waves = self._simulate_wave_detection(test_signal, 100.0)
                execution_time = time.time() - start_time
                
                # Calculate accuracy metrics
                accuracy = self._calculate_detection_accuracy(detected_waves, ground_truth)
                
                result = BenchmarkResult(
                    benchmark_name="analysis_accuracy",
                    test_parameters={
                        "noise_level": noise_level,
                        "signal_strength": signal_strength
                    },
                    execution_time=execution_time,
                    throughput_mbps=0.0,  # Not applicable for accuracy test
                    memory_peak_mb=0.0,   # Not measured for accuracy test
                    accuracy_score=accuracy,
                    success_rate=1.0 if accuracy > 0.5 else 0.0
                )
                
                results.append(result)
                self.benchmark_results.append(result)
        
        return results
    
    def benchmark_scalability(self, concurrent_operations: List[int] = None,
                            operation_duration: float = 10.0) -> List[ScalabilityTestResult]:
        """
        Benchmark system scalability under concurrent load.
        
        Args:
            concurrent_operations: List of concurrent operation counts to test
            operation_duration: Duration of each test operation in seconds
            
        Returns:
            List of scalability test results
        """
        if concurrent_operations is None:
            concurrent_operations = [1, 2, 4, 8, 16]
        
        results = []
        
        for num_ops in concurrent_operations:
            logger.info(f"Testing scalability with {num_ops} concurrent operations")
            
            # Generate test data for each operation
            test_data = []
            for i in range(num_ops):
                signal, _ = self.data_generator.generate_complete_earthquake(operation_duration)
                test_data.append(signal)
            
            # Run concurrent operations
            start_time = time.time()
            success_count = 0
            failure_count = 0
            response_times = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_ops) as executor:
                # Submit all operations
                futures = []
                for i, signal in enumerate(test_data):
                    future = executor.submit(self._benchmark_single_operation, signal, i)
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        op_time = future.result()
                        response_times.append(op_time)
                        success_count += 1
                    except Exception as e:
                        failure_count += 1
                        logger.error(f"Operation failed: {e}")
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            avg_response_time = np.mean(response_times) if response_times else 0.0
            throughput_ops_per_sec = num_ops / total_time if total_time > 0 else 0.0
            error_rate = failure_count / num_ops if num_ops > 0 else 0.0
            
            result = ScalabilityTestResult(
                concurrent_operations=num_ops,
                total_execution_time=total_time,
                average_response_time=avg_response_time,
                throughput_ops_per_second=throughput_ops_per_sec,
                memory_usage_mb=self.profiler.memory_monitor.get_peak_memory(),
                success_count=success_count,
                failure_count=failure_count,
                error_rate=error_rate
            )
            
            results.append(result)
            
            logger.info(f"Scalability test: {num_ops} ops in {total_time:.3f}s "
                       f"({throughput_ops_per_sec:.2f} ops/s, error rate: {error_rate:.3f})")
        
        return results
    
    def _simulate_wave_detection(self, signal: np.ndarray, sampling_rate: float) -> List[Dict[str, Any]]:
        """
        Simulate wave detection for benchmarking purposes.
        
        Args:
            signal: Input seismic signal
            sampling_rate: Sampling rate in Hz
            
        Returns:
            List of detected wave information
        """
        # This is a simplified simulation for benchmarking
        # In real implementation, this would call actual wave detectors
        
        detected_waves = []
        
        # Simple peak detection as simulation
        threshold = np.std(signal) * 2.0
        peaks = np.where(np.abs(signal) > threshold)[0]
        
        if len(peaks) > 0:
            # Group peaks into wave segments
            wave_starts = [peaks[0]]
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[i-1] > sampling_rate:  # 1 second gap
                    wave_starts.append(peaks[i])
            
            for i, start_idx in enumerate(wave_starts):
                wave_type = 'P' if i == 0 else 'S' if i == 1 else 'Surface'
                detected_waves.append({
                    'type': wave_type,
                    'start_time': start_idx / sampling_rate,
                    'amplitude': np.max(np.abs(signal[start_idx:start_idx + int(sampling_rate * 5)]))
                })
        
        return detected_waves
    
    def _calculate_detection_accuracy(self, detected_waves: List[Dict[str, Any]], 
                                    ground_truth: Dict[str, float]) -> float:
        """
        Calculate detection accuracy against ground truth.
        
        Args:
            detected_waves: List of detected wave information
            ground_truth: Ground truth arrival times
            
        Returns:
            Accuracy score between 0 and 1
        """
        if not detected_waves:
            return 0.0
        
        # Simple accuracy calculation based on timing
        accuracy_scores = []
        
        for wave in detected_waves:
            wave_type = wave['type'].lower()
            detected_time = wave['start_time']
            
            if wave_type == 'p' and 'p_arrival' in ground_truth:
                time_error = abs(detected_time - ground_truth['p_arrival'])
                accuracy = max(0, 1 - time_error / 5.0)  # 5 second tolerance
                accuracy_scores.append(accuracy)
            elif wave_type == 's' and 's_arrival' in ground_truth:
                time_error = abs(detected_time - ground_truth['s_arrival'])
                accuracy = max(0, 1 - time_error / 5.0)
                accuracy_scores.append(accuracy)
            elif wave_type == 'surface' and 'surface_arrival' in ground_truth:
                time_error = abs(detected_time - ground_truth['surface_arrival'])
                accuracy = max(0, 1 - time_error / 10.0)  # 10 second tolerance
                accuracy_scores.append(accuracy)
        
        return np.mean(accuracy_scores) if accuracy_scores else 0.0
    
    def _generate_test_signal_with_conditions(self, noise_level: float, 
                                            signal_strength: float) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate test signal with specific noise and signal conditions.
        
        Args:
            noise_level: Noise level as fraction of signal
            signal_strength: Signal strength multiplier
            
        Returns:
            Tuple of (test signal, ground truth)
        """
        # Generate base earthquake
        signal, ground_truth = self.data_generator.generate_complete_earthquake(30.0)
        
        # Apply signal strength
        signal *= signal_strength
        
        # Add additional noise
        additional_noise = noise_level * np.std(signal) * np.random.normal(0, 1, len(signal))
        signal += additional_noise
        
        return signal, ground_truth
    
    def _benchmark_single_operation(self, signal: np.ndarray, operation_id: int) -> float:
        """
        Benchmark a single wave analysis operation.
        
        Args:
            signal: Input seismic signal
            operation_id: Unique identifier for the operation
            
        Returns:
            Operation execution time in seconds
        """
        start_time = time.time()
        
        try:
            # Simulate wave analysis operation
            detected_waves = self._simulate_wave_detection(signal, 100.0)
            
            # Add some processing delay to simulate real analysis
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Operation {operation_id} failed: {e}")
            raise
        
        return time.time() - start_time
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite covering all aspects.
        
        Returns:
            Complete benchmark results summary
        """
        logger.info("Starting comprehensive performance benchmark suite")
        
        results = {
            'timestamp': time.time(),
            'speed_benchmarks': [],
            'accuracy_benchmarks': [],
            'scalability_benchmarks': []
        }
        
        try:
            # Speed benchmarks
            logger.info("Running speed benchmarks...")
            speed_results = self.benchmark_wave_detection_speed()
            results['speed_benchmarks'] = [self._serialize_benchmark_result(r) for r in speed_results]
            
            # Accuracy benchmarks
            logger.info("Running accuracy benchmarks...")
            accuracy_results = self.benchmark_analysis_accuracy()
            results['accuracy_benchmarks'] = [self._serialize_benchmark_result(r) for r in accuracy_results]
            
            # Scalability benchmarks
            logger.info("Running scalability benchmarks...")
            scalability_results = self.benchmark_scalability()
            results['scalability_benchmarks'] = [self._serialize_scalability_result(r) for r in scalability_results]
            
            # Overall summary
            results['summary'] = self._generate_benchmark_summary()
            
            logger.info("Comprehensive benchmark suite completed")
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _serialize_benchmark_result(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Convert BenchmarkResult to serializable dictionary."""
        return {
            'benchmark_name': result.benchmark_name,
            'test_parameters': result.test_parameters,
            'execution_time': result.execution_time,
            'throughput_mbps': result.throughput_mbps,
            'memory_peak_mb': result.memory_peak_mb,
            'accuracy_score': result.accuracy_score,
            'success_rate': result.success_rate,
            'error_messages': result.error_messages,
            'detailed_metrics': result.detailed_metrics,
            'timestamp': result.timestamp
        }
    
    def _serialize_scalability_result(self, result: ScalabilityTestResult) -> Dict[str, Any]:
        """Convert ScalabilityTestResult to serializable dictionary."""
        return {
            'concurrent_operations': result.concurrent_operations,
            'total_execution_time': result.total_execution_time,
            'average_response_time': result.average_response_time,
            'throughput_ops_per_second': result.throughput_ops_per_second,
            'memory_usage_mb': result.memory_usage_mb,
            'success_count': result.success_count,
            'failure_count': result.failure_count,
            'error_rate': result.error_rate
        }
    
    def _generate_benchmark_summary(self) -> Dict[str, Any]:
        """Generate overall benchmark summary."""
        if not self.benchmark_results:
            return {}
        
        # Calculate overall statistics
        execution_times = [r.execution_time for r in self.benchmark_results]
        throughputs = [r.throughput_mbps for r in self.benchmark_results if r.throughput_mbps > 0]
        accuracies = [r.accuracy_score for r in self.benchmark_results if r.accuracy_score > 0]
        
        summary = {
            'total_benchmarks': len(self.benchmark_results),
            'average_execution_time': np.mean(execution_times),
            'max_execution_time': np.max(execution_times),
            'min_execution_time': np.min(execution_times)
        }
        
        if throughputs:
            summary['average_throughput'] = np.mean(throughputs)
            summary['max_throughput'] = np.max(throughputs)
        
        if accuracies:
            summary['average_accuracy'] = np.mean(accuracies)
            summary['min_accuracy'] = np.min(accuracies)
        
        return summary
    
    def export_benchmark_results(self, filepath: str) -> None:
        """
        Export benchmark results to file.
        
        Args:
            filepath: Path to export file (JSON format)
        """
        results = self.run_comprehensive_benchmark()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results exported to {filepath}")