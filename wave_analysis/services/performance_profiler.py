"""
Performance profiling utilities for wave analysis operations.

This module provides comprehensive performance monitoring capabilities
including execution time tracking, memory usage monitoring, and
detailed profiling of wave analysis operations.
"""

import time
import psutil
import functools
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    data_size_mb: float
    throughput_mbps: float
    timestamp: float = field(default_factory=time.time)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a specific point in time."""
    timestamp: float
    memory_mb: float
    virtual_memory_mb: float
    swap_memory_mb: float
    memory_percent: float


class MemoryMonitor:
    """
    Monitors memory usage during wave analysis operations.
    
    Provides real-time memory tracking and peak usage detection
    for large seismic file processing.
    """
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Initialize memory monitor.
        
        Args:
            sampling_interval: Time between memory samples in seconds
        """
        self.sampling_interval = sampling_interval
        self.snapshots: List[MemorySnapshot] = []
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
    
    def start_monitoring(self) -> None:
        """Start continuous memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.debug("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring and return final snapshot."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.debug(f"Memory monitoring stopped. Collected {len(self.snapshots)} snapshots")
    
    def _monitor_loop(self) -> None:
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                virtual_memory = psutil.virtual_memory()
                swap_memory = psutil.swap_memory()
                
                snapshot = MemorySnapshot(
                    timestamp=time.time(),
                    memory_mb=memory_info.rss / 1024 / 1024,
                    virtual_memory_mb=memory_info.vms / 1024 / 1024,
                    swap_memory_mb=swap_memory.used / 1024 / 1024,
                    memory_percent=virtual_memory.percent
                )
                self.snapshots.append(snapshot)
                
                time.sleep(self.sampling_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                break
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return max(snapshot.memory_mb for snapshot in self.snapshots)
    
    def get_average_memory(self) -> float:
        """Get average memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return sum(snapshot.memory_mb for snapshot in self.snapshots) / len(self.snapshots)
    
    def get_memory_trend(self) -> Dict[str, float]:
        """Get memory usage trend analysis."""
        if len(self.snapshots) < 2:
            return {"trend": 0.0, "volatility": 0.0}
        
        memory_values = [s.memory_mb for s in self.snapshots]
        
        # Calculate trend (linear regression slope)
        n = len(memory_values)
        x = np.arange(n)
        trend = np.polyfit(x, memory_values, 1)[0]
        
        # Calculate volatility (standard deviation)
        volatility = np.std(memory_values)
        
        return {
            "trend": trend,
            "volatility": volatility,
            "min_memory": min(memory_values),
            "max_memory": max(memory_values)
        }


class PerformanceProfiler:
    """
    Comprehensive performance profiler for wave analysis operations.
    
    Tracks execution time, memory usage, CPU utilization, and throughput
    for all major wave analysis components.
    """
    
    def __init__(self):
        """Initialize performance profiler."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.memory_monitor = MemoryMonitor()
        self.active_operations: Dict[str, Dict[str, Any]] = {}
    
    @contextmanager
    def profile_operation(self, operation_name: str, data_size_mb: float = 0.0):
        """
        Context manager for profiling wave analysis operations.
        
        Args:
            operation_name: Name of the operation being profiled
            data_size_mb: Size of data being processed in MB
            
        Yields:
            Dictionary to store additional metrics during operation
        """
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = self.memory_monitor.process.memory_info().rss / 1024 / 1024
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Container for additional metrics
        additional_metrics = {}
        
        try:
            yield additional_metrics
        finally:
            # Stop monitoring and calculate metrics
            end_time = time.time()
            self.memory_monitor.stop_monitoring()
            
            execution_time = end_time - start_time
            end_memory = self.memory_monitor.process.memory_info().rss / 1024 / 1024
            peak_memory = self.memory_monitor.get_peak_memory()
            end_cpu = psutil.cpu_percent()
            
            # Calculate throughput
            throughput = data_size_mb / execution_time if execution_time > 0 else 0.0
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage_mb=end_memory - start_memory,
                peak_memory_mb=peak_memory,
                cpu_usage_percent=(start_cpu + end_cpu) / 2,
                data_size_mb=data_size_mb,
                throughput_mbps=throughput,
                additional_metrics=additional_metrics
            )
            
            self.metrics_history.append(metrics)
            logger.info(f"Operation '{operation_name}' completed in {execution_time:.3f}s, "
                       f"peak memory: {peak_memory:.1f}MB, throughput: {throughput:.2f}MB/s")
    
    def profile_function(self, operation_name: str = None, data_size_mb: float = 0.0):
        """
        Decorator for profiling individual functions.
        
        Args:
            operation_name: Name for the operation (defaults to function name)
            data_size_mb: Size of data being processed in MB
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                with self.profile_operation(op_name, data_size_mb) as metrics:
                    result = func(*args, **kwargs)
                    
                    # Try to extract data size from result if not provided
                    if data_size_mb == 0.0 and hasattr(result, '__len__'):
                        try:
                            if isinstance(result, np.ndarray):
                                metrics['result_size_mb'] = result.nbytes / 1024 / 1024
                        except:
                            pass
                
                return result
            return wrapper
        return decorator
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistical summary for a specific operation.
        
        Args:
            operation_name: Name of the operation to analyze
            
        Returns:
            Dictionary with statistical metrics
        """
        operation_metrics = [m for m in self.metrics_history if m.operation_name == operation_name]
        
        if not operation_metrics:
            return {}
        
        execution_times = [m.execution_time for m in operation_metrics]
        memory_usage = [m.memory_usage_mb for m in operation_metrics]
        throughputs = [m.throughput_mbps for m in operation_metrics if m.throughput_mbps > 0]
        
        stats = {
            'operation_name': operation_name,
            'total_executions': len(operation_metrics),
            'execution_time': {
                'mean': np.mean(execution_times),
                'median': np.median(execution_times),
                'std': np.std(execution_times),
                'min': np.min(execution_times),
                'max': np.max(execution_times)
            },
            'memory_usage': {
                'mean': np.mean(memory_usage),
                'median': np.median(memory_usage),
                'std': np.std(memory_usage),
                'min': np.min(memory_usage),
                'max': np.max(memory_usage)
            }
        }
        
        if throughputs:
            stats['throughput'] = {
                'mean': np.mean(throughputs),
                'median': np.median(throughputs),
                'std': np.std(throughputs),
                'min': np.min(throughputs),
                'max': np.max(throughputs)
            }
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for all operations.
        
        Returns:
            Dictionary with performance summary statistics
        """
        if not self.metrics_history:
            return {'total_operations': 0}
        
        # Group by operation name
        operations = {}
        for metric in self.metrics_history:
            if metric.operation_name not in operations:
                operations[metric.operation_name] = []
            operations[metric.operation_name].append(metric)
        
        summary = {
            'total_operations': len(self.metrics_history),
            'unique_operations': len(operations),
            'operations': {}
        }
        
        for op_name, op_metrics in operations.items():
            summary['operations'][op_name] = self.get_operation_stats(op_name)
        
        # Overall statistics
        all_times = [m.execution_time for m in self.metrics_history]
        all_memory = [m.peak_memory_mb for m in self.metrics_history]
        
        summary['overall'] = {
            'total_execution_time': sum(all_times),
            'average_execution_time': np.mean(all_times),
            'peak_memory_usage': max(all_memory),
            'average_memory_usage': np.mean(all_memory)
        }
        
        return summary
    
    def clear_history(self) -> None:
        """Clear performance metrics history."""
        self.metrics_history.clear()
        logger.info("Performance metrics history cleared")
    
    def export_metrics(self, filepath: str) -> None:
        """
        Export performance metrics to file.
        
        Args:
            filepath: Path to export file (JSON format)
        """
        import json
        
        # Convert metrics to serializable format
        serializable_metrics = []
        for metric in self.metrics_history:
            metric_dict = {
                'operation_name': metric.operation_name,
                'execution_time': metric.execution_time,
                'memory_usage_mb': metric.memory_usage_mb,
                'peak_memory_mb': metric.peak_memory_mb,
                'cpu_usage_percent': metric.cpu_usage_percent,
                'data_size_mb': metric.data_size_mb,
                'throughput_mbps': metric.throughput_mbps,
                'timestamp': metric.timestamp,
                'additional_metrics': metric.additional_metrics
            }
            serializable_metrics.append(metric_dict)
        
        with open(filepath, 'w') as f:
            json.dump({
                'summary': self.get_performance_summary(),
                'detailed_metrics': serializable_metrics
            }, f, indent=2)
        
        logger.info(f"Performance metrics exported to {filepath}")


# Global profiler instance
global_profiler = PerformanceProfiler()


def profile_wave_operation(operation_name: str = None, data_size_mb: float = 0.0):
    """
    Convenience decorator for profiling wave analysis operations.
    
    Args:
        operation_name: Name for the operation
        data_size_mb: Size of data being processed in MB
    """
    return global_profiler.profile_function(operation_name, data_size_mb)