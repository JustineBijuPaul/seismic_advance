"""
Logging and monitoring configuration for wave analysis operations.
Provides structured logging with performance metrics and alert tracking.
"""

import os
import logging
import logging.handlers
import time
import functools
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json

from .config import config_manager


class WaveAnalysisLogger:
    """Centralized logging system for wave analysis operations."""
    
    def __init__(self):
        self.config = config_manager.logging
        self.loggers = {}
        self.performance_metrics = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration based on settings."""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.config.format)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for main logs
        if self.config.enable_file_logging:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.log_file_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        
        # Performance logging
        if self.config.enable_performance_logging:
            self._setup_performance_logging()
        
        # Alert logging
        if self.config.enable_alert_logging:
            self._setup_alert_logging()
    
    def _setup_performance_logging(self):
        """Set up performance-specific logging."""
        perf_logger = logging.getLogger('wave_analysis.performance')
        perf_logger.setLevel(logging.INFO)
        
        # Create performance log directory
        perf_log_path = Path(self.config.performance_log_file)
        perf_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        perf_handler = logging.handlers.RotatingFileHandler(
            self.config.performance_log_file,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count
        )
        
        perf_formatter = logging.Formatter(
            '%(asctime)s - PERFORMANCE - %(message)s'
        )
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        
        self.loggers['performance'] = perf_logger
    
    def _setup_alert_logging(self):
        """Set up alert-specific logging."""
        alert_logger = logging.getLogger('wave_analysis.alerts')
        alert_logger.setLevel(logging.INFO)
        
        # Create alert log directory
        alert_log_path = Path(self.config.alert_log_file)
        alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        alert_handler = logging.handlers.RotatingFileHandler(
            self.config.alert_log_file,
            maxBytes=self.config.max_file_size_mb * 1024 * 1024,
            backupCount=self.config.backup_count
        )
        
        alert_formatter = logging.Formatter(
            '%(asctime)s - ALERT - %(message)s'
        )
        alert_handler.setFormatter(alert_formatter)
        alert_logger.addHandler(alert_handler)
        
        self.loggers['alerts'] = alert_logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific component."""
        return logging.getLogger(f'wave_analysis.{name}')
    
    def log_performance_metric(self, operation: str, duration: float, 
                             metadata: Optional[Dict[str, Any]] = None):
        """Log performance metrics for wave analysis operations."""
        if 'performance' not in self.loggers:
            return
        
        metric_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        # Store in memory for aggregation
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = []
        
        self.performance_metrics[operation].append({
            'duration': duration,
            'timestamp': datetime.utcnow(),
            'metadata': metadata or {}
        })
        
        # Keep only recent metrics (last 1000 entries per operation)
        if len(self.performance_metrics[operation]) > 1000:
            self.performance_metrics[operation] = self.performance_metrics[operation][-1000:]
        
        # Log to file
        self.loggers['performance'].info(json.dumps(metric_data))
    
    def log_alert(self, alert_type: str, message: str, severity: str = 'INFO',
                  metadata: Optional[Dict[str, Any]] = None):
        """Log alert information."""
        if 'alerts' not in self.loggers:
            return
        
        alert_data = {
            'alert_type': alert_type,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        self.loggers['alerts'].info(json.dumps(alert_data))
    
    def get_performance_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics summary."""
        if operation:
            metrics = self.performance_metrics.get(operation, [])
            if not metrics:
                return {}
            
            durations = [m['duration'] for m in metrics]
            return {
                'operation': operation,
                'count': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'recent_count': len([m for m in metrics 
                                   if (datetime.utcnow() - m['timestamp']).seconds < 3600])
            }
        else:
            summary = {}
            for op, metrics in self.performance_metrics.items():
                if metrics:
                    durations = [m['duration'] for m in metrics]
                    summary[op] = {
                        'count': len(durations),
                        'avg_duration': sum(durations) / len(durations),
                        'min_duration': min(durations),
                        'max_duration': max(durations)
                    }
            return summary


def performance_monitor(operation_name: str):
    """Decorator to monitor performance of wave analysis operations."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = wave_logger.get_logger('performance')
            
            try:
                logger.debug(f"Starting {operation_name}")
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                wave_logger.log_performance_metric(
                    operation_name, 
                    duration,
                    {
                        'success': True,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                )
                
                logger.debug(f"Completed {operation_name} in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                wave_logger.log_performance_metric(
                    operation_name,
                    duration,
                    {
                        'success': False,
                        'error': str(e),
                        'error_type': type(e).__name__
                    }
                )
                
                logger.error(f"Failed {operation_name} after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


class HealthMonitor:
    """System health monitoring for wave analysis operations."""
    
    def __init__(self):
        self.config = config_manager.deployment
        self.logger = wave_logger.get_logger('health')
        self.last_check = None
        self.health_status = {}
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check."""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        # Check configuration validity
        config_issues = config_manager.validate_configuration()
        health_status['checks']['configuration'] = {
            'status': 'healthy' if not config_issues['errors'] else 'unhealthy',
            'errors': config_issues['errors'],
            'warnings': config_issues['warnings']
        }
        
        # Check logging system
        health_status['checks']['logging'] = self._check_logging_health()
        
        # Check performance metrics
        health_status['checks']['performance'] = self._check_performance_health()
        
        # Check disk space for logs
        health_status['checks']['disk_space'] = self._check_disk_space()
        
        # Determine overall status
        unhealthy_checks = [
            check for check in health_status['checks'].values()
            if check['status'] == 'unhealthy'
        ]
        
        if unhealthy_checks:
            health_status['overall_status'] = 'unhealthy'
        elif any(check.get('warnings') for check in health_status['checks'].values()):
            health_status['overall_status'] = 'warning'
        
        self.health_status = health_status
        self.last_check = datetime.utcnow()
        
        # Log health status
        if health_status['overall_status'] != 'healthy':
            self.logger.warning(f"System health check: {health_status['overall_status']}")
            wave_logger.log_alert(
                'system_health',
                f"System health status: {health_status['overall_status']}",
                'WARNING' if health_status['overall_status'] == 'warning' else 'ERROR',
                health_status
            )
        
        return health_status
    
    def _check_logging_health(self) -> Dict[str, Any]:
        """Check logging system health."""
        try:
            # Test log writing
            test_logger = wave_logger.get_logger('health_test')
            test_logger.info("Health check test log entry")
            
            return {
                'status': 'healthy',
                'message': 'Logging system operational'
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Logging system error: {str(e)}'
            }
    
    def _check_performance_health(self) -> Dict[str, Any]:
        """Check performance metrics health."""
        try:
            summary = wave_logger.get_performance_summary()
            
            # Check for operations taking too long
            slow_operations = []
            for operation, metrics in summary.items():
                if metrics.get('avg_duration', 0) > 30:  # 30 seconds threshold
                    slow_operations.append(operation)
            
            if slow_operations:
                return {
                    'status': 'warning',
                    'message': f'Slow operations detected: {", ".join(slow_operations)}',
                    'slow_operations': slow_operations
                }
            
            return {
                'status': 'healthy',
                'message': 'Performance metrics within normal range',
                'operations_count': len(summary)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Performance monitoring error: {str(e)}'
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space for log files."""
        try:
            log_dir = Path(config_manager.logging.log_file_path).parent
            stat = os.statvfs(log_dir)
            
            # Calculate available space in MB
            available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            
            if available_mb < 100:  # Less than 100MB available
                return {
                    'status': 'unhealthy',
                    'message': f'Low disk space: {available_mb:.1f}MB available',
                    'available_mb': available_mb
                }
            elif available_mb < 500:  # Less than 500MB available
                return {
                    'status': 'warning',
                    'message': f'Disk space getting low: {available_mb:.1f}MB available',
                    'available_mb': available_mb
                }
            
            return {
                'status': 'healthy',
                'message': f'Sufficient disk space: {available_mb:.1f}MB available',
                'available_mb': available_mb
            }
        except Exception as e:
            return {
                'status': 'warning',
                'message': f'Could not check disk space: {str(e)}'
            }


# Global instances
wave_logger = WaveAnalysisLogger()
health_monitor = HealthMonitor()