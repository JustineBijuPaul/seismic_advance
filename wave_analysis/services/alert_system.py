"""
Alert System for Earthquake Event Notifications

This module provides real-time alerting capabilities for significant earthquake events
based on wave analysis results and configurable thresholds.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json
from enum import Enum

from ..models import DetailedAnalysis, WaveAnalysisResult, MagnitudeEstimate


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts that can be triggered"""
    MAGNITUDE_THRESHOLD = "magnitude_threshold"
    WAVE_AMPLITUDE = "wave_amplitude"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    MULTIPLE_EVENTS = "multiple_events"
    DATA_QUALITY = "data_quality"


@dataclass
class AlertThreshold:
    """Configuration for alert thresholds"""
    alert_type: AlertType
    severity: AlertSeverity
    threshold_value: float
    wave_types: List[str]  # Which wave types to monitor
    enabled: bool = True
    description: str = ""


@dataclass
class AlertEvent:
    """Represents an alert event"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    analysis_data: Dict[str, Any]
    threshold_exceeded: Dict[str, float]
    location_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for JSON serialization"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'analysis_data': self.analysis_data,
            'threshold_exceeded': self.threshold_exceeded,
            'location_info': self.location_info
        }


class AlertHandlerInterface(ABC):
    """Interface for alert handlers"""
    
    @abstractmethod
    def handle_alert(self, alert: AlertEvent) -> bool:
        """Handle an alert event"""
        pass


class WebSocketAlertHandler(AlertHandlerInterface):
    """Handler for WebSocket-based real-time alerts"""
    
    def __init__(self, socketio_instance):
        self.socketio = socketio_instance
        self.logger = logging.getLogger(__name__)
    
    def handle_alert(self, alert: AlertEvent) -> bool:
        """Send alert via WebSocket to connected clients"""
        try:
            self.socketio.emit('earthquake_alert', alert.to_dict(), namespace='/alerts')
            self.logger.info(f"WebSocket alert sent: {alert.alert_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send WebSocket alert: {e}")
            return False


class LogAlertHandler(AlertHandlerInterface):
    """Handler for logging alerts"""
    
    def __init__(self):
        self.logger = logging.getLogger('earthquake_alerts')
    
    def handle_alert(self, alert: AlertEvent) -> bool:
        """Log alert to file"""
        try:
            log_message = f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}"
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                self.logger.error(log_message)
            elif alert.severity == AlertSeverity.MEDIUM:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            return True
        except Exception as e:
            self.logger.error(f"Failed to log alert: {e}")
            return False


class AlertSystem:
    """
    Main alert system for earthquake event notifications.
    
    Monitors wave analysis results and triggers alerts based on configurable
    thresholds for magnitude, wave characteristics, and other parameters.
    """
    
    def __init__(self):
        self.thresholds: List[AlertThreshold] = []
        self.handlers: List[AlertHandlerInterface] = []
        self.alert_history: List[AlertEvent] = []
        self.logger = logging.getLogger(__name__)
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self):
        """Setup default alert thresholds"""
        default_thresholds = [
            AlertThreshold(
                alert_type=AlertType.MAGNITUDE_THRESHOLD,
                severity=AlertSeverity.MEDIUM,
                threshold_value=4.0,
                wave_types=['P', 'S'],
                description="Medium magnitude earthquake detected"
            ),
            AlertThreshold(
                alert_type=AlertType.MAGNITUDE_THRESHOLD,
                severity=AlertSeverity.HIGH,
                threshold_value=5.5,
                wave_types=['P', 'S'],
                description="High magnitude earthquake detected"
            ),
            AlertThreshold(
                alert_type=AlertType.MAGNITUDE_THRESHOLD,
                severity=AlertSeverity.CRITICAL,
                threshold_value=7.0,
                wave_types=['P', 'S'],
                description="Critical magnitude earthquake detected"
            ),
            AlertThreshold(
                alert_type=AlertType.WAVE_AMPLITUDE,
                severity=AlertSeverity.HIGH,
                threshold_value=1000.0,  # Amplitude threshold
                wave_types=['P', 'S', 'Love', 'Rayleigh'],
                description="High amplitude waves detected"
            ),
            AlertThreshold(
                alert_type=AlertType.FREQUENCY_ANOMALY,
                severity=AlertSeverity.MEDIUM,
                threshold_value=0.1,  # Frequency deviation threshold
                wave_types=['P', 'S'],
                description="Unusual frequency patterns detected"
            )
        ]
        self.thresholds.extend(default_thresholds)
    
    def add_handler(self, handler: AlertHandlerInterface):
        """Add an alert handler"""
        self.handlers.append(handler)
        self.logger.info(f"Added alert handler: {type(handler).__name__}")
    
    def remove_handler(self, handler: AlertHandlerInterface):
        """Remove an alert handler"""
        if handler in self.handlers:
            self.handlers.remove(handler)
            self.logger.info(f"Removed alert handler: {type(handler).__name__}")
    
    def add_threshold(self, threshold: AlertThreshold):
        """Add a custom alert threshold"""
        self.thresholds.append(threshold)
        self.logger.info(f"Added alert threshold: {threshold.alert_type.value}")
    
    def update_threshold(self, alert_type: AlertType, **kwargs):
        """Update an existing threshold"""
        for threshold in self.thresholds:
            if threshold.alert_type == alert_type:
                for key, value in kwargs.items():
                    if hasattr(threshold, key):
                        setattr(threshold, key, value)
                self.logger.info(f"Updated threshold for {alert_type.value}")
                return
        self.logger.warning(f"Threshold not found for {alert_type.value}")
    
    def check_analysis_for_alerts(self, analysis: DetailedAnalysis) -> List[AlertEvent]:
        """
        Check analysis results against alert thresholds.
        
        Args:
            analysis: Detailed wave analysis results
            
        Returns:
            List of triggered alert events
        """
        triggered_alerts = []
        
        # Check magnitude thresholds
        magnitude_alerts = self._check_magnitude_thresholds(analysis)
        triggered_alerts.extend(magnitude_alerts)
        
        # Check wave amplitude thresholds
        amplitude_alerts = self._check_amplitude_thresholds(analysis)
        triggered_alerts.extend(amplitude_alerts)
        
        # Check frequency anomalies
        frequency_alerts = self._check_frequency_anomalies(analysis)
        triggered_alerts.extend(frequency_alerts)
        
        # Process all triggered alerts
        for alert in triggered_alerts:
            self._process_alert(alert)
        
        return triggered_alerts
    
    def _check_magnitude_thresholds(self, analysis: DetailedAnalysis) -> List[AlertEvent]:
        """Check magnitude-based alert thresholds"""
        alerts = []
        
        magnitude_thresholds = [t for t in self.thresholds 
                              if t.alert_type == AlertType.MAGNITUDE_THRESHOLD and t.enabled]
        
        for estimate in analysis.magnitude_estimates:
            magnitude = estimate.magnitude
            
            # Find the highest severity threshold that is exceeded
            triggered_threshold = None
            for threshold in magnitude_thresholds:
                if magnitude >= threshold.threshold_value:
                    if (triggered_threshold is None or 
                        threshold.threshold_value > triggered_threshold.threshold_value):
                        triggered_threshold = threshold
            
            if triggered_threshold:
                alert = AlertEvent(
                    alert_id=f"mag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timestamp=datetime.now(),
                    alert_type=AlertType.MAGNITUDE_THRESHOLD,
                    severity=triggered_threshold.severity,
                    title=f"Magnitude {magnitude:.1f} Earthquake Detected",
                    message=f"Earthquake with magnitude {magnitude:.1f} detected using {estimate.method} method. {triggered_threshold.description}",
                    analysis_data={
                        'magnitude': magnitude,
                        'method': estimate.method,
                        'confidence': estimate.confidence,
                        'wave_type': estimate.wave_type_used
                    },
                    threshold_exceeded={'magnitude': magnitude - triggered_threshold.threshold_value}
                )
                alerts.append(alert)
        
        return alerts
    
    def _check_amplitude_thresholds(self, analysis: DetailedAnalysis) -> List[AlertEvent]:
        """Check wave amplitude thresholds"""
        alerts = []
        
        amplitude_thresholds = [t for t in self.thresholds 
                              if t.alert_type == AlertType.WAVE_AMPLITUDE and t.enabled]
        
        # Check amplitudes for each wave type
        wave_types = ['p_waves', 's_waves', 'surface_waves']
        for wave_type in wave_types:
            waves = getattr(analysis.wave_result, wave_type, [])
            
            for wave in waves:
                for threshold in amplitude_thresholds:
                    if wave.wave_type in threshold.wave_types and wave.peak_amplitude >= threshold.threshold_value:
                        alert = AlertEvent(
                            alert_id=f"amp_{wave.wave_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            timestamp=datetime.now(),
                            alert_type=AlertType.WAVE_AMPLITUDE,
                            severity=threshold.severity,
                            title=f"High Amplitude {wave.wave_type}-Wave Detected",
                            message=f"High amplitude {wave.wave_type}-wave detected with peak amplitude {wave.peak_amplitude:.2f}",
                            analysis_data={
                                'wave_type': wave.wave_type,
                                'peak_amplitude': wave.peak_amplitude,
                                'dominant_frequency': wave.dominant_frequency,
                                'arrival_time': wave.arrival_time
                            },
                            threshold_exceeded={'amplitude': wave.peak_amplitude - threshold.threshold_value}
                        )
                        alerts.append(alert)
        
        return alerts
    
    def _check_frequency_anomalies(self, analysis: DetailedAnalysis) -> List[AlertEvent]:
        """Check for frequency anomalies"""
        alerts = []
        
        frequency_thresholds = [t for t in self.thresholds 
                              if t.alert_type == AlertType.FREQUENCY_ANOMALY and t.enabled]
        
        # Check for unusual frequency patterns
        for wave_type, freq_data in analysis.frequency_analysis.items():
            expected_freq_ranges = {
                'P': (1.0, 10.0),
                'S': (0.5, 5.0),
                'Love': (0.02, 0.5),
                'Rayleigh': (0.02, 0.5)
            }
            
            if wave_type in expected_freq_ranges:
                expected_min, expected_max = expected_freq_ranges[wave_type]
                dominant_freq = freq_data.dominant_frequency
                
                # Check if frequency is outside expected range
                if dominant_freq < expected_min or dominant_freq > expected_max:
                    for threshold in frequency_thresholds:
                        if wave_type in threshold.wave_types:
                            alert = AlertEvent(
                                alert_id=f"freq_{wave_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                timestamp=datetime.now(),
                                alert_type=AlertType.FREQUENCY_ANOMALY,
                                severity=threshold.severity,
                                title=f"Frequency Anomaly in {wave_type}-Waves",
                                message=f"Unusual frequency pattern detected in {wave_type}-waves: {dominant_freq:.2f} Hz",
                                analysis_data={
                                    'wave_type': wave_type,
                                    'dominant_frequency': dominant_freq,
                                    'expected_range': expected_freq_ranges[wave_type],
                                    'power_spectrum': freq_data.power_spectrum[:10].tolist()  # First 10 values
                                },
                                threshold_exceeded={'frequency_deviation': abs(dominant_freq - (expected_min + expected_max) / 2)}
                            )
                            alerts.append(alert)
        
        return alerts
    
    def _process_alert(self, alert: AlertEvent):
        """Process an alert by sending it to all handlers"""
        self.alert_history.append(alert)
        
        # Keep only last 1000 alerts in memory
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Send alert to all handlers
        for handler in self.handlers:
            try:
                handler.handle_alert(alert)
            except Exception as e:
                self.logger.error(f"Handler {type(handler).__name__} failed to process alert: {e}")
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts as dictionaries"""
        recent_alerts = self.alert_history[-limit:] if limit > 0 else self.alert_history
        return [alert.to_dict() for alert in reversed(recent_alerts)]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent alerts"""
        if not self.alert_history:
            return {'total_alerts': 0}
        
        # Count alerts by severity and type
        severity_counts = {}
        type_counts = {}
        
        for alert in self.alert_history:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            type_counts[alert.alert_type.value] = type_counts.get(alert.alert_type.value, 0) + 1
        
        return {
            'total_alerts': len(self.alert_history),
            'by_severity': severity_counts,
            'by_type': type_counts,
            'latest_alert': self.alert_history[-1].to_dict() if self.alert_history else None
        }
    
    def clear_alert_history(self):
        """Clear alert history"""
        self.alert_history.clear()
        self.logger.info("Alert history cleared")