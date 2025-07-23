"""
Integration tests for the Alert System

Tests the alert system functionality including threshold-based alerting,
WebSocket notifications, and various earthquake scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime
import json
import time

from wave_analysis.services.alert_system import (
    AlertSystem, AlertEvent, AlertThreshold, AlertSeverity, AlertType,
    WebSocketAlertHandler, LogAlertHandler
)
from wave_analysis.models import (
    DetailedAnalysis, WaveAnalysisResult, WaveSegment, ArrivalTimes, 
    MagnitudeEstimate, FrequencyData, QualityMetrics
)


class TestAlertSystem(unittest.TestCase):
    """Test cases for the AlertSystem class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.alert_system = AlertSystem()
        self.mock_socketio = Mock()
        self.websocket_handler = WebSocketAlertHandler(self.mock_socketio)
        self.log_handler = LogAlertHandler()
        
        # Add handlers to alert system
        self.alert_system.add_handler(self.websocket_handler)
        self.alert_system.add_handler(self.log_handler)
    
    def create_mock_analysis(self, magnitude=4.5, p_amplitude=500.0, s_amplitude=300.0):
        """Create a mock DetailedAnalysis object for testing"""
        # Create wave segments
        p_wave = WaveSegment(
            wave_type='P',
            start_time=10.0,
            end_time=15.0,
            data=np.random.randn(500),
            sampling_rate=100.0,
            peak_amplitude=p_amplitude,
            dominant_frequency=5.0,
            arrival_time=12.0
        )
        
        s_wave = WaveSegment(
            wave_type='S',
            start_time=20.0,
            end_time=30.0,
            data=np.random.randn(1000),
            sampling_rate=100.0,
            peak_amplitude=s_amplitude,
            dominant_frequency=2.0,
            arrival_time=25.0
        )
        
        # Create wave analysis result
        wave_result = WaveAnalysisResult(
            original_data=np.random.randn(5000),
            sampling_rate=100.0,
            p_waves=[p_wave],
            s_waves=[s_wave],
            surface_waves=[],
            metadata={}
        )
        
        # Create magnitude estimates
        magnitude_estimates = [
            MagnitudeEstimate(
                method='ML',
                magnitude=magnitude,
                confidence=0.8,
                wave_type_used='P'
            )
        ]
        
        # Create arrival times
        arrival_times = ArrivalTimes(
            p_wave_arrival=12.0,
            s_wave_arrival=25.0,
            sp_time_difference=13.0,
            surface_wave_arrival=0.0
        )
        
        # Create frequency analysis
        frequency_analysis = {
            'P': FrequencyData(
                frequencies=np.linspace(0, 50, 100),
                power_spectrum=np.random.randn(100),
                dominant_frequency=5.0,
                frequency_range=(1.0, 10.0),
                spectral_centroid=5.2,
                bandwidth=2.0
            ),
            'S': FrequencyData(
                frequencies=np.linspace(0, 50, 100),
                power_spectrum=np.random.randn(100),
                dominant_frequency=2.0,
                frequency_range=(0.5, 5.0),
                spectral_centroid=2.1,
                bandwidth=1.5
            )
        }
        
        # Create quality metrics
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=10.0,
            detection_confidence=0.85,
            analysis_quality_score=0.85,
            data_completeness=1.0
        )
        
        # Create detailed analysis
        return DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates,
            epicenter_distance=50.0,
            frequency_analysis=frequency_analysis,
            quality_metrics=quality_metrics
        )
    
    def test_default_thresholds_setup(self):
        """Test that default thresholds are properly set up"""
        self.assertGreater(len(self.alert_system.thresholds), 0)
        
        # Check for magnitude thresholds
        magnitude_thresholds = [t for t in self.alert_system.thresholds 
                              if t.alert_type == AlertType.MAGNITUDE_THRESHOLD]
        self.assertGreater(len(magnitude_thresholds), 0)
        
        # Check for amplitude thresholds
        amplitude_thresholds = [t for t in self.alert_system.thresholds 
                              if t.alert_type == AlertType.WAVE_AMPLITUDE]
        self.assertGreater(len(amplitude_thresholds), 0)
    
    def test_magnitude_threshold_alert(self):
        """Test magnitude-based alert triggering"""
        # Create analysis with high magnitude
        analysis = self.create_mock_analysis(magnitude=6.0)
        
        # Check for alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should trigger magnitude alert
        magnitude_alerts = [a for a in alerts if a.alert_type == AlertType.MAGNITUDE_THRESHOLD]
        self.assertGreater(len(magnitude_alerts), 0)
        
        # Check alert properties
        alert = magnitude_alerts[0]
        self.assertEqual(alert.alert_type, AlertType.MAGNITUDE_THRESHOLD)
        self.assertIn('6.0', alert.message)
        self.assertGreater(alert.threshold_exceeded['magnitude'], 0)
    
    def test_amplitude_threshold_alert(self):
        """Test amplitude-based alert triggering"""
        # Create analysis with high amplitude waves
        analysis = self.create_mock_analysis(p_amplitude=1500.0, s_amplitude=1200.0)
        
        # Check for alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should trigger amplitude alert
        amplitude_alerts = [a for a in alerts if a.alert_type == AlertType.WAVE_AMPLITUDE]
        self.assertGreater(len(amplitude_alerts), 0)
        
        # Check alert properties
        alert = amplitude_alerts[0]
        self.assertEqual(alert.alert_type, AlertType.WAVE_AMPLITUDE)
        self.assertIn('amplitude', alert.message.lower())
    
    def test_frequency_anomaly_alert(self):
        """Test frequency anomaly detection"""
        # Create analysis with unusual frequency
        analysis = self.create_mock_analysis()
        
        # Modify frequency data to be outside normal range
        analysis.frequency_analysis['P'].dominant_frequency = 15.0  # Too high for P-wave
        
        # Check for alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should trigger frequency anomaly alert
        freq_alerts = [a for a in alerts if a.alert_type == AlertType.FREQUENCY_ANOMALY]
        self.assertGreater(len(freq_alerts), 0)
        
        # Check alert properties
        alert = freq_alerts[0]
        self.assertEqual(alert.alert_type, AlertType.FREQUENCY_ANOMALY)
        self.assertIn('frequency', alert.message.lower())
    
    def test_multiple_alerts_scenario(self):
        """Test scenario with multiple alert conditions"""
        # Create analysis that triggers multiple alerts
        analysis = self.create_mock_analysis(magnitude=7.5, p_amplitude=2000.0)
        analysis.frequency_analysis['S'].dominant_frequency = 8.0  # Unusual for S-wave
        
        # Check for alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should trigger multiple types of alerts
        alert_types = {alert.alert_type for alert in alerts}
        self.assertGreater(len(alert_types), 1)
        
        # Should include magnitude and amplitude alerts at minimum
        self.assertIn(AlertType.MAGNITUDE_THRESHOLD, alert_types)
        self.assertIn(AlertType.WAVE_AMPLITUDE, alert_types)
    
    def test_websocket_handler(self):
        """Test WebSocket alert handler"""
        # Create a test alert
        alert = AlertEvent(
            alert_id='test_alert_001',
            timestamp=datetime.now(),
            alert_type=AlertType.MAGNITUDE_THRESHOLD,
            severity=AlertSeverity.HIGH,
            title='Test Alert',
            message='Test alert message',
            analysis_data={'magnitude': 5.5},
            threshold_exceeded={'magnitude': 1.5}
        )
        
        # Handle the alert
        result = self.websocket_handler.handle_alert(alert)
        
        # Should succeed
        self.assertTrue(result)
        
        # Should have called socketio.emit
        self.mock_socketio.emit.assert_called_once()
        call_args = self.mock_socketio.emit.call_args
        self.assertEqual(call_args[0][0], 'earthquake_alert')
        self.assertEqual(call_args[1]['namespace'], '/alerts')
    
    def test_log_handler(self):
        """Test log alert handler"""
        # Create test alerts with different severities
        alerts = [
            AlertEvent(
                alert_id='test_info',
                timestamp=datetime.now(),
                alert_type=AlertType.MAGNITUDE_THRESHOLD,
                severity=AlertSeverity.LOW,
                title='Low Severity Alert',
                message='Low severity test',
                analysis_data={},
                threshold_exceeded={}
            ),
            AlertEvent(
                alert_id='test_critical',
                timestamp=datetime.now(),
                alert_type=AlertType.MAGNITUDE_THRESHOLD,
                severity=AlertSeverity.CRITICAL,
                title='Critical Alert',
                message='Critical test',
                analysis_data={},
                threshold_exceeded={}
            )
        ]
        
        # Handle alerts
        for alert in alerts:
            result = self.log_handler.handle_alert(alert)
            self.assertTrue(result)
    
    def test_alert_history_management(self):
        """Test alert history storage and retrieval"""
        # Create multiple test alerts
        for i in range(5):
            analysis = self.create_mock_analysis(magnitude=4.0 + i * 0.5)
            self.alert_system.check_analysis_for_alerts(analysis)
        
        # Check alert history
        recent_alerts = self.alert_system.get_recent_alerts(limit=10)
        self.assertGreater(len(recent_alerts), 0)
        
        # Check alert statistics
        stats = self.alert_system.get_alert_statistics()
        self.assertIn('total_alerts', stats)
        self.assertIn('by_severity', stats)
        self.assertIn('by_type', stats)
        self.assertGreater(stats['total_alerts'], 0)
    
    def test_threshold_management(self):
        """Test adding and updating alert thresholds"""
        # Add custom threshold
        custom_threshold = AlertThreshold(
            alert_type=AlertType.MAGNITUDE_THRESHOLD,
            severity=AlertSeverity.CRITICAL,
            threshold_value=8.0,
            wave_types=['P', 'S'],
            description='Custom critical threshold'
        )
        
        initial_count = len(self.alert_system.thresholds)
        self.alert_system.add_threshold(custom_threshold)
        self.assertEqual(len(self.alert_system.thresholds), initial_count + 1)
        
        # Update existing threshold
        self.alert_system.update_threshold(
            AlertType.MAGNITUDE_THRESHOLD,
            threshold_value=3.5,
            enabled=False
        )
        
        # Verify update
        updated_threshold = next(
            (t for t in self.alert_system.thresholds 
             if t.alert_type == AlertType.MAGNITUDE_THRESHOLD and t.threshold_value == 3.5),
            None
        )
        self.assertIsNotNone(updated_threshold)
        self.assertFalse(updated_threshold.enabled)
    
    def test_alert_serialization(self):
        """Test alert event serialization to dictionary"""
        alert = AlertEvent(
            alert_id='test_serialization',
            timestamp=datetime.now(),
            alert_type=AlertType.WAVE_AMPLITUDE,
            severity=AlertSeverity.MEDIUM,
            title='Serialization Test',
            message='Test message',
            analysis_data={'test_key': 'test_value'},
            threshold_exceeded={'amplitude': 100.0},
            location_info={'latitude': 37.7749, 'longitude': -122.4194}
        )
        
        # Convert to dictionary
        alert_dict = alert.to_dict()
        
        # Verify all fields are present
        required_fields = [
            'alert_id', 'timestamp', 'alert_type', 'severity',
            'title', 'message', 'analysis_data', 'threshold_exceeded',
            'location_info'
        ]
        
        for field in required_fields:
            self.assertIn(field, alert_dict)
        
        # Verify timestamp is ISO format string
        self.assertIsInstance(alert_dict['timestamp'], str)
        
        # Verify enum values are converted to strings
        self.assertEqual(alert_dict['alert_type'], 'wave_amplitude')
        self.assertEqual(alert_dict['severity'], 'medium')
    
    def test_no_alerts_for_low_magnitude(self):
        """Test that low magnitude earthquakes don't trigger alerts"""
        # Create analysis with very low magnitude
        analysis = self.create_mock_analysis(magnitude=2.0, p_amplitude=50.0)
        
        # Check for alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should not trigger any alerts
        self.assertEqual(len(alerts), 0)
    
    def test_alert_system_with_disabled_thresholds(self):
        """Test that disabled thresholds don't trigger alerts"""
        # Disable all magnitude thresholds
        for threshold in self.alert_system.thresholds:
            if threshold.alert_type == AlertType.MAGNITUDE_THRESHOLD:
                threshold.enabled = False
        
        # Create analysis with high magnitude
        analysis = self.create_mock_analysis(magnitude=7.0)
        
        # Check for alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should not trigger magnitude alerts
        magnitude_alerts = [a for a in alerts if a.alert_type == AlertType.MAGNITUDE_THRESHOLD]
        self.assertEqual(len(magnitude_alerts), 0)
    
    def test_clear_alert_history(self):
        """Test clearing alert history"""
        # Generate some alerts
        analysis = self.create_mock_analysis(magnitude=5.0)
        self.alert_system.check_analysis_for_alerts(analysis)
        
        # Verify alerts exist
        self.assertGreater(len(self.alert_system.alert_history), 0)
        
        # Clear history
        self.alert_system.clear_alert_history()
        
        # Verify history is cleared
        self.assertEqual(len(self.alert_system.alert_history), 0)
        
        # Verify statistics reflect empty history
        stats = self.alert_system.get_alert_statistics()
        self.assertEqual(stats['total_alerts'], 0)


class TestAlertSystemIntegration(unittest.TestCase):
    """Integration tests for alert system with various earthquake scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.alert_system = AlertSystem()
        self.mock_socketio = Mock()
        self.websocket_handler = WebSocketAlertHandler(self.mock_socketio)
        self.alert_system.add_handler(self.websocket_handler)
    
    def test_small_local_earthquake_scenario(self):
        """Test alert behavior for small local earthquake (M3.5)"""
        # Create analysis for small local earthquake
        analysis = self.create_earthquake_scenario(
            magnitude=3.5,
            distance=10.0,  # km
            p_amplitude=200.0,
            s_amplitude=150.0
        )
        
        # Check alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Small earthquake should not trigger alerts with default thresholds
        self.assertEqual(len(alerts), 0)
    
    def test_moderate_regional_earthquake_scenario(self):
        """Test alert behavior for moderate regional earthquake (M5.2)"""
        # Create analysis for moderate regional earthquake
        analysis = self.create_earthquake_scenario(
            magnitude=5.2,
            distance=100.0,  # km
            p_amplitude=800.0,
            s_amplitude=600.0
        )
        
        # Check alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should trigger medium severity magnitude alert
        magnitude_alerts = [a for a in alerts if a.alert_type == AlertType.MAGNITUDE_THRESHOLD]
        self.assertGreater(len(magnitude_alerts), 0)
        
        alert = magnitude_alerts[0]
        self.assertEqual(alert.severity, AlertSeverity.MEDIUM)
    
    def test_large_distant_earthquake_scenario(self):
        """Test alert behavior for large distant earthquake (M7.1)"""
        # Create analysis for large distant earthquake
        analysis = self.create_earthquake_scenario(
            magnitude=7.1,
            distance=500.0,  # km
            p_amplitude=1200.0,
            s_amplitude=900.0
        )
        
        # Check alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should trigger critical magnitude alert and high amplitude alert
        alert_types = {alert.alert_type for alert in alerts}
        severities = {alert.severity for alert in alerts}
        
        self.assertIn(AlertType.MAGNITUDE_THRESHOLD, alert_types)
        self.assertIn(AlertType.WAVE_AMPLITUDE, alert_types)
        self.assertIn(AlertSeverity.CRITICAL, severities)
    
    def test_deep_earthquake_scenario(self):
        """Test alert behavior for deep earthquake with unusual characteristics"""
        # Create analysis for deep earthquake (different frequency characteristics)
        analysis = self.create_earthquake_scenario(
            magnitude=6.0,
            distance=200.0,
            p_amplitude=600.0,
            s_amplitude=400.0,
            p_frequency=12.0,  # Higher frequency for deep earthquake
            s_frequency=6.0
        )
        
        # Check alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should trigger magnitude alert and frequency anomaly
        alert_types = {alert.alert_type for alert in alerts}
        self.assertIn(AlertType.MAGNITUDE_THRESHOLD, alert_types)
        self.assertIn(AlertType.FREQUENCY_ANOMALY, alert_types)
    
    def test_tsunami_generating_earthquake_scenario(self):
        """Test alert behavior for large shallow earthquake (potential tsunami)"""
        # Create analysis for large shallow earthquake
        analysis = self.create_earthquake_scenario(
            magnitude=8.2,
            distance=300.0,
            p_amplitude=2500.0,
            s_amplitude=2000.0,
            surface_wave_amplitude=1500.0  # Strong surface waves
        )
        
        # Check alerts
        alerts = self.alert_system.check_analysis_for_alerts(analysis)
        
        # Should trigger multiple critical alerts
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        self.assertGreater(len(critical_alerts), 0)
        
        # Should include magnitude and amplitude alerts
        alert_types = {alert.alert_type for alert in alerts}
        self.assertIn(AlertType.MAGNITUDE_THRESHOLD, alert_types)
        self.assertIn(AlertType.WAVE_AMPLITUDE, alert_types)
    
    def test_earthquake_swarm_scenario(self):
        """Test alert behavior for earthquake swarm (multiple events)"""
        # Simulate multiple earthquakes in sequence
        magnitudes = [3.2, 3.8, 4.1, 4.5, 3.9, 4.2]
        all_alerts = []
        
        for magnitude in magnitudes:
            analysis = self.create_earthquake_scenario(
                magnitude=magnitude,
                distance=50.0,
                p_amplitude=magnitude * 100,
                s_amplitude=magnitude * 80
            )
            
            alerts = self.alert_system.check_analysis_for_alerts(analysis)
            all_alerts.extend(alerts)
        
        # Should trigger alerts for larger events in the swarm
        magnitude_alerts = [a for a in all_alerts if a.alert_type == AlertType.MAGNITUDE_THRESHOLD]
        self.assertGreater(len(magnitude_alerts), 0)
        
        # Check alert history accumulation
        stats = self.alert_system.get_alert_statistics()
        self.assertGreater(stats['total_alerts'], 0)
    
    def create_earthquake_scenario(self, magnitude, distance, p_amplitude, s_amplitude, 
                                 p_frequency=5.0, s_frequency=2.0, surface_wave_amplitude=0.0):
        """Helper method to create earthquake scenario analysis"""
        from wave_analysis.models import (
            DetailedAnalysis, WaveAnalysisResult, WaveSegment, ArrivalTimes, 
            MagnitudeEstimate, FrequencyData, QualityMetrics
        )
        
        # Calculate arrival times based on distance
        p_velocity = 6.0  # km/s
        s_velocity = 3.5  # km/s
        
        p_arrival = distance / p_velocity
        s_arrival = distance / s_velocity
        sp_time = s_arrival - p_arrival
        
        # Create wave segments
        waves = []
        
        # P-wave
        p_wave = WaveSegment(
            wave_type='P',
            start_time=p_arrival,
            end_time=p_arrival + 5.0,
            data=np.random.randn(500),
            sampling_rate=100.0,
            peak_amplitude=p_amplitude,
            dominant_frequency=p_frequency,
            arrival_time=p_arrival
        )
        waves.append(p_wave)
        
        # S-wave
        s_wave = WaveSegment(
            wave_type='S',
            start_time=s_arrival,
            end_time=s_arrival + 10.0,
            data=np.random.randn(1000),
            sampling_rate=100.0,
            peak_amplitude=s_amplitude,
            dominant_frequency=s_frequency,
            arrival_time=s_arrival
        )
        waves.append(s_wave)
        
        # Surface waves (if present)
        surface_waves = []
        if surface_wave_amplitude > 0:
            surface_wave = WaveSegment(
                wave_type='Rayleigh',
                start_time=s_arrival + 20.0,
                end_time=s_arrival + 60.0,
                data=np.random.randn(4000),
                sampling_rate=100.0,
                peak_amplitude=surface_wave_amplitude,
                dominant_frequency=0.1,
                arrival_time=s_arrival + 20.0
            )
            surface_waves.append(surface_wave)
        
        # Create wave analysis result
        wave_result = WaveAnalysisResult(
            original_data=np.random.randn(10000),
            sampling_rate=100.0,
            p_waves=[p_wave],
            s_waves=[s_wave],
            surface_waves=surface_waves,
            metadata={'distance_km': distance}
        )
        
        # Create magnitude estimates
        magnitude_estimates = [
            MagnitudeEstimate(
                method='ML',
                magnitude=magnitude,
                confidence=0.85,
                wave_type_used='P'
            ),
            MagnitudeEstimate(
                method='Mb',
                magnitude=magnitude - 0.2,
                confidence=0.80,
                wave_type_used='P'
            )
        ]
        
        # Create arrival times
        arrival_times = ArrivalTimes(
            p_wave_arrival=p_arrival,
            s_wave_arrival=s_arrival,
            sp_time_difference=sp_time,
            surface_wave_arrival=s_arrival + 20.0 if surface_waves else 0.0
        )
        
        # Create frequency analysis
        frequency_analysis = {
            'P': FrequencyData(
                frequencies=np.linspace(0, 50, 100),
                power_spectrum=np.random.randn(100),
                dominant_frequency=p_frequency,
                frequency_range=(1.0, 10.0),
                spectral_centroid=p_frequency + 0.2,
                bandwidth=2.0
            ),
            'S': FrequencyData(
                frequencies=np.linspace(0, 50, 100),
                power_spectrum=np.random.randn(100),
                dominant_frequency=s_frequency,
                frequency_range=(0.5, 5.0),
                spectral_centroid=s_frequency + 0.1,
                bandwidth=1.5
            )
        }
        
        # Create quality metrics
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=15.0,
            detection_confidence=0.90,
            analysis_quality_score=0.90,
            data_completeness=1.0
        )
        
        # Create detailed analysis
        return DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates,
            epicenter_distance=distance,
            frequency_analysis=frequency_analysis,
            quality_metrics=quality_metrics
        )


if __name__ == '__main__':
    unittest.main()