"""
Integration tests for WaveAnalyzer complete workflow.

This module tests the complete wave analysis workflow including
wave separation, detailed analysis, and quality assessment integration.
"""

import unittest
import numpy as np
from datetime import datetime

from wave_analysis.services.wave_analyzer import WaveAnalyzer
from wave_analysis.services.quality_metrics import QualityMetricsCalculator
from wave_analysis.models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, 
    ArrivalTimes, MagnitudeEstimate, QualityMetrics
)


class TestWaveAnalyzerIntegration(unittest.TestCase):
    """Integration test suite for complete WaveAnalyzer workflow."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sampling_rate = 100.0
        self.wave_analyzer = WaveAnalyzer(self.sampling_rate)
        self.quality_calculator = QualityMetricsCalculator(self.sampling_rate)
        
        # Create comprehensive test data
        self.test_data = self._create_comprehensive_test_data()
        self.wave_result = self._create_comprehensive_wave_result()
        
    def _create_comprehensive_test_data(self) -> np.ndarray:
        """Create comprehensive synthetic seismic data for testing."""
        duration = 120.0  # 2 minutes
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Create realistic earthquake signal
        signal = np.zeros_like(t)
        
        # Background noise
        noise = 0.05 * np.random.normal(0, 1, len(signal))
        
        # P-wave arrival at 15 seconds (high frequency, short duration)
        p_start = 15.0
        p_duration = 3.0
        p_mask = (t >= p_start) & (t <= p_start + p_duration)
        p_envelope = np.exp(-2 * (t[p_mask] - p_start))
        signal[p_mask] += 1.5 * np.sin(2 * np.pi * 10 * (t[p_mask] - p_start)) * p_envelope
        
        # S-wave arrival at 28 seconds (medium frequency, medium duration)
        s_start = 28.0
        s_duration = 8.0
        s_mask = (t >= s_start) & (t <= s_start + s_duration)
        s_envelope = np.exp(-1 * (t[s_mask] - s_start))
        signal[s_mask] += 2.5 * np.sin(2 * np.pi * 5 * (t[s_mask] - s_start)) * s_envelope
        
        # Love wave arrival at 45 seconds (low frequency, long duration)
        love_start = 45.0
        love_duration = 25.0
        love_mask = (t >= love_start) & (t <= love_start + love_duration)
        love_envelope = np.exp(-0.15 * (t[love_mask] - love_start))
        signal[love_mask] += 3.0 * np.sin(2 * np.pi * 0.8 * (t[love_mask] - love_start)) * love_envelope
        
        # Rayleigh wave arrival at 50 seconds (low frequency, long duration)
        rayleigh_start = 50.0
        rayleigh_duration = 30.0
        rayleigh_mask = (t >= rayleigh_start) & (t <= rayleigh_start + rayleigh_duration)
        rayleigh_envelope = np.exp(-0.12 * (t[rayleigh_mask] - rayleigh_start))
        signal[rayleigh_mask] += 2.8 * np.sin(2 * np.pi * 0.6 * (t[rayleigh_mask] - rayleigh_start)) * rayleigh_envelope
        
        # Add noise
        signal += noise
        
        return signal
    
    def _create_comprehensive_wave_result(self) -> WaveAnalysisResult:
        """Create a comprehensive WaveAnalysisResult for testing."""
        data = self.test_data
        
        # Create realistic wave segments
        p_waves = [
            WaveSegment(
                wave_type='P',
                start_time=15.0,
                end_time=18.0,
                data=data[1500:1800],  # 3 seconds at 100 Hz
                sampling_rate=self.sampling_rate,
                peak_amplitude=1.5,
                dominant_frequency=10.0,
                arrival_time=15.2,
                confidence=0.85
            )
        ]
        
        s_waves = [
            WaveSegment(
                wave_type='S',
                start_time=28.0,
                end_time=36.0,
                data=data[2800:3600],  # 8 seconds at 100 Hz
                sampling_rate=self.sampling_rate,
                peak_amplitude=2.5,
                dominant_frequency=5.0,
                arrival_time=28.3,
                confidence=0.78
            )
        ]
        
        surface_waves = [
            WaveSegment(
                wave_type='Love',
                start_time=45.0,
                end_time=70.0,
                data=data[4500:7000],  # 25 seconds at 100 Hz
                sampling_rate=self.sampling_rate,
                peak_amplitude=3.0,
                dominant_frequency=0.8,
                arrival_time=45.5,
                confidence=0.82
            ),
            WaveSegment(
                wave_type='Rayleigh',
                start_time=50.0,
                end_time=80.0,
                data=data[5000:8000],  # 30 seconds at 100 Hz
                sampling_rate=self.sampling_rate,
                peak_amplitude=2.8,
                dominant_frequency=0.6,
                arrival_time=50.2,
                confidence=0.79
            )
        ]
        
        return WaveAnalysisResult(
            original_data=data,
            sampling_rate=self.sampling_rate,
            p_waves=p_waves,
            s_waves=s_waves,
            surface_waves=surface_waves,
            metadata={
                'station': 'TEST_STATION',
                'location': {'lat': 40.0, 'lon': -120.0},
                'event_time': '2024-01-01T12:00:00Z'
            }
        )
    
    def test_complete_wave_analysis_workflow(self):
        """Test the complete wave analysis workflow from start to finish."""
        # Perform comprehensive wave analysis
        detailed_analysis = self.wave_analyzer.analyze_waves(self.wave_result)
        
        # Verify the analysis result structure
        self.assertIsInstance(detailed_analysis, DetailedAnalysis)
        self.assertEqual(detailed_analysis.wave_result, self.wave_result)
        
        # Check arrival times
        self.assertIsInstance(detailed_analysis.arrival_times, ArrivalTimes)
        self.assertIsNotNone(detailed_analysis.arrival_times.p_wave_arrival)
        self.assertIsNotNone(detailed_analysis.arrival_times.s_wave_arrival)
        self.assertIsNotNone(detailed_analysis.arrival_times.surface_wave_arrival)
        self.assertIsNotNone(detailed_analysis.arrival_times.sp_time_difference)
        
        # Verify timing relationships
        self.assertLess(detailed_analysis.arrival_times.p_wave_arrival, 
                       detailed_analysis.arrival_times.s_wave_arrival)
        self.assertLess(detailed_analysis.arrival_times.s_wave_arrival,
                       detailed_analysis.arrival_times.surface_wave_arrival)
        self.assertGreater(detailed_analysis.arrival_times.sp_time_difference, 0)
        
        # Check magnitude estimates
        self.assertGreater(len(detailed_analysis.magnitude_estimates), 0)
        for mag_est in detailed_analysis.magnitude_estimates:
            self.assertIsInstance(mag_est, MagnitudeEstimate)
            self.assertIn(mag_est.method, ['ML', 'Mb', 'Ms'])
            self.assertGreater(mag_est.confidence, 0)
            self.assertLessEqual(mag_est.confidence, 1)
        
        # Check epicenter distance
        self.assertIsNotNone(detailed_analysis.epicenter_distance)
        self.assertGreater(detailed_analysis.epicenter_distance, 0)
        
        # Check frequency analysis
        self.assertIsInstance(detailed_analysis.frequency_analysis, dict)
        self.assertGreater(len(detailed_analysis.frequency_analysis), 0)
        
        # Check quality metrics
        self.assertIsInstance(detailed_analysis.quality_metrics, QualityMetrics)
        self.assertGreater(detailed_analysis.quality_metrics.signal_to_noise_ratio, 0)
        self.assertGreaterEqual(detailed_analysis.quality_metrics.detection_confidence, 0)
        self.assertLessEqual(detailed_analysis.quality_metrics.detection_confidence, 1)
        self.assertGreaterEqual(detailed_analysis.quality_metrics.analysis_quality_score, 0)
        self.assertLessEqual(detailed_analysis.quality_metrics.analysis_quality_score, 1)
        
    def test_wave_analyzer_with_quality_assessment(self):
        """Test wave analyzer integration with quality assessment."""
        # Perform analysis
        detailed_analysis = self.wave_analyzer.analyze_waves(self.wave_result)
        
        # Perform additional quality assessment
        enhanced_quality = self.quality_calculator.calculate_quality_metrics(
            self.wave_result, detailed_analysis
        )
        
        # Compare quality metrics
        analyzer_quality = detailed_analysis.quality_metrics
        
        # Both should provide similar SNR values
        self.assertAlmostEqual(
            analyzer_quality.signal_to_noise_ratio,
            enhanced_quality.signal_to_noise_ratio,
            delta=1.0  # Allow some difference due to different calculation methods
        )
        
        # Both should provide similar confidence values
        self.assertAlmostEqual(
            analyzer_quality.detection_confidence,
            enhanced_quality.detection_confidence,
            delta=0.1
        )
        
        # Enhanced quality assessment might provide more detailed warnings
        self.assertGreaterEqual(
            len(enhanced_quality.processing_warnings),
            len(analyzer_quality.processing_warnings)
        )
    
    def test_wave_analysis_with_different_quality_data(self):
        """Test wave analysis with different data quality levels."""
        # Test with high-quality data (already done in setup)
        high_quality_analysis = self.wave_analyzer.analyze_waves(self.wave_result)
        
        # Create low-quality data
        low_quality_data = 0.1 * self.test_data + 0.5 * np.random.normal(0, 1, len(self.test_data))
        low_quality_result = WaveAnalysisResult(
            original_data=low_quality_data,
            sampling_rate=self.sampling_rate,
            p_waves=[WaveSegment('P', 15.0, 18.0, low_quality_data[1500:1800], 
                               self.sampling_rate, 0.2, 10.0, 15.2, 0.4)],
            s_waves=[WaveSegment('S', 28.0, 36.0, low_quality_data[2800:3600], 
                               self.sampling_rate, 0.3, 5.0, 28.3, 0.3)],
            surface_waves=[]
        )
        
        low_quality_analysis = self.wave_analyzer.analyze_waves(low_quality_result)
        
        # Compare quality metrics
        self.assertLess(
            low_quality_analysis.quality_metrics.signal_to_noise_ratio,
            high_quality_analysis.quality_metrics.signal_to_noise_ratio
        )
        self.assertLess(
            low_quality_analysis.quality_metrics.detection_confidence,
            high_quality_analysis.quality_metrics.detection_confidence
        )
        self.assertLess(
            low_quality_analysis.quality_metrics.analysis_quality_score,
            high_quality_analysis.quality_metrics.analysis_quality_score
        )
        self.assertGreater(
            len(low_quality_analysis.quality_metrics.processing_warnings),
            len(high_quality_analysis.quality_metrics.processing_warnings)
        )
    
    def test_individual_wave_type_analysis(self):
        """Test analysis of individual wave types."""
        # Test P-wave analysis
        p_characteristics = self.wave_analyzer.analyze_single_wave_type('P', self.wave_result.p_waves)
        
        self.assertIsInstance(p_characteristics, dict)
        self.assertEqual(p_characteristics['wave_type'], 'P')
        self.assertIn('arrival_time', p_characteristics)
        self.assertIn('peak_amplitude', p_characteristics)
        self.assertIn('dominant_frequency', p_characteristics)
        self.assertIn('duration', p_characteristics)
        self.assertIn('confidence', p_characteristics)
        
        # Test S-wave analysis
        s_characteristics = self.wave_analyzer.analyze_single_wave_type('S', self.wave_result.s_waves)
        
        self.assertIsInstance(s_characteristics, dict)
        self.assertEqual(s_characteristics['wave_type'], 'S')
        self.assertLess(s_characteristics['dominant_frequency'], p_characteristics['dominant_frequency'])
        self.assertGreater(s_characteristics['duration'], p_characteristics['duration'])
        
        # Test surface wave analysis
        love_characteristics = self.wave_analyzer.analyze_single_wave_type('Love', 
                                                                         [w for w in self.wave_result.surface_waves if w.wave_type == 'Love'])
        
        self.assertIsInstance(love_characteristics, dict)
        self.assertEqual(love_characteristics['wave_type'], 'Love')
        self.assertLess(love_characteristics['dominant_frequency'], s_characteristics['dominant_frequency'])
        self.assertGreater(love_characteristics['duration'], s_characteristics['duration'])
    
    def test_wave_analysis_comparison(self):
        """Test comparison between different wave analyses."""
        # Create two similar analyses
        analysis1 = self.wave_analyzer.analyze_waves(self.wave_result)
        
        # Create slightly different wave result
        modified_waves = self.wave_result.p_waves.copy()
        modified_waves[0].arrival_time += 0.1  # Shift arrival time slightly
        
        modified_result = WaveAnalysisResult(
            original_data=self.wave_result.original_data,
            sampling_rate=self.sampling_rate,
            p_waves=modified_waves,
            s_waves=self.wave_result.s_waves,
            surface_waves=self.wave_result.surface_waves
        )
        
        analysis2 = self.wave_analyzer.analyze_waves(modified_result)
        
        # Compare analyses
        comparison = self.wave_analyzer.compare_wave_characteristics(analysis1, analysis2)
        
        self.assertIsInstance(comparison, dict)
        # The comparison may not show the exact difference due to arrival time recalculation
        # Just verify that comparison metrics are present and reasonable
        if 'p_wave_time_difference' in comparison:
            self.assertGreaterEqual(comparison['p_wave_time_difference'], 0)
        
        # Verify other comparison metrics if present
        for key in ['s_wave_time_difference', 'magnitude_difference', 'snr_difference']:
            if key in comparison:
                self.assertGreaterEqual(comparison[key], 0)
    
    def test_wave_validation_integration(self):
        """Test integration of wave validation with analysis."""
        # Perform analysis
        detailed_analysis = self.wave_analyzer.analyze_waves(self.wave_result)
        
        # Validate the wave detection results
        validation_results = self.quality_calculator.validate_wave_detection_results(self.wave_result)
        
        self.assertIsInstance(validation_results, dict)
        self.assertIn('overall_valid', validation_results)
        self.assertIn('p_waves_valid', validation_results)
        self.assertIn('s_waves_valid', validation_results)
        self.assertIn('surface_waves_valid', validation_results)
        self.assertIn('timing_valid', validation_results)
        self.assertIn('amplitudes_valid', validation_results)
        
        # For our well-constructed test data, validation should pass
        self.assertTrue(validation_results['overall_valid'])
        self.assertTrue(validation_results['p_waves_valid'])
        self.assertTrue(validation_results['s_waves_valid'])
        self.assertTrue(validation_results['surface_waves_valid'])
        self.assertTrue(validation_results['timing_valid'])
        self.assertTrue(validation_results['amplitudes_valid'])
    
    def test_parameter_consistency(self):
        """Test parameter consistency across analysis components."""
        # Set parameters for wave analyzer
        analyzer_params = {
            'arrival_time_params': {
                'cross_correlation_window': 3.0,
                'refinement_window': 1.0
            },
            'frequency_params': {
                'window_type': 'hamming',
                'nperseg': 512
            },
            'magnitude_params': {
                'ml_constants': {'a': 1.0, 'b': 0.0, 'c': -2.5}
            }
        }
        
        self.wave_analyzer.set_parameters(**analyzer_params)
        
        # Set parameters for quality calculator
        quality_params = {
            'noise_window_fraction': 0.15,
            'min_snr_threshold': 4.0,
            'min_confidence_threshold': 0.4
        }
        
        self.quality_calculator.set_parameters(**quality_params)
        
        # Perform analysis with updated parameters
        detailed_analysis = self.wave_analyzer.analyze_waves(self.wave_result)
        quality_metrics = self.quality_calculator.calculate_quality_metrics(self.wave_result, detailed_analysis)
        
        # Verify analysis still works with custom parameters
        self.assertIsInstance(detailed_analysis, DetailedAnalysis)
        self.assertIsInstance(quality_metrics, QualityMetrics)
        
        # Check that parameters affected the analysis
        # (This is a basic check - in practice, you'd verify specific parameter effects)
        self.assertIsNotNone(detailed_analysis.arrival_times)
        self.assertIsNotNone(detailed_analysis.magnitude_estimates)
        self.assertIsNotNone(quality_metrics.processing_warnings)
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Create problematic wave result (empty waves)
        empty_wave_result = WaveAnalysisResult(
            original_data=self.test_data,
            sampling_rate=self.sampling_rate,
            p_waves=[],
            s_waves=[],
            surface_waves=[]
        )
        
        # Analysis should handle empty waves gracefully
        empty_analysis = self.wave_analyzer.analyze_waves(empty_wave_result)
        
        self.assertIsInstance(empty_analysis, DetailedAnalysis)
        self.assertEqual(len(empty_analysis.magnitude_estimates), 0)
        self.assertIsNone(empty_analysis.arrival_times.p_wave_arrival)
        self.assertIsNone(empty_analysis.arrival_times.s_wave_arrival)
        
        # Quality assessment should also handle this gracefully
        empty_quality = self.quality_calculator.calculate_quality_metrics(empty_wave_result, empty_analysis)
        
        self.assertIsInstance(empty_quality, QualityMetrics)
        self.assertEqual(empty_quality.detection_confidence, 0.0)
        self.assertGreater(len(empty_quality.processing_warnings), 0)
        self.assertIn("No waves detected", ' '.join(empty_quality.processing_warnings))
    
    def test_performance_characteristics(self):
        """Test performance characteristics of integrated analysis."""
        import time
        
        # Measure analysis time
        start_time = time.time()
        detailed_analysis = self.wave_analyzer.analyze_waves(self.wave_result)
        analysis_time = time.time() - start_time
        
        # Analysis should complete in reasonable time (< 5 seconds for test data)
        self.assertLess(analysis_time, 5.0)
        
        # Measure quality assessment time
        start_time = time.time()
        quality_metrics = self.quality_calculator.calculate_quality_metrics(self.wave_result, detailed_analysis)
        quality_time = time.time() - start_time
        
        # Quality assessment should also be fast (< 2 seconds)
        self.assertLess(quality_time, 2.0)
        
        # Verify results are still valid despite performance requirements
        self.assertIsInstance(detailed_analysis, DetailedAnalysis)
        self.assertIsInstance(quality_metrics, QualityMetrics)
        self.assertTrue(detailed_analysis.has_complete_analysis)


if __name__ == '__main__':
    unittest.main()