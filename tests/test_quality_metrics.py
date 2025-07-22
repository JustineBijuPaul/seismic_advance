"""
Unit tests for QualityMetricsCalculator.

This module tests the quality assessment functionality for wave analysis results,
including signal-to-noise ratio calculations, confidence scoring, and validation
of wave detection results with various data quality scenarios.
"""

import unittest
import numpy as np
from datetime import datetime

from wave_analysis.services.quality_metrics import QualityMetricsCalculator
from wave_analysis.models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, 
    ArrivalTimes, MagnitudeEstimate, QualityMetrics
)


class TestQualityMetricsCalculator(unittest.TestCase):
    """Test suite for QualityMetricsCalculator."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sampling_rate = 100.0
        self.quality_calculator = QualityMetricsCalculator(self.sampling_rate)
        
        # Create sample data with different quality levels
        self.high_quality_data = self._create_high_quality_data()
        self.low_quality_data = self._create_low_quality_data()
        self.noisy_data = self._create_noisy_data()
        self.clipped_data = self._create_clipped_data()
        
    def _create_high_quality_data(self) -> np.ndarray:
        """Create high-quality synthetic seismic data."""
        duration = 60.0  # 60 seconds
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Clean signal with clear P, S, and surface waves
        signal = np.zeros_like(t)
        
        # P-wave at 10 seconds
        p_mask = (t >= 10) & (t <= 12)
        signal[p_mask] += 2.0 * np.sin(2 * np.pi * 8 * (t[p_mask] - 10)) * np.exp(-2 * (t[p_mask] - 10))
        
        # S-wave at 18 seconds
        s_mask = (t >= 18) & (t <= 23)
        signal[s_mask] += 3.0 * np.sin(2 * np.pi * 4 * (t[s_mask] - 18)) * np.exp(-1 * (t[s_mask] - 18))
        
        # Surface wave at 25 seconds
        surf_mask = (t >= 25) & (t <= 40)
        signal[surf_mask] += 4.0 * np.sin(2 * np.pi * 1 * (t[surf_mask] - 25)) * np.exp(-0.2 * (t[surf_mask] - 25))
        
        # Add minimal noise
        noise = 0.1 * np.random.normal(0, 1, len(signal))
        signal += noise
        
        return signal
    
    def _create_low_quality_data(self) -> np.ndarray:
        """Create low-quality synthetic seismic data."""
        duration = 60.0
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Weak signal
        signal = np.zeros_like(t)
        
        # Weak P-wave
        p_mask = (t >= 10) & (t <= 12)
        signal[p_mask] += 0.2 * np.sin(2 * np.pi * 8 * (t[p_mask] - 10))
        
        # Add significant noise
        noise = 0.5 * np.random.normal(0, 1, len(signal))
        signal += noise
        
        return signal
    
    def _create_noisy_data(self) -> np.ndarray:
        """Create very noisy synthetic seismic data."""
        duration = 60.0
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Mostly noise with very weak signal
        signal = 0.1 * np.sin(2 * np.pi * 5 * t)
        noise = 2.0 * np.random.normal(0, 1, len(signal))
        
        return signal + noise
    
    def _create_clipped_data(self) -> np.ndarray:
        """Create clipped/saturated synthetic seismic data."""
        duration = 60.0
        t = np.linspace(0, duration, int(duration * self.sampling_rate))
        
        # Signal that gets clipped
        signal = 5.0 * np.sin(2 * np.pi * 2 * t)
        
        # Clip the signal
        signal = np.clip(signal, -3.0, 3.0)
        
        return signal
    
    def _create_sample_wave_result(self, data: np.ndarray, quality_level: str = 'high') -> WaveAnalysisResult:
        """Create a sample WaveAnalysisResult for testing."""
        if quality_level == 'high':
            # High-quality detections
            p_waves = [
                WaveSegment('P', 10.0, 12.0, data[1000:1200], self.sampling_rate, 2.0, 8.0, 10.5, 0.9)
            ]
            s_waves = [
                WaveSegment('S', 18.0, 23.0, data[1800:2300], self.sampling_rate, 3.0, 4.0, 18.5, 0.8)
            ]
            surface_waves = [
                WaveSegment('Love', 25.0, 40.0, data[2500:4000], self.sampling_rate, 4.0, 1.0, 25.5, 0.85)
            ]
        elif quality_level == 'medium':
            # Medium-quality detections
            p_waves = [
                WaveSegment('P', 10.0, 12.0, data[1000:1200], self.sampling_rate, 1.0, 8.0, 10.5, 0.6)
            ]
            s_waves = [
                WaveSegment('S', 18.0, 23.0, data[1800:2300], self.sampling_rate, 1.5, 4.0, 18.5, 0.5)
            ]
            surface_waves = []
        else:  # low quality
            # Low-quality detections
            p_waves = [
                WaveSegment('P', 10.0, 12.0, data[1000:1200], self.sampling_rate, 0.3, 8.0, 10.5, 0.3)
            ]
            s_waves = []
            surface_waves = []
        
        return WaveAnalysisResult(
            original_data=data,
            sampling_rate=self.sampling_rate,
            p_waves=p_waves,
            s_waves=s_waves,
            surface_waves=surface_waves
        )
    
    def test_calculator_initialization(self):
        """Test proper initialization of QualityMetricsCalculator."""
        calculator = QualityMetricsCalculator(self.sampling_rate)
        
        self.assertEqual(calculator.sampling_rate, self.sampling_rate)
        self.assertEqual(calculator.noise_window_fraction, 0.1)
        self.assertEqual(calculator.min_snr_threshold, 3.0)
        self.assertEqual(calculator.min_confidence_threshold, 0.3)
        self.assertIsInstance(calculator.quality_thresholds, dict)
    
    def test_signal_to_noise_ratio_calculation(self):
        """Test SNR calculation with different data quality levels."""
        # High-quality data should have high SNR
        high_snr = self.quality_calculator._calculate_signal_to_noise_ratio(self.high_quality_data)
        self.assertGreater(high_snr, 10.0)
        
        # Low-quality data should have lower SNR
        low_snr = self.quality_calculator._calculate_signal_to_noise_ratio(self.low_quality_data)
        self.assertLess(low_snr, high_snr)
        
        # Very noisy data should have very low SNR
        noisy_snr = self.quality_calculator._calculate_signal_to_noise_ratio(self.noisy_data)
        self.assertLess(noisy_snr, 5.0)
        
        # Test edge case: constant data (should have 0 SNR since signal power = noise power)
        constant_data = np.ones(1000)
        constant_snr = self.quality_calculator._calculate_signal_to_noise_ratio(constant_data)
        self.assertEqual(constant_snr, 0.0)  # Constant data has no signal above noise
    
    def test_detection_confidence_calculation(self):
        """Test detection confidence calculation."""
        # High-quality wave result
        high_quality_result = self._create_sample_wave_result(self.high_quality_data, 'high')
        high_confidence = self.quality_calculator._calculate_detection_confidence(high_quality_result)
        self.assertGreater(high_confidence, 0.7)
        
        # Medium-quality wave result
        medium_quality_result = self._create_sample_wave_result(self.high_quality_data, 'medium')
        medium_confidence = self.quality_calculator._calculate_detection_confidence(medium_quality_result)
        self.assertLess(medium_confidence, high_confidence)
        
        # Low-quality wave result
        low_quality_result = self._create_sample_wave_result(self.low_quality_data, 'low')
        low_confidence = self.quality_calculator._calculate_detection_confidence(low_quality_result)
        self.assertLess(low_confidence, medium_confidence)
        
        # No waves detected
        no_waves_result = WaveAnalysisResult(
            original_data=self.noisy_data,
            sampling_rate=self.sampling_rate
        )
        no_confidence = self.quality_calculator._calculate_detection_confidence(no_waves_result)
        self.assertEqual(no_confidence, 0.0)
    
    def test_data_completeness_calculation(self):
        """Test data completeness assessment."""
        # Clean data should have high completeness
        clean_result = self._create_sample_wave_result(self.high_quality_data)
        clean_completeness = self.quality_calculator._calculate_data_completeness(clean_result)
        self.assertGreater(clean_completeness, 0.9)
        
        # Clipped data should have lower completeness
        clipped_result = self._create_sample_wave_result(self.clipped_data)
        clipped_completeness = self.quality_calculator._calculate_data_completeness(clipped_result)
        self.assertLess(clipped_completeness, clean_completeness)
        
        # Data with NaN values
        nan_data = self.high_quality_data.copy()
        nan_data[100:200] = np.nan
        nan_result = self._create_sample_wave_result(nan_data)
        nan_completeness = self.quality_calculator._calculate_data_completeness(nan_result)
        self.assertLess(nan_completeness, clean_completeness)
        
        # Constant data should have very low completeness
        constant_data = np.ones(6000)
        constant_result = self._create_sample_wave_result(constant_data)
        constant_completeness = self.quality_calculator._calculate_data_completeness(constant_result)
        self.assertLess(constant_completeness, 0.5)
    
    def test_snr_normalization(self):
        """Test SNR to quality score normalization."""
        # Test different SNR levels
        excellent_quality = self.quality_calculator._normalize_snr_to_quality(25.0)
        self.assertEqual(excellent_quality, 1.0)
        
        good_quality = self.quality_calculator._normalize_snr_to_quality(15.0)
        self.assertEqual(good_quality, 0.8)
        
        fair_quality = self.quality_calculator._normalize_snr_to_quality(7.0)
        self.assertEqual(fair_quality, 0.6)
        
        poor_quality = self.quality_calculator._normalize_snr_to_quality(2.0)
        self.assertEqual(poor_quality, 0.4)
        
        very_poor_quality = self.quality_calculator._normalize_snr_to_quality(-5.0)
        self.assertEqual(very_poor_quality, 0.2)
    
    def test_timing_consistency_assessment(self):
        """Test timing consistency assessment."""
        # Good timing (P < S < Surface)
        good_timing = ArrivalTimes(
            p_wave_arrival=10.0,
            s_wave_arrival=18.0,
            surface_wave_arrival=25.0
        )
        good_score = self.quality_calculator._assess_timing_consistency(good_timing)
        self.assertEqual(good_score, 1.0)
        
        # Bad timing (S before P)
        bad_timing = ArrivalTimes(
            p_wave_arrival=18.0,
            s_wave_arrival=10.0,
            surface_wave_arrival=25.0
        )
        bad_score = self.quality_calculator._assess_timing_consistency(bad_timing)
        self.assertLess(bad_score, 1.0)
        
        # Surface waves before S-waves
        surface_early = ArrivalTimes(
            p_wave_arrival=10.0,
            s_wave_arrival=25.0,
            surface_wave_arrival=18.0
        )
        surface_score = self.quality_calculator._assess_timing_consistency(surface_early)
        self.assertLess(surface_score, 1.0)
        
        # Unrealistic S-P time
        unrealistic_timing = ArrivalTimes(
            p_wave_arrival=10.0,
            s_wave_arrival=10.05  # Only 0.05 seconds difference
        )
        unrealistic_score = self.quality_calculator._assess_timing_consistency(unrealistic_timing)
        self.assertLess(unrealistic_score, 1.0)
    
    def test_magnitude_consistency_assessment(self):
        """Test magnitude consistency assessment."""
        # Consistent magnitude estimates
        consistent_estimates = [
            MagnitudeEstimate('ML', 4.2, 0.8, 'P'),
            MagnitudeEstimate('Mb', 4.1, 0.7, 'P'),
            MagnitudeEstimate('Ms', 4.3, 0.6, 'Love')
        ]
        consistent_score = self.quality_calculator._assess_magnitude_consistency(consistent_estimates)
        self.assertEqual(consistent_score, 1.0)
        
        # Inconsistent magnitude estimates
        inconsistent_estimates = [
            MagnitudeEstimate('ML', 3.0, 0.8, 'P'),
            MagnitudeEstimate('Mb', 5.5, 0.7, 'P'),
            MagnitudeEstimate('Ms', 4.2, 0.6, 'Love')
        ]
        inconsistent_score = self.quality_calculator._assess_magnitude_consistency(inconsistent_estimates)
        self.assertLess(inconsistent_score, 1.0)
        
        # Single estimate (should return 1.0)
        single_estimate = [MagnitudeEstimate('ML', 4.0, 0.8, 'P')]
        single_score = self.quality_calculator._assess_magnitude_consistency(single_estimate)
        self.assertEqual(single_score, 1.0)
    
    def test_quality_warnings_generation(self):
        """Test generation of quality warnings."""
        # High-quality data should generate few warnings
        high_quality_result = self._create_sample_wave_result(self.high_quality_data, 'high')
        high_snr = 15.0
        high_confidence = 0.8
        high_completeness = 0.95
        
        high_warnings = self.quality_calculator._generate_quality_warnings(
            high_quality_result, high_snr, high_confidence, high_completeness
        )
        self.assertLessEqual(len(high_warnings), 1)
        
        # Low-quality data should generate multiple warnings
        low_quality_result = self._create_sample_wave_result(self.low_quality_data, 'low')
        low_snr = 1.0
        low_confidence = 0.2
        low_completeness = 0.7
        
        low_warnings = self.quality_calculator._generate_quality_warnings(
            low_quality_result, low_snr, low_confidence, low_completeness
        )
        self.assertGreater(len(low_warnings), 2)
        
        # Check specific warning types
        warning_text = ' '.join(low_warnings)
        self.assertIn('Low signal-to-noise ratio', warning_text)
        self.assertIn('Low average detection confidence', warning_text)
        self.assertIn('Data completeness issues', warning_text)
    
    def test_wave_detection_validation(self):
        """Test validation of wave detection results."""
        # Valid wave result
        valid_result = self._create_sample_wave_result(self.high_quality_data, 'high')
        validation = self.quality_calculator.validate_wave_detection_results(valid_result)
        
        self.assertTrue(validation['p_waves_valid'])
        self.assertTrue(validation['s_waves_valid'])
        self.assertTrue(validation['surface_waves_valid'])
        self.assertTrue(validation['timing_valid'])
        self.assertTrue(validation['amplitudes_valid'])
        self.assertTrue(validation['overall_valid'])
        
        # Invalid wave result (bad timing)
        invalid_waves = [
            WaveSegment('P', 18.0, 20.0, np.random.normal(0, 1, 200), 100.0, 1.0, 8.0, 18.5, 0.8),  # P after S
            WaveSegment('S', 10.0, 15.0, np.random.normal(0, 1, 500), 100.0, 1.5, 4.0, 10.5, 0.7)   # S before P
        ]
        
        invalid_result = WaveAnalysisResult(
            original_data=self.high_quality_data,
            sampling_rate=self.sampling_rate,
            p_waves=[invalid_waves[0]],
            s_waves=[invalid_waves[1]]
        )
        
        invalid_validation = self.quality_calculator.validate_wave_detection_results(invalid_result)
        self.assertFalse(invalid_validation['timing_valid'])
        self.assertFalse(invalid_validation['overall_valid'])
    
    def test_p_wave_validation(self):
        """Test P-wave specific validation."""
        # Valid P-waves
        valid_p_waves = [
            WaveSegment('P', 10.0, 12.0, np.random.normal(0, 1, 200), 100.0, 1.0, 8.0, 10.5, 0.8)
        ]
        self.assertTrue(self.quality_calculator._validate_p_waves(valid_p_waves))
        
        # Invalid P-waves (wrong frequency)
        invalid_freq_p = [
            WaveSegment('P', 10.0, 12.0, np.random.normal(0, 1, 200), 100.0, 1.0, 25.0, 10.5, 0.8)  # Too high freq
        ]
        self.assertFalse(self.quality_calculator._validate_p_waves(invalid_freq_p))
        
        # Invalid P-waves (too long duration)
        invalid_duration_p = [
            WaveSegment('P', 10.0, 30.0, np.random.normal(0, 1, 2000), 100.0, 1.0, 8.0, 10.5, 0.8)  # 20 seconds
        ]
        self.assertFalse(self.quality_calculator._validate_p_waves(invalid_duration_p))
        
        # Invalid P-waves (low confidence)
        invalid_conf_p = [
            WaveSegment('P', 10.0, 12.0, np.random.normal(0, 1, 200), 100.0, 1.0, 8.0, 10.5, 0.05)  # Very low confidence
        ]
        self.assertFalse(self.quality_calculator._validate_p_waves(invalid_conf_p))
    
    def test_s_wave_validation(self):
        """Test S-wave specific validation."""
        # Valid S-waves
        valid_s_waves = [
            WaveSegment('S', 18.0, 23.0, np.random.normal(0, 1, 500), 100.0, 1.5, 4.0, 18.5, 0.7)
        ]
        self.assertTrue(self.quality_calculator._validate_s_waves(valid_s_waves))
        
        # Invalid S-waves (wrong frequency)
        invalid_freq_s = [
            WaveSegment('S', 18.0, 23.0, np.random.normal(0, 1, 500), 100.0, 1.5, 20.0, 18.5, 0.7)  # Too high freq
        ]
        self.assertFalse(self.quality_calculator._validate_s_waves(invalid_freq_s))
        
        # Invalid S-waves (too long duration)
        invalid_duration_s = [
            WaveSegment('S', 18.0, 50.0, np.random.normal(0, 1, 3200), 100.0, 1.5, 4.0, 18.5, 0.7)  # 32 seconds
        ]
        self.assertFalse(self.quality_calculator._validate_s_waves(invalid_duration_s))
    
    def test_surface_wave_validation(self):
        """Test surface wave specific validation."""
        # Valid surface waves
        valid_surface_waves = [
            WaveSegment('Love', 25.0, 40.0, np.random.normal(0, 1, 1500), 100.0, 2.0, 0.2, 25.5, 0.8),
            WaveSegment('Rayleigh', 26.0, 45.0, np.random.normal(0, 1, 1900), 100.0, 1.8, 0.15, 26.5, 0.75)
        ]
        self.assertTrue(self.quality_calculator._validate_surface_waves(valid_surface_waves))
        
        # Invalid surface waves (wrong frequency)
        invalid_freq_surf = [
            WaveSegment('Love', 25.0, 40.0, np.random.normal(0, 1, 1500), 100.0, 2.0, 5.0, 25.5, 0.8)  # Too high freq
        ]
        self.assertFalse(self.quality_calculator._validate_surface_waves(invalid_freq_surf))
        
        # Invalid surface waves (too short duration)
        invalid_duration_surf = [
            WaveSegment('Love', 25.0, 27.0, np.random.normal(0, 1, 200), 100.0, 2.0, 0.2, 25.5, 0.8)  # 2 seconds
        ]
        self.assertFalse(self.quality_calculator._validate_surface_waves(invalid_duration_surf))
        
        # Invalid surface waves (wrong type) - test the validation logic directly
        # since WaveSegment constructor validates wave types
        # We'll test this by creating a mock wave with invalid type after construction
        valid_wave = WaveSegment('Love', 25.0, 40.0, np.random.normal(0, 1, 1500), 100.0, 2.0, 0.2, 25.5, 0.8)
        # Manually change the wave type to test validation
        valid_wave.wave_type = 'Unknown'
        invalid_type_surf = [valid_wave]
        self.assertFalse(self.quality_calculator._validate_surface_waves(invalid_type_surf))
    
    def test_comprehensive_quality_metrics_calculation(self):
        """Test comprehensive quality metrics calculation."""
        # High-quality scenario
        high_quality_result = self._create_sample_wave_result(self.high_quality_data, 'high')
        
        # Create detailed analysis
        arrival_times = ArrivalTimes(
            p_wave_arrival=10.5,
            s_wave_arrival=18.5,
            surface_wave_arrival=25.5
        )
        
        magnitude_estimates = [
            MagnitudeEstimate('ML', 4.2, 0.8, 'P'),
            MagnitudeEstimate('Ms', 4.1, 0.7, 'Love')
        ]
        
        detailed_analysis = DetailedAnalysis(
            wave_result=high_quality_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates
        )
        
        quality_metrics = self.quality_calculator.calculate_quality_metrics(
            high_quality_result, detailed_analysis
        )
        
        self.assertIsInstance(quality_metrics, QualityMetrics)
        self.assertGreater(quality_metrics.signal_to_noise_ratio, 10.0)
        self.assertGreater(quality_metrics.detection_confidence, 0.7)
        self.assertGreater(quality_metrics.analysis_quality_score, 0.7)
        self.assertGreater(quality_metrics.data_completeness, 0.9)
        self.assertIsInstance(quality_metrics.processing_warnings, list)
        
        # Low-quality scenario
        low_quality_result = self._create_sample_wave_result(self.low_quality_data, 'low')
        low_quality_metrics = self.quality_calculator.calculate_quality_metrics(low_quality_result)
        
        self.assertLess(low_quality_metrics.signal_to_noise_ratio, quality_metrics.signal_to_noise_ratio)
        self.assertLess(low_quality_metrics.detection_confidence, quality_metrics.detection_confidence)
        self.assertLess(low_quality_metrics.analysis_quality_score, quality_metrics.analysis_quality_score)
        self.assertGreater(len(low_quality_metrics.processing_warnings), len(quality_metrics.processing_warnings))
    
    def test_data_quality_assessment(self):
        """Test raw data quality assessment."""
        # High-quality data
        high_quality_assessment = self.quality_calculator.assess_data_quality_for_analysis(self.high_quality_data)
        
        self.assertIn('mean', high_quality_assessment)
        self.assertIn('std', high_quality_assessment)
        self.assertIn('finite_fraction', high_quality_assessment)
        self.assertIn('dynamic_range', high_quality_assessment)
        self.assertIn('variability', high_quality_assessment)
        self.assertIn('clipping_fraction', high_quality_assessment)
        
        self.assertEqual(high_quality_assessment['finite_fraction'], 1.0)
        self.assertLess(high_quality_assessment['clipping_fraction'], 0.1)
        
        # Clipped data should have higher clipping fraction
        clipped_assessment = self.quality_calculator.assess_data_quality_for_analysis(self.clipped_data)
        # The clipped data should show signs of clipping in the assessment
        self.assertGreaterEqual(clipped_assessment['clipping_fraction'], high_quality_assessment['clipping_fraction'])
    
    def test_parameter_setting(self):
        """Test parameter setting functionality."""
        # Set new parameters
        new_params = {
            'noise_window_fraction': 0.2,
            'min_snr_threshold': 5.0,
            'min_confidence_threshold': 0.5,
            'quality_thresholds': {'excellent': 0.9}
        }
        
        self.quality_calculator.set_parameters(**new_params)
        
        self.assertEqual(self.quality_calculator.noise_window_fraction, 0.2)
        self.assertEqual(self.quality_calculator.min_snr_threshold, 5.0)
        self.assertEqual(self.quality_calculator.min_confidence_threshold, 0.5)
        self.assertEqual(self.quality_calculator.quality_thresholds['excellent'], 0.9)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Test with empty data - WaveAnalysisResult validation will prevent this
        # so we test the quality calculator methods directly
        
        # Test SNR calculation with empty data
        try:
            empty_snr = self.quality_calculator._calculate_signal_to_noise_ratio(np.array([]))
        except (ValueError, ZeroDivisionError, IndexError):
            # Expected to fail with empty data
            pass
        
        # Single sample data - test if WaveAnalysisResult allows it
        try:
            single_sample = np.array([1.0])
            single_result = WaveAnalysisResult(
                original_data=single_sample,
                sampling_rate=self.sampling_rate
            )
            single_quality = self.quality_calculator.calculate_quality_metrics(single_result)
            self.assertIsInstance(single_quality, QualityMetrics)
        except ValueError:
            # WaveAnalysisResult may reject single sample data
            pass
        
        # All-zero data (should be valid but low quality)
        zero_data = np.zeros(1000)
        zero_result = WaveAnalysisResult(
            original_data=zero_data,
            sampling_rate=self.sampling_rate
        )
        
        zero_quality = self.quality_calculator.calculate_quality_metrics(zero_result)
        self.assertIsInstance(zero_quality, QualityMetrics)
        self.assertLessEqual(zero_quality.analysis_quality_score, 0.5)
        
        # Test with very short data
        short_data = np.random.normal(0, 1, 10)  # Only 10 samples
        try:
            short_result = WaveAnalysisResult(
                original_data=short_data,
                sampling_rate=self.sampling_rate
            )
            short_quality = self.quality_calculator.calculate_quality_metrics(short_result)
            self.assertIsInstance(short_quality, QualityMetrics)
        except ValueError:
            # May be rejected as too short
            pass


if __name__ == '__main__':
    unittest.main()