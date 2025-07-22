"""
Integration tests for WaveSeparationEngine.

This module tests the complete wave separation workflow including
P-wave, S-wave, and surface wave detection integration.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from wave_analysis.services.wave_separation_engine import (
    WaveSeparationEngine, 
    WaveSeparationParameters,
    WaveSeparationResult
)
from wave_analysis.services.wave_detectors import (
    PWaveDetectionParameters,
    SWaveDetectionParameters, 
    SurfaceWaveDetectionParameters
)
from wave_analysis.models import WaveSegment, WaveAnalysisResult, QualityMetrics


class TestWaveSeparationEngine(unittest.TestCase):
    """Test suite for WaveSeparationEngine integration."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_seismic_data = self._create_sample_seismic_data()
        self.wave_separation_params = self._create_wave_separation_params()
        self.wave_separation_engine = WaveSeparationEngine(self.wave_separation_params)
    
    def _create_sample_seismic_data(self):
        """Generate synthetic seismic data for testing."""
        sampling_rate = 100.0
        duration = 60.0  # 60 seconds
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Create synthetic earthquake signal with P, S, and surface waves
        signal = np.zeros_like(t)
        
        # Add P-wave arrival at 10 seconds (high frequency, short duration)
        p_start = 10.0
        p_mask = (t >= p_start) & (t <= p_start + 2.0)
        signal[p_mask] += 0.5 * np.sin(2 * np.pi * 8 * (t[p_mask] - p_start)) * np.exp(-2 * (t[p_mask] - p_start))
        
        # Add S-wave arrival at 18 seconds (medium frequency, medium duration)
        s_start = 18.0
        s_mask = (t >= s_start) & (t <= s_start + 5.0)
        signal[s_mask] += 0.8 * np.sin(2 * np.pi * 4 * (t[s_mask] - s_start)) * np.exp(-1 * (t[s_mask] - s_start))
        
        # Add surface waves at 25 seconds (low frequency, long duration)
        surf_start = 25.0
        surf_mask = (t >= surf_start) & (t <= surf_start + 15.0)
        signal[surf_mask] += 1.0 * np.sin(2 * np.pi * 1 * (t[surf_mask] - surf_start)) * np.exp(-0.2 * (t[surf_mask] - surf_start))
        
        # Add some noise
        noise = 0.1 * np.random.normal(0, 1, len(signal))
        signal += noise
        
        return signal, sampling_rate
    
    def _create_wave_separation_params(self):
        """Create wave separation parameters for testing."""
        sampling_rate = 100.0
        
        p_params = PWaveDetectionParameters(
            sampling_rate=sampling_rate,
            sta_window=1.0,
            lta_window=10.0,
            trigger_threshold=2.0,
            confidence_threshold=0.3
        )
        
        s_params = SWaveDetectionParameters(
            sampling_rate=sampling_rate,
            sta_window=2.0,
            lta_window=15.0,
            trigger_threshold=2.0,
            confidence_threshold=0.3
        )
        
        surf_params = SurfaceWaveDetectionParameters(
            sampling_rate=sampling_rate,
            min_wave_duration=5.0,
            confidence_threshold=0.3
        )
        
        return WaveSeparationParameters(
            sampling_rate=sampling_rate,
            p_wave_params=p_params,
            s_wave_params=s_params,
            surface_wave_params=surf_params,
            min_snr=1.0,
            min_detection_confidence=0.2
        )
    
    def test_engine_initialization(self):
        """Test proper initialization of WaveSeparationEngine."""
        engine = WaveSeparationEngine(self.wave_separation_params)
        
        self.assertEqual(engine.params, self.wave_separation_params)
        self.assertIsNotNone(engine.p_wave_detector)
        self.assertIsNotNone(engine.s_wave_detector)
        self.assertIsNotNone(engine.surface_wave_detector)
        self.assertIsNotNone(engine.signal_processor)
        self.assertEqual(engine.processing_stats['total_analyses'], 0)
    
    def test_separate_waves_complete_workflow(self):
        """Test complete wave separation workflow."""
        seismic_data, sampling_rate = self.sample_seismic_data
        
        # Run wave separation
        result = self.wave_separation_engine.separate_waves(seismic_data)
        
        # Verify result structure
        self.assertIsInstance(result, WaveSeparationResult)
        self.assertIsInstance(result.wave_analysis_result, WaveAnalysisResult)
        self.assertIsInstance(result.quality_metrics, QualityMetrics)
        self.assertIsInstance(result.processing_metadata, dict)
        self.assertIsInstance(result.warnings, list)
        self.assertIsInstance(result.errors, list)
        
        # Verify wave analysis result
        wave_result = result.wave_analysis_result
        self.assertTrue(np.array_equal(wave_result.original_data, seismic_data))
        self.assertEqual(wave_result.sampling_rate, sampling_rate)
        self.assertIsInstance(wave_result.p_waves, list)
        self.assertIsInstance(wave_result.s_waves, list)
        self.assertIsInstance(wave_result.surface_waves, list)
        
        # Verify processing metadata
        metadata = result.processing_metadata
        self.assertIn('start_time', metadata)
        self.assertIn('end_time', metadata)
        self.assertIn('processing_time_seconds', metadata)
        self.assertIn('data_length', metadata)
        self.assertEqual(metadata['data_length'], len(seismic_data))
        self.assertTrue(metadata['success'])
    
    def test_separate_waves_with_metadata(self):
        """Test wave separation with additional metadata."""
        seismic_data, _ = self.sample_seismic_data
        metadata = {
            'station': 'TEST',
            'location': {'lat': 40.0, 'lon': -120.0},
            'event_time': '2024-01-01T00:00:00Z'
        }
        
        result = self.wave_separation_engine.separate_waves(seismic_data, metadata)
        
        self.assertEqual(result.wave_analysis_result.metadata, metadata)
        self.assertTrue(result.processing_metadata['success'])
    
    def test_individual_wave_detection_methods(self):
        """Test individual wave detection methods."""
        seismic_data, _ = self.sample_seismic_data
        
        # Test P-wave detection
        p_waves = self.wave_separation_engine.detect_p_waves(seismic_data)
        self.assertIsInstance(p_waves, list)
        for wave in p_waves:
            self.assertIsInstance(wave, WaveSegment)
            self.assertEqual(wave.wave_type, 'P')
        
        # Test S-wave detection
        s_waves = self.wave_separation_engine.detect_s_waves(seismic_data, p_waves)
        self.assertIsInstance(s_waves, list)
        for wave in s_waves:
            self.assertIsInstance(wave, WaveSegment)
            self.assertEqual(wave.wave_type, 'S')
        
        # Test surface wave detection
        surface_waves = self.wave_separation_engine.detect_surface_waves(seismic_data)
        self.assertIsInstance(surface_waves, list)
        for wave in surface_waves:
            self.assertIsInstance(wave, WaveSegment)
            self.assertIn(wave.wave_type, ['Love', 'Rayleigh'])
    
    def test_input_validation(self):
        """Test input data validation."""
        # Test empty data
        with self.assertRaises(ValueError):
            self.wave_separation_engine.separate_waves(np.array([]))
        
        # Test data with non-finite values
        bad_data = np.array([1.0, 2.0, np.inf, 4.0])
        with self.assertRaises(ValueError):
            self.wave_separation_engine.separate_waves(bad_data)
        
        # Test data that's too short
        short_data = np.random.normal(0, 1, 100)  # Less than 10 seconds at 100 Hz
        with self.assertRaises(ValueError):
            self.wave_separation_engine.separate_waves(short_data)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment functionality."""
        seismic_data, _ = self.sample_seismic_data
        
        # Access private method for testing
        processed_data = self.wave_separation_engine._preprocess_data(seismic_data)
        quality = self.wave_separation_engine._assess_data_quality(processed_data)
        
        self.assertIsInstance(quality, dict)
        self.assertIn('snr', quality)
        self.assertIn('completeness', quality)
        self.assertIn('dynamic_range', quality)
        self.assertIn('max_amplitude', quality)
        self.assertIn('rms_amplitude', quality)
        
        self.assertGreater(quality['snr'], 0)
        self.assertGreaterEqual(quality['completeness'], 0)
        self.assertLessEqual(quality['completeness'], 1)
        self.assertGreater(quality['dynamic_range'], 0)
    
    def test_wave_validation_and_overlap_removal(self):
        """Test wave validation and overlap removal functionality."""
        # Create mock wave segments with overlaps
        p_waves = [
            WaveSegment('P', 10.0, 12.0, np.random.normal(0, 1, 200), 100.0, 1.0, 5.0, 10.5, 0.8),
            WaveSegment('P', 11.0, 13.0, np.random.normal(0, 1, 200), 100.0, 0.8, 5.0, 11.5, 0.6)
        ]
        
        s_waves = [
            WaveSegment('S', 18.0, 22.0, np.random.normal(0, 1, 400), 100.0, 1.2, 3.0, 18.5, 0.7),
            WaveSegment('S', 20.0, 24.0, np.random.normal(0, 1, 400), 100.0, 1.0, 3.0, 20.5, 0.5)
        ]
        
        surface_waves = [
            WaveSegment('Love', 25.0, 35.0, np.random.normal(0, 1, 1000), 100.0, 1.5, 1.0, 25.5, 0.9)
        ]
        
        # Test validation (should filter low confidence waves)
        validated = self.wave_separation_engine._validate_wave_separation(p_waves, s_waves, surface_waves)
        val_p, val_s, val_surf = validated
        
        # Should keep waves with confidence >= min_detection_confidence (0.2)
        self.assertLessEqual(len(val_p), len(p_waves))
        self.assertLessEqual(len(val_s), len(s_waves))
        self.assertLessEqual(len(val_surf), len(surface_waves))
        
        # Test overlap removal
        filtered = self.wave_separation_engine._remove_overlapping_detections(p_waves, s_waves, surface_waves)
        filt_p, filt_s, filt_surf = filtered
        
        # Should have fewer or equal waves after overlap removal
        self.assertLessEqual(len(filt_p), len(p_waves))
        self.assertLessEqual(len(filt_s), len(s_waves))
        self.assertLessEqual(len(filt_surf), len(surface_waves))
    
    def test_overlap_calculation(self):
        """Test overlap calculation between wave segments."""
        wave1 = WaveSegment('P', 10.0, 15.0, np.array([1, 2, 3]), 100.0, 1.0, 5.0, 10.5, 0.8)
        wave2 = WaveSegment('S', 12.0, 17.0, np.array([1, 2, 3]), 100.0, 1.0, 3.0, 12.5, 0.7)
        wave3 = WaveSegment('Love', 20.0, 25.0, np.array([1, 2, 3]), 100.0, 1.0, 1.0, 20.5, 0.9)
        
        # Test overlapping waves
        overlap1 = self.wave_separation_engine._calculate_overlap(wave1, wave2)
        self.assertGreater(overlap1, 0)
        self.assertLessEqual(overlap1, 1)
        
        # Test non-overlapping waves
        overlap2 = self.wave_separation_engine._calculate_overlap(wave1, wave3)
        self.assertEqual(overlap2, 0.0)
        
        # Test identical waves
        overlap3 = self.wave_separation_engine._calculate_overlap(wave1, wave1)
        self.assertEqual(overlap3, 1.0)
    
    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        seismic_data, sampling_rate = self.sample_seismic_data
        
        # Create mock wave result
        wave_result = WaveAnalysisResult(
            original_data=seismic_data,
            sampling_rate=sampling_rate,
            p_waves=[WaveSegment('P', 10.0, 12.0, np.random.normal(0, 1, 200), 100.0, 1.0, 5.0, 10.5, 0.8)],
            s_waves=[WaveSegment('S', 18.0, 22.0, np.random.normal(0, 1, 400), 100.0, 1.2, 3.0, 18.5, 0.7)],
            surface_waves=[WaveSegment('Love', 25.0, 35.0, np.random.normal(0, 1, 1000), 100.0, 1.5, 1.0, 25.5, 0.9)]
        )
        
        data_quality = {'snr': 5.0, 'completeness': 0.95, 'dynamic_range': 10.0}
        processing_metadata = {'processing_time_seconds': 1.5}
        
        quality_metrics = self.wave_separation_engine._calculate_quality_metrics(
            wave_result, data_quality, processing_metadata
        )
        
        self.assertIsInstance(quality_metrics, QualityMetrics)
        self.assertEqual(quality_metrics.signal_to_noise_ratio, 5.0)
        self.assertGreaterEqual(quality_metrics.detection_confidence, 0)
        self.assertLessEqual(quality_metrics.detection_confidence, 1)
        self.assertGreaterEqual(quality_metrics.analysis_quality_score, 0)
        self.assertLessEqual(quality_metrics.analysis_quality_score, 1)
        self.assertEqual(quality_metrics.data_completeness, 0.95)
        self.assertIsInstance(quality_metrics.processing_warnings, list)
    
    def test_processing_statistics(self):
        """Test processing statistics tracking."""
        seismic_data, _ = self.sample_seismic_data
        
        # Initial statistics
        initial_stats = self.wave_separation_engine.get_processing_statistics()
        self.assertEqual(initial_stats['total_analyses'], 0)
        self.assertEqual(initial_stats['successful_analyses'], 0)
        self.assertEqual(initial_stats['failed_analyses'], 0)
        
        # Run successful analysis
        result = self.wave_separation_engine.separate_waves(seismic_data)
        
        # Check updated statistics
        stats = self.wave_separation_engine.get_processing_statistics()
        self.assertEqual(stats['total_analyses'], 1)
        self.assertEqual(stats['successful_analyses'], 1)
        self.assertEqual(stats['failed_analyses'], 0)
        self.assertGreater(stats['average_processing_time'], 0)
        
        # Reset statistics
        self.wave_separation_engine.reset_statistics()
        reset_stats = self.wave_separation_engine.get_processing_statistics()
        self.assertEqual(reset_stats['total_analyses'], 0)
    
    def test_parameter_updates(self):
        """Test parameter updates functionality."""
        # Create new parameters
        new_params = WaveSeparationParameters(
            sampling_rate=50.0,  # Different sampling rate
            min_snr=5.0,
            min_detection_confidence=0.5
        )
        
        # Update parameters
        self.wave_separation_engine.update_parameters(new_params)
        
        self.assertEqual(self.wave_separation_engine.params, new_params)
        self.assertEqual(self.wave_separation_engine.params.sampling_rate, 50.0)
        self.assertEqual(self.wave_separation_engine.params.min_snr, 5.0)
    
    def test_error_handling(self):
        """Test error handling in wave separation."""
        # Create data that will cause processing errors
        problematic_data = np.random.normal(0, 1, 2000)  # Valid length but may cause detector issues
        
        # Mock detector to raise exception
        with patch.object(self.wave_separation_engine.p_wave_detector, 'detect_waves', side_effect=Exception("Mock error")):
            result = self.wave_separation_engine.separate_waves(problematic_data)
            
            # Should handle error gracefully
            self.assertGreater(len(result.errors), 0)
            self.assertFalse(result.processing_metadata['success'])
            self.assertEqual(result.quality_metrics.analysis_quality_score, 0.0)
    
    def test_low_quality_data_warnings(self):
        """Test warnings for low quality data."""
        # Create low SNR data (mostly noise)
        low_snr_data = 0.01 * np.random.normal(0, 1, 5000)  # Very low amplitude signal
        
        result = self.wave_separation_engine.separate_waves(low_snr_data)
        
        # Should generate warnings for low SNR
        self.assertGreater(len(result.warnings), 0)
        self.assertTrue(any("Low SNR" in warning for warning in result.warnings))
    
    def test_metadata_propagation(self):
        """Test that metadata is properly propagated through the workflow."""
        seismic_data, _ = self.sample_seismic_data
        
        # Test S-wave metadata creation
        p_waves = [WaveSegment('P', 10.0, 12.0, np.array([1, 2, 3]), 100.0, 1.0, 5.0, 10.5, 0.8)]
        s_metadata = self.wave_separation_engine._create_s_wave_metadata({'test': 'value'}, p_waves)
        
        self.assertIn('test', s_metadata)
        self.assertIn('p_wave_arrivals', s_metadata)
        self.assertIn('p_waves', s_metadata)
        self.assertEqual(s_metadata['p_wave_arrivals'], [10.5])
        
        # Test surface wave metadata creation
        s_waves = [WaveSegment('S', 18.0, 22.0, np.array([1, 2, 3]), 100.0, 1.2, 3.0, 18.5, 0.7)]
        surf_metadata = self.wave_separation_engine._create_surface_wave_metadata({'test': 'value'}, p_waves, s_waves)
        
        self.assertIn('test', surf_metadata)
        self.assertIn('p_waves', surf_metadata)
        self.assertIn('s_waves', surf_metadata)


if __name__ == '__main__':
    unittest.main()