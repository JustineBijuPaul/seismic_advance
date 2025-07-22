"""
Unit tests for SAC format export functionality.

This module tests the SACExporter class and SAC format support
in the DataExporter class, including header validation and
wave separation markers.
"""

import unittest
import numpy as np
import io
import zipfile
from datetime import datetime
from typing import Dict, Any

try:
    from obspy import Trace, UTCDateTime
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False

from wave_analysis.services.data_exporter import SACExporter, DataExporter
from wave_analysis.models.wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, 
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)


class TestSACExporter(unittest.TestCase):
    """Test cases for SAC format export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not OBSPY_AVAILABLE:
            self.skipTest("ObsPy not available for SAC export tests")
        
        # Create synthetic wave data
        self.sampling_rate = 100.0
        self.duration = 10.0
        self.time_samples = int(self.duration * self.sampling_rate)
        
        # Generate synthetic seismic data
        t = np.linspace(0, self.duration, self.time_samples)
        
        # P-wave: High frequency, early arrival
        p_wave_data = 0.5 * np.sin(2 * np.pi * 15 * t) * np.exp(-t/2)
        
        # S-wave: Medium frequency, later arrival
        s_wave_data = 0.8 * np.sin(2 * np.pi * 8 * t) * np.exp(-(t-3)/3)
        
        # Surface wave: Low frequency, latest arrival
        surface_wave_data = 1.2 * np.sin(2 * np.pi * 3 * t) * np.exp(-(t-6)/4)
        
        # Combined signal
        self.original_data = p_wave_data + s_wave_data + surface_wave_data
        
        # Create wave segments
        self.p_wave = WaveSegment(
            wave_type="P",
            start_time=0.5,
            end_time=3.0,
            data=p_wave_data[50:300],
            sampling_rate=self.sampling_rate,
            peak_amplitude=0.5,
            dominant_frequency=15.0,
            arrival_time=1.0,
            confidence=0.95,
            metadata={"detection_method": "STA/LTA"}
        )
        
        self.s_wave = WaveSegment(
            wave_type="S",
            start_time=3.0,
            end_time=6.0,
            data=s_wave_data[300:600],
            sampling_rate=self.sampling_rate,
            peak_amplitude=0.8,
            dominant_frequency=8.0,
            arrival_time=3.5,
            confidence=0.88,
            metadata={"detection_method": "polarization"}
        )
        
        self.love_wave = WaveSegment(
            wave_type="Love",
            start_time=6.0,
            end_time=10.0,
            data=surface_wave_data[600:1000],
            sampling_rate=self.sampling_rate,
            peak_amplitude=1.2,
            dominant_frequency=3.0,
            arrival_time=6.5,
            confidence=0.82,
            metadata={"detection_method": "frequency_analysis"}
        )
        
        # Create wave analysis result
        self.wave_result = WaveAnalysisResult(
            original_data=self.original_data,
            sampling_rate=self.sampling_rate,
            p_waves=[self.p_wave],
            s_waves=[self.s_wave],
            surface_waves=[self.love_wave],
            metadata={
                "processing_time": 2.5,
                "model_version": "v1.0",
                "analysis_timestamp": datetime.now()
            }
        )
        
        self.sac_exporter = SACExporter()
    
    def test_sac_exporter_initialization(self):
        """Test SAC exporter initialization."""
        exporter = SACExporter()
        self.assertIsInstance(exporter, SACExporter)
    
    def test_sac_exporter_requires_obspy(self):
        """Test that SAC exporter requires ObsPy."""
        # This test would only fail if ObsPy is not available
        # Since we skip the entire test class if ObsPy is not available,
        # this test mainly documents the requirement
        self.assertTrue(OBSPY_AVAILABLE)
    
    def test_export_separated_waves(self):
        """Test exporting separated waves as individual SAC files."""
        sac_files = self.sac_exporter.export_separated_waves(self.wave_result)
        
        # Check that we get the expected number of files
        expected_files = 4  # original + P + S + Love
        self.assertEqual(len(sac_files), expected_files)
        
        # Check file naming convention
        expected_filenames = [
            "XX.WAVE.00.HHZ.original.sac",
            "XX.WAVE.00.HHP00.sac",
            "XX.WAVE.00.HHS00.sac",
            "XX.WAVE.00.HHL00.sac"
        ]
        
        for filename in expected_filenames:
            self.assertIn(filename, sac_files)
            self.assertIsInstance(sac_files[filename], bytes)
            self.assertGreater(len(sac_files[filename]), 0)
    
    def test_export_separated_waves_custom_codes(self):
        """Test exporting with custom station and network codes."""
        sac_files = self.sac_exporter.export_separated_waves(
            self.wave_result,
            station_code="TEST",
            network_code="YY",
            location_code="01"
        )
        
        # Check custom naming
        expected_filename = "YY.TEST.01.HHZ.original.sac"
        self.assertIn(expected_filename, sac_files)
    
    def test_export_single_sac_file(self):
        """Test exporting all data in a single SAC file."""
        sac_data = self.sac_exporter.export_single_sac_file(self.wave_result)
        
        self.assertIsInstance(sac_data, bytes)
        self.assertGreater(len(sac_data), 0)
    
    def test_export_single_sac_file_with_markers(self):
        """Test exporting single SAC file with wave markers."""
        sac_data = self.sac_exporter.export_single_sac_file(
            self.wave_result, 
            include_markers=True
        )
        
        self.assertIsInstance(sac_data, bytes)
        self.assertGreater(len(sac_data), 0)
        
        # Validate that markers were added
        validation_result = self.sac_exporter.validate_sac_export(sac_data)
        self.assertTrue(validation_result['valid'])
        self.assertIn('wave_markers', validation_result)
        
        # Check for expected markers
        markers = validation_result['wave_markers']
        self.assertIn('T0', markers)  # P-wave marker
        self.assertIn('T1', markers)  # S-wave marker
        self.assertIn('T2', markers)  # Surface wave marker
    
    def test_export_single_sac_file_without_markers(self):
        """Test exporting single SAC file without wave markers."""
        sac_data = self.sac_exporter.export_single_sac_file(
            self.wave_result, 
            include_markers=False
        )
        
        self.assertIsInstance(sac_data, bytes)
        self.assertGreater(len(sac_data), 0)
    
    def test_validate_sac_export(self):
        """Test SAC export validation functionality."""
        # Export a SAC file
        sac_files = self.sac_exporter.export_separated_waves(self.wave_result)
        original_sac = sac_files["XX.WAVE.00.HHZ.original.sac"]
        
        # Validate the export
        validation_result = self.sac_exporter.validate_sac_export(original_sac)
        
        self.assertTrue(validation_result['valid'])
        self.assertIn('header_info', validation_result)
        self.assertIn('sac_headers', validation_result)
        
        # Check header information
        header_info = validation_result['header_info']
        self.assertEqual(header_info['station'], 'WAVE')
        self.assertEqual(header_info['network'], 'XX')
        self.assertEqual(header_info['sampling_rate'], self.sampling_rate)
        self.assertEqual(header_info['npts'], len(self.original_data))
    
    def test_validate_p_wave_sac_headers(self):
        """Test validation of P-wave specific SAC headers."""
        sac_files = self.sac_exporter.export_separated_waves(self.wave_result)
        p_wave_sac = sac_files["XX.WAVE.00.HHP00.sac"]
        
        validation_result = self.sac_exporter.validate_sac_export(p_wave_sac)
        
        self.assertTrue(validation_result['valid'])
        
        # Check SAC-specific headers
        sac_headers = validation_result['sac_headers']
        self.assertEqual(sac_headers['wave_type'], 'P')
        self.assertAlmostEqual(sac_headers['arrival_time'], self.p_wave.arrival_time, places=2)
        self.assertAlmostEqual(sac_headers['peak_amplitude'], self.p_wave.peak_amplitude, places=2)
        self.assertAlmostEqual(sac_headers['dominant_frequency'], self.p_wave.dominant_frequency, places=1)
        self.assertAlmostEqual(sac_headers['confidence'], self.p_wave.confidence, places=2)
    
    def test_validate_invalid_sac_data(self):
        """Test validation with invalid SAC data."""
        invalid_data = b"not a sac file"
        
        validation_result = self.sac_exporter.validate_sac_export(invalid_data)
        
        self.assertFalse(validation_result['valid'])
        self.assertIn('error', validation_result)
    
    def test_multiple_waves_same_type(self):
        """Test exporting multiple waves of the same type."""
        # Create additional P-wave
        p_wave_2 = WaveSegment(
            wave_type="P",
            start_time=1.5,
            end_time=4.0,
            data=np.sin(2 * np.pi * 12 * np.linspace(0, 2.5, 250)),
            sampling_rate=self.sampling_rate,
            peak_amplitude=0.3,
            dominant_frequency=12.0,
            arrival_time=2.0,
            confidence=0.75
        )
        
        # Create wave result with multiple P-waves
        wave_result_multi = WaveAnalysisResult(
            original_data=self.original_data,
            sampling_rate=self.sampling_rate,
            p_waves=[self.p_wave, p_wave_2],
            s_waves=[self.s_wave],
            surface_waves=[self.love_wave]
        )
        
        sac_files = self.sac_exporter.export_separated_waves(wave_result_multi)
        
        # Should have 5 files: original + 2 P-waves + 1 S-wave + 1 Love wave
        self.assertEqual(len(sac_files), 5)
        
        # Check for both P-wave files
        self.assertIn("XX.WAVE.00.HHP00.sac", sac_files)
        self.assertIn("XX.WAVE.00.HHP01.sac", sac_files)
    
    def test_rayleigh_wave_export(self):
        """Test exporting Rayleigh waves with correct naming."""
        rayleigh_wave = WaveSegment(
            wave_type="Rayleigh",
            start_time=7.0,
            end_time=10.0,
            data=np.sin(2 * np.pi * 2 * np.linspace(0, 3, 300)),
            sampling_rate=self.sampling_rate,
            peak_amplitude=1.0,
            dominant_frequency=2.0,
            arrival_time=7.5,
            confidence=0.80
        )
        
        wave_result_rayleigh = WaveAnalysisResult(
            original_data=self.original_data,
            sampling_rate=self.sampling_rate,
            p_waves=[self.p_wave],
            s_waves=[self.s_wave],
            surface_waves=[rayleigh_wave]
        )
        
        sac_files = self.sac_exporter.export_separated_waves(wave_result_rayleigh)
        
        # Check for Rayleigh wave file with 'R' designation
        self.assertIn("XX.WAVE.00.HHR00.sac", sac_files)
    
    def test_wave_markers_multiple_waves(self):
        """Test wave markers with multiple waves of different types."""
        # Create additional waves
        p_wave_2 = WaveSegment(
            wave_type="P", start_time=1.5, end_time=4.0,
            data=np.sin(2 * np.pi * 12 * np.linspace(0, 2.5, 250)),
            sampling_rate=self.sampling_rate, peak_amplitude=0.3,
            dominant_frequency=12.0, arrival_time=2.0, confidence=0.75
        )
        
        s_wave_2 = WaveSegment(
            wave_type="S", start_time=4.0, end_time=7.0,
            data=np.sin(2 * np.pi * 6 * np.linspace(0, 3, 300)),
            sampling_rate=self.sampling_rate, peak_amplitude=0.6,
            dominant_frequency=6.0, arrival_time=4.5, confidence=0.70
        )
        
        wave_result_multi = WaveAnalysisResult(
            original_data=self.original_data,
            sampling_rate=self.sampling_rate,
            p_waves=[self.p_wave, p_wave_2],
            s_waves=[self.s_wave, s_wave_2],
            surface_waves=[self.love_wave]
        )
        
        sac_data = self.sac_exporter.export_single_sac_file(
            wave_result_multi, 
            include_markers=True
        )
        
        validation_result = self.sac_exporter.validate_sac_export(sac_data)
        markers = validation_result['wave_markers']
        
        # Should have markers for multiple waves
        self.assertGreaterEqual(len(markers), 3)  # At least T0, T1, T2
        
        # Check that we have P and S markers
        marker_labels = [marker['label'] for marker in markers.values()]
        self.assertIn('P', marker_labels)
        self.assertIn('S', marker_labels)


class TestDataExporterSACIntegration(unittest.TestCase):
    """Test SAC format integration with main DataExporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not OBSPY_AVAILABLE:
            self.skipTest("ObsPy not available for SAC export tests")
        
        self.data_exporter = DataExporter()
        
        # Create test wave segments
        self.p_wave = WaveSegment(
            wave_type="P", start_time=0.5, end_time=3.0,
            data=np.sin(2 * np.pi * 15 * np.linspace(0, 2.5, 250)),
            sampling_rate=100.0, peak_amplitude=0.5,
            dominant_frequency=15.0, arrival_time=1.0, confidence=0.95
        )
        
        self.s_wave = WaveSegment(
            wave_type="S", start_time=3.0, end_time=6.0,
            data=np.sin(2 * np.pi * 8 * np.linspace(0, 3, 300)),
            sampling_rate=100.0, peak_amplitude=0.8,
            dominant_frequency=8.0, arrival_time=3.5, confidence=0.88
        )
        
        self.waves_dict = {
            'P': [self.p_wave],
            'S': [self.s_wave]
        }
    
    def test_sac_format_supported(self):
        """Test that SAC format is in supported formats."""
        supported_formats = self.data_exporter.get_supported_formats()
        self.assertIn('sac', supported_formats)
    
    def test_export_waves_sac_format(self):
        """Test exporting waves in SAC format through DataExporter."""
        sac_data = self.data_exporter.export_waves(self.waves_dict, 'sac')
        
        self.assertIsInstance(sac_data, bytes)
        self.assertGreater(len(sac_data), 0)
        
        # Verify it's a ZIP file
        zip_buffer = io.BytesIO(sac_data)
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            file_list = zip_file.namelist()
            self.assertGreater(len(file_list), 0)
            
            # Check for SAC files in the ZIP
            sac_files = [f for f in file_list if f.endswith('.sac')]
            self.assertGreater(len(sac_files), 0)
    
    def test_export_waves_sac_case_insensitive(self):
        """Test that SAC format export is case insensitive."""
        sac_data_lower = self.data_exporter.export_waves(self.waves_dict, 'sac')
        sac_data_upper = self.data_exporter.export_waves(self.waves_dict, 'SAC')
        
        # Both should work and produce data
        self.assertIsInstance(sac_data_lower, bytes)
        self.assertIsInstance(sac_data_upper, bytes)
        self.assertGreater(len(sac_data_lower), 0)
        self.assertGreater(len(sac_data_upper), 0)
    
    def test_export_waves_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        with self.assertRaises(ValueError) as context:
            self.data_exporter.export_waves(self.waves_dict, 'unsupported')
        
        self.assertIn("Unsupported format", str(context.exception))
    
    def test_export_waves_empty_data(self):
        """Test exporting empty wave data."""
        empty_waves = {}
        sac_data = self.data_exporter.export_waves(empty_waves, 'sac')
        
        self.assertIsInstance(sac_data, bytes)
        # Should still produce a valid ZIP file, even if empty
        zip_buffer = io.BytesIO(sac_data)
        with zipfile.ZipFile(zip_buffer, 'r') as zip_file:
            # May have an original data file even with no waves
            pass  # Just verify it doesn't crash


class TestSACExportErrorHandling(unittest.TestCase):
    """Test error handling in SAC export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not OBSPY_AVAILABLE:
            self.skipTest("ObsPy not available for SAC export tests")
    
    def test_invalid_wave_segment_data(self):
        """Test handling of invalid wave segment data."""
        # Create wave segment with invalid data
        invalid_wave = WaveSegment(
            wave_type="P", start_time=0.0, end_time=1.0,
            data=np.array([]),  # Empty data array
            sampling_rate=100.0, peak_amplitude=0.5,
            dominant_frequency=15.0, arrival_time=0.5, confidence=0.95
        )
        
        wave_result = WaveAnalysisResult(
            original_data=np.sin(2 * np.pi * np.linspace(0, 10, 1000)),
            sampling_rate=100.0,
            p_waves=[invalid_wave]
        )
        
        sac_exporter = SACExporter()
        
        # Should handle empty data gracefully
        sac_files = sac_exporter.export_separated_waves(wave_result)
        self.assertIsInstance(sac_files, dict)
    
    def test_zero_sampling_rate(self):
        """Test handling of zero sampling rate."""
        with self.assertRaises(ValueError):
            WaveSegment(
                wave_type="P", start_time=0.0, end_time=1.0,
                data=np.sin(2 * np.pi * np.linspace(0, 1, 100)),
                sampling_rate=0.0,  # Invalid sampling rate
                peak_amplitude=0.5, dominant_frequency=15.0,
                arrival_time=0.5, confidence=0.95
            )
    
    def test_negative_confidence(self):
        """Test handling of invalid confidence values."""
        with self.assertRaises(ValueError):
            WaveSegment(
                wave_type="P", start_time=0.0, end_time=1.0,
                data=np.sin(2 * np.pi * np.linspace(0, 1, 100)),
                sampling_rate=100.0, peak_amplitude=0.5,
                dominant_frequency=15.0, arrival_time=0.5,
                confidence=-0.1  # Invalid confidence
            )


if __name__ == '__main__':
    unittest.main()