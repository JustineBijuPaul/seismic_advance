"""
Unit tests for data export functionality.

This module tests the data export capabilities including MSEED, CSV, and JSON
export formats with validation of output formats and data integrity.
"""

import pytest
import numpy as np
import json
import csv
import io
from datetime import datetime
from unittest.mock import Mock, patch

# Import the classes to test
from wave_analysis.services.data_exporter import DataExporter, MSEEDExporter, CSVExporter
from wave_analysis.models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, ArrivalTimes,
    MagnitudeEstimate, FrequencyData, QualityMetrics
)


class TestDataExporter:
    """Test cases for the main DataExporter class."""
    
    @pytest.fixture
    def sample_wave_segments(self):
        """Create sample wave segments for testing."""
        # Create synthetic wave data
        sampling_rate = 100.0
        duration = 2.0
        t = np.linspace(0, duration, int(sampling_rate * duration))
        
        # P-wave: higher frequency, shorter duration
        p_data = np.sin(2 * np.pi * 10 * t) * np.exp(-t * 2)
        p_wave = WaveSegment(
            wave_type='P',
            start_time=1.0,
            end_time=3.0,
            data=p_data,
            sampling_rate=sampling_rate,
            peak_amplitude=1.0,
            dominant_frequency=10.0,
            arrival_time=1.2,
            confidence=0.9,
            metadata={'detector': 'test'}
        )
        
        # S-wave: lower frequency, longer duration
        s_data = np.sin(2 * np.pi * 5 * t) * np.exp(-t * 1)
        s_wave = WaveSegment(
            wave_type='S',
            start_time=3.5,
            end_time=5.5,
            data=s_data,
            sampling_rate=sampling_rate,
            peak_amplitude=0.8,
            dominant_frequency=5.0,
            arrival_time=3.7,
            confidence=0.85
        )
        
        # Surface wave: very low frequency, long duration
        surface_data = np.sin(2 * np.pi * 2 * t) * np.exp(-t * 0.5)
        surface_wave = WaveSegment(
            wave_type='Love',
            start_time=8.0,
            end_time=10.0,
            data=surface_data,
            sampling_rate=sampling_rate,
            peak_amplitude=1.2,
            dominant_frequency=2.0,
            arrival_time=8.2,
            confidence=0.75
        )
        
        return {
            'P': [p_wave],
            'S': [s_wave],
            'Love': [surface_wave]
        }
    
    @pytest.fixture
    def sample_analysis(self, sample_wave_segments):
        """Create sample detailed analysis for testing."""
        # Create wave analysis result
        original_data = np.random.randn(1000)
        wave_result = WaveAnalysisResult(
            original_data=original_data,
            sampling_rate=100.0,
            p_waves=sample_wave_segments['P'],
            s_waves=sample_wave_segments['S'],
            surface_waves=sample_wave_segments['Love']
        )
        
        # Create arrival times
        arrival_times = ArrivalTimes(
            p_wave_arrival=1.2,
            s_wave_arrival=3.7,
            surface_wave_arrival=8.2,
            sp_time_difference=2.5
        )
        
        # Create magnitude estimates
        magnitude_estimates = [
            MagnitudeEstimate(
                method='ML',
                magnitude=4.2,
                confidence=0.8,
                wave_type_used='P'
            ),
            MagnitudeEstimate(
                method='Ms',
                magnitude=4.1,
                confidence=0.75,
                wave_type_used='Love'
            )
        ]
        
        # Create frequency data
        frequencies = np.linspace(0, 50, 100)
        power_spectrum = np.exp(-frequencies / 10)
        freq_data = FrequencyData(
            frequencies=frequencies,
            power_spectrum=power_spectrum,
            dominant_frequency=5.0,
            frequency_range=(1.0, 20.0),
            spectral_centroid=7.5,
            bandwidth=15.0
        )
        
        # Create quality metrics
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=15.0,
            detection_confidence=0.82,
            analysis_quality_score=0.88,
            data_completeness=0.95,
            processing_warnings=['Low SNR in surface waves']
        )
        
        # Create detailed analysis
        analysis = DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates,
            epicenter_distance=45.2,
            quality_metrics=quality_metrics
        )
        analysis.frequency_analysis['P'] = freq_data
        
        return analysis
    
    def test_init(self):
        """Test DataExporter initialization."""
        exporter = DataExporter()
        assert isinstance(exporter.supported_formats, list)
        assert 'csv' in exporter.supported_formats
        assert 'json' in exporter.supported_formats
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        exporter = DataExporter()
        formats = exporter.get_supported_formats()
        assert isinstance(formats, list)
        assert len(formats) > 0
        assert 'csv' in formats
        assert 'json' in formats
    
    def test_export_waves_invalid_format(self, sample_wave_segments):
        """Test export with invalid format raises ValueError."""
        exporter = DataExporter()
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export_waves(sample_wave_segments, 'invalid_format')
    
    def test_export_json(self, sample_wave_segments):
        """Test JSON export functionality."""
        exporter = DataExporter()
        result = exporter.export_waves(sample_wave_segments, 'json')
        
        # Verify it's bytes
        assert isinstance(result, bytes)
        
        # Parse JSON and verify structure
        data = json.loads(result.decode('utf-8'))
        assert 'export_timestamp' in data
        assert 'wave_data' in data
        
        # Verify wave data structure
        wave_data = data['wave_data']
        assert 'P' in wave_data
        assert 'S' in wave_data
        assert 'Love' in wave_data
        
        # Verify P-wave data
        p_wave_data = wave_data['P'][0]
        assert p_wave_data['wave_type'] == 'P'
        assert p_wave_data['start_time'] == 1.0
        assert p_wave_data['end_time'] == 3.0
        assert p_wave_data['dominant_frequency'] == 10.0
        assert p_wave_data['confidence'] == 0.9
        assert isinstance(p_wave_data['data'], list)
        assert p_wave_data['metadata']['detector'] == 'test'
    
    def test_export_csv(self, sample_wave_segments):
        """Test CSV export functionality."""
        exporter = DataExporter()
        result = exporter.export_waves(sample_wave_segments, 'csv')
        
        # Verify it's bytes
        assert isinstance(result, bytes)
        
        # Parse CSV and verify structure
        csv_content = result.decode('utf-8')
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)
        
        # Verify header
        header = rows[0]
        expected_columns = [
            'wave_type', 'segment_id', 'start_time', 'end_time', 'duration',
            'arrival_time', 'peak_amplitude', 'dominant_frequency', 'confidence',
            'sampling_rate', 'sample_count', 'data_points'
        ]
        assert header == expected_columns
        
        # Verify data rows (should have 3 waves)
        data_rows = rows[1:]
        assert len(data_rows) == 3
        
        # Verify P-wave row
        p_row = data_rows[0]
        assert p_row[0] == 'P'  # wave_type
        assert float(p_row[2]) == 1.0  # start_time
        assert float(p_row[3]) == 3.0  # end_time
        assert float(p_row[7]) == 10.0  # dominant_frequency
    
    @patch('wave_analysis.services.data_exporter.OBSPY_AVAILABLE', True)
    @patch('wave_analysis.services.data_exporter.Stream')
    @patch('wave_analysis.services.data_exporter.Trace')
    @patch('wave_analysis.services.data_exporter.Stats')
    @patch('wave_analysis.services.data_exporter.UTCDateTime')
    def test_export_mseed(self, mock_utc, mock_stats, mock_trace, mock_stream, sample_wave_segments):
        """Test MSEED export functionality."""
        # Setup mocks
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance
        mock_stream_instance.write = Mock()
        mock_stream_instance.append = Mock()
        
        mock_trace_instance = Mock()
        mock_trace.return_value = mock_trace_instance
        
        mock_stats_instance = Mock()
        mock_stats.return_value = mock_stats_instance
        
        mock_utc_instance = Mock()
        mock_utc.return_value = mock_utc_instance
        
        # Mock the buffer write to return bytes
        buffer_mock = Mock()
        buffer_mock.getvalue.return_value = b'mock_mseed_data'
        mock_stream_instance.write.return_value = None
        
        with patch('io.BytesIO', return_value=buffer_mock):
            exporter = DataExporter()
            result = exporter.export_waves(sample_wave_segments, 'mseed')
        
        # Verify mocks were called
        assert mock_stream.called
        assert mock_trace.called
        assert mock_stats.called
        assert mock_stream_instance.write.called
        assert isinstance(result, bytes)
    
    def test_export_analysis_json(self, sample_analysis):
        """Test JSON export of analysis results."""
        exporter = DataExporter()
        result = exporter.export_analysis_results(sample_analysis, 'json')
        
        # Verify it's bytes
        assert isinstance(result, bytes)
        
        # Parse JSON and verify structure
        data = json.loads(result.decode('utf-8'))
        assert 'analysis_timestamp' in data
        assert 'arrival_times' in data
        assert 'magnitude_estimates' in data
        assert 'wave_summary' in data
        
        # Verify arrival times
        arrival_times = data['arrival_times']
        assert arrival_times['p_wave_arrival'] == 1.2
        assert arrival_times['s_wave_arrival'] == 3.7
        assert arrival_times['sp_time_difference'] == 2.5
        
        # Verify magnitude estimates
        mag_estimates = data['magnitude_estimates']
        assert len(mag_estimates) == 2
        assert mag_estimates[0]['method'] == 'ML'
        assert mag_estimates[0]['magnitude'] == 4.2
        
        # Verify wave summary
        wave_summary = data['wave_summary']
        assert wave_summary['total_waves_detected'] == 3
        assert 'P' in wave_summary['wave_types_detected']
        assert 'S' in wave_summary['wave_types_detected']
    
    def test_export_analysis_csv(self, sample_analysis):
        """Test CSV export of analysis results."""
        exporter = DataExporter()
        result = exporter.export_analysis_results(sample_analysis, 'csv')
        
        # Verify it's bytes
        assert isinstance(result, bytes)
        
        # Parse CSV content
        csv_content = result.decode('utf-8')
        lines = csv_content.strip().split('\n')
        
        # Verify key sections are present
        assert any('Analysis Summary' in line for line in lines)
        assert any('Arrival Times' in line for line in lines)
        assert any('Magnitude Estimates' in line for line in lines)
        assert any('Quality Metrics' in line for line in lines)
    
    def test_export_analysis_invalid_format(self, sample_analysis):
        """Test export analysis with invalid format."""
        exporter = DataExporter()
        with pytest.raises(ValueError, match="Unsupported analysis export format"):
            exporter.export_analysis_results(sample_analysis, 'invalid')


class TestMSEEDExporter:
    """Test cases for the specialized MSEED exporter."""
    
    @pytest.fixture
    def sample_wave_result(self):
        """Create sample wave analysis result."""
        original_data = np.random.randn(1000)
        
        # Create sample waves
        p_wave = WaveSegment(
            wave_type='P',
            start_time=1.0,
            end_time=2.0,
            data=np.random.randn(100),
            sampling_rate=100.0,
            peak_amplitude=1.0,
            dominant_frequency=10.0,
            arrival_time=1.2
        )
        
        return WaveAnalysisResult(
            original_data=original_data,
            sampling_rate=100.0,
            p_waves=[p_wave],
            s_waves=[],
            surface_waves=[]
        )
    
    @patch('wave_analysis.services.data_exporter.OBSPY_AVAILABLE', False)
    def test_init_without_obspy(self):
        """Test initialization without ObsPy raises error."""
        with pytest.raises(RuntimeError, match="ObsPy is required"):
            MSEEDExporter()
    
    @patch('wave_analysis.services.data_exporter.OBSPY_AVAILABLE', True)
    @patch('wave_analysis.services.data_exporter.Stream')
    @patch('wave_analysis.services.data_exporter.Trace')
    @patch('wave_analysis.services.data_exporter.Stats')
    @patch('wave_analysis.services.data_exporter.UTCDateTime')
    def test_export_separated_waves(self, mock_utc, mock_stats, mock_trace, mock_stream, sample_wave_result):
        """Test export of separated waves."""
        mock_stream_instance = Mock()
        mock_stream.return_value = mock_stream_instance
        mock_stream_instance.write = Mock()
        mock_stream_instance.append = Mock()
        
        mock_trace_instance = Mock()
        mock_trace.return_value = mock_trace_instance
        
        mock_stats_instance = Mock()
        mock_stats.return_value = mock_stats_instance
        
        mock_utc_instance = Mock()
        mock_utc.return_value = mock_utc_instance
        # Mock addition operation for UTCDateTime
        mock_utc_instance.__add__ = Mock(return_value=mock_utc_instance)
        
        # Mock the buffer write to return bytes
        buffer_mock = Mock()
        buffer_mock.getvalue.return_value = b'mock_mseed_data'
        mock_stream_instance.write.return_value = None
        
        with patch('io.BytesIO', return_value=buffer_mock):
            exporter = MSEEDExporter()
            result = exporter.export_separated_waves(sample_wave_result)
        
        # Verify stream operations were called
        assert mock_stream.called
        assert mock_stream_instance.write.called
        assert isinstance(result, bytes)


class TestCSVExporter:
    """Test cases for the specialized CSV exporter."""
    
    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis for CSV export testing."""
        # Create minimal analysis structure
        original_data = np.random.randn(500)
        wave_result = WaveAnalysisResult(
            original_data=original_data,
            sampling_rate=100.0
        )
        
        arrival_times = ArrivalTimes(
            p_wave_arrival=1.0,
            s_wave_arrival=2.5,
            sp_time_difference=1.5
        )
        
        analysis = DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times
        )
        
        return analysis
    
    def test_export_wave_characteristics(self, sample_analysis):
        """Test export of wave characteristics."""
        exporter = CSVExporter()
        result = exporter.export_wave_characteristics(sample_analysis)
        
        # Verify it's bytes
        assert isinstance(result, bytes)
        
        # Parse CSV content
        csv_content = result.decode('utf-8')
        lines = csv_content.strip().split('\n')
        
        # Verify header comments
        assert any('Wave Analysis Export' in line for line in lines)
        assert any('Generated:' in line for line in lines)
        
        # Verify column headers are present
        header_line = None
        for line in lines:
            if 'Wave Type' in line and 'Segment ID' in line:
                header_line = line
                break
        
        assert header_line is not None
        assert 'Peak Amplitude' in header_line
        assert 'Dominant Frequency' in header_line
    
    def test_export_timing_data(self, sample_analysis):
        """Test export of timing data."""
        exporter = CSVExporter()
        result = exporter.export_timing_data(sample_analysis)
        
        # Verify it's bytes
        assert isinstance(result, bytes)
        
        # Parse CSV content
        csv_content = result.decode('utf-8')
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)
        
        # Find timing analysis section
        timing_found = False
        for row in rows:
            if row and 'Timing Analysis' in row[0]:
                timing_found = True
                break
        
        assert timing_found
        
        # Verify timing data is present
        csv_text = csv_content.lower()
        assert 'p-wave arrival' in csv_text
        assert 's-wave arrival' in csv_text
        assert 's-p time difference' in csv_text


class TestDataExportIntegration:
    """Integration tests for data export functionality."""
    
    def test_export_workflow_json(self):
        """Test complete export workflow with JSON format."""
        # Create realistic test data
        sampling_rate = 100.0
        duration = 10.0
        t = np.linspace(0, duration, int(sampling_rate * duration))
        
        # Create earthquake-like signal
        p_signal = np.sin(2 * np.pi * 15 * t) * np.exp(-t * 3) * (t > 1.0) * (t < 3.0)
        s_signal = np.sin(2 * np.pi * 8 * t) * np.exp(-t * 2) * (t > 3.0) * (t < 6.0)
        noise = np.random.randn(len(t)) * 0.1
        
        original_data = p_signal + s_signal + noise
        
        # Create wave segments
        p_wave = WaveSegment(
            wave_type='P',
            start_time=1.0,
            end_time=3.0,
            data=original_data[100:300],  # 1-3 seconds
            sampling_rate=sampling_rate,
            peak_amplitude=np.max(np.abs(p_signal)),
            dominant_frequency=15.0,
            arrival_time=1.2
        )
        
        s_wave = WaveSegment(
            wave_type='S',
            start_time=3.0,
            end_time=6.0,
            data=original_data[300:600],  # 3-6 seconds
            sampling_rate=sampling_rate,
            peak_amplitude=np.max(np.abs(s_signal)),
            dominant_frequency=8.0,
            arrival_time=3.2
        )
        
        waves = {'P': [p_wave], 'S': [s_wave]}
        
        # Test export
        exporter = DataExporter()
        result = exporter.export_waves(waves, 'json')
        
        # Verify export
        assert isinstance(result, bytes)
        data = json.loads(result.decode('utf-8'))
        
        # Verify data integrity
        assert len(data['wave_data']['P']) == 1
        assert len(data['wave_data']['S']) == 1
        
        exported_p = data['wave_data']['P'][0]
        assert exported_p['dominant_frequency'] == 15.0
        assert exported_p['wave_type'] == 'P'
        assert len(exported_p['data']) == 200  # 2 seconds at 100 Hz
    
    def test_export_workflow_csv(self):
        """Test complete export workflow with CSV format."""
        # Create simple test data
        wave = WaveSegment(
            wave_type='P',
            start_time=0.0,
            end_time=1.0,
            data=np.sin(2 * np.pi * np.linspace(0, 1, 100)),
            sampling_rate=100.0,
            peak_amplitude=1.0,
            dominant_frequency=1.0,
            arrival_time=0.1
        )
        
        waves = {'P': [wave]}
        
        # Test export
        exporter = DataExporter()
        result = exporter.export_waves(waves, 'csv')
        
        # Verify export
        assert isinstance(result, bytes)
        
        # Parse and verify CSV structure
        csv_content = result.decode('utf-8')
        reader = csv.reader(io.StringIO(csv_content))
        rows = list(reader)
        
        assert len(rows) == 2  # Header + 1 data row
        assert rows[0][0] == 'wave_type'
        assert rows[1][0] == 'P'
        assert float(rows[1][7]) == 1.0  # dominant_frequency


if __name__ == '__main__':
    pytest.main([__file__])