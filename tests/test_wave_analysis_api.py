"""
Tests for wave analysis API endpoints.

This module tests the /api/analyze_waves and /api/wave_results endpoints
to ensure proper request handling, response formatting, and error handling.
"""

import pytest
import json
import tempfile
import numpy as np
import os
from unittest.mock import patch, MagicMock
from bson import ObjectId
import soundfile as sf

# Import the Flask app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app, db, fs


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_seismic_data():
    """Generate sample seismic data for testing."""
    # Create synthetic seismic data with P-wave, S-wave, and surface wave patterns
    duration = 60  # 60 seconds
    sampling_rate = 100  # 100 Hz
    t = np.linspace(0, duration, duration * sampling_rate)
    
    # Background noise
    noise = np.random.normal(0, 0.1, len(t))
    
    # P-wave arrival at 10 seconds
    p_wave_start = 10 * sampling_rate
    p_wave_duration = 2 * sampling_rate
    p_wave = np.zeros(len(t))
    p_wave[p_wave_start:p_wave_start + p_wave_duration] = (
        0.5 * np.sin(2 * np.pi * 8 * t[p_wave_start:p_wave_start + p_wave_duration]) *
        np.exp(-0.5 * (t[p_wave_start:p_wave_start + p_wave_duration] - t[p_wave_start]))
    )
    
    # S-wave arrival at 18 seconds (S-P time = 8 seconds)
    s_wave_start = 18 * sampling_rate
    s_wave_duration = 4 * sampling_rate
    s_wave = np.zeros(len(t))
    s_wave[s_wave_start:s_wave_start + s_wave_duration] = (
        0.8 * np.sin(2 * np.pi * 4 * t[s_wave_start:s_wave_start + s_wave_duration]) *
        np.exp(-0.3 * (t[s_wave_start:s_wave_start + s_wave_duration] - t[s_wave_start]))
    )
    
    # Surface wave arrival at 30 seconds
    surface_wave_start = 30 * sampling_rate
    surface_wave_duration = 10 * sampling_rate
    surface_wave = np.zeros(len(t))
    surface_wave[surface_wave_start:surface_wave_start + surface_wave_duration] = (
        0.6 * np.sin(2 * np.pi * 1 * t[surface_wave_start:surface_wave_start + surface_wave_duration]) *
        np.exp(-0.1 * (t[surface_wave_start:surface_wave_start + surface_wave_duration] - t[surface_wave_start]))
    )
    
    # Combine all components
    seismic_signal = noise + p_wave + s_wave + surface_wave
    
    return seismic_signal.astype(np.float32), sampling_rate


@pytest.fixture
def sample_file_id(sample_seismic_data):
    """Create a sample file in GridFS for testing."""
    seismic_data, sampling_rate = sample_seismic_data
    
    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        sf.write(temp_file.name, seismic_data, sampling_rate)
        temp_file_path = temp_file.name
    
    try:
        # Store in GridFS
        with open(temp_file_path, 'rb') as f:
            file_id = fs.put(f, filename='test_seismic.wav')
        
        yield str(file_id)
        
        # Cleanup
        try:
            fs.delete(file_id)
        except:
            pass
    finally:
        # Remove temporary file
        try:
            os.remove(temp_file_path)
        except:
            pass


class TestAnalyzeWavesEndpoint:
    """Test cases for the /api/analyze_waves endpoint."""
    
    def test_analyze_waves_missing_json(self, client):
        """Test analyze_waves endpoint with missing JSON data."""
        response = client.post('/api/analyze_waves')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'No JSON data provided'
    
    def test_analyze_waves_missing_file_id(self, client):
        """Test analyze_waves endpoint with missing file_id."""
        response = client.post('/api/analyze_waves', 
                             json={})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'file_id is required'
    
    def test_analyze_waves_invalid_file_id(self, client):
        """Test analyze_waves endpoint with invalid file_id."""
        response = client.post('/api/analyze_waves', 
                             json={'file_id': 'invalid_id'})
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Failed to load seismic data' in data['error']
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', False)
    def test_analyze_waves_unavailable(self, client):
        """Test analyze_waves endpoint when wave analysis is unavailable."""
        response = client.post('/api/analyze_waves', 
                             json={'file_id': 'test_id'})
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Wave analysis components not available'
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', True)
    def test_analyze_waves_success(self, client, sample_file_id):
        """Test successful wave analysis."""
        # Mock the wave analysis components
        with patch('app.WaveSeparationEngine') as mock_engine_class, \
             patch('app.WaveAnalyzer') as mock_analyzer_class:
            
            # Mock wave separation result
            mock_wave_result = MagicMock()
            mock_wave_result.p_waves = []
            mock_wave_result.s_waves = []
            mock_wave_result.surface_waves = []
            
            mock_quality_metrics = MagicMock()
            mock_quality_metrics.signal_to_noise_ratio = 10.0
            mock_quality_metrics.detection_confidence = 0.8
            mock_quality_metrics.analysis_quality_score = 0.9
            mock_quality_metrics.data_completeness = 1.0
            mock_quality_metrics.processing_warnings = []
            
            mock_separation_result = MagicMock()
            mock_separation_result.wave_analysis_result = mock_wave_result
            mock_separation_result.quality_metrics = mock_quality_metrics
            mock_separation_result.processing_metadata = {'success': True}
            mock_separation_result.warnings = []
            mock_separation_result.errors = []
            
            mock_engine = MagicMock()
            mock_engine.separate_waves.return_value = mock_separation_result
            mock_engine_class.return_value = mock_engine
            
            # Mock detailed analysis
            mock_arrival_times = MagicMock()
            mock_arrival_times.p_wave_arrival = 10.0
            mock_arrival_times.s_wave_arrival = 18.0
            mock_arrival_times.surface_wave_arrival = 30.0
            mock_arrival_times.sp_time_difference = 8.0
            
            mock_detailed_analysis = MagicMock()
            mock_detailed_analysis.arrival_times = mock_arrival_times
            mock_detailed_analysis.magnitude_estimates = []
            mock_detailed_analysis.epicenter_distance = 100.0
            mock_detailed_analysis.frequency_analysis = {}
            
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_waves.return_value = mock_detailed_analysis
            mock_analyzer_class.return_value = mock_analyzer
            
            # Make request
            response = client.post('/api/analyze_waves', 
                                 json={'file_id': sample_file_id})
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify response structure
            assert 'analysis_id' in data
            assert 'file_id' in data
            assert data['file_id'] == sample_file_id
            assert data['status'] == 'success'
            assert 'wave_separation' in data
            assert 'detailed_analysis' in data
            assert 'quality_metrics' in data
            assert 'processing_metadata' in data
            
            # Verify wave separation structure
            wave_sep = data['wave_separation']
            assert 'p_waves' in wave_sep
            assert 's_waves' in wave_sep
            assert 'surface_waves' in wave_sep
            
            # Verify detailed analysis structure
            detailed = data['detailed_analysis']
            assert 'arrival_times' in detailed
            assert 'magnitude_estimates' in detailed
            assert 'epicenter_distance' in detailed
            assert 'frequency_analysis' in detailed
            
            # Verify arrival times
            arrivals = detailed['arrival_times']
            assert arrivals['p_wave_arrival'] == 10.0
            assert arrivals['s_wave_arrival'] == 18.0
            assert arrivals['sp_time_difference'] == 8.0
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', True)
    def test_analyze_waves_with_parameters(self, client, sample_file_id):
        """Test wave analysis with custom parameters."""
        with patch('app.WaveSeparationEngine') as mock_engine_class, \
             patch('app.WaveAnalyzer') as mock_analyzer_class:
            
            # Setup mocks (similar to previous test)
            mock_wave_result = MagicMock()
            mock_wave_result.p_waves = []
            mock_wave_result.s_waves = []
            mock_wave_result.surface_waves = []
            
            mock_quality_metrics = MagicMock()
            mock_quality_metrics.signal_to_noise_ratio = 5.0
            mock_quality_metrics.detection_confidence = 0.6
            mock_quality_metrics.analysis_quality_score = 0.7
            mock_quality_metrics.data_completeness = 0.95
            mock_quality_metrics.processing_warnings = ['Low SNR detected']
            
            mock_separation_result = MagicMock()
            mock_separation_result.wave_analysis_result = mock_wave_result
            mock_separation_result.quality_metrics = mock_quality_metrics
            mock_separation_result.processing_metadata = {'success': True}
            mock_separation_result.warnings = ['Low SNR detected']
            mock_separation_result.errors = []
            
            mock_engine = MagicMock()
            mock_engine.separate_waves.return_value = mock_separation_result
            mock_engine_class.return_value = mock_engine
            
            mock_detailed_analysis = MagicMock()
            mock_detailed_analysis.arrival_times = MagicMock()
            mock_detailed_analysis.magnitude_estimates = []
            mock_detailed_analysis.epicenter_distance = None
            mock_detailed_analysis.frequency_analysis = {}
            
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_waves.return_value = mock_detailed_analysis
            mock_analyzer_class.return_value = mock_analyzer
            
            # Make request with custom parameters
            custom_params = {
                'sampling_rate': 50,
                'min_snr': 1.5,
                'min_detection_confidence': 0.2
            }
            
            response = client.post('/api/analyze_waves', 
                                 json={
                                     'file_id': sample_file_id,
                                     'parameters': custom_params
                                 })
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify that WaveSeparationParameters was called with custom values
            mock_engine_class.assert_called_once()
            call_args = mock_engine_class.call_args[0][0]  # First positional argument
            assert call_args.sampling_rate == 50
            assert call_args.min_snr == 1.5
            assert call_args.min_detection_confidence == 0.2


class TestWaveResultsEndpoint:
    """Test cases for the /api/wave_results/<analysis_id> endpoint."""
    
    def test_wave_results_invalid_analysis_id(self, client):
        """Test wave_results endpoint with invalid analysis_id."""
        response = client.get('/api/wave_results/invalid_id')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Invalid analysis_id format'
    
    def test_wave_results_not_found(self, client):
        """Test wave_results endpoint with non-existent analysis_id."""
        fake_id = str(ObjectId())
        response = client.get(f'/api/wave_results/{fake_id}')
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Analysis result not found'
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', False)
    def test_wave_results_unavailable(self, client):
        """Test wave_results endpoint when wave analysis is unavailable."""
        fake_id = str(ObjectId())
        response = client.get(f'/api/wave_results/{fake_id}')
        assert response.status_code == 503
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Wave analysis components not available'
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', True)
    def test_wave_results_success(self, client):
        """Test successful retrieval of wave analysis results."""
        # Create a mock analysis result in the database
        analysis_id = ObjectId()
        mock_analysis = {
            '_id': analysis_id,
            'file_id': 'test_file_id',
            'analysis_timestamp': '2024-01-01T12:00:00',
            'parameters': {'sampling_rate': 100},
            'wave_separation': {
                'p_waves_count': 2,
                's_waves_count': 1,
                'surface_waves_count': 3
            },
            'quality_metrics': {
                'snr': 10.0,
                'detection_confidence': 0.8,
                'analysis_quality_score': 0.9,
                'data_completeness': 1.0
            },
            'processing_metadata': {'success': True}
        }
        
        with patch.object(db.wave_analyses, 'find_one') as mock_find:
            mock_find.return_value = mock_analysis
            
            response = client.get(f'/api/wave_results/{str(analysis_id)}')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify response structure
            assert data['analysis_id'] == str(analysis_id)
            assert data['file_id'] == 'test_file_id'
            assert 'analysis_timestamp' in data
            assert 'parameters' in data
            assert 'wave_separation' in data
            assert 'quality_metrics' in data
            assert 'processing_metadata' in data
            
            # Verify wave separation data
            wave_sep = data['wave_separation']
            assert wave_sep['p_waves_count'] == 2
            assert wave_sep['s_waves_count'] == 1
            assert wave_sep['surface_waves_count'] == 3
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', True)
    def test_wave_results_with_raw_data(self, client, sample_file_id):
        """Test wave_results endpoint with raw data inclusion."""
        analysis_id = ObjectId()
        mock_analysis = {
            '_id': analysis_id,
            'file_id': sample_file_id,
            'analysis_timestamp': '2024-01-01T12:00:00',
            'parameters': {'sampling_rate': 100},
            'wave_separation': {'p_waves_count': 1},
            'quality_metrics': {'snr': 10.0},
            'processing_metadata': {'success': True}
        }
        
        with patch.object(db.wave_analyses, 'find_one') as mock_find, \
             patch('app.WaveSeparationEngine') as mock_engine_class:
            
            mock_find.return_value = mock_analysis
            
            # Mock wave separation result with raw data
            mock_wave_segment = MagicMock()
            mock_wave_segment.wave_type = 'P'
            mock_wave_segment.start_time = 10.0
            mock_wave_segment.end_time = 12.0
            mock_wave_segment.arrival_time = 10.5
            mock_wave_segment.peak_amplitude = 0.5
            mock_wave_segment.dominant_frequency = 8.0
            mock_wave_segment.confidence = 0.9
            mock_wave_segment.duration = 2.0
            mock_wave_segment.data = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
            mock_wave_segment.sampling_rate = 100
            
            mock_wave_result = MagicMock()
            mock_wave_result.p_waves = [mock_wave_segment]
            mock_wave_result.s_waves = []
            mock_wave_result.surface_waves = []
            
            mock_separation_result = MagicMock()
            mock_separation_result.wave_analysis_result = mock_wave_result
            
            mock_engine = MagicMock()
            mock_engine.separate_waves.return_value = mock_separation_result
            mock_engine_class.return_value = mock_engine
            
            response = client.get(f'/api/wave_results/{str(analysis_id)}?include_raw_data=true')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify raw data is included
            assert 'raw_wave_data' in data
            raw_data = data['raw_wave_data']
            assert 'p_waves' in raw_data
            assert len(raw_data['p_waves']) == 1
            
            p_wave = raw_data['p_waves'][0]
            assert p_wave['wave_type'] == 'P'
            assert p_wave['start_time'] == 10.0
            assert p_wave['raw_data'] == [0.1, 0.2, 0.3, 0.2, 0.1]
            assert p_wave['sampling_rate'] == 100
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', True)
    def test_wave_results_with_wave_type_filter(self, client, sample_file_id):
        """Test wave_results endpoint with wave type filtering."""
        analysis_id = ObjectId()
        mock_analysis = {
            '_id': analysis_id,
            'file_id': sample_file_id,
            'analysis_timestamp': '2024-01-01T12:00:00',
            'parameters': {'sampling_rate': 100},
            'wave_separation': {'p_waves_count': 1, 's_waves_count': 1},
            'quality_metrics': {'snr': 10.0},
            'processing_metadata': {'success': True}
        }
        
        with patch.object(db.wave_analyses, 'find_one') as mock_find, \
             patch('app.WaveSeparationEngine') as mock_engine_class:
            
            mock_find.return_value = mock_analysis
            
            # Mock wave separation result
            mock_p_wave = MagicMock()
            mock_p_wave.wave_type = 'P'
            mock_p_wave.data = np.array([0.1, 0.2])
            mock_p_wave.sampling_rate = 100
            
            mock_s_wave = MagicMock()
            mock_s_wave.wave_type = 'S'
            mock_s_wave.data = np.array([0.3, 0.4])
            mock_s_wave.sampling_rate = 100
            
            mock_wave_result = MagicMock()
            mock_wave_result.p_waves = [mock_p_wave]
            mock_wave_result.s_waves = [mock_s_wave]
            mock_wave_result.surface_waves = []
            
            mock_separation_result = MagicMock()
            mock_separation_result.wave_analysis_result = mock_wave_result
            
            mock_engine = MagicMock()
            mock_engine.separate_waves.return_value = mock_separation_result
            mock_engine_class.return_value = mock_engine
            
            # Request only P-waves
            response = client.get(f'/api/wave_results/{str(analysis_id)}?include_raw_data=true&wave_types=P')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Verify only P-waves are included
            assert 'raw_wave_data' in data
            raw_data = data['raw_wave_data']
            assert 'p_waves' in raw_data
            assert 's_waves' not in raw_data or len(raw_data['s_waves']) == 0


class TestAPIErrorHandling:
    """Test error handling across both API endpoints."""
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', True)
    def test_analyze_waves_processing_error(self, client, sample_file_id):
        """Test analyze_waves endpoint with processing error."""
        with patch('app.WaveSeparationEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.separate_waves.side_effect = Exception("Processing failed")
            mock_engine_class.return_value = mock_engine
            
            response = client.post('/api/analyze_waves', 
                                 json={'file_id': sample_file_id})
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data
            assert data['error'] == 'Wave analysis failed'
            assert 'message' in data
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', True)
    def test_wave_results_database_error(self, client):
        """Test wave_results endpoint with database error."""
        analysis_id = ObjectId()
        
        with patch.object(db.wave_analyses, 'find_one') as mock_find:
            mock_find.side_effect = Exception("Database error")
            
            response = client.get(f'/api/wave_results/{str(analysis_id)}')
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert 'error' in data
            assert data['error'] == 'Failed to retrieve wave results'


class TestAPIIntegration:
    """Integration tests for the complete API workflow."""
    
    @patch('app.WAVE_ANALYSIS_AVAILABLE', True)
    def test_complete_workflow(self, client, sample_file_id):
        """Test complete workflow from analysis to results retrieval."""
        with patch('app.WaveSeparationEngine') as mock_engine_class, \
             patch('app.WaveAnalyzer') as mock_analyzer_class:
            
            # Setup mocks for analysis
            mock_wave_result = MagicMock()
            mock_wave_result.p_waves = []
            mock_wave_result.s_waves = []
            mock_wave_result.surface_waves = []
            
            mock_quality_metrics = MagicMock()
            mock_quality_metrics.signal_to_noise_ratio = 10.0
            mock_quality_metrics.detection_confidence = 0.8
            mock_quality_metrics.analysis_quality_score = 0.9
            mock_quality_metrics.data_completeness = 1.0
            mock_quality_metrics.processing_warnings = []
            
            mock_separation_result = MagicMock()
            mock_separation_result.wave_analysis_result = mock_wave_result
            mock_separation_result.quality_metrics = mock_quality_metrics
            mock_separation_result.processing_metadata = {'success': True}
            mock_separation_result.warnings = []
            mock_separation_result.errors = []
            
            mock_engine = MagicMock()
            mock_engine.separate_waves.return_value = mock_separation_result
            mock_engine_class.return_value = mock_engine
            
            mock_detailed_analysis = MagicMock()
            mock_detailed_analysis.arrival_times = MagicMock()
            mock_detailed_analysis.magnitude_estimates = []
            mock_detailed_analysis.epicenter_distance = None
            mock_detailed_analysis.frequency_analysis = {}
            
            mock_analyzer = MagicMock()
            mock_analyzer.analyze_waves.return_value = mock_detailed_analysis
            mock_analyzer_class.return_value = mock_analyzer
            
            # Step 1: Perform analysis
            analysis_response = client.post('/api/analyze_waves', 
                                          json={'file_id': sample_file_id})
            
            assert analysis_response.status_code == 200
            analysis_data = json.loads(analysis_response.data)
            analysis_id = analysis_data['analysis_id']
            
            # Step 2: Retrieve results
            results_response = client.get(f'/api/wave_results/{analysis_id}')
            
            assert results_response.status_code == 200
            results_data = json.loads(results_response.data)
            
            # Verify consistency between analysis and results
            assert results_data['analysis_id'] == analysis_id
            assert results_data['file_id'] == sample_file_id


if __name__ == '__main__':
    pytest.main([__file__])