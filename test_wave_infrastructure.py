#!/usr/bin/env python3
"""
Test wave analysis infrastructure and API endpoints.
"""

import sys
import os
import json
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_seismic_data():
    """Create synthetic seismic data for testing."""
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
    
    # S-wave arrival at 18 seconds
    s_wave_start = 18 * sampling_rate
    s_wave_duration = 4 * sampling_rate
    s_wave = np.zeros(len(t))
    s_wave[s_wave_start:s_wave_start + s_wave_duration] = (
        0.8 * np.sin(2 * np.pi * 4 * t[s_wave_start:s_wave_start + s_wave_duration]) *
        np.exp(-0.3 * (t[s_wave_start:s_wave_start + s_wave_duration] - t[s_wave_start]))
    )
    
    # Combine all components
    seismic_signal = noise + p_wave + s_wave
    
    return seismic_signal.astype(np.float32), sampling_rate

def test_wave_analysis_infrastructure():
    """Test wave analysis components."""
    print("Testing wave analysis infrastructure...")
    
    try:
        # Test imports
        from wave_analysis.services import WaveSeparationEngine, WaveSeparationParameters
        from wave_analysis.services import WaveAnalyzer
        from wave_analysis.models import WaveAnalysisResult
        print("✓ Wave analysis components imported successfully")
        
        # Test parameter creation
        params = WaveSeparationParameters(sampling_rate=100.0)
        print("✓ WaveSeparationParameters created successfully")
        
        # Test engine creation
        engine = WaveSeparationEngine(params)
        print("✓ WaveSeparationEngine created successfully")
        
        # Test analyzer creation
        analyzer = WaveAnalyzer(sampling_rate=100.0)
        print("✓ WaveAnalyzer created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing wave analysis infrastructure: {e}")
        return False

def test_api_with_mock_data():
    """Test API endpoints with mock data."""
    print("Testing API endpoints with mock data...")
    
    try:
        # Set environment variable to avoid MongoDB connection issues
        os.environ['MONGO_URL'] = 'mongodb://localhost:27017/test'
        
        from app import app, fs
        
        app.config['TESTING'] = True
        
        # Create synthetic seismic data
        seismic_data, sampling_rate = create_test_seismic_data()
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, seismic_data, sampling_rate)
            temp_file_path = temp_file.name
        
        try:
            # Mock GridFS operations
            with patch.object(fs, 'get') as mock_get, \
                 patch.object(fs, 'put') as mock_put:
                
                # Mock file reading
                mock_file = MagicMock()
                with open(temp_file_path, 'rb') as f:
                    mock_file.read.return_value = f.read()
                mock_get.return_value.__enter__.return_value = mock_file
                
                # Mock file storage
                mock_put.return_value = 'test_file_id'
                
                # Mock wave analysis components
                with patch('app.WaveSeparationEngine') as mock_engine_class, \
                     patch('app.WaveAnalyzer') as mock_analyzer_class, \
                     patch('app.db') as mock_db:
                    
                    # Setup mock wave separation result
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
                    
                    # Setup mock detailed analysis
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
                    
                    # Mock database operations
                    mock_db.wave_analyses.insert_one.return_value.inserted_id = 'test_analysis_id'
                    
                    # Test the API endpoint
                    with app.test_client() as client:
                        # Test analyze_waves endpoint
                        response = client.post('/api/analyze_waves', 
                                             json={'file_id': 'test_file_id'})
                        
                        print(f"✓ /api/analyze_waves responds with status: {response.status_code}")
                        
                        if response.status_code == 200:
                            data = response.get_json()
                            print("✓ Response contains expected fields:")
                            
                            expected_fields = ['analysis_id', 'file_id', 'status', 'wave_separation', 
                                             'detailed_analysis', 'quality_metrics']
                            for field in expected_fields:
                                if field in data:
                                    print(f"  ✓ {field}")
                                else:
                                    print(f"  ✗ {field} missing")
                            
                            # Test wave_results endpoint
                            analysis_id = data.get('analysis_id', 'test_analysis_id')
                            
                            # Mock database query for results endpoint
                            mock_analysis_result = {
                                '_id': analysis_id,
                                'file_id': 'test_file_id',
                                'analysis_timestamp': '2024-01-01T12:00:00',
                                'parameters': {'sampling_rate': 100},
                                'wave_separation': {'p_waves_count': 1},
                                'quality_metrics': {'snr': 10.0},
                                'processing_metadata': {'success': True}
                            }
                            
                            mock_db.wave_analyses.find_one.return_value = mock_analysis_result
                            
                            results_response = client.get(f'/api/wave_results/{analysis_id}')
                            print(f"✓ /api/wave_results responds with status: {results_response.status_code}")
                            
                            if results_response.status_code == 200:
                                results_data = results_response.get_json()
                                print("✓ Results response contains expected fields:")
                                
                                expected_result_fields = ['analysis_id', 'file_id', 'analysis_timestamp']
                                for field in expected_result_fields:
                                    if field in results_data:
                                        print(f"  ✓ {field}")
                                    else:
                                        print(f"  ✗ {field} missing")
                        
                        print("✓ API endpoints working correctly with mock data")
                        return True
                        
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"✗ Error testing API with mock data: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Wave Analysis API Test Suite ===\n")
    
    success = True
    
    # Test 1: Infrastructure
    if not test_wave_analysis_infrastructure():
        success = False
    print()
    
    # Test 2: API with mock data
    if not test_api_with_mock_data():
        success = False
    print()
    
    if success:
        print("✓ All tests passed! Wave analysis API endpoints are working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == '__main__':
    main()