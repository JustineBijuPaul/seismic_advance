#!/usr/bin/env python3
"""
Test upload workflow integration with wave analysis.
"""

import os
import sys
import tempfile
import numpy as np
import soundfile as sf
from unittest.mock import patch, MagicMock

# Mock MongoDB before importing app
with patch('pymongo.MongoClient') as mock_client, \
     patch('gridfs.GridFS') as mock_gridfs_class:
    
    mock_db = MagicMock()
    mock_client_instance = MagicMock()
    mock_client_instance.__getitem__.return_value = mock_db
    mock_client.return_value = mock_client_instance
    
    mock_gridfs = MagicMock()
    mock_gridfs_class.return_value = mock_gridfs
    
    # Mock environment variable
    with patch.dict(os.environ, {'MONGO_URL': 'mongodb://localhost:27017/test'}):
        try:
            from app import app
            print("✅ Flask app imported successfully")
            
            # Create test audio file
            def create_test_audio():
                duration = 10  # seconds
                sample_rate = 100  # Hz
                t = np.linspace(0, duration, duration * sample_rate)
                
                # Create synthetic seismic signal
                signal = np.random.normal(0, 0.1, len(t))  # noise
                
                # Add P-wave at 3 seconds
                p_start = int(3 * sample_rate)
                p_duration = int(1 * sample_rate)
                signal[p_start:p_start + p_duration] += 0.5 * np.sin(2 * np.pi * 8 * t[p_start:p_start + p_duration])
                
                # Add S-wave at 5 seconds
                s_start = int(5 * sample_rate)
                s_duration = int(2 * sample_rate)
                signal[s_start:s_start + s_duration] += 0.8 * np.sin(2 * np.pi * 4 * t[s_start:s_start + s_duration])
                
                return signal, sample_rate
            
            # Test upload workflow with wave analysis
            with app.test_client() as client:
                # Create temporary audio file
                signal, sample_rate = create_test_audio()
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    sf.write(temp_file.name, signal, sample_rate)
                    temp_file_path = temp_file.name
                
                try:
                    # Mock GridFS file storage
                    mock_gridfs.put.return_value = 'test_file_id_123'
                    
                    # Test upload without wave analysis
                    with open(temp_file_path, 'rb') as f:
                        response = client.post('/upload', data={
                            'file': (f, 'test_seismic.wav'),
                            'enable_wave_analysis': 'false'
                        })
                    
                    print(f"✅ Upload without wave analysis status: {response.status_code}")
                    if response.status_code == 200:
                        data = response.get_json()
                        print(f"   - Wave analysis enabled: {data.get('wave_analysis_enabled', False)}")
                    
                    # Test upload with wave analysis enabled
                    with open(temp_file_path, 'rb') as f:
                        response = client.post('/upload', data={
                            'file': (f, 'test_seismic.wav'),
                            'enable_wave_analysis': 'true'
                        })
                    
                    print(f"✅ Upload with wave analysis status: {response.status_code}")
                    if response.status_code == 200:
                        data = response.get_json()
                        print(f"   - Wave analysis enabled: {data.get('wave_analysis_enabled', False)}")
                        if 'wave_analysis' in data:
                            print(f"   - Wave analysis results included: ✅")
                            wave_analysis = data['wave_analysis']
                            print(f"   - P-waves detected: {wave_analysis.get('wave_separation', {}).get('p_waves_count', 0)}")
                            print(f"   - S-waves detected: {wave_analysis.get('wave_separation', {}).get('s_waves_count', 0)}")
                        elif 'wave_analysis_error' in data:
                            print(f"   - Wave analysis error: {data['wave_analysis_error']}")
                    
                    # Test async upload
                    with open(temp_file_path, 'rb') as f:
                        response = client.post('/upload', data={
                            'file': (f, 'test_seismic.wav'),
                            'enable_wave_analysis': 'true',
                            'async_processing': 'true'
                        })
                    
                    print(f"✅ Async upload with wave analysis status: {response.status_code}")
                    if response.status_code == 200:
                        data = response.get_json()
                        print(f"   - Async processing: {data.get('async', False)}")
                        print(f"   - Task ID: {data.get('task_id', 'N/A')}")
                    
                finally:
                    # Clean up
                    os.unlink(temp_file_path)
            
            print("\n✅ Upload integration tests completed!")
            
        except Exception as e:
            print(f"❌ Upload integration test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)