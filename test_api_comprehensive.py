#!/usr/bin/env python3

import os
import sys
import json
from unittest.mock import patch, MagicMock
from bson import ObjectId

# Mock MongoDB to avoid connection issues during testing
sys.modules['pymongo'] = MagicMock()
sys.modules['gridfs'] = MagicMock()

def test_api_endpoints():
    """Comprehensive test of API endpoints"""
    
    # Mock the database connections
    with patch('app.MongoClient') as mock_client, \
         patch('app.gridfs.GridFS') as mock_gridfs, \
         patch('app.db') as mock_db:
        
        mock_client.return_value = {'seismic_quake': mock_db}
        mock_gridfs.return_value = MagicMock()
        
        # Import the app after mocking
        from app import app
        
        app.config['TESTING'] = True
        
        with app.test_client() as client:
            print("=== Testing /api/analyze_waves endpoint ===")
            
            # Test 1: Missing JSON data
            print("Test 1: Missing JSON data")
            response = client.post('/api/analyze_waves', 
                                 headers={'Content-Type': 'application/json'})
            print(f"Status: {response.status_code}")
            if response.status_code == 400:
                data = response.get_json()
                print(f"Response: {data}")
                assert 'Invalid JSON data' in data['error'] or 'No JSON data provided' in data['error']
                print("✓ PASS")
            else:
                print("✗ FAIL")
            
            # Test 2: Missing file_id
            print("\nTest 2: Missing file_id")
            response = client.post('/api/analyze_waves', 
                                 json={},
                                 headers={'Content-Type': 'application/json'})
            print(f"Status: {response.status_code}")
            if response.status_code == 400:
                data = response.get_json()
                print(f"Response: {data}")
                assert data['error'] == 'file_id is required'
                print("✓ PASS")
            else:
                print("✗ FAIL")
            
            # Test 3: Invalid file_id
            print("\nTest 3: Invalid file_id")
            response = client.post('/api/analyze_waves', 
                                 json={'file_id': 'invalid_id'},
                                 headers={'Content-Type': 'application/json'})
            print(f"Status: {response.status_code}")
            if response.status_code == 400:
                data = response.get_json()
                print(f"Response: {data}")
                assert 'Failed to load seismic data' in data['error']
                print("✓ PASS")
            else:
                print("✗ FAIL")
            
            # Test 4: Wave analysis unavailable
            print("\nTest 4: Wave analysis unavailable")
            with patch('app.WAVE_ANALYSIS_AVAILABLE', False):
                response = client.post('/api/analyze_waves', 
                                     json={'file_id': 'test_id'},
                                     headers={'Content-Type': 'application/json'})
                print(f"Status: {response.status_code}")
                if response.status_code == 503:
                    data = response.get_json()
                    print(f"Response: {data}")
                    assert data['error'] == 'Wave analysis components not available'
                    print("✓ PASS")
                else:
                    print("✗ FAIL")
            
            print("\n=== Testing /api/wave_results endpoint ===")
            
            # Test 5: Invalid analysis_id format
            print("Test 5: Invalid analysis_id format")
            response = client.get('/api/wave_results/invalid_id')
            print(f"Status: {response.status_code}")
            if response.status_code == 400:
                data = response.get_json()
                print(f"Response: {data}")
                assert data['error'] == 'Invalid analysis_id format'
                print("✓ PASS")
            else:
                print("✗ FAIL")
            
            # Test 6: Analysis result not found
            print("\nTest 6: Analysis result not found")
            fake_id = str(ObjectId())
            mock_db.wave_analyses.find_one.return_value = None
            response = client.get(f'/api/wave_results/{fake_id}')
            print(f"Status: {response.status_code}")
            if response.status_code == 404:
                data = response.get_json()
                print(f"Response: {data}")
                assert data['error'] == 'Analysis result not found'
                print("✓ PASS")
            else:
                print("✗ FAIL")
            
            # Test 7: Wave analysis unavailable for results
            print("\nTest 7: Wave analysis unavailable for results")
            with patch('app.WAVE_ANALYSIS_AVAILABLE', False):
                response = client.get(f'/api/wave_results/{fake_id}')
                print(f"Status: {response.status_code}")
                if response.status_code == 503:
                    data = response.get_json()
                    print(f"Response: {data}")
                    assert data['error'] == 'Wave analysis components not available'
                    print("✓ PASS")
                else:
                    print("✗ FAIL")
            
            # Test 8: Successful results retrieval
            print("\nTest 8: Successful results retrieval")
            analysis_id = ObjectId()
            from datetime import datetime
            mock_analysis = {
                '_id': analysis_id,
                'file_id': 'test_file_id',
                'analysis_timestamp': datetime(2024, 1, 1, 12, 0, 0),
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
            
            mock_db.wave_analyses.find_one.return_value = mock_analysis
            response = client.get(f'/api/wave_results/{str(analysis_id)}')
            print(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"Response keys: {list(data.keys())}")
                assert data['analysis_id'] == str(analysis_id)
                assert data['file_id'] == 'test_file_id'
                assert 'wave_separation' in data
                assert 'quality_metrics' in data
                print("✓ PASS")
            else:
                print("✗ FAIL")
            
            print("\n=== All API tests completed! ===")

if __name__ == '__main__':
    test_api_endpoints()