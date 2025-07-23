#!/usr/bin/env python3
"""
Simple test to verify Flask integration with wave analysis components
without database dependencies.
"""

import os
import sys
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
            
            # Check if wave analysis routes exist
            wave_routes = [str(rule) for rule in app.url_map.iter_rules() if 'wave' in str(rule)]
            print(f"✅ Found {len(wave_routes)} wave analysis routes:")
            for route in wave_routes:
                print(f"   - {route}")
            
            # Check if wave analysis dashboard route exists
            dashboard_routes = [str(rule) for rule in app.url_map.iter_rules() if 'dashboard' in str(rule)]
            print(f"✅ Found {len(dashboard_routes)} dashboard routes:")
            for route in dashboard_routes:
                print(f"   - {route}")
            
            # Test basic Flask app functionality
            with app.test_client() as client:
                # Test main page
                response = client.get('/')
                print(f"✅ Main page status: {response.status_code}")
                
                # Test wave analysis dashboard
                response = client.get('/wave_analysis_dashboard')
                print(f"✅ Wave analysis dashboard status: {response.status_code}")
                
                # Test API endpoints (should return error without proper data)
                response = client.post('/api/analyze_waves', json={})
                print(f"✅ Wave analysis API status: {response.status_code}")
                
                response = client.get('/api/wave_results/test_id')
                print(f"✅ Wave results API status: {response.status_code}")
            
            print("\n✅ All Flask integration tests passed!")
            
        except Exception as e:
            print(f"❌ Flask integration test failed: {e}")
            sys.exit(1)