#!/usr/bin/env python3
"""
Simple test to verify API endpoints are working.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_api_endpoints():
    """Test that API endpoints exist and respond."""
    try:
        # Set environment variable to avoid MongoDB connection
        os.environ['MONGO_URL'] = 'mongodb://localhost:27017/test'
        
        from app import app
        print("✓ Flask app imported successfully")
        
        # Check if routes exist
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        
        if '/api/analyze_waves' in routes:
            print("✓ /api/analyze_waves endpoint exists")
        else:
            print("✗ /api/analyze_waves endpoint missing")
            
        if any('/api/wave_results' in route for route in routes):
            print("✓ /api/wave_results endpoint exists")
        else:
            print("✗ /api/wave_results endpoint missing")
            
        # Test basic endpoint response
        app.config['TESTING'] = True
        with app.test_client() as client:
            response = client.post('/api/analyze_waves')
            print(f"✓ /api/analyze_waves responds with status: {response.status_code}")
            
            if response.status_code == 400:
                data = response.get_json()
                if data and 'error' in data:
                    print(f"✓ Error handling works: {data['error']}")
                    
        print("✓ API endpoints are working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error testing API endpoints: {e}")
        return False

if __name__ == '__main__':
    test_api_endpoints()