#!/usr/bin/env python3

import sys
import os

# Test basic imports
print("Testing basic imports...")
try:
    import numpy as np
    import flask
    from bson import ObjectId
    print("✓ Basic imports successful")
except ImportError as e:
    print(f"✗ Basic import failed: {e}")
    sys.exit(1)

# Test wave analysis imports
print("Testing wave analysis imports...")
try:
    from wave_analysis import WaveAnalysisResult, DetailedAnalysis
    from wave_analysis.services import WaveAnalyzer
    from wave_analysis.services.wave_separation_engine import WaveSeparationEngine, WaveSeparationParameters
    print("✓ Wave analysis imports successful")
    WAVE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"✗ Wave analysis import failed: {e}")
    WAVE_ANALYSIS_AVAILABLE = False

# Test Flask app creation
print("Testing Flask app...")
try:
    from app import app
    print("✓ Flask app imported successfully")
    
    # Test that the new routes exist
    with app.test_request_context():
        print("✓ Flask app context created")
        
        # Check if routes are registered
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        
        if '/api/analyze_waves' in routes:
            print("✓ /api/analyze_waves route registered")
        else:
            print("✗ /api/analyze_waves route not found")
            
        if '/api/wave_results/<analysis_id>' in routes:
            print("✓ /api/wave_results/<analysis_id> route registered")
        else:
            print("✗ /api/wave_results/<analysis_id> route not found")
            
except Exception as e:
    print(f"✗ Flask app test failed: {e}")

print(f"\nWAVE_ANALYSIS_AVAILABLE: {WAVE_ANALYSIS_AVAILABLE}")
print("Infrastructure test completed!")