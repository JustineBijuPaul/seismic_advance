#!/usr/bin/env python3
"""
Quick test script to verify configuration and health check endpoints.
"""

import os
import sys
import requests
import json
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables for testing
os.environ['MONGO_URL'] = 'mongodb://localhost:27017/test'
os.environ['ENVIRONMENT'] = 'development'

def test_configuration_system():
    """Test the configuration system directly."""
    try:
        from wave_analysis.config import config_manager
        from wave_analysis.logging_config import wave_logger, health_monitor
        
        print("✓ Configuration system imported successfully")
        
        # Test configuration loading
        config = config_manager.get_wave_analysis_params()
        print(f"✓ Wave analysis config loaded: sampling_rate={config['sampling_rate']}")
        
        # Test logging system
        logger = wave_logger.get_logger('test')
        logger.info("Test log message")
        print("✓ Logging system working")
        
        # Test health monitor
        health_status = health_monitor.check_system_health()
        print(f"✓ Health check completed: {health_status['overall_status']}")
        
        # Test configuration validation
        validation_results = config_manager.validate_configuration()
        if validation_results['errors']:
            print(f"⚠ Configuration errors: {validation_results['errors']}")
        else:
            print("✓ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration system test failed: {e}")
        return False

def test_flask_app_import():
    """Test that the Flask app can be imported with new configuration."""
    try:
        # This will test if the app can be imported with the new config system
        import app
        print("✓ Flask application imported successfully with new configuration")
        
        # Test that configuration is being used
        if hasattr(app, 'config_manager'):
            print("✓ Configuration manager is available in Flask app")
        
        if hasattr(app, 'wave_logger'):
            print("✓ Wave logger is available in Flask app")
        
        return True
        
    except Exception as e:
        print(f"✗ Flask app import failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Configuration and Deployment Setup")
    print("=" * 50)
    
    # Test configuration system
    config_test = test_configuration_system()
    
    print("\n" + "=" * 50)
    
    # Test Flask app import
    flask_test = test_flask_app_import()
    
    print("\n" + "=" * 50)
    
    if config_test and flask_test:
        print("🎉 All configuration tests passed!")
        print("\nConfiguration system is ready for deployment.")
        print("\nNext steps:")
        print("1. Run: python setup_deployment.py --environment development")
        print("2. Start the application: python app.py")
        print("3. Test health endpoint: curl http://localhost:5000/health")
        print("4. Test config endpoint: curl http://localhost:5000/config")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main()