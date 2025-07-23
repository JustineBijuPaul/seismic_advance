#!/usr/bin/env python3
"""
Deployment setup script for wave analysis system.
Validates configuration, sets up logging, and prepares the system for deployment.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available, environment variables from .env file won't be loaded")

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from wave_analysis.config import ConfigManager
from wave_analysis.logging_config import WaveAnalysisLogger, HealthMonitor


def setup_logging_directories(config: ConfigManager) -> bool:
    """Create necessary logging directories."""
    try:
        log_paths = [
            config.logging.log_file_path,
            config.logging.performance_log_file,
            config.logging.alert_log_file
        ]
        
        for log_path in log_paths:
            log_dir = Path(log_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created log directory: {log_dir}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create log directories: {e}")
        return False


def validate_environment_variables(environment: str) -> Dict[str, Any]:
    """Validate required environment variables for deployment."""
    validation_results = {
        'valid': True,
        'missing': [],
        'warnings': []
    }
    
    # Required environment variables by environment
    required_vars = {
        'development': ['MONGO_URL'],
        'staging': ['MONGO_URL', 'SECRET_KEY'],
        'production': ['MONGO_URL', 'SECRET_KEY', 'REDIS_URL']
    }
    
    # Optional but recommended variables
    recommended_vars = {
        'development': ['LOG_LEVEL'],
        'staging': ['LOG_LEVEL', 'WAVE_MIN_SNR'],
        'production': ['LOG_LEVEL', 'WAVE_MIN_SNR', 'DB_MAX_POOL_SIZE']
    }
    
    # Check required variables
    for var in required_vars.get(environment, []):
        if not os.getenv(var):
            validation_results['missing'].append(var)
            validation_results['valid'] = False
    
    # Check recommended variables
    for var in recommended_vars.get(environment, []):
        if not os.getenv(var):
            validation_results['warnings'].append(f"Recommended variable {var} not set")
    
    return validation_results


def validate_configuration(config: ConfigManager) -> bool:
    """Validate the configuration for deployment."""
    print("Validating configuration...")
    
    issues = config.validate_configuration()
    
    if issues['errors']:
        print("✗ Configuration validation failed:")
        for error in issues['errors']:
            print(f"  - ERROR: {error}")
        return False
    
    if issues['warnings']:
        print("⚠ Configuration warnings:")
        for warning in issues['warnings']:
            print(f"  - WARNING: {warning}")
    
    print("✓ Configuration validation passed")
    return True


def test_database_connection(config: ConfigManager) -> bool:
    """Test database connectivity."""
    try:
        from pymongo import MongoClient
        
        mongo_url = os.getenv('MONGO_URL')
        if not mongo_url:
            print("✗ MONGO_URL environment variable not set")
            return False
        
        # Test connection with configuration parameters
        client = MongoClient(
            mongo_url,
            connectTimeoutMS=config.database.connection_timeout_ms,
            socketTimeoutMS=config.database.socket_timeout_ms,
            maxPoolSize=config.database.max_pool_size,
            minPoolSize=config.database.min_pool_size
        )
        
        # Test basic operation
        client.admin.command('ping')
        print("✓ Database connection successful")
        return True
        
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def test_logging_system(logger: WaveAnalysisLogger) -> bool:
    """Test the logging system."""
    try:
        # Test basic logging
        test_logger = logger.get_logger('deployment_test')
        test_logger.info("Deployment test log entry")
        
        # Test performance logging
        logger.log_performance_metric('deployment_test', 0.1, {'test': True})
        
        # Test alert logging
        logger.log_alert('deployment_test', 'Test alert', 'INFO', {'test': True})
        
        print("✓ Logging system test passed")
        return True
        
    except Exception as e:
        print(f"✗ Logging system test failed: {e}")
        return False


def run_health_check(monitor: HealthMonitor) -> bool:
    """Run comprehensive health check."""
    try:
        health_status = monitor.check_system_health()
        
        print(f"System health status: {health_status['overall_status']}")
        
        for check_name, check_result in health_status['checks'].items():
            status_symbol = "✓" if check_result['status'] == 'healthy' else "⚠" if check_result['status'] == 'warning' else "✗"
            print(f"{status_symbol} {check_name}: {check_result['message']}")
        
        return health_status['overall_status'] in ['healthy', 'warning']
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def generate_deployment_report(config: ConfigManager, environment: str) -> Dict[str, Any]:
    """Generate a deployment readiness report."""
    report = {
        'environment': environment,
        'timestamp': config.deployment.__dict__.get('timestamp', 'unknown'),
        'configuration': {
            'wave_analysis': config.get_wave_analysis_params(),
            'database': config.get_database_params(),
            'logging': config.get_logging_params(),
            'deployment': config.get_deployment_params()
        },
        'validation_results': config.validate_configuration(),
        'ready_for_deployment': True
    }
    
    return report


def main():
    """Main deployment setup function."""
    parser = argparse.ArgumentParser(description='Setup deployment for wave analysis system')
    parser.add_argument('--environment', '-e', 
                       choices=['development', 'staging', 'production'],
                       default=os.getenv('ENVIRONMENT', 'development'),
                       help='Deployment environment')
    parser.add_argument('--config-file', '-c',
                       help='Custom configuration file path')
    parser.add_argument('--skip-db-test', action='store_true',
                       help='Skip database connection test')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate deployment report')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration without setup')
    
    args = parser.parse_args()
    
    print(f"🚀 Setting up deployment for {args.environment} environment")
    print("=" * 60)
    
    # Set environment variable
    os.environ['ENVIRONMENT'] = args.environment
    
    # Initialize configuration
    try:
        config = ConfigManager(args.config_file)
        print(f"✓ Configuration loaded for {args.environment}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate environment variables
    env_validation = validate_environment_variables(args.environment)
    if not env_validation['valid']:
        print("✗ Environment validation failed:")
        for missing_var in env_validation['missing']:
            print(f"  - Missing required variable: {missing_var}")
        sys.exit(1)
    
    for warning in env_validation['warnings']:
        print(f"⚠ {warning}")
    
    # Validate configuration
    if not validate_configuration(config):
        sys.exit(1)
    
    if args.validate_only:
        print("✓ Configuration validation completed")
        sys.exit(0)
    
    # Setup logging directories
    if not setup_logging_directories(config):
        sys.exit(1)
    
    # Initialize logging system
    try:
        logger = WaveAnalysisLogger()
        print("✓ Logging system initialized")
    except Exception as e:
        print(f"✗ Failed to initialize logging: {e}")
        sys.exit(1)
    
    # Test logging system
    if not test_logging_system(logger):
        sys.exit(1)
    
    # Test database connection
    if not args.skip_db_test:
        if not test_database_connection(config):
            sys.exit(1)
    
    # Run health check
    try:
        monitor = HealthMonitor()
        if not run_health_check(monitor):
            print("⚠ Health check completed with warnings")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        sys.exit(1)
    
    # Generate deployment report
    if args.generate_report:
        try:
            report = generate_deployment_report(config, args.environment)
            report_file = f"deployment_report_{args.environment}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"✓ Deployment report generated: {report_file}")
        except Exception as e:
            print(f"⚠ Failed to generate deployment report: {e}")
    
    print("=" * 60)
    print(f"🎉 Deployment setup completed successfully for {args.environment}")
    print("\nNext steps:")
    print("1. Review any warnings above")
    print("2. Test the application with: python app.py")
    print("3. Run tests with: python -m pytest tests/test_deployment_config.py")
    
    if args.environment == 'production':
        print("4. Configure reverse proxy (nginx/apache)")
        print("5. Set up process manager (systemd/supervisor)")
        print("6. Configure monitoring and alerting")


if __name__ == '__main__':
    main()