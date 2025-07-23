"""
Tests for deployment configuration and system validation.
Ensures proper configuration management and deployment readiness.
"""

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from wave_analysis.config import (
    ConfigManager, WaveAnalysisConfig, DatabaseConfig, 
    LoggingConfig, DeploymentConfig, config_manager
)
from wave_analysis.logging_config import (
    WaveAnalysisLogger, HealthMonitor, performance_monitor
)


class TestConfigManager:
    """Test configuration management functionality."""
    
    def test_default_configuration_creation(self):
        """Test that default configuration is created properly."""
        config = ConfigManager()
        
        assert isinstance(config.wave_analysis, WaveAnalysisConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.deployment, DeploymentConfig)
        
        # Test default values (may be overridden by environment-specific config)
        assert config.wave_analysis.sampling_rate == 100.0
        assert config.wave_analysis.min_snr >= 2.0  # May be higher in some environments
        assert config.database.max_pool_size >= 50  # May vary by environment
        assert config.logging.level in ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    def test_environment_specific_configuration(self):
        """Test loading environment-specific configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_config.json"
            
            # Create test configuration
            test_config = {
                "wave_analysis": {
                    "sampling_rate": 200.0,
                    "min_snr": 5.0
                },
                "logging": {
                    "level": "DEBUG"
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(test_config, f)
            
            # Load configuration
            config = ConfigManager(str(config_file))
            
            assert config.wave_analysis.sampling_rate == 200.0
            assert config.wave_analysis.min_snr == 5.0
            assert config.logging.level == "DEBUG"
    
    @patch.dict(os.environ, {
        'WAVE_SAMPLING_RATE': '150.0',
        'WAVE_MIN_SNR': '3.5',
        'LOG_LEVEL': 'WARNING',
        'DEBUG': 'false'
    })
    def test_environment_variable_overrides(self):
        """Test that environment variables override configuration."""
        config = ConfigManager()
        
        assert config.wave_analysis.sampling_rate == 150.0
        assert config.wave_analysis.min_snr == 3.5
        assert config.logging.level == "WARNING"
        assert config.deployment.debug == False
    
    def test_configuration_validation(self):
        """Test configuration validation functionality."""
        config = ConfigManager()
        
        # Test valid configuration
        issues = config.validate_configuration()
        assert len(issues['errors']) == 0
        
        # Test invalid configuration
        config.wave_analysis.sampling_rate = -1
        config.wave_analysis.min_detection_confidence = 2.0
        config.database.max_pool_size = 5
        config.database.min_pool_size = 10
        
        issues = config.validate_configuration()
        assert len(issues['errors']) > 0
        assert any("Sampling rate must be positive" in error for error in issues['errors'])
        assert any("Detection confidence must be between 0 and 1" in error for error in issues['errors'])
        assert any("Max pool size must be >= min pool size" in error for error in issues['errors'])
    
    def test_configuration_save_and_load(self):
        """Test saving and loading configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "save_test.json"
            
            # Create and modify configuration
            config1 = ConfigManager(str(config_file))
            config1.wave_analysis.sampling_rate = 250.0
            config1.logging.level = "ERROR"
            config1.save_configuration()
            
            # Load configuration in new instance
            config2 = ConfigManager(str(config_file))
            
            assert config2.wave_analysis.sampling_rate == 250.0
            assert config2.logging.level == "ERROR"
    
    def test_get_parameter_methods(self):
        """Test parameter retrieval methods."""
        config = ConfigManager()
        
        wave_params = config.get_wave_analysis_params()
        assert isinstance(wave_params, dict)
        assert 'sampling_rate' in wave_params
        assert 'min_snr' in wave_params
        
        db_params = config.get_database_params()
        assert isinstance(db_params, dict)
        assert 'max_pool_size' in db_params
        
        logging_params = config.get_logging_params()
        assert isinstance(logging_params, dict)
        assert 'level' in logging_params
        
        deployment_params = config.get_deployment_params()
        assert isinstance(deployment_params, dict)
        assert 'environment' in deployment_params


class TestLoggingConfiguration:
    """Test logging configuration and functionality."""
    
    def test_logger_initialization(self):
        """Test that logger initializes properly."""
        logger = WaveAnalysisLogger()
        
        assert hasattr(logger, 'config')
        assert hasattr(logger, 'loggers')
        assert hasattr(logger, 'performance_metrics')
    
    def test_get_logger(self):
        """Test logger retrieval."""
        logger = WaveAnalysisLogger()
        
        component_logger = logger.get_logger('test_component')
        assert component_logger.name == 'wave_analysis.test_component'
    
    def test_performance_metric_logging(self):
        """Test performance metric logging."""
        logger = WaveAnalysisLogger()
        
        # Log a performance metric
        logger.log_performance_metric('test_operation', 1.5, {'test': 'data'})
        
        # Check that metric was stored
        assert 'test_operation' in logger.performance_metrics
        assert len(logger.performance_metrics['test_operation']) == 1
        assert logger.performance_metrics['test_operation'][0]['duration'] == 1.5
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        logger = WaveAnalysisLogger()
        
        # Add some test metrics
        logger.log_performance_metric('operation1', 1.0)
        logger.log_performance_metric('operation1', 2.0)
        logger.log_performance_metric('operation2', 0.5)
        
        # Test specific operation summary
        summary = logger.get_performance_summary('operation1')
        assert summary['count'] == 2
        assert summary['avg_duration'] == 1.5
        assert summary['min_duration'] == 1.0
        assert summary['max_duration'] == 2.0
        
        # Test overall summary
        overall_summary = logger.get_performance_summary()
        assert 'operation1' in overall_summary
        assert 'operation2' in overall_summary
    
    def test_alert_logging(self):
        """Test alert logging functionality."""
        logger = WaveAnalysisLogger()
        
        # This should not raise an exception even if alert logging is not set up
        logger.log_alert('test_alert', 'Test message', 'INFO', {'key': 'value'})
    
    @patch('wave_analysis.logging_config.wave_logger')
    def test_performance_monitor_decorator(self, mock_logger):
        """Test performance monitoring decorator."""
        mock_logger.get_logger.return_value = MagicMock()
        mock_logger.log_performance_metric = MagicMock()
        
        @performance_monitor('test_function')
        def test_function(x, y=None):
            return x + (y or 0)
        
        result = test_function(5, y=3)
        
        assert result == 8
        mock_logger.log_performance_metric.assert_called_once()
        
        # Check that the call included the operation name and success metadata
        call_args = mock_logger.log_performance_metric.call_args
        assert call_args[0][0] == 'test_function'  # operation name
        assert isinstance(call_args[0][1], float)  # duration
        assert call_args[0][2]['success'] == True  # metadata
    
    @patch('wave_analysis.logging_config.wave_logger')
    def test_performance_monitor_decorator_with_exception(self, mock_logger):
        """Test performance monitoring decorator with exceptions."""
        mock_logger.get_logger.return_value = MagicMock()
        mock_logger.log_performance_metric = MagicMock()
        
        @performance_monitor('failing_function')
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Check that the error was logged
        call_args = mock_logger.log_performance_metric.call_args
        assert call_args[0][0] == 'failing_function'
        assert call_args[0][2]['success'] == False
        assert 'error' in call_args[0][2]


class TestHealthMonitor:
    """Test system health monitoring functionality."""
    
    @patch('wave_analysis.logging_config.config_manager')
    def test_health_monitor_initialization(self, mock_config):
        """Test health monitor initialization."""
        mock_config.deployment = MagicMock()
        mock_config.validate_configuration.return_value = {'errors': [], 'warnings': []}
        
        monitor = HealthMonitor()
        
        assert hasattr(monitor, 'config')
        assert hasattr(monitor, 'logger')
        assert hasattr(monitor, 'health_status')
    
    @patch('wave_analysis.logging_config.config_manager')
    @patch('wave_analysis.logging_config.wave_logger')
    def test_system_health_check(self, mock_logger, mock_config):
        """Test comprehensive system health check."""
        # Mock configuration validation
        mock_config.validate_configuration.return_value = {
            'errors': [],
            'warnings': ['Test warning']
        }
        
        # Mock logger
        mock_logger.get_logger.return_value = MagicMock()
        mock_logger.get_performance_summary.return_value = {
            'test_operation': {'avg_duration': 1.0}
        }
        mock_logger.log_alert = MagicMock()
        
        monitor = HealthMonitor()
        
        with patch.object(monitor, '_check_disk_space') as mock_disk_check:
            mock_disk_check.return_value = {
                'status': 'healthy',
                'message': 'Sufficient disk space'
            }
            
            health_status = monitor.check_system_health()
        
        assert 'timestamp' in health_status
        assert 'overall_status' in health_status
        assert 'checks' in health_status
        assert 'configuration' in health_status['checks']
        assert 'logging' in health_status['checks']
        assert 'performance' in health_status['checks']
        assert 'disk_space' in health_status['checks']
    
    @patch('wave_analysis.logging_config.wave_logger')
    def test_logging_health_check(self, mock_logger):
        """Test logging system health check."""
        mock_test_logger = MagicMock()
        mock_logger.get_logger.return_value = mock_test_logger
        
        monitor = HealthMonitor()
        result = monitor._check_logging_health()
        
        assert result['status'] == 'healthy'
        mock_test_logger.info.assert_called_once()
    
    @patch('wave_analysis.logging_config.wave_logger')
    def test_performance_health_check(self, mock_logger):
        """Test performance health check."""
        # Test normal performance
        mock_logger.get_performance_summary.return_value = {
            'fast_operation': {'avg_duration': 1.0},
            'normal_operation': {'avg_duration': 5.0}
        }
        
        monitor = HealthMonitor()
        result = monitor._check_performance_health()
        
        assert result['status'] == 'healthy'
        
        # Test slow performance
        mock_logger.get_performance_summary.return_value = {
            'slow_operation': {'avg_duration': 35.0}
        }
        
        result = monitor._check_performance_health()
        
        assert result['status'] == 'warning'
        assert 'slow_operation' in result['slow_operations']
    
    @patch('wave_analysis.logging_config.config_manager')
    def test_disk_space_check(self, mock_config):
        """Test disk space health check."""
        mock_config.logging.log_file_path = '/tmp/test.log'
        
        monitor = HealthMonitor()
        
        # Test that the method runs without error (actual disk space checking is platform-specific)
        result = monitor._check_disk_space()
        
        # Should return a valid result structure
        assert 'status' in result
        assert 'message' in result
        assert result['status'] in ['healthy', 'warning', 'unhealthy']


class TestDeploymentValidation:
    """Test deployment-specific validation and configuration."""
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_production_configuration_validation(self):
        """Test that production configuration is secure and optimized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Load production configuration
            prod_config_path = Path(__file__).parent.parent / "wave_analysis" / "config" / "production.json"
            
            if prod_config_path.exists():
                config = ConfigManager(str(prod_config_path))
                
                # Validate production-specific settings
                assert config.deployment.environment == "production"
                assert config.deployment.debug == False
                assert config.wave_analysis.min_snr >= 3.0  # Higher threshold for production
                assert config.wave_analysis.min_detection_confidence >= 0.5
                assert config.database.max_pool_size >= 50  # Adequate for production load
    
    def test_development_configuration_validation(self):
        """Test that development configuration is suitable for debugging."""
        dev_config_path = Path(__file__).parent.parent / "wave_analysis" / "config" / "development.json"
        
        if dev_config_path.exists():
            config = ConfigManager(str(dev_config_path))
            
            # Validate development-specific settings
            assert config.deployment.environment == "development"
            assert config.deployment.debug == True
            assert config.logging.level == "DEBUG"
            assert config.wave_analysis.processing_timeout_seconds <= 120  # Faster feedback
    
    def test_configuration_file_existence(self):
        """Test that all required configuration files exist."""
        config_dir = Path(__file__).parent.parent / "wave_analysis" / "config"
        
        required_configs = ["development.json", "staging.json", "production.json"]
        
        for config_file in required_configs:
            config_path = config_dir / config_file
            assert config_path.exists(), f"Missing configuration file: {config_file}"
            
            # Validate JSON format
            with open(config_path) as f:
                config_data = json.load(f)
                assert isinstance(config_data, dict)
                assert "wave_analysis" in config_data
                assert "database" in config_data
                assert "logging" in config_data
                assert "deployment" in config_data
    
    @patch.dict(os.environ, {'ENVIRONMENT': 'production'})
    def test_production_environment_detection(self):
        """Test that production environment is detected correctly."""
        config = ConfigManager()
        assert config.environment == 'production'
        assert config.deployment.environment == 'production'
    
    def test_configuration_parameter_ranges(self):
        """Test that configuration parameters are within reasonable ranges."""
        config = ConfigManager()
        
        # Wave analysis parameters
        assert 50 <= config.wave_analysis.sampling_rate <= 1000
        assert 1.0 <= config.wave_analysis.min_snr <= 10.0
        assert 0.1 <= config.wave_analysis.min_detection_confidence <= 1.0
        assert 10 <= config.wave_analysis.max_file_size_mb <= 1000
        assert 30 <= config.wave_analysis.processing_timeout_seconds <= 600
        
        # Database parameters
        assert 1 <= config.database.min_pool_size <= config.database.max_pool_size
        assert config.database.max_pool_size <= 200
        assert 1000 <= config.database.connection_timeout_ms <= 30000
        
        # Logging parameters
        assert config.logging.level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        assert 1 <= config.logging.max_file_size_mb <= 100
        assert 1 <= config.logging.backup_count <= 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])