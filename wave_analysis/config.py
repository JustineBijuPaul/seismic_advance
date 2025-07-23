"""
Configuration management for wave analysis parameters.
Supports environment-specific settings for development and production.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class WaveAnalysisConfig:
    """Configuration parameters for wave analysis operations."""
    
    # Signal Processing Parameters
    sampling_rate: float = 100.0
    min_snr: float = 2.0
    min_detection_confidence: float = 0.3
    
    # P-Wave Detection Parameters
    p_wave_sta_window: float = 0.5  # seconds
    p_wave_lta_window: float = 10.0  # seconds
    p_wave_trigger_ratio: float = 3.0
    p_wave_detrigger_ratio: float = 1.5
    
    # S-Wave Detection Parameters
    s_wave_sta_window: float = 1.0  # seconds
    s_wave_lta_window: float = 20.0  # seconds
    s_wave_trigger_ratio: float = 2.5
    s_wave_detrigger_ratio: float = 1.2
    
    # Surface Wave Detection Parameters
    surface_wave_min_period: float = 10.0  # seconds
    surface_wave_max_period: float = 100.0  # seconds
    surface_wave_min_group_velocity: float = 2.5  # km/s
    surface_wave_max_group_velocity: float = 4.5  # km/s
    
    # Analysis Parameters
    magnitude_estimation_methods: list = None
    frequency_analysis_window: float = 4.0  # seconds
    arrival_time_precision: float = 0.01  # seconds
    
    # Performance Parameters
    max_file_size_mb: int = 500
    processing_timeout_seconds: int = 300
    cache_ttl_seconds: int = 3600
    max_concurrent_analyses: int = 5
    
    # Alert System Parameters
    magnitude_alert_threshold: float = 4.0
    distance_alert_threshold_km: float = 100.0
    alert_cooldown_seconds: int = 60
    
    def __post_init__(self):
        if self.magnitude_estimation_methods is None:
            self.magnitude_estimation_methods = ['ML', 'Mb', 'Ms']


@dataclass
class DatabaseConfig:
    """Database configuration parameters."""
    
    connection_timeout_ms: int = 5000
    socket_timeout_ms: int = 30000
    max_pool_size: int = 100
    min_pool_size: int = 10
    max_idle_time_ms: int = 60000
    
    # GridFS Configuration
    gridfs_chunk_size: int = 261120  # 255KB chunks
    
    # Collection Names
    wave_analyses_collection: str = "wave_analyses"
    analysis_cache_collection: str = "analysis_cache"
    monitoring_results_collection: str = "monitoring_results"
    async_analyses_collection: str = "async_analyses"


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File Logging
    enable_file_logging: bool = True
    log_file_path: str = "logs/wave_analysis.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    # Performance Logging
    enable_performance_logging: bool = True
    performance_log_file: str = "logs/performance.log"
    
    # Alert Logging
    enable_alert_logging: bool = True
    alert_log_file: str = "logs/alerts.log"


@dataclass
class DeploymentConfig:
    """Deployment-specific configuration."""
    
    environment: str = "development"  # development, staging, production
    debug: bool = True
    
    # Security
    secret_key: str = "earthquake-analysis-secret-key"
    enable_cors: bool = True
    allowed_origins: list = None
    
    # Performance
    enable_caching: bool = True
    cache_backend: str = "memory"  # memory, redis
    redis_url: Optional[str] = None
    
    # Monitoring
    enable_metrics: bool = False
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["*"] if self.debug else []


class ConfigManager:
    """Manages configuration loading and environment-specific settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.config_file = config_file or self._get_default_config_file()
        
        # Initialize configurations
        self.wave_analysis = WaveAnalysisConfig()
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()
        self.deployment = DeploymentConfig()
        
        # Load configuration from file and environment
        self._load_configuration()
        self._apply_environment_overrides()
    
    def _get_default_config_file(self) -> str:
        """Get the default configuration file path."""
        config_dir = Path(__file__).parent / "config"
        config_dir.mkdir(exist_ok=True)
        return str(config_dir / f"{self.environment}.json")
    
    def _load_configuration(self):
        """Load configuration from file if it exists."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update configurations with loaded data
                if 'wave_analysis' in config_data:
                    self._update_dataclass(self.wave_analysis, config_data['wave_analysis'])
                
                if 'database' in config_data:
                    self._update_dataclass(self.database, config_data['database'])
                
                if 'logging' in config_data:
                    self._update_dataclass(self.logging, config_data['logging'])
                
                if 'deployment' in config_data:
                    self._update_dataclass(self.deployment, config_data['deployment'])
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logging.warning(f"Could not load configuration file {self.config_file}: {e}")
    
    def _update_dataclass(self, instance, data: Dict[str, Any]):
        """Update dataclass instance with dictionary data."""
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Wave Analysis overrides
        if os.getenv('WAVE_SAMPLING_RATE'):
            self.wave_analysis.sampling_rate = float(os.getenv('WAVE_SAMPLING_RATE'))
        
        if os.getenv('WAVE_MIN_SNR'):
            self.wave_analysis.min_snr = float(os.getenv('WAVE_MIN_SNR'))
        
        if os.getenv('WAVE_MIN_CONFIDENCE'):
            self.wave_analysis.min_detection_confidence = float(os.getenv('WAVE_MIN_CONFIDENCE'))
        
        # Database overrides
        if os.getenv('DB_CONNECTION_TIMEOUT'):
            self.database.connection_timeout_ms = int(os.getenv('DB_CONNECTION_TIMEOUT'))
        
        if os.getenv('DB_MAX_POOL_SIZE'):
            self.database.max_pool_size = int(os.getenv('DB_MAX_POOL_SIZE'))
        
        # Logging overrides
        if os.getenv('LOG_LEVEL'):
            self.logging.level = os.getenv('LOG_LEVEL')
        
        if os.getenv('LOG_FILE_PATH'):
            self.logging.log_file_path = os.getenv('LOG_FILE_PATH')
        
        # Deployment overrides
        self.deployment.environment = self.environment
        
        if os.getenv('SECRET_KEY'):
            self.deployment.secret_key = os.getenv('SECRET_KEY')
        
        if os.getenv('DEBUG'):
            self.deployment.debug = os.getenv('DEBUG').lower() == 'true'
        
        if os.getenv('REDIS_URL'):
            self.deployment.redis_url = os.getenv('REDIS_URL')
            self.deployment.cache_backend = 'redis'
    
    def save_configuration(self):
        """Save current configuration to file."""
        config_data = {
            'wave_analysis': asdict(self.wave_analysis),
            'database': asdict(self.database),
            'logging': asdict(self.logging),
            'deployment': asdict(self.deployment)
        }
        
        # Ensure config directory exists
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def get_wave_analysis_params(self) -> Dict[str, Any]:
        """Get wave analysis parameters as dictionary."""
        return asdict(self.wave_analysis)
    
    def get_database_params(self) -> Dict[str, Any]:
        """Get database parameters as dictionary."""
        return asdict(self.database)
    
    def get_logging_params(self) -> Dict[str, Any]:
        """Get logging parameters as dictionary."""
        return asdict(self.logging)
    
    def get_deployment_params(self) -> Dict[str, Any]:
        """Get deployment parameters as dictionary."""
        return asdict(self.deployment)
    
    def validate_configuration(self) -> Dict[str, list]:
        """Validate configuration parameters and return any issues."""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Validate wave analysis parameters
        if self.wave_analysis.sampling_rate <= 0:
            issues['errors'].append("Sampling rate must be positive")
        
        if self.wave_analysis.min_snr < 0:
            issues['errors'].append("Minimum SNR cannot be negative")
        
        if not (0 < self.wave_analysis.min_detection_confidence <= 1):
            issues['errors'].append("Detection confidence must be between 0 and 1")
        
        # Validate database parameters
        if self.database.max_pool_size < self.database.min_pool_size:
            issues['errors'].append("Max pool size must be >= min pool size")
        
        # Validate deployment parameters
        if self.deployment.environment not in ['development', 'staging', 'production']:
            issues['warnings'].append(f"Unknown environment: {self.deployment.environment}")
        
        if self.deployment.environment == 'production' and self.deployment.debug:
            issues['warnings'].append("Debug mode enabled in production environment")
        
        return issues


# Global configuration instance
config_manager = ConfigManager()