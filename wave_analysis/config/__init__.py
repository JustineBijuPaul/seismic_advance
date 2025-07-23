"""
Configuration package for wave analysis system.

This package provides configuration management for different deployment environments
including development, staging, and production configurations.
"""

from .config import (
    ConfigManager,
    WaveAnalysisConfig,
    DatabaseConfig,
    LoggingConfig,
    DeploymentConfig,
    config_manager
)

__all__ = [
    'ConfigManager',
    'WaveAnalysisConfig', 
    'DatabaseConfig',
    'LoggingConfig',
    'DeploymentConfig',
    'config_manager'
]

__version__ = '1.0.0'