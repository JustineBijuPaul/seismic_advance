"""
Cached wave analyzer that integrates caching with wave analysis operations.

This module provides a cached wrapper around wave analysis operations
to demonstrate cache integration and improve performance.
"""

import logging
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime

from .analysis_cache import AnalysisCacheManager, CacheDecorator
from .wave_analyzer import WaveAnalyzer
from .wave_separation_engine import WaveSeparationEngine, WaveSeparationParameters
from ..models.wave_models import WaveAnalysisResult, DetailedAnalysis
from ..interfaces import WaveAnalyzerInterface


class CachedWaveAnalyzer(WaveAnalyzerInterface):
    """
    Wave analyzer with integrated caching for improved performance.
    
    This class wraps the standard WaveAnalyzer with caching capabilities
    to avoid redundant computations for frequently accessed analyses.
    """
    
    def __init__(self, 
                 wave_analyzer: WaveAnalyzer,
                 wave_separation_engine: WaveSeparationEngine,
                 cache_manager: AnalysisCacheManager,
                 enable_caching: bool = True):
        """
        Initialize the cached wave analyzer.
        
        Args:
            wave_analyzer: Core wave analyzer instance
            wave_separation_engine: Wave separation engine instance
            cache_manager: Cache manager for storing results
            enable_caching: Whether to enable caching (for testing/debugging)
        """
        self.wave_analyzer = wave_analyzer
        self.wave_separation_engine = wave_separation_engine
        self.cache_manager = cache_manager
        self.enable_caching = enable_caching
        self.logger = logging.getLogger(__name__)
        
        # Cache invalidation tracking
        self._last_parameters: Dict[str, Any] = {}
    
    def _generate_parameters_hash(self, parameters: Dict[str, Any]) -> str:
        """
        Generate a hash for analysis parameters to detect changes.
        
        Args:
            parameters: Analysis parameters
            
        Returns:
            Parameter hash string
        """
        import hashlib
        import json
        
        # Sort parameters for consistent hashing
        sorted_params = dict(sorted(parameters.items()))
        param_string = json.dumps(sorted_params, sort_keys=True, default=str)
        return hashlib.md5(param_string.encode()).hexdigest()
    
    def _should_invalidate_cache(self, file_id: str, parameters: Dict[str, Any]) -> bool:
        """
        Check if cache should be invalidated due to parameter changes.
        
        Args:
            file_id: File identifier
            parameters: Current analysis parameters
            
        Returns:
            True if cache should be invalidated
        """
        if not self.enable_caching:
            return False
        
        cache_key = f"{file_id}_params"
        current_hash = self._generate_parameters_hash(parameters)
        
        if cache_key in self._last_parameters:
            last_hash = self._last_parameters[cache_key]
            if current_hash != last_hash:
                self.logger.info(f"Parameters changed for {file_id}, invalidating cache")
                self._last_parameters[cache_key] = current_hash
                return True
        else:
            self._last_parameters[cache_key] = current_hash
        
        return False
    
    def separate_waves_cached(self, 
                            file_id: str,
                            seismic_data: np.ndarray, 
                            sampling_rate: float,
                            parameters: Optional[WaveSeparationParameters] = None) -> WaveAnalysisResult:
        """
        Perform wave separation with caching.
        
        Args:
            file_id: Unique identifier for the seismic data file
            seismic_data: Raw seismic time series data
            sampling_rate: Sampling rate in Hz
            parameters: Wave separation parameters
            
        Returns:
            Wave analysis result with separated waves
        """
        if not self.enable_caching:
            return self.wave_separation_engine.separate_waves(seismic_data, sampling_rate, parameters)
        
        # Prepare parameters for caching
        cache_params = {
            'sampling_rate': sampling_rate,
            'data_shape': seismic_data.shape,
            'data_checksum': hash(seismic_data.tobytes()),
        }
        
        if parameters:
            cache_params.update({
                'p_wave_params': parameters.p_wave_params.__dict__ if parameters.p_wave_params else None,
                's_wave_params': parameters.s_wave_params.__dict__ if parameters.s_wave_params else None,
                'surface_wave_params': parameters.surface_wave_params.__dict__ if parameters.surface_wave_params else None,
            })
        
        # Check for parameter changes
        if self._should_invalidate_cache(file_id, cache_params):
            self.cache_manager.invalidate_cache(file_id=file_id)
        
        # Try to get from cache
        cached_result = self.cache_manager.get_cached_result(
            'wave_separation', file_id, cache_params
        )
        
        if cached_result is not None:
            self.logger.debug(f"Cache hit for wave separation: {file_id}")
            return cached_result
        
        # Perform wave separation
        self.logger.debug(f"Cache miss for wave separation: {file_id}")
        start_time = datetime.now()
        
        result = self.wave_separation_engine.separate_waves(seismic_data, sampling_rate, parameters)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Cache the result
        self.cache_manager.cache_analysis_result(
            'wave_separation', file_id, result, cache_params, 
            ttl_hours=48,  # Longer TTL for expensive operations
            cache_type='wave_separation'
        )
        
        self.logger.info(f"Wave separation completed in {processing_time:.3f}s for {file_id}")
        return result
    
    def analyze_waves(self, wave_result: WaveAnalysisResult, file_id: Optional[str] = None) -> DetailedAnalysis:
        """
        Perform detailed wave analysis with caching.
        
        Args:
            wave_result: Result from wave separation
            file_id: Optional file identifier for caching
            
        Returns:
            Detailed analysis results
        """
        if not self.enable_caching or file_id is None:
            return self.wave_analyzer.analyze_waves(wave_result)
        
        # Prepare parameters for caching
        cache_params = {
            'wave_counts': {
                'p_waves': len(wave_result.p_waves),
                's_waves': len(wave_result.s_waves),
                'surface_waves': len(wave_result.surface_waves)
            },
            'sampling_rate': wave_result.sampling_rate,
            'data_length': len(wave_result.original_data),
            'metadata': wave_result.metadata
        }
        
        # Try to get from cache
        cached_result = self.cache_manager.get_cached_result(
            'detailed_analysis', file_id, cache_params
        )
        
        if cached_result is not None:
            self.logger.debug(f"Cache hit for detailed analysis: {file_id}")
            return cached_result
        
        # Perform detailed analysis
        self.logger.debug(f"Cache miss for detailed analysis: {file_id}")
        start_time = datetime.now()
        
        result = self.wave_analyzer.analyze_waves(wave_result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Cache the result
        self.cache_manager.cache_analysis_result(
            'detailed_analysis', file_id, result, cache_params,
            ttl_hours=24,
            cache_type='detailed_analysis'
        )
        
        self.logger.info(f"Detailed analysis completed in {processing_time:.3f}s for {file_id}")
        return result
    
    def calculate_arrival_times(self, waves: Dict[str, List], file_id: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate arrival times with caching.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            file_id: Optional file identifier for caching
            
        Returns:
            Dictionary mapping wave types to arrival times
        """
        if not self.enable_caching or file_id is None:
            return self.wave_analyzer.calculate_arrival_times(waves)
        
        # Prepare parameters for caching
        cache_params = {
            'wave_counts': {wave_type: len(segments) for wave_type, segments in waves.items()},
            'wave_types': list(waves.keys())
        }
        
        # Try to get from cache
        cached_result = self.cache_manager.get_cached_result(
            'arrival_times', file_id, cache_params
        )
        
        if cached_result is not None:
            self.logger.debug(f"Cache hit for arrival times: {file_id}")
            return cached_result
        
        # Calculate arrival times
        self.logger.debug(f"Cache miss for arrival times: {file_id}")
        result = self.wave_analyzer.calculate_arrival_times(waves)
        
        # Cache the result
        self.cache_manager.cache_analysis_result(
            'arrival_times', file_id, result, cache_params,
            ttl_hours=12,
            cache_type='arrival_times'
        )
        
        return result
    
    def estimate_magnitude(self, waves: Dict[str, List], file_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Estimate magnitude with caching.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            file_id: Optional file identifier for caching
            
        Returns:
            List of magnitude estimates
        """
        if not self.enable_caching or file_id is None:
            return self.wave_analyzer.estimate_magnitude(waves)
        
        # Prepare parameters for caching
        cache_params = {
            'wave_counts': {wave_type: len(segments) for wave_type, segments in waves.items()},
            'wave_types': list(waves.keys())
        }
        
        # Try to get from cache
        cached_result = self.cache_manager.get_cached_result(
            'magnitude_estimate', file_id, cache_params
        )
        
        if cached_result is not None:
            self.logger.debug(f"Cache hit for magnitude estimate: {file_id}")
            return cached_result
        
        # Estimate magnitude
        self.logger.debug(f"Cache miss for magnitude estimate: {file_id}")
        result = self.wave_analyzer.estimate_magnitude(waves)
        
        # Cache the result
        self.cache_manager.cache_analysis_result(
            'magnitude_estimate', file_id, result, cache_params,
            ttl_hours=24,
            cache_type='magnitude_estimate'
        )
        
        return result
    
    def analyze_complete_workflow(self, 
                                file_id: str,
                                seismic_data: np.ndarray,
                                sampling_rate: float,
                                parameters: Optional[WaveSeparationParameters] = None) -> DetailedAnalysis:
        """
        Perform complete wave analysis workflow with caching at each step.
        
        Args:
            file_id: Unique identifier for the seismic data file
            seismic_data: Raw seismic time series data
            sampling_rate: Sampling rate in Hz
            parameters: Wave separation parameters
            
        Returns:
            Complete detailed analysis results
        """
        self.logger.info(f"Starting complete wave analysis workflow for {file_id}")
        
        # Step 1: Wave separation (cached)
        wave_result = self.separate_waves_cached(file_id, seismic_data, sampling_rate, parameters)
        
        # Step 2: Detailed analysis (cached)
        detailed_analysis = self.analyze_waves(wave_result, file_id)
        
        self.logger.info(f"Complete wave analysis workflow finished for {file_id}")
        return detailed_analysis
    
    def get_cache_statistics_for_file(self, file_id: str) -> Dict[str, Any]:
        """
        Get cache statistics for a specific file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Dictionary with cache statistics for the file
        """
        # Query cache collection for file-specific statistics
        pipeline = [
            {'$match': {'file_id': file_id}},
            {
                '$group': {
                    '_id': '$operation',
                    'cache_entries': {'$sum': 1},
                    'total_accesses': {'$sum': '$access_count'},
                    'avg_access_count': {'$avg': '$access_count'},
                    'last_access': {'$max': '$created_at'},
                    'total_size': {'$sum': '$data_size'}
                }
            }
        ]
        
        results = list(self.cache_manager.cache_collection.aggregate(pipeline))
        
        file_stats = {
            'file_id': file_id,
            'operations': {},
            'total_cache_entries': 0,
            'total_accesses': 0,
            'total_cache_size': 0
        }
        
        for result in results:
            operation = result['_id']
            file_stats['operations'][operation] = {
                'cache_entries': result['cache_entries'],
                'total_accesses': result['total_accesses'],
                'avg_access_count': result['avg_access_count'],
                'last_access': result['last_access'],
                'size_bytes': result['total_size']
            }
            
            file_stats['total_cache_entries'] += result['cache_entries']
            file_stats['total_accesses'] += result['total_accesses']
            file_stats['total_cache_size'] += result['total_size']
        
        return file_stats
    
    def clear_file_cache(self, file_id: str) -> int:
        """
        Clear all cached results for a specific file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Number of cache entries cleared
        """
        self.logger.info(f"Clearing cache for file: {file_id}")
        
        # Clear from parameter tracking
        keys_to_remove = [key for key in self._last_parameters.keys() if key.startswith(f"{file_id}_")]
        for key in keys_to_remove:
            del self._last_parameters[key]
        
        # Clear from cache manager
        return self.cache_manager.invalidate_cache(file_id=file_id)
    
    def enable_cache_warming(self, warming_service) -> None:
        """
        Enable cache warming for this analyzer.
        
        Args:
            warming_service: Cache warming service instance
        """
        # Register warming functions with the service
        def warm_wave_separation(file_id: str):
            """Warming function for wave separation."""
            # This would need actual file data retrieval implementation
            # For now, it's a placeholder
            return None
        
        def warm_detailed_analysis(file_id: str):
            """Warming function for detailed analysis."""
            # This would need actual analysis implementation
            # For now, it's a placeholder
            return None
        
        warming_functions = [warm_wave_separation, warm_detailed_analysis]
        warming_service.create_warming_functions = lambda: warming_functions
        
        self.logger.info("Cache warming enabled for wave analyzer")


# Decorator-based caching for individual functions
def cached_wave_operation(operation_name: str, ttl_hours: int = 24):
    """
    Decorator for caching individual wave analysis operations.
    
    Args:
        operation_name: Name of the operation for cache key generation
        ttl_hours: Time to live for cached results
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Check if caching is enabled and cache manager is available
            if (hasattr(self, 'cache_manager') and 
                hasattr(self, 'enable_caching') and 
                self.enable_caching):
                
                # Extract file_id if available
                file_id = kwargs.get('file_id')
                if not file_id and args:
                    # Try to extract from first argument if it has file_id attribute
                    if hasattr(args[0], 'file_id'):
                        file_id = args[0].file_id
                
                if file_id:
                    # Generate cache parameters
                    cache_params = {
                        'function': func.__name__,
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys())
                    }
                    
                    # Try cache first
                    cached_result = self.cache_manager.get_cached_result(
                        operation_name, file_id, cache_params
                    )
                    
                    if cached_result is not None:
                        return cached_result
                    
                    # Execute function
                    result = func(self, *args, **kwargs)
                    
                    # Cache result
                    self.cache_manager.cache_analysis_result(
                        operation_name, file_id, result, cache_params, ttl_hours
                    )
                    
                    return result
            
            # Fall back to normal execution
            return func(self, *args, **kwargs)
        
        return wrapper
    return decorator