"""
Analysis result caching service for wave analysis.

This module provides a comprehensive caching layer for wave analysis results
to improve performance and reduce redundant computations.
"""

import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import asdict
import numpy as np
from pymongo.database import Database
from pymongo.collection import Collection
import redis
from threading import Lock
import logging

from ..models.wave_models import (
    WaveAnalysisResult, DetailedAnalysis, WaveSegment,
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)


class AnalysisCacheManager:
    """
    Manages caching of wave analysis results with multiple storage backends
    and intelligent cache warming strategies.
    """
    
    def __init__(self, 
                 mongodb: Database,
                 redis_client: Optional[redis.Redis] = None,
                 default_ttl_hours: int = 24,
                 max_memory_cache_size: int = 100):
        """
        Initialize the cache manager.
        
        Args:
            mongodb: MongoDB database instance
            redis_client: Optional Redis client for distributed caching
            default_ttl_hours: Default time-to-live for cache entries in hours
            max_memory_cache_size: Maximum number of entries in memory cache
        """
        self.db = mongodb
        self.redis_client = redis_client
        self.default_ttl_hours = default_ttl_hours
        self.max_memory_cache_size = max_memory_cache_size
        
        # In-memory cache for frequently accessed items
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_cache_lock = Lock()
        self._access_counts: Dict[str, int] = {}
        
        # MongoDB collection for persistent cache
        self.cache_collection: Collection = self.db.analysis_cache
        
        # Ensure indexes
        self._ensure_indexes()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'mongodb_hits': 0,
            'redis_hits': 0,
            'invalidations': 0,
            'warming_operations': 0
        }
    
    def _ensure_indexes(self) -> None:
        """Create necessary indexes for cache collection."""
        self.cache_collection.create_index([
            ("cache_key", 1)
        ], unique=True, name="cache_key_unique_idx")
        
        self.cache_collection.create_index([
            ("expires_at", 1)
        ], expireAfterSeconds=0, name="cache_expiry_idx")
        
        self.cache_collection.create_index([
            ("created_at", 1)
        ], name="cache_created_idx")
        
        self.cache_collection.create_index([
            ("access_count", -1)
        ], name="cache_access_count_idx")
        
        self.cache_collection.create_index([
            ("cache_type", 1),
            ("file_id", 1)
        ], name="cache_type_file_idx")
    
    def _generate_cache_key(self, 
                          operation: str, 
                          file_id: str, 
                          parameters: Dict[str, Any] = None) -> str:
        """
        Generate a unique cache key for an operation.
        
        Args:
            operation: Type of operation (e.g., 'wave_separation', 'detailed_analysis')
            file_id: File identifier
            parameters: Operation parameters that affect the result
            
        Returns:
            Unique cache key string
        """
        key_data = {
            'operation': operation,
            'file_id': file_id,
            'parameters': parameters or {}
        }
        
        # Sort parameters for consistent key generation
        if isinstance(key_data['parameters'], dict):
            key_data['parameters'] = dict(sorted(key_data['parameters'].items()))
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _serialize_for_cache(self, data: Any) -> bytes:
        """
        Serialize data for cache storage.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized data as bytes
        """
        # Handle numpy arrays and dataclass objects
        if hasattr(data, '__dict__') and hasattr(data, '__dataclass_fields__'):
            # Convert dataclass to dict
            data = asdict(data)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_data = convert_numpy(data)
        return pickle.dumps(serializable_data)
    
    def _deserialize_from_cache(self, data: bytes) -> Any:
        """
        Deserialize data from cache storage.
        
        Args:
            data: Serialized data bytes
            
        Returns:
            Deserialized data
        """
        return pickle.loads(data)
    
    def _update_memory_cache(self, cache_key: str, data: Any, expires_at: datetime) -> None:
        """
        Update the in-memory cache with new data.
        
        Args:
            cache_key: Cache key
            data: Data to cache
            expires_at: Expiration timestamp
        """
        with self._memory_cache_lock:
            # Remove oldest entries if cache is full
            if len(self._memory_cache) >= self.max_memory_cache_size:
                # Remove least accessed entries
                sorted_keys = sorted(self._access_counts.items(), key=lambda x: x[1])
                keys_to_remove = [k for k, _ in sorted_keys[:10]]  # Remove 10 oldest
                
                for key in keys_to_remove:
                    self._memory_cache.pop(key, None)
                    self._access_counts.pop(key, None)
            
            self._memory_cache[cache_key] = {
                'data': data,
                'expires_at': expires_at,
                'cached_at': datetime.now()
            }
            self._access_counts[cache_key] = self._access_counts.get(cache_key, 0)
    
    def _get_from_memory_cache(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve data from memory cache.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached data or None if not found/expired
        """
        with self._memory_cache_lock:
            if cache_key not in self._memory_cache:
                return None
            
            entry = self._memory_cache[cache_key]
            
            # Check expiration
            if datetime.now() > entry['expires_at']:
                del self._memory_cache[cache_key]
                self._access_counts.pop(cache_key, None)
                return None
            
            # Update access count
            self._access_counts[cache_key] = self._access_counts.get(cache_key, 0) + 1
            self.stats['memory_hits'] += 1
            
            return entry['data']
    
    def cache_analysis_result(self, 
                            operation: str,
                            file_id: str,
                            result: Any,
                            parameters: Dict[str, Any] = None,
                            ttl_hours: Optional[int] = None,
                            cache_type: str = 'analysis') -> str:
        """
        Cache an analysis result.
        
        Args:
            operation: Operation type (e.g., 'wave_separation', 'detailed_analysis')
            file_id: File identifier
            result: Analysis result to cache
            parameters: Parameters that affect the result
            ttl_hours: Time to live in hours (uses default if None)
            cache_type: Type of cache entry for organization
            
        Returns:
            Cache key used for storage
        """
        cache_key = self._generate_cache_key(operation, file_id, parameters)
        ttl_hours = ttl_hours or self.default_ttl_hours
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        # Serialize data
        serialized_data = self._serialize_for_cache(result)
        
        # Store in memory cache
        self._update_memory_cache(cache_key, result, expires_at)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                redis_key = f"wave_analysis:{cache_key}"
                self.redis_client.setex(
                    redis_key,
                    int(ttl_hours * 3600),  # Convert to seconds
                    serialized_data
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache in Redis: {e}")
        
        # Store in MongoDB
        cache_doc = {
            'cache_key': cache_key,
            'operation': operation,
            'file_id': file_id,
            'parameters': parameters or {},
            'cache_type': cache_type,
            'data': serialized_data,
            'created_at': datetime.now(),
            'expires_at': expires_at,
            'access_count': 0,
            'data_size': len(serialized_data)
        }
        
        try:
            self.cache_collection.replace_one(
                {'cache_key': cache_key},
                cache_doc,
                upsert=True
            )
        except Exception as e:
            self.logger.error(f"Failed to cache in MongoDB: {e}")
        
        self.logger.debug(f"Cached {operation} result for file {file_id}")
        return cache_key
    
    def get_cached_result(self, 
                         operation: str,
                         file_id: str,
                         parameters: Dict[str, Any] = None) -> Optional[Any]:
        """
        Retrieve a cached analysis result.
        
        Args:
            operation: Operation type
            file_id: File identifier
            parameters: Parameters that affect the result
            
        Returns:
            Cached result or None if not found/expired
        """
        cache_key = self._generate_cache_key(operation, file_id, parameters)
        
        # Try memory cache first
        result = self._get_from_memory_cache(cache_key)
        if result is not None:
            self.stats['hits'] += 1
            return result
        
        # Try Redis cache
        if self.redis_client:
            try:
                redis_key = f"wave_analysis:{cache_key}"
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    result = self._deserialize_from_cache(cached_data)
                    # Update memory cache
                    expires_at = datetime.now() + timedelta(hours=self.default_ttl_hours)
                    self._update_memory_cache(cache_key, result, expires_at)
                    self.stats['hits'] += 1
                    self.stats['redis_hits'] += 1
                    return result
            except Exception as e:
                self.logger.warning(f"Failed to retrieve from Redis: {e}")
        
        # Try MongoDB cache
        try:
            doc = self.cache_collection.find_one({
                'cache_key': cache_key,
                'expires_at': {'$gt': datetime.now()}
            })
            
            if doc:
                result = self._deserialize_from_cache(doc['data'])
                
                # Update access count
                self.cache_collection.update_one(
                    {'cache_key': cache_key},
                    {'$inc': {'access_count': 1}}
                )
                
                # Update memory cache
                expires_at = doc['expires_at']
                self._update_memory_cache(cache_key, result, expires_at)
                
                self.stats['hits'] += 1
                self.stats['mongodb_hits'] += 1
                return result
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve from MongoDB cache: {e}")
        
        self.stats['misses'] += 1
        return None
    
    def invalidate_cache(self, 
                        operation: Optional[str] = None,
                        file_id: Optional[str] = None,
                        cache_pattern: Optional[str] = None) -> int:
        """
        Invalidate cached results based on criteria.
        
        Args:
            operation: Specific operation to invalidate (optional)
            file_id: Specific file to invalidate (optional)
            cache_pattern: Pattern to match cache keys (optional)
            
        Returns:
            Number of cache entries invalidated
        """
        invalidated_count = 0
        
        # Build MongoDB query
        query = {}
        if operation:
            query['operation'] = operation
        if file_id:
            query['file_id'] = file_id
        
        # Get cache keys to invalidate
        cache_keys = []
        if query:
            cursor = self.cache_collection.find(query, {'cache_key': 1})
            cache_keys = [doc['cache_key'] for doc in cursor]
        elif cache_pattern:
            # For pattern matching, we need to get all keys and filter
            cursor = self.cache_collection.find({}, {'cache_key': 1})
            cache_keys = [
                doc['cache_key'] for doc in cursor 
                if cache_pattern in doc['cache_key']
            ]
        
        # Remove from memory cache
        with self._memory_cache_lock:
            for key in cache_keys:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    self._access_counts.pop(key, None)
                    invalidated_count += 1
        
        # Remove from Redis
        if self.redis_client and cache_keys:
            try:
                redis_keys = [f"wave_analysis:{key}" for key in cache_keys]
                deleted = self.redis_client.delete(*redis_keys)
                invalidated_count += deleted
            except Exception as e:
                self.logger.warning(f"Failed to invalidate Redis cache: {e}")
        
        # Remove from MongoDB
        if query:
            result = self.cache_collection.delete_many(query)
            invalidated_count += result.deleted_count
        elif cache_keys:
            result = self.cache_collection.delete_many({
                'cache_key': {'$in': cache_keys}
            })
            invalidated_count += result.deleted_count
        
        self.stats['invalidations'] += invalidated_count
        self.logger.info(f"Invalidated {invalidated_count} cache entries")
        return invalidated_count
    
    def warm_cache(self, 
                   warming_functions: List[Callable],
                   file_ids: Optional[List[str]] = None,
                   max_concurrent: int = 5) -> Dict[str, Any]:
        """
        Warm the cache with commonly accessed analysis results.
        
        Args:
            warming_functions: List of functions that perform analysis operations
            file_ids: List of file IDs to warm cache for (optional)
            max_concurrent: Maximum concurrent warming operations
            
        Returns:
            Dictionary with warming statistics
        """
        import concurrent.futures
        import threading
        
        warming_stats = {
            'operations_attempted': 0,
            'operations_successful': 0,
            'operations_failed': 0,
            'cache_entries_created': 0,
            'total_time': 0
        }
        
        start_time = time.time()
        
        # Get file IDs to warm if not provided
        if file_ids is None:
            # Get most recently analyzed files
            recent_files = self.cache_collection.aggregate([
                {'$group': {'_id': '$file_id', 'last_access': {'$max': '$created_at'}}},
                {'$sort': {'last_access': -1}},
                {'$limit': 50}
            ])
            file_ids = [doc['_id'] for doc in recent_files]
        
        def warm_file(file_id: str, func: Callable) -> Dict[str, Any]:
            """Warm cache for a specific file and function."""
            try:
                result = func(file_id)
                if result:
                    return {'success': True, 'file_id': file_id, 'function': func.__name__}
                else:
                    return {'success': False, 'file_id': file_id, 'function': func.__name__, 'error': 'No result'}
            except Exception as e:
                return {'success': False, 'file_id': file_id, 'function': func.__name__, 'error': str(e)}
        
        # Execute warming operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            
            for file_id in file_ids:
                for func in warming_functions:
                    future = executor.submit(warm_file, file_id, func)
                    futures.append(future)
                    warming_stats['operations_attempted'] += 1
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result['success']:
                        warming_stats['operations_successful'] += 1
                        warming_stats['cache_entries_created'] += 1
                    else:
                        warming_stats['operations_failed'] += 1
                        self.logger.warning(f"Cache warming failed: {result}")
                except Exception as e:
                    warming_stats['operations_failed'] += 1
                    self.logger.error(f"Cache warming error: {e}")
        
        warming_stats['total_time'] = time.time() - start_time
        self.stats['warming_operations'] += warming_stats['operations_attempted']
        
        self.logger.info(f"Cache warming completed: {warming_stats}")
        return warming_stats
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache performance statistics
        """
        # MongoDB cache stats
        mongodb_stats = self.cache_collection.aggregate([
            {
                '$group': {
                    '_id': None,
                    'total_entries': {'$sum': 1},
                    'total_size': {'$sum': '$data_size'},
                    'avg_access_count': {'$avg': '$access_count'},
                    'total_accesses': {'$sum': '$access_count'}
                }
            }
        ])
        
        mongodb_result = list(mongodb_stats)
        if mongodb_result:
            mongodb_info = mongodb_result[0]
            del mongodb_info['_id']
        else:
            mongodb_info = {
                'total_entries': 0,
                'total_size': 0,
                'avg_access_count': 0,
                'total_accesses': 0
            }
        
        # Memory cache stats
        with self._memory_cache_lock:
            memory_info = {
                'entries': len(self._memory_cache),
                'max_size': self.max_memory_cache_size,
                'utilization': len(self._memory_cache) / self.max_memory_cache_size
            }
        
        # Hit rate calculation
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'performance': {
                'hit_rate': hit_rate,
                'total_hits': self.stats['hits'],
                'total_misses': self.stats['misses'],
                'memory_hits': self.stats['memory_hits'],
                'mongodb_hits': self.stats['mongodb_hits'],
                'redis_hits': self.stats['redis_hits'],
                'invalidations': self.stats['invalidations'],
                'warming_operations': self.stats['warming_operations']
            },
            'storage': {
                'memory_cache': memory_info,
                'mongodb_cache': mongodb_info,
                'redis_available': self.redis_client is not None
            },
            'configuration': {
                'default_ttl_hours': self.default_ttl_hours,
                'max_memory_cache_size': self.max_memory_cache_size
            }
        }
    
    def cleanup_expired_entries(self) -> int:
        """
        Clean up expired cache entries from all storage backends.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned_count = 0
        
        # Clean memory cache
        with self._memory_cache_lock:
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if datetime.now() > entry['expires_at']:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
                self._access_counts.pop(key, None)
                cleaned_count += 1
        
        # MongoDB cleanup is handled by TTL index
        # But we can manually clean up for immediate effect
        result = self.cache_collection.delete_many({
            'expires_at': {'$lt': datetime.now()}
        })
        cleaned_count += result.deleted_count
        
        self.logger.info(f"Cleaned up {cleaned_count} expired cache entries")
        return cleaned_count


class CacheDecorator:
    """
    Decorator for automatically caching function results.
    """
    
    def __init__(self, 
                 cache_manager: AnalysisCacheManager,
                 operation: str,
                 ttl_hours: int = 24,
                 cache_type: str = 'analysis'):
        """
        Initialize the cache decorator.
        
        Args:
            cache_manager: Cache manager instance
            operation: Operation name for cache key generation
            ttl_hours: Time to live for cached results
            cache_type: Type of cache entry
        """
        self.cache_manager = cache_manager
        self.operation = operation
        self.ttl_hours = ttl_hours
        self.cache_type = cache_type
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorate a function with caching.
        
        Args:
            func: Function to decorate
            
        Returns:
            Decorated function
        """
        def wrapper(*args, **kwargs):
            # Extract file_id and parameters for cache key
            file_id = None
            parameters = {}
            
            # Try to extract file_id from arguments
            if args:
                if isinstance(args[0], str):
                    file_id = args[0]
                elif hasattr(args[0], 'file_id'):
                    file_id = args[0].file_id
            
            if 'file_id' in kwargs:
                file_id = kwargs['file_id']
            
            # Use function arguments as parameters
            if len(args) > 1:
                parameters['args'] = args[1:]
            if kwargs:
                parameters['kwargs'] = kwargs
            
            if file_id:
                # Try to get from cache
                cached_result = self.cache_manager.get_cached_result(
                    self.operation, file_id, parameters
                )
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result if file_id is available
            if file_id and result is not None:
                self.cache_manager.cache_analysis_result(
                    self.operation, file_id, result, parameters, 
                    self.ttl_hours, self.cache_type
                )
            
            return result
        
        return wrapper