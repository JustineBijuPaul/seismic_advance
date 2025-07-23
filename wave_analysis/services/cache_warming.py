"""
Cache warming strategies for wave analysis results.

This module provides intelligent cache warming strategies to preload
frequently accessed analysis results and improve system performance.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from pymongo.database import Database

from .analysis_cache import AnalysisCacheManager
from ..models.wave_models import WaveAnalysisResult, DetailedAnalysis


@dataclass
class WarmingStrategy:
    """Configuration for a cache warming strategy."""
    name: str
    description: str
    priority: int  # Higher priority strategies run first
    enabled: bool = True
    max_files: int = 50
    max_age_days: int = 30


class CacheWarmingService:
    """
    Service for implementing intelligent cache warming strategies.
    """
    
    def __init__(self, 
                 cache_manager: AnalysisCacheManager,
                 db: Database,
                 wave_analyzer: Any = None,
                 wave_separation_engine: Any = None):
        """
        Initialize the cache warming service.
        
        Args:
            cache_manager: Cache manager instance
            db: MongoDB database instance
            wave_analyzer: Wave analyzer service instance
            wave_separation_engine: Wave separation engine instance
        """
        self.cache_manager = cache_manager
        self.db = db
        self.wave_analyzer = wave_analyzer
        self.wave_separation_engine = wave_separation_engine
        self.logger = logging.getLogger(__name__)
        
        # Define warming strategies
        self.strategies = [
            WarmingStrategy(
                name="recent_files",
                description="Warm cache for recently uploaded files",
                priority=1,
                max_files=20,
                max_age_days=7
            ),
            WarmingStrategy(
                name="high_quality_analyses",
                description="Warm cache for high-quality analysis results",
                priority=2,
                max_files=30,
                max_age_days=14
            ),
            WarmingStrategy(
                name="frequently_accessed",
                description="Warm cache for frequently accessed files",
                priority=3,
                max_files=25,
                max_age_days=30
            ),
            WarmingStrategy(
                name="significant_events",
                description="Warm cache for significant earthquake events",
                priority=4,
                max_files=15,
                max_age_days=60
            ),
            WarmingStrategy(
                name="educational_examples",
                description="Warm cache for educational example files",
                priority=5,
                max_files=10,
                max_age_days=90
            )
        ]
    
    def get_recent_files(self, strategy: WarmingStrategy) -> List[str]:
        """
        Get recently uploaded files for cache warming.
        
        Args:
            strategy: Warming strategy configuration
            
        Returns:
            List of file IDs
        """
        cutoff_date = datetime.now() - timedelta(days=strategy.max_age_days)
        
        pipeline = [
            {
                '$match': {
                    'uploadDate': {'$gte': cutoff_date}
                }
            },
            {
                '$sort': {'uploadDate': -1}
            },
            {
                '$limit': strategy.max_files
            },
            {
                '$project': {'_id': 1}
            }
        ]
        
        results = list(self.db.fs.files.aggregate(pipeline))
        return [str(doc['_id']) for doc in results]
    
    def get_high_quality_analyses(self, strategy: WarmingStrategy) -> List[str]:
        """
        Get files with high-quality analysis results.
        
        Args:
            strategy: Warming strategy configuration
            
        Returns:
            List of file IDs
        """
        cutoff_date = datetime.now() - timedelta(days=strategy.max_age_days)
        
        pipeline = [
            {
                '$match': {
                    'analysis_timestamp': {'$gte': cutoff_date},
                    'processing_metadata.quality_score': {'$gte': 0.8}
                }
            },
            {
                '$sort': {'processing_metadata.quality_score': -1}
            },
            {
                '$limit': strategy.max_files
            },
            {
                '$project': {'file_id': 1}
            }
        ]
        
        results = list(self.db.wave_analyses.aggregate(pipeline))
        return [str(doc['file_id']) for doc in results]
    
    def get_frequently_accessed_files(self, strategy: WarmingStrategy) -> List[str]:
        """
        Get frequently accessed files based on cache statistics.
        
        Args:
            strategy: Warming strategy configuration
            
        Returns:
            List of file IDs
        """
        cutoff_date = datetime.now() - timedelta(days=strategy.max_age_days)
        
        pipeline = [
            {
                '$match': {
                    'created_at': {'$gte': cutoff_date}
                }
            },
            {
                '$group': {
                    '_id': '$file_id',
                    'total_accesses': {'$sum': '$access_count'},
                    'last_access': {'$max': '$created_at'}
                }
            },
            {
                '$sort': {'total_accesses': -1}
            },
            {
                '$limit': strategy.max_files
            }
        ]
        
        results = list(self.cache_manager.cache_collection.aggregate(pipeline))
        return [doc['_id'] for doc in results if doc['_id']]
    
    def get_significant_events(self, strategy: WarmingStrategy) -> List[str]:
        """
        Get files containing significant earthquake events.
        
        Args:
            strategy: Warming strategy configuration
            
        Returns:
            List of file IDs
        """
        cutoff_date = datetime.now() - timedelta(days=strategy.max_age_days)
        
        pipeline = [
            {
                '$match': {
                    'analysis_timestamp': {'$gte': cutoff_date},
                    'detailed_analysis.magnitude_estimates.magnitude': {'$gte': 4.0}
                }
            },
            {
                '$sort': {'detailed_analysis.magnitude_estimates.magnitude': -1}
            },
            {
                '$limit': strategy.max_files
            },
            {
                '$project': {'file_id': 1}
            }
        ]
        
        results = list(self.db.wave_analyses.aggregate(pipeline))
        return [str(doc['file_id']) for doc in results]
    
    def get_educational_examples(self, strategy: WarmingStrategy) -> List[str]:
        """
        Get educational example files for cache warming.
        
        Args:
            strategy: Warming strategy configuration
            
        Returns:
            List of file IDs
        """
        # Look for files marked as educational or with specific metadata
        pipeline = [
            {
                '$match': {
                    '$or': [
                        {'metadata.educational': True},
                        {'metadata.example': True},
                        {'filename': {'$regex': 'example|demo|tutorial', '$options': 'i'}}
                    ]
                }
            },
            {
                '$sort': {'uploadDate': -1}
            },
            {
                '$limit': strategy.max_files
            },
            {
                '$project': {'_id': 1}
            }
        ]
        
        results = list(self.db.fs.files.aggregate(pipeline))
        return [str(doc['_id']) for doc in results]
    
    def create_warming_functions(self) -> List[Callable]:
        """
        Create warming functions for different analysis operations.
        
        Returns:
            List of warming functions
        """
        warming_functions = []
        
        if self.wave_separation_engine:
            def warm_wave_separation(file_id: str) -> Optional[WaveAnalysisResult]:
                """Warm cache for wave separation operation."""
                try:
                    # Check if already cached
                    cached = self.cache_manager.get_cached_result(
                        'wave_separation', file_id
                    )
                    if cached:
                        return cached
                    
                    # Get file data
                    file_data = self._get_file_data(file_id)
                    if file_data is None:
                        return None
                    
                    seismic_data, sampling_rate = file_data
                    
                    # Perform wave separation
                    result = self.wave_separation_engine.separate_waves(
                        seismic_data, sampling_rate
                    )
                    
                    # Cache the result
                    self.cache_manager.cache_analysis_result(
                        'wave_separation', file_id, result
                    )
                    
                    return result
                    
                except Exception as e:
                    self.logger.warning(f"Failed to warm wave separation for {file_id}: {e}")
                    return None
            
            warming_functions.append(warm_wave_separation)
        
        if self.wave_analyzer:
            def warm_detailed_analysis(file_id: str) -> Optional[DetailedAnalysis]:
                """Warm cache for detailed analysis operation."""
                try:
                    # Check if already cached
                    cached = self.cache_manager.get_cached_result(
                        'detailed_analysis', file_id
                    )
                    if cached:
                        return cached
                    
                    # Get wave separation result (may be cached)
                    wave_result = self.cache_manager.get_cached_result(
                        'wave_separation', file_id
                    )
                    
                    if wave_result is None:
                        # Need to perform wave separation first
                        file_data = self._get_file_data(file_id)
                        if file_data is None:
                            return None
                        
                        seismic_data, sampling_rate = file_data
                        wave_result = self.wave_separation_engine.separate_waves(
                            seismic_data, sampling_rate
                        )
                    
                    # Perform detailed analysis
                    detailed_result = self.wave_analyzer.analyze_waves(wave_result)
                    
                    # Cache the result
                    self.cache_manager.cache_analysis_result(
                        'detailed_analysis', file_id, detailed_result
                    )
                    
                    return detailed_result
                    
                except Exception as e:
                    self.logger.warning(f"Failed to warm detailed analysis for {file_id}: {e}")
                    return None
            
            warming_functions.append(warm_detailed_analysis)
        
        def warm_frequency_analysis(file_id: str) -> Optional[Dict[str, Any]]:
            """Warm cache for frequency analysis operation."""
            try:
                # Check if already cached
                cached = self.cache_manager.get_cached_result(
                    'frequency_analysis', file_id
                )
                if cached:
                    return cached
                
                # Get detailed analysis (may be cached)
                detailed_analysis = self.cache_manager.get_cached_result(
                    'detailed_analysis', file_id
                )
                
                if detailed_analysis and detailed_analysis.frequency_analysis:
                    # Cache the frequency analysis separately
                    self.cache_manager.cache_analysis_result(
                        'frequency_analysis', file_id, detailed_analysis.frequency_analysis
                    )
                    return detailed_analysis.frequency_analysis
                
                return None
                
            except Exception as e:
                self.logger.warning(f"Failed to warm frequency analysis for {file_id}: {e}")
                return None
        
        warming_functions.append(warm_frequency_analysis)
        
        return warming_functions
    
    def _get_file_data(self, file_id: str) -> Optional[tuple]:
        """
        Get seismic data and sampling rate for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Tuple of (seismic_data, sampling_rate) or None if not found
        """
        try:
            from bson import ObjectId
            import gridfs
            
            fs = gridfs.GridFS(self.db)
            
            # Get file
            file_obj = fs.get(ObjectId(file_id))
            
            # This is a simplified version - in practice, you'd need to
            # handle different file formats and extract the actual seismic data
            # For now, we'll assume the data is stored in a specific format
            
            # Read file content
            file_content = file_obj.read()
            
            # Parse based on file type (this would need to be implemented
            # based on your specific file format handling)
            # For now, return None to indicate we need actual implementation
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get file data for {file_id}: {e}")
            return None
    
    def execute_warming_strategy(self, strategy: WarmingStrategy) -> Dict[str, Any]:
        """
        Execute a specific warming strategy.
        
        Args:
            strategy: Warming strategy to execute
            
        Returns:
            Dictionary with execution results
        """
        if not strategy.enabled:
            return {'strategy': strategy.name, 'status': 'disabled', 'files_warmed': 0}
        
        self.logger.info(f"Executing warming strategy: {strategy.name}")
        
        # Get file IDs based on strategy
        file_ids = []
        
        if strategy.name == "recent_files":
            file_ids = self.get_recent_files(strategy)
        elif strategy.name == "high_quality_analyses":
            file_ids = self.get_high_quality_analyses(strategy)
        elif strategy.name == "frequently_accessed":
            file_ids = self.get_frequently_accessed_files(strategy)
        elif strategy.name == "significant_events":
            file_ids = self.get_significant_events(strategy)
        elif strategy.name == "educational_examples":
            file_ids = self.get_educational_examples(strategy)
        
        if not file_ids:
            return {
                'strategy': strategy.name,
                'status': 'no_files_found',
                'files_warmed': 0
            }
        
        # Create warming functions
        warming_functions = self.create_warming_functions()
        
        if not warming_functions:
            return {
                'strategy': strategy.name,
                'status': 'no_warming_functions',
                'files_warmed': 0
            }
        
        # Execute warming
        warming_result = self.cache_manager.warm_cache(
            warming_functions, file_ids, max_concurrent=3
        )
        
        return {
            'strategy': strategy.name,
            'status': 'completed',
            'files_processed': len(file_ids),
            'files_warmed': warming_result['operations_successful'],
            'warming_stats': warming_result
        }
    
    def execute_all_strategies(self) -> Dict[str, Any]:
        """
        Execute all enabled warming strategies in priority order.
        
        Returns:
            Dictionary with overall execution results
        """
        self.logger.info("Starting comprehensive cache warming")
        
        # Sort strategies by priority
        sorted_strategies = sorted(self.strategies, key=lambda x: x.priority)
        
        overall_results = {
            'total_strategies': len(sorted_strategies),
            'strategies_executed': 0,
            'total_files_warmed': 0,
            'strategy_results': [],
            'start_time': datetime.now(),
            'end_time': None,
            'duration_seconds': 0
        }
        
        for strategy in sorted_strategies:
            try:
                result = self.execute_warming_strategy(strategy)
                overall_results['strategy_results'].append(result)
                
                if result['status'] == 'completed':
                    overall_results['strategies_executed'] += 1
                    overall_results['total_files_warmed'] += result['files_warmed']
                
            except Exception as e:
                self.logger.error(f"Failed to execute strategy {strategy.name}: {e}")
                overall_results['strategy_results'].append({
                    'strategy': strategy.name,
                    'status': 'error',
                    'error': str(e),
                    'files_warmed': 0
                })
        
        overall_results['end_time'] = datetime.now()
        overall_results['duration_seconds'] = (
            overall_results['end_time'] - overall_results['start_time']
        ).total_seconds()
        
        self.logger.info(f"Cache warming completed: {overall_results['total_files_warmed']} files warmed")
        return overall_results
    
    def schedule_warming(self, interval_hours: int = 6) -> None:
        """
        Schedule periodic cache warming.
        
        Args:
            interval_hours: Interval between warming operations in hours
        """
        from apscheduler.schedulers.background import BackgroundScheduler
        
        scheduler = BackgroundScheduler()
        
        def warming_job():
            """Background job for cache warming."""
            try:
                self.execute_all_strategies()
            except Exception as e:
                self.logger.error(f"Scheduled cache warming failed: {e}")
        
        scheduler.add_job(
            warming_job,
            'interval',
            hours=interval_hours,
            id='cache_warming',
            replace_existing=True
        )
        
        scheduler.start()
        self.logger.info(f"Scheduled cache warming every {interval_hours} hours")
    
    def get_warming_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations for cache warming based on system usage patterns.
        
        Returns:
            List of warming recommendations
        """
        recommendations = []
        
        # Analyze cache hit rates
        cache_stats = self.cache_manager.get_cache_statistics()
        hit_rate = cache_stats['performance']['hit_rate']
        
        if hit_rate < 0.5:
            recommendations.append({
                'type': 'low_hit_rate',
                'priority': 'high',
                'message': f"Cache hit rate is low ({hit_rate:.2%}). Consider more aggressive warming.",
                'suggested_action': 'Increase warming frequency and file count'
            })
        
        # Analyze frequently missed operations
        if cache_stats['performance']['total_misses'] > 100:
            recommendations.append({
                'type': 'high_miss_count',
                'priority': 'medium',
                'message': f"High number of cache misses ({cache_stats['performance']['total_misses']})",
                'suggested_action': 'Focus warming on most common operations'
            })
        
        # Check memory cache utilization
        memory_util = cache_stats['storage']['memory_cache']['utilization']
        if memory_util > 0.9:
            recommendations.append({
                'type': 'memory_pressure',
                'priority': 'medium',
                'message': f"Memory cache utilization is high ({memory_util:.1%})",
                'suggested_action': 'Consider increasing memory cache size or reducing TTL'
            })
        
        return recommendations