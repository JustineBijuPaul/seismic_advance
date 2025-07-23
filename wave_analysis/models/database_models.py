"""
Database models and data access layer for wave analysis results.

This module provides MongoDB integration for storing and retrieving
wave analysis results, with proper indexing and query optimization.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import asdict
import numpy as np
from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
import gridfs
import pickle
import json

from .wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis,
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)


class WaveAnalysisRepository:
    """
    Repository class for managing wave analysis data in MongoDB.
    
    Handles storage, retrieval, and querying of wave analysis results
    with proper indexing and performance optimization.
    """
    
    def __init__(self, db: Database, gridfs_instance: gridfs.GridFS):
        """
        Initialize the repository with database and GridFS instances.
        
        Args:
            db: MongoDB database instance
            gridfs_instance: GridFS instance for large data storage
        """
        self.db = db
        self.fs = gridfs_instance
        
        # Collection references
        self.wave_analyses = db.wave_analyses
        self.analysis_cache = db.analysis_cache
        
        # Ensure indexes are created
        self._ensure_indexes()
    
    def _ensure_indexes(self) -> None:
        """Create necessary indexes for efficient querying."""
        
        # Wave analyses collection indexes
        self.wave_analyses.create_index([
            ("file_id", ASCENDING),
            ("analysis_timestamp", DESCENDING)
        ], name="file_timestamp_idx")
        
        self.wave_analyses.create_index([
            ("analysis_timestamp", DESCENDING)
        ], name="timestamp_idx")
        
        self.wave_analyses.create_index([
            ("processing_metadata.quality_score", DESCENDING)
        ], name="quality_score_idx")
        
        self.wave_analyses.create_index([
            ("detailed_analysis.magnitude_estimates.magnitude", DESCENDING)
        ], name="magnitude_idx")
        
        self.wave_analyses.create_index([
            ("detailed_analysis.arrival_times.sp_time_difference", ASCENDING)
        ], name="sp_time_idx")
        
        # Analysis cache collection indexes
        self.analysis_cache.create_index([
            ("cache_key", ASCENDING)
        ], unique=True, name="cache_key_idx")
        
        self.analysis_cache.create_index([
            ("expires_at", ASCENDING)
        ], expireAfterSeconds=0, name="cache_expiry_idx")
    
    def _serialize_numpy_arrays(self, obj: Any) -> Any:
        """
        Recursively serialize numpy arrays to lists for MongoDB storage.
        
        Args:
            obj: Object that may contain numpy arrays
            
        Returns:
            Object with numpy arrays converted to lists
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._serialize_numpy_arrays(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy_arrays(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle dataclass objects
            if hasattr(obj, '__dataclass_fields__'):
                return self._serialize_numpy_arrays(asdict(obj))
            else:
                return self._serialize_numpy_arrays(obj.__dict__)
        else:
            return obj
    
    def _deserialize_numpy_arrays(self, obj: Any, array_fields: List[str] = None) -> Any:
        """
        Recursively deserialize lists back to numpy arrays.
        
        Args:
            obj: Object that may contain serialized arrays
            array_fields: List of field names that should be numpy arrays
            
        Returns:
            Object with lists converted back to numpy arrays
        """
        if array_fields is None:
            array_fields = ['data', 'original_data', 'frequencies', 'power_spectrum', 'array_field', 'another_array']
        
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key in array_fields and isinstance(value, list):
                    result[key] = np.array(value)
                else:
                    result[key] = self._deserialize_numpy_arrays(value, array_fields)
            return result
        elif isinstance(obj, list) and not any(isinstance(item, (dict, list)) for item in obj):
            # If it's a simple list of numbers, it might be a serialized array
            try:
                return np.array(obj)
            except:
                return [self._deserialize_numpy_arrays(item, array_fields) for item in obj]
        elif isinstance(obj, list):
            return [self._deserialize_numpy_arrays(item, array_fields) for item in obj]
        else:
            return obj
    
    def store_wave_analysis(self, 
                          file_id: Union[str, ObjectId], 
                          analysis_result: DetailedAnalysis) -> str:
        """
        Store a complete wave analysis result in the database.
        
        Args:
            file_id: GridFS file ID of the original seismic data
            analysis_result: Complete analysis result to store
            
        Returns:
            String ID of the stored analysis document
        """
        # Prepare document for storage
        doc = {
            'file_id': ObjectId(file_id) if isinstance(file_id, str) else file_id,
            'analysis_timestamp': analysis_result.analysis_timestamp,
            'wave_separation': {
                'p_waves': [self._serialize_numpy_arrays(asdict(wave)) 
                           for wave in analysis_result.wave_result.p_waves],
                's_waves': [self._serialize_numpy_arrays(asdict(wave)) 
                           for wave in analysis_result.wave_result.s_waves],
                'surface_waves': [self._serialize_numpy_arrays(asdict(wave)) 
                                 for wave in analysis_result.wave_result.surface_waves],
                'sampling_rate': analysis_result.wave_result.sampling_rate,
                'metadata': analysis_result.wave_result.metadata
            },
            'detailed_analysis': {
                'arrival_times': asdict(analysis_result.arrival_times),
                'magnitude_estimates': [asdict(est) for est in analysis_result.magnitude_estimates],
                'epicenter_distance': analysis_result.epicenter_distance,
                'frequency_analysis': {
                    wave_type: self._serialize_numpy_arrays(asdict(freq_data))
                    for wave_type, freq_data in analysis_result.frequency_analysis.items()
                },
                'quality_metrics': asdict(analysis_result.quality_metrics) if analysis_result.quality_metrics else None
            },
            'processing_metadata': {
                'processing_time': analysis_result.processing_metadata.get('processing_time'),
                'model_versions': analysis_result.processing_metadata.get('model_versions', {}),
                'quality_score': analysis_result.quality_metrics.analysis_quality_score if analysis_result.quality_metrics else None,
                'total_waves_detected': analysis_result.wave_result.total_waves_detected,
                'wave_types_detected': analysis_result.wave_result.wave_types_detected
            }
        }
        
        # Store large numpy arrays in GridFS if needed
        original_data = analysis_result.wave_result.original_data
        if len(original_data) > 1000000:  # Store large arrays separately
            data_id = self.fs.put(pickle.dumps(original_data), 
                                filename=f"original_data_{file_id}")
            doc['original_data_gridfs_id'] = data_id
        else:
            doc['original_data'] = original_data.tolist()
        
        # Insert document
        result = self.wave_analyses.insert_one(doc)
        return str(result.inserted_id)
    
    def get_wave_analysis(self, analysis_id: Union[str, ObjectId]) -> Optional[DetailedAnalysis]:
        """
        Retrieve a wave analysis result by ID.
        
        Args:
            analysis_id: ID of the analysis to retrieve
            
        Returns:
            DetailedAnalysis object or None if not found
        """
        doc = self.wave_analyses.find_one({
            '_id': ObjectId(analysis_id) if isinstance(analysis_id, str) else analysis_id
        })
        
        if not doc:
            return None
        
        return self._document_to_analysis(doc)
    
    def get_analyses_by_file(self, file_id: Union[str, ObjectId]) -> List[DetailedAnalysis]:
        """
        Get all wave analyses for a specific file.
        
        Args:
            file_id: GridFS file ID
            
        Returns:
            List of DetailedAnalysis objects
        """
        cursor = self.wave_analyses.find({
            'file_id': ObjectId(file_id) if isinstance(file_id, str) else file_id
        }).sort('analysis_timestamp', DESCENDING)
        
        return [self._document_to_analysis(doc) for doc in cursor]
    
    def get_recent_analyses(self, 
                          limit: int = 50, 
                          min_quality_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Get recent wave analyses with optional quality filtering.
        
        Args:
            limit: Maximum number of results to return
            min_quality_score: Minimum quality score filter
            
        Returns:
            List of analysis summary dictionaries
        """
        pipeline = [
            {
                '$match': {
                    'processing_metadata.quality_score': {'$gte': min_quality_score}
                }
            },
            {
                '$sort': {'analysis_timestamp': -1}
            },
            {
                '$limit': limit
            },
            {
                '$project': {
                    'file_id': 1,
                    'analysis_timestamp': 1,
                    'processing_metadata.quality_score': 1,
                    'processing_metadata.total_waves_detected': 1,
                    'processing_metadata.wave_types_detected': 1,
                    'detailed_analysis.magnitude_estimates': 1,
                    'detailed_analysis.arrival_times.sp_time_difference': 1
                }
            }
        ]
        
        return list(self.wave_analyses.aggregate(pipeline))
    
    def search_analyses_by_magnitude(self, 
                                   min_magnitude: float, 
                                   max_magnitude: float = None) -> List[Dict[str, Any]]:
        """
        Search analyses by magnitude range.
        
        Args:
            min_magnitude: Minimum magnitude
            max_magnitude: Maximum magnitude (optional)
            
        Returns:
            List of matching analysis summaries
        """
        match_filter = {
            'detailed_analysis.magnitude_estimates.magnitude': {'$gte': min_magnitude}
        }
        
        if max_magnitude is not None:
            match_filter['detailed_analysis.magnitude_estimates.magnitude']['$lte'] = max_magnitude
        
        pipeline = [
            {'$match': match_filter},
            {'$sort': {'analysis_timestamp': -1}},
            {
                '$project': {
                    'file_id': 1,
                    'analysis_timestamp': 1,
                    'detailed_analysis.magnitude_estimates': 1,
                    'detailed_analysis.arrival_times': 1,
                    'processing_metadata.quality_score': 1
                }
            }
        ]
        
        return list(self.wave_analyses.aggregate(pipeline))
    
    def delete_analysis(self, analysis_id: Union[str, ObjectId]) -> bool:
        """
        Delete a wave analysis and associated GridFS data.
        
        Args:
            analysis_id: ID of the analysis to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        # Get document first to check for GridFS references
        doc = self.wave_analyses.find_one({
            '_id': ObjectId(analysis_id) if isinstance(analysis_id, str) else analysis_id
        })
        
        if not doc:
            return False
        
        # Delete GridFS data if present
        if 'original_data_gridfs_id' in doc:
            try:
                self.fs.delete(doc['original_data_gridfs_id'])
            except gridfs.errors.NoFile:
                pass  # File already deleted or doesn't exist
        
        # Delete the analysis document
        result = self.wave_analyses.delete_one({
            '_id': ObjectId(analysis_id) if isinstance(analysis_id, str) else analysis_id
        })
        
        return result.deleted_count > 0
    
    def _document_to_analysis(self, doc: Dict[str, Any]) -> DetailedAnalysis:
        """
        Convert a MongoDB document back to a DetailedAnalysis object.
        
        Args:
            doc: MongoDB document
            
        Returns:
            DetailedAnalysis object
        """
        # Reconstruct original data
        if 'original_data_gridfs_id' in doc:
            with self.fs.get(doc['original_data_gridfs_id']) as f:
                original_data = pickle.loads(f.read())
        else:
            original_data = np.array(doc['original_data'])
        
        # Reconstruct wave segments
        def reconstruct_wave_segments(wave_list: List[Dict]) -> List[WaveSegment]:
            segments = []
            for wave_dict in wave_list:
                wave_dict = self._deserialize_numpy_arrays(wave_dict)
                segments.append(WaveSegment(**wave_dict))
            return segments
        
        # Reconstruct WaveAnalysisResult
        wave_sep = doc['wave_separation']
        wave_result = WaveAnalysisResult(
            original_data=original_data,
            sampling_rate=wave_sep['sampling_rate'],
            p_waves=reconstruct_wave_segments(wave_sep['p_waves']),
            s_waves=reconstruct_wave_segments(wave_sep['s_waves']),
            surface_waves=reconstruct_wave_segments(wave_sep['surface_waves']),
            metadata=wave_sep['metadata'],
            processing_timestamp=doc['analysis_timestamp']
        )
        
        # Reconstruct detailed analysis components
        detailed = doc['detailed_analysis']
        
        arrival_times = ArrivalTimes(**detailed['arrival_times'])
        
        magnitude_estimates = [
            MagnitudeEstimate(**est_dict) 
            for est_dict in detailed['magnitude_estimates']
        ]
        
        frequency_analysis = {}
        for wave_type, freq_dict in detailed['frequency_analysis'].items():
            freq_dict = self._deserialize_numpy_arrays(freq_dict)
            frequency_analysis[wave_type] = FrequencyData(**freq_dict)
        
        quality_metrics = None
        if detailed['quality_metrics']:
            quality_metrics = QualityMetrics(**detailed['quality_metrics'])
        
        return DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates,
            epicenter_distance=detailed['epicenter_distance'],
            frequency_analysis=frequency_analysis,
            quality_metrics=quality_metrics,
            analysis_timestamp=doc['analysis_timestamp'],
            processing_metadata=doc.get('processing_metadata', {})
        )
    
    # Cache management methods
    def cache_analysis_result(self, 
                            cache_key: str, 
                            analysis_data: Any, 
                            ttl_hours: int = 24) -> None:
        """
        Cache analysis result for faster retrieval.
        
        Args:
            cache_key: Unique cache key
            analysis_data: Data to cache
            ttl_hours: Time to live in hours
        """
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        doc = {
            'cache_key': cache_key,
            'analysis_data': self._serialize_numpy_arrays(analysis_data),
            'created_at': datetime.now(),
            'expires_at': expires_at
        }
        
        self.analysis_cache.replace_one(
            {'cache_key': cache_key},
            doc,
            upsert=True
        )
    
    def get_cached_analysis(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached analysis result.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached data or None if not found/expired
        """
        doc = self.analysis_cache.find_one({
            'cache_key': cache_key,
            'expires_at': {'$gt': datetime.now()}
        })
        
        if not doc:
            return None
        
        return self._deserialize_numpy_arrays(doc['analysis_data'])
    
    def clear_expired_cache(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        result = self.analysis_cache.delete_many({
            'expires_at': {'$lt': datetime.now()}
        })
        return result.deleted_count
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored wave analyses.
        
        Returns:
            Dictionary with analysis statistics
        """
        pipeline = [
            {
                '$group': {
                    '_id': None,
                    'total_analyses': {'$sum': 1},
                    'avg_quality_score': {'$avg': '$processing_metadata.quality_score'},
                    'total_waves_detected': {'$sum': '$processing_metadata.total_waves_detected'},
                    'latest_analysis': {'$max': '$analysis_timestamp'},
                    'oldest_analysis': {'$min': '$analysis_timestamp'}
                }
            }
        ]
        
        result = list(self.wave_analyses.aggregate(pipeline))
        if not result:
            return {
                'total_analyses': 0,
                'avg_quality_score': 0,
                'total_waves_detected': 0,
                'latest_analysis': None,
                'oldest_analysis': None
            }
        
        stats = result[0]
        del stats['_id']
        
        # Add wave type distribution
        wave_type_pipeline = [
            {'$unwind': '$processing_metadata.wave_types_detected'},
            {
                '$group': {
                    '_id': '$processing_metadata.wave_types_detected',
                    'count': {'$sum': 1}
                }
            }
        ]
        
        wave_type_stats = list(self.wave_analyses.aggregate(wave_type_pipeline))
        stats['wave_type_distribution'] = {
            item['_id']: item['count'] for item in wave_type_stats
        }
        
        return stats


def create_wave_analysis_repository(mongo_uri: str, db_name: str) -> WaveAnalysisRepository:
    """
    Factory function to create a WaveAnalysisRepository instance.
    
    Args:
        mongo_uri: MongoDB connection URI
        db_name: Database name
        
    Returns:
        Configured WaveAnalysisRepository instance
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    return WaveAnalysisRepository(db, fs)