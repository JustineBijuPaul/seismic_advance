"""
Tests for wave analysis database models and data access layer.

This module tests the MongoDB integration for wave analysis results,
including storage, retrieval, indexing, and query performance.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta
from bson import ObjectId
import tempfile
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wave_analysis.models.database_models import WaveAnalysisRepository, create_wave_analysis_repository
from wave_analysis.models.wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis,
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)


class TestWaveAnalysisRepository(unittest.TestCase):
    """Test cases for WaveAnalysisRepository class."""
    
    def setUp(self):
        """Set up test fixtures with mock database and GridFS."""
        self.mock_db = Mock()
        self.mock_gridfs = Mock()
        
        # Mock collections
        self.mock_wave_analyses = Mock()
        self.mock_analysis_cache = Mock()
        
        self.mock_db.wave_analyses = self.mock_wave_analyses
        self.mock_db.analysis_cache = self.mock_analysis_cache
        
        # Create repository instance
        self.repository = WaveAnalysisRepository(self.mock_db, self.mock_gridfs)
        
        # Create test data
        self.test_file_id = ObjectId()
        self.test_analysis = self._create_test_analysis()
    
    def _create_test_analysis(self) -> DetailedAnalysis:
        """Create a test DetailedAnalysis object."""
        # Create test wave segments
        p_wave = WaveSegment(
            wave_type='P',
            start_time=10.0,
            end_time=15.0,
            data=np.random.randn(500),
            sampling_rate=100.0,
            peak_amplitude=0.8,
            dominant_frequency=8.0,
            arrival_time=12.0,
            confidence=0.9
        )
        
        s_wave = WaveSegment(
            wave_type='S',
            start_time=20.0,
            end_time=30.0,
            data=np.random.randn(1000),
            sampling_rate=100.0,
            peak_amplitude=1.2,
            dominant_frequency=4.0,
            arrival_time=22.0,
            confidence=0.85
        )
        
        surface_wave = WaveSegment(
            wave_type='Love',
            start_time=40.0,
            end_time=60.0,
            data=np.random.randn(2000),
            sampling_rate=100.0,
            peak_amplitude=0.6,
            dominant_frequency=2.0,
            arrival_time=42.0,
            confidence=0.75
        )
        
        # Create wave analysis result
        wave_result = WaveAnalysisResult(
            original_data=np.random.randn(10000),
            sampling_rate=100.0,
            p_waves=[p_wave],
            s_waves=[s_wave],
            surface_waves=[surface_wave],
            metadata={'station': 'TEST01', 'location': 'Test Location'}
        )
        
        # Create arrival times
        arrival_times = ArrivalTimes(
            p_wave_arrival=12.0,
            s_wave_arrival=22.0,
            surface_wave_arrival=42.0,
            sp_time_difference=10.0
        )
        
        # Create magnitude estimates
        magnitude_estimates = [
            MagnitudeEstimate(
                method='ML',
                magnitude=4.2,
                confidence=0.8,
                wave_type_used='P'
            ),
            MagnitudeEstimate(
                method='Ms',
                magnitude=4.1,
                confidence=0.75,
                wave_type_used='Love'
            )
        ]
        
        # Create frequency data
        frequencies = np.linspace(0, 50, 100)
        power_spectrum = np.random.randn(100)
        frequency_data = FrequencyData(
            frequencies=frequencies,
            power_spectrum=power_spectrum,
            dominant_frequency=8.0,
            frequency_range=(2.0, 20.0),
            spectral_centroid=10.0,
            bandwidth=5.0
        )
        
        # Create quality metrics
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=15.0,
            detection_confidence=0.85,
            analysis_quality_score=0.8,
            data_completeness=0.95,
            processing_warnings=['Low SNR in surface waves']
        )
        
        # Create detailed analysis
        return DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates,
            epicenter_distance=120.5,
            frequency_analysis={'P': frequency_data},
            quality_metrics=quality_metrics,
            processing_metadata={
                'processing_time': 45.2,
                'model_versions': {'p_wave': '1.0', 's_wave': '1.1'}
            }
        )
    
    def test_ensure_indexes_called(self):
        """Test that indexes are created during initialization."""
        # Verify that create_index was called multiple times
        self.assertTrue(self.mock_wave_analyses.create_index.called)
        self.assertTrue(self.mock_analysis_cache.create_index.called)
        
        # Check that specific indexes were created
        call_args_list = self.mock_wave_analyses.create_index.call_args_list
        index_names = [call.kwargs.get('name') for call in call_args_list if 'name' in call.kwargs]
        
        expected_indexes = [
            'file_timestamp_idx',
            'timestamp_idx',
            'quality_score_idx',
            'magnitude_idx',
            'sp_time_idx'
        ]
        
        for expected_index in expected_indexes:
            self.assertIn(expected_index, index_names)
    
    def test_serialize_numpy_arrays(self):
        """Test numpy array serialization for MongoDB storage."""
        test_array = np.array([1, 2, 3, 4, 5])
        test_dict = {
            'data': test_array,
            'nested': {
                'more_data': np.array([6, 7, 8])
            }
        }
        
        serialized = self.repository._serialize_numpy_arrays(test_dict)
        
        self.assertEqual(serialized['data'], [1, 2, 3, 4, 5])
        self.assertEqual(serialized['nested']['more_data'], [6, 7, 8])
        self.assertIsInstance(serialized['data'], list)
        self.assertIsInstance(serialized['nested']['more_data'], list)
    
    def test_deserialize_numpy_arrays(self):
        """Test numpy array deserialization from MongoDB storage."""
        test_dict = {
            'data': [1, 2, 3, 4, 5],
            'nested': {
                'frequencies': [0.1, 0.2, 0.3]
            },
            'other_field': 'string_value'
        }
        
        deserialized = self.repository._deserialize_numpy_arrays(test_dict)
        
        self.assertIsInstance(deserialized['data'], np.ndarray)
        self.assertIsInstance(deserialized['nested']['frequencies'], np.ndarray)
        self.assertEqual(deserialized['other_field'], 'string_value')
        np.testing.assert_array_equal(deserialized['data'], np.array([1, 2, 3, 4, 5]))
    
    def test_store_wave_analysis_small_data(self):
        """Test storing wave analysis with small data (no GridFS)."""
        # Mock insert_one to return a result with inserted_id
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        self.mock_wave_analyses.insert_one.return_value = mock_result
        
        # Store the analysis
        analysis_id = self.repository.store_wave_analysis(self.test_file_id, self.test_analysis)
        
        # Verify insert_one was called
        self.mock_wave_analyses.insert_one.assert_called_once()
        
        # Get the document that was inserted
        inserted_doc = self.mock_wave_analyses.insert_one.call_args[0][0]
        
        # Verify document structure
        self.assertEqual(inserted_doc['file_id'], self.test_file_id)
        self.assertIn('wave_separation', inserted_doc)
        self.assertIn('detailed_analysis', inserted_doc)
        self.assertIn('processing_metadata', inserted_doc)
        
        # Verify wave separation data
        wave_sep = inserted_doc['wave_separation']
        self.assertEqual(len(wave_sep['p_waves']), 1)
        self.assertEqual(len(wave_sep['s_waves']), 1)
        self.assertEqual(len(wave_sep['surface_waves']), 1)
        
        # Verify analysis ID is returned as string
        self.assertIsInstance(analysis_id, str)
    
    def test_store_wave_analysis_large_data(self):
        """Test storing wave analysis with large data (uses GridFS)."""
        # Create analysis with large original data
        large_analysis = self.test_analysis
        large_analysis.wave_result.original_data = np.random.randn(2000000)  # Large array
        
        # Mock GridFS put
        mock_data_id = ObjectId()
        self.mock_gridfs.put.return_value = mock_data_id
        
        # Mock insert_one
        mock_result = Mock()
        mock_result.inserted_id = ObjectId()
        self.mock_wave_analyses.insert_one.return_value = mock_result
        
        # Store the analysis
        analysis_id = self.repository.store_wave_analysis(self.test_file_id, large_analysis)
        
        # Verify GridFS was used
        self.mock_gridfs.put.assert_called_once()
        
        # Verify document contains GridFS reference
        inserted_doc = self.mock_wave_analyses.insert_one.call_args[0][0]
        self.assertIn('original_data_gridfs_id', inserted_doc)
        self.assertEqual(inserted_doc['original_data_gridfs_id'], mock_data_id)
        self.assertNotIn('original_data', inserted_doc)
    
    def test_get_wave_analysis_found(self):
        """Test retrieving an existing wave analysis."""
        test_id = ObjectId()
        
        # Mock document from database
        mock_doc = {
            '_id': test_id,
            'file_id': self.test_file_id,
            'analysis_timestamp': datetime.now(),
            'original_data': np.random.randn(1000).tolist(),
            'wave_separation': {
                'p_waves': [],
                's_waves': [],
                'surface_waves': [],
                'sampling_rate': 100.0,
                'metadata': {}
            },
            'detailed_analysis': {
                'arrival_times': {
                    'p_wave_arrival': 12.0,
                    's_wave_arrival': 22.0,
                    'surface_wave_arrival': None,
                    'sp_time_difference': 10.0
                },
                'magnitude_estimates': [],
                'epicenter_distance': None,
                'frequency_analysis': {},
                'quality_metrics': None
            },
            'processing_metadata': {}
        }
        
        self.mock_wave_analyses.find_one.return_value = mock_doc
        
        # Get the analysis
        result = self.repository.get_wave_analysis(test_id)
        
        # Verify find_one was called with correct ID
        self.mock_wave_analyses.find_one.assert_called_once_with({'_id': test_id})
        
        # Verify result is DetailedAnalysis object
        self.assertIsInstance(result, DetailedAnalysis)
        self.assertEqual(result.arrival_times.p_wave_arrival, 12.0)
        self.assertEqual(result.arrival_times.s_wave_arrival, 22.0)
    
    def test_get_wave_analysis_not_found(self):
        """Test retrieving a non-existent wave analysis."""
        test_id = ObjectId()
        self.mock_wave_analyses.find_one.return_value = None
        
        result = self.repository.get_wave_analysis(test_id)
        
        self.assertIsNone(result)
        self.mock_wave_analyses.find_one.assert_called_once_with({'_id': test_id})
    
    def test_get_analyses_by_file(self):
        """Test retrieving all analyses for a specific file."""
        # Mock cursor with multiple documents
        mock_docs = [
            {
                '_id': ObjectId(),
                'file_id': self.test_file_id,
                'analysis_timestamp': datetime.now(),
                'original_data': np.random.randn(100).tolist(),
                'wave_separation': {
                    'p_waves': [], 's_waves': [], 'surface_waves': [],
                    'sampling_rate': 100.0, 'metadata': {}
                },
                'detailed_analysis': {
                    'arrival_times': {'p_wave_arrival': None, 's_wave_arrival': None, 
                                    'surface_wave_arrival': None, 'sp_time_difference': None},
                    'magnitude_estimates': [], 'epicenter_distance': None,
                    'frequency_analysis': {}, 'quality_metrics': None
                },
                'processing_metadata': {}
            }
        ]
        
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_docs))
        mock_cursor.sort.return_value = mock_cursor
        
        self.mock_wave_analyses.find.return_value = mock_cursor
        
        # Get analyses by file
        results = self.repository.get_analyses_by_file(self.test_file_id)
        
        # Verify query was made correctly
        self.mock_wave_analyses.find.assert_called_once_with({'file_id': self.test_file_id})
        mock_cursor.sort.assert_called_once_with('analysis_timestamp', -1)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], DetailedAnalysis)
    
    def test_get_recent_analyses(self):
        """Test retrieving recent analyses with quality filtering."""
        mock_results = [
            {
                '_id': ObjectId(),
                'file_id': ObjectId(),
                'analysis_timestamp': datetime.now(),
                'processing_metadata': {
                    'quality_score': 0.8,
                    'total_waves_detected': 3,
                    'wave_types_detected': ['P', 'S']
                },
                'detailed_analysis': {
                    'magnitude_estimates': [{'magnitude': 4.2}],
                    'arrival_times': {'sp_time_difference': 10.0}
                }
            }
        ]
        
        self.mock_wave_analyses.aggregate.return_value = mock_results
        
        # Get recent analyses
        results = self.repository.get_recent_analyses(limit=10, min_quality_score=0.5)
        
        # Verify aggregate was called
        self.mock_wave_analyses.aggregate.assert_called_once()
        
        # Verify pipeline structure
        pipeline = self.mock_wave_analyses.aggregate.call_args[0][0]
        self.assertEqual(len(pipeline), 4)  # match, sort, limit, project
        self.assertIn('$match', pipeline[0])
        self.assertIn('$sort', pipeline[1])
        self.assertIn('$limit', pipeline[2])
        self.assertIn('$project', pipeline[3])
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['processing_metadata']['quality_score'], 0.8)
    
    def test_search_analyses_by_magnitude(self):
        """Test searching analyses by magnitude range."""
        mock_results = [
            {
                '_id': ObjectId(),
                'file_id': ObjectId(),
                'detailed_analysis': {
                    'magnitude_estimates': [{'magnitude': 4.5}]
                }
            }
        ]
        
        self.mock_wave_analyses.aggregate.return_value = mock_results
        
        # Search by magnitude
        results = self.repository.search_analyses_by_magnitude(4.0, 5.0)
        
        # Verify aggregate was called
        self.mock_wave_analyses.aggregate.assert_called_once()
        
        # Verify pipeline has magnitude filter
        pipeline = self.mock_wave_analyses.aggregate.call_args[0][0]
        match_stage = pipeline[0]['$match']
        self.assertIn('detailed_analysis.magnitude_estimates.magnitude', match_stage)
        
        # Verify results
        self.assertEqual(len(results), 1)
    
    def test_delete_analysis_success(self):
        """Test successful deletion of wave analysis."""
        test_id = ObjectId()
        
        # Mock document with GridFS reference
        mock_doc = {
            '_id': test_id,
            'original_data_gridfs_id': ObjectId()
        }
        
        self.mock_wave_analyses.find_one.return_value = mock_doc
        
        # Mock successful deletion
        mock_delete_result = Mock()
        mock_delete_result.deleted_count = 1
        self.mock_wave_analyses.delete_one.return_value = mock_delete_result
        
        # Delete analysis
        result = self.repository.delete_analysis(test_id)
        
        # Verify GridFS deletion was attempted
        self.mock_gridfs.delete.assert_called_once_with(mock_doc['original_data_gridfs_id'])
        
        # Verify document deletion
        self.mock_wave_analyses.delete_one.assert_called_once_with({'_id': test_id})
        
        # Verify success
        self.assertTrue(result)
    
    def test_delete_analysis_not_found(self):
        """Test deletion of non-existent analysis."""
        test_id = ObjectId()
        self.mock_wave_analyses.find_one.return_value = None
        
        result = self.repository.delete_analysis(test_id)
        
        self.assertFalse(result)
        self.mock_gridfs.delete.assert_not_called()
        self.mock_wave_analyses.delete_one.assert_not_called()
    
    def test_cache_analysis_result(self):
        """Test caching analysis results."""
        cache_key = 'test_analysis_123'
        test_data = {'result': 'test_value', 'array': np.array([1, 2, 3])}
        
        # Cache the result
        self.repository.cache_analysis_result(cache_key, test_data, ttl_hours=12)
        
        # Verify replace_one was called
        self.mock_analysis_cache.replace_one.assert_called_once()
        
        # Get the cached document
        call_args = self.mock_analysis_cache.replace_one.call_args
        filter_doc = call_args[0][0]
        cached_doc = call_args[0][1]
        
        self.assertEqual(filter_doc['cache_key'], cache_key)
        self.assertEqual(cached_doc['cache_key'], cache_key)
        self.assertIn('expires_at', cached_doc)
        self.assertIn('analysis_data', cached_doc)
    
    def test_get_cached_analysis_found(self):
        """Test retrieving cached analysis result."""
        cache_key = 'test_cache_key'
        cached_data = {'result': 'cached_value', 'array': [1, 2, 3]}
        
        mock_doc = {
            'cache_key': cache_key,
            'analysis_data': cached_data,
            'expires_at': datetime.now() + timedelta(hours=1)
        }
        
        self.mock_analysis_cache.find_one.return_value = mock_doc
        
        # Get cached result
        result = self.repository.get_cached_analysis(cache_key)
        
        # Verify query
        self.mock_analysis_cache.find_one.assert_called_once()
        query = self.mock_analysis_cache.find_one.call_args[0][0]
        self.assertEqual(query['cache_key'], cache_key)
        self.assertIn('expires_at', query)
        
        # Verify result (handle numpy arrays in comparison)
        self.assertEqual(result['result'], cached_data['result'])
        np.testing.assert_array_equal(result['array'], cached_data['array'])
    
    def test_get_cached_analysis_not_found(self):
        """Test retrieving non-existent cached analysis."""
        self.mock_analysis_cache.find_one.return_value = None
        
        result = self.repository.get_cached_analysis('nonexistent_key')
        
        self.assertIsNone(result)
    
    def test_clear_expired_cache(self):
        """Test clearing expired cache entries."""
        mock_result = Mock()
        mock_result.deleted_count = 5
        self.mock_analysis_cache.delete_many.return_value = mock_result
        
        # Clear expired cache
        count = self.repository.clear_expired_cache()
        
        # Verify delete_many was called with expiry filter
        self.mock_analysis_cache.delete_many.assert_called_once()
        query = self.mock_analysis_cache.delete_many.call_args[0][0]
        self.assertIn('expires_at', query)
        
        # Verify count returned
        self.assertEqual(count, 5)
    
    def test_get_analysis_statistics(self):
        """Test getting analysis statistics."""
        mock_stats = [{
            '_id': None,
            'total_analyses': 100,
            'avg_quality_score': 0.75,
            'total_waves_detected': 300,
            'latest_analysis': datetime.now(),
            'oldest_analysis': datetime.now() - timedelta(days=30)
        }]
        
        mock_wave_type_stats = [
            {'_id': 'P', 'count': 80},
            {'_id': 'S', 'count': 75},
            {'_id': 'Love', 'count': 45}
        ]
        
        # Mock aggregate calls
        self.mock_wave_analyses.aggregate.side_effect = [mock_stats, mock_wave_type_stats]
        
        # Get statistics
        stats = self.repository.get_analysis_statistics()
        
        # Verify aggregate was called twice
        self.assertEqual(self.mock_wave_analyses.aggregate.call_count, 2)
        
        # Verify statistics structure
        self.assertEqual(stats['total_analyses'], 100)
        self.assertEqual(stats['avg_quality_score'], 0.75)
        self.assertIn('wave_type_distribution', stats)
        self.assertEqual(stats['wave_type_distribution']['P'], 80)
        self.assertEqual(stats['wave_type_distribution']['S'], 75)
        self.assertEqual(stats['wave_type_distribution']['Love'], 45)
    
    def test_get_analysis_statistics_empty(self):
        """Test getting statistics when no analyses exist."""
        # Mock empty results
        self.mock_wave_analyses.aggregate.side_effect = [[], []]
        
        stats = self.repository.get_analysis_statistics()
        
        # Verify default values
        self.assertEqual(stats['total_analyses'], 0)
        self.assertEqual(stats['avg_quality_score'], 0)
        self.assertEqual(stats['total_waves_detected'], 0)
        self.assertIsNone(stats['latest_analysis'])
        self.assertIsNone(stats['oldest_analysis'])


class TestCreateWaveAnalysisRepository(unittest.TestCase):
    """Test cases for the repository factory function."""
    
    @patch('wave_analysis.models.database_models.MongoClient')
    @patch('wave_analysis.models.database_models.gridfs.GridFS')
    def test_create_wave_analysis_repository(self, mock_gridfs_class, mock_mongo_client_class):
        """Test the factory function creates repository correctly."""
        # Mock MongoDB client and database
        mock_client = MagicMock()
        mock_db = Mock()
        mock_gridfs = Mock()
        
        mock_mongo_client_class.return_value = mock_client
        mock_client.__getitem__.return_value = mock_db
        mock_gridfs_class.return_value = mock_gridfs
        
        # Create repository
        repo = create_wave_analysis_repository('mongodb://localhost:27017', 'test_db')
        
        # Verify MongoDB client was created
        mock_mongo_client_class.assert_called_once_with('mongodb://localhost:27017')
        
        # Verify database was accessed
        mock_client.__getitem__.assert_called_once_with('test_db')
        
        # Verify GridFS was created
        mock_gridfs_class.assert_called_once_with(mock_db)
        
        # Verify repository was created
        self.assertIsInstance(repo, WaveAnalysisRepository)


class TestWaveAnalysisDatabaseIntegration(unittest.TestCase):
    """Integration tests for wave analysis database operations."""
    
    def setUp(self):
        """Set up integration test environment."""
        # These tests would require a real MongoDB instance
        # For now, we'll skip them unless MongoDB is available
        self.skipTest("Integration tests require MongoDB instance")
    
    def test_full_workflow_integration(self):
        """Test complete workflow from storage to retrieval."""
        # This would test the full workflow with a real database
        pass
    
    def test_performance_with_large_datasets(self):
        """Test performance with large wave analysis datasets."""
        # This would test performance characteristics
        pass
    
    def test_concurrent_access(self):
        """Test concurrent read/write operations."""
        # This would test thread safety and concurrent access
        pass


if __name__ == '__main__':
    unittest.main()