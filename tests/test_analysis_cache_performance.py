"""
Performance tests for the analysis caching system.

This module contains comprehensive performance tests to validate
the effectiveness of the caching implementation.
"""

import pytest
import time
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import threading
import concurrent.futures
from typing import List, Dict, Any

from wave_analysis.services.analysis_cache import AnalysisCacheManager, CacheDecorator
from wave_analysis.services.cache_warming import CacheWarmingService, WarmingStrategy
from wave_analysis.models.wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis,
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)


class TestAnalysisCachePerformance:
    """Performance tests for the analysis cache manager."""
    
    @pytest.fixture
    def mock_mongodb(self):
        """Create a mock MongoDB database."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_db.analysis_cache = mock_collection
        
        # Mock collection methods
        mock_collection.create_index = Mock()
        mock_collection.replace_one = Mock()
        mock_collection.find_one = Mock()
        mock_collection.delete_many = Mock()
        mock_collection.aggregate = Mock()
        mock_collection.update_one = Mock()
        
        return mock_db
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock_redis = Mock()
        mock_redis.setex = Mock()
        mock_redis.get = Mock()
        mock_redis.delete = Mock()
        return mock_redis
    
    @pytest.fixture
    def cache_manager(self, mock_mongodb, mock_redis):
        """Create a cache manager instance for testing."""
        return AnalysisCacheManager(
            mongodb=mock_mongodb,
            redis_client=mock_redis,
            default_ttl_hours=24,
            max_memory_cache_size=100
        )
    
    @pytest.fixture
    def sample_wave_segment(self):
        """Create a sample wave segment for testing."""
        return WaveSegment(
            wave_type='P',
            start_time=10.0,
            end_time=15.0,
            data=np.random.random(1000),
            sampling_rate=100.0,
            peak_amplitude=0.5,
            dominant_frequency=5.0,
            arrival_time=12.0,
            confidence=0.9
        )
    
    @pytest.fixture
    def sample_analysis_result(self, sample_wave_segment):
        """Create a sample analysis result for testing."""
        original_data = np.random.random(10000)
        
        wave_result = WaveAnalysisResult(
            original_data=original_data,
            sampling_rate=100.0,
            p_waves=[sample_wave_segment],
            s_waves=[],
            surface_waves=[]
        )
        
        arrival_times = ArrivalTimes(
            p_wave_arrival=12.0,
            s_wave_arrival=18.0,
            sp_time_difference=6.0
        )
        
        magnitude_estimate = MagnitudeEstimate(
            method='ML',
            magnitude=4.5,
            confidence=0.8,
            wave_type_used='P'
        )
        
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=15.0,
            detection_confidence=0.85,
            analysis_quality_score=0.9,
            data_completeness=1.0
        )
        
        return DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=[magnitude_estimate],
            quality_metrics=quality_metrics
        )
    
    def test_cache_write_performance(self, cache_manager, sample_analysis_result):
        """Test cache write performance with various data sizes."""
        file_id = "test_file_123"
        operation = "detailed_analysis"
        
        # Test with different data sizes
        data_sizes = [1000, 10000, 100000]
        write_times = []
        
        for size in data_sizes:
            # Create analysis result with specific data size
            large_data = np.random.random(size)
            sample_analysis_result.wave_result.original_data = large_data
            
            # Measure write time
            start_time = time.time()
            cache_key = cache_manager.cache_analysis_result(
                operation, file_id, sample_analysis_result
            )
            write_time = time.time() - start_time
            write_times.append(write_time)
            
            # Verify cache key was generated
            assert cache_key is not None
            assert len(cache_key) == 64  # SHA256 hash length
        
        # Performance assertions
        assert all(t < 1.0 for t in write_times), "Cache writes should complete within 1 second"
        
        # Write time should scale reasonably with data size
        assert write_times[2] > write_times[0], "Larger data should take more time to cache"
        
        print(f"Cache write times: {write_times}")
    
    def test_cache_read_performance(self, cache_manager, sample_analysis_result):
        """Test cache read performance from different storage backends."""
        file_id = "test_file_456"
        operation = "detailed_analysis"
        
        # Cache the result first
        cache_manager.cache_analysis_result(operation, file_id, sample_analysis_result)
        
        # Test memory cache read performance
        memory_read_times = []
        for _ in range(100):
            start_time = time.time()
            result = cache_manager.get_cached_result(operation, file_id)
            read_time = time.time() - start_time
            memory_read_times.append(read_time)
            assert result is not None
        
        avg_memory_read_time = sum(memory_read_times) / len(memory_read_times)
        
        # Clear memory cache to test MongoDB read
        cache_manager._memory_cache.clear()
        
        # Mock MongoDB response
        cache_manager.cache_collection.find_one.return_value = {
            'cache_key': 'test_key',
            'data': cache_manager._serialize_for_cache(sample_analysis_result),
            'expires_at': datetime.now() + timedelta(hours=24)
        }
        
        mongodb_read_times = []
        for _ in range(10):
            start_time = time.time()
            result = cache_manager.get_cached_result(operation, file_id)
            read_time = time.time() - start_time
            mongodb_read_times.append(read_time)
        
        avg_mongodb_read_time = sum(mongodb_read_times) / len(mongodb_read_times)
        
        # Performance assertions
        assert avg_memory_read_time < 0.001, "Memory cache reads should be very fast"
        assert avg_mongodb_read_time < 0.1, "MongoDB cache reads should be reasonably fast"
        assert avg_memory_read_time < avg_mongodb_read_time, "Memory cache should be faster than MongoDB"
        
        print(f"Average memory read time: {avg_memory_read_time:.6f}s")
        print(f"Average MongoDB read time: {avg_mongodb_read_time:.6f}s")
    
    def test_concurrent_cache_access(self, cache_manager, sample_analysis_result):
        """Test cache performance under concurrent access."""
        file_id = "test_file_concurrent"
        operation = "detailed_analysis"
        num_threads = 10
        operations_per_thread = 20
        
        # Pre-populate cache
        cache_manager.cache_analysis_result(operation, file_id, sample_analysis_result)
        
        def cache_access_worker(worker_id: int) -> List[float]:
            """Worker function for concurrent cache access."""
            access_times = []
            for i in range(operations_per_thread):
                start_time = time.time()
                result = cache_manager.get_cached_result(operation, f"{file_id}_{worker_id}_{i}")
                access_time = time.time() - start_time
                access_times.append(access_time)
                
                # Also test cache writes
                if i % 5 == 0:
                    cache_manager.cache_analysis_result(
                        operation, f"{file_id}_{worker_id}_{i}", sample_analysis_result
                    )
            
            return access_times
        
        # Execute concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(cache_access_worker, i) 
                for i in range(num_threads)
            ]
            
            all_access_times = []
            for future in concurrent.futures.as_completed(futures):
                access_times = future.result()
                all_access_times.extend(access_times)
        
        # Performance analysis
        avg_access_time = sum(all_access_times) / len(all_access_times)
        max_access_time = max(all_access_times)
        
        # Performance assertions
        assert avg_access_time < 0.01, "Average concurrent access time should be reasonable"
        assert max_access_time < 0.1, "Maximum access time should not be excessive"
        
        print(f"Concurrent access - Average: {avg_access_time:.6f}s, Max: {max_access_time:.6f}s")
    
    def test_cache_memory_usage(self, cache_manager, sample_analysis_result):
        """Test cache memory usage and cleanup behavior."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Fill cache with many entries
        num_entries = 200  # More than max_memory_cache_size
        for i in range(num_entries):
            file_id = f"test_file_{i}"
            cache_manager.cache_analysis_result(
                "detailed_analysis", file_id, sample_analysis_result
            )
        
        # Check memory usage
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Verify cache size limit is respected
        assert len(cache_manager._memory_cache) <= cache_manager.max_memory_cache_size
        
        # Clean up expired entries
        cleaned_count = cache_manager.cleanup_expired_entries()
        
        # Check memory after cleanup
        final_memory = process.memory_info().rss
        
        print(f"Memory usage - Initial: {initial_memory/1024/1024:.1f}MB, "
              f"Peak: {peak_memory/1024/1024:.1f}MB, "
              f"Final: {final_memory/1024/1024:.1f}MB")
        print(f"Cleaned {cleaned_count} expired entries")
        
        # Memory should not grow excessively
        assert memory_increase < 100 * 1024 * 1024, "Memory increase should be reasonable (< 100MB)"
    
    def test_cache_hit_rate_improvement(self, cache_manager, sample_analysis_result):
        """Test that caching improves hit rates over time."""
        file_ids = [f"test_file_{i}" for i in range(20)]
        operation = "detailed_analysis"
        
        # Simulate initial requests (all misses)
        initial_stats = cache_manager.get_cache_statistics()
        initial_misses = initial_stats['performance']['total_misses']
        
        for file_id in file_ids:
            result = cache_manager.get_cached_result(operation, file_id)
            assert result is None  # Should be cache miss
            
            # Cache the result
            cache_manager.cache_analysis_result(operation, file_id, sample_analysis_result)
        
        # Simulate repeated requests (should be hits)
        for _ in range(3):  # Multiple rounds
            for file_id in file_ids:
                result = cache_manager.get_cached_result(operation, file_id)
                assert result is not None  # Should be cache hit
        
        # Check final statistics
        final_stats = cache_manager.get_cache_statistics()
        
        # Calculate hit rate improvement
        total_requests = final_stats['performance']['total_hits'] + final_stats['performance']['total_misses']
        hit_rate = final_stats['performance']['hit_rate']
        
        print(f"Final hit rate: {hit_rate:.2%}")
        print(f"Total requests: {total_requests}")
        
        # Hit rate should be good after warming
        assert hit_rate > 0.5, "Hit rate should improve significantly with repeated access"
    
    def test_cache_invalidation_performance(self, cache_manager, sample_analysis_result):
        """Test performance of cache invalidation operations."""
        # Populate cache with many entries
        num_entries = 100
        file_ids = []
        
        for i in range(num_entries):
            file_id = f"test_file_{i}"
            file_ids.append(file_id)
            cache_manager.cache_analysis_result(
                "detailed_analysis", file_id, sample_analysis_result
            )
        
        # Test selective invalidation performance
        start_time = time.time()
        invalidated_count = cache_manager.invalidate_cache(
            operation="detailed_analysis",
            file_id=file_ids[0]
        )
        selective_invalidation_time = time.time() - start_time
        
        # Test bulk invalidation performance
        start_time = time.time()
        bulk_invalidated_count = cache_manager.invalidate_cache(
            operation="detailed_analysis"
        )
        bulk_invalidation_time = time.time() - start_time
        
        # Performance assertions
        assert selective_invalidation_time < 0.1, "Selective invalidation should be fast"
        assert bulk_invalidation_time < 1.0, "Bulk invalidation should complete reasonably quickly"
        
        print(f"Selective invalidation: {selective_invalidation_time:.6f}s ({invalidated_count} entries)")
        print(f"Bulk invalidation: {bulk_invalidation_time:.6f}s ({bulk_invalidated_count} entries)")


class TestCacheWarmingPerformance:
    """Performance tests for cache warming strategies."""
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock database for warming tests."""
        mock_db = Mock()
        
        # Mock GridFS files collection
        mock_fs_files = Mock()
        mock_db.fs = Mock()
        mock_db.fs.files = mock_fs_files
        
        # Mock wave analyses collection
        mock_wave_analyses = Mock()
        mock_db.wave_analyses = mock_wave_analyses
        
        return mock_db
    
    @pytest.fixture
    def cache_warming_service(self, mock_db):
        """Create a cache warming service for testing."""
        mock_cache_manager = Mock(spec=AnalysisCacheManager)
        mock_cache_manager.cache_collection = Mock()
        
        return CacheWarmingService(
            cache_manager=mock_cache_manager,
            db=mock_db
        )
    
    def test_warming_strategy_execution_time(self, cache_warming_service):
        """Test execution time of different warming strategies."""
        # Mock file IDs for different strategies
        recent_files = [f"recent_{i}" for i in range(20)]
        quality_files = [f"quality_{i}" for i in range(15)]
        
        # Mock database responses
        cache_warming_service.db.fs.files.aggregate.return_value = [
            {'_id': file_id} for file_id in recent_files
        ]
        
        cache_warming_service.db.wave_analyses.aggregate.return_value = [
            {'file_id': file_id} for file_id in quality_files
        ]
        
        # Mock cache manager warming
        cache_warming_service.cache_manager.warm_cache.return_value = {
            'operations_attempted': 20,
            'operations_successful': 18,
            'operations_failed': 2,
            'total_time': 5.2
        }
        
        # Test recent files strategy
        recent_strategy = WarmingStrategy(
            name="recent_files",
            description="Test recent files",
            priority=1,
            max_files=20
        )
        
        start_time = time.time()
        result = cache_warming_service.execute_warming_strategy(recent_strategy)
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert execution_time < 10.0, "Strategy execution should complete within 10 seconds"
        assert result['status'] == 'completed'
        assert result['files_warmed'] > 0
        
        print(f"Recent files strategy execution time: {execution_time:.3f}s")
    
    def test_concurrent_warming_performance(self, cache_warming_service):
        """Test performance of concurrent warming operations."""
        # Create multiple strategies
        strategies = [
            WarmingStrategy(f"strategy_{i}", f"Test strategy {i}", i, max_files=10)
            for i in range(5)
        ]
        
        cache_warming_service.strategies = strategies
        
        # Mock database responses for all strategies
        cache_warming_service.db.fs.files.aggregate.return_value = [
            {'_id': f"file_{i}"} for i in range(50)
        ]
        
        cache_warming_service.db.wave_analyses.aggregate.return_value = [
            {'file_id': f"file_{i}"} for i in range(30)
        ]
        
        cache_warming_service.cache_manager.cache_collection.aggregate.return_value = [
            {'_id': f"file_{i}", 'total_accesses': 10 - i} for i in range(25)
        ]
        
        # Mock warming function
        cache_warming_service.cache_manager.warm_cache.return_value = {
            'operations_attempted': 10,
            'operations_successful': 8,
            'operations_failed': 2,
            'total_time': 2.0
        }
        
        # Execute all strategies
        start_time = time.time()
        results = cache_warming_service.execute_all_strategies()
        total_execution_time = time.time() - start_time
        
        # Performance assertions
        assert total_execution_time < 30.0, "All strategies should complete within 30 seconds"
        assert results['strategies_executed'] > 0
        assert results['total_files_warmed'] > 0
        
        print(f"All strategies execution time: {total_execution_time:.3f}s")
        print(f"Strategies executed: {results['strategies_executed']}")
        print(f"Files warmed: {results['total_files_warmed']}")
    
    def test_warming_memory_efficiency(self, cache_warming_service):
        """Test memory efficiency of warming operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Mock large number of files for warming
        large_file_list = [f"file_{i}" for i in range(1000)]
        
        cache_warming_service.db.fs.files.aggregate.return_value = [
            {'_id': file_id} for file_id in large_file_list
        ]
        
        # Mock warming with memory-efficient processing
        def mock_warm_cache(functions, file_ids, max_concurrent=5):
            # Simulate processing in batches to test memory efficiency
            batch_size = 50
            total_successful = 0
            
            for i in range(0, len(file_ids), batch_size):
                batch = file_ids[i:i + batch_size]
                # Simulate processing batch
                time.sleep(0.01)  # Small delay to simulate work
                total_successful += len(batch)
            
            return {
                'operations_attempted': len(file_ids),
                'operations_successful': total_successful,
                'operations_failed': 0,
                'total_time': len(file_ids) * 0.01
            }
        
        cache_warming_service.cache_manager.warm_cache = mock_warm_cache
        
        # Execute warming strategy
        strategy = WarmingStrategy(
            name="memory_test",
            description="Memory efficiency test",
            priority=1,
            max_files=1000
        )
        
        result = cache_warming_service.execute_warming_strategy(strategy)
        
        # Check memory usage
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Memory efficiency assertions
        assert memory_increase < 50 * 1024 * 1024, "Memory increase should be reasonable (< 50MB)"
        assert result['files_warmed'] > 0
        
        print(f"Memory increase during warming: {memory_increase / 1024 / 1024:.1f}MB")
        print(f"Files warmed: {result['files_warmed']}")


class TestCacheDecoratorPerformance:
    """Performance tests for the cache decorator."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Create a mock cache manager for decorator tests."""
        mock_manager = Mock(spec=AnalysisCacheManager)
        mock_manager.get_cached_result.return_value = None  # Initially no cache
        mock_manager.cache_analysis_result = Mock()
        return mock_manager
    
    def test_decorator_overhead(self, mock_cache_manager):
        """Test the performance overhead of using the cache decorator."""
        
        @CacheDecorator(mock_cache_manager, "test_operation")
        def expensive_function(file_id: str, data_size: int = 1000) -> np.ndarray:
            """Simulate an expensive computation."""
            time.sleep(0.01)  # Simulate work
            return np.random.random(data_size)
        
        # Test without caching (first call)
        start_time = time.time()
        result1 = expensive_function("test_file", 1000)
        first_call_time = time.time() - start_time
        
        # Mock cache hit for second call
        mock_cache_manager.get_cached_result.return_value = result1
        
        # Test with caching (second call)
        start_time = time.time()
        result2 = expensive_function("test_file", 1000)
        second_call_time = time.time() - start_time
        
        # Performance assertions
        assert first_call_time > 0.005, "First call should take some time"
        assert second_call_time < 0.001, "Cached call should be very fast"
        assert second_call_time < first_call_time / 10, "Cache should provide significant speedup"
        
        # Verify cache was used
        mock_cache_manager.get_cached_result.assert_called()
        
        print(f"First call time: {first_call_time:.6f}s")
        print(f"Cached call time: {second_call_time:.6f}s")
        print(f"Speedup: {first_call_time / second_call_time:.1f}x")
    
    def test_decorator_with_different_parameters(self, mock_cache_manager):
        """Test decorator performance with different parameter combinations."""
        
        call_times = {}
        
        @CacheDecorator(mock_cache_manager, "parameterized_operation")
        def parameterized_function(file_id: str, param1: int, param2: str = "default") -> str:
            """Function with multiple parameters."""
            time.sleep(0.005)  # Simulate work
            return f"result_{file_id}_{param1}_{param2}"
        
        # Test different parameter combinations
        test_cases = [
            ("file1", 100, "test1"),
            ("file1", 200, "test1"),  # Different param1
            ("file1", 100, "test2"),  # Different param2
            ("file2", 100, "test1"),  # Different file_id
            ("file1", 100, "test1"),  # Repeat first case
        ]
        
        for i, (file_id, param1, param2) in enumerate(test_cases):
            # Mock cache behavior - only hit on exact repeat
            if i == 4:  # Last case is repeat of first
                mock_cache_manager.get_cached_result.return_value = "cached_result"
            else:
                mock_cache_manager.get_cached_result.return_value = None
            
            start_time = time.time()
            result = parameterized_function(file_id, param1, param2)
            call_time = time.time() - start_time
            call_times[f"call_{i}"] = call_time
        
        # Performance assertions
        assert call_times["call_4"] < call_times["call_0"] / 5, "Cached call should be much faster"
        
        print("Parameter-based caching performance:")
        for call, time_taken in call_times.items():
            print(f"  {call}: {time_taken:.6f}s")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])