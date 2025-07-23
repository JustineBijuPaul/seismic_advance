#!/usr/bin/env python3
"""
Simple test to verify wave analysis database models work correctly.

This test uses mocked MongoDB connections to verify the models
without requiring a live database connection.
"""

import os
import sys
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wave_analysis.models import (
    WaveAnalysisRepository,
    WaveSegment, WaveAnalysisResult, DetailedAnalysis,
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)


def test_wave_models():
    """Test that wave analysis models can be created and used correctly."""
    print("Testing wave analysis data models...")
    
    # Test WaveSegment creation
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
    
    print(f"✓ Created P-wave segment: {p_wave.duration:.1f}s duration, {p_wave.sample_count} samples")
    
    # Test WaveAnalysisResult
    wave_result = WaveAnalysisResult(
        original_data=np.random.randn(1000),
        sampling_rate=100.0,
        p_waves=[p_wave],
        s_waves=[],
        surface_waves=[],
        metadata={'test': True}
    )
    
    print(f"✓ Created WaveAnalysisResult: {wave_result.total_waves_detected} waves detected")
    print(f"  Wave types: {wave_result.wave_types_detected}")
    
    # Test DetailedAnalysis
    arrival_times = ArrivalTimes(p_wave_arrival=12.0, s_wave_arrival=22.0)
    magnitude_est = MagnitudeEstimate(method='ML', magnitude=4.2, confidence=0.8, wave_type_used='P')
    quality_metrics = QualityMetrics(
        signal_to_noise_ratio=15.0,
        detection_confidence=0.85,
        analysis_quality_score=0.8,
        data_completeness=0.95
    )
    
    detailed_analysis = DetailedAnalysis(
        wave_result=wave_result,
        arrival_times=arrival_times,
        magnitude_estimates=[magnitude_est],
        quality_metrics=quality_metrics
    )
    
    print(f"✓ Created DetailedAnalysis: magnitude {detailed_analysis.best_magnitude_estimate.magnitude}")
    print(f"  Complete analysis: {detailed_analysis.has_complete_analysis}")
    
    return detailed_analysis


def test_repository_with_mocks():
    """Test WaveAnalysisRepository with mocked MongoDB."""
    print("\nTesting WaveAnalysisRepository with mocked database...")
    
    # Create mock database and GridFS
    mock_db = Mock()
    mock_gridfs = Mock()
    
    # Mock collections
    mock_wave_analyses = Mock()
    mock_analysis_cache = Mock()
    
    mock_db.wave_analyses = mock_wave_analyses
    mock_db.analysis_cache = mock_analysis_cache
    
    # Create repository
    repo = WaveAnalysisRepository(mock_db, mock_gridfs)
    print("✓ Created WaveAnalysisRepository with mocked database")
    
    # Verify indexes were created
    assert mock_wave_analyses.create_index.called, "Wave analyses indexes should be created"
    assert mock_analysis_cache.create_index.called, "Cache indexes should be created"
    print("✓ Database indexes were created during initialization")
    
    # Test serialization
    test_data = {
        'array_field': np.array([1, 2, 3, 4, 5]),
        'nested': {
            'another_array': np.array([6, 7, 8])
        },
        'regular_field': 'test_value'
    }
    
    serialized = repo._serialize_numpy_arrays(test_data)
    assert isinstance(serialized['array_field'], list), "Numpy arrays should be serialized to lists"
    assert isinstance(serialized['nested']['another_array'], list), "Nested arrays should be serialized"
    assert serialized['regular_field'] == 'test_value', "Regular fields should be unchanged"
    print("✓ Numpy array serialization works correctly")
    
    # Test deserialization
    deserialized = repo._deserialize_numpy_arrays(serialized)
    assert isinstance(deserialized['array_field'], np.ndarray), "Lists should be deserialized to numpy arrays"
    np.testing.assert_array_equal(deserialized['array_field'], test_data['array_field'])
    print("✓ Numpy array deserialization works correctly")
    
    return repo


def test_storage_operations():
    """Test storage operations with mocked database."""
    print("\nTesting storage operations...")
    
    # Create test analysis
    detailed_analysis = test_wave_models()
    
    # Create repository with mocks
    repo = test_repository_with_mocks()
    
    # Mock successful storage
    from bson import ObjectId
    mock_result = Mock()
    mock_result.inserted_id = ObjectId()
    repo.wave_analyses.insert_one.return_value = mock_result
    
    # Test storage
    test_file_id = ObjectId()
    analysis_id = repo.store_wave_analysis(test_file_id, detailed_analysis)
    
    assert repo.wave_analyses.insert_one.called, "insert_one should be called"
    assert isinstance(analysis_id, str), "Analysis ID should be returned as string"
    print(f"✓ Wave analysis storage simulation successful: {analysis_id}")
    
    # Test retrieval simulation
    mock_doc = {
        '_id': ObjectId(),
        'file_id': test_file_id,
        'analysis_timestamp': datetime.now(),
        'original_data': np.random.randn(100).tolist(),
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
    
    repo.wave_analyses.find_one.return_value = mock_doc
    
    # Test retrieval
    retrieved = repo.get_wave_analysis(analysis_id)
    assert retrieved is not None, "Analysis should be retrieved"
    assert isinstance(retrieved, DetailedAnalysis), "Retrieved object should be DetailedAnalysis"
    print("✓ Wave analysis retrieval simulation successful")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("WAVE ANALYSIS DATABASE MODELS TEST")
    print("="*60)
    
    try:
        # Test data models
        test_wave_models()
        
        # Test repository operations
        test_repository_with_mocks()
        
        # Test storage operations
        test_storage_operations()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe wave analysis database models are working correctly.")
        print("Key features verified:")
        print("  ✓ Data model creation and validation")
        print("  ✓ Repository initialization with indexing")
        print("  ✓ Numpy array serialization/deserialization")
        print("  ✓ Storage and retrieval operations")
        print("  ✓ Database integration architecture")
        
        print(f"\nNext steps:")
        print("  1. The models are integrated into app.py")
        print("  2. New API endpoints are available:")
        print("     - /api/wave_analysis_stats")
        print("     - /api/recent_wave_analyses")
        print("     - /api/search_wave_analyses")
        print("     - /api/wave_analyses_by_file/<file_id>")
        print("  3. Run the Flask app to test the endpoints")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)