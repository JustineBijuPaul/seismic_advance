#!/usr/bin/env python3
"""
Demonstration script for wave analysis database models.

This script shows how to use the WaveAnalysisRepository to store,
retrieve, and query wave analysis results in MongoDB.
"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from wave_analysis.models import (
    WaveAnalysisRepository, create_wave_analysis_repository,
    WaveSegment, WaveAnalysisResult, DetailedAnalysis,
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)


def create_sample_wave_analysis() -> DetailedAnalysis:
    """Create a sample DetailedAnalysis object for demonstration."""
    print("Creating sample wave analysis data...")
    
    # Create sample wave segments
    p_wave = WaveSegment(
        wave_type='P',
        start_time=10.0,
        end_time=15.0,
        data=np.random.randn(500),
        sampling_rate=100.0,
        peak_amplitude=0.8,
        dominant_frequency=8.0,
        arrival_time=12.0,
        confidence=0.9,
        metadata={'detector_version': '1.0', 'snr': 15.2}
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
        confidence=0.85,
        metadata={'detector_version': '1.0', 'snr': 12.8}
    )
    
    love_wave = WaveSegment(
        wave_type='Love',
        start_time=40.0,
        end_time=60.0,
        data=np.random.randn(2000),
        sampling_rate=100.0,
        peak_amplitude=0.6,
        dominant_frequency=2.0,
        arrival_time=42.0,
        confidence=0.75,
        metadata={'detector_version': '1.0', 'snr': 8.5}
    )
    
    # Create wave analysis result
    wave_result = WaveAnalysisResult(
        original_data=np.random.randn(10000),
        sampling_rate=100.0,
        p_waves=[p_wave],
        s_waves=[s_wave],
        surface_waves=[love_wave],
        metadata={
            'station': 'DEMO01',
            'location': 'Demo Location',
            'instrument': 'Demo Seismometer',
            'processing_date': datetime.now().isoformat()
        }
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
            wave_type_used='P',
            metadata={'station_correction': -0.1}
        ),
        MagnitudeEstimate(
            method='Ms',
            magnitude=4.1,
            confidence=0.75,
            wave_type_used='Love',
            metadata={'period': 20.0}
        ),
        MagnitudeEstimate(
            method='Mb',
            magnitude=4.3,
            confidence=0.82,
            wave_type_used='P',
            metadata={'period': 1.0}
        )
    ]
    
    # Create frequency data
    frequencies = np.linspace(0, 50, 100)
    power_spectrum = np.random.exponential(1.0, 100)  # Realistic power spectrum shape
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
        processing_warnings=['Low SNR in surface waves', 'Possible noise contamination']
    )
    
    # Create detailed analysis
    detailed_analysis = DetailedAnalysis(
        wave_result=wave_result,
        arrival_times=arrival_times,
        magnitude_estimates=magnitude_estimates,
        epicenter_distance=120.5,
        frequency_analysis={'P': frequency_data, 'S': frequency_data, 'Love': frequency_data},
        quality_metrics=quality_metrics,
        processing_metadata={
            'processing_time': 45.2,
            'model_versions': {
                'p_wave': '1.0',
                's_wave': '1.1',
                'surface_wave': '1.0'
            },
            'algorithm_parameters': {
                'min_snr': 2.0,
                'detection_threshold': 0.3
            }
        }
    )
    
    print(f"✓ Created sample analysis with {wave_result.total_waves_detected} waves detected")
    print(f"  - Wave types: {', '.join(wave_result.wave_types_detected)}")
    print(f"  - Best magnitude estimate: {detailed_analysis.best_magnitude_estimate.magnitude} ({detailed_analysis.best_magnitude_estimate.method})")
    
    return detailed_analysis


def demonstrate_database_operations():
    """Demonstrate various database operations with the wave analysis repository."""
    print("\n" + "="*60)
    print("WAVE ANALYSIS DATABASE DEMONSTRATION")
    print("="*60)
    
    # Get MongoDB connection details
    mongo_uri = os.getenv('MONGO_URL')
    if not mongo_uri:
        print("❌ Error: MONGO_URL environment variable not set")
        print("Please set MONGO_URL in your .env file")
        return
    
    db_name = 'seismic_quake'
    
    try:
        # Create repository
        print(f"\n1. Connecting to MongoDB...")
        print(f"   URI: {mongo_uri[:20]}...")
        print(f"   Database: {db_name}")
        
        repo = create_wave_analysis_repository(mongo_uri, db_name)
        print("✓ Successfully connected to MongoDB and created repository")
        
        # Create sample data
        print(f"\n2. Creating sample wave analysis data...")
        sample_analysis = create_sample_wave_analysis()
        
        # Store analysis
        print(f"\n3. Storing wave analysis in database...")
        from bson import ObjectId
        sample_file_id = ObjectId()  # Simulate a GridFS file ID
        
        analysis_id = repo.store_wave_analysis(sample_file_id, sample_analysis)
        print(f"✓ Stored analysis with ID: {analysis_id}")
        
        # Retrieve analysis
        print(f"\n4. Retrieving stored analysis...")
        retrieved_analysis = repo.get_wave_analysis(analysis_id)
        
        if retrieved_analysis:
            print("✓ Successfully retrieved analysis")
            print(f"  - Analysis timestamp: {retrieved_analysis.analysis_timestamp}")
            print(f"  - Total waves: {retrieved_analysis.wave_result.total_waves_detected}")
            print(f"  - Quality score: {retrieved_analysis.quality_metrics.analysis_quality_score}")
            print(f"  - Epicenter distance: {retrieved_analysis.epicenter_distance} km")
        else:
            print("❌ Failed to retrieve analysis")
            return
        
        # Test caching
        print(f"\n5. Testing analysis result caching...")
        cache_key = f"demo_analysis_{analysis_id}"
        test_cache_data = {
            'analysis_id': analysis_id,
            'summary': 'Demo analysis cache test',
            'cached_at': datetime.now().isoformat()
        }
        
        repo.cache_analysis_result(cache_key, test_cache_data, ttl_hours=1)
        print("✓ Cached analysis result")
        
        cached_data = repo.get_cached_analysis(cache_key)
        if cached_data:
            print("✓ Successfully retrieved cached data")
            print(f"  - Cache key: {cache_key}")
            print(f"  - Cached at: {cached_data['cached_at']}")
        else:
            print("❌ Failed to retrieve cached data")
        
        # Create additional sample analyses for search testing
        print(f"\n6. Creating additional sample analyses for search testing...")
        additional_analyses = []
        
        for i in range(3):
            # Create variations of the sample analysis
            additional_analysis = create_sample_wave_analysis()
            
            # Modify magnitude estimates for search testing
            for est in additional_analysis.magnitude_estimates:
                est.magnitude += (i * 0.5)  # Create different magnitudes
            
            # Store additional analysis
            additional_file_id = ObjectId()
            additional_id = repo.store_wave_analysis(additional_file_id, additional_analysis)
            additional_analyses.append(additional_id)
        
        print(f"✓ Created {len(additional_analyses)} additional analyses")
        
        # Test search by magnitude
        print(f"\n7. Testing magnitude-based search...")
        search_results = repo.search_analyses_by_magnitude(min_magnitude=4.0, max_magnitude=5.0)
        print(f"✓ Found {len(search_results)} analyses with magnitude 4.0-5.0")
        
        for result in search_results[:2]:  # Show first 2 results
            mag_estimates = result.get('detailed_analysis', {}).get('magnitude_estimates', [])
            if mag_estimates:
                best_mag = max(mag_estimates, key=lambda x: x.get('confidence', 0))
                print(f"  - Analysis: {result['_id']}, Magnitude: {best_mag.get('magnitude', 'N/A')}")
        
        # Test recent analyses
        print(f"\n8. Testing recent analyses retrieval...")
        recent_analyses = repo.get_recent_analyses(limit=5, min_quality_score=0.5)
        print(f"✓ Found {len(recent_analyses)} recent analyses with quality score ≥ 0.5")
        
        # Test analyses by file
        print(f"\n9. Testing file-based analysis retrieval...")
        file_analyses = repo.get_analyses_by_file(sample_file_id)
        print(f"✓ Found {len(file_analyses)} analyses for file {sample_file_id}")
        
        # Get statistics
        print(f"\n10. Getting analysis statistics...")
        stats = repo.get_analysis_statistics()
        print("✓ Analysis Statistics:")
        print(f"   - Total analyses: {stats['total_analyses']}")
        print(f"   - Average quality score: {stats['avg_quality_score']:.3f}")
        print(f"   - Total waves detected: {stats['total_waves_detected']}")
        print(f"   - Wave type distribution: {stats['wave_type_distribution']}")
        
        if stats['latest_analysis']:
            print(f"   - Latest analysis: {stats['latest_analysis']}")
        if stats['oldest_analysis']:
            print(f"   - Oldest analysis: {stats['oldest_analysis']}")
        
        # Clean up - delete test analyses
        print(f"\n11. Cleaning up test data...")
        deleted_count = 0
        
        # Delete main analysis
        if repo.delete_analysis(analysis_id):
            deleted_count += 1
        
        # Delete additional analyses
        for add_id in additional_analyses:
            if repo.delete_analysis(add_id):
                deleted_count += 1
        
        print(f"✓ Cleaned up {deleted_count} test analyses")
        
        # Clear cache
        cache_cleared = repo.clear_expired_cache()
        print(f"✓ Cleared {cache_cleared} expired cache entries")
        
        print(f"\n" + "="*60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe wave analysis database models are working correctly.")
        print("You can now use the WaveAnalysisRepository in your Flask application")
        print("to store, retrieve, and query wave analysis results.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")


if __name__ == '__main__':
    demonstrate_database_operations()