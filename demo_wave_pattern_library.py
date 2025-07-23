#!/usr/bin/env python3
"""
Demo script for Wave Pattern Library functionality

This script demonstrates the wave pattern library and comparison features
for educational purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from wave_analysis.services.wave_pattern_library import WavePatternLibrary, WavePatternType, PatternCategory
from wave_analysis.services.pattern_comparison import PatternComparisonService
from wave_analysis.models.wave_models import WaveSegment, WaveAnalysisResult, DetailedAnalysis, ArrivalTimes


def create_demo_wave_segment(wave_type: str, frequency: float, duration: float = 10.0) -> WaveSegment:
    """Create a demo wave segment for testing."""
    sampling_rate = 100.0
    t = np.linspace(0, duration, int(duration * sampling_rate))
    
    if wave_type.upper() == 'P':
        # P-wave: sharp onset, high frequency
        onset_time = 2.0
        wave_duration = 3.0
        amplitude = 1.0
        
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        wave_t = t[onset_idx:end_idx] - onset_time
        envelope[onset_idx:end_idx] = amplitude * np.exp(-wave_t * 2.0)
        
        data = envelope * np.sin(2 * np.pi * frequency * t)
        
        return WaveSegment(
            wave_type='P',
            start_time=onset_time,
            end_time=onset_time + wave_duration,
            data=data,
            sampling_rate=sampling_rate,
            peak_amplitude=amplitude,
            dominant_frequency=frequency,
            arrival_time=onset_time + 0.2
        )
    
    elif wave_type.upper() == 'S':
        # S-wave: larger amplitude, lower frequency
        onset_time = 5.0
        wave_duration = 8.0
        amplitude = 1.5
        
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        wave_t = t[onset_idx:end_idx] - onset_time
        envelope[onset_idx:end_idx] = amplitude * (1 - np.exp(-wave_t * 1.5)) * np.exp(-wave_t * 0.8)
        
        data = envelope * np.sin(2 * np.pi * frequency * t)
        
        return WaveSegment(
            wave_type='S',
            start_time=onset_time,
            end_time=onset_time + wave_duration,
            data=data,
            sampling_rate=sampling_rate,
            peak_amplitude=amplitude,
            dominant_frequency=frequency,
            arrival_time=onset_time + 0.5
        )
    
    else:
        raise ValueError(f"Unsupported wave type: {wave_type}")


def demo_pattern_library():
    """Demonstrate the wave pattern library functionality."""
    print("=== Wave Pattern Library Demo ===\n")
    
    # Initialize the library
    print("1. Initializing Wave Pattern Library...")
    library = WavePatternLibrary()
    print(f"   Loaded {len(library.patterns)} patterns")
    
    # Show available patterns by category
    print("\n2. Available Pattern Categories:")
    for category in PatternCategory:
        patterns = library.get_patterns_by_category(category)
        print(f"   {category.value}: {len(patterns)} patterns")
        for pattern in patterns:
            print(f"     - {pattern.name}")
    
    # Create a test P-wave
    print("\n3. Creating test P-wave...")
    test_p_wave = create_demo_wave_segment('P', frequency=8.0)
    print(f"   Created P-wave: {test_p_wave.dominant_frequency} Hz, {test_p_wave.peak_amplitude} amplitude")
    
    # Compare with library
    print("\n4. Comparing with pattern library...")
    comparisons = library.compare_with_library(test_p_wave, max_comparisons=3)
    
    for i, comparison in enumerate(comparisons, 1):
        print(f"\n   Match {i}: {comparison.library_pattern.name}")
        print(f"   Similarity: {comparison.similarity_score:.3f}")
        print(f"   Matching features: {', '.join(comparison.matching_features)}")
        if comparison.differences:
            print(f"   Differences: {', '.join(comparison.differences)}")
        print(f"   Interpretation: {comparison.interpretation[:100]}...")
    
    # Show educational content for a specific pattern
    print("\n5. Educational Content Example:")
    p_pattern = library.get_pattern('basic_p_wave_001')
    if p_pattern:
        print(f"   Pattern: {p_pattern.name}")
        print(f"   Description: {p_pattern.description}")
        print(f"   Educational Text: {p_pattern.educational_text[:200]}...")
        print(f"   Key Features: {p_pattern.key_features}")


def demo_pattern_comparison_service():
    """Demonstrate the pattern comparison service."""
    print("\n\n=== Pattern Comparison Service Demo ===\n")
    
    # Initialize service
    print("1. Initializing Pattern Comparison Service...")
    service = PatternComparisonService()
    
    # Create a complete analysis example
    print("\n2. Creating example earthquake analysis...")
    
    # Create wave segments
    p_wave = create_demo_wave_segment('P', frequency=8.0)
    s_wave = create_demo_wave_segment('S', frequency=4.0)
    
    # Create analysis result
    original_data = np.concatenate([p_wave.data, s_wave.data])
    wave_result = WaveAnalysisResult(
        original_data=original_data,
        sampling_rate=100.0,
        p_waves=[p_wave],
        s_waves=[s_wave],
        surface_waves=[],
        metadata={'demo': True}
    )
    
    # Create arrival times
    arrival_times = ArrivalTimes(
        p_wave_arrival=p_wave.arrival_time,
        s_wave_arrival=s_wave.arrival_time,
        sp_time_difference=s_wave.arrival_time - p_wave.arrival_time,
        surface_wave_arrival=0.0
    )
    
    # Create detailed analysis
    detailed_analysis = DetailedAnalysis(
        wave_result=wave_result,
        arrival_times=arrival_times,
        magnitude_estimates=[],
        epicenter_distance=(s_wave.arrival_time - p_wave.arrival_time) * 8.0,  # Rough estimate
        frequency_analysis={
            'P': {'dominant_frequency': p_wave.dominant_frequency},
            'S': {'dominant_frequency': s_wave.dominant_frequency}
        },
        quality_metrics=None
    )
    
    # Compare with library
    print("\n3. Performing comprehensive pattern comparison...")
    comparison = service.compare_analysis_with_library(detailed_analysis)
    
    print(f"   Overall confidence: {comparison.confidence_score:.3f}")
    print(f"   Pattern matches found: {sum(len(matches) for matches in comparison.pattern_matches.values())}")
    
    # Show interpretation
    print("\n4. Overall Interpretation:")
    print(f"   {comparison.overall_interpretation[:300]}...")
    
    # Show educational insights
    print("\n5. Educational Insights:")
    for i, insight in enumerate(comparison.educational_insights, 1):
        print(f"   {i}. {insight}")
    
    # Show unusual patterns if any
    if comparison.unusual_patterns:
        print("\n6. Unusual Pattern Guidance:")
        for pattern_type, guidance in comparison.unusual_patterns.items():
            print(f"   {pattern_type}: {guidance[:100]}...")
    
    # Get learning resources
    print("\n7. Suggested Learning Resources:")
    resources = service.suggest_learning_resources(detailed_analysis)
    for resource in resources:
        print(f"   - {resource['title']} ({resource['difficulty']})")
        print(f"     {resource['description'][:80]}...")


def demo_unusual_patterns():
    """Demonstrate unusual pattern detection."""
    print("\n\n=== Unusual Pattern Detection Demo ===\n")
    
    library = WavePatternLibrary()
    
    # Create unusual high-frequency P-wave
    print("1. Testing unusual high-frequency P-wave...")
    unusual_p_wave = create_demo_wave_segment('P', frequency=25.0)  # Very high frequency
    
    comparisons = library.compare_with_library(unusual_p_wave, max_comparisons=2)
    
    if comparisons:
        best_match = comparisons[0]
        print(f"   Best match: {best_match.library_pattern.name}")
        print(f"   Similarity: {best_match.similarity_score:.3f}")
        print(f"   Differences: {', '.join(best_match.differences)}")
        print(f"   Educational notes: {best_match.educational_notes[:150]}...")
    
    # Test unusual pattern guidance
    print("\n2. Getting guidance for unusual patterns...")
    analysis_results = {
        'frequency_analysis': {
            'P': {'dominant_frequency': 25.0},  # Very high
            'S': {'dominant_frequency': 0.8}    # Very low
        },
        'wave_amplitudes': {
            'P': 1.0,
            'S': 0.8  # Low S/P ratio
        }
    }
    
    guidance = library.get_unusual_pattern_guidance(analysis_results)
    
    if guidance:
        print("   Unusual pattern guidance:")
        for pattern_type, advice in guidance.items():
            print(f"   - {pattern_type}: {advice[:100]}...")
    else:
        print("   No unusual patterns detected in this example.")


def main():
    """Run all demos."""
    try:
        demo_pattern_library()
        demo_pattern_comparison_service()
        demo_unusual_patterns()
        
        print("\n\n=== Demo Complete ===")
        print("The wave pattern library provides:")
        print("- Library of typical earthquake wave patterns")
        print("- Pattern comparison functionality")
        print("- Educational explanations and interpretations")
        print("- Guidance for unusual wave patterns")
        print("- Learning resource suggestions")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()