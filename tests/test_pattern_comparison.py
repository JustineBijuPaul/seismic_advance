"""
Tests for Pattern Comparison Service

Tests the pattern comparison service functionality including analysis comparison
and educational insights generation.
"""

import unittest
import numpy as np
from datetime import datetime

from wave_analysis.services.pattern_comparison import PatternComparisonService, AnalysisComparison
from wave_analysis.services.wave_pattern_library import WavePatternLibrary
from wave_analysis.models.wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis, ArrivalTimes, MagnitudeEstimate
)


class TestPatternComparisonService(unittest.TestCase):
    """Test cases for PatternComparisonService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = PatternComparisonService()
        
        # Create test wave segments
        p_wave_data = np.sin(2 * np.pi * 8.0 * np.linspace(0, 5, 500))
        self.p_wave = WaveSegment(
            wave_type='P',
            start_time=2.0,
            end_time=7.0,
            data=p_wave_data,
            sampling_rate=100.0,
            peak_amplitude=1.0,
            dominant_frequency=8.0,
            arrival_time=2.5
        )
        
        s_wave_data = 1.5 * np.sin(2 * np.pi * 4.0 * np.linspace(0, 8, 800))
        self.s_wave = WaveSegment(
            wave_type='S',
            start_time=7.0,
            end_time=15.0,
            data=s_wave_data,
            sampling_rate=100.0,
            peak_amplitude=1.5,
            dominant_frequency=4.0,
            arrival_time=8.0
        )
        
        surface_wave_data = 2.0 * np.sin(2 * np.pi * 0.5 * np.linspace(0, 20, 2000))
        self.surface_wave = WaveSegment(
            wave_type='Rayleigh',
            start_time=15.0,
            end_time=35.0,
            data=surface_wave_data,
            sampling_rate=100.0,
            peak_amplitude=2.0,
            dominant_frequency=0.5,
            arrival_time=16.0
        )
        
        # Create test analysis
        original_data = np.concatenate([
            np.zeros(200),  # 2 seconds of quiet
            p_wave_data,
            np.zeros(200),  # 2 seconds between waves
            s_wave_data,
            surface_wave_data
        ])
        
        wave_result = WaveAnalysisResult(
            original_data=original_data,
            sampling_rate=100.0,
            p_waves=[self.p_wave],
            s_waves=[self.s_wave],
            surface_waves=[self.surface_wave],
            metadata={'test': True}
        )
        
        arrival_times = ArrivalTimes(
            p_wave_arrival=2.5,
            s_wave_arrival=8.0,
            sp_time_difference=5.5,
            surface_wave_arrival=16.0
        )
        
        magnitude_estimate = MagnitudeEstimate(
            method='ML',
            magnitude=4.2,
            confidence=0.8,
            wave_type_used='P'
        )
        
        self.detailed_analysis = DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=[magnitude_estimate],
            epicenter_distance=44.0,  # 5.5 * 8 km
            frequency_analysis={
                'P': {'dominant_frequency': 8.0, 'frequency_range': [5, 15]},
                'S': {'dominant_frequency': 4.0, 'frequency_range': [2, 8]},
                'Surface': {'dominant_frequency': 0.5, 'frequency_range': [0.1, 2.0]}
            },
            quality_metrics=None
        )
    
    def test_service_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.service.pattern_library, WavePatternLibrary)
        self.assertGreater(len(self.service.pattern_library.patterns), 0)
    
    def test_analysis_comparison(self):
        """Test complete analysis comparison with library."""
        comparison = self.service.compare_analysis_with_library(self.detailed_analysis)
        
        self.assertIsInstance(comparison, AnalysisComparison)
        self.assertIsInstance(comparison.pattern_matches, dict)
        self.assertIsInstance(comparison.overall_interpretation, str)
        self.assertIsInstance(comparison.educational_insights, list)
        self.assertIsInstance(comparison.unusual_patterns, dict)
        self.assertIsInstance(comparison.confidence_score, float)
        
        # Should have matches for different wave types
        self.assertIn('P', comparison.pattern_matches)
        self.assertIn('S', comparison.pattern_matches)
        self.assertIn('Surface', comparison.pattern_matches)
        
        # Should have meaningful content
        self.assertGreater(len(comparison.overall_interpretation), 50)
        self.assertGreater(len(comparison.educational_insights), 0)
        self.assertGreaterEqual(comparison.confidence_score, 0.0)
        self.assertLessEqual(comparison.confidence_score, 1.0)
    
    def test_educational_explanations(self):
        """Test educational explanations for different wave types."""
        # Test P-wave explanation
        p_characteristics = {'dominant_frequency': 8.0, 'peak_amplitude': 1.0}
        p_explanation = self.service.get_educational_explanation('P', p_characteristics)
        
        self.assertIsInstance(p_explanation, str)
        self.assertGreater(len(p_explanation), 50)
        self.assertIn('P-wave', p_explanation)
        self.assertIn('compressional', p_explanation.lower())
        
        # Test S-wave explanation
        s_characteristics = {'dominant_frequency': 4.0, 'peak_amplitude': 1.5}
        s_explanation = self.service.get_educational_explanation('S', s_characteristics)
        
        self.assertIsInstance(s_explanation, str)
        self.assertGreater(len(s_explanation), 50)
        self.assertIn('S-wave', s_explanation)
        self.assertIn('shear', s_explanation.lower())
        
        # Test Love wave explanation
        love_characteristics = {'dominant_frequency': 0.8}
        love_explanation = self.service.get_educational_explanation('Love', love_characteristics)
        
        self.assertIn('Love', love_explanation)
        self.assertIn('horizontal', love_explanation.lower())
        
        # Test Rayleigh wave explanation
        rayleigh_characteristics = {'dominant_frequency': 0.3}
        rayleigh_explanation = self.service.get_educational_explanation('Rayleigh', rayleigh_characteristics)
        
        self.assertIn('Rayleigh', rayleigh_explanation)
        self.assertIn('elliptical', rayleigh_explanation.lower())
    
    def test_learning_resource_suggestions(self):
        """Test learning resource suggestions."""
        resources = self.service.suggest_learning_resources(self.detailed_analysis)
        
        self.assertIsInstance(resources, list)
        self.assertGreater(len(resources), 0)
        
        for resource in resources:
            self.assertIsInstance(resource, dict)
            self.assertIn('title', resource)
            self.assertIn('description', resource)
            self.assertIn('topics', resource)
            self.assertIn('difficulty', resource)
            
            # Check content quality
            self.assertGreater(len(resource['title']), 5)
            self.assertGreater(len(resource['description']), 20)
            self.assertIsInstance(resource['topics'], list)
            self.assertIn(resource['difficulty'], ['beginner', 'intermediate', 'advanced'])
    
    def test_overall_interpretation_generation(self):
        """Test overall interpretation generation."""
        comparison = self.service.compare_analysis_with_library(self.detailed_analysis)
        interpretation = comparison.overall_interpretation
        
        # Should mention different wave types
        self.assertIn('P-Wave', interpretation)
        self.assertIn('S-Wave', interpretation)
        
        # Should include distance estimate
        self.assertIn('Distance', interpretation)
        self.assertIn('44', interpretation)  # Expected distance
        
        # Should include magnitude
        self.assertIn('Magnitude', interpretation)
        self.assertIn('4.2', interpretation)
    
    def test_educational_insights_generation(self):
        """Test educational insights generation."""
        comparison = self.service.compare_analysis_with_library(self.detailed_analysis)
        insights = comparison.educational_insights
        
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        # Should mention wave identification
        wave_identification_found = any('identified' in insight.lower() for insight in insights)
        self.assertTrue(wave_identification_found)
        
        # Should mention S-P time
        sp_time_found = any('S-P time' in insight for insight in insights)
        self.assertTrue(sp_time_found)
    
    def test_unusual_pattern_identification(self):
        """Test identification of unusual patterns."""
        # Create analysis with unusual characteristics
        unusual_analysis = self.detailed_analysis
        unusual_analysis.frequency_analysis = {
            'P': {'dominant_frequency': 25.0},  # Very high
            'S': {'dominant_frequency': 0.8}    # Very low
        }
        
        comparison = self.service.compare_analysis_with_library(unusual_analysis)
        unusual_patterns = comparison.unusual_patterns
        
        self.assertIsInstance(unusual_patterns, dict)
        
        # Should detect unusual frequencies
        if unusual_patterns:
            # Check that guidance is provided
            for pattern_type, guidance in unusual_patterns.items():
                self.assertIsInstance(guidance, str)
                self.assertGreater(len(guidance), 30)
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        comparison = self.service.compare_analysis_with_library(self.detailed_analysis)
        
        # Should have reasonable confidence
        self.assertGreaterEqual(comparison.confidence_score, 0.0)
        self.assertLessEqual(comparison.confidence_score, 1.0)
        
        # With good matches, should have decent confidence
        if comparison.pattern_matches:
            total_matches = sum(len(matches) for matches in comparison.pattern_matches.values())
            if total_matches > 0:
                self.assertGreater(comparison.confidence_score, 0.1)
    
    def test_frequency_based_insights(self):
        """Test insights based on frequency characteristics."""
        # Test high-frequency P-wave
        high_freq_analysis = self.detailed_analysis
        high_freq_analysis.frequency_analysis['P']['dominant_frequency'] = 15.0
        
        comparison = self.service.compare_analysis_with_library(high_freq_analysis)
        insights = comparison.educational_insights
        
        # Should mention high frequency implications
        high_freq_mentioned = any('high-frequency' in insight.lower() or 'nearby' in insight.lower() 
                                 for insight in insights)
        self.assertTrue(high_freq_mentioned)
        
        # Test low-frequency S-wave
        low_freq_analysis = self.detailed_analysis
        low_freq_analysis.frequency_analysis['S']['dominant_frequency'] = 2.0
        
        comparison = self.service.compare_analysis_with_library(low_freq_analysis)
        insights = comparison.educational_insights
        
        # Should mention low frequency implications
        low_freq_mentioned = any('low-frequency' in insight.lower() or 'distant' in insight.lower() 
                                for insight in insights)
        self.assertTrue(low_freq_mentioned)
    
    def test_pattern_matching_quality(self):
        """Test quality of pattern matching."""
        comparison = self.service.compare_analysis_with_library(self.detailed_analysis)
        
        # Check P-wave matches
        if 'P' in comparison.pattern_matches:
            p_matches = comparison.pattern_matches['P']
            if p_matches:
                best_p_match = max(p_matches, key=lambda x: x.similarity_score)
                
                # Should have reasonable similarity for typical P-wave
                self.assertGreater(best_p_match.similarity_score, 0.3)
                
                # Should have matching features
                self.assertGreater(len(best_p_match.matching_features), 0)
                
                # Should have educational content
                self.assertGreater(len(best_p_match.educational_notes), 50)
        
        # Check S-wave matches
        if 'S' in comparison.pattern_matches:
            s_matches = comparison.pattern_matches['S']
            if s_matches:
                best_s_match = max(s_matches, key=lambda x: x.similarity_score)
                self.assertGreater(best_s_match.similarity_score, 0.2)
    
    def test_interpretation_accuracy(self):
        """Test accuracy of interpretations."""
        comparison = self.service.compare_analysis_with_library(self.detailed_analysis)
        
        # Check that interpretations mention appropriate confidence levels
        interpretation = comparison.overall_interpretation
        
        # Should mention wave types correctly
        self.assertIn('P-Wave', interpretation)
        self.assertIn('S-Wave', interpretation)
        
        # Should provide distance estimate
        distance_mentioned = 'distance' in interpretation.lower() or 'km' in interpretation.lower()
        self.assertTrue(distance_mentioned)
        
        # Should mention magnitude
        magnitude_mentioned = 'magnitude' in interpretation.lower()
        self.assertTrue(magnitude_mentioned)
    
    def test_educational_content_accuracy(self):
        """Test accuracy of educational content."""
        # Test P-wave educational content
        p_explanation = self.service.get_educational_explanation('P', {'dominant_frequency': 8.0})
        
        # Should contain accurate information
        self.assertIn('fastest', p_explanation.lower())
        self.assertIn('first', p_explanation.lower())
        self.assertIn('compressional', p_explanation.lower())
        
        # Test S-wave educational content
        s_explanation = self.service.get_educational_explanation('S', {'dominant_frequency': 4.0})
        
        # Should contain accurate information
        self.assertIn('after', s_explanation.lower())
        self.assertIn('larger amplitude', s_explanation.lower())
        self.assertIn('shear', s_explanation.lower())
        self.assertIn('liquid', s_explanation.lower())


class TestPatternComparisonIntegration(unittest.TestCase):
    """Integration tests for pattern comparison with real-world scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = PatternComparisonService()
    
    def test_local_earthquake_scenario(self):
        """Test pattern comparison for local earthquake scenario."""
        # Create local earthquake characteristics
        local_p_wave = WaveSegment(
            wave_type='P',
            start_time=1.0,
            end_time=3.0,
            data=np.sin(2 * np.pi * 12.0 * np.linspace(0, 2, 200)),
            sampling_rate=100.0,
            peak_amplitude=0.8,
            dominant_frequency=12.0,  # High frequency for local
            arrival_time=1.2
        )
        
        local_s_wave = WaveSegment(
            wave_type='S',
            start_time=3.0,
            end_time=8.0,
            data=1.2 * np.sin(2 * np.pi * 6.0 * np.linspace(0, 5, 500)),
            sampling_rate=100.0,
            peak_amplitude=1.2,
            dominant_frequency=6.0,
            arrival_time=3.5
        )
        
        # Short S-P time for local event
        arrival_times = ArrivalTimes(
            p_wave_arrival=1.2,
            s_wave_arrival=3.5,
            sp_time_difference=2.3,  # Short for local
            surface_wave_arrival=0.0
        )
        
        wave_result = WaveAnalysisResult(
            original_data=np.concatenate([local_p_wave.data, local_s_wave.data]),
            sampling_rate=100.0,
            p_waves=[local_p_wave],
            s_waves=[local_s_wave],
            surface_waves=[],
            metadata={'scenario': 'local'}
        )
        
        local_analysis = DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=[],
            epicenter_distance=18.4,  # 2.3 * 8 km
            frequency_analysis={
                'P': {'dominant_frequency': 12.0},
                'S': {'dominant_frequency': 6.0}
            },
            quality_metrics=None
        )
        
        comparison = self.service.compare_analysis_with_library(local_analysis)
        
        # Should identify as local earthquake
        insights = comparison.educational_insights
        local_indicators = any('nearby' in insight.lower() or 'local' in insight.lower() 
                              for insight in insights)
        self.assertTrue(local_indicators)
    
    def test_distant_earthquake_scenario(self):
        """Test pattern comparison for distant earthquake scenario."""
        # Create distant earthquake characteristics
        distant_p_wave = WaveSegment(
            wave_type='P',
            start_time=5.0,
            end_time=15.0,
            data=0.6 * np.sin(2 * np.pi * 3.0 * np.linspace(0, 10, 1000)),
            sampling_rate=100.0,
            peak_amplitude=0.6,
            dominant_frequency=3.0,  # Low frequency for distant
            arrival_time=6.0
        )
        
        distant_s_wave = WaveSegment(
            wave_type='S',
            start_time=25.0,
            end_time=45.0,
            data=0.8 * np.sin(2 * np.pi * 1.5 * np.linspace(0, 20, 2000)),
            sampling_rate=100.0,
            peak_amplitude=0.8,
            dominant_frequency=1.5,
            arrival_time=26.0
        )
        
        # Long S-P time for distant event
        arrival_times = ArrivalTimes(
            p_wave_arrival=6.0,
            s_wave_arrival=26.0,
            sp_time_difference=20.0,  # Long for distant
            surface_wave_arrival=45.0
        )
        
        wave_result = WaveAnalysisResult(
            original_data=np.concatenate([distant_p_wave.data, distant_s_wave.data]),
            sampling_rate=100.0,
            p_waves=[distant_p_wave],
            s_waves=[distant_s_wave],
            surface_waves=[],
            metadata={'scenario': 'distant'}
        )
        
        distant_analysis = DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=[],
            epicenter_distance=160.0,  # 20.0 * 8 km
            frequency_analysis={
                'P': {'dominant_frequency': 3.0},
                'S': {'dominant_frequency': 1.5}
            },
            quality_metrics=None
        )
        
        comparison = self.service.compare_analysis_with_library(distant_analysis)
        
        # Should identify as distant earthquake
        insights = comparison.educational_insights
        distant_indicators = any('distant' in insight.lower() or 'far' in insight.lower() 
                                for insight in insights)
        self.assertTrue(distant_indicators)


if __name__ == '__main__':
    unittest.main()