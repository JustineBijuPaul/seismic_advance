"""
Tests for Wave Pattern Library

Tests the wave pattern library functionality including pattern creation,
comparison, and educational content accuracy.
"""

import unittest
import numpy as np
from datetime import datetime
import tempfile
import os

from wave_analysis.services.wave_pattern_library import (
    WavePatternLibrary, WavePattern, WavePatternType, PatternCategory,
    PatternComparison
)
from wave_analysis.models.wave_models import WaveSegment


class TestWavePatternLibrary(unittest.TestCase):
    """Test cases for WavePatternLibrary class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.library = WavePatternLibrary()
        
        # Create test wave segment
        test_data = np.sin(2 * np.pi * 5.0 * np.linspace(0, 10, 1000))
        self.test_wave = WaveSegment(
            wave_type='P',
            start_time=0.0,
            end_time=10.0,
            data=test_data,
            sampling_rate=100.0,
            peak_amplitude=1.0,
            dominant_frequency=5.0,
            arrival_time=2.0
        )
    
    def test_library_initialization(self):
        """Test that library initializes with patterns."""
        self.assertGreater(len(self.library.patterns), 0)
        self.assertIn('basic_p_wave_001', self.library.patterns)
        self.assertIn('basic_s_wave_001', self.library.patterns)
        self.assertIn('basic_love_wave_001', self.library.patterns)
        self.assertIn('basic_rayleigh_wave_001', self.library.patterns)
    
    def test_pattern_structure(self):
        """Test that patterns have correct structure."""
        pattern = self.library.get_pattern('basic_p_wave_001')
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, WavePatternType.TYPICAL_P_WAVE)
        self.assertEqual(pattern.category, PatternCategory.BASIC_WAVES)
        self.assertIsInstance(pattern.synthetic_data, np.ndarray)
        self.assertGreater(len(pattern.synthetic_data), 0)
        self.assertIsInstance(pattern.educational_text, str)
        self.assertGreater(len(pattern.educational_text), 50)
        self.assertIsInstance(pattern.interpretation_guide, str)
        self.assertGreater(len(pattern.interpretation_guide), 30)
    
    def test_get_patterns_by_category(self):
        """Test filtering patterns by category."""
        basic_patterns = self.library.get_patterns_by_category(PatternCategory.BASIC_WAVES)
        self.assertGreater(len(basic_patterns), 0)
        
        for pattern in basic_patterns:
            self.assertEqual(pattern.category, PatternCategory.BASIC_WAVES)
    
    def test_get_patterns_by_type(self):
        """Test filtering patterns by type."""
        p_patterns = self.library.get_patterns_by_type(WavePatternType.TYPICAL_P_WAVE)
        self.assertGreater(len(p_patterns), 0)
        
        for pattern in p_patterns:
            self.assertEqual(pattern.pattern_type, WavePatternType.TYPICAL_P_WAVE)
    
    def test_pattern_comparison(self):
        """Test wave pattern comparison functionality."""
        comparisons = self.library.compare_with_library(self.test_wave, max_comparisons=3)
        
        self.assertIsInstance(comparisons, list)
        self.assertLessEqual(len(comparisons), 3)
        
        if comparisons:
            comparison = comparisons[0]
            self.assertIsInstance(comparison, PatternComparison)
            self.assertIsInstance(comparison.similarity_score, float)
            self.assertGreaterEqual(comparison.similarity_score, 0.0)
            self.assertLessEqual(comparison.similarity_score, 1.0)
            self.assertIsInstance(comparison.matching_features, list)
            self.assertIsInstance(comparison.differences, list)
            self.assertIsInstance(comparison.interpretation, str)
            self.assertIsInstance(comparison.educational_notes, str)
    
    def test_similarity_calculation(self):
        """Test similarity calculation between waves."""
        # Create a wave similar to P-wave pattern
        p_pattern = self.library.get_pattern('basic_p_wave_001')
        
        # Create similar wave
        similar_wave = WaveSegment(
            wave_type='P',
            start_time=0.0,
            end_time=10.0,
            data=p_pattern.synthetic_data[:len(self.test_wave.data)],
            sampling_rate=100.0,
            peak_amplitude=1.0,
            dominant_frequency=8.0,  # Close to pattern's 8.0 Hz
            arrival_time=2.0
        )
        
        similarity = self.library._calculate_similarity(similar_wave, p_pattern)
        self.assertGreater(similarity, 0.3)  # Should have reasonable similarity
    
    def test_feature_matching(self):
        """Test feature matching between waves and patterns."""
        p_pattern = self.library.get_pattern('basic_p_wave_001')
        
        # Test wave with matching frequency
        matching_wave = WaveSegment(
            wave_type='P',
            start_time=0.0,
            end_time=10.0,
            data=self.test_wave.data,
            sampling_rate=100.0,
            peak_amplitude=1.0,
            dominant_frequency=8.0,  # Matches pattern
            arrival_time=2.0
        )
        
        matching_features = self.library._find_matching_features(matching_wave, p_pattern)
        self.assertIsInstance(matching_features, list)
        
        # Should find frequency match
        freq_match_found = any('Frequency' in feature for feature in matching_features)
        self.assertTrue(freq_match_found)
    
    def test_unusual_pattern_guidance(self):
        """Test unusual pattern guidance generation."""
        # Test with unusual frequency
        analysis_results = {
            'frequency_analysis': {
                'P': {'dominant_frequency': 25.0},  # Very high for P-wave
                'S': {'dominant_frequency': 0.5}    # Very low for S-wave
            },
            'wave_amplitudes': {
                'P': 1.0,
                'S': 0.8  # Unusually low S/P ratio
            }
        }
        
        guidance = self.library.get_unusual_pattern_guidance(analysis_results)
        self.assertIsInstance(guidance, dict)
        
        # Should detect high frequency P-wave
        self.assertIn('high_freq_p', guidance)
        self.assertIn('high P-wave frequencies', guidance['high_freq_p'])
    
    def test_synthetic_data_generation(self):
        """Test synthetic wave data generation methods."""
        # Test P-wave generation
        p_data = self.library._generate_p_wave_synthetic(10.0, 100.0, 8.0, 1.0)
        self.assertEqual(len(p_data), 1000)  # 10 seconds at 100 Hz
        self.assertGreater(np.max(np.abs(p_data)), 0.1)  # Has significant amplitude
        
        # Test S-wave generation
        s_data = self.library._generate_s_wave_synthetic(15.0, 100.0, 4.0, 1.5)
        self.assertEqual(len(s_data), 1500)  # 15 seconds at 100 Hz
        self.assertGreater(np.max(np.abs(s_data)), 0.1)
        
        # Test Love wave generation
        love_data = self.library._generate_love_wave_synthetic(30.0, 100.0, 0.5, 2.0)
        self.assertEqual(len(love_data), 3000)  # 30 seconds at 100 Hz
        self.assertGreater(np.max(np.abs(love_data)), 0.1)
    
    def test_pattern_export_import(self):
        """Test pattern library export and import functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = os.path.join(temp_dir, 'test_library.json')
            
            # Export library
            success = self.library.export_library(export_path)
            self.assertTrue(success)
            self.assertTrue(os.path.exists(export_path))
            
            # Create new library and import
            new_library = WavePatternLibrary()
            original_count = len(new_library.patterns)
            
            import_success = new_library.load_library(export_path)
            self.assertTrue(import_success)
            
            # Should have loaded patterns (though synthetic data would be empty)
            self.assertGreaterEqual(len(new_library.patterns), original_count)
    
    def test_educational_content_quality(self):
        """Test quality and accuracy of educational content."""
        for pattern in self.library.list_all_patterns():
            # Educational text should be substantial
            self.assertGreater(len(pattern.educational_text), 50)
            
            # Should contain key terms
            if pattern.pattern_type == WavePatternType.TYPICAL_P_WAVE:
                self.assertIn('P-wave', pattern.educational_text)
                self.assertIn('primary', pattern.educational_text.lower())
                self.assertIn('compressional', pattern.educational_text.lower())
            
            elif pattern.pattern_type == WavePatternType.TYPICAL_S_WAVE:
                self.assertIn('S-wave', pattern.educational_text)
                self.assertIn('secondary', pattern.educational_text.lower())
                self.assertIn('shear', pattern.educational_text.lower())
            
            elif pattern.pattern_type == WavePatternType.TYPICAL_LOVE_WAVE:
                self.assertIn('Love', pattern.educational_text)
                self.assertIn('surface', pattern.educational_text.lower())
                self.assertIn('horizontal', pattern.educational_text.lower())
            
            elif pattern.pattern_type == WavePatternType.TYPICAL_RAYLEIGH_WAVE:
                self.assertIn('Rayleigh', pattern.educational_text)
                self.assertIn('surface', pattern.educational_text.lower())
                self.assertIn('elliptical', pattern.educational_text.lower())
            
            # Interpretation guide should provide actionable guidance
            self.assertGreater(len(pattern.interpretation_guide), 30)
            self.assertIn('Look for', pattern.interpretation_guide)
    
    def test_pattern_parameters_validity(self):
        """Test that pattern parameters are realistic."""
        for pattern in self.library.list_all_patterns():
            # Check frequency ranges
            dom_freq = pattern.typical_parameters.get('dominant_frequency', 0)
            if dom_freq > 0:
                if pattern.pattern_type == WavePatternType.TYPICAL_P_WAVE:
                    self.assertGreater(dom_freq, 1.0)  # P-waves should be > 1 Hz
                    self.assertLess(dom_freq, 50.0)    # But < 50 Hz for typical
                
                elif pattern.pattern_type == WavePatternType.TYPICAL_S_WAVE:
                    self.assertGreater(dom_freq, 0.5)  # S-waves should be > 0.5 Hz
                    self.assertLess(dom_freq, 20.0)    # But < 20 Hz for typical
                
                elif pattern.pattern_type in [WavePatternType.TYPICAL_LOVE_WAVE, 
                                             WavePatternType.TYPICAL_RAYLEIGH_WAVE]:
                    self.assertGreater(dom_freq, 0.01)  # Surface waves > 0.01 Hz
                    self.assertLess(dom_freq, 5.0)      # But < 5 Hz for typical
            
            # Check durations
            duration = pattern.typical_parameters.get('duration', 0)
            if duration > 0:
                self.assertGreater(duration, 0.5)   # At least 0.5 seconds
                self.assertLess(duration, 300.0)    # Less than 5 minutes
            
            # Check amplitudes
            amplitude = pattern.typical_parameters.get('peak_amplitude', 0)
            if amplitude > 0:
                self.assertGreater(amplitude, 0.01)  # Reasonable minimum
                self.assertLess(amplitude, 100.0)    # Reasonable maximum


class TestPatternComparison(unittest.TestCase):
    """Test cases for pattern comparison functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.library = WavePatternLibrary()
        
        # Create test P-wave similar to library pattern
        self.p_wave = WaveSegment(
            wave_type='P',
            start_time=0.0,
            end_time=5.0,
            data=np.sin(2 * np.pi * 8.0 * np.linspace(0, 5, 500)),
            sampling_rate=100.0,
            peak_amplitude=1.0,
            dominant_frequency=8.0,
            arrival_time=1.0
        )
        
        # Create test S-wave
        self.s_wave = WaveSegment(
            wave_type='S',
            start_time=5.0,
            end_time=15.0,
            data=np.sin(2 * np.pi * 4.0 * np.linspace(0, 10, 1000)),
            sampling_rate=100.0,
            peak_amplitude=1.5,
            dominant_frequency=4.0,
            arrival_time=6.0
        )
    
    def test_p_wave_comparison(self):
        """Test P-wave comparison with library."""
        comparisons = self.library.compare_with_library(self.p_wave)
        
        self.assertGreater(len(comparisons), 0)
        
        # Should find P-wave patterns
        p_wave_matches = [c for c in comparisons 
                         if 'P-wave' in c.library_pattern.name or 'p_wave' in c.library_pattern.pattern_id]
        self.assertGreater(len(p_wave_matches), 0)
        
        # Best match should have reasonable similarity
        best_match = max(comparisons, key=lambda x: x.similarity_score)
        self.assertGreater(best_match.similarity_score, 0.2)
    
    def test_s_wave_comparison(self):
        """Test S-wave comparison with library."""
        comparisons = self.library.compare_with_library(self.s_wave)
        
        self.assertGreater(len(comparisons), 0)
        
        # Should find S-wave patterns
        s_wave_matches = [c for c in comparisons 
                         if 'S-wave' in c.library_pattern.name or 's_wave' in c.library_pattern.pattern_id]
        self.assertGreater(len(s_wave_matches), 0)
    
    def test_comparison_interpretation(self):
        """Test that comparisons include meaningful interpretations."""
        comparisons = self.library.compare_with_library(self.p_wave, max_comparisons=1)
        
        if comparisons:
            comparison = comparisons[0]
            
            # Should have interpretation text
            self.assertGreater(len(comparison.interpretation), 20)
            
            # Should have educational notes
            self.assertGreater(len(comparison.educational_notes), 50)
            
            # Should mention confidence level
            confidence_terms = ['high', 'moderate', 'low', 'very high']
            has_confidence = any(term in comparison.interpretation.lower() 
                               for term in confidence_terms)
            self.assertTrue(has_confidence)
    
    def test_unusual_wave_detection(self):
        """Test detection of unusual wave characteristics."""
        # Create unusual high-frequency P-wave
        unusual_data = np.sin(2 * np.pi * 25.0 * np.linspace(0, 3, 300))
        unusual_wave = WaveSegment(
            wave_type='P',
            start_time=0.0,
            end_time=3.0,
            data=unusual_data,
            sampling_rate=100.0,
            peak_amplitude=0.8,
            dominant_frequency=25.0,  # Very high for P-wave
            arrival_time=1.0
        )
        
        comparisons = self.library.compare_with_library(unusual_wave)
        
        # Should still get comparisons but with lower similarity
        self.assertGreater(len(comparisons), 0)
        
        # Check if differences are noted
        best_match = max(comparisons, key=lambda x: x.similarity_score)
        self.assertGreater(len(best_match.differences), 0)


if __name__ == '__main__':
    unittest.main()