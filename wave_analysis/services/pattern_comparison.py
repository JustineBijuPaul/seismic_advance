"""
Pattern Comparison Service

This module provides functionality to compare user wave analysis results
with the wave pattern library for educational purposes.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from ..models.wave_models import WaveSegment, WaveAnalysisResult, DetailedAnalysis
from .wave_pattern_library import WavePatternLibrary, PatternComparison, WavePattern


@dataclass
class AnalysisComparison:
    """Results of comparing complete analysis with library patterns."""
    analysis_id: str
    pattern_matches: Dict[str, List[PatternComparison]]  # wave_type -> comparisons
    overall_interpretation: str
    educational_insights: List[str]
    unusual_patterns: Dict[str, str]
    confidence_score: float


class PatternComparisonService:
    """
    Service for comparing wave analysis results with educational patterns.
    
    Provides educational insights and interpretations based on pattern matching.
    """
    
    def __init__(self, pattern_library: Optional[WavePatternLibrary] = None):
        """
        Initialize the pattern comparison service.
        
        Args:
            pattern_library: Optional pre-initialized pattern library
        """
        self.pattern_library = pattern_library or WavePatternLibrary()
    
    def compare_analysis_with_library(self, analysis: DetailedAnalysis) -> AnalysisComparison:
        """
        Compare complete wave analysis with pattern library.
        
        Args:
            analysis: Detailed wave analysis results
            
        Returns:
            Comprehensive comparison results with educational insights
        """
        pattern_matches = {}
        all_comparisons = []
        
        # Compare P-waves
        if analysis.wave_result.p_waves:
            p_comparisons = []
            for p_wave in analysis.wave_result.p_waves:
                comparisons = self.pattern_library.compare_with_library(p_wave, max_comparisons=3)
                p_comparisons.extend(comparisons)
            pattern_matches['P'] = p_comparisons
            all_comparisons.extend(p_comparisons)
        
        # Compare S-waves
        if analysis.wave_result.s_waves:
            s_comparisons = []
            for s_wave in analysis.wave_result.s_waves:
                comparisons = self.pattern_library.compare_with_library(s_wave, max_comparisons=3)
                s_comparisons.extend(comparisons)
            pattern_matches['S'] = s_comparisons
            all_comparisons.extend(s_comparisons)
        
        # Compare surface waves
        if analysis.wave_result.surface_waves:
            surface_comparisons = []
            for surface_wave in analysis.wave_result.surface_waves:
                comparisons = self.pattern_library.compare_with_library(surface_wave, max_comparisons=3)
                surface_comparisons.extend(comparisons)
            pattern_matches['Surface'] = surface_comparisons
            all_comparisons.extend(surface_comparisons)
        
        # Generate overall interpretation
        overall_interpretation = self._generate_overall_interpretation(analysis, pattern_matches)
        
        # Generate educational insights
        educational_insights = self._generate_educational_insights(analysis, pattern_matches)
        
        # Check for unusual patterns
        unusual_patterns = self._identify_unusual_patterns(analysis)
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(all_comparisons)
        
        return AnalysisComparison(
            analysis_id=f"analysis_{hash(str(analysis))}",
            pattern_matches=pattern_matches,
            overall_interpretation=overall_interpretation,
            educational_insights=educational_insights,
            unusual_patterns=unusual_patterns,
            confidence_score=confidence_score
        )
    
    def get_educational_explanation(self, wave_type: str, characteristics: Dict[str, Any]) -> str:
        """
        Get educational explanation for specific wave characteristics.
        
        Args:
            wave_type: Type of wave (P, S, Love, Rayleigh)
            characteristics: Dictionary of wave characteristics
            
        Returns:
            Educational explanation text
        """
        explanations = {
            'P': self._get_p_wave_explanation(characteristics),
            'S': self._get_s_wave_explanation(characteristics),
            'Love': self._get_love_wave_explanation(characteristics),
            'Rayleigh': self._get_rayleigh_wave_explanation(characteristics)
        }
        
        return explanations.get(wave_type, "Unknown wave type")
    
    def suggest_learning_resources(self, analysis: DetailedAnalysis) -> List[Dict[str, str]]:
        """
        Suggest learning resources based on analysis results.
        
        Args:
            analysis: Wave analysis results
            
        Returns:
            List of learning resource suggestions
        """
        resources = []
        
        # Basic wave identification resources
        if analysis.wave_result.p_waves or analysis.wave_result.s_waves:
            resources.append({
                'title': 'Seismic Wave Basics',
                'description': 'Learn about P-waves and S-waves, their characteristics and how to identify them',
                'topics': ['wave_propagation', 'arrival_times', 'amplitude_differences'],
                'difficulty': 'beginner'
            })
        
        # Surface wave resources
        if analysis.wave_result.surface_waves:
            resources.append({
                'title': 'Surface Wave Analysis',
                'description': 'Understanding Love and Rayleigh waves, dispersion, and surface wave magnitude',
                'topics': ['dispersion', 'group_velocity', 'surface_wave_magnitude'],
                'difficulty': 'intermediate'
            })
        
        # Magnitude estimation resources
        if hasattr(analysis, 'magnitude_estimates') and analysis.magnitude_estimates:
            resources.append({
                'title': 'Earthquake Magnitude Scales',
                'description': 'Learn about different magnitude scales and how they are calculated',
                'topics': ['local_magnitude', 'body_wave_magnitude', 'surface_wave_magnitude'],
                'difficulty': 'intermediate'
            })
        
        # Advanced analysis resources
        if hasattr(analysis, 'frequency_analysis') and analysis.frequency_analysis:
            resources.append({
                'title': 'Frequency Domain Analysis',
                'description': 'Understanding spectral analysis of seismic waves',
                'topics': ['fourier_transform', 'spectrograms', 'frequency_content'],
                'difficulty': 'advanced'
            })
        
        return resources
    
    def _generate_overall_interpretation(self, analysis: DetailedAnalysis, 
                                       pattern_matches: Dict[str, List[PatternComparison]]) -> str:
        """Generate overall interpretation of the analysis."""
        
        interpretation = "Wave Analysis Interpretation:\n\n"
        
        # Analyze P-waves
        if 'P' in pattern_matches and pattern_matches['P']:
            best_p_match = max(pattern_matches['P'], key=lambda x: x.similarity_score)
            interpretation += f"P-Wave Analysis: {best_p_match.interpretation}\n\n"
        
        # Analyze S-waves
        if 'S' in pattern_matches and pattern_matches['S']:
            best_s_match = max(pattern_matches['S'], key=lambda x: x.similarity_score)
            interpretation += f"S-Wave Analysis: {best_s_match.interpretation}\n\n"
        
        # Analyze surface waves
        if 'Surface' in pattern_matches and pattern_matches['Surface']:
            best_surface_match = max(pattern_matches['Surface'], key=lambda x: x.similarity_score)
            interpretation += f"Surface Wave Analysis: {best_surface_match.interpretation}\n\n"
        
        # Overall earthquake characteristics
        if hasattr(analysis, 'arrival_times') and analysis.arrival_times:
            s_p_time = analysis.arrival_times.sp_time_difference
            if s_p_time > 0:
                distance_km = s_p_time * 8  # Rough approximation
                interpretation += f"Estimated Distance: Approximately {distance_km:.0f} km from source "
                interpretation += f"(based on S-P time of {s_p_time:.1f} seconds)\n\n"
        
        # Magnitude information
        if hasattr(analysis, 'magnitude_estimates') and analysis.magnitude_estimates:
            mag_est = analysis.magnitude_estimates[0]  # Use first estimate
            interpretation += f"Magnitude Estimate: {mag_est.magnitude:.1f} ({mag_est.method}) "
            interpretation += f"with {mag_est.confidence:.0%} confidence\n\n"
        
        return interpretation
    
    def _generate_educational_insights(self, analysis: DetailedAnalysis,
                                     pattern_matches: Dict[str, List[PatternComparison]]) -> List[str]:
        """Generate educational insights from the analysis."""
        
        insights = []
        
        # Wave identification insights
        wave_types_found = []
        if analysis.wave_result.p_waves:
            wave_types_found.append("P-waves")
        if analysis.wave_result.s_waves:
            wave_types_found.append("S-waves")
        if analysis.wave_result.surface_waves:
            wave_types_found.append("surface waves")
        
        if wave_types_found:
            insights.append(f"Successfully identified {', '.join(wave_types_found)} in this earthquake recording")
        
        # Timing insights
        if hasattr(analysis, 'arrival_times') and analysis.arrival_times:
            if analysis.arrival_times.p_wave_arrival > 0 and analysis.arrival_times.s_wave_arrival > 0:
                insights.append(
                    "The S-P time difference can be used to estimate distance to the earthquake epicenter. "
                    "Each second of S-P time represents approximately 8 km of distance."
                )
        
        # Frequency insights
        if hasattr(analysis, 'frequency_analysis') and analysis.frequency_analysis:
            for wave_type, freq_data in analysis.frequency_analysis.items():
                dominant_freq = freq_data.get('dominant_frequency', 0)
                if dominant_freq > 0:
                    if wave_type.upper() == 'P' and dominant_freq > 10:
                        insights.append(
                            f"High-frequency P-waves ({dominant_freq:.1f} Hz) suggest a nearby or shallow earthquake"
                        )
                    elif wave_type.upper() == 'S' and dominant_freq < 5:
                        insights.append(
                            f"Low-frequency S-waves ({dominant_freq:.1f} Hz) are typical of larger, more distant earthquakes"
                        )
        
        # Pattern matching insights
        for wave_type, comparisons in pattern_matches.items():
            if comparisons:
                best_match = max(comparisons, key=lambda x: x.similarity_score)
                if best_match.similarity_score > 0.7:
                    insights.append(f"The {wave_type}-wave pattern closely matches typical earthquake signatures")
                elif best_match.similarity_score < 0.4:
                    insights.append(f"The {wave_type}-wave pattern shows unusual characteristics that warrant further investigation")
        
        return insights
    
    def _identify_unusual_patterns(self, analysis: DetailedAnalysis) -> Dict[str, str]:
        """Identify unusual patterns in the analysis."""
        
        # Convert analysis to dictionary format for the library method
        analysis_dict = {}
        
        if hasattr(analysis, 'frequency_analysis'):
            analysis_dict['frequency_analysis'] = analysis.frequency_analysis
        
        if hasattr(analysis, 'arrival_times'):
            analysis_dict['arrival_times'] = {
                'p_wave_arrival': analysis.arrival_times.p_wave_arrival,
                's_wave_arrival': analysis.arrival_times.s_wave_arrival
            }
        
        # Extract wave amplitudes
        wave_amplitudes = {}
        if analysis.wave_result.p_waves:
            wave_amplitudes['P'] = max([w.peak_amplitude for w in analysis.wave_result.p_waves])
        if analysis.wave_result.s_waves:
            wave_amplitudes['S'] = max([w.peak_amplitude for w in analysis.wave_result.s_waves])
        
        if wave_amplitudes:
            analysis_dict['wave_amplitudes'] = wave_amplitudes
        
        return self.pattern_library.get_unusual_pattern_guidance(analysis_dict)
    
    def _calculate_overall_confidence(self, comparisons: List[PatternComparison]) -> float:
        """Calculate overall confidence score for the analysis."""
        
        if not comparisons:
            return 0.0
        
        # Use weighted average of similarity scores
        total_weight = 0
        weighted_sum = 0
        
        for comparison in comparisons:
            weight = 1.0  # Could be adjusted based on wave type importance
            weighted_sum += comparison.similarity_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_p_wave_explanation(self, characteristics: Dict[str, Any]) -> str:
        """Get educational explanation for P-wave characteristics."""
        
        explanation = (
            "P-waves (Primary or Pressure waves) are the fastest seismic waves and always arrive first. "
            "They are compressional waves that alternately compress and expand the material they travel through. "
        )
        
        frequency = characteristics.get('dominant_frequency', 0)
        if frequency > 0:
            if frequency > 15:
                explanation += f"The high frequency content ({frequency:.1f} Hz) suggests a nearby or shallow source. "
            elif frequency < 3:
                explanation += f"The low frequency content ({frequency:.1f} Hz) suggests a distant or deep source. "
            else:
                explanation += f"The frequency content ({frequency:.1f} Hz) is typical for regional earthquakes. "
        
        amplitude = characteristics.get('peak_amplitude', 0)
        if amplitude > 0:
            explanation += f"P-waves typically have smaller amplitudes than S-waves, which helps in identification. "
        
        return explanation
    
    def _get_s_wave_explanation(self, characteristics: Dict[str, Any]) -> str:
        """Get educational explanation for S-wave characteristics."""
        
        explanation = (
            "S-waves (Secondary or Shear waves) arrive after P-waves and typically have larger amplitudes. "
            "They are shear waves that move material perpendicular to their direction of travel. "
            "S-waves cannot travel through liquids, which is why they don't pass through the Earth's outer core. "
        )
        
        frequency = characteristics.get('dominant_frequency', 0)
        if frequency > 0:
            if frequency > 10:
                explanation += f"The high frequency content ({frequency:.1f} Hz) is unusual for S-waves and may indicate special conditions. "
            elif frequency < 2:
                explanation += f"The low frequency content ({frequency:.1f} Hz) suggests a large or distant earthquake. "
            else:
                explanation += f"The frequency content ({frequency:.1f} Hz) is typical for S-waves. "
        
        return explanation
    
    def _get_love_wave_explanation(self, characteristics: Dict[str, Any]) -> str:
        """Get educational explanation for Love wave characteristics."""
        
        explanation = (
            "Love waves are surface waves that cause horizontal ground motion perpendicular to the direction "
            "of wave propagation. They are typically faster than Rayleigh waves and show strong dispersion, "
            "meaning different frequencies travel at different speeds. "
        )
        
        frequency = characteristics.get('dominant_frequency', 0)
        if frequency > 0:
            explanation += f"The dominant frequency of {frequency:.2f} Hz is typical for Love waves from regional earthquakes. "
        
        explanation += "Love waves are important for determining surface wave magnitude (Ms)."
        
        return explanation
    
    def _get_rayleigh_wave_explanation(self, characteristics: Dict[str, Any]) -> str:
        """Get educational explanation for Rayleigh wave characteristics."""
        
        explanation = (
            "Rayleigh waves are surface waves that cause elliptical particle motion in the vertical plane. "
            "They typically have the largest amplitudes and are responsible for much of the damage in earthquakes. "
            "Rayleigh waves show very strong dispersion and can be observed for hours after large earthquakes. "
        )
        
        frequency = characteristics.get('dominant_frequency', 0)
        if frequency > 0:
            explanation += f"The dominant frequency of {frequency:.2f} Hz indicates the period and helps determine magnitude. "
        
        explanation += "Rayleigh waves are crucial for surface wave magnitude (Ms) calculations and provide information about the earthquake source."
        
        return explanation