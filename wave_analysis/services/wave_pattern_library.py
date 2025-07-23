"""
Wave Pattern Library Service

This module provides a library of typical earthquake wave patterns for educational purposes,
including pattern comparison functionality and explanatory text for interpretations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import os
from datetime import datetime

from ..models.wave_models import WaveSegment, WaveAnalysisResult


class WavePatternType(Enum):
    """Types of wave patterns in the library."""
    TYPICAL_P_WAVE = "typical_p_wave"
    TYPICAL_S_WAVE = "typical_s_wave"
    TYPICAL_LOVE_WAVE = "typical_love_wave"
    TYPICAL_RAYLEIGH_WAVE = "typical_rayleigh_wave"
    HIGH_FREQUENCY_P = "high_frequency_p"
    LOW_FREQUENCY_S = "low_frequency_s"
    EMERGENT_P_ARRIVAL = "emergent_p_arrival"
    IMPULSIVE_P_ARRIVAL = "impulsive_p_arrival"
    CLEAR_S_ARRIVAL = "clear_s_arrival"
    COMPLEX_S_ARRIVAL = "complex_s_arrival"
    DISPERSED_SURFACE_WAVES = "dispersed_surface_waves"
    NOISE_CONTAMINATED = "noise_contaminated"
    AFTERSHOCK_SEQUENCE = "aftershock_sequence"
    TELESEISMIC_P = "teleseismic_p"
    REGIONAL_S = "regional_s"


class PatternCategory(Enum):
    """Categories for organizing wave patterns."""
    BASIC_WAVES = "basic_waves"
    ARRIVAL_CHARACTERISTICS = "arrival_characteristics"
    FREQUENCY_VARIATIONS = "frequency_variations"
    COMPLEX_PATTERNS = "complex_patterns"
    UNUSUAL_PATTERNS = "unusual_patterns"


@dataclass
class WavePattern:
    """Represents a wave pattern in the library."""
    pattern_id: str
    pattern_type: WavePatternType
    category: PatternCategory
    name: str
    description: str
    educational_text: str
    interpretation_guide: str
    synthetic_data: np.ndarray
    sampling_rate: float
    duration: float
    key_features: Dict[str, Any]
    typical_parameters: Dict[str, float]
    created_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary for serialization."""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'category': self.category.value,
            'name': self.name,
            'description': self.description,
            'educational_text': self.educational_text,
            'interpretation_guide': self.interpretation_guide,
            'sampling_rate': self.sampling_rate,
            'duration': self.duration,
            'key_features': self.key_features,
            'typical_parameters': self.typical_parameters,
            'created_date': self.created_date.isoformat()
        }


@dataclass
class PatternComparison:
    """Results of comparing user data with library patterns."""
    user_pattern_id: str
    library_pattern: WavePattern
    similarity_score: float
    matching_features: List[str]
    differences: List[str]
    interpretation: str
    educational_notes: str


class WavePatternLibrary:
    """
    Library of typical earthquake wave patterns for educational purposes.
    
    Provides pattern comparison functionality and explanatory text for
    wave pattern interpretations.
    """
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize the wave pattern library.
        
        Args:
            library_path: Optional path to load existing library data
        """
        self.patterns: Dict[str, WavePattern] = {}
        self.library_path = library_path or "static/education/wave_patterns"
        self._initialize_library()
    
    def _initialize_library(self):
        """Initialize the library with typical wave patterns."""
        # Create basic wave patterns
        self._create_basic_wave_patterns()
        
        # Create arrival characteristic patterns
        self._create_arrival_patterns()
        
        # Create frequency variation patterns
        self._create_frequency_patterns()
        
        # Create complex and unusual patterns
        self._create_complex_patterns()
    
    def _create_basic_wave_patterns(self):
        """Create basic P, S, and surface wave patterns."""
        
        # Typical P-wave pattern
        p_wave_data = self._generate_p_wave_synthetic(
            duration=10.0, sampling_rate=100.0, frequency=8.0, amplitude=1.0
        )
        
        p_pattern = WavePattern(
            pattern_id="basic_p_wave_001",
            pattern_type=WavePatternType.TYPICAL_P_WAVE,
            category=PatternCategory.BASIC_WAVES,
            name="Typical P-Wave",
            description="A classic primary wave with sharp onset and high frequency content",
            educational_text=(
                "P-waves (Primary waves) are the fastest seismic waves and arrive first at recording stations. "
                "They are compressional waves that can travel through both solid and liquid materials. "
                "P-waves typically have frequencies between 1-20 Hz and show a sharp, impulsive onset."
            ),
            interpretation_guide=(
                "Look for: Sharp onset, high frequency content (5-15 Hz), first arrival, "
                "compressional motion. P-waves indicate the initial rupture of an earthquake."
            ),
            synthetic_data=p_wave_data,
            sampling_rate=100.0,
            duration=10.0,
            key_features={
                'onset_sharpness': 'high',
                'frequency_range': [5, 15],
                'arrival_order': 'first',
                'wave_type': 'compressional'
            },
            typical_parameters={
                'dominant_frequency': 8.0,
                'peak_amplitude': 1.0,
                'onset_time': 2.0,
                'duration': 3.0
            },
            created_date=datetime.now()
        )
        self.patterns[p_pattern.pattern_id] = p_pattern
        
        # Typical S-wave pattern
        s_wave_data = self._generate_s_wave_synthetic(
            duration=15.0, sampling_rate=100.0, frequency=4.0, amplitude=1.5
        )
        
        s_pattern = WavePattern(
            pattern_id="basic_s_wave_001",
            pattern_type=WavePatternType.TYPICAL_S_WAVE,
            category=PatternCategory.BASIC_WAVES,
            name="Typical S-Wave",
            description="A classic secondary wave with larger amplitude and lower frequency than P-waves",
            educational_text=(
                "S-waves (Secondary waves) arrive after P-waves and have larger amplitudes. "
                "They are shear waves that cannot travel through liquids, only through solid materials. "
                "S-waves typically have frequencies between 1-10 Hz and show more complex waveforms."
            ),
            interpretation_guide=(
                "Look for: Larger amplitude than P-waves, lower frequency content (1-8 Hz), "
                "second arrival, shear motion. S-waves provide information about earthquake magnitude."
            ),
            synthetic_data=s_wave_data,
            sampling_rate=100.0,
            duration=15.0,
            key_features={
                'amplitude_ratio': 'larger_than_p',
                'frequency_range': [1, 8],
                'arrival_order': 'second',
                'wave_type': 'shear'
            },
            typical_parameters={
                'dominant_frequency': 4.0,
                'peak_amplitude': 1.5,
                'onset_time': 5.0,
                'duration': 8.0
            },
            created_date=datetime.now()
        )
        self.patterns[s_pattern.pattern_id] = s_pattern
        
        # Typical Love wave pattern
        love_wave_data = self._generate_love_wave_synthetic(
            duration=30.0, sampling_rate=100.0, frequency=0.5, amplitude=2.0
        )
        
        love_pattern = WavePattern(
            pattern_id="basic_love_wave_001",
            pattern_type=WavePatternType.TYPICAL_LOVE_WAVE,
            category=PatternCategory.BASIC_WAVES,
            name="Typical Love Wave",
            description="A surface wave with horizontal motion and dispersive characteristics",
            educational_text=(
                "Love waves are surface waves that cause horizontal ground motion perpendicular "
                "to the direction of wave propagation. They are typically the fastest surface waves "
                "and show strong dispersion, with longer periods arriving later."
            ),
            interpretation_guide=(
                "Look for: Horizontal motion, dispersive arrival (frequency decreases with time), "
                "long duration, periods typically 10-100 seconds for regional earthquakes."
            ),
            synthetic_data=love_wave_data,
            sampling_rate=100.0,
            duration=30.0,
            key_features={
                'motion_type': 'horizontal',
                'dispersion': 'strong',
                'frequency_range': [0.1, 2.0],
                'wave_type': 'surface'
            },
            typical_parameters={
                'dominant_frequency': 0.5,
                'peak_amplitude': 2.0,
                'onset_time': 15.0,
                'duration': 20.0
            },
            created_date=datetime.now()
        )
        self.patterns[love_pattern.pattern_id] = love_pattern   
     # Typical Rayleigh wave pattern
        rayleigh_wave_data = self._generate_rayleigh_wave_synthetic(
            duration=40.0, sampling_rate=100.0, frequency=0.3, amplitude=2.5
        )
        
        rayleigh_pattern = WavePattern(
            pattern_id="basic_rayleigh_wave_001",
            pattern_type=WavePatternType.TYPICAL_RAYLEIGH_WAVE,
            category=PatternCategory.BASIC_WAVES,
            name="Typical Rayleigh Wave",
            description="A surface wave with elliptical particle motion and strong dispersion",
            educational_text=(
                "Rayleigh waves are surface waves that cause elliptical particle motion in the "
                "vertical plane. They are typically slower than Love waves but have larger amplitudes "
                "and are responsible for much of the damage in earthquakes."
            ),
            interpretation_guide=(
                "Look for: Elliptical motion, strong dispersion, largest amplitudes, "
                "long duration, periods typically 15-300 seconds for regional earthquakes."
            ),
            synthetic_data=rayleigh_wave_data,
            sampling_rate=100.0,
            duration=40.0,
            key_features={
                'motion_type': 'elliptical',
                'dispersion': 'very_strong',
                'frequency_range': [0.05, 1.0],
                'wave_type': 'surface'
            },
            typical_parameters={
                'dominant_frequency': 0.3,
                'peak_amplitude': 2.5,
                'onset_time': 20.0,
                'duration': 30.0
            },
            created_date=datetime.now()
        )
        self.patterns[rayleigh_pattern.pattern_id] = rayleigh_pattern
    
    def _create_arrival_patterns(self):
        """Create patterns showing different arrival characteristics."""
        
        # Emergent P-wave arrival
        emergent_data = self._generate_emergent_p_wave(
            duration=12.0, sampling_rate=100.0, frequency=6.0
        )
        
        emergent_pattern = WavePattern(
            pattern_id="arrival_emergent_p_001",
            pattern_type=WavePatternType.EMERGENT_P_ARRIVAL,
            category=PatternCategory.ARRIVAL_CHARACTERISTICS,
            name="Emergent P-Wave Arrival",
            description="A P-wave with gradual, unclear onset typical of distant earthquakes",
            educational_text=(
                "Emergent P-wave arrivals have gradual onsets that make precise timing difficult. "
                "This is common for distant earthquakes where high frequencies are attenuated "
                "during propagation through the Earth."
            ),
            interpretation_guide=(
                "Look for: Gradual amplitude increase, unclear onset time, lower frequency content, "
                "typical of teleseismic (distant) earthquakes. Timing uncertainty is higher."
            ),
            synthetic_data=emergent_data,
            sampling_rate=100.0,
            duration=12.0,
            key_features={
                'onset_sharpness': 'low',
                'timing_uncertainty': 'high',
                'distance_indicator': 'far',
                'frequency_content': 'attenuated_high_freq'
            },
            typical_parameters={
                'dominant_frequency': 6.0,
                'peak_amplitude': 0.8,
                'onset_time': 3.0,
                'duration': 6.0
            },
            created_date=datetime.now()
        )
        self.patterns[emergent_pattern.pattern_id] = emergent_pattern
        
        # Impulsive P-wave arrival
        impulsive_data = self._generate_impulsive_p_wave(
            duration=8.0, sampling_rate=100.0, frequency=12.0
        )
        
        impulsive_pattern = WavePattern(
            pattern_id="arrival_impulsive_p_001",
            pattern_type=WavePatternType.IMPULSIVE_P_ARRIVAL,
            category=PatternCategory.ARRIVAL_CHARACTERISTICS,
            name="Impulsive P-Wave Arrival",
            description="A P-wave with very sharp, clear onset typical of nearby earthquakes",
            educational_text=(
                "Impulsive P-wave arrivals have very sharp, clear onsets that allow precise timing. "
                "This is typical of nearby earthquakes where high frequencies are preserved "
                "and the signal-to-noise ratio is high."
            ),
            interpretation_guide=(
                "Look for: Very sharp onset, high frequency content, clear timing, "
                "typical of local earthquakes. Excellent for precise location determination."
            ),
            synthetic_data=impulsive_data,
            sampling_rate=100.0,
            duration=8.0,
            key_features={
                'onset_sharpness': 'very_high',
                'timing_uncertainty': 'very_low',
                'distance_indicator': 'near',
                'frequency_content': 'high_freq_preserved'
            },
            typical_parameters={
                'dominant_frequency': 12.0,
                'peak_amplitude': 1.2,
                'onset_time': 1.5,
                'duration': 2.5
            },
            created_date=datetime.now()
        )
        self.patterns[impulsive_pattern.pattern_id] = impulsive_pattern
    
    def _create_frequency_patterns(self):
        """Create patterns showing frequency variations."""
        
        # High-frequency P-wave
        high_freq_data = self._generate_high_frequency_p_wave(
            duration=6.0, sampling_rate=100.0, frequency=18.0
        )
        
        high_freq_pattern = WavePattern(
            pattern_id="freq_high_p_001",
            pattern_type=WavePatternType.HIGH_FREQUENCY_P,
            category=PatternCategory.FREQUENCY_VARIATIONS,
            name="High-Frequency P-Wave",
            description="P-wave with unusually high frequency content",
            educational_text=(
                "High-frequency P-waves (>15 Hz) are typically observed for very shallow, "
                "nearby earthquakes or explosions. The high frequencies indicate minimal "
                "attenuation during wave propagation."
            ),
            interpretation_guide=(
                "Look for: Dominant frequencies >15 Hz, short duration, high amplitude. "
                "May indicate shallow source, nearby event, or artificial explosion."
            ),
            synthetic_data=high_freq_data,
            sampling_rate=100.0,
            duration=6.0,
            key_features={
                'frequency_range': [15, 25],
                'source_depth': 'shallow',
                'attenuation': 'minimal',
                'event_type': 'local_or_explosion'
            },
            typical_parameters={
                'dominant_frequency': 18.0,
                'peak_amplitude': 0.9,
                'onset_time': 1.0,
                'duration': 2.0
            },
            created_date=datetime.now()
        )
        self.patterns[high_freq_pattern.pattern_id] = high_freq_pattern
    
    def _create_complex_patterns(self):
        """Create complex and unusual wave patterns."""
        
        # Noise-contaminated pattern
        noisy_data = self._generate_noisy_wave_pattern(
            duration=20.0, sampling_rate=100.0, noise_level=0.5
        )
        
        noisy_pattern = WavePattern(
            pattern_id="complex_noisy_001",
            pattern_type=WavePatternType.NOISE_CONTAMINATED,
            category=PatternCategory.UNUSUAL_PATTERNS,
            name="Noise-Contaminated Signal",
            description="Seismic signal heavily contaminated with noise",
            educational_text=(
                "Noise-contaminated signals are common in seismic recordings and can make "
                "wave identification challenging. Sources include cultural noise, wind, "
                "ocean waves, and instrumental problems."
            ),
            interpretation_guide=(
                "Look for: High background noise, unclear wave arrivals, multiple false triggers. "
                "Requires careful filtering and analysis. Consider data quality before interpretation."
            ),
            synthetic_data=noisy_data,
            sampling_rate=100.0,
            duration=20.0,
            key_features={
                'noise_level': 'high',
                'signal_clarity': 'poor',
                'analysis_difficulty': 'high',
                'reliability': 'questionable'
            },
            typical_parameters={
                'signal_to_noise_ratio': 2.0,
                'noise_amplitude': 0.5,
                'onset_time': 5.0,
                'duration': 10.0
            },
            created_date=datetime.now()
        )
        self.patterns[noisy_pattern.pattern_id] = noisy_pattern
    
    def _generate_p_wave_synthetic(self, duration: float, sampling_rate: float, 
                                 frequency: float, amplitude: float) -> np.ndarray:
        """Generate synthetic P-wave data."""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # P-wave characteristics: sharp onset, high frequency
        onset_time = 2.0
        wave_duration = 3.0
        
        # Create envelope
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        # Sharp onset with exponential decay
        wave_t = t[onset_idx:end_idx] - onset_time
        envelope[onset_idx:end_idx] = amplitude * np.exp(-wave_t * 2.0)
        
        # High-frequency oscillation
        signal = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        noise = 0.05 * amplitude * np.random.normal(0, 1, len(t))
        
        return signal + noise
    
    def _generate_s_wave_synthetic(self, duration: float, sampling_rate: float,
                                 frequency: float, amplitude: float) -> np.ndarray:
        """Generate synthetic S-wave data."""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # S-wave characteristics: larger amplitude, lower frequency, later arrival
        onset_time = 5.0
        wave_duration = 8.0
        
        # Create envelope
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        # Gradual onset with slower decay
        wave_t = t[onset_idx:end_idx] - onset_time
        envelope[onset_idx:end_idx] = amplitude * (1 - np.exp(-wave_t * 1.5)) * np.exp(-wave_t * 0.8)
        
        # Lower frequency oscillation
        signal = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Add some noise
        noise = 0.03 * amplitude * np.random.normal(0, 1, len(t))
        
        return signal + noise
    
    def _generate_love_wave_synthetic(self, duration: float, sampling_rate: float,
                                    frequency: float, amplitude: float) -> np.ndarray:
        """Generate synthetic Love wave data with dispersion."""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Love wave characteristics: dispersive, long duration
        onset_time = 15.0
        wave_duration = 20.0
        
        # Create dispersive envelope
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        wave_t = t[onset_idx:end_idx] - onset_time
        
        # Dispersive envelope - amplitude builds up then decays
        envelope[onset_idx:end_idx] = amplitude * np.exp(-((wave_t - wave_duration/3) / (wave_duration/4))**2)
        
        # Frequency modulation for dispersion effect
        freq_mod = frequency * (1 + 0.3 * np.exp(-wave_t / 5.0))
        phase = np.zeros_like(t)
        phase[onset_idx:end_idx] = 2 * np.pi * np.cumsum(freq_mod) / sampling_rate
        signal = envelope * np.sin(phase)
        
        # Add some noise
        noise = 0.02 * amplitude * np.random.normal(0, 1, len(t))
        
        return signal + noise
    
    def _generate_rayleigh_wave_synthetic(self, duration: float, sampling_rate: float,
                                        frequency: float, amplitude: float) -> np.ndarray:
        """Generate synthetic Rayleigh wave data with strong dispersion."""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Rayleigh wave characteristics: very dispersive, largest amplitude
        onset_time = 20.0
        wave_duration = 30.0
        
        # Create strongly dispersive envelope
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        wave_t = t[onset_idx:end_idx] - onset_time
        
        # Strong dispersion - frequency decreases significantly with time
        envelope[onset_idx:end_idx] = amplitude * (1 - np.exp(-wave_t / 3.0)) * np.exp(-wave_t / 15.0)
        
        # Strong frequency modulation
        freq_mod = frequency * (1 + 0.8 * np.exp(-wave_t / 8.0))
        phase = np.zeros_like(t)
        phase[onset_idx:end_idx] = 2 * np.pi * np.cumsum(freq_mod) / sampling_rate
        signal = envelope * np.sin(phase)
        
        # Add some noise
        noise = 0.01 * amplitude * np.random.normal(0, 1, len(t))
        
        return signal + noise
    
    def _generate_emergent_p_wave(self, duration: float, sampling_rate: float,
                                frequency: float) -> np.ndarray:
        """Generate emergent P-wave with gradual onset."""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        onset_time = 3.0
        wave_duration = 6.0
        amplitude = 0.8
        
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        wave_t = t[onset_idx:end_idx] - onset_time
        
        # Very gradual onset
        envelope[onset_idx:end_idx] = amplitude * (1 - np.exp(-wave_t / 2.0)) * np.exp(-wave_t / 4.0)
        
        signal = envelope * np.sin(2 * np.pi * frequency * t)
        noise = 0.1 * amplitude * np.random.normal(0, 1, len(t))
        
        return signal + noise
    
    def _generate_impulsive_p_wave(self, duration: float, sampling_rate: float,
                                 frequency: float) -> np.ndarray:
        """Generate impulsive P-wave with sharp onset."""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        onset_time = 1.5
        wave_duration = 2.5
        amplitude = 1.2
        
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        wave_t = t[onset_idx:end_idx] - onset_time
        
        # Very sharp onset
        envelope[onset_idx:end_idx] = amplitude * np.exp(-wave_t * 4.0)
        
        signal = envelope * np.sin(2 * np.pi * frequency * t)
        noise = 0.02 * amplitude * np.random.normal(0, 1, len(t))
        
        return signal + noise
    
    def _generate_high_frequency_p_wave(self, duration: float, sampling_rate: float,
                                      frequency: float) -> np.ndarray:
        """Generate high-frequency P-wave."""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        onset_time = 1.0
        wave_duration = 2.0
        amplitude = 0.9
        
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        wave_t = t[onset_idx:end_idx] - onset_time
        
        # Sharp, short duration
        envelope[onset_idx:end_idx] = amplitude * np.exp(-wave_t * 6.0)
        
        signal = envelope * np.sin(2 * np.pi * frequency * t)
        noise = 0.03 * amplitude * np.random.normal(0, 1, len(t))
        
        return signal + noise
    
    def _generate_noisy_wave_pattern(self, duration: float, sampling_rate: float,
                                   noise_level: float) -> np.ndarray:
        """Generate noise-contaminated wave pattern."""
        t = np.linspace(0, duration, int(duration * sampling_rate))
        
        # Weak signal buried in noise
        signal_amplitude = 0.3
        onset_time = 5.0
        wave_duration = 10.0
        frequency = 5.0
        
        envelope = np.zeros_like(t)
        onset_idx = int(onset_time * sampling_rate)
        end_idx = int((onset_time + wave_duration) * sampling_rate)
        
        if end_idx > len(envelope):
            end_idx = len(envelope)
        
        wave_t = t[onset_idx:end_idx] - onset_time
        envelope[onset_idx:end_idx] = signal_amplitude * np.exp(-wave_t / 5.0)
        
        signal = envelope * np.sin(2 * np.pi * frequency * t)
        
        # Heavy noise
        noise = noise_level * np.random.normal(0, 1, len(t))
        
        return signal + noise 
   
    def get_pattern(self, pattern_id: str) -> Optional[WavePattern]:
        """Get a specific pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_category(self, category: PatternCategory) -> List[WavePattern]:
        """Get all patterns in a specific category."""
        return [pattern for pattern in self.patterns.values() 
                if pattern.category == category]
    
    def get_patterns_by_type(self, pattern_type: WavePatternType) -> List[WavePattern]:
        """Get all patterns of a specific type."""
        return [pattern for pattern in self.patterns.values() 
                if pattern.pattern_type == pattern_type]
    
    def list_all_patterns(self) -> List[WavePattern]:
        """Get all patterns in the library."""
        return list(self.patterns.values())
    
    def compare_with_library(self, user_wave: WaveSegment, 
                           max_comparisons: int = 5) -> List[PatternComparison]:
        """
        Compare user wave data with library patterns.
        
        Args:
            user_wave: User's wave segment to compare
            max_comparisons: Maximum number of comparisons to return
            
        Returns:
            List of pattern comparisons sorted by similarity score
        """
        comparisons = []
        
        for pattern in self.patterns.values():
            similarity = self._calculate_similarity(user_wave, pattern)
            
            if similarity > 0.1:  # Only include meaningful similarities
                comparison = PatternComparison(
                    user_pattern_id=f"user_{hash(str(user_wave.data))}",
                    library_pattern=pattern,
                    similarity_score=similarity,
                    matching_features=self._find_matching_features(user_wave, pattern),
                    differences=self._find_differences(user_wave, pattern),
                    interpretation=self._generate_interpretation(user_wave, pattern, similarity),
                    educational_notes=self._generate_educational_notes(pattern, similarity)
                )
                comparisons.append(comparison)
        
        # Sort by similarity score (highest first)
        comparisons.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return comparisons[:max_comparisons]
    
    def _calculate_similarity(self, user_wave: WaveSegment, pattern: WavePattern) -> float:
        """Calculate similarity between user wave and library pattern."""
        
        # Normalize both signals
        user_data = self._normalize_signal(user_wave.data)
        pattern_data = self._normalize_signal(pattern.synthetic_data)
        
        # Resample to same length if needed
        if len(user_data) != len(pattern_data):
            min_len = min(len(user_data), len(pattern_data))
            user_data = user_data[:min_len]
            pattern_data = pattern_data[:min_len]
        
        # Calculate cross-correlation
        correlation = np.corrcoef(user_data, pattern_data)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Feature-based similarity
        feature_similarity = self._calculate_feature_similarity(user_wave, pattern)
        
        # Combined similarity score
        similarity = 0.6 * abs(correlation) + 0.4 * feature_similarity
        
        return max(0.0, min(1.0, similarity))
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        if np.std(signal) == 0:
            return signal - np.mean(signal)
        return (signal - np.mean(signal)) / np.std(signal)
    
    def _calculate_feature_similarity(self, user_wave: WaveSegment, pattern: WavePattern) -> float:
        """Calculate similarity based on wave features."""
        similarity_score = 0.0
        total_features = 0
        
        # Compare dominant frequency
        if hasattr(user_wave, 'dominant_frequency') and user_wave.dominant_frequency:
            pattern_freq = pattern.typical_parameters.get('dominant_frequency', 0)
            if pattern_freq > 0:
                freq_diff = abs(user_wave.dominant_frequency - pattern_freq) / pattern_freq
                similarity_score += max(0, 1 - freq_diff)
                total_features += 1
        
        # Compare peak amplitude (relative)
        if hasattr(user_wave, 'peak_amplitude') and user_wave.peak_amplitude:
            pattern_amp = pattern.typical_parameters.get('peak_amplitude', 1)
            # Normalize amplitudes
            user_amp_norm = user_wave.peak_amplitude / np.max(np.abs(user_wave.data))
            pattern_amp_norm = pattern_amp / np.max(np.abs(pattern.synthetic_data))
            amp_diff = abs(user_amp_norm - pattern_amp_norm)
            similarity_score += max(0, 1 - amp_diff)
            total_features += 1
        
        # Compare duration
        user_duration = user_wave.end_time - user_wave.start_time
        pattern_duration = pattern.typical_parameters.get('duration', user_duration)
        if pattern_duration > 0:
            duration_diff = abs(user_duration - pattern_duration) / pattern_duration
            similarity_score += max(0, 1 - duration_diff)
            total_features += 1
        
        return similarity_score / max(1, total_features)
    
    def _find_matching_features(self, user_wave: WaveSegment, pattern: WavePattern) -> List[str]:
        """Find features that match between user wave and pattern."""
        matching_features = []
        
        # Check frequency range
        if hasattr(user_wave, 'dominant_frequency') and user_wave.dominant_frequency:
            pattern_freq_range = pattern.key_features.get('frequency_range', [0, 100])
            if pattern_freq_range[0] <= user_wave.dominant_frequency <= pattern_freq_range[1]:
                matching_features.append(f"Frequency in expected range ({pattern_freq_range[0]}-{pattern_freq_range[1]} Hz)")
        
        # Check wave type
        if user_wave.wave_type.lower() in pattern.name.lower():
            matching_features.append(f"Wave type matches ({user_wave.wave_type})")
        
        # Check duration
        user_duration = user_wave.end_time - user_wave.start_time
        pattern_duration = pattern.typical_parameters.get('duration', 0)
        if pattern_duration > 0 and abs(user_duration - pattern_duration) / pattern_duration < 0.5:
            matching_features.append(f"Duration similar ({user_duration:.1f}s vs {pattern_duration:.1f}s)")
        
        return matching_features
    
    def _find_differences(self, user_wave: WaveSegment, pattern: WavePattern) -> List[str]:
        """Find differences between user wave and pattern."""
        differences = []
        
        # Check frequency differences
        if hasattr(user_wave, 'dominant_frequency') and user_wave.dominant_frequency:
            pattern_freq = pattern.typical_parameters.get('dominant_frequency', 0)
            if pattern_freq > 0:
                freq_diff_pct = abs(user_wave.dominant_frequency - pattern_freq) / pattern_freq * 100
                if freq_diff_pct > 20:
                    differences.append(f"Frequency differs by {freq_diff_pct:.1f}% from typical")
        
        # Check amplitude differences
        if hasattr(user_wave, 'peak_amplitude') and user_wave.peak_amplitude:
            user_amp_norm = user_wave.peak_amplitude / np.max(np.abs(user_wave.data))
            pattern_amp_norm = pattern.typical_parameters.get('peak_amplitude', 1) / np.max(np.abs(pattern.synthetic_data))
            amp_diff_pct = abs(user_amp_norm - pattern_amp_norm) / pattern_amp_norm * 100
            if amp_diff_pct > 30:
                differences.append(f"Amplitude differs by {amp_diff_pct:.1f}% from typical")
        
        # Check duration differences
        user_duration = user_wave.end_time - user_wave.start_time
        pattern_duration = pattern.typical_parameters.get('duration', 0)
        if pattern_duration > 0:
            duration_diff_pct = abs(user_duration - pattern_duration) / pattern_duration * 100
            if duration_diff_pct > 40:
                differences.append(f"Duration differs by {duration_diff_pct:.1f}% from typical")
        
        return differences
    
    def _generate_interpretation(self, user_wave: WaveSegment, pattern: WavePattern, 
                               similarity: float) -> str:
        """Generate interpretation text for the comparison."""
        
        if similarity > 0.8:
            confidence = "very high"
        elif similarity > 0.6:
            confidence = "high"
        elif similarity > 0.4:
            confidence = "moderate"
        else:
            confidence = "low"
        
        interpretation = f"This wave shows {confidence} similarity to a {pattern.name.lower()}. "
        
        if similarity > 0.6:
            interpretation += f"The pattern characteristics strongly suggest {pattern.wave_type if hasattr(pattern, 'wave_type') else pattern.pattern_type.value.replace('_', ' ')} behavior. "
            interpretation += pattern.interpretation_guide
        else:
            interpretation += f"While some features match {pattern.name.lower()}, significant differences suggest either: "
            interpretation += "1) Different wave type, 2) Unusual source characteristics, or 3) Propagation effects. "
            interpretation += "Consider additional analysis or comparison with other patterns."
        
        return interpretation
    
    def _generate_educational_notes(self, pattern: WavePattern, similarity: float) -> str:
        """Generate educational notes for the comparison."""
        
        notes = f"Educational Context: {pattern.educational_text}\n\n"
        
        if similarity > 0.6:
            notes += "This is a good match! Key learning points:\n"
            notes += f"• {pattern.description}\n"
            notes += f"• Typical characteristics: {', '.join([f'{k}: {v}' for k, v in pattern.key_features.items()])}\n"
        else:
            notes += "This shows some similarities but also differences. Learning opportunities:\n"
            notes += f"• Compare with typical {pattern.name.lower()} characteristics\n"
            notes += "• Consider what factors might cause variations from typical patterns\n"
            notes += "• Examine other wave types that might be a better match\n"
        
        notes += f"\nFor reference: {pattern.interpretation_guide}"
        
        return notes
    
    def get_unusual_pattern_guidance(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Provide guidance for unusual wave patterns detected in analysis.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Dictionary with guidance for unusual patterns
        """
        guidance = {}
        
        # Check for unusual frequency patterns
        if 'frequency_analysis' in analysis_results:
            freq_analysis = analysis_results['frequency_analysis']
            
            for wave_type, freq_data in freq_analysis.items():
                dominant_freq = freq_data.get('dominant_frequency', 0)
                
                # Unusual P-wave frequencies
                if wave_type.lower() == 'p' and dominant_freq > 20:
                    guidance['high_freq_p'] = (
                        "Unusually high P-wave frequencies (>20 Hz) detected. "
                        "This may indicate: 1) Very shallow earthquake source, "
                        "2) Nearby explosion or quarry blast, 3) Instrumental artifact. "
                        "Consider source depth and local seismic activity."
                    )
                elif wave_type.lower() == 'p' and dominant_freq < 2:
                    guidance['low_freq_p'] = (
                        "Unusually low P-wave frequencies (<2 Hz) detected. "
                        "This may indicate: 1) Very distant earthquake, "
                        "2) Deep earthquake source, 3) Strong attenuation effects. "
                        "Check epicentral distance and source depth."
                    )
                
                # Unusual S-wave frequencies
                if wave_type.lower() == 's' and dominant_freq > 15:
                    guidance['high_freq_s'] = (
                        "Unusually high S-wave frequencies (>15 Hz) detected. "
                        "This is rare and may indicate: 1) Very shallow, nearby source, "
                        "2) Unusual source mechanism, 3) Site amplification effects."
                    )
        
        # Check for unusual amplitude ratios
        if 'wave_amplitudes' in analysis_results:
            amplitudes = analysis_results['wave_amplitudes']
            
            p_amp = amplitudes.get('P', 0)
            s_amp = amplitudes.get('S', 0)
            
            if p_amp > 0 and s_amp > 0:
                s_p_ratio = s_amp / p_amp
                
                if s_p_ratio < 1.5:
                    guidance['low_s_p_ratio'] = (
                        "Unusually low S/P amplitude ratio (<1.5) detected. "
                        "This may indicate: 1) Deep earthquake source, "
                        "2) Unusual focal mechanism, 3) Directivity effects. "
                        "Consider source parameters and station azimuth."
                    )
                elif s_p_ratio > 10:
                    guidance['high_s_p_ratio'] = (
                        "Unusually high S/P amplitude ratio (>10) detected. "
                        "This may indicate: 1) Shallow earthquake, "
                        "2) Local site effects, 3) S-wave amplification. "
                        "Check local geology and site conditions."
                    )
        
        # Check for unusual timing
        if 'arrival_times' in analysis_results:
            arrivals = analysis_results['arrival_times']
            
            p_arrival = arrivals.get('p_wave_arrival', 0)
            s_arrival = arrivals.get('s_wave_arrival', 0)
            
            if p_arrival > 0 and s_arrival > 0:
                s_p_time = s_arrival - p_arrival
                
                if s_p_time < 1:
                    guidance['short_s_p_time'] = (
                        "Very short S-P time (<1 second) indicates extremely close earthquake. "
                        "Verify: 1) Correct wave identification, 2) Local seismic activity, "
                        "3) Possible explosion or induced seismicity."
                    )
                elif s_p_time > 100:
                    guidance['long_s_p_time'] = (
                        "Very long S-P time (>100 seconds) indicates distant earthquake. "
                        "Consider: 1) Teleseismic event, 2) Regional earthquake, "
                        "3) Check global earthquake catalogs for correlation."
                    )
        
        return guidance
    
    def export_library(self, filepath: str) -> bool:
        """
        Export the pattern library to a JSON file.
        
        Args:
            filepath: Path to save the library
            
        Returns:
            True if successful, False otherwise
        """
        try:
            library_data = {
                'metadata': {
                    'version': '1.0',
                    'created_date': datetime.now().isoformat(),
                    'pattern_count': len(self.patterns)
                },
                'patterns': {pid: pattern.to_dict() for pid, pattern in self.patterns.items()}
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(library_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Error exporting library: {e}")
            return False
    
    def load_library(self, filepath: str) -> bool:
        """
        Load pattern library from a JSON file.
        
        Args:
            filepath: Path to load the library from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                library_data = json.load(f)
            
            patterns_data = library_data.get('patterns', {})
            
            for pattern_id, pattern_dict in patterns_data.items():
                # Reconstruct WavePattern object
                # Note: synthetic_data would need to be stored separately or regenerated
                pattern = WavePattern(
                    pattern_id=pattern_dict['pattern_id'],
                    pattern_type=WavePatternType(pattern_dict['pattern_type']),
                    category=PatternCategory(pattern_dict['category']),
                    name=pattern_dict['name'],
                    description=pattern_dict['description'],
                    educational_text=pattern_dict['educational_text'],
                    interpretation_guide=pattern_dict['interpretation_guide'],
                    synthetic_data=np.array([]),  # Would need to regenerate
                    sampling_rate=pattern_dict['sampling_rate'],
                    duration=pattern_dict['duration'],
                    key_features=pattern_dict['key_features'],
                    typical_parameters=pattern_dict['typical_parameters'],
                    created_date=datetime.fromisoformat(pattern_dict['created_date'])
                )
                
                self.patterns[pattern_id] = pattern
            
            return True
        except Exception as e:
            print(f"Error loading library: {e}")
            return False