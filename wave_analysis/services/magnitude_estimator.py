"""
Magnitude Estimator for earthquake magnitude calculations.
"""

import numpy as np
from typing import List, Dict, Optional
import logging
import math

from ..models import WaveSegment, MagnitudeEstimate


logger = logging.getLogger(__name__)


class MagnitudeEstimator:
    """
    Estimator for earthquake magnitude using various seismological methods.
    """
    
    def __init__(self, sampling_rate: float):
        """Initialize the magnitude estimator."""
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__)
        
        # Standard magnitude calculation parameters
        self.ml_constants = {'a': 1.0, 'b': 0.0, 'c': -2.48}
        self.mb_constants = {'a': 1.0, 'b': 0.0, 'c': 5.1}
        self.ms_constants = {'a': 1.0, 'b': 0.0, 'c': 3.3}
        
        # Frequency ranges for different magnitude types
        self.frequency_ranges = {
            'ML': (1.0, 3.0),
            'Mb': (0.5, 2.0),
            'Ms': (0.05, 0.25)
        }
        
    def estimate_all_magnitudes(self, waves: Dict[str, List[WaveSegment]], 
                              epicenter_distance: Optional[float] = None) -> List[MagnitudeEstimate]:
        """Estimate earthquake magnitude using all available methods."""
        magnitude_estimates = []
        
        # Local magnitude (ML) using P-waves
        if 'P' in waves and waves['P']:
            ml_estimate = self.calculate_local_magnitude(waves['P'], epicenter_distance)
            if ml_estimate:
                magnitude_estimates.append(ml_estimate)
        
        # Body wave magnitude (Mb) using P-waves
        if 'P' in waves and waves['P']:
            mb_estimate = self.calculate_body_wave_magnitude(waves['P'], epicenter_distance)
            if mb_estimate:
                magnitude_estimates.append(mb_estimate)
        
        # Surface wave magnitude (Ms) using surface waves
        surface_waves = []
        if 'Love' in waves:
            surface_waves.extend(waves['Love'])
        if 'Rayleigh' in waves:
            surface_waves.extend(waves['Rayleigh'])
            
        if surface_waves:
            ms_estimate = self.calculate_surface_wave_magnitude(surface_waves, epicenter_distance)
            if ms_estimate:
                magnitude_estimates.append(ms_estimate)
        
        return magnitude_estimates

    def calculate_local_magnitude(self, p_waves: List[WaveSegment], 
                                distance: Optional[float] = None) -> Optional[MagnitudeEstimate]:
        """Calculate local magnitude (ML) using P-wave amplitudes."""
        if not p_waves:
            return None
        
        try:
            # Use the P-wave with highest amplitude
            best_p_wave = max(p_waves, key=lambda w: w.peak_amplitude)
            
            # Basic Richter formula: ML = log10(A) + corrections
            ml = math.log10(best_p_wave.peak_amplitude)
            
            # Apply distance correction if distance is known
            if distance is not None:
                distance_correction = math.log10(distance / 100.0) if distance > 0 else 0.0
                ml += distance_correction
            
            # Apply station correction
            ml += self.ml_constants['c']
            
            # Calculate confidence based on signal quality
            confidence = min(1.0, best_p_wave.confidence)
            
            return MagnitudeEstimate(
                method='ML',
                magnitude=ml,
                confidence=confidence,
                wave_type_used='P',
                metadata={
                    'amplitude': best_p_wave.peak_amplitude,
                    'distance': distance,
                    'wave_duration': best_p_wave.duration
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating local magnitude: {e}")
            return None

    def calculate_body_wave_magnitude(self, p_waves: List[WaveSegment],
                                    distance: Optional[float] = None) -> Optional[MagnitudeEstimate]:
        """Calculate body wave magnitude (Mb) using P-wave periods and amplitudes."""
        if not p_waves:
            return None
        
        try:
            # Use the P-wave with best signal characteristics
            best_p_wave = max(p_waves, key=lambda w: w.peak_amplitude * w.confidence)
            
            # Calculate period
            period = 1.0 / best_p_wave.dominant_frequency if best_p_wave.dominant_frequency > 0 else 1.0
            
            # Basic Mb formula
            mb = math.log10(best_p_wave.peak_amplitude / period)
            
            # Apply constant term
            mb += self.mb_constants['c']
            
            # Calculate confidence
            confidence = min(1.0, best_p_wave.confidence)
            
            return MagnitudeEstimate(
                method='Mb',
                magnitude=mb,
                confidence=confidence,
                wave_type_used='P',
                metadata={
                    'amplitude': best_p_wave.peak_amplitude,
                    'period': period,
                    'distance': distance
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating body wave magnitude: {e}")
            return None

    def calculate_surface_wave_magnitude(self, surface_waves: List[WaveSegment],
                                       distance: Optional[float] = None) -> Optional[MagnitudeEstimate]:
        """Calculate surface wave magnitude (Ms) using surface wave amplitudes."""
        if not surface_waves:
            return None
        
        try:
            # Use the surface wave with highest amplitude
            best_surface_wave = max(surface_waves, key=lambda w: w.peak_amplitude)
            
            # Calculate period
            period = 1.0 / best_surface_wave.dominant_frequency if best_surface_wave.dominant_frequency > 0 else 20.0
            
            # Basic Ms formula
            ms = math.log10(best_surface_wave.peak_amplitude / period)
            
            # Apply distance correction if distance is known
            if distance is not None:
                distance_degrees = distance / 111.0  # km per degree
                if distance_degrees > 0:
                    distance_correction = 1.66 * math.log10(distance_degrees)
                    ms += distance_correction
            
            # Apply constant term
            ms += self.ms_constants['c']
            
            # Calculate confidence
            confidence = min(1.0, best_surface_wave.confidence)
            
            return MagnitudeEstimate(
                method='Ms',
                magnitude=ms,
                confidence=confidence,
                wave_type_used=best_surface_wave.wave_type,
                metadata={
                    'amplitude': best_surface_wave.peak_amplitude,
                    'period': period,
                    'distance': distance,
                    'wave_type': best_surface_wave.wave_type
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating surface wave magnitude: {e}")
            return None

    def set_parameters(self, **kwargs):
        """Set magnitude calculation parameters."""
        if 'ml_constants' in kwargs:
            self.ml_constants.update(kwargs['ml_constants'])
        if 'mb_constants' in kwargs:
            self.mb_constants.update(kwargs['mb_constants'])
        if 'ms_constants' in kwargs:
            self.ms_constants.update(kwargs['ms_constants'])
        if 'frequency_ranges' in kwargs:
            self.frequency_ranges.update(kwargs['frequency_ranges'])