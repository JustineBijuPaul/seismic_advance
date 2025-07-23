"""
Test Data Management System for Wave Analysis Testing.

This module provides comprehensive test data generation and management
capabilities for validating wave analysis algorithms and system components.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import tempfile
import requests
from urllib.parse import urljoin
import logging

from wave_analysis.models.wave_models import (
    WaveSegment, WaveAnalysisResult, DetailedAnalysis,
    ArrivalTimes, MagnitudeEstimate, FrequencyData, QualityMetrics
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SyntheticEarthquakeParams:
    """Parameters for generating synthetic earthquake data."""
    magnitude: float  # Earthquake magnitude (Richter scale)
    distance: float   # Distance from epicenter in km
    depth: float      # Earthquake depth in km
    duration: float   # Total recording duration in seconds
    sampling_rate: float = 100.0  # Sampling rate in Hz
    noise_level: float = 0.1  # Background noise level (0-1)
    p_velocity: float = 6.0   # P-wave velocity in km/s
    s_velocity: float = 3.5   # S-wave velocity in km/s
    surface_velocity: float = 3.0  # Surface wave velocity in km/s
    
    def __post_init__(self):
        """Validate earthquake parameters."""
        if self.magnitude < 0 or self.magnitude > 10:
            raise ValueError("Magnitude must be between 0 and 10")
        if self.distance <= 0:
            raise ValueError("Distance must be positive")
        if self.depth <= 0:
            raise ValueError("Depth must be positive")
        if self.duration <= 0:
            raise ValueError("Duration must be positive")
        if self.sampling_rate <= 0:
            raise ValueError("Sampling rate must be positive")


@dataclass
class ReferenceEarthquake:
    """Reference earthquake data from seismic databases."""
    event_id: str
    magnitude: float
    location: Tuple[float, float]  # (latitude, longitude)
    depth: float
    origin_time: datetime
    data_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NoiseProfile:
    """Noise characteristics for testing robustness."""
    noise_type: str  # 'white', 'pink', 'brown', 'seismic'
    amplitude: float  # Noise amplitude relative to signal
    frequency_range: Tuple[float, float]  # (min_freq, max_freq) in Hz
    duration: float  # Noise duration in seconds
    sampling_rate: float = 100.0


class TestDataManager:
    """
    Comprehensive test data management system for wave analysis testing.
    
    Provides synthetic earthquake generation, reference data loading,
    noise generation, and validation capabilities.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize test data manager.
        
        Args:
            cache_dir: Directory for caching test data (optional)
        """
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix='wave_test_data_')
        self.reference_earthquakes: List[ReferenceEarthquake] = []
        self.synthetic_data_cache: Dict[str, Any] = {}
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"TestDataManager initialized with cache dir: {self.cache_dir}")
    
    def create_synthetic_earthquake(self, params: SyntheticEarthquakeParams) -> np.ndarray:
        """
        Generate synthetic earthquake data with realistic wave characteristics.
        
        Args:
            params: Parameters for earthquake generation
            
        Returns:
            Synthetic seismic time series data
        """
        logger.info(f"Generating synthetic earthquake: M{params.magnitude}, {params.distance}km")
        
        # Create cache key for this parameter set
        cache_key = self._create_cache_key(params)
        if cache_key in self.synthetic_data_cache:
            logger.debug("Returning cached synthetic earthquake data")
            return self.synthetic_data_cache[cache_key]
        
        # Calculate time parameters
        total_samples = int(params.duration * params.sampling_rate)
        time_array = np.linspace(0, params.duration, total_samples)
        
        # Calculate arrival times based on distance and velocities
        p_arrival_time = params.distance / params.p_velocity
        s_arrival_time = params.distance / params.s_velocity
        surface_arrival_time = params.distance / params.surface_velocity
        
        # Initialize output array
        seismic_data = np.zeros(total_samples)
        
        # Generate P-wave
        p_wave = self._generate_p_wave(
            time_array, p_arrival_time, params.magnitude, params.distance
        )
        seismic_data += p_wave
        
        # Generate S-wave
        s_wave = self._generate_s_wave(
            time_array, s_arrival_time, params.magnitude, params.distance
        )
        seismic_data += s_wave
        
        # Generate surface waves
        surface_wave = self._generate_surface_wave(
            time_array, surface_arrival_time, params.magnitude, params.distance
        )
        seismic_data += surface_wave
        
        # Add background noise
        noise = self._generate_noise(total_samples, params.noise_level, 'seismic')
        seismic_data += noise
        
        # Cache the result
        self.synthetic_data_cache[cache_key] = seismic_data
        
        logger.info(f"Generated {len(seismic_data)} samples of synthetic earthquake data")
        return seismic_data
    
    def _generate_p_wave(self, time_array: np.ndarray, arrival_time: float, 
                        magnitude: float, distance: float) -> np.ndarray:
        """Generate realistic P-wave signal."""
        # P-wave characteristics
        dominant_freq = 8.0  # Hz
        duration = 2.0 + magnitude * 0.5  # Duration increases with magnitude
        
        # Amplitude scaling based on magnitude and distance
        amplitude = (10 ** (magnitude - 3)) / (distance + 1)
        
        # Create P-wave envelope
        p_wave = np.zeros_like(time_array)
        wave_mask = (time_array >= arrival_time) & (time_array <= arrival_time + duration)
        
        if np.any(wave_mask):
            wave_time = time_array[wave_mask] - arrival_time
            
            # Exponential decay envelope
            envelope = amplitude * np.exp(-wave_time / (duration / 3))
            
            # High-frequency oscillation
            oscillation = np.sin(2 * np.pi * dominant_freq * wave_time)
            
            # Add some frequency modulation
            freq_mod = np.sin(2 * np.pi * (dominant_freq * 1.5) * wave_time)
            
            p_wave[wave_mask] = envelope * (oscillation + 0.3 * freq_mod)
        
        return p_wave
    
    def _generate_s_wave(self, time_array: np.ndarray, arrival_time: float,
                        magnitude: float, distance: float) -> np.ndarray:
        """Generate realistic S-wave signal."""
        # S-wave characteristics
        dominant_freq = 4.0  # Hz (lower than P-wave)
        duration = 3.0 + magnitude * 0.7  # Longer duration than P-wave
        
        # Amplitude scaling (S-waves typically larger than P-waves)
        amplitude = (10 ** (magnitude - 2.5)) / (distance + 1)
        
        # Create S-wave envelope
        s_wave = np.zeros_like(time_array)
        wave_mask = (time_array >= arrival_time) & (time_array <= arrival_time + duration)
        
        if np.any(wave_mask):
            wave_time = time_array[wave_mask] - arrival_time
            
            # Exponential decay envelope with slower decay than P-wave
            envelope = amplitude * np.exp(-wave_time / (duration / 2))
            
            # Lower frequency oscillation
            oscillation = np.sin(2 * np.pi * dominant_freq * wave_time)
            
            # Add harmonic content
            harmonic = 0.4 * np.sin(2 * np.pi * (dominant_freq * 2) * wave_time)
            
            s_wave[wave_mask] = envelope * (oscillation + harmonic)
        
        return s_wave
    
    def _generate_surface_wave(self, time_array: np.ndarray, arrival_time: float,
                              magnitude: float, distance: float) -> np.ndarray:
        """Generate realistic surface wave signal (Love and Rayleigh waves)."""
        # Surface wave characteristics
        love_freq = 0.5  # Hz (very low frequency)
        rayleigh_freq = 0.3  # Hz (even lower)
        duration = 10.0 + magnitude * 2.0  # Very long duration
        
        # Amplitude scaling (surface waves can be very large at distance)
        amplitude = (10 ** (magnitude - 2)) / np.sqrt(distance + 1)
        
        # Create surface wave envelope
        surface_wave = np.zeros_like(time_array)
        wave_mask = (time_array >= arrival_time) & (time_array <= arrival_time + duration)
        
        if np.any(wave_mask):
            wave_time = time_array[wave_mask] - arrival_time
            
            # Slow decay envelope
            envelope = amplitude * np.exp(-wave_time / (duration / 4))
            
            # Love wave component
            love_wave = np.sin(2 * np.pi * love_freq * wave_time)
            
            # Rayleigh wave component
            rayleigh_wave = np.sin(2 * np.pi * rayleigh_freq * wave_time)
            
            # Combine with different phases
            surface_wave[wave_mask] = envelope * (0.6 * love_wave + 0.4 * rayleigh_wave)
        
        return surface_wave
    
    def _generate_noise(self, num_samples: int, amplitude: float, 
                       noise_type: str = 'seismic') -> np.ndarray:
        """Generate various types of noise for testing robustness."""
        if noise_type == 'white':
            return amplitude * np.random.normal(0, 1, num_samples)
        
        elif noise_type == 'pink':
            # Pink noise (1/f spectrum)
            white_noise = np.random.normal(0, 1, num_samples)
            freqs = np.fft.fftfreq(num_samples)
            freqs[0] = 1e-10  # Avoid division by zero
            pink_filter = 1 / np.sqrt(np.abs(freqs))
            pink_noise = np.fft.ifft(np.fft.fft(white_noise) * pink_filter).real
            return amplitude * pink_noise / np.std(pink_noise)
        
        elif noise_type == 'seismic':
            # Realistic seismic background noise
            base_noise = amplitude * np.random.normal(0, 1, num_samples)
            
            # Add low-frequency cultural noise
            time_array = np.arange(num_samples) / 100.0  # Assuming 100 Hz sampling
            cultural_noise = 0.3 * amplitude * np.sin(2 * np.pi * 0.1 * time_array)
            
            # Add microseismic noise (ocean waves)
            microseismic = 0.2 * amplitude * np.sin(2 * np.pi * 0.2 * time_array)
            
            return base_noise + cultural_noise + microseismic
        
        else:
            return amplitude * np.random.normal(0, 1, num_samples)
    
    def load_reference_earthquakes(self, source: str = 'usgs', 
                                 limit: int = 10) -> List[ReferenceEarthquake]:
        """
        Load reference earthquake data from seismic databases.
        
        Args:
            source: Data source ('usgs', 'iris', 'local')
            limit: Maximum number of earthquakes to load
            
        Returns:
            List of reference earthquake records
        """
        logger.info(f"Loading reference earthquakes from {source}, limit={limit}")
        
        if source == 'usgs':
            return self._load_usgs_earthquakes(limit)
        elif source == 'local':
            return self._load_local_reference_data()
        else:
            logger.warning(f"Unknown source: {source}, returning empty list")
            return []
    
    def _load_usgs_earthquakes(self, limit: int) -> List[ReferenceEarthquake]:
        """Load earthquake data from USGS API."""
        try:
            # USGS Earthquake API endpoint
            base_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
            params = {
                'format': 'geojson',
                'minmagnitude': 5.0,  # Only significant earthquakes
                'limit': limit,
                'orderby': 'time-desc'
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            earthquakes = []
            
            for feature in data.get('features', []):
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                earthquake = ReferenceEarthquake(
                    event_id=feature['id'],
                    magnitude=props.get('mag', 0.0),
                    location=(coords[1], coords[0]),  # lat, lon
                    depth=coords[2] if len(coords) > 2 else 0.0,
                    origin_time=datetime.fromtimestamp(props['time'] / 1000),
                    data_url=props.get('url'),
                    metadata={
                        'place': props.get('place', ''),
                        'type': props.get('type', ''),
                        'status': props.get('status', ''),
                        'tsunami': props.get('tsunami', 0)
                    }
                )
                earthquakes.append(earthquake)
            
            self.reference_earthquakes.extend(earthquakes)
            logger.info(f"Loaded {len(earthquakes)} reference earthquakes from USGS")
            return earthquakes
            
        except Exception as e:
            logger.error(f"Failed to load USGS earthquake data: {e}")
            return []
    
    def _load_local_reference_data(self) -> List[ReferenceEarthquake]:
        """Load reference earthquake data from local files."""
        reference_file = os.path.join(self.cache_dir, 'reference_earthquakes.json')
        
        if not os.path.exists(reference_file):
            logger.warning("No local reference earthquake file found")
            return []
        
        try:
            with open(reference_file, 'r') as f:
                data = json.load(f)
            
            earthquakes = []
            for item in data:
                earthquake = ReferenceEarthquake(
                    event_id=item['event_id'],
                    magnitude=item['magnitude'],
                    location=tuple(item['location']),
                    depth=item['depth'],
                    origin_time=datetime.fromisoformat(item['origin_time']),
                    data_url=item.get('data_url'),
                    metadata=item.get('metadata', {})
                )
                earthquakes.append(earthquake)
            
            logger.info(f"Loaded {len(earthquakes)} reference earthquakes from local file")
            return earthquakes
            
        except Exception as e:
            logger.error(f"Failed to load local reference data: {e}")
            return []
    
    def generate_noise_samples(self, profiles: List[NoiseProfile]) -> Dict[str, np.ndarray]:
        """
        Generate various noise samples for testing algorithm robustness.
        
        Args:
            profiles: List of noise profiles to generate
            
        Returns:
            Dictionary mapping noise type to generated samples
        """
        logger.info(f"Generating {len(profiles)} noise samples")
        
        noise_samples = {}
        
        for profile in profiles:
            num_samples = int(profile.duration * profile.sampling_rate)
            
            if profile.noise_type == 'white':
                noise = np.random.normal(0, profile.amplitude, num_samples)
            
            elif profile.noise_type == 'pink':
                noise = self._generate_noise(num_samples, profile.amplitude, 'pink')
            
            elif profile.noise_type == 'brown':
                # Brown noise (1/f^2 spectrum)
                white_noise = np.random.normal(0, 1, num_samples)
                freqs = np.fft.fftfreq(num_samples)
                freqs[0] = 1e-10
                brown_filter = 1 / np.abs(freqs)
                noise = np.fft.ifft(np.fft.fft(white_noise) * brown_filter).real
                noise = profile.amplitude * noise / np.std(noise)
            
            elif profile.noise_type == 'seismic':
                noise = self._generate_noise(num_samples, profile.amplitude, 'seismic')
            
            else:
                noise = np.random.normal(0, profile.amplitude, num_samples)
            
            # Apply frequency filtering if specified
            if profile.frequency_range != (0, np.inf):
                noise = self._apply_bandpass_filter(
                    noise, profile.sampling_rate, profile.frequency_range
                )
            
            noise_samples[f"{profile.noise_type}_{profile.amplitude}"] = noise
        
        logger.info(f"Generated noise samples: {list(noise_samples.keys())}")
        return noise_samples
    
    def _apply_bandpass_filter(self, data: np.ndarray, sampling_rate: float,
                              freq_range: Tuple[float, float]) -> np.ndarray:
        """Apply bandpass filter to data."""
        from scipy import signal
        
        nyquist = sampling_rate / 2
        low = freq_range[0] / nyquist
        high = min(freq_range[1] / nyquist, 0.99)  # Avoid Nyquist frequency
        
        if low >= high:
            return data  # Invalid frequency range
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            return signal.filtfilt(b, a, data)
        except Exception as e:
            logger.warning(f"Filter application failed: {e}")
            return data
    
    def create_multi_channel_data(self, channels: int, 
                                 params: SyntheticEarthquakeParams) -> np.ndarray:
        """
        Create multi-channel seismic data for testing.
        
        Args:
            channels: Number of channels to generate
            params: Base earthquake parameters
            
        Returns:
            Multi-channel seismic data array (channels x samples)
        """
        logger.info(f"Generating {channels}-channel synthetic earthquake data")
        
        base_data = self.create_synthetic_earthquake(params)
        multi_channel_data = np.zeros((channels, len(base_data)))
        
        for i in range(channels):
            # Add slight variations between channels
            channel_params = SyntheticEarthquakeParams(
                magnitude=params.magnitude + np.random.normal(0, 0.1),
                distance=params.distance + np.random.normal(0, 5),
                depth=params.depth,
                duration=params.duration,
                sampling_rate=params.sampling_rate,
                noise_level=params.noise_level * (1 + np.random.normal(0, 0.2))
            )
            
            multi_channel_data[i] = self.create_synthetic_earthquake(channel_params)
        
        return multi_channel_data
    
    def validate_test_data_quality(self, data: np.ndarray, 
                                  expected_params: SyntheticEarthquakeParams) -> Dict[str, Any]:
        """
        Validate the quality and consistency of generated test data.
        
        Args:
            data: Generated test data
            expected_params: Expected parameters for validation
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating test data quality")
        
        validation_results = {
            'data_length_valid': False,
            'amplitude_range_valid': False,
            'frequency_content_valid': False,
            'noise_level_valid': False,
            'wave_arrivals_detected': False,
            'quality_score': 0.0,
            'warnings': []
        }
        
        try:
            # Check data length
            expected_samples = int(expected_params.duration * expected_params.sampling_rate)
            if len(data) == expected_samples:
                validation_results['data_length_valid'] = True
            else:
                validation_results['warnings'].append(
                    f"Data length mismatch: expected {expected_samples}, got {len(data)}"
                )
            
            # Check amplitude range
            max_amplitude = np.max(np.abs(data))
            expected_max = 10 ** (expected_params.magnitude - 2)  # Rough estimate
            if 0.1 * expected_max <= max_amplitude <= 10 * expected_max:
                validation_results['amplitude_range_valid'] = True
            else:
                validation_results['warnings'].append(
                    f"Amplitude out of expected range: {max_amplitude:.2e}"
                )
            
            # Check frequency content
            freqs, psd = self._compute_power_spectrum(data, expected_params.sampling_rate)
            dominant_freq = freqs[np.argmax(psd)]
            if 0.1 <= dominant_freq <= 20:  # Reasonable seismic frequency range
                validation_results['frequency_content_valid'] = True
            else:
                validation_results['warnings'].append(
                    f"Dominant frequency unusual: {dominant_freq:.2f} Hz"
                )
            
            # Check noise level
            noise_estimate = np.std(data[:int(0.1 * len(data))])  # First 10% as noise
            signal_estimate = np.max(np.abs(data))
            snr = signal_estimate / (noise_estimate + 1e-10)
            if snr > 2:  # Reasonable SNR
                validation_results['noise_level_valid'] = True
            else:
                validation_results['warnings'].append(f"Low SNR detected: {snr:.2f}")
            
            # Simple wave arrival detection
            envelope = np.abs(data)
            smoothed = np.convolve(envelope, np.ones(100)/100, mode='same')
            peaks = np.where(smoothed > 0.5 * np.max(smoothed))[0]
            if len(peaks) > 0:
                validation_results['wave_arrivals_detected'] = True
            else:
                validation_results['warnings'].append("No clear wave arrivals detected")
            
            # Calculate overall quality score
            valid_checks = sum([
                validation_results['data_length_valid'],
                validation_results['amplitude_range_valid'],
                validation_results['frequency_content_valid'],
                validation_results['noise_level_valid'],
                validation_results['wave_arrivals_detected']
            ])
            validation_results['quality_score'] = valid_checks / 5.0
            
        except Exception as e:
            validation_results['warnings'].append(f"Validation error: {e}")
            logger.error(f"Test data validation failed: {e}")
        
        logger.info(f"Test data validation complete. Quality score: {validation_results['quality_score']:.2f}")
        return validation_results
    
    def _compute_power_spectrum(self, data: np.ndarray, 
                               sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum of data."""
        freqs = np.fft.fftfreq(len(data), 1/sampling_rate)
        fft_data = np.fft.fft(data)
        psd = np.abs(fft_data) ** 2
        
        # Return only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_psd = psd[:len(psd)//2]
        
        return positive_freqs, positive_psd
    
    def _create_cache_key(self, params: SyntheticEarthquakeParams) -> str:
        """Create a cache key for earthquake parameters."""
        return f"eq_{params.magnitude}_{params.distance}_{params.depth}_{params.duration}_{params.noise_level}"
    
    def save_test_dataset(self, filename: str, data: Dict[str, Any]) -> None:
        """Save test dataset to file for reuse."""
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2, default=str)
            
            logger.info(f"Test dataset saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save test dataset: {e}")
    
    def load_test_dataset(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load test dataset from file."""
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Test dataset file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays where appropriate
            for key, value in data.items():
                if isinstance(value, list) and key.endswith('_data'):
                    data[key] = np.array(value)
            
            logger.info(f"Test dataset loaded from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load test dataset: {e}")
            return None
    
    def cleanup_cache(self) -> None:
        """Clean up cached test data."""
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cleaned up cache directory: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")