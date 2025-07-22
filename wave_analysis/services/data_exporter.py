"""
Data export service for wave analysis results.

This module provides comprehensive export capabilities for separated wave data
in multiple formats including MSEED, CSV, and analysis results.
"""

import io
import csv
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import json

try:
    from obspy import Stream, Trace, UTCDateTime
    from obspy.core.stats import Stats
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False

from ..interfaces import WaveExporterInterface
from ..models import WaveSegment, WaveAnalysisResult, DetailedAnalysis


class DataExporter(WaveExporterInterface):
    """
    Main data exporter class supporting multiple export formats.
    
    Supports export of separated wave data in MSEED, CSV formats
    and comprehensive analysis results.
    """
    
    def __init__(self):
        """Initialize the data exporter."""
        self.supported_formats = ['mseed', 'csv', 'json']
        if not OBSPY_AVAILABLE:
            # Remove MSEED support if ObsPy is not available
            self.supported_formats = ['csv', 'json']
    
    def export_waves(self, waves: Dict[str, List[WaveSegment]], 
                    format_type: str) -> bytes:
        """
        Export wave data in specified format.
        
        Args:
            waves: Dictionary mapping wave types to their segments
            format_type: Export format ('mseed', 'csv', 'json')
            
        Returns:
            Exported data as bytes
            
        Raises:
            ValueError: If format is not supported
            RuntimeError: If export fails
        """
        format_type = format_type.lower()
        
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. "
                           f"Supported formats: {self.supported_formats}")
        
        try:
            if format_type == 'mseed':
                return self._export_mseed(waves)
            elif format_type == 'csv':
                return self._export_csv(waves)
            elif format_type == 'json':
                return self._export_json(waves)
        except Exception as e:
            raise RuntimeError(f"Export failed for format {format_type}: {str(e)}")
    
    def export_analysis_results(self, analysis: DetailedAnalysis, 
                              format_type: str = 'json') -> bytes:
        """
        Export comprehensive analysis results.
        
        Args:
            analysis: Detailed analysis results to export
            format_type: Export format ('json', 'csv')
            
        Returns:
            Exported analysis data as bytes
        """
        format_type = format_type.lower()
        
        if format_type == 'json':
            return self._export_analysis_json(analysis)
        elif format_type == 'csv':
            return self._export_analysis_csv(analysis)
        else:
            raise ValueError(f"Unsupported analysis export format: {format_type}")
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported export formats.
        
        Returns:
            List of supported format strings
        """
        return self.supported_formats.copy()
    
    def _export_mseed(self, waves: Dict[str, List[WaveSegment]]) -> bytes:
        """
        Export wave data in MSEED format.
        
        Args:
            waves: Dictionary of wave segments by type
            
        Returns:
            MSEED data as bytes
        """
        if not OBSPY_AVAILABLE:
            raise RuntimeError("ObsPy is required for MSEED export but not available")
        
        stream = Stream()
        
        for wave_type, wave_segments in waves.items():
            for i, segment in enumerate(wave_segments):
                # Create ObsPy Stats object with metadata
                stats = Stats()
                stats.network = "XX"  # Default network code
                stats.station = f"{wave_type}{i:02d}"  # Station code based on wave type
                stats.channel = "HHZ"  # Default channel
                stats.sampling_rate = segment.sampling_rate
                stats.npts = len(segment.data)
                stats.starttime = UTCDateTime(segment.start_time)
                
                # Add custom headers for wave characteristics
                stats.mseed = {
                    'dataquality': 'D',
                    'wave_type': wave_type,
                    'arrival_time': segment.arrival_time,
                    'peak_amplitude': segment.peak_amplitude,
                    'dominant_frequency': segment.dominant_frequency,
                    'confidence': segment.confidence
                }
                
                # Create trace and add to stream
                trace = Trace(data=segment.data.astype(np.float32), header=stats)
                stream.append(trace)
        
        # Export to MSEED format
        buffer = io.BytesIO()
        stream.write(buffer, format='MSEED')
        return buffer.getvalue()
    
    def _export_csv(self, waves: Dict[str, List[WaveSegment]]) -> bytes:
        """
        Export wave data and characteristics in CSV format.
        
        Args:
            waves: Dictionary of wave segments by type
            
        Returns:
            CSV data as bytes
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        
        # Write header
        header = [
            'wave_type', 'segment_id', 'start_time', 'end_time', 'duration',
            'arrival_time', 'peak_amplitude', 'dominant_frequency', 'confidence',
            'sampling_rate', 'sample_count', 'data_points'
        ]
        writer.writerow(header)
        
        # Write wave data
        for wave_type, wave_segments in waves.items():
            for i, segment in enumerate(wave_segments):
                # Convert data array to comma-separated string
                data_str = ','.join(map(str, segment.data.tolist()))
                
                row = [
                    wave_type,
                    i,
                    segment.start_time,
                    segment.end_time,
                    segment.duration,
                    segment.arrival_time,
                    segment.peak_amplitude,
                    segment.dominant_frequency,
                    segment.confidence,
                    segment.sampling_rate,
                    segment.sample_count,
                    data_str
                ]
                writer.writerow(row)
        
        return buffer.getvalue().encode('utf-8')
    
    def _export_json(self, waves: Dict[str, List[WaveSegment]]) -> bytes:
        """
        Export wave data in JSON format.
        
        Args:
            waves: Dictionary of wave segments by type
            
        Returns:
            JSON data as bytes
        """
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'wave_data': {}
        }
        
        for wave_type, wave_segments in waves.items():
            export_data['wave_data'][wave_type] = []
            
            for i, segment in enumerate(wave_segments):
                segment_data = {
                    'segment_id': i,
                    'wave_type': segment.wave_type,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'duration': segment.duration,
                    'arrival_time': segment.arrival_time,
                    'peak_amplitude': segment.peak_amplitude,
                    'dominant_frequency': segment.dominant_frequency,
                    'confidence': segment.confidence,
                    'sampling_rate': segment.sampling_rate,
                    'sample_count': segment.sample_count,
                    'data': segment.data.tolist(),
                    'metadata': segment.metadata
                }
                export_data['wave_data'][wave_type].append(segment_data)
        
        return json.dumps(export_data, indent=2).encode('utf-8')
    
    def _export_analysis_json(self, analysis: DetailedAnalysis) -> bytes:
        """
        Export detailed analysis results in JSON format.
        
        Args:
            analysis: Detailed analysis results
            
        Returns:
            JSON data as bytes
        """
        # Convert analysis to dictionary
        analysis_data = {
            'analysis_timestamp': analysis.analysis_timestamp.isoformat(),
            'arrival_times': {
                'p_wave_arrival': analysis.arrival_times.p_wave_arrival,
                's_wave_arrival': analysis.arrival_times.s_wave_arrival,
                'surface_wave_arrival': analysis.arrival_times.surface_wave_arrival,
                'sp_time_difference': analysis.arrival_times.sp_time_difference
            },
            'magnitude_estimates': [
                {
                    'method': est.method,
                    'magnitude': est.magnitude,
                    'confidence': est.confidence,
                    'wave_type_used': est.wave_type_used,
                    'metadata': est.metadata
                }
                for est in analysis.magnitude_estimates
            ],
            'epicenter_distance': analysis.epicenter_distance,
            'frequency_analysis': {},
            'quality_metrics': None,
            'processing_metadata': analysis.processing_metadata,
            'wave_summary': {
                'total_waves_detected': analysis.wave_result.total_waves_detected,
                'wave_types_detected': analysis.wave_result.wave_types_detected,
                'p_waves_count': len(analysis.wave_result.p_waves),
                's_waves_count': len(analysis.wave_result.s_waves),
                'surface_waves_count': len(analysis.wave_result.surface_waves)
            }
        }
        
        # Add frequency analysis data
        for wave_type, freq_data in analysis.frequency_analysis.items():
            analysis_data['frequency_analysis'][wave_type] = {
                'dominant_frequency': freq_data.dominant_frequency,
                'frequency_range': freq_data.frequency_range,
                'spectral_centroid': freq_data.spectral_centroid,
                'bandwidth': freq_data.bandwidth,
                'frequencies': freq_data.frequencies.tolist(),
                'power_spectrum': freq_data.power_spectrum.tolist()
            }
        
        # Add quality metrics if available
        if analysis.quality_metrics:
            analysis_data['quality_metrics'] = {
                'signal_to_noise_ratio': analysis.quality_metrics.signal_to_noise_ratio,
                'detection_confidence': analysis.quality_metrics.detection_confidence,
                'analysis_quality_score': analysis.quality_metrics.analysis_quality_score,
                'data_completeness': analysis.quality_metrics.data_completeness,
                'processing_warnings': analysis.quality_metrics.processing_warnings
            }
        
        return json.dumps(analysis_data, indent=2).encode('utf-8')
    
    def _export_analysis_csv(self, analysis: DetailedAnalysis) -> bytes:
        """
        Export analysis results in CSV format.
        
        Args:
            analysis: Detailed analysis results
            
        Returns:
            CSV data as bytes
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        
        # Write analysis summary
        writer.writerow(['Analysis Summary'])
        writer.writerow(['Timestamp', analysis.analysis_timestamp.isoformat()])
        writer.writerow(['Total Waves Detected', analysis.wave_result.total_waves_detected])
        writer.writerow(['Wave Types', ', '.join(analysis.wave_result.wave_types_detected)])
        writer.writerow([])
        
        # Write arrival times
        writer.writerow(['Arrival Times'])
        writer.writerow(['P-wave Arrival', analysis.arrival_times.p_wave_arrival])
        writer.writerow(['S-wave Arrival', analysis.arrival_times.s_wave_arrival])
        writer.writerow(['Surface Wave Arrival', analysis.arrival_times.surface_wave_arrival])
        writer.writerow(['S-P Time Difference', analysis.arrival_times.sp_time_difference])
        writer.writerow([])
        
        # Write magnitude estimates
        writer.writerow(['Magnitude Estimates'])
        writer.writerow(['Method', 'Magnitude', 'Confidence', 'Wave Type Used'])
        for est in analysis.magnitude_estimates:
            writer.writerow([est.method, est.magnitude, est.confidence, est.wave_type_used])
        writer.writerow([])
        
        # Write epicenter distance
        writer.writerow(['Epicenter Distance (km)', analysis.epicenter_distance])
        writer.writerow([])
        
        # Write quality metrics if available
        if analysis.quality_metrics:
            writer.writerow(['Quality Metrics'])
            writer.writerow(['Signal-to-Noise Ratio', analysis.quality_metrics.signal_to_noise_ratio])
            writer.writerow(['Detection Confidence', analysis.quality_metrics.detection_confidence])
            writer.writerow(['Analysis Quality Score', analysis.quality_metrics.analysis_quality_score])
            writer.writerow(['Data Completeness', analysis.quality_metrics.data_completeness])
            writer.writerow([])
        
        # Write frequency analysis summary
        if analysis.frequency_analysis:
            writer.writerow(['Frequency Analysis Summary'])
            writer.writerow(['Wave Type', 'Dominant Frequency', 'Spectral Centroid', 'Bandwidth'])
            for wave_type, freq_data in analysis.frequency_analysis.items():
                writer.writerow([
                    wave_type,
                    freq_data.dominant_frequency,
                    freq_data.spectral_centroid,
                    freq_data.bandwidth
                ])
        
        return buffer.getvalue().encode('utf-8')


class MSEEDExporter:
    """
    Specialized MSEED format exporter for separated wave data.
    
    Provides advanced MSEED export capabilities with proper metadata
    and wave separation markers.
    """
    
    def __init__(self):
        """Initialize MSEED exporter."""
        if not OBSPY_AVAILABLE:
            raise RuntimeError("ObsPy is required for MSEED export but not available")
    
    def export_separated_waves(self, wave_result: WaveAnalysisResult, 
                             station_code: str = "WAVE",
                             network_code: str = "XX") -> bytes:
        """
        Export separated waves with enhanced metadata.
        
        Args:
            wave_result: Wave analysis result containing separated waves
            station_code: Station code for MSEED headers
            network_code: Network code for MSEED headers
            
        Returns:
            MSEED data as bytes
        """
        stream = Stream()
        
        # Create trace for original data
        original_stats = Stats()
        original_stats.network = network_code
        original_stats.station = station_code
        original_stats.channel = "HHZ"
        original_stats.sampling_rate = wave_result.sampling_rate
        original_stats.npts = len(wave_result.original_data)
        original_stats.starttime = UTCDateTime()
        
        original_trace = Trace(
            data=wave_result.original_data.astype(np.float32),
            header=original_stats
        )
        stream.append(original_trace)
        
        # Add separated wave traces
        wave_types = [
            ('P', wave_result.p_waves),
            ('S', wave_result.s_waves),
            ('Surface', wave_result.surface_waves)
        ]
        
        for wave_type, waves in wave_types:
            for i, wave in enumerate(waves):
                stats = Stats()
                stats.network = network_code
                stats.station = f"{station_code}{wave_type[0]}{i:02d}"
                stats.channel = f"HH{wave_type[0]}"
                stats.sampling_rate = wave.sampling_rate
                stats.npts = len(wave.data)
                stats.starttime = UTCDateTime() + wave.start_time
                
                # Add wave-specific metadata
                stats.mseed = {
                    'dataquality': 'D',
                    'wave_type': wave.wave_type,
                    'arrival_time': wave.arrival_time,
                    'peak_amplitude': wave.peak_amplitude,
                    'dominant_frequency': wave.dominant_frequency,
                    'confidence': wave.confidence
                }
                
                trace = Trace(data=wave.data.astype(np.float32), header=stats)
                stream.append(trace)
        
        # Export to buffer
        buffer = io.BytesIO()
        stream.write(buffer, format='MSEED')
        return buffer.getvalue()


class CSVExporter:
    """
    Specialized CSV exporter for wave characteristics and timing data.
    
    Provides detailed CSV export with comprehensive wave analysis data.
    """
    
    def export_wave_characteristics(self, analysis: DetailedAnalysis) -> bytes:
        """
        Export comprehensive wave characteristics in CSV format.
        
        Args:
            analysis: Detailed analysis results
            
        Returns:
            CSV data as bytes
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        
        # Write metadata header
        writer.writerow(['# Wave Analysis Export'])
        writer.writerow(['# Generated:', datetime.now().isoformat()])
        writer.writerow(['# Total Waves:', analysis.wave_result.total_waves_detected])
        writer.writerow([])
        
        # Write detailed wave characteristics
        writer.writerow([
            'Wave Type', 'Segment ID', 'Start Time (s)', 'End Time (s)', 
            'Duration (s)', 'Arrival Time (s)', 'Peak Amplitude', 
            'Dominant Frequency (Hz)', 'Confidence', 'Sampling Rate (Hz)',
            'Sample Count', 'RMS Amplitude', 'Energy'
        ])
        
        # Process all wave types
        all_waves = []
        for wave_type in ['P', 'S', 'Love', 'Rayleigh']:
            waves = analysis.wave_result.get_waves_by_type(wave_type)
            for i, wave in enumerate(waves):
                # Calculate additional characteristics
                rms_amplitude = np.sqrt(np.mean(wave.data**2))
                energy = np.sum(wave.data**2)
                
                all_waves.append([
                    wave.wave_type,
                    i,
                    wave.start_time,
                    wave.end_time,
                    wave.duration,
                    wave.arrival_time,
                    wave.peak_amplitude,
                    wave.dominant_frequency,
                    wave.confidence,
                    wave.sampling_rate,
                    wave.sample_count,
                    rms_amplitude,
                    energy
                ])
        
        # Sort by arrival time
        all_waves.sort(key=lambda x: x[5])  # Sort by arrival_time
        
        for wave_data in all_waves:
            writer.writerow(wave_data)
        
        return buffer.getvalue().encode('utf-8')
    
    def export_timing_data(self, analysis: DetailedAnalysis) -> bytes:
        """
        Export precise timing data for wave arrivals.
        
        Args:
            analysis: Detailed analysis results
            
        Returns:
            CSV data as bytes with timing information
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        
        # Write timing summary
        writer.writerow(['Timing Analysis'])
        writer.writerow(['Parameter', 'Value (seconds)', 'Notes'])
        writer.writerow(['P-wave Arrival', analysis.arrival_times.p_wave_arrival, 'First P-wave detection'])
        writer.writerow(['S-wave Arrival', analysis.arrival_times.s_wave_arrival, 'First S-wave detection'])
        writer.writerow(['Surface Wave Arrival', analysis.arrival_times.surface_wave_arrival, 'First surface wave detection'])
        writer.writerow(['S-P Time Difference', analysis.arrival_times.sp_time_difference, 'Used for distance estimation'])
        
        if analysis.epicenter_distance:
            writer.writerow(['Estimated Distance (km)', analysis.epicenter_distance, 'Based on S-P time'])
        
        writer.writerow([])
        
        # Write individual wave timings
        writer.writerow(['Individual Wave Timings'])
        writer.writerow(['Wave Type', 'Segment', 'Arrival Time', 'Peak Time', 'End Time'])
        
        all_waves = []
        for wave_type in ['P', 'S', 'Love', 'Rayleigh']:
            waves = analysis.wave_result.get_waves_by_type(wave_type)
            for i, wave in enumerate(waves):
                # Find peak time
                peak_idx = np.argmax(np.abs(wave.data))
                peak_time = wave.start_time + (peak_idx / wave.sampling_rate)
                
                all_waves.append([
                    wave.wave_type,
                    i,
                    wave.arrival_time,
                    peak_time,
                    wave.end_time
                ])
        
        # Sort by arrival time
        all_waves.sort(key=lambda x: x[2])
        
        for wave_data in all_waves:
            writer.writerow(wave_data)
        
        return buffer.getvalue().encode('utf-8')