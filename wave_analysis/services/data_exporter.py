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
    from obspy.core import Stats
    OBSPY_AVAILABLE = True
except ImportError:
    OBSPY_AVAILABLE = False
    # Create dummy classes for when ObsPy is not available
    class Stream:
        pass
    class Trace:
        pass
    class UTCDateTime:
        pass
    class Stats:
        pass

from ..interfaces import WaveExporterInterface
from ..models.wave_models import WaveSegment, WaveAnalysisResult, DetailedAnalysis


class DataExporter(WaveExporterInterface):
    """
    Main data exporter class supporting multiple export formats.
    
    Supports export of separated wave data in MSEED, CSV formats
    and comprehensive analysis results.
    """
    
    def __init__(self):
        """Initialize the data exporter."""
        self.supported_formats = ['mseed', 'csv', 'json', 'sac']
        if not OBSPY_AVAILABLE:
            # Remove MSEED and SAC support if ObsPy is not available
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
            elif format_type == 'sac':
                return self._export_sac(waves)
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
    
    def _export_sac(self, waves: Dict[str, List[WaveSegment]]) -> bytes:
        """
        Export wave data in SAC format as a ZIP archive containing multiple SAC files.
        
        Args:
            waves: Dictionary of wave segments by type
            
        Returns:
            ZIP archive containing SAC files as bytes
        """
        if not OBSPY_AVAILABLE:
            raise RuntimeError("ObsPy is required for SAC export but not available")
        
        import zipfile
        
        # Create a ZIP buffer to contain multiple SAC files
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Create SAC exporter instance
            sac_exporter = SACExporter()
            
            # Create a mock WaveAnalysisResult for the SAC exporter
            # This is a simplified approach for the basic export_waves interface
            mock_wave_result = type('MockWaveAnalysisResult', (), {
                'original_data': np.concatenate([seg.data for segments in waves.values() for seg in segments]) if waves else np.array([]),
                'sampling_rate': next(iter(next(iter(waves.values())))).sampling_rate if waves else 100.0,
                'p_waves': waves.get('P', []),
                's_waves': waves.get('S', []),
                'surface_waves': waves.get('Love', []) + waves.get('Rayleigh', []),
                'metadata': {}
            })()
            
            # Export individual SAC files
            sac_files = sac_exporter.export_separated_waves(mock_wave_result)
            
            # Add each SAC file to the ZIP archive
            for filename, sac_data in sac_files.items():
                zip_file.writestr(filename, sac_data)
        
        return zip_buffer.getvalue()
    
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


class SACExporter:
    """
    Specialized SAC format exporter for seismic data.
    
    Provides SAC format export with proper metadata preservation
    and wave separation markers in SAC file headers.
    """
    
    def __init__(self):
        """Initialize SAC exporter."""
        if not OBSPY_AVAILABLE:
            raise RuntimeError("ObsPy is required for SAC export but not available")
    
    def export_separated_waves(self, wave_result: WaveAnalysisResult,
                             station_code: str = "WAVE",
                             network_code: str = "XX",
                             location_code: str = "00") -> Dict[str, bytes]:
        """
        Export separated waves as individual SAC files.
        
        Args:
            wave_result: Wave analysis result containing separated waves
            station_code: Station code for SAC headers
            network_code: Network code for SAC headers
            location_code: Location code for SAC headers
            
        Returns:
            Dictionary mapping filenames to SAC file data as bytes
        """
        sac_files = {}
        
        # Export original data
        original_filename = f"{network_code}.{station_code}.{location_code}.HHZ.original.sac"
        sac_files[original_filename] = self._create_sac_file(
            data=wave_result.original_data,
            sampling_rate=wave_result.sampling_rate,
            station_code=station_code,
            network_code=network_code,
            location_code=location_code,
            channel_code="HHZ",
            wave_type="original",
            metadata=wave_result.metadata
        )
        
        # Export P-waves
        for i, p_wave in enumerate(wave_result.p_waves):
            filename = f"{network_code}.{station_code}.{location_code}.HHP{i:02d}.sac"
            sac_files[filename] = self._create_sac_file(
                data=p_wave.data,
                sampling_rate=p_wave.sampling_rate,
                station_code=station_code,
                network_code=network_code,
                location_code=location_code,
                channel_code=f"HHP{i:02d}",
                wave_type="P",
                wave_segment=p_wave,
                start_time_offset=p_wave.start_time
            )
        
        # Export S-waves
        for i, s_wave in enumerate(wave_result.s_waves):
            filename = f"{network_code}.{station_code}.{location_code}.HHS{i:02d}.sac"
            sac_files[filename] = self._create_sac_file(
                data=s_wave.data,
                sampling_rate=s_wave.sampling_rate,
                station_code=station_code,
                network_code=network_code,
                location_code=location_code,
                channel_code=f"HHS{i:02d}",
                wave_type="S",
                wave_segment=s_wave,
                start_time_offset=s_wave.start_time
            )
        
        # Export surface waves
        for i, surface_wave in enumerate(wave_result.surface_waves):
            wave_code = "L" if surface_wave.wave_type == "Love" else "R"
            filename = f"{network_code}.{station_code}.{location_code}.HH{wave_code}{i:02d}.sac"
            sac_files[filename] = self._create_sac_file(
                data=surface_wave.data,
                sampling_rate=surface_wave.sampling_rate,
                station_code=station_code,
                network_code=network_code,
                location_code=location_code,
                channel_code=f"HH{wave_code}{i:02d}",
                wave_type=surface_wave.wave_type,
                wave_segment=surface_wave,
                start_time_offset=surface_wave.start_time
            )
        
        return sac_files
    
    def export_single_sac_file(self, wave_result: WaveAnalysisResult,
                              include_markers: bool = True,
                              station_code: str = "WAVE",
                              network_code: str = "XX") -> bytes:
        """
        Export all wave data in a single SAC file with wave markers.
        
        Args:
            wave_result: Wave analysis result
            include_markers: Whether to include wave arrival markers
            station_code: Station code for SAC headers
            network_code: Network code for SAC headers
            
        Returns:
            SAC file data as bytes
        """
        # Create SAC file with original data
        sac_data = self._create_sac_file(
            data=wave_result.original_data,
            sampling_rate=wave_result.sampling_rate,
            station_code=station_code,
            network_code=network_code,
            location_code="00",
            channel_code="HHZ",
            wave_type="composite",
            metadata=wave_result.metadata
        )
        
        if include_markers:
            # Add wave arrival markers to the SAC file
            sac_data = self._add_wave_markers(sac_data, wave_result)
        
        return sac_data
    
    def _create_sac_file(self, data: np.ndarray, sampling_rate: float,
                        station_code: str, network_code: str, location_code: str,
                        channel_code: str, wave_type: str,
                        wave_segment: Optional[WaveSegment] = None,
                        start_time_offset: float = 0.0,
                        metadata: Optional[Dict[str, Any]] = None) -> bytes:
        """
        Create a SAC file with proper headers and metadata.
        
        Args:
            data: Seismic data array
            sampling_rate: Sampling rate in Hz
            station_code: Station code
            network_code: Network code
            location_code: Location code
            channel_code: Channel code
            wave_type: Type of wave data
            wave_segment: Optional wave segment for additional metadata
            start_time_offset: Time offset from original data start
            metadata: Additional metadata
            
        Returns:
            SAC file data as bytes
        """
        # Create ObsPy Stats object for SAC headers
        stats = Stats()
        stats.network = network_code
        stats.station = station_code
        stats.location = location_code
        stats.channel = channel_code
        stats.sampling_rate = sampling_rate
        stats.npts = len(data)
        stats.starttime = UTCDateTime() + start_time_offset
        
        # Set SAC-specific headers
        stats.sac = {}
        
        # Basic SAC headers
        stats.sac.delta = 1.0 / sampling_rate
        stats.sac.npts = len(data)
        stats.sac.b = 0.0  # Begin time
        stats.sac.e = (len(data) - 1) / sampling_rate  # End time
        stats.sac.iftype = 1  # Time series file
        stats.sac.leven = 1  # Evenly spaced data
        
        # Station information
        stats.sac.kstnm = station_code
        stats.sac.knetwk = network_code
        stats.sac.kcmpnm = channel_code
        stats.sac.khole = location_code
        
        # Wave type information in user-defined headers
        stats.sac.kuser0 = wave_type[:8]  # SAC string limit is 8 chars
        
        # Add wave segment specific information
        if wave_segment:
            stats.sac.user0 = wave_segment.arrival_time
            stats.sac.user1 = wave_segment.peak_amplitude
            stats.sac.user2 = wave_segment.dominant_frequency
            stats.sac.user3 = wave_segment.confidence
            stats.sac.user4 = wave_segment.duration
            
            # Add wave characteristics to header comments
            stats.sac.kuser1 = f"AMP{wave_segment.peak_amplitude:.3f}"[:8]
            stats.sac.kuser2 = f"FRQ{wave_segment.dominant_frequency:.2f}"[:8]
        
        # Add metadata if provided
        if metadata:
            # Store processing information
            if 'processing_time' in metadata:
                stats.sac.user5 = metadata['processing_time']
            if 'model_version' in metadata:
                stats.sac.kuser3 = str(metadata['model_version'])[:8]
        
        # Set data quality and processing flags
        stats.sac.idep = 50  # Displacement in nm (generic units)
        stats.sac.iztype = 9  # Reference time is arbitrary
        
        # Create trace and export to SAC
        trace = Trace(data=data.astype(np.float32), header=stats)
        
        # Export to buffer
        buffer = io.BytesIO()
        trace.write(buffer, format='SAC')
        return buffer.getvalue()
    
    def _add_wave_markers(self, sac_data: bytes, wave_result: WaveAnalysisResult) -> bytes:
        """
        Add wave arrival markers to existing SAC file data.
        
        Args:
            sac_data: Original SAC file data
            wave_result: Wave analysis result with arrival times
            
        Returns:
            Modified SAC file data with markers
        """
        # Read the SAC file back to modify headers
        buffer = io.BytesIO(sac_data)
        trace = Trace()
        trace.read(buffer, format='SAC')
        
        # Add arrival time markers
        if wave_result.p_waves:
            # T0 marker for first P-wave arrival
            trace.stats.sac.t0 = wave_result.p_waves[0].arrival_time
            trace.stats.sac.kt0 = "P"
        
        if wave_result.s_waves:
            # T1 marker for first S-wave arrival
            trace.stats.sac.t1 = wave_result.s_waves[0].arrival_time
            trace.stats.sac.kt1 = "S"
        
        if wave_result.surface_waves:
            # T2 marker for first surface wave arrival
            trace.stats.sac.t2 = wave_result.surface_waves[0].arrival_time
            trace.stats.sac.kt2 = "SURF"
        
        # Add additional markers for multiple waves of same type
        marker_index = 3
        for wave_type, waves in [("P", wave_result.p_waves[1:]), 
                                ("S", wave_result.s_waves[1:]),
                                ("L", [w for w in wave_result.surface_waves[1:] if w.wave_type == "Love"]),
                                ("R", [w for w in wave_result.surface_waves[1:] if w.wave_type == "Rayleigh"])]:
            for wave in waves:
                if marker_index <= 9:  # SAC supports T0-T9 markers
                    setattr(trace.stats.sac, f't{marker_index}', wave.arrival_time)
                    setattr(trace.stats.sac, f'kt{marker_index}', wave_type)
                    marker_index += 1
                else:
                    break
        
        # Export modified trace
        buffer = io.BytesIO()
        trace.write(buffer, format='SAC')
        return buffer.getvalue()
    
    def validate_sac_export(self, sac_data: bytes) -> Dict[str, Any]:
        """
        Validate exported SAC file and return header information.
        
        Args:
            sac_data: SAC file data to validate
            
        Returns:
            Dictionary with validation results and header information
        """
        try:
            # Read SAC file to validate
            buffer = io.BytesIO(sac_data)
            trace = Trace()
            trace.read(buffer, format='SAC')
            
            validation_result = {
                'valid': True,
                'file_size': len(sac_data),
                'header_info': {
                    'station': trace.stats.station,
                    'network': trace.stats.network,
                    'channel': trace.stats.channel,
                    'location': trace.stats.location,
                    'sampling_rate': trace.stats.sampling_rate,
                    'npts': trace.stats.npts,
                    'starttime': str(trace.stats.starttime),
                    'endtime': str(trace.stats.endtime)
                },
                'sac_headers': {},
                'wave_markers': {}
            }
            
            # Extract SAC-specific headers
            if hasattr(trace.stats, 'sac'):
                sac_headers = trace.stats.sac
                validation_result['sac_headers'] = {
                    'delta': getattr(sac_headers, 'delta', None),
                    'b': getattr(sac_headers, 'b', None),
                    'e': getattr(sac_headers, 'e', None),
                    'wave_type': getattr(sac_headers, 'kuser0', None),
                    'arrival_time': getattr(sac_headers, 'user0', None),
                    'peak_amplitude': getattr(sac_headers, 'user1', None),
                    'dominant_frequency': getattr(sac_headers, 'user2', None),
                    'confidence': getattr(sac_headers, 'user3', None)
                }
                
                # Extract wave markers
                for i in range(10):  # T0-T9 markers
                    t_attr = f't{i}'
                    kt_attr = f'kt{i}'
                    if hasattr(sac_headers, t_attr) and hasattr(sac_headers, kt_attr):
                        t_val = getattr(sac_headers, t_attr)
                        kt_val = getattr(sac_headers, kt_attr)
                        if t_val is not None and kt_val:
                            validation_result['wave_markers'][f'T{i}'] = {
                                'time': t_val,
                                'label': kt_val
                            }
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_size': len(sac_data)
            }


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