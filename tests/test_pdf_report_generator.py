"""
Unit tests for PDF report generation functionality.

This module tests the PDFReportGenerator class with various analysis
scenarios and validates PDF generation with sample analysis results.
"""

import unittest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import io

# Import the classes we're testing
from wave_analysis.services.pdf_report_generator import PDFReportGenerator, PDFReportError
from wave_analysis.models import (
    DetailedAnalysis, WaveAnalysisResult, WaveSegment, ArrivalTimes,
    MagnitudeEstimate, FrequencyData, QualityMetrics
)


class TestPDFReportGenerator(unittest.TestCase):
    """Test cases for PDFReportGenerator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample wave segments
        self.sample_p_wave = WaveSegment(
            wave_type='P',
            start_time=10.0,
            end_time=15.0,
            data=np.random.randn(500),
            sampling_rate=100.0,
            peak_amplitude=0.05,
            dominant_frequency=8.5,
            arrival_time=10.2,
            confidence=0.85
        )
        
        self.sample_s_wave = WaveSegment(
            wave_type='S',
            start_time=20.0,
            end_time=30.0,
            data=np.random.randn(1000),
            sampling_rate=100.0,
            peak_amplitude=0.08,
            dominant_frequency=4.2,
            arrival_time=20.5,
            confidence=0.78
        )
        
        self.sample_surface_wave = WaveSegment(
            wave_type='Love',
            start_time=45.0,
            end_time=60.0,
            data=np.random.randn(1500),
            sampling_rate=100.0,
            peak_amplitude=0.12,
            dominant_frequency=1.8,
            arrival_time=45.3,
            confidence=0.72
        )
        
        # Create sample wave analysis result
        self.sample_wave_result = WaveAnalysisResult(
            original_data=np.random.randn(8000),
            sampling_rate=100.0,
            p_waves=[self.sample_p_wave],
            s_waves=[self.sample_s_wave],
            surface_waves=[self.sample_surface_wave],
            metadata={'source': 'test_data'},
            processing_timestamp=datetime.now()
        )
        
        # Create sample arrival times
        self.sample_arrival_times = ArrivalTimes(
            p_wave_arrival=10.2,
            s_wave_arrival=20.5,
            surface_wave_arrival=45.3,
            sp_time_difference=10.3
        )
        
        # Create sample magnitude estimates
        self.sample_magnitude_estimates = [
            MagnitudeEstimate(
                method='ML',
                magnitude=4.2,
                confidence=0.85,
                wave_type_used='P',
                metadata={'station_correction': 0.1}
            ),
            MagnitudeEstimate(
                method='Mb',
                magnitude=4.1,
                confidence=0.78,
                wave_type_used='P',
                metadata={'period': 1.2}
            ),
            MagnitudeEstimate(
                method='Ms',
                magnitude=4.3,
                confidence=0.72,
                wave_type_used='Love',
                metadata={'period': 20.0}
            )
        ]
        
        # Create sample frequency data
        frequencies = np.linspace(0, 50, 1000)
        self.sample_frequency_data = {
            'P': FrequencyData(
                frequencies=frequencies,
                power_spectrum=np.exp(-((frequencies - 8.5) / 3.0) ** 2),
                dominant_frequency=8.5,
                frequency_range=(5.0, 15.0),
                spectral_centroid=8.8,
                bandwidth=6.2
            ),
            'S': FrequencyData(
                frequencies=frequencies,
                power_spectrum=np.exp(-((frequencies - 4.2) / 2.0) ** 2),
                dominant_frequency=4.2,
                frequency_range=(2.0, 8.0),
                spectral_centroid=4.5,
                bandwidth=4.1
            )
        }
        
        # Create sample quality metrics
        self.sample_quality_metrics = QualityMetrics(
            signal_to_noise_ratio=15.2,
            detection_confidence=0.78,
            analysis_quality_score=0.82,
            data_completeness=0.95,
            processing_warnings=['Low frequency noise detected']
        )
        
        # Create comprehensive detailed analysis
        self.sample_detailed_analysis = DetailedAnalysis(
            wave_result=self.sample_wave_result,
            arrival_times=self.sample_arrival_times,
            magnitude_estimates=self.sample_magnitude_estimates,
            epicenter_distance=85.4,
            frequency_analysis=self.sample_frequency_data,
            quality_metrics=self.sample_quality_metrics,
            analysis_timestamp=datetime.now(),
            processing_metadata={
                'processing_time': 2.34,
                'model_version': '1.0.0',
                'algorithm': 'STA/LTA'
            }
        )
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_pdf_generator_initialization(self):
        """Test PDF generator initialization."""
        generator = PDFReportGenerator()
        self.assertIsNotNone(generator)
        self.assertEqual(generator.page_size, (612.0, 792.0))  # letter size
        self.assertIsNotNone(generator.styles)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', False)
    def test_pdf_generator_initialization_no_reportlab(self):
        """Test PDF generator initialization without ReportLab."""
        with self.assertRaises(RuntimeError) as context:
            PDFReportGenerator()
        self.assertIn("ReportLab is required", str(context.exception))
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.SimpleDocTemplate')
    @patch('wave_analysis.services.pdf_report_generator.getSampleStyleSheet')
    def test_generate_basic_report(self, mock_styles, mock_doc):
        """Test basic PDF report generation."""
        # Mock the document and styles
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance
        
        # Create a proper mock styles object that is subscriptable
        mock_styles_obj = Mock()
        mock_styles_obj.__getitem__ = Mock(return_value=Mock())
        mock_styles_obj.add = Mock()
        mock_styles.return_value = mock_styles_obj
        
        generator = PDFReportGenerator()
        
        # Generate report
        result = generator.generate_report(self.sample_detailed_analysis)
        
        # Verify document was created and built
        mock_doc.assert_called_once()
        mock_doc_instance.build.assert_called_once()
        self.assertIsInstance(result, bytes)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_title_page(self):
        """Test title page creation."""
        generator = PDFReportGenerator()
        
        title_content = generator._create_title_page(
            "Test Earthquake Report", 
            self.sample_detailed_analysis
        )
        
        self.assertIsInstance(title_content, list)
        self.assertGreater(len(title_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_executive_summary(self):
        """Test executive summary creation."""
        generator = PDFReportGenerator()
        
        summary_content = generator._create_executive_summary(self.sample_detailed_analysis)
        
        self.assertIsInstance(summary_content, list)
        self.assertGreater(len(summary_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_wave_detection_section(self):
        """Test wave detection section creation."""
        generator = PDFReportGenerator()
        
        wave_content = generator._create_wave_detection_section(self.sample_detailed_analysis)
        
        self.assertIsInstance(wave_content, list)
        self.assertGreater(len(wave_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_timing_analysis_section(self):
        """Test timing analysis section creation."""
        generator = PDFReportGenerator()
        
        timing_content = generator._create_timing_analysis_section(self.sample_detailed_analysis)
        
        self.assertIsInstance(timing_content, list)
        self.assertGreater(len(timing_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_magnitude_section(self):
        """Test magnitude estimation section creation."""
        generator = PDFReportGenerator()
        
        magnitude_content = generator._create_magnitude_section(self.sample_detailed_analysis)
        
        self.assertIsInstance(magnitude_content, list)
        self.assertGreater(len(magnitude_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_magnitude_section_no_estimates(self):
        """Test magnitude section with no estimates."""
        # Create analysis without magnitude estimates
        analysis_no_mag = DetailedAnalysis(
            wave_result=self.sample_wave_result,
            arrival_times=self.sample_arrival_times,
            magnitude_estimates=[],
            epicenter_distance=None,
            frequency_analysis={},
            quality_metrics=None
        )
        
        generator = PDFReportGenerator()
        magnitude_content = generator._create_magnitude_section(analysis_no_mag)
        
        self.assertIsInstance(magnitude_content, list)
        self.assertGreater(len(magnitude_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_frequency_analysis_section(self):
        """Test frequency analysis section creation."""
        generator = PDFReportGenerator()
        
        freq_content = generator._create_frequency_analysis_section(self.sample_detailed_analysis)
        
        self.assertIsInstance(freq_content, list)
        self.assertGreater(len(freq_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_frequency_analysis_section_no_data(self):
        """Test frequency analysis section with no data."""
        # Create analysis without frequency data
        analysis_no_freq = DetailedAnalysis(
            wave_result=self.sample_wave_result,
            arrival_times=self.sample_arrival_times,
            magnitude_estimates=[],
            epicenter_distance=None,
            frequency_analysis={},
            quality_metrics=None
        )
        
        generator = PDFReportGenerator()
        freq_content = generator._create_frequency_analysis_section(analysis_no_freq)
        
        self.assertIsInstance(freq_content, list)
        self.assertGreater(len(freq_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_quality_metrics_section(self):
        """Test quality metrics section creation."""
        generator = PDFReportGenerator()
        
        quality_content = generator._create_quality_metrics_section(self.sample_detailed_analysis)
        
        self.assertIsInstance(quality_content, list)
        self.assertGreater(len(quality_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_quality_metrics_section_no_metrics(self):
        """Test quality metrics section with no metrics."""
        # Create analysis without quality metrics
        analysis_no_quality = DetailedAnalysis(
            wave_result=self.sample_wave_result,
            arrival_times=self.sample_arrival_times,
            magnitude_estimates=[],
            epicenter_distance=None,
            frequency_analysis={},
            quality_metrics=None
        )
        
        generator = PDFReportGenerator()
        quality_content = generator._create_quality_metrics_section(analysis_no_quality)
        
        self.assertIsInstance(quality_content, list)
        self.assertGreater(len(quality_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_assess_snr_quality(self):
        """Test SNR quality assessment."""
        generator = PDFReportGenerator()
        
        self.assertEqual(generator._assess_snr(25.0), "Excellent")
        self.assertEqual(generator._assess_snr(15.0), "Good")
        self.assertEqual(generator._assess_snr(7.0), "Fair")
        self.assertEqual(generator._assess_snr(3.0), "Poor")
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_assess_confidence_quality(self):
        """Test confidence quality assessment."""
        generator = PDFReportGenerator()
        
        self.assertEqual(generator._assess_confidence(0.95), "Very High")
        self.assertEqual(generator._assess_confidence(0.8), "High")
        self.assertEqual(generator._assess_confidence(0.6), "Moderate")
        self.assertEqual(generator._assess_confidence(0.3), "Low")
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_assess_quality_score(self):
        """Test quality score assessment."""
        generator = PDFReportGenerator()
        
        self.assertEqual(generator._assess_quality_score(0.9), "Excellent")
        self.assertEqual(generator._assess_quality_score(0.7), "Good")
        self.assertEqual(generator._assess_quality_score(0.5), "Fair")
        self.assertEqual(generator._assess_quality_score(0.3), "Poor")
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_assess_completeness(self):
        """Test data completeness assessment."""
        generator = PDFReportGenerator()
        
        self.assertEqual(generator._assess_completeness(0.98), "Complete")
        self.assertEqual(generator._assess_completeness(0.85), "Nearly Complete")
        self.assertEqual(generator._assess_completeness(0.7), "Mostly Complete")
        self.assertEqual(generator._assess_completeness(0.5), "Incomplete")
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.MATPLOTLIB_AVAILABLE', False)
    def test_create_visualization_section_no_matplotlib(self):
        """Test visualization section without matplotlib."""
        generator = PDFReportGenerator()
        
        viz_content = generator._create_visualization_section(self.sample_detailed_analysis)
        
        self.assertIsInstance(viz_content, list)
        self.assertGreater(len(viz_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.MATPLOTLIB_AVAILABLE', True)
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_time_series_plot(self, mock_close, mock_savefig, mock_subplots):
        """Test time series plot creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        generator = PDFReportGenerator()
        
        plot_image = generator._create_time_series_plot(self.sample_detailed_analysis)
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.MATPLOTLIB_AVAILABLE', True)
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_frequency_spectrum_plot(self, mock_close, mock_savefig, mock_subplots):
        """Test frequency spectrum plot creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        generator = PDFReportGenerator()
        
        plot_image = generator._create_frequency_spectrum_plot(self.sample_detailed_analysis)
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.MATPLOTLIB_AVAILABLE', True)
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_arrival_timeline(self, mock_close, mock_savefig, mock_subplots):
        """Test arrival timeline creation."""
        # Mock matplotlib components
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        generator = PDFReportGenerator()
        
        timeline_image = generator._create_arrival_timeline(self.sample_detailed_analysis)
        
        # Verify matplotlib was called
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_raw_data_section(self):
        """Test raw data section creation."""
        generator = PDFReportGenerator()
        
        raw_data_content = generator._create_raw_data_section(self.sample_detailed_analysis)
        
        self.assertIsInstance(raw_data_content, list)
        self.assertGreater(len(raw_data_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    def test_create_appendix(self):
        """Test appendix creation."""
        generator = PDFReportGenerator()
        
        appendix_content = generator._create_appendix(self.sample_detailed_analysis)
        
        self.assertIsInstance(appendix_content, list)
        self.assertGreater(len(appendix_content), 0)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.SimpleDocTemplate')
    @patch('wave_analysis.services.pdf_report_generator.getSampleStyleSheet')
    def test_generate_report_with_raw_data(self, mock_styles, mock_doc):
        """Test PDF report generation with raw data included."""
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance
        
        # Create a proper mock styles object that is subscriptable
        mock_styles_obj = Mock()
        mock_styles_obj.__getitem__ = Mock(return_value=Mock())
        mock_styles_obj.add = Mock()
        mock_styles.return_value = mock_styles_obj
        
        generator = PDFReportGenerator()
        
        result = generator.generate_report(
            self.sample_detailed_analysis,
            title="Test Report with Raw Data",
            include_raw_data=True
        )
        
        mock_doc_instance.build.assert_called_once()
        self.assertIsInstance(result, bytes)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.SimpleDocTemplate')
    def test_generate_report_error_handling(self, mock_doc):
        """Test PDF report generation error handling."""
        mock_doc_instance = Mock()
        mock_doc_instance.build.side_effect = Exception("PDF generation failed")
        mock_doc.return_value = mock_doc_instance
        
        generator = PDFReportGenerator()
        generator.styles = Mock()
        generator.styles.add = Mock()
        
        with self.assertRaises(RuntimeError) as context:
            generator.generate_report(self.sample_detailed_analysis)
        
        self.assertIn("PDF report generation failed", str(context.exception))
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.SimpleDocTemplate')
    @patch('wave_analysis.services.pdf_report_generator.getSampleStyleSheet')
    def test_minimal_analysis_data(self, mock_styles, mock_doc):
        """Test PDF generation with minimal analysis data."""
        # Create minimal analysis with only basic wave result
        minimal_wave_result = WaveAnalysisResult(
            original_data=np.random.randn(1000),
            sampling_rate=100.0,
            p_waves=[],
            s_waves=[],
            surface_waves=[],
            metadata={}
        )
        
        minimal_analysis = DetailedAnalysis(
            wave_result=minimal_wave_result,
            arrival_times=ArrivalTimes(),
            magnitude_estimates=[],
            epicenter_distance=None,
            frequency_analysis={},
            quality_metrics=None
        )
        
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance
        
        # Create a proper mock styles object that is subscriptable
        mock_styles_obj = Mock()
        mock_styles_obj.__getitem__ = Mock(return_value=Mock())
        mock_styles_obj.add = Mock()
        mock_styles.return_value = mock_styles_obj
        
        generator = PDFReportGenerator()
        
        result = generator.generate_report(minimal_analysis)
        
        mock_doc_instance.build.assert_called_once()
        self.assertIsInstance(result, bytes)


class TestPDFReportIntegration(unittest.TestCase):
    """Integration tests for PDF report generation."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Create realistic earthquake analysis data
        self.realistic_analysis = self._create_realistic_analysis()
    
    def _create_realistic_analysis(self) -> DetailedAnalysis:
        """Create realistic earthquake analysis data for testing."""
        # Simulate a magnitude 4.5 earthquake at 50km distance
        
        # P-wave: arrives first, higher frequency
        p_wave_data = self._generate_realistic_wave_data(
            duration=3.0, sampling_rate=100.0, 
            dominant_freq=12.0, amplitude=0.02
        )
        p_wave = WaveSegment(
            wave_type='P',
            start_time=8.5,
            end_time=11.5,
            data=p_wave_data,
            sampling_rate=100.0,
            peak_amplitude=0.025,
            dominant_frequency=12.0,
            arrival_time=8.7,
            confidence=0.89
        )
        
        # S-wave: arrives later, intermediate frequency, higher amplitude
        s_wave_data = self._generate_realistic_wave_data(
            duration=5.0, sampling_rate=100.0,
            dominant_freq=6.0, amplitude=0.04
        )
        s_wave = WaveSegment(
            wave_type='S',
            start_time=15.2,
            end_time=20.2,
            data=s_wave_data,
            sampling_rate=100.0,
            peak_amplitude=0.048,
            dominant_frequency=6.0,
            arrival_time=15.4,
            confidence=0.82
        )
        
        # Love wave: surface wave, lower frequency, highest amplitude
        love_wave_data = self._generate_realistic_wave_data(
            duration=8.0, sampling_rate=100.0,
            dominant_freq=2.5, amplitude=0.06
        )
        love_wave = WaveSegment(
            wave_type='Love',
            start_time=35.0,
            end_time=43.0,
            data=love_wave_data,
            sampling_rate=100.0,
            peak_amplitude=0.065,
            dominant_frequency=2.5,
            arrival_time=35.3,
            confidence=0.75
        )
        
        # Create original data combining all waves with noise
        original_data = np.random.randn(6000) * 0.005  # Background noise
        # Add P-wave
        p_start_idx = int(8.5 * 100)
        p_end_idx = int(11.5 * 100)
        original_data[p_start_idx:p_end_idx] += p_wave_data
        # Add S-wave
        s_start_idx = int(15.2 * 100)
        s_end_idx = int(20.2 * 100)
        original_data[s_start_idx:s_end_idx] += s_wave_data
        # Add Love wave
        l_start_idx = int(35.0 * 100)
        l_end_idx = int(43.0 * 100)
        original_data[l_start_idx:l_end_idx] += love_wave_data
        
        wave_result = WaveAnalysisResult(
            original_data=original_data,
            sampling_rate=100.0,
            p_waves=[p_wave],
            s_waves=[s_wave],
            surface_waves=[love_wave],
            metadata={
                'station': 'TEST01',
                'location': 'Test Location',
                'event_id': 'test_earthquake_001'
            }
        )
        
        arrival_times = ArrivalTimes(
            p_wave_arrival=8.7,
            s_wave_arrival=15.4,
            surface_wave_arrival=35.3,
            sp_time_difference=6.7
        )
        
        magnitude_estimates = [
            MagnitudeEstimate('ML', 4.5, 0.88, 'P'),
            MagnitudeEstimate('Mb', 4.4, 0.82, 'P'),
            MagnitudeEstimate('Ms', 4.6, 0.79, 'Love')
        ]
        
        # Create realistic frequency data
        frequencies = np.linspace(0, 50, 1000)
        frequency_analysis = {
            'P': FrequencyData(
                frequencies=frequencies,
                power_spectrum=np.exp(-((frequencies - 12.0) / 4.0) ** 2),
                dominant_frequency=12.0,
                frequency_range=(8.0, 18.0),
                spectral_centroid=12.5,
                bandwidth=8.2
            ),
            'S': FrequencyData(
                frequencies=frequencies,
                power_spectrum=np.exp(-((frequencies - 6.0) / 2.5) ** 2),
                dominant_frequency=6.0,
                frequency_range=(3.0, 10.0),
                spectral_centroid=6.3,
                bandwidth=5.1
            ),
            'Love': FrequencyData(
                frequencies=frequencies,
                power_spectrum=np.exp(-((frequencies - 2.5) / 1.2) ** 2),
                dominant_frequency=2.5,
                frequency_range=(1.0, 5.0),
                spectral_centroid=2.7,
                bandwidth=2.8
            )
        }
        
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=18.5,
            detection_confidence=0.83,
            analysis_quality_score=0.85,
            data_completeness=0.98,
            processing_warnings=[]
        )
        
        return DetailedAnalysis(
            wave_result=wave_result,
            arrival_times=arrival_times,
            magnitude_estimates=magnitude_estimates,
            epicenter_distance=55.8,  # Estimated from S-P time
            frequency_analysis=frequency_analysis,
            quality_metrics=quality_metrics,
            processing_metadata={
                'processing_time': 3.45,
                'model_version': '2.1.0',
                'algorithm': 'Advanced STA/LTA with ML enhancement'
            }
        )
    
    def _generate_realistic_wave_data(self, duration: float, sampling_rate: float,
                                    dominant_freq: float, amplitude: float) -> np.ndarray:
        """Generate realistic seismic wave data."""
        n_samples = int(duration * sampling_rate)
        t = np.linspace(0, duration, n_samples)
        
        # Create wave with dominant frequency and some harmonics
        wave = (amplitude * np.sin(2 * np.pi * dominant_freq * t) * 
                np.exp(-t / (duration * 0.3)))  # Exponential decay
        
        # Add some harmonics
        wave += (amplitude * 0.3 * np.sin(2 * np.pi * dominant_freq * 2 * t) * 
                np.exp(-t / (duration * 0.2)))
        
        # Add noise
        wave += np.random.randn(n_samples) * amplitude * 0.1
        
        return wave
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.SimpleDocTemplate')
    @patch('wave_analysis.services.pdf_report_generator.getSampleStyleSheet')
    def test_realistic_earthquake_report(self, mock_styles, mock_doc):
        """Test PDF generation with realistic earthquake data."""
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance
        
        # Create a proper mock styles object that is subscriptable
        mock_styles_obj = Mock()
        mock_styles_obj.__getitem__ = Mock(return_value=Mock())
        mock_styles_obj.add = Mock()
        mock_styles.return_value = mock_styles_obj
        
        generator = PDFReportGenerator()
        
        result = generator.generate_report(
            self.realistic_analysis,
            title="M4.5 Earthquake Analysis Report - Station TEST01",
            include_raw_data=True
        )
        
        mock_doc_instance.build.assert_called_once()
        self.assertIsInstance(result, bytes)
    
    @patch('wave_analysis.services.pdf_report_generator.REPORTLAB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.MATPLOTLIB_AVAILABLE', True)
    @patch('wave_analysis.services.pdf_report_generator.SimpleDocTemplate')
    @patch('wave_analysis.services.pdf_report_generator.getSampleStyleSheet')
    def test_report_with_visualizations(self, mock_styles, mock_doc):
        """Test PDF generation with visualization components."""
        mock_doc_instance = Mock()
        mock_doc.return_value = mock_doc_instance
        
        # Create a proper mock styles object that is subscriptable
        mock_styles_obj = Mock()
        mock_styles_obj.__getitem__ = Mock(return_value=Mock())
        mock_styles_obj.add = Mock()
        mock_styles.return_value = mock_styles_obj
        
        generator = PDFReportGenerator()
        
        # Mock matplotlib functions
        with patch('matplotlib.pyplot.subplots') as mock_subplots:
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    mock_fig = Mock()
                    mock_axes = [Mock(), Mock(), Mock()]
                    mock_subplots.return_value = (mock_fig, mock_axes)
                    
                    result = generator.generate_report(self.realistic_analysis)
                    
                    mock_doc_instance.build.assert_called_once()
                    self.assertIsInstance(result, bytes)


if __name__ == '__main__':
    unittest.main()