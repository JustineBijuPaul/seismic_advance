"""
PDF report generation service for wave analysis results.

This module provides comprehensive PDF report generation capabilities
for earthquake wave analysis results, including visualizations and
detailed measurements.
"""

import io
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import base64

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import black, blue, red, green, orange
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.graphics.shapes import Drawing, Line, Rect
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Create dummy classes and constants for when ReportLab is not available
    letter = (612.0, 792.0)
    A4 = (595.276, 841.89)
    inch = 72.0
    TA_CENTER = 1
    TA_LEFT = 0
    TA_RIGHT = 2
    
    class MockColors:
        darkblue = 'darkblue'
        darkgreen = 'darkgreen'
        darkorange = 'darkorange'
        darkred = 'darkred'
        purple = 'purple'
        whitesmoke = 'whitesmoke'
        beige = 'beige'
        lightgrey = 'lightgrey'
        lightyellow = 'lightyellow'
        lightpink = 'lightpink'
        lavender = 'lavender'
        black = 'black'
        blue = 'blue'
        red = 'red'
        green = 'green'
        orange = 'orange'
    
    colors = MockColors()
    black = blue = red = green = orange = 'color'
    
    class Image:
        def __init__(self, *args, **kwargs):
            pass
    
    class SimpleDocTemplate:
        def __init__(self, *args, **kwargs):
            pass
        def build(self, story):
            pass
    
    class Paragraph:
        def __init__(self, *args, **kwargs):
            pass
    
    class Spacer:
        def __init__(self, *args, **kwargs):
            pass
    
    class Table:
        def __init__(self, *args, **kwargs):
            pass
        def setStyle(self, style):
            pass
    
    class TableStyle:
        def __init__(self, *args, **kwargs):
            pass
    
    class PageBreak:
        def __init__(self, *args, **kwargs):
            pass
    
    class KeepTogether:
        def __init__(self, *args, **kwargs):
            pass
    
    class ParagraphStyle:
        def __init__(self, *args, **kwargs):
            pass
    
    def getSampleStyleSheet():
        class MockStyles:
            def __init__(self):
                self.styles = {
                    'Title': ParagraphStyle(),
                    'Heading1': ParagraphStyle(),
                    'Heading2': ParagraphStyle(),
                    'Normal': ParagraphStyle()
                }
            def __getitem__(self, key):
                return self.styles.get(key, ParagraphStyle())
            def add(self, style):
                if hasattr(style, 'name'):
                    self.styles[style.name] = style
        return MockStyles()

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..models import DetailedAnalysis, WaveSegment, MagnitudeEstimate


class PDFReportGenerator:
    """
    PDF report generator for comprehensive wave analysis results.
    
    Creates professional PDF reports with wave analysis summary,
    visualizations, and detailed measurements.
    """
    
    def __init__(self):
        """Initialize the PDF report generator."""
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab is required for PDF generation but not available")
        
        self.page_size = letter
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the report."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkgreen
        ))
        
        # Data style for measurements
        self.styles.add(ParagraphStyle(
            name='DataStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            fontName='Courier',
            leftIndent=20
        ))
        
        # Warning style
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            leftIndent=20
        ))
    
    def generate_report(self, analysis: DetailedAnalysis, 
                       title: str = "Earthquake Wave Analysis Report",
                       include_raw_data: bool = False) -> bytes:
        """
        Generate comprehensive PDF report for wave analysis results.
        
        Args:
            analysis: Detailed analysis results to include in report
            title: Report title
            include_raw_data: Whether to include raw wave data tables
            
        Returns:
            PDF report as bytes
            
        Raises:
            RuntimeError: If report generation fails
        """
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=self.page_size,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build report content
            story = []
            
            # Title page
            story.extend(self._create_title_page(title, analysis))
            story.append(PageBreak())
            
            # Executive summary
            story.extend(self._create_executive_summary(analysis))
            story.append(PageBreak())
            
            # Wave detection results
            story.extend(self._create_wave_detection_section(analysis))
            
            # Timing analysis
            story.extend(self._create_timing_analysis_section(analysis))
            
            # Magnitude estimates
            story.extend(self._create_magnitude_section(analysis))
            
            # Frequency analysis
            story.extend(self._create_frequency_analysis_section(analysis))
            
            # Quality metrics
            story.extend(self._create_quality_metrics_section(analysis))
            
            # Visualizations
            if MATPLOTLIB_AVAILABLE:
                story.append(PageBreak())
                story.extend(self._create_visualization_section(analysis))
            
            # Raw data tables (optional)
            if include_raw_data:
                story.append(PageBreak())
                story.extend(self._create_raw_data_section(analysis))
            
            # Appendix
            story.append(PageBreak())
            story.extend(self._create_appendix(analysis))
            
            # Build PDF
            doc.build(story)
            return buffer.getvalue()
            
        except Exception as e:
            raise RuntimeError(f"PDF report generation failed: {str(e)}")
    
    def _create_title_page(self, title: str, analysis: DetailedAnalysis) -> List[Any]:
        """Create the title page content."""
        story = []
        
        # Main title
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 0.5*inch))
        
        # Analysis timestamp
        timestamp_text = f"Analysis Date: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        story.append(Paragraph(timestamp_text, self.styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Summary statistics table
        summary_data = [
            ['Parameter', 'Value'],
            ['Total Waves Detected', str(analysis.wave_result.total_waves_detected)],
            ['Wave Types Found', ', '.join(analysis.wave_result.wave_types_detected)],
            ['P-waves', str(len(analysis.wave_result.p_waves))],
            ['S-waves', str(len(analysis.wave_result.s_waves))],
            ['Surface Waves', str(len(analysis.wave_result.surface_waves))],
        ]
        
        if analysis.best_magnitude_estimate:
            summary_data.append(['Best Magnitude Estimate', 
                               f"{analysis.best_magnitude_estimate.magnitude:.2f} ({analysis.best_magnitude_estimate.method})"])
        
        if analysis.epicenter_distance:
            summary_data.append(['Estimated Distance', f"{analysis.epicenter_distance:.1f} km"])
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Quality indicator
        if analysis.quality_metrics:
            quality_score = analysis.quality_metrics.analysis_quality_score
            quality_text = f"Analysis Quality Score: {quality_score:.2f}/1.00"
            if quality_score >= 0.8:
                quality_color = colors.green
                quality_desc = "Excellent"
            elif quality_score >= 0.6:
                quality_color = colors.orange
                quality_desc = "Good"
            else:
                quality_color = colors.red
                quality_desc = "Fair"
            
            quality_para = Paragraph(
                f'<font color="{quality_color}">{quality_text} ({quality_desc})</font>',
                self.styles['Normal']
            )
            story.append(quality_para)
        
        return story
    
    def _create_executive_summary(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create executive summary section."""
        story = []
        
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Generate summary text based on analysis results
        summary_parts = []
        
        # Wave detection summary
        total_waves = analysis.wave_result.total_waves_detected
        wave_types = analysis.wave_result.wave_types_detected
        summary_parts.append(f"This analysis detected {total_waves} seismic wave segments across {len(wave_types)} different wave types: {', '.join(wave_types)}.")
        
        # Timing summary
        if analysis.arrival_times.p_wave_arrival and analysis.arrival_times.s_wave_arrival:
            sp_diff = analysis.arrival_times.sp_time_difference
            summary_parts.append(f"P-wave arrival was detected at {analysis.arrival_times.p_wave_arrival:.2f} seconds, followed by S-wave arrival at {analysis.arrival_times.s_wave_arrival:.2f} seconds, giving an S-P time difference of {sp_diff:.2f} seconds.")
        
        # Distance estimation
        if analysis.epicenter_distance:
            summary_parts.append(f"Based on the S-P time difference, the estimated epicentral distance is approximately {analysis.epicenter_distance:.1f} kilometers.")
        
        # Magnitude summary
        if analysis.magnitude_estimates:
            best_mag = analysis.best_magnitude_estimate
            summary_parts.append(f"Magnitude analysis yielded {len(analysis.magnitude_estimates)} estimates, with the most reliable being {best_mag.magnitude:.2f} using the {best_mag.method} method (confidence: {best_mag.confidence:.2f}).")
        
        # Quality assessment
        if analysis.quality_metrics:
            quality = analysis.quality_metrics
            summary_parts.append(f"The analysis achieved a quality score of {quality.analysis_quality_score:.2f} with a signal-to-noise ratio of {quality.signal_to_noise_ratio:.1f} dB.")
        
        # Combine summary parts
        summary_text = " ".join(summary_parts)
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Key findings
        story.append(Paragraph("Key Findings:", self.styles['SubsectionHeader']))
        
        findings = []
        if len(analysis.wave_result.p_waves) > 0:
            findings.append(f"• {len(analysis.wave_result.p_waves)} P-wave segments identified")
        if len(analysis.wave_result.s_waves) > 0:
            findings.append(f"• {len(analysis.wave_result.s_waves)} S-wave segments identified")
        if len(analysis.wave_result.surface_waves) > 0:
            surface_types = set(w.wave_type for w in analysis.wave_result.surface_waves)
            findings.append(f"• {len(analysis.wave_result.surface_waves)} surface wave segments ({', '.join(surface_types)})")
        
        if analysis.epicenter_distance:
            findings.append(f"• Estimated epicentral distance: {analysis.epicenter_distance:.1f} km")
        
        if analysis.best_magnitude_estimate:
            findings.append(f"• Best magnitude estimate: {analysis.best_magnitude_estimate.magnitude:.2f} ({analysis.best_magnitude_estimate.method})")
        
        for finding in findings:
            story.append(Paragraph(finding, self.styles['Normal']))
        
        return story   
 
    def _create_wave_detection_section(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create wave detection results section."""
        story = []
        
        story.append(Paragraph("Wave Detection Results", self.styles['SectionHeader']))
        
        # Create table for wave detection summary
        wave_data = [
            ['Wave Type', 'Count', 'Avg Confidence', 'Avg Amplitude', 'Avg Frequency (Hz)']
        ]
        
        wave_types = [
            ('P-waves', analysis.wave_result.p_waves),
            ('S-waves', analysis.wave_result.s_waves),
            ('Surface waves', analysis.wave_result.surface_waves)
        ]
        
        for wave_name, waves in wave_types:
            if waves:
                avg_confidence = np.mean([w.confidence for w in waves])
                avg_amplitude = np.mean([w.peak_amplitude for w in waves])
                avg_frequency = np.mean([w.dominant_frequency for w in waves])
                
                wave_data.append([
                    wave_name,
                    str(len(waves)),
                    f"{avg_confidence:.3f}",
                    f"{avg_amplitude:.2e}",
                    f"{avg_frequency:.1f}"
                ])
            else:
                wave_data.append([wave_name, "0", "N/A", "N/A", "N/A"])
        
        wave_table = Table(wave_data, colWidths=[1.5*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
        wave_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(wave_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Detailed wave information
        for wave_name, waves in wave_types:
            if waves:
                story.append(Paragraph(f"{wave_name} Details:", self.styles['SubsectionHeader']))
                
                for i, wave in enumerate(waves[:5]):  # Limit to first 5 waves
                    wave_info = [
                        f"Segment {i+1}:",
                        f"  Arrival time: {wave.arrival_time:.3f} s",
                        f"  Duration: {wave.duration:.3f} s",
                        f"  Peak amplitude: {wave.peak_amplitude:.2e}",
                        f"  Dominant frequency: {wave.dominant_frequency:.1f} Hz",
                        f"  Confidence: {wave.confidence:.3f}"
                    ]
                    
                    for info in wave_info:
                        story.append(Paragraph(info, self.styles['DataStyle']))
                
                if len(waves) > 5:
                    story.append(Paragraph(f"... and {len(waves) - 5} more segments", self.styles['DataStyle']))
                
                story.append(Spacer(1, 0.1*inch))
        
        return story
    
    def _create_timing_analysis_section(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create timing analysis section."""
        story = []
        
        story.append(Paragraph("Timing Analysis", self.styles['SectionHeader']))
        
        # Arrival times table
        timing_data = [['Parameter', 'Time (seconds)', 'Notes']]
        
        if analysis.arrival_times.p_wave_arrival is not None:
            timing_data.append(['P-wave Arrival', f"{analysis.arrival_times.p_wave_arrival:.3f}", 'First P-wave detection'])
        
        if analysis.arrival_times.s_wave_arrival is not None:
            timing_data.append(['S-wave Arrival', f"{analysis.arrival_times.s_wave_arrival:.3f}", 'First S-wave detection'])
        
        if analysis.arrival_times.surface_wave_arrival is not None:
            timing_data.append(['Surface Wave Arrival', f"{analysis.arrival_times.surface_wave_arrival:.3f}", 'First surface wave detection'])
        
        if analysis.arrival_times.sp_time_difference is not None:
            timing_data.append(['S-P Time Difference', f"{analysis.arrival_times.sp_time_difference:.3f}", 'Used for distance estimation'])
        
        if analysis.epicenter_distance is not None:
            timing_data.append(['Estimated Distance', f"{analysis.epicenter_distance:.1f} km", 'Based on S-P time'])
        
        timing_table = Table(timing_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        timing_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(timing_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Distance estimation explanation
        if analysis.arrival_times.sp_time_difference and analysis.epicenter_distance:
            explanation = f"""
            The epicentral distance is estimated using the S-P time difference of {analysis.arrival_times.sp_time_difference:.3f} seconds.
            This calculation assumes average crustal velocities of 6.0 km/s for P-waves and 3.5 km/s for S-waves.
            The estimated distance of {analysis.epicenter_distance:.1f} km should be considered approximate.
            """
            story.append(Paragraph(explanation, self.styles['Normal']))
        
        return story
    
    def _create_magnitude_section(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create magnitude estimation section."""
        story = []
        
        story.append(Paragraph("Magnitude Estimation", self.styles['SectionHeader']))
        
        if not analysis.magnitude_estimates:
            story.append(Paragraph("No magnitude estimates available.", self.styles['Normal']))
            return story
        
        # Magnitude estimates table
        mag_data = [['Method', 'Magnitude', 'Confidence', 'Wave Type Used', 'Notes']]
        
        for estimate in analysis.magnitude_estimates:
            notes = "Best estimate" if estimate == analysis.best_magnitude_estimate else ""
            mag_data.append([
                estimate.method,
                f"{estimate.magnitude:.2f}",
                f"{estimate.confidence:.3f}",
                estimate.wave_type_used,
                notes
            ])
        
        mag_table = Table(mag_data, colWidths=[1*inch, 1*inch, 1*inch, 1.2*inch, 1.8*inch])
        mag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkorange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(mag_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Best magnitude highlight
        if analysis.best_magnitude_estimate:
            best = analysis.best_magnitude_estimate
            best_text = f"""
            <b>Recommended Magnitude:</b> {best.magnitude:.2f} ({best.method})<br/>
            <b>Confidence:</b> {best.confidence:.3f}<br/>
            <b>Based on:</b> {best.wave_type_used} wave analysis
            """
            story.append(Paragraph(best_text, self.styles['Normal']))
        
        return story
    
    def _create_frequency_analysis_section(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create frequency analysis section."""
        story = []
        
        story.append(Paragraph("Frequency Analysis", self.styles['SectionHeader']))
        
        if not analysis.frequency_analysis:
            story.append(Paragraph("No frequency analysis data available.", self.styles['Normal']))
            return story
        
        # Frequency analysis table
        freq_data = [['Wave Type', 'Dominant Freq (Hz)', 'Spectral Centroid (Hz)', 'Bandwidth (Hz)', 'Freq Range (Hz)']]
        
        for wave_type, freq_info in analysis.frequency_analysis.items():
            freq_range_str = f"{freq_info.frequency_range[0]:.1f} - {freq_info.frequency_range[1]:.1f}"
            freq_data.append([
                wave_type,
                f"{freq_info.dominant_frequency:.2f}",
                f"{freq_info.spectral_centroid:.2f}",
                f"{freq_info.bandwidth:.2f}",
                freq_range_str
            ])
        
        freq_table = Table(freq_data, colWidths=[1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        freq_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightpink),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(freq_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Frequency analysis interpretation
        interpretation = """
        <b>Frequency Analysis Interpretation:</b><br/>
        • P-waves typically show higher frequencies (5-20 Hz)<br/>
        • S-waves generally have intermediate frequencies (2-10 Hz)<br/>
        • Surface waves usually exhibit lower frequencies (0.1-5 Hz)<br/>
        • Spectral centroid indicates the "center of mass" of the frequency spectrum<br/>
        • Bandwidth shows the spread of significant frequency content
        """
        story.append(Paragraph(interpretation, self.styles['Normal']))
        
        return story
    
    def _create_quality_metrics_section(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create quality metrics section."""
        story = []
        
        story.append(Paragraph("Quality Assessment", self.styles['SectionHeader']))
        
        if not analysis.quality_metrics:
            story.append(Paragraph("No quality metrics available.", self.styles['Normal']))
            return story
        
        quality = analysis.quality_metrics
        
        # Quality metrics table
        quality_data = [
            ['Metric', 'Value', 'Assessment'],
            ['Signal-to-Noise Ratio', f"{quality.signal_to_noise_ratio:.1f} dB", self._assess_snr(quality.signal_to_noise_ratio)],
            ['Detection Confidence', f"{quality.detection_confidence:.3f}", self._assess_confidence(quality.detection_confidence)],
            ['Analysis Quality Score', f"{quality.analysis_quality_score:.3f}", self._assess_quality_score(quality.analysis_quality_score)],
            ['Data Completeness', f"{quality.data_completeness:.1%}", self._assess_completeness(quality.data_completeness)]
        ]
        
        quality_table = Table(quality_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(quality_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Processing warnings
        if quality.processing_warnings:
            story.append(Paragraph("Processing Warnings:", self.styles['SubsectionHeader']))
            for warning in quality.processing_warnings:
                story.append(Paragraph(f"• {warning}", self.styles['Warning']))
        
        return story
    
    def _assess_snr(self, snr: float) -> str:
        """Assess signal-to-noise ratio quality."""
        if snr >= 20:
            return "Excellent"
        elif snr >= 10:
            return "Good"
        elif snr >= 5:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_confidence(self, confidence: float) -> str:
        """Assess detection confidence quality."""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_quality_score(self, score: float) -> str:
        """Assess overall quality score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _assess_completeness(self, completeness: float) -> str:
        """Assess data completeness."""
        if completeness >= 0.95:
            return "Complete"
        elif completeness >= 0.8:
            return "Nearly Complete"
        elif completeness >= 0.6:
            return "Mostly Complete"
        else:
            return "Incomplete"    

    def _create_visualization_section(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create visualization section with wave plots."""
        story = []
        
        story.append(Paragraph("Wave Visualizations", self.styles['SectionHeader']))
        
        if not MATPLOTLIB_AVAILABLE:
            story.append(Paragraph("Matplotlib not available - visualizations skipped.", self.styles['Warning']))
            return story
        
        try:
            # Create time series plot
            time_plot_img = self._create_time_series_plot(analysis)
            if time_plot_img:
                story.append(Paragraph("Time Series Analysis", self.styles['SubsectionHeader']))
                story.append(time_plot_img)
                story.append(Spacer(1, 0.2*inch))
            
            # Create frequency spectrum plot
            freq_plot_img = self._create_frequency_spectrum_plot(analysis)
            if freq_plot_img:
                story.append(Paragraph("Frequency Spectrum Analysis", self.styles['SubsectionHeader']))
                story.append(freq_plot_img)
                story.append(Spacer(1, 0.2*inch))
            
            # Create wave arrival timeline
            timeline_img = self._create_arrival_timeline(analysis)
            if timeline_img:
                story.append(Paragraph("Wave Arrival Timeline", self.styles['SubsectionHeader']))
                story.append(timeline_img)
                
        except Exception as e:
            story.append(Paragraph(f"Error generating visualizations: {str(e)}", self.styles['Warning']))
        
        return story
    
    def _create_time_series_plot(self, analysis: DetailedAnalysis) -> Optional[Image]:
        """Create time series plot of wave data."""
        try:
            fig, axes = plt.subplots(3, 1, figsize=(10, 8))
            fig.suptitle('Wave Analysis - Time Series', fontsize=16, fontweight='bold')
            
            # Original data plot
            original_data = analysis.wave_result.original_data
            sampling_rate = analysis.wave_result.sampling_rate
            time_axis = np.arange(len(original_data)) / sampling_rate
            
            axes[0].plot(time_axis, original_data, 'k-', linewidth=0.5, alpha=0.7)
            axes[0].set_title('Original Seismic Data')
            axes[0].set_ylabel('Amplitude')
            axes[0].grid(True, alpha=0.3)
            
            # P-wave segments
            if analysis.wave_result.p_waves:
                for wave in analysis.wave_result.p_waves:
                    wave_time = np.arange(len(wave.data)) / wave.sampling_rate + wave.start_time
                    axes[1].plot(wave_time, wave.data, 'b-', linewidth=1, alpha=0.8)
                axes[1].set_title('P-wave Segments')
                axes[1].set_ylabel('Amplitude')
                axes[1].grid(True, alpha=0.3)
            
            # S-wave segments
            if analysis.wave_result.s_waves:
                for wave in analysis.wave_result.s_waves:
                    wave_time = np.arange(len(wave.data)) / wave.sampling_rate + wave.start_time
                    axes[2].plot(wave_time, wave.data, 'r-', linewidth=1, alpha=0.8)
                axes[2].set_title('S-wave Segments')
                axes[2].set_ylabel('Amplitude')
                axes[2].set_xlabel('Time (seconds)')
                axes[2].grid(True, alpha=0.3)
            
            # Mark arrival times
            for i, ax in enumerate(axes):
                if i == 0:  # Original data
                    if analysis.arrival_times.p_wave_arrival:
                        ax.axvline(analysis.arrival_times.p_wave_arrival, color='blue', 
                                 linestyle='--', alpha=0.7, label='P-wave arrival')
                    if analysis.arrival_times.s_wave_arrival:
                        ax.axvline(analysis.arrival_times.s_wave_arrival, color='red', 
                                 linestyle='--', alpha=0.7, label='S-wave arrival')
                    if analysis.arrival_times.surface_wave_arrival:
                        ax.axvline(analysis.arrival_times.surface_wave_arrival, color='green', 
                                 linestyle='--', alpha=0.7, label='Surface wave arrival')
                    ax.legend()
            
            plt.tight_layout()
            
            # Save to buffer and create Image
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            img_buffer.seek(0)
            return Image(img_buffer, width=6*inch, height=4.8*inch)
            
        except Exception as e:
            print(f"Error creating time series plot: {e}")
            return None
    
    def _create_frequency_spectrum_plot(self, analysis: DetailedAnalysis) -> Optional[Image]:
        """Create frequency spectrum plot."""
        try:
            if not analysis.frequency_analysis:
                return None
            
            fig, axes = plt.subplots(len(analysis.frequency_analysis), 1, 
                                   figsize=(10, 2*len(analysis.frequency_analysis)))
            if len(analysis.frequency_analysis) == 1:
                axes = [axes]
            
            fig.suptitle('Frequency Spectrum Analysis', fontsize=16, fontweight='bold')
            
            colors_map = {'P': 'blue', 'S': 'red', 'Love': 'green', 'Rayleigh': 'orange'}
            
            for i, (wave_type, freq_data) in enumerate(analysis.frequency_analysis.items()):
                color = colors_map.get(wave_type, 'black')
                
                axes[i].plot(freq_data.frequencies, freq_data.power_spectrum, 
                           color=color, linewidth=1.5)
                axes[i].axvline(freq_data.dominant_frequency, color=color, 
                              linestyle='--', alpha=0.7, label=f'Dominant: {freq_data.dominant_frequency:.1f} Hz')
                axes[i].axvline(freq_data.spectral_centroid, color=color, 
                              linestyle=':', alpha=0.7, label=f'Centroid: {freq_data.spectral_centroid:.1f} Hz')
                
                axes[i].set_title(f'{wave_type}-wave Frequency Spectrum')
                axes[i].set_ylabel('Power')
                axes[i].set_xlim(0, min(50, freq_data.frequencies.max()))
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
            
            axes[-1].set_xlabel('Frequency (Hz)')
            plt.tight_layout()
            
            # Save to buffer and create Image
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            img_buffer.seek(0)
            return Image(img_buffer, width=6*inch, height=2*len(analysis.frequency_analysis)*inch)
            
        except Exception as e:
            print(f"Error creating frequency spectrum plot: {e}")
            return None
    
    def _create_arrival_timeline(self, analysis: DetailedAnalysis) -> Optional[Image]:
        """Create wave arrival timeline visualization."""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            
            arrivals = []
            colors = []
            labels = []
            
            if analysis.arrival_times.p_wave_arrival is not None:
                arrivals.append(analysis.arrival_times.p_wave_arrival)
                colors.append('blue')
                labels.append('P-wave')
            
            if analysis.arrival_times.s_wave_arrival is not None:
                arrivals.append(analysis.arrival_times.s_wave_arrival)
                colors.append('red')
                labels.append('S-wave')
            
            if analysis.arrival_times.surface_wave_arrival is not None:
                arrivals.append(analysis.arrival_times.surface_wave_arrival)
                colors.append('green')
                labels.append('Surface wave')
            
            if not arrivals:
                return None
            
            # Create timeline
            y_pos = [1] * len(arrivals)
            ax.scatter(arrivals, y_pos, c=colors, s=200, alpha=0.7, edgecolors='black')
            
            # Add labels
            for i, (arrival, label) in enumerate(zip(arrivals, labels)):
                ax.annotate(f'{label}\n{arrival:.2f}s', 
                          (arrival, 1), 
                          xytext=(0, 20), 
                          textcoords='offset points',
                          ha='center',
                          fontsize=10,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
            
            # Add S-P time difference if available
            if (analysis.arrival_times.p_wave_arrival is not None and 
                analysis.arrival_times.s_wave_arrival is not None):
                p_time = analysis.arrival_times.p_wave_arrival
                s_time = analysis.arrival_times.s_wave_arrival
                ax.annotate('', xy=(s_time, 0.8), xytext=(p_time, 0.8),
                          arrowprops=dict(arrowstyle='<->', color='black', lw=2))
                ax.text((p_time + s_time) / 2, 0.7, 
                       f'S-P: {analysis.arrival_times.sp_time_difference:.2f}s',
                       ha='center', fontsize=10, fontweight='bold')
            
            ax.set_ylim(0.5, 1.5)
            ax.set_xlabel('Time (seconds)')
            ax.set_title('Wave Arrival Timeline', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yticks([])
            
            plt.tight_layout()
            
            # Save to buffer and create Image
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            img_buffer.seek(0)
            return Image(img_buffer, width=6*inch, height=2.4*inch)
            
        except Exception as e:
            print(f"Error creating arrival timeline: {e}")
            return None
    
    def _create_raw_data_section(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create raw data tables section."""
        story = []
        
        story.append(Paragraph("Raw Data Tables", self.styles['SectionHeader']))
        
        # Individual wave segments table
        story.append(Paragraph("Individual Wave Segments", self.styles['SubsectionHeader']))
        
        segment_data = [['Type', 'ID', 'Start (s)', 'End (s)', 'Duration (s)', 
                        'Arrival (s)', 'Peak Amp', 'Dom Freq (Hz)', 'Confidence']]
        
        all_waves = []
        for wave_type in ['P', 'S', 'Love', 'Rayleigh']:
            waves = analysis.wave_result.get_waves_by_type(wave_type)
            for i, wave in enumerate(waves):
                all_waves.append((wave, i))
        
        # Sort by arrival time
        all_waves.sort(key=lambda x: x[0].arrival_time)
        
        for wave, segment_id in all_waves:
            segment_data.append([
                wave.wave_type,
                str(segment_id),
                f"{wave.start_time:.3f}",
                f"{wave.end_time:.3f}",
                f"{wave.duration:.3f}",
                f"{wave.arrival_time:.3f}",
                f"{wave.peak_amplitude:.2e}",
                f"{wave.dominant_frequency:.1f}",
                f"{wave.confidence:.3f}"
            ])
        
        segment_table = Table(segment_data, colWidths=[0.6*inch, 0.4*inch, 0.7*inch, 0.7*inch, 
                                                     0.7*inch, 0.7*inch, 0.8*inch, 0.8*inch, 0.7*inch])
        segment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(segment_table)
        
        return story
    
    def _create_appendix(self, analysis: DetailedAnalysis) -> List[Any]:
        """Create appendix with technical details."""
        story = []
        
        story.append(Paragraph("Appendix", self.styles['SectionHeader']))
        
        # Technical parameters
        story.append(Paragraph("Technical Parameters", self.styles['SubsectionHeader']))
        
        tech_info = [
            f"• Sampling Rate: {analysis.wave_result.sampling_rate} Hz",
            f"• Original Data Length: {len(analysis.wave_result.original_data)} samples",
            f"• Analysis Duration: {len(analysis.wave_result.original_data) / analysis.wave_result.sampling_rate:.1f} seconds",
            f"• Processing Timestamp: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]
        
        for info in tech_info:
            story.append(Paragraph(info, self.styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Processing metadata
        if analysis.processing_metadata:
            story.append(Paragraph("Processing Metadata", self.styles['SubsectionHeader']))
            for key, value in analysis.processing_metadata.items():
                story.append(Paragraph(f"• {key}: {value}", self.styles['DataStyle']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Methodology notes
        story.append(Paragraph("Methodology Notes", self.styles['SubsectionHeader']))
        
        methodology = """
        This analysis was performed using automated wave detection algorithms:
        
        • P-wave detection: STA/LTA algorithm with characteristic function analysis
        • S-wave detection: Polarization analysis and particle motion evaluation
        • Surface wave detection: Frequency-time analysis for Love and Rayleigh waves
        • Magnitude estimation: Multiple methods including ML, Mb, and Ms calculations
        • Quality assessment: Signal-to-noise ratio and confidence scoring
        
        Results should be validated by experienced seismologists for critical applications.
        """
        
        story.append(Paragraph(methodology, self.styles['Normal']))
        
        return story


class PDFReportError(Exception):
    """Custom exception for PDF report generation errors."""
    pass