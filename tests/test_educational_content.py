"""
Tests for educational content accuracy and completeness
"""

import unittest
import json
import os
from pathlib import Path


class TestEducationalContent(unittest.TestCase):
    """Test educational content for wave analysis features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.educational_content = {
            "waveTypes": {
                "p-wave": {
                    "title": "P-Waves (Primary Waves)",
                    "description": "P-waves are the fastest seismic waves and arrive first at recording stations.",
                    "characteristics": [
                        "Speed: 6-8 km/s in Earth's crust",
                        "Motion: Compressional (push-pull)",
                        "Can travel through solids and liquids",
                        "Smaller amplitude than S-waves",
                        "First to be detected on seismograms"
                    ],
                    "detection": "Detected using STA/LTA algorithm and characteristic function analysis",
                    "importance": "Critical for initial earthquake detection and location determination",
                    "examples": [
                        "Typical frequency: 8-15 Hz",
                        "Duration: Usually 1-1.5 seconds",
                        "Amplitude: 0.1-1.0 relative units"
                    ]
                },
                "s-wave": {
                    "title": "S-Waves (Secondary Waves)",
                    "description": "S-waves are slower than P-waves but cause more ground shaking.",
                    "characteristics": [
                        "Speed: 3-4 km/s in Earth's crust",
                        "Motion: Shear (side-to-side)",
                        "Cannot travel through liquids",
                        "Larger amplitude than P-waves",
                        "Arrive after P-waves"
                    ],
                    "detection": "Identified through polarization analysis and particle motion studies",
                    "importance": "Essential for magnitude estimation and damage assessment",
                    "examples": [
                        "Typical frequency: 1.5-5 Hz",
                        "Duration: Usually 2-6 seconds",
                        "Amplitude: 0.5-2.0 relative units"
                    ]
                },
                "surface": {
                    "title": "Surface Waves",
                    "description": "Surface waves travel along Earth's surface and cause the most damage.",
                    "characteristics": [
                        "Speed: 1.5-2.8 km/s (slower than body waves)",
                        "Two types: Love waves and Rayleigh waves",
                        "Love waves: horizontal shearing motion",
                        "Rayleigh waves: elliptical rolling motion",
                        "Largest amplitude and longest duration"
                    ],
                    "detection": "Identified through frequency-time analysis and group velocity calculations",
                    "importance": "Dominant in earthquake damage, used for surface wave magnitude (Ms)",
                    "examples": [
                        "Typical frequency: 0.05-1 Hz",
                        "Duration: 10-60 seconds",
                        "Amplitude: 1.0-5.0 relative units"
                    ]
                }
            },
            "analysisTerms": {
                "arrival-time": {
                    "title": "Arrival Time",
                    "definition": "The time when a seismic wave first arrives at a recording station",
                    "calculation": "Determined using onset detection algorithms and cross-correlation methods",
                    "importance": "Used to calculate epicentral distance and earthquake location",
                    "units": "Seconds from earthquake origin time"
                },
                "sp-time": {
                    "title": "S-P Time Difference",
                    "definition": "Time difference between S-wave and P-wave arrivals",
                    "calculation": "S-wave arrival time minus P-wave arrival time",
                    "importance": "Directly related to distance from earthquake epicenter",
                    "formula": "Distance ≈ S-P time × 8 km/s (approximate)"
                },
                "magnitude": {
                    "title": "Earthquake Magnitude",
                    "definition": "Logarithmic measure of earthquake energy release",
                    "types": [
                        "ML (Local): Based on maximum amplitude",
                        "Mb (Body wave): Based on P-wave amplitude and period",
                        "Ms (Surface wave): Based on surface wave amplitude",
                        "Mw (Moment): Based on seismic moment"
                    ],
                    "scale": "Each unit increase represents 10x more amplitude, 32x more energy"
                },
                "snr": {
                    "title": "Signal-to-Noise Ratio (SNR)",
                    "definition": "Ratio of signal power to background noise power",
                    "calculation": "SNR = Signal Power / Noise Power (often in dB)",
                    "importance": "Indicates data quality and detection reliability",
                    "threshold": "Minimum SNR of 2-3 typically required for reliable detection"
                }
            }
        }

    def test_wave_type_content_completeness(self):
        """Test that all wave types have complete educational content"""
        required_fields = ['title', 'description', 'characteristics', 'detection', 'importance', 'examples']
        
        for wave_type, content in self.educational_content['waveTypes'].items():
            with self.subTest(wave_type=wave_type):
                for field in required_fields:
                    self.assertIn(field, content, f"Missing {field} for {wave_type}")
                    self.assertTrue(content[field], f"Empty {field} for {wave_type}")
                
                # Test characteristics is a list with content
                self.assertIsInstance(content['characteristics'], list)
                self.assertGreater(len(content['characteristics']), 0)
                
                # Test examples is a list with content
                self.assertIsInstance(content['examples'], list)
                self.assertGreater(len(content['examples']), 0)

    def test_p_wave_content_accuracy(self):
        """Test P-wave educational content for scientific accuracy"""
        p_wave = self.educational_content['waveTypes']['p-wave']
        
        # Test title
        self.assertEqual(p_wave['title'], "P-Waves (Primary Waves)")
        
        # Test speed information is present and reasonable
        speed_info = [char for char in p_wave['characteristics'] if 'Speed:' in char]
        self.assertEqual(len(speed_info), 1)
        self.assertIn('6-8 km/s', speed_info[0])
        
        # Test motion description
        motion_info = [char for char in p_wave['characteristics'] if 'Motion:' in char]
        self.assertEqual(len(motion_info), 1)
        self.assertIn('Compressional', motion_info[0])
        
        # Test frequency range in examples
        freq_examples = [ex for ex in p_wave['examples'] if 'frequency:' in ex.lower()]
        self.assertGreater(len(freq_examples), 0)
        self.assertIn('8-15 Hz', freq_examples[0])

    def test_s_wave_content_accuracy(self):
        """Test S-wave educational content for scientific accuracy"""
        s_wave = self.educational_content['waveTypes']['s-wave']
        
        # Test title
        self.assertEqual(s_wave['title'], "S-Waves (Secondary Waves)")
        
        # Test speed information
        speed_info = [char for char in s_wave['characteristics'] if 'Speed:' in char]
        self.assertEqual(len(speed_info), 1)
        self.assertIn('3-4 km/s', speed_info[0])
        
        # Test liquid propagation characteristic
        liquid_info = [char for char in s_wave['characteristics'] if 'liquid' in char.lower()]
        self.assertGreater(len(liquid_info), 0)
        self.assertIn('Cannot travel through liquids', liquid_info[0])
        
        # Test frequency range
        freq_examples = [ex for ex in s_wave['examples'] if 'frequency:' in ex.lower()]
        self.assertGreater(len(freq_examples), 0)
        self.assertIn('1.5-5 Hz', freq_examples[0])

    def test_surface_wave_content_accuracy(self):
        """Test surface wave educational content for scientific accuracy"""
        surface_wave = self.educational_content['waveTypes']['surface']
        
        # Test title
        self.assertEqual(surface_wave['title'], "Surface Waves")
        
        # Test speed information
        speed_info = [char for char in surface_wave['characteristics'] if 'Speed:' in char]
        self.assertEqual(len(speed_info), 1)
        self.assertIn('1.5-2.8 km/s', speed_info[0])
        
        # Test wave types mentioned
        love_info = [char for char in surface_wave['characteristics'] if 'Love' in char]
        rayleigh_info = [char for char in surface_wave['characteristics'] if 'Rayleigh' in char]
        self.assertGreater(len(love_info), 0)
        self.assertGreater(len(rayleigh_info), 0)
        
        # Test frequency range
        freq_examples = [ex for ex in surface_wave['examples'] if 'frequency:' in ex.lower()]
        self.assertGreater(len(freq_examples), 0)
        self.assertIn('0.05-1 Hz', freq_examples[0])

    def test_wave_speed_relationships(self):
        """Test that wave speeds follow correct physical relationships"""
        p_speed = self.extract_speed_range('p-wave')
        s_speed = self.extract_speed_range('s-wave')
        surface_speed = self.extract_speed_range('surface')
        
        # P-waves should be fastest
        self.assertGreater(p_speed[0], s_speed[1], "P-wave minimum should be > S-wave maximum")
        self.assertGreater(s_speed[0], surface_speed[1], "S-wave minimum should be > Surface wave maximum")

    def extract_speed_range(self, wave_type):
        """Extract speed range from wave characteristics"""
        characteristics = self.educational_content['waveTypes'][wave_type]['characteristics']
        speed_info = [char for char in characteristics if 'Speed:' in char][0]
        
        # Extract numbers from speed info (e.g., "6-8 km/s" -> [6, 8])
        import re
        numbers = re.findall(r'\d+\.?\d*', speed_info)
        return [float(n) for n in numbers[:2]]

    def test_analysis_terms_completeness(self):
        """Test that analysis terms have complete definitions"""
        required_fields = ['title', 'definition']
        
        for term, content in self.educational_content['analysisTerms'].items():
            with self.subTest(term=term):
                for field in required_fields:
                    self.assertIn(field, content, f"Missing {field} for {term}")
                    self.assertTrue(content[field], f"Empty {field} for {term}")

    def test_magnitude_scale_accuracy(self):
        """Test magnitude scale information for accuracy"""
        magnitude_info = self.educational_content['analysisTerms']['magnitude']
        
        # Test that all major magnitude types are mentioned
        magnitude_types = magnitude_info['types']
        type_codes = [t.split()[0] for t in magnitude_types]
        
        expected_types = ['ML', 'Mb', 'Ms', 'Mw']
        for expected in expected_types:
            self.assertIn(expected, type_codes, f"Missing magnitude type {expected}")
        
        # Test scale description
        self.assertIn('10x more amplitude', magnitude_info['scale'])
        self.assertIn('32x more energy', magnitude_info['scale'])

    def test_snr_threshold_accuracy(self):
        """Test SNR threshold information for accuracy"""
        snr_info = self.educational_content['analysisTerms']['snr']
        
        # Test threshold information
        self.assertIn('threshold', snr_info)
        self.assertIn('2-3', snr_info['threshold'])
        
        # Test calculation description
        self.assertIn('Signal Power / Noise Power', snr_info['calculation'])

    def test_sp_time_formula_accuracy(self):
        """Test S-P time formula for accuracy"""
        sp_info = self.educational_content['analysisTerms']['sp-time']
        
        # Test formula presence
        self.assertIn('formula', sp_info)
        self.assertIn('8 km/s', sp_info['formula'])
        self.assertIn('Distance', sp_info['formula'])

    def test_frequency_ranges_consistency(self):
        """Test that frequency ranges are consistent across wave types"""
        p_freq = self.extract_frequency_range('p-wave')
        s_freq = self.extract_frequency_range('s-wave')
        surface_freq = self.extract_frequency_range('surface')
        
        # P-waves should have higher frequencies than S-waves
        self.assertGreater(p_freq[0], s_freq[1], "P-wave frequencies should be higher than S-wave")
        
        # S-waves should have higher frequencies than surface waves
        self.assertGreater(s_freq[0], surface_freq[1], "S-wave frequencies should be higher than surface waves")

    def extract_frequency_range(self, wave_type):
        """Extract frequency range from wave examples"""
        examples = self.educational_content['waveTypes'][wave_type]['examples']
        freq_example = [ex for ex in examples if 'frequency:' in ex.lower()][0]
        
        # Extract numbers from frequency info
        import re
        numbers = re.findall(r'[\d.]+', freq_example)
        return [float(n) for n in numbers[:2]]

    def test_duration_relationships(self):
        """Test that wave durations follow expected relationships"""
        p_duration = self.extract_duration_range('p-wave')
        s_duration = self.extract_duration_range('s-wave')
        surface_duration = self.extract_duration_range('surface')
        
        # Surface waves should have longest duration
        self.assertGreater(surface_duration[0], s_duration[1], "Surface waves should be longer than S-waves")
        self.assertGreater(s_duration[0], p_duration[1], "S-waves should be longer than P-waves")

    def extract_duration_range(self, wave_type):
        """Extract duration range from wave examples"""
        examples = self.educational_content['waveTypes'][wave_type]['examples']
        duration_example = [ex for ex in examples if 'Duration:' in ex][0]
        
        # Extract numbers from duration info
        import re
        numbers = re.findall(r'\d+', duration_example)
        return [int(n) for n in numbers[:2]]

    def test_amplitude_relationships(self):
        """Test that amplitude relationships are correct"""
        p_amp = self.extract_amplitude_range('p-wave')
        s_amp = self.extract_amplitude_range('s-wave')
        surface_amp = self.extract_amplitude_range('surface')
        
        # Surface waves should have largest amplitudes
        self.assertGreater(surface_amp[1], s_amp[1], "Surface waves should have larger amplitude than S-waves")
        self.assertGreater(s_amp[1], p_amp[1], "S-waves should have larger amplitude than P-waves")

    def extract_amplitude_range(self, wave_type):
        """Extract amplitude range from wave examples"""
        examples = self.educational_content['waveTypes'][wave_type]['examples']
        amp_example = [ex for ex in examples if 'Amplitude:' in ex][0]
        
        # Extract numbers from amplitude info
        import re
        numbers = re.findall(r'[\d.]+', amp_example)
        return [float(n) for n in numbers[:2]]

    def test_educational_content_structure(self):
        """Test the overall structure of educational content"""
        # Test main sections exist
        self.assertIn('waveTypes', self.educational_content)
        self.assertIn('analysisTerms', self.educational_content)
        
        # Test wave types section has all expected types
        expected_wave_types = ['p-wave', 's-wave', 'surface']
        for wave_type in expected_wave_types:
            self.assertIn(wave_type, self.educational_content['waveTypes'])
        
        # Test analysis terms section has key terms
        expected_terms = ['arrival-time', 'sp-time', 'magnitude', 'snr']
        for term in expected_terms:
            self.assertIn(term, self.educational_content['analysisTerms'])

    def test_content_language_appropriateness(self):
        """Test that content uses appropriate scientific language"""
        # Test for technical accuracy in descriptions
        p_wave = self.educational_content['waveTypes']['p-wave']
        self.assertIn('compressional', p_wave['description'].lower() + ' '.join(p_wave['characteristics']).lower())
        
        s_wave = self.educational_content['waveTypes']['s-wave']
        self.assertIn('shear', s_wave['description'].lower() + ' '.join(s_wave['characteristics']).lower())
        
        # Test for educational appropriateness (not too technical)
        for wave_type, content in self.educational_content['waveTypes'].items():
            description_length = len(content['description'].split())
            self.assertLess(description_length, 50, f"{wave_type} description too long for educational use")
            self.assertGreater(description_length, 5, f"{wave_type} description too short")

    def test_tooltip_content_validation(self):
        """Test content suitable for tooltips"""
        for wave_type, content in self.educational_content['waveTypes'].items():
            with self.subTest(wave_type=wave_type):
                # Test that examples are concise enough for tooltips
                for example in content['examples']:
                    self.assertLess(len(example), 100, f"Example too long for tooltip: {example}")
                
                # Test that characteristics are bite-sized
                for char in content['characteristics']:
                    self.assertLess(len(char), 80, f"Characteristic too long for tooltip: {char}")

    def test_guided_tour_content(self):
        """Test content suitable for guided tour"""
        tour_steps = [
            "This dashboard helps you analyze seismic waves from earthquake data",
            "Choose which types of seismic waves to analyze",
            "Adjust these settings to fine-tune the wave detection algorithms",
            "This chart displays your seismic data and detected wave arrivals",
            "View detailed information about detected waves",
            "Learn about different wave types and analysis concepts",
            "Use these buttons to start analysis, export results, or generate reports"
        ]
        
        for step in tour_steps:
            # Test step length is appropriate
            self.assertLess(len(step), 200, f"Tour step too long: {step}")
            self.assertGreater(len(step), 20, f"Tour step too short: {step}")
            
            # Test step is informative
            self.assertTrue(any(word in step.lower() for word in ['analyze', 'wave', 'data', 'detect', 'result']),
                          f"Tour step not informative enough: {step}")


class TestEducationalSystemIntegration(unittest.TestCase):
    """Test integration of educational system with wave analysis dashboard"""
    
    def test_javascript_file_exists(self):
        """Test that educational system JavaScript file exists"""
        js_file = Path('static/js/educational_system.js')
        self.assertTrue(js_file.exists(), "Educational system JavaScript file not found")
    
    def test_javascript_file_content(self):
        """Test that JavaScript file contains required classes and methods"""
        js_file = Path('static/js/educational_system.js')
        if js_file.exists():
            content = js_file.read_text()
            
            # Test for main class
            self.assertIn('class EducationalSystem', content)
            
            # Test for key methods
            required_methods = [
                'setupTooltips',
                'setupGuidedTour',
                'showTooltip',
                'hideTooltip',
                'startGuidedTour',
                'validateContent'
            ]
            
            for method in required_methods:
                self.assertIn(method, content, f"Missing method: {method}")
    
    def test_template_integration(self):
        """Test that wave analysis dashboard template includes educational system"""
        template_file = Path('templates/wave_analysis_dashboard.html')
        if template_file.exists():
            try:
                content = template_file.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                content = template_file.read_text(encoding='latin-1')
            
            # Test for educational system script inclusion
            self.assertIn('educational_system.js', content)
            
            # Test for help indicators in template
            self.assertTrue('help-indicator' in content.lower() or 'educational' in content.lower())


if __name__ == '__main__':
    unittest.main()