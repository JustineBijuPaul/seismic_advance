#!/usr/bin/env python3
"""
Frontend tests for wave analysis dashboard functionality
Tests the dashboard interface, controls, and JavaScript interactions
"""

import unittest
import os
import sys

# Add the parent directory to the path to import the Flask app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import Selenium for browser tests, but make it optional
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    import time
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

@unittest.skipUnless(SELENIUM_AVAILABLE, "Selenium not available")
class WaveAnalysisDashboardTests(unittest.TestCase):
    """Test suite for wave analysis dashboard frontend functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment"""
        if not SELENIUM_AVAILABLE:
            raise unittest.SkipTest("Selenium not available")
            
        # Configure Chrome options for headless testing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            cls.driver = webdriver.Chrome(options=chrome_options)
            cls.driver.implicitly_wait(10)
            cls.wait = WebDriverWait(cls.driver, 10)
            
            # Start Flask app in test mode (this would need to be running)
            cls.base_url = "http://localhost:5000"
            
        except Exception as e:
            print(f"Failed to initialize WebDriver: {e}")
            print("Note: This test requires Chrome/Chromium and chromedriver to be installed")
            raise unittest.SkipTest("WebDriver not available")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests"""
        if hasattr(cls, 'driver'):
            cls.driver.quit()
    
    def setUp(self):
        """Set up each test"""
        try:
            self.driver.get(f"{self.base_url}/wave_analysis_dashboard")
            # Wait for page to load
            self.wait.until(EC.presence_of_element_located((By.CLASS_NAME, "dashboard-container")))
        except TimeoutException:
            raise unittest.SkipTest("Dashboard page not accessible - Flask app may not be running")
    
    def test_dashboard_page_loads(self):
        """Test that the dashboard page loads correctly"""
        # Check page title
        self.assertIn("Wave Analysis Dashboard", self.driver.title)
        
        # Check main dashboard elements are present
        dashboard_container = self.driver.find_element(By.CLASS_NAME, "dashboard-container")
        self.assertTrue(dashboard_container.is_displayed())
        
        # Check dashboard header
        dashboard_header = self.driver.find_element(By.CLASS_NAME, "dashboard-header")
        self.assertTrue(dashboard_header.is_displayed())
        
        # Check dashboard title
        title_element = self.driver.find_element(By.CLASS_NAME, "dashboard-title")
        self.assertEqual(title_element.text, "Wave Analysis Dashboard")
    
    def test_wave_type_selection_controls(self):
        """Test wave type selection functionality"""
        # Find all wave type buttons
        wave_type_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".wave-type-btn[data-wave-type]")
        self.assertEqual(len(wave_type_buttons), 4)  # all, p-wave, s-wave, surface
        
        # Check initial state - "All Waves" should be active
        all_waves_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-wave-type='all']")
        self.assertIn("active", all_waves_btn.get_attribute("class"))
        
        # Test clicking P-wave button
        p_wave_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-wave-type='p-wave']")
        p_wave_btn.click()
        
        # Wait for state change
        time.sleep(0.5)
        
        # Check that P-wave button is now active
        self.assertIn("active", p_wave_btn.get_attribute("class"))
        
        # Check that All Waves button is no longer active
        self.assertNotIn("active", all_waves_btn.get_attribute("class"))
    
    def test_analysis_parameters_controls(self):
        """Test analysis parameter input controls"""
        # Test sampling rate input
        sampling_rate_input = self.driver.find_element(By.ID, "samplingRate")
        self.assertEqual(sampling_rate_input.get_attribute("value"), "100")
        
        # Change sampling rate
        sampling_rate_input.clear()
        sampling_rate_input.send_keys("200")
        self.assertEqual(sampling_rate_input.get_attribute("value"), "200")
        
        # Test minimum SNR input
        min_snr_input = self.driver.find_element(By.ID, "minSnr")
        self.assertEqual(min_snr_input.get_attribute("value"), "2.0")
        
        # Test minimum confidence input
        min_confidence_input = self.driver.find_element(By.ID, "minConfidence")
        self.assertEqual(min_confidence_input.get_attribute("value"), "0.3")
        
        # Test filter frequency input
        filter_freq_input = self.driver.find_element(By.ID, "filterFreq")
        self.assertEqual(filter_freq_input.get_attribute("value"), "20")
    
    def test_chart_type_selection(self):
        """Test chart type selection controls"""
        # Find chart control buttons
        chart_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".chart-control-btn[data-chart]")
        self.assertEqual(len(chart_buttons), 3)  # waveform, frequency, spectrogram
        
        # Check initial state - "Waveform" should be active
        waveform_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-chart='waveform']")
        self.assertIn("active", waveform_btn.get_attribute("class"))
        
        # Test clicking frequency button
        frequency_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-chart='frequency']")
        frequency_btn.click()
        
        # Wait for state change
        time.sleep(0.5)
        
        # Check that frequency button is now active
        self.assertIn("active", frequency_btn.get_attribute("class"))
        
        # Check that waveform button is no longer active
        self.assertNotIn("active", waveform_btn.get_attribute("class"))
    
    def test_educational_content_display(self):
        """Test educational content functionality"""
        # Find educational buttons
        education_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".wave-type-btn[data-education]")
        self.assertEqual(len(education_buttons), 4)  # p-wave, s-wave, surface, analysis
        
        # Test clicking P-wave education button
        p_wave_edu_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-education='p-wave']")
        p_wave_edu_btn.click()
        
        # Wait for content change
        time.sleep(0.5)
        
        # Check that P-wave educational content is displayed
        p_wave_content = self.driver.find_element(By.ID, "education-p-wave")
        self.assertIn("active", p_wave_content.get_attribute("class"))
        
        # Check content text
        self.assertIn("P-Waves (Primary Waves)", p_wave_content.text)
        self.assertIn("fastest seismic waves", p_wave_content.text)
    
    def test_chart_canvas_presence(self):
        """Test that the chart canvas is present and properly sized"""
        # Find the main chart canvas
        chart_canvas = self.driver.find_element(By.ID, "mainChart")
        self.assertTrue(chart_canvas.is_displayed())
        
        # Check canvas dimensions
        canvas_width = chart_canvas.size['width']
        canvas_height = chart_canvas.size['height']
        
        # Canvas should have reasonable dimensions
        self.assertGreater(canvas_width, 400)
        self.assertGreater(canvas_height, 300)
    
    def test_analysis_results_display(self):
        """Test analysis results display elements"""
        # Check wave detection results panel
        wave_detection_panel = self.driver.find_element(By.XPATH, "//h3[text()='Wave Detection Results']/parent::div")
        self.assertTrue(wave_detection_panel.is_displayed())
        
        # Check metric items
        metric_items = wave_detection_panel.find_elements(By.CLASS_NAME, "metric-item")
        self.assertEqual(len(metric_items), 4)  # P-Waves, S-Waves, Surface Waves, Quality Score
        
        # Check arrival times panel
        arrival_times_panel = self.driver.find_element(By.XPATH, "//h3[text()='Arrival Times']/parent::div")
        self.assertTrue(arrival_times_panel.is_displayed())
        
        # Check magnitude estimates panel
        magnitude_panel = self.driver.find_element(By.XPATH, "//h3[text()='Magnitude Estimates']/parent::div")
        self.assertTrue(magnitude_panel.is_displayed())
    
    def test_toolbar_buttons_presence(self):
        """Test that all toolbar buttons are present and clickable"""
        # Find toolbar buttons
        analyze_btn = self.driver.find_element(By.ID, "analyzeBtn")
        export_btn = self.driver.find_element(By.ID, "exportBtn")
        report_btn = self.driver.find_element(By.ID, "reportBtn")
        reset_btn = self.driver.find_element(By.ID, "resetBtn")
        
        # Check buttons are displayed and enabled
        buttons = [analyze_btn, export_btn, report_btn, reset_btn]
        for button in buttons:
            self.assertTrue(button.is_displayed())
            self.assertTrue(button.is_enabled())
    
    def test_system_status_indicator(self):
        """Test system status indicator functionality"""
        # Find status indicator
        status_indicator = self.driver.find_element(By.ID, "systemStatus")
        self.assertTrue(status_indicator.is_displayed())
        
        # Check initial status
        self.assertIn("ready", status_indicator.get_attribute("class"))
        
        # Check status text
        status_text = status_indicator.find_element(By.TAG_NAME, "span").text
        self.assertIn("System Ready", status_text)
    
    def test_loading_overlay_hidden_initially(self):
        """Test that loading overlay is hidden initially"""
        loading_overlay = self.driver.find_element(By.ID, "loadingOverlay")
        
        # Loading overlay should be hidden initially
        self.assertEqual(loading_overlay.value_of_css_property("display"), "none")
    
    def test_reset_functionality(self):
        """Test dashboard reset functionality"""
        # Change some parameters first
        sampling_rate_input = self.driver.find_element(By.ID, "samplingRate")
        sampling_rate_input.clear()
        sampling_rate_input.send_keys("200")
        
        # Select a different wave type
        s_wave_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-wave-type='s-wave']")
        s_wave_btn.click()
        
        # Wait for changes
        time.sleep(0.5)
        
        # Click reset button
        reset_btn = self.driver.find_element(By.ID, "resetBtn")
        reset_btn.click()
        
        # Wait for reset
        time.sleep(1)
        
        # Check that parameters are reset
        self.assertEqual(sampling_rate_input.get_attribute("value"), "100")
        
        # Check that wave type is reset to "all"
        all_waves_btn = self.driver.find_element(By.CSS_SELECTOR, "[data-wave-type='all']")
        self.assertIn("active", all_waves_btn.get_attribute("class"))
    
    def test_responsive_design_elements(self):
        """Test responsive design elements"""
        # Test with different window sizes
        original_size = self.driver.get_window_size()
        
        try:
            # Test mobile size
            self.driver.set_window_size(375, 667)
            time.sleep(0.5)
            
            # Dashboard should still be visible
            dashboard_container = self.driver.find_element(By.CLASS_NAME, "dashboard-container")
            self.assertTrue(dashboard_container.is_displayed())
            
            # Test tablet size
            self.driver.set_window_size(768, 1024)
            time.sleep(0.5)
            
            # Dashboard should still be functional
            wave_type_buttons = self.driver.find_elements(By.CSS_SELECTOR, ".wave-type-btn[data-wave-type]")
            self.assertEqual(len(wave_type_buttons), 4)
            
        finally:
            # Restore original window size
            self.driver.set_window_size(original_size['width'], original_size['height'])
    
    def test_javascript_functionality(self):
        """Test JavaScript functionality is working"""
        # Execute JavaScript to check if dashboard object exists
        dashboard_exists = self.driver.execute_script("return typeof window.waveDashboard !== 'undefined';")
        self.assertTrue(dashboard_exists, "Wave dashboard JavaScript object should be initialized")
        
        # Test that Chart.js is loaded
        chartjs_loaded = self.driver.execute_script("return typeof Chart !== 'undefined';")
        self.assertTrue(chartjs_loaded, "Chart.js should be loaded")
        
        # Test that axios is loaded
        axios_loaded = self.driver.execute_script("return typeof axios !== 'undefined';")
        self.assertTrue(axios_loaded, "Axios should be loaded")


class WaveAnalysisDashboardUnitTests(unittest.TestCase):
    """Unit tests for dashboard functionality without browser"""
    
    def test_dashboard_route_exists(self):
        """Test that the dashboard route exists in Flask app"""
        try:
            from app import app
            
            with app.test_client() as client:
                response = client.get('/wave_analysis_dashboard')
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'Wave Analysis Dashboard', response.data)
                
        except ImportError:
            self.skipTest("Flask app not available for testing")
    
    def test_dashboard_template_structure(self):
        """Test dashboard template structure"""
        template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'wave_analysis_dashboard.html')
        
        if os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                
            # Check for essential elements
            self.assertIn('dashboard-container', template_content)
            self.assertIn('wave-type-selector', template_content)
            self.assertIn('analysis-parameters', template_content)
            self.assertIn('mainChart', template_content)
            self.assertIn('educational-content', template_content)
            self.assertIn('toolbar-buttons', template_content)
        else:
            self.skipTest("Dashboard template file not found")
    
    def test_javascript_file_exists(self):
        """Test that the JavaScript file exists"""
        js_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'js', 'wave_dashboard.js')
        self.assertTrue(os.path.exists(js_path), "wave_dashboard.js should exist")
        
        if os.path.exists(js_path):
            with open(js_path, 'r', encoding='utf-8') as f:
                js_content = f.read()
                
            # Check for essential JavaScript components
            self.assertIn('WaveAnalysisDashboard', js_content)
            self.assertIn('selectWaveType', js_content)
            self.assertIn('updateChart', js_content)
            self.assertIn('startAnalysis', js_content)
            self.assertIn('exportResults', js_content)


def run_dashboard_tests():
    """Run all dashboard tests"""
    print("Running Wave Analysis Dashboard Tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add unit tests (these don't require browser)
    suite.addTest(unittest.makeSuite(WaveAnalysisDashboardUnitTests))
    
    # Add browser tests (these require Chrome/chromedriver)
    try:
        suite.addTest(unittest.makeSuite(WaveAnalysisDashboardTests))
        print("Browser tests included - requires Chrome/chromedriver")
    except Exception as e:
        print(f"Browser tests skipped: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_dashboard_tests()
    sys.exit(0 if success else 1)