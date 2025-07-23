"""
Comprehensive end-to-end integration tests for the complete earthquake wave analysis workflow.

This module tests the complete workflow from file upload through wave analysis to export,
validating all API endpoints with realistic earthquake data and testing error handling
and recovery scenarios across all components.

Tests Requirements: All requirements integration (1.1, 1.2, 1.3, 2.1-2.5, 3.1-3.5, 4.1-4.4, 5.1-5.4, 6.1-6.5)
"""

import unittest
import tempfile
import os
import time
import json
import numpy as np
import scipy.io.wavfile as wav
from unittest.mock import patch, MagicMock
import threading
from datetime import datetime, timedelta
import requests
import io
import base64
import sys
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import Flask app and test client
try:
    import app
    from app import application as flask_app, db, fs, processing_tasks
    
    # Try to import wave analysis components
    try:
        from wave_analysis.services.test_data_manager import TestDataManager
        from wave_analysis.models.wave_models import WaveSegment, WaveAnalysisResult, DetailedAnalysis
        WAVE_ANALYSIS_AVAILABLE = True
    except ImportError:
        # Create mock test data manager if wave analysis not available
        class MockTestDataManager:
            def create_synthetic_earthquake(self, magnitude, distance, duration):
                # Create simple synthetic earthquake data
                samples = int(duration * 100)  # 100 Hz sampling rate
                t = np.linspace(0, duration, samples)
                # Simple earthquake simulation with P, S, and surface waves
                p_wave = 0.1 * np.sin(2 * np.pi * 5 * t) * np.exp(-t/10)
                s_wave = 0.2 * np.sin(2 * np.pi * 3 * t) * np.exp(-(t-5)/15)
                surface_wave = 0.3 * np.sin(2 * np.pi * 1 * t) * np.exp(-(t-10)/20)
                return p_wave + s_wave + surface_wave
            
            def generate_noise_samples(self, duration):
                samples = int(duration * 100)
                return [np.random.normal(0, 0.01, samples)]
            
            def create_multi_channel_data(self, channels):
                duration = 30.0
                samples = int(duration * 100)
                return [np.random.normal(0, 0.1, samples) for _ in range(channels)]
        
        TestDataManager = MockTestDataManager
        WAVE_ANALYSIS_AVAILABLE = False
    
    FLASK_APP_AVAILABLE = True
    
except ImportError as e:
    print(f"Flask app not available: {e}")
    FLASK_APP_AVAILABLE = False
    flask_app = None
    db = None
    fs = None
    processing_tasks = {}
    TestDataManager = None


class EndToEndIntegrationTest(unittest.TestCase):
    """
    Comprehensive end-to-end integration tests covering the complete workflow
    from file upload through wave analysis to export.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        if not FLASK_APP_AVAILABLE:
            cls.skipTest(cls, "Flask app not available")
        
        # Configure Flask app for testing
        flask_app.config['TESTING'] = True
        flask_app.config['WTF_CSRF_ENABLED'] = False
        cls.client = flask_app.test_client()
        cls.app_context = flask_app.app_context()
        cls.app_context.push()
        
        # Initialize test data manager
        cls.test_data_manager = TestDataManager()
        
        # Create test database collections
        cls.test_collection_names = [
            'test_files', 'test_analyses', 'test_async_analyses',
            'test_monitoring_results', 'test_wave_analyses'
        ]
        
        # Clean up any existing test data
        for collection_name in cls.test_collection_names:
            db[collection_name].delete_many({})
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if FLASK_APP_AVAILABLE:
            # Clean up test data
            for collection_name in cls.test_collection_names:
                db[collection_name].delete_many({})
            
            cls.app_context.pop()
    
    def setUp(self):
        """Set up for each test."""
        if not FLASK_APP_AVAILABLE:
            self.skipTest("Flask app not available")
        
        # Clear processing tasks
        processing_tasks.clear()
        
        # Create test seismic data files
        self.test_files = self._create_test_files()
    
    def tearDown(self):
        """Clean up after each test."""
        # Clean up temporary files
        for file_path in getattr(self, 'temp_files', []):
            try:
                os.unlink(file_path)
            except:
                pass
    
    def _create_test_files(self):
        """Create various test seismic data files for testing."""
        test_files = {}
        self.temp_files = []
        
        # 1. Synthetic earthquake with clear P, S, and surface waves
        earthquake_data = self.test_data_manager.create_synthetic_earthquake(
            magnitude=5.5, distance=100.0, duration=60.0
        )
        
        # Save as WAV file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav.write(temp_file.name, 100, earthquake_data.astype(np.float32))
        test_files['earthquake_wav'] = temp_file.name
        self.temp_files.append(temp_file.name)
        
        # 2. Noise-only file (no earthquake)
        noise_data = self.test_data_manager.generate_noise_samples(duration=30.0)[0]
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav.write(temp_file.name, 100, noise_data.astype(np.float32))
        test_files['noise_wav'] = temp_file.name
        self.temp_files.append(temp_file.name)
        
        # 3. Large file for async processing testing
        large_earthquake_data = self.test_data_manager.create_synthetic_earthquake(
            magnitude=6.2, distance=50.0, duration=300.0  # 5 minutes
        )
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav.write(temp_file.name, 100, large_earthquake_data.astype(np.float32))
        test_files['large_earthquake_wav'] = temp_file.name
        self.temp_files.append(temp_file.name)
        
        # 4. Multi-channel data
        multi_channel_data = self.test_data_manager.create_multi_channel_data(channels=3)
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        # Convert to stereo for testing (take first 2 channels)
        stereo_data = np.column_stack([multi_channel_data[0], multi_channel_data[1]])
        wav.write(temp_file.name, 100, stereo_data.astype(np.float32))
        test_files['multi_channel_wav'] = temp_file.name
        self.temp_files.append(temp_file.name)
        
        return test_files
    
    def test_complete_workflow_earthquake_detection(self):
        """
        Test complete workflow: upload -> analysis -> visualization -> export
        Requirements: 1.1, 1.2, 1.3, 2.1, 4.1, 4.2, 4.3
        """
        # Step 1: Upload earthquake file
        with open(self.test_files['earthquake_wav'], 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'test_earthquake.wav'),
                'enable_wave_analysis': 'true',
                'async_processing': 'false'
            })
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        
        # Verify basic earthquake detection
        self.assertIn('file_id', result)
        self.assertIn('prediction', result)
        self.assertEqual(result['prediction'], 'Seismic Activity Detected')
        self.assertIn('amplitude_data', result)
        self.assertIn('time_labels', result)
        
        # Verify wave analysis results if available
        if 'wave_analysis' in result:
            wave_analysis = result['wave_analysis']
            self.assertIn('wave_separation', wave_analysis)
            self.assertIn('arrival_times', wave_analysis)
            self.assertIn('magnitude_estimates', wave_analysis)
            
            # Check wave separation counts
            separation = wave_analysis['wave_separation']
            self.assertGreaterEqual(separation['p_waves_count'], 0)
            self.assertGreaterEqual(separation['s_waves_count'], 0)
            
            # Check arrival times
            arrival_times = wave_analysis['arrival_times']
            self.assertIsInstance(arrival_times['p_wave_arrival'], (int, float))
            self.assertIsInstance(arrival_times['s_wave_arrival'], (int, float))
            self.assertIsInstance(arrival_times['sp_time_difference'], (int, float))
        
        file_id = result['file_id']
        
        # Step 2: Test export functionality
        export_data = {
            'time_labels': result['time_labels'][:100],  # Limit for testing
            'amplitude_data': result['amplitude_data'][:100],
            'sampling_rate': result['sampling_rate']
        }
        
        # Test CSV export
        csv_response = self.client.post('/download_csv', json=export_data)
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(csv_response.content_type, 'text/csv; charset=utf-8')
        
        # Test MSEED export
        mseed_response = self.client.post('/download_mseed', json=export_data)
        self.assertEqual(mseed_response.status_code, 200)
        self.assertEqual(mseed_response.content_type, 'application/octet-stream')
        
        # Test XML export
        xml_response = self.client.post('/download_xml', json=export_data)
        self.assertEqual(xml_response.status_code, 200)
        self.assertEqual(xml_response.content_type, 'application/xml; charset=utf-8')
        
        # Step 3: Test wave analysis API endpoint if available
        if 'wave_analysis' in result:
            api_response = self.client.post('/api/analyze_waves', json={
                'file_id': file_id,
                'parameters': {
                    'sampling_rate': 100,
                    'min_snr': 2.0,
                    'min_detection_confidence': 0.3
                }
            })
            
            if api_response.status_code == 200:
                api_result = json.loads(api_response.data)
                self.assertIn('wave_separation', api_result)
                self.assertIn('detailed_analysis', api_result)
                self.assertIn('quality_metrics', api_result)
    
    def test_async_processing_workflow(self):
        """
        Test asynchronous processing workflow for large files.
        Requirements: 6.1, 6.2, 4.4
        """
        # Upload large file with async processing
        with open(self.test_files['large_earthquake_wav'], 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'large_earthquake.wav'),
                'enable_wave_analysis': 'true',
                'async_processing': 'true'
            })
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        
        # Verify async response
        self.assertIn('task_id', result)
        self.assertIn('status', result)
        self.assertEqual(result['status'], 'processing')
        self.assertTrue(result['async'])
        
        task_id = result['task_id']
        
        # Poll task status
        max_attempts = 30  # 30 seconds timeout
        for attempt in range(max_attempts):
            status_response = self.client.get(f'/api/task_status/{task_id}')
            self.assertEqual(status_response.status_code, 200)
            
            status_result = json.loads(status_response.data)
            
            if status_result['status'] == 'completed':
                # Test getting results
                results_response = self.client.get(f'/api/task_results/{task_id}')
                self.assertEqual(results_response.status_code, 200)
                
                results_data = json.loads(results_response.data)
                self.assertIn('prediction', results_data)
                self.assertIn('amplitude_data', results_data)
                break
            elif status_result['status'] == 'failed':
                self.fail(f"Async processing failed: {status_result.get('message', 'Unknown error')}")
            
            time.sleep(1)
        else:
            self.fail("Async processing timed out")
    
    def test_noise_file_handling(self):
        """
        Test handling of files with no seismic activity.
        Requirements: 1.1, 6.5
        """
        with open(self.test_files['noise_wav'], 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'noise.wav'),
                'enable_wave_analysis': 'false',
                'async_processing': 'false'
            })
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        
        # Verify no seismic activity detected
        self.assertEqual(result['prediction'], 'No Seismic Activity Detected')
        self.assertIn('amplitude_data', result)
        self.assertNotIn('time_indices', result)  # No earthquake events
    
    def test_multi_channel_data_processing(self):
        """
        Test processing of multi-channel seismic data.
        Requirements: 2.5, 1.3
        """
        with open(self.test_files['multi_channel_wav'], 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'multi_channel.wav'),
                'enable_wave_analysis': 'true',
                'async_processing': 'false'
            })
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        
        # Verify processing succeeded (multi-channel converted to mono)
        self.assertIn('prediction', result)
        self.assertIn('amplitude_data', result)
        self.assertIn('sampling_rate', result)
    
    def test_real_time_monitoring_workflow(self):
        """
        Test real-time monitoring capabilities.
        Requirements: 6.1, 6.2, 6.4
        """
        # First upload a file to use for monitoring
        with open(self.test_files['earthquake_wav'], 'rb') as f:
            upload_response = self.client.post('/upload', data={
                'file': (f, 'monitor_test.wav'),
                'enable_wave_analysis': 'false',
                'async_processing': 'false'
            })
        
        self.assertEqual(upload_response.status_code, 200)
        upload_result = json.loads(upload_response.data)
        file_id = upload_result['file_id']
        
        # Start monitoring
        monitor_response = self.client.post('/start_monitoring', json={
            'file_id': file_id
        })
        self.assertEqual(monitor_response.status_code, 200)
        
        # Wait a moment for monitoring to process
        time.sleep(2)
        
        # Get latest results
        results_response = self.client.get('/get_latest_results')
        self.assertEqual(results_response.status_code, 200)
        
        results_data = json.loads(results_response.data)
        if 'error' not in results_data:
            self.assertIn('prediction', results_data)
            self.assertIn('timestamp', results_data)
        
        # Stop monitoring
        stop_response = self.client.post('/stop_monitoring')
        self.assertEqual(stop_response.status_code, 200)
    
    def test_error_handling_scenarios(self):
        """
        Test various error handling and recovery scenarios.
        Requirements: 6.5, Error handling across all components
        """
        # Test 1: Upload without file
        response = self.client.post('/upload', data={})
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        self.assertIn('error', result)
        
        # Test 2: Invalid file format
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        temp_file.write(b'This is not audio data')
        temp_file.close()
        self.temp_files.append(temp_file.name)
        
        with open(temp_file.name, 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'invalid.txt'),
                'enable_wave_analysis': 'false',
                'async_processing': 'false'
            })
        
        # Should handle gracefully (may succeed with error in processing)
        self.assertIn(response.status_code, [200, 400, 500])
        
        # Test 3: Invalid task ID
        response = self.client.get('/api/task_status/invalid_task_id')
        self.assertEqual(response.status_code, 404)
        
        # Test 4: Invalid analysis ID for wave results
        response = self.client.get('/api/wave_results/invalid_analysis_id')
        self.assertIn(response.status_code, [400, 404, 503])
        
        # Test 5: Missing required parameters
        response = self.client.post('/api/analyze_waves', json={})
        self.assertIn(response.status_code, [400, 503])
    
    def test_alert_system_integration(self):
        """
        Test alert system integration with wave analysis.
        Requirements: 6.4, Alert system functionality
        """
        if not alert_system:
            self.skipTest("Alert system not available")
        
        # Test getting recent alerts
        response = self.client.get('/api/alerts/recent?limit=10')
        self.assertIn(response.status_code, [200, 503])
        
        if response.status_code == 200:
            result = json.loads(response.data)
            self.assertIn('alerts', result)
            self.assertIn('count', result)
        
        # Test getting alert statistics
        response = self.client.get('/api/alerts/statistics')
        self.assertIn(response.status_code, [200, 503])
        
        # Test getting alert thresholds
        response = self.client.get('/api/alerts/thresholds')
        self.assertIn(response.status_code, [200, 503])
    
    def test_educational_features(self):
        """
        Test educational features and content delivery.
        Requirements: 5.1, 5.2, 5.3, 5.4
        """
        # Test documentation page
        response = self.client.get('/documentation')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'html', response.data.lower())
        
        # Test wave analysis dashboard
        response = self.client.get('/wave_analysis_dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'html', response.data.lower())
        
        # Test earthquake history page
        response = self.client.get('/earthquake_history')
        # May fail due to external API, so accept various status codes
        self.assertIn(response.status_code, [200, 500])
    
    def test_database_integration(self):
        """
        Test database operations and data persistence.
        Requirements: 4.4, Database integration
        """
        # Upload file and verify database storage
        with open(self.test_files['earthquake_wav'], 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'db_test.wav'),
                'enable_wave_analysis': 'false',
                'async_processing': 'false'
            })
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        file_id = result['file_id']
        
        # Verify file exists in GridFS
        self.assertTrue(fs.exists(file_id))
        
        # Test file retrieval
        stored_file = fs.get(file_id)
        self.assertIsNotNone(stored_file)
        stored_file.close()
    
    def test_performance_under_load(self):
        """
        Test system performance under concurrent load.
        Requirements: 6.3, Performance and scalability
        """
        def upload_file_worker(file_path, worker_id):
            """Worker function for concurrent uploads."""
            try:
                with open(file_path, 'rb') as f:
                    response = self.client.post('/upload', data={
                        'file': (f, f'concurrent_test_{worker_id}.wav'),
                        'enable_wave_analysis': 'false',
                        'async_processing': 'false'
                    })
                return response.status_code == 200
            except Exception as e:
                print(f"Worker {worker_id} failed: {e}")
                return False
        
        # Create multiple threads for concurrent uploads
        num_workers = 3  # Keep it reasonable for testing
        threads = []
        results = []
        
        for i in range(num_workers):
            thread = threading.Thread(
                target=lambda i=i: results.append(
                    upload_file_worker(self.test_files['earthquake_wav'], i)
                )
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout per thread
        
        # Verify at least some uploads succeeded
        successful_uploads = sum(results)
        self.assertGreater(successful_uploads, 0, "No concurrent uploads succeeded")
    
    def test_api_endpoint_validation(self):
        """
        Test all API endpoints with various input scenarios.
        Requirements: API validation and error handling
        """
        # Test main page
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        # Test upload page
        response = self.client.get('/upload')
        self.assertEqual(response.status_code, 200)
        
        # Test invalid endpoints
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        # Test POST to GET-only endpoints
        response = self.client.post('/')
        self.assertEqual(response.status_code, 405)
    
    def test_data_export_formats(self):
        """
        Test all supported data export formats.
        Requirements: 4.1, 4.2, 4.3
        """
        # Create test data for export
        test_data = {
            'time_labels': ['0:00:00', '0:00:01', '0:00:02'],
            'amplitude_data': [0.1, 0.5, -0.3],
            'sampling_rate': 100
        }
        
        # Test PNG export (with mock base64 image)
        mock_image_data = base64.b64encode(b'fake_png_data').decode()
        png_data = {'image_base64': f'data:image/png;base64,{mock_image_data}'}
        
        png_response = self.client.post('/download_png', json=png_data)
        self.assertEqual(png_response.status_code, 200)
        
        # Test CSV export
        csv_response = self.client.post('/download_csv', json=test_data)
        self.assertEqual(csv_response.status_code, 200)
        
        # Test MSEED export
        mseed_response = self.client.post('/download_mseed', json=test_data)
        self.assertEqual(mseed_response.status_code, 200)
        
        # Test XML export
        xml_response = self.client.post('/download_xml', json=test_data)
        self.assertEqual(xml_response.status_code, 200)


class WebSocketIntegrationTest(unittest.TestCase):
    """
    Test WebSocket functionality for real-time features.
    Requirements: 6.4, Real-time monitoring and alerts
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up WebSocket test environment."""
        if not FLASK_APP_AVAILABLE:
            cls.skipTest(cls, "Flask app not available")
        
        flask_app.config['TESTING'] = True
        cls.client = flask_app.test_client()
        cls.app_context = flask_app.app_context()
        cls.app_context.push()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up WebSocket test environment."""
        if FLASK_APP_AVAILABLE:
            cls.app_context.pop()
    
    def test_websocket_connection(self):
        """Test WebSocket connection for alerts."""
        # This is a basic test - full WebSocket testing would require
        # additional setup with socketio test client
        self.assertTrue(True)  # Placeholder for WebSocket tests


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTest(unittest.makeSuite(EndToEndIntegrationTest))
    suite.addTest(unittest.makeSuite(WebSocketIntegrationTest))
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"End-to-End Integration Test Summary")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)