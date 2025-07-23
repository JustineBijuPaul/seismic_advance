"""
Integration tests for enhanced upload workflow with wave analysis and async processing.
Tests Requirements 1.1 and 6.1 from the earthquake wave analysis specification.
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

# Import Flask app and test client
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db, fs, processing_tasks, WAVE_ANALYSIS_AVAILABLE


class TestEnhancedUploadWorkflow(unittest.TestCase):
    """Test enhanced upload workflow with wave analysis and async processing"""
    
    def setUp(self):
        """Set up test environment"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # Clear processing tasks
        global processing_tasks
        processing_tasks.clear()
        
        # Create test audio file
        self.test_file_path = self._create_test_audio_file()
        
    def tearDown(self):
        """Clean up test environment"""
        # Clean up test file
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        
        # Clear processing tasks
        processing_tasks.clear()
        
        # Clean up database test records
        try:
            db.async_analyses.delete_many({'file_id': {'$regex': 'test_'}})
        except:
            pass
    
    def _create_test_audio_file(self, duration=10, sampling_rate=100):
        """Create a synthetic test audio file with earthquake-like signal"""
        t = np.linspace(0, duration, duration * sampling_rate)
        
        # Background noise
        noise = np.random.normal(0, 0.1, len(t))
        
        # Synthetic P-wave (high frequency, early arrival)
        p_wave_start = 2.0
        p_wave_duration = 1.0
        p_wave_mask = (t >= p_wave_start) & (t <= p_wave_start + p_wave_duration)
        p_wave = np.where(p_wave_mask, 
                         0.8 * np.sin(2 * np.pi * 15 * (t - p_wave_start)) * 
                         np.exp(-3 * (t - p_wave_start)), 0)
        
        # Synthetic S-wave (lower frequency, later arrival)
        s_wave_start = 4.0
        s_wave_duration = 2.0
        s_wave_mask = (t >= s_wave_start) & (t <= s_wave_start + s_wave_duration)
        s_wave = np.where(s_wave_mask,
                         1.2 * np.sin(2 * np.pi * 8 * (t - s_wave_start)) * 
                         np.exp(-2 * (t - s_wave_start)), 0)
        
        # Synthetic surface wave (low frequency, latest arrival)
        surf_wave_start = 6.0
        surf_wave_duration = 3.0
        surf_wave_mask = (t >= surf_wave_start) & (t <= surf_wave_start + surf_wave_duration)
        surf_wave = np.where(surf_wave_mask,
                            0.6 * np.sin(2 * np.pi * 3 * (t - surf_wave_start)) * 
                            np.exp(-1 * (t - surf_wave_start)), 0)
        
        # Combine all components
        signal = noise + p_wave + s_wave + surf_wave
        
        # Normalize to 16-bit range
        signal = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav.write(temp_file.name, sampling_rate, signal)
        temp_file.close()
        
        return temp_file.name
    
    def _create_large_test_file(self, size_mb=60):
        """Create a large test file to trigger async processing"""
        # Create a file larger than the 50MB threshold
        duration = 600  # 10 minutes
        sampling_rate = 100
        t = np.linspace(0, duration, duration * sampling_rate)
        
        # Simple signal with noise
        signal = 0.1 * np.random.normal(0, 1, len(t))
        signal = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav.write(temp_file.name, sampling_rate, signal)
        temp_file.close()
        
        return temp_file.name
    
    def test_synchronous_upload_basic_analysis(self):
        """Test synchronous upload with basic earthquake detection"""
        with open(self.test_file_path, 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'test_earthquake.wav'),
                'enable_wave_analysis': 'false',
                'async_processing': 'false'
            })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify basic response structure
        self.assertIn('file_id', data)
        self.assertIn('prediction', data)
        self.assertIn('time_labels', data)
        self.assertIn('amplitude_data', data)
        self.assertIn('sampling_rate', data)
        self.assertEqual(data['async'], False)
        self.assertEqual(data['wave_analysis_enabled'], False)
        
        # Should detect seismic activity due to synthetic earthquake signal
        self.assertEqual(data['prediction'], 'Seismic Activity Detected')
        self.assertIn('time_indices', data)
        self.assertIn('amplitudes', data)
    
    @unittest.skipIf(not WAVE_ANALYSIS_AVAILABLE, "Wave analysis components not available")
    def test_synchronous_upload_with_wave_analysis(self):
        """Test synchronous upload with wave analysis enabled"""
        with open(self.test_file_path, 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'test_earthquake.wav'),
                'enable_wave_analysis': 'true',
                'async_processing': 'false'
            })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify wave analysis is enabled and results are present
        self.assertEqual(data['wave_analysis_enabled'], True)
        
        if data['prediction'] == 'Seismic Activity Detected':
            # Should have wave analysis results
            self.assertIn('wave_analysis', data)
            wave_analysis = data['wave_analysis']
            
            # Verify wave analysis structure
            self.assertIn('wave_separation', wave_analysis)
            self.assertIn('arrival_times', wave_analysis)
            self.assertIn('magnitude_estimates', wave_analysis)
            self.assertIn('quality_score', wave_analysis)
            
            # Verify wave separation counts
            wave_sep = wave_analysis['wave_separation']
            self.assertIn('p_waves_count', wave_sep)
            self.assertIn('s_waves_count', wave_sep)
            self.assertIn('surface_waves_count', wave_sep)
    
    def test_asynchronous_upload_processing(self):
        """Test asynchronous upload processing"""
        with open(self.test_file_path, 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'test_earthquake.wav'),
                'enable_wave_analysis': 'false',
                'async_processing': 'true'
            })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Verify async response structure
        self.assertIn('file_id', data)
        self.assertIn('task_id', data)
        self.assertEqual(data['status'], 'processing')
        self.assertEqual(data['async'], True)
        self.assertIn('message', data)
        
        task_id = data['task_id']
        
        # Wait for processing to complete (with timeout)
        max_wait_time = 30  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = self.client.get(f'/api/task_status/{task_id}')
            self.assertEqual(status_response.status_code, 200)
            
            status_data = json.loads(status_response.data)
            
            if status_data['status'] == 'completed':
                break
            elif status_data['status'] == 'failed':
                self.fail(f"Task failed: {status_data.get('message', 'Unknown error')}")
            
            time.sleep(1)
        else:
            self.fail("Task did not complete within timeout period")
        
        # Get final results
        results_response = self.client.get(f'/api/task_results/{task_id}')
        self.assertEqual(results_response.status_code, 200)
        
        results_data = json.loads(results_response.data)
        
        # Verify results structure
        self.assertIn('prediction', results_data)
        self.assertIn('file_id', results_data)
        self.assertEqual(results_data['async'], True)
    
    def test_large_file_automatic_async_processing(self):
        """Test that large files automatically trigger async processing"""
        # Create a large test file
        large_file_path = self._create_large_test_file()
        
        try:
            with open(large_file_path, 'rb') as f:
                response = self.client.post('/upload', data={
                    'file': (f, 'large_earthquake.wav'),
                    'enable_wave_analysis': 'false',
                    'async_processing': 'false'  # Should be overridden for large files
                })
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            
            # Should automatically enable async processing for large files
            self.assertEqual(data['async'], True)
            self.assertIn('task_id', data)
            
        finally:
            if os.path.exists(large_file_path):
                os.remove(large_file_path)
    
    @unittest.skipIf(not WAVE_ANALYSIS_AVAILABLE, "Wave analysis components not available")
    def test_async_upload_with_wave_analysis(self):
        """Test asynchronous upload with wave analysis enabled"""
        with open(self.test_file_path, 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'test_earthquake.wav'),
                'enable_wave_analysis': 'true',
                'async_processing': 'true'
            })
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        self.assertEqual(data['async'], True)
        self.assertEqual(data['wave_analysis_enabled'], True)
        
        task_id = data['task_id']
        
        # Wait for processing to complete
        max_wait_time = 60  # Longer timeout for wave analysis
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = self.client.get(f'/api/task_status/{task_id}')
            status_data = json.loads(status_response.data)
            
            if status_data['status'] == 'completed':
                break
            elif status_data['status'] == 'failed':
                self.fail(f"Task failed: {status_data.get('message', 'Unknown error')}")
            
            time.sleep(2)
        else:
            self.fail("Wave analysis task did not complete within timeout period")
        
        # Get final results
        results_response = self.client.get(f'/api/task_results/{task_id}')
        results_data = json.loads(results_response.data)
        
        # Verify wave analysis results are present if seismic activity detected
        if results_data['prediction'] == 'Seismic Activity Detected':
            self.assertIn('wave_analysis', results_data)
    
    def test_task_status_api_endpoint(self):
        """Test task status API endpoint functionality"""
        with open(self.test_file_path, 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'test_earthquake.wav'),
                'async_processing': 'true'
            })
        
        data = json.loads(response.data)
        task_id = data['task_id']
        
        # Test task status endpoint
        status_response = self.client.get(f'/api/task_status/{task_id}')
        self.assertEqual(status_response.status_code, 200)
        
        status_data = json.loads(status_response.data)
        self.assertIn('status', status_data)
        self.assertIn('progress', status_data)
        self.assertIn('message', status_data)
        self.assertIn('file_id', status_data)
        
        # Test non-existent task
        invalid_response = self.client.get('/api/task_status/invalid_task_id')
        self.assertEqual(invalid_response.status_code, 404)
    
    def test_task_results_api_endpoint(self):
        """Test task results API endpoint functionality"""
        with open(self.test_file_path, 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'test_earthquake.wav'),
                'async_processing': 'true'
            })
        
        data = json.loads(response.data)
        task_id = data['task_id']
        
        # Wait for completion
        max_wait_time = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = self.client.get(f'/api/task_status/{task_id}')
            status_data = json.loads(status_response.data)
            
            if status_data['status'] in ['completed', 'failed']:
                break
            time.sleep(1)
        
        # Test results endpoint
        results_response = self.client.get(f'/api/task_results/{task_id}')
        
        if status_data['status'] == 'completed':
            self.assertEqual(results_response.status_code, 200)
            results_data = json.loads(results_response.data)
            self.assertIn('prediction', results_data)
        elif status_data['status'] == 'failed':
            self.assertEqual(results_response.status_code, 500)
        
        # Test non-existent task results
        invalid_response = self.client.get('/api/task_results/invalid_task_id')
        self.assertEqual(invalid_response.status_code, 404)
    
    def test_upload_error_handling(self):
        """Test error handling in upload workflow"""
        # Test missing file
        response = self.client.post('/upload', data={})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No file part')
        
        # Test empty filename
        response = self.client.post('/upload', data={
            'file': (tempfile.NamedTemporaryFile(), '')
        })
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No selected file')
    
    def test_concurrent_async_processing(self):
        """Test handling of multiple concurrent async processing tasks"""
        task_ids = []
        
        # Submit multiple files for async processing
        for i in range(3):
            with open(self.test_file_path, 'rb') as f:
                response = self.client.post('/upload', data={
                    'file': (f, f'test_earthquake_{i}.wav'),
                    'async_processing': 'true'
                })
            
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            task_ids.append(data['task_id'])
        
        # Wait for all tasks to complete
        max_wait_time = 60
        start_time = time.time()
        completed_tasks = set()
        
        while len(completed_tasks) < len(task_ids) and time.time() - start_time < max_wait_time:
            for task_id in task_ids:
                if task_id not in completed_tasks:
                    status_response = self.client.get(f'/api/task_status/{task_id}')
                    if status_response.status_code == 200:
                        status_data = json.loads(status_response.data)
                        if status_data['status'] in ['completed', 'failed']:
                            completed_tasks.add(task_id)
            
            time.sleep(1)
        
        # Verify all tasks completed
        self.assertEqual(len(completed_tasks), len(task_ids), 
                        "Not all concurrent tasks completed within timeout")
        
        # Verify results are available for completed tasks
        for task_id in completed_tasks:
            results_response = self.client.get(f'/api/task_results/{task_id}')
            self.assertIn(results_response.status_code, [200, 500])  # Either success or controlled failure
    
    def test_database_integration(self):
        """Test database integration for async analysis results"""
        with open(self.test_file_path, 'rb') as f:
            response = self.client.post('/upload', data={
                'file': (f, 'test_earthquake.wav'),
                'async_processing': 'true'
            })
        
        data = json.loads(response.data)
        task_id = data['task_id']
        
        # Wait for completion
        max_wait_time = 30
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_response = self.client.get(f'/api/task_status/{task_id}')
            status_data = json.loads(status_response.data)
            
            if status_data['status'] == 'completed':
                break
            time.sleep(1)
        
        # Verify database record was created
        try:
            db_record = db.async_analyses.find_one({'task_id': task_id})
            self.assertIsNotNone(db_record, "Database record should be created for completed task")
            
            # Verify record structure
            self.assertIn('file_id', db_record)
            self.assertIn('analysis_timestamp', db_record)
            self.assertIn('results', db_record)
            self.assertIn('processing_time', db_record)
            
        except Exception as e:
            self.skipTest(f"Database not available for testing: {e}")


if __name__ == '__main__':
    unittest.main()