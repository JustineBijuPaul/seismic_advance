#!/usr/bin/env python3
"""
Basic test for upload functionality
"""

import tempfile
import numpy as np
import scipy.io.wavfile as wav
from app import app

def test_basic_upload():
    """Test basic upload functionality"""
    # Create test file
    t = np.linspace(0, 5, 500)
    signal = 0.5 * np.sin(2 * np.pi * 10 * t)
    signal = (signal * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav.write(f.name, 100, signal)
        test_file = f.name

    print(f'Created test file: {test_file}')

    # Test app creation
    with app.test_client() as client:
        response = client.get('/upload')
        print(f'Upload page status: {response.status_code}')
        
        # Test file upload
        with open(test_file, 'rb') as f:
            response = client.post('/upload', data={
                'file': (f, 'test.wav'),
                'enable_wave_analysis': 'false',
                'async_processing': 'false'
            })
        
        print(f'Upload response status: {response.status_code}')
        if response.status_code == 200:
            print('Upload test passed!')
        else:
            print(f'Upload failed: {response.data}')

    print('Basic functionality test completed')

if __name__ == '__main__':
    test_basic_upload()