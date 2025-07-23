#!/usr/bin/env python3
"""
Test just the new functions without database dependencies
"""

import numpy as np
import tempfile
import scipy.io.wavfile as wav

def test_function_imports():
    """Test that new functions can be imported"""
    try:
        from app import start_async_analysis, perform_wave_analysis, processing_tasks
        print("✓ Functions imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_task_counter():
    """Test task counter functionality"""
    try:
        from app import processing_tasks, task_counter
        print(f"✓ Task counter initialized: {task_counter}")
        print(f"✓ Processing tasks dict: {len(processing_tasks)} tasks")
        return True
    except Exception as e:
        print(f"✗ Task counter test failed: {e}")
        return False

def test_synthetic_data_creation():
    """Test synthetic data creation for testing"""
    try:
        # Create synthetic earthquake data
        duration = 10
        sampling_rate = 100
        t = np.linspace(0, duration, duration * sampling_rate)
        
        # Background noise
        noise = np.random.normal(0, 0.1, len(t))
        
        # Synthetic P-wave
        p_wave_start = 2.0
        p_wave_duration = 1.0
        p_wave_mask = (t >= p_wave_start) & (t <= p_wave_start + p_wave_duration)
        p_wave = np.where(p_wave_mask, 
                         0.8 * np.sin(2 * np.pi * 15 * (t - p_wave_start)) * 
                         np.exp(-3 * (t - p_wave_start)), 0)
        
        # Combine components
        signal = noise + p_wave
        signal = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav.write(temp_file.name, sampling_rate, signal)
        temp_file.close()
        
        print(f"✓ Created synthetic test file: {temp_file.name}")
        print(f"✓ Signal length: {len(signal)} samples")
        print(f"✓ Signal range: {signal.min()} to {signal.max()}")
        
        return temp_file.name
    except Exception as e:
        print(f"✗ Synthetic data creation failed: {e}")
        return None

if __name__ == '__main__':
    print("Testing enhanced upload workflow functions...")
    
    # Test imports
    if not test_function_imports():
        exit(1)
    
    # Test task counter
    if not test_task_counter():
        exit(1)
    
    # Test synthetic data
    test_file = test_synthetic_data_creation()
    if not test_file:
        exit(1)
    
    print("\n✓ All basic function tests passed!")
    print("Enhanced upload workflow implementation is ready for integration testing.")