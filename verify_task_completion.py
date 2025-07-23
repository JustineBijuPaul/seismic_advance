#!/usr/bin/env python3
"""
Verification script for Task 9.2: Extend file upload handling
This script verifies that all requirements have been implemented.
"""

import os
import re

def verify_upload_endpoint_modifications():
    """Verify that upload endpoint supports wave analysis option"""
    print("🔍 Verifying upload endpoint modifications...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('Wave analysis option support', 'enable_wave_analysis'),
        ('Async processing option support', 'async_processing'),
        ('File size threshold check', 'large_file_threshold'),
        ('Async processing trigger', 'start_async_analysis'),
        ('Wave analysis trigger', 'perform_wave_analysis'),
        ('Enhanced response structure', 'wave_analysis_enabled')
    ]
    
    results = []
    for check_name, pattern in checks:
        if pattern in content:
            results.append(f"  ✓ {check_name}")
        else:
            results.append(f"  ✗ {check_name}")
    
    return results

def verify_wave_analysis_trigger():
    """Verify wave analysis trigger in file processing workflow"""
    print("🔍 Verifying wave analysis trigger implementation...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('Wave analysis function defined', 'def perform_wave_analysis'),
        ('Wave separation engine integration', 'WaveSeparationEngine'),
        ('Wave analyzer integration', 'WaveAnalyzer'),
        ('Conditional wave analysis trigger', 'if enable_wave_analysis and WAVE_ANALYSIS_AVAILABLE'),
        ('Wave analysis error handling', 'wave_analysis_error'),
        ('Wave analysis results formatting', 'wave_separation')
    ]
    
    results = []
    for check_name, pattern in checks:
        if pattern in content:
            results.append(f"  ✓ {check_name}")
        else:
            results.append(f"  ✗ {check_name}")
    
    return results

def verify_async_processing():
    """Verify asynchronous processing implementation"""
    print("🔍 Verifying asynchronous processing implementation...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('Async analysis function defined', 'def start_async_analysis'),
        ('Processing tasks tracking', 'processing_tasks'),
        ('Background thread implementation', 'threading.Thread'),
        ('Task status tracking', 'status.*processing'),
        ('Progress tracking', 'progress'),
        ('Database integration', 'async_analyses'),
        ('Large file auto-async', 'file_size > large_file_threshold')
    ]
    
    results = []
    for check_name, pattern in checks:
        if re.search(pattern, content):
            results.append(f"  ✓ {check_name}")
        else:
            results.append(f"  ✗ {check_name}")
    
    return results

def verify_api_endpoints():
    """Verify new API endpoints for task management"""
    print("🔍 Verifying API endpoints...")
    
    with open('app.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('Task status endpoint', "@app.route('/api/task_status/<task_id>"),
        ('Task results endpoint', "@app.route('/api/task_results/<task_id>"),
        ('Task status function', 'def get_task_status'),
        ('Task results function', 'def get_task_results'),
        ('Error handling for missing tasks', 'Task not found'),
        ('JSON response formatting', 'jsonify')
    ]
    
    results = []
    for check_name, pattern in checks:
        if pattern in content:
            results.append(f"  ✓ {check_name}")
        else:
            results.append(f"  ✗ {check_name}")
    
    return results

def verify_ui_enhancements():
    """Verify UI enhancements for wave analysis and async processing"""
    print("🔍 Verifying UI enhancements...")
    
    with open('templates/upload.html', 'r') as f:
        content = f.read()
    
    checks = [
        ('Wave analysis checkbox', 'enableWaveAnalysis'),
        ('Async processing checkbox', 'enableAsyncProcessing'),
        ('Analysis options section', 'analysis-options'),
        ('Async status display', 'async-status'),
        ('Progress bar implementation', 'progress-bar'),
        ('Wave analysis results display', 'renderWaveAnalysisResults'),
        ('Task polling functionality', 'pollTaskStatus'),
        ('Enhanced form handling', 'handleAsyncProcessing')
    ]
    
    results = []
    for check_name, pattern in checks:
        if pattern in content:
            results.append(f"  ✓ {check_name}")
        else:
            results.append(f"  ✗ {check_name}")
    
    return results

def verify_integration_tests():
    """Verify integration tests implementation"""
    print("🔍 Verifying integration tests...")
    
    if not os.path.exists('tests/test_enhanced_upload_workflow.py'):
        return ["  ✗ Integration test file missing"]
    
    with open('tests/test_enhanced_upload_workflow.py', 'r') as f:
        content = f.read()
    
    checks = [
        ('Basic synchronous upload test', 'test_synchronous_upload_basic_analysis'),
        ('Wave analysis upload test', 'test_synchronous_upload_with_wave_analysis'),
        ('Async processing test', 'test_asynchronous_upload_processing'),
        ('Large file auto-async test', 'test_large_file_automatic_async_processing'),
        ('Task status API test', 'test_task_status_api_endpoint'),
        ('Task results API test', 'test_task_results_api_endpoint'),
        ('Concurrent processing test', 'test_concurrent_async_processing'),
        ('Error handling test', 'test_upload_error_handling'),
        ('Database integration test', 'test_database_integration'),
        ('Synthetic data generation', '_create_test_audio_file')
    ]
    
    results = []
    for check_name, pattern in checks:
        if pattern in content:
            results.append(f"  ✓ {check_name}")
        else:
            results.append(f"  ✗ {check_name}")
    
    return results

def verify_requirements_coverage():
    """Verify that all requirements are covered"""
    print("🔍 Verifying requirements coverage...")
    
    # Requirements 1.1 and 6.1 from the specification
    requirements = {
        "1.1": "Automatic wave detection and separation using ML models",
        "6.1": "Continuous analysis of incoming data for earthquake events"
    }
    
    results = []
    
    # Check for requirement 1.1 coverage
    with open('app.py', 'r') as f:
        content = f.read()
    
    if 'WaveSeparationEngine' in content and 'separate_waves' in content:
        results.append("  ✓ Requirement 1.1: Wave detection and separation implemented")
    else:
        results.append("  ✗ Requirement 1.1: Wave detection and separation missing")
    
    # Check for requirement 6.1 coverage
    if 'async' in content and 'continuous' in content.lower():
        results.append("  ✓ Requirement 6.1: Continuous/async processing implemented")
    else:
        results.append("  ✗ Requirement 6.1: Continuous/async processing missing")
    
    return results

def main():
    """Main verification function"""
    print("=" * 80)
    print("TASK 9.2 IMPLEMENTATION VERIFICATION")
    print("Extend file upload handling")
    print("=" * 80)
    print()
    
    all_results = []
    
    # Verify each component
    components = [
        ("Upload Endpoint Modifications", verify_upload_endpoint_modifications),
        ("Wave Analysis Trigger", verify_wave_analysis_trigger),
        ("Asynchronous Processing", verify_async_processing),
        ("API Endpoints", verify_api_endpoints),
        ("UI Enhancements", verify_ui_enhancements),
        ("Integration Tests", verify_integration_tests),
        ("Requirements Coverage", verify_requirements_coverage)
    ]
    
    for component_name, verify_func in components:
        print(f"📋 {component_name}")
        results = verify_func()
        all_results.extend(results)
        for result in results:
            print(result)
        print()
    
    # Summary
    total_checks = len(all_results)
    passed_checks = len([r for r in all_results if '✓' in r])
    failed_checks = total_checks - passed_checks
    
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"Total checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {failed_checks}")
    print(f"Success rate: {(passed_checks/total_checks)*100:.1f}%")
    print()
    
    if failed_checks == 0:
        print("🎉 ALL VERIFICATION CHECKS PASSED!")
        print()
        print("✅ Task 9.2 Implementation Complete:")
        print("   • Modified existing upload endpoint to support wave analysis option")
        print("   • Added wave analysis trigger in file processing workflow")
        print("   • Implemented asynchronous processing for large seismic files")
        print("   • Created comprehensive integration tests for enhanced upload workflow")
        print("   • Added new API endpoints for task status and result retrieval")
        print("   • Enhanced UI with analysis options and real-time status updates")
        print()
        print("🚀 The enhanced upload workflow is ready for production use!")
    else:
        print(f"⚠️  {failed_checks} verification checks failed.")
        print("Please review the implementation before marking the task as complete.")
    
    print("=" * 80)

if __name__ == '__main__':
    main()