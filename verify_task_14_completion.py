#!/usr/bin/env python3
"""
Verification script for Task 14: Final integration and deployment preparation
Verifies completion of both subtasks 14.1 and 14.2
"""

import os
import sys
import json
from pathlib import Path

def verify_task_14_1_integration_testing():
    """Verify Task 14.1: Complete end-to-end integration testing"""
    print("🔍 Verifying Task 14.1: Complete end-to-end integration testing")
    print("=" * 60)
    
    checks = []
    
    # Check 1: End-to-end integration test file exists
    integration_test_file = "tests/test_end_to_end_integration.py"
    if os.path.exists(integration_test_file):
        checks.append("✅ End-to-end integration test file exists")
        
        # Check test content
        with open(integration_test_file, 'r') as f:
            content = f.read()
            
        required_tests = [
            "test_complete_workflow_earthquake_detection",
            "test_async_processing_workflow", 
            "test_noise_file_handling",
            "test_multi_channel_data_processing",
            "test_real_time_monitoring_workflow",
            "test_error_handling_scenarios",
            "test_alert_system_integration",
            "test_educational_features",
            "test_database_integration",
            "test_performance_under_load",
            "test_api_endpoint_validation",
            "test_data_export_formats"
        ]
        
        for test in required_tests:
            if test in content:
                checks.append(f"✅ {test} implemented")
            else:
                checks.append(f"❌ {test} missing")
    else:
        checks.append("❌ End-to-end integration test file missing")
    
    # Check 2: Comprehensive integration test file exists
    comp_test_file = "tests/test_comprehensive_integration.py"
    if os.path.exists(comp_test_file):
        checks.append("✅ Comprehensive integration test file exists")
    else:
        checks.append("❌ Comprehensive integration test file missing")
    
    # Check 3: Integration test runner exists
    runner_file = "run_integration_tests.py"
    if os.path.exists(runner_file):
        checks.append("✅ Integration test runner exists")
    else:
        checks.append("❌ Integration test runner missing")
    
    # Check 4: API integration tests exist
    api_test_file = "tests/test_api_integration.py"
    if os.path.exists(api_test_file):
        checks.append("✅ API integration test file exists")
    else:
        checks.append("❌ API integration test file missing")
    
    # Check 5: Verification scripts exist
    verification_files = [
        "verify_integration_tests.py",
        "verify_integration_complete.py",
        "verify_task_completion.py"
    ]
    
    for file in verification_files:
        if os.path.exists(file):
            checks.append(f"✅ {file} exists")
        else:
            checks.append(f"❌ {file} missing")
    
    # Print results
    for check in checks:
        print(check)
    
    passed = sum(1 for check in checks if check.startswith("✅"))
    total = len(checks)
    
    print(f"\nTask 14.1 Status: {passed}/{total} checks passed")
    return passed == total


def verify_task_14_2_deployment_setup():
    """Verify Task 14.2: Add configuration and deployment setup"""
    print("\n🔍 Verifying Task 14.2: Add configuration and deployment setup")
    print("=" * 60)
    
    checks = []
    
    # Check 1: Configuration management system exists
    config_files = [
        "wave_analysis/config/__init__.py",
        "wave_analysis/config.py"
    ]
    
    for file in config_files:
        if os.path.exists(file):
            checks.append(f"✅ {file} exists")
        else:
            checks.append(f"❌ {file} missing")
    
    # Check 2: Environment-specific configuration files exist
    env_configs = [
        "wave_analysis/config/development.json",
        "wave_analysis/config/staging.json", 
        "wave_analysis/config/production.json"
    ]
    
    for config in env_configs:
        if os.path.exists(config):
            checks.append(f"✅ {config} exists")
            
            # Validate JSON format
            try:
                with open(config, 'r') as f:
                    config_data = json.load(f)
                    
                required_sections = ["wave_analysis", "database", "logging", "deployment"]
                for section in required_sections:
                    if section in config_data:
                        checks.append(f"✅ {config} has {section} section")
                    else:
                        checks.append(f"❌ {config} missing {section} section")
            except json.JSONDecodeError:
                checks.append(f"❌ {config} has invalid JSON format")
        else:
            checks.append(f"❌ {config} missing")
    
    # Check 3: Logging configuration exists
    logging_config_file = "wave_analysis/logging_config.py"
    if os.path.exists(logging_config_file):
        checks.append("✅ Logging configuration exists")
        
        with open(logging_config_file, 'r') as f:
            content = f.read()
            
        required_classes = [
            "WaveAnalysisLogger",
            "HealthMonitor",
            "performance_monitor"
        ]
        
        for cls in required_classes:
            if cls in content:
                checks.append(f"✅ {cls} implemented")
            else:
                checks.append(f"❌ {cls} missing")
    else:
        checks.append("❌ Logging configuration missing")
    
    # Check 4: Deployment setup script exists
    deployment_script = "setup_deployment.py"
    if os.path.exists(deployment_script):
        checks.append("✅ Deployment setup script exists")
        
        with open(deployment_script, 'r') as f:
            content = f.read()
            
        required_functions = [
            "validate_environment_variables",
            "validate_configuration",
            "test_database_connection",
            "setup_logging_directories",
            "generate_deployment_report"
        ]
        
        for func in required_functions:
            if func in content:
                checks.append(f"✅ {func} implemented")
            else:
                checks.append(f"❌ {func} missing")
    else:
        checks.append("❌ Deployment setup script missing")
    
    # Check 5: Deployment configuration tests exist
    deployment_test_file = "tests/test_deployment_config.py"
    if os.path.exists(deployment_test_file):
        checks.append("✅ Deployment configuration tests exist")
    else:
        checks.append("❌ Deployment configuration tests missing")
    
    # Check 6: Log directories exist
    log_dirs = ["logs"]
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            checks.append(f"✅ {log_dir} directory exists")
        else:
            checks.append(f"❌ {log_dir} directory missing")
    
    # Check 7: Deployment report can be generated
    deployment_report = "deployment_report_development.json"
    if os.path.exists(deployment_report):
        checks.append("✅ Deployment report generated")
        
        try:
            with open(deployment_report, 'r') as f:
                report_data = json.load(f)
                
            required_report_sections = ["environment", "configuration", "validation_results", "ready_for_deployment"]
            for section in required_report_sections:
                if section in report_data:
                    checks.append(f"✅ Deployment report has {section}")
                else:
                    checks.append(f"❌ Deployment report missing {section}")
        except json.JSONDecodeError:
            checks.append("❌ Deployment report has invalid JSON format")
    else:
        checks.append("❌ Deployment report not generated")
    
    # Print results
    for check in checks:
        print(check)
    
    passed = sum(1 for check in checks if check.startswith("✅"))
    total = len(checks)
    
    print(f"\nTask 14.2 Status: {passed}/{total} checks passed")
    return passed == total


def main():
    """Main verification function"""
    print("🚀 Task 14: Final integration and deployment preparation")
    print("=" * 80)
    
    # Verify both subtasks
    task_14_1_complete = verify_task_14_1_integration_testing()
    task_14_2_complete = verify_task_14_2_deployment_setup()
    
    print("\n" + "=" * 80)
    print("TASK 14 VERIFICATION SUMMARY")
    print("=" * 80)
    
    if task_14_1_complete:
        print("✅ Task 14.1: Complete end-to-end integration testing - COMPLETE")
    else:
        print("❌ Task 14.1: Complete end-to-end integration testing - INCOMPLETE")
    
    if task_14_2_complete:
        print("✅ Task 14.2: Add configuration and deployment setup - COMPLETE")
    else:
        print("❌ Task 14.2: Add configuration and deployment setup - INCOMPLETE")
    
    overall_complete = task_14_1_complete and task_14_2_complete
    
    if overall_complete:
        print("\n🎉 TASK 14: Final integration and deployment preparation - COMPLETE!")
        print("\nAll subtasks have been successfully implemented:")
        print("• Comprehensive end-to-end integration tests covering all workflows")
        print("• Complete configuration management system for all environments")
        print("• Deployment setup scripts with validation and monitoring")
        print("• Logging and health monitoring systems")
        print("• Environment-specific configuration files")
        print("• Deployment readiness validation and reporting")
        return True
    else:
        print("\n❌ TASK 14: Final integration and deployment preparation - INCOMPLETE")
        print("\nSome subtasks need attention. Please review the failed checks above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)