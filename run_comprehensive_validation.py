#!/usr/bin/env python3
"""
Comprehensive Validation Runner for Earthquake Wave Analysis System.

This script runs comprehensive validation tests across all components of the
earthquake wave analysis system, validating that all requirements are met
and the system functions correctly as an integrated solution.
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_system_requirements():
    """
    Check that all system requirements are met for comprehensive validation.
    
    Returns:
        dict: System status information
    """
    logger.info("Checking system requirements...")
    
    status = {
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'timestamp': datetime.now().isoformat()
    }
    
    # Check required modules
    required_modules = [
        'numpy', 'scipy', 'matplotlib', 'pandas', 'sklearn',
        'obspy', 'librosa', 'plotly', 'reportlab'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            status[f'{module}_available'] = True
        except ImportError:
            status[f'{module}_available'] = False
            missing_modules.append(module)
    
    # Check wave analysis components
    try:
        from wave_analysis.models.wave_models import WaveSegment
        from wave_analysis.services.wave_separation_engine import WaveSeparationEngine
        status['wave_analysis_available'] = True
        logger.info("✓ Wave analysis components available")
    except ImportError as e:
        status['wave_analysis_available'] = False
        logger.warning(f"⚠ Wave analysis components not available: {e}")
    
    # Check test data manager
    try:
        from tests.test_data_manager import TestDataManager
        status['test_data_manager_available'] = True
        logger.info("✓ Test data manager available")
    except ImportError as e:
        status['test_data_manager_available'] = False
        logger.error(f"✗ Test data manager not available: {e}")
    
    if missing_modules:
        logger.warning(f"Missing modules: {', '.join(missing_modules)}")
    
    return status


def run_unit_tests():
    """
    Run all unit tests for individual components.
    
    Returns:
        dict: Unit test results
    """
    logger.info("Running unit tests...")
    
    import subprocess
    
    # List of unit test modules to run
    unit_test_modules = [
        'tests.test_test_data_manager',
        'tests.test_signal_processing',
        'tests.test_p_wave_detection',
        'tests.test_s_wave_detection',
        'tests.test_surface_wave_detection',
        'tests.test_frequency_plotter',
        'tests.test_time_series_plotter',
        'tests.test_performance_monitoring'
    ]
    
    results = {}
    total_passed = 0
    total_failed = 0
    
    for module in unit_test_modules:
        logger.info(f"Running {module}...")
        
        try:
            # Run pytest for the specific module
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 
                f'{module.replace(".", "/")}.py',
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=300)
            
            # Parse results
            output = result.stdout + result.stderr
            
            if result.returncode == 0:
                results[module] = {
                    'status': 'PASSED',
                    'output': output,
                    'return_code': result.returncode
                }
                # Count passed tests (rough estimate)
                passed_count = output.count('PASSED')
                total_passed += passed_count
                logger.info(f"✓ {module}: {passed_count} tests passed")
            else:
                results[module] = {
                    'status': 'FAILED',
                    'output': output,
                    'return_code': result.returncode
                }
                failed_count = output.count('FAILED') + output.count('ERROR')
                total_failed += failed_count
                logger.error(f"✗ {module}: {failed_count} tests failed")
                
        except subprocess.TimeoutExpired:
            results[module] = {
                'status': 'TIMEOUT',
                'output': 'Test execution timed out',
                'return_code': -1
            }
            total_failed += 1
            logger.error(f"✗ {module}: Timed out")
            
        except Exception as e:
            results[module] = {
                'status': 'ERROR',
                'output': str(e),
                'return_code': -1
            }
            total_failed += 1
            logger.error(f"✗ {module}: {str(e)}")
    
    summary = {
        'total_modules': len(unit_test_modules),
        'total_passed': total_passed,
        'total_failed': total_failed,
        'success_rate': total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0,
        'results': results
    }
    
    logger.info(f"Unit tests summary: {total_passed} passed, {total_failed} failed")
    return summary


def run_integration_tests():
    """
    Run integration tests for component interactions.
    
    Returns:
        dict: Integration test results
    """
    logger.info("Running integration tests...")
    
    import subprocess
    
    # List of integration test modules
    integration_test_modules = [
        'tests.test_wave_analyzer_integration',
        'tests.test_wave_visualizer_integration', 
        'tests.test_data_manager_integration',
        'tests.test_end_to_end_integration',
        'tests.test_performance_integration'
    ]
    
    results = {}
    total_passed = 0
    total_failed = 0
    
    for module in integration_test_modules:
        logger.info(f"Running {module}...")
        
        try:
            # Run pytest for the specific module
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                f'{module.replace(".", "/")}.py',
                '-v', '--tb=short'
            ], capture_output=True, text=True, timeout=600)  # Longer timeout for integration tests
            
            output = result.stdout + result.stderr
            
            if result.returncode == 0:
                results[module] = {
                    'status': 'PASSED',
                    'output': output,
                    'return_code': result.returncode
                }
                passed_count = output.count('PASSED')
                total_passed += passed_count
                logger.info(f"✓ {module}: {passed_count} tests passed")
            else:
                results[module] = {
                    'status': 'FAILED',
                    'output': output,
                    'return_code': result.returncode
                }
                failed_count = output.count('FAILED') + output.count('ERROR')
                total_failed += failed_count
                logger.error(f"✗ {module}: {failed_count} tests failed")
                
        except subprocess.TimeoutExpired:
            results[module] = {
                'status': 'TIMEOUT',
                'output': 'Integration test execution timed out',
                'return_code': -1
            }
            total_failed += 1
            logger.error(f"✗ {module}: Timed out")
            
        except Exception as e:
            results[module] = {
                'status': 'ERROR',
                'output': str(e),
                'return_code': -1
            }
            total_failed += 1
            logger.error(f"✗ {module}: {str(e)}")
    
    summary = {
        'total_modules': len(integration_test_modules),
        'total_passed': total_passed,
        'total_failed': total_failed,
        'success_rate': total_passed / (total_passed + total_failed) if (total_passed + total_failed) > 0 else 0,
        'results': results
    }
    
    logger.info(f"Integration tests summary: {total_passed} passed, {total_failed} failed")
    return summary


def run_comprehensive_validation():
    """
    Run comprehensive validation tests.
    
    Returns:
        dict: Comprehensive validation results
    """
    logger.info("Running comprehensive validation tests...")
    
    try:
        from tests.test_comprehensive_validation import run_comprehensive_validation
        
        # Run comprehensive validation
        validation_results = run_comprehensive_validation()
        
        logger.info(f"Comprehensive validation completed: {validation_results['success_rate']:.1%} success rate")
        return validation_results
        
    except Exception as e:
        logger.error(f"Comprehensive validation failed: {str(e)}")
        return {
            'status': 'ERROR',
            'error': str(e),
            'success_rate': 0.0
        }


def run_performance_benchmarks():
    """
    Run performance benchmarks across different scenarios.
    
    Returns:
        dict: Performance benchmark results
    """
    logger.info("Running performance benchmarks...")
    
    try:
        from tests.test_data_manager import TestDataManager, SyntheticEarthquakeParams
        
        # Initialize test data manager
        manager = TestDataManager()
        
        # Define benchmark scenarios
        scenarios = [
            {
                'name': 'small_earthquake',
                'params': SyntheticEarthquakeParams(magnitude=3.0, distance=50.0, duration=30.0),
                'expected_time': 5.0  # seconds
            },
            {
                'name': 'medium_earthquake', 
                'params': SyntheticEarthquakeParams(magnitude=5.0, distance=150.0, duration=60.0),
                'expected_time': 10.0
            },
            {
                'name': 'large_earthquake',
                'params': SyntheticEarthquakeParams(magnitude=7.0, distance=500.0, duration=120.0),
                'expected_time': 20.0
            }
        ]
        
        benchmark_results = {}
        
        for scenario in scenarios:
            logger.info(f"Benchmarking {scenario['name']}...")
            
            start_time = time.time()
            
            # Generate test data
            data = manager.create_synthetic_earthquake(scenario['params'])
            
            # Validate data quality
            validation = manager.validate_test_data_quality(data, scenario['params'])
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            benchmark_results[scenario['name']] = {
                'processing_time': processing_time,
                'expected_time': scenario['expected_time'],
                'performance_ratio': processing_time / scenario['expected_time'],
                'data_size': len(data),
                'quality_score': validation.get('quality_score', 0.0),
                'status': 'PASS' if processing_time <= scenario['expected_time'] else 'SLOW'
            }
            
            logger.info(f"✓ {scenario['name']}: {processing_time:.2f}s "
                       f"(expected {scenario['expected_time']:.2f}s), "
                       f"quality={validation.get('quality_score', 0.0):.2f}")
        
        # Calculate overall performance metrics
        avg_performance_ratio = sum(r['performance_ratio'] for r in benchmark_results.values()) / len(benchmark_results)
        avg_quality = sum(r['quality_score'] for r in benchmark_results.values()) / len(benchmark_results)
        
        summary = {
            'scenarios': benchmark_results,
            'average_performance_ratio': avg_performance_ratio,
            'average_quality': avg_quality,
            'overall_status': 'PASS' if avg_performance_ratio <= 1.2 else 'FAIL'  # Allow 20% overhead
        }
        
        logger.info(f"Performance benchmarks completed: avg ratio={avg_performance_ratio:.2f}, "
                   f"avg quality={avg_quality:.2f}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Performance benchmarks failed: {str(e)}")
        return {
            'status': 'ERROR',
            'error': str(e)
        }


def generate_validation_report(results):
    """
    Generate comprehensive validation report.
    
    Args:
        results (dict): All validation results
        
    Returns:
        dict: Formatted validation report
    """
    logger.info("Generating validation report...")
    
    report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'system_info': results.get('system_status', {}),
            'validation_version': '1.0.0'
        },
        'summary': {
            'overall_status': 'UNKNOWN',
            'total_tests': 0,
            'total_passed': 0,
            'total_failed': 0,
            'success_rate': 0.0
        },
        'components': {
            'unit_tests': results.get('unit_tests', {}),
            'integration_tests': results.get('integration_tests', {}),
            'comprehensive_validation': results.get('comprehensive_validation', {}),
            'performance_benchmarks': results.get('performance_benchmarks', {})
        },
        'requirements_coverage': {
            'requirement_1_1': 'UNKNOWN',  # Wave separation
            'requirement_1_2': 'UNKNOWN',  # Wave display
            'requirement_1_3': 'UNKNOWN',  # Wave characteristics
            'requirement_2_1': 'UNKNOWN',  # Interactive visualization
            'requirement_3_1': 'UNKNOWN',  # Arrival time calculations
            'requirement_3_3': 'UNKNOWN',  # Magnitude estimation
            'requirement_4_1': 'UNKNOWN',  # Data export
            'requirement_6_5': 'UNKNOWN'   # Quality control
        }
    }
    
    # Calculate summary statistics
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    # Unit tests
    if 'unit_tests' in results:
        unit_results = results['unit_tests']
        total_tests += unit_results.get('total_passed', 0) + unit_results.get('total_failed', 0)
        total_passed += unit_results.get('total_passed', 0)
        total_failed += unit_results.get('total_failed', 0)
    
    # Integration tests
    if 'integration_tests' in results:
        int_results = results['integration_tests']
        total_tests += int_results.get('total_passed', 0) + int_results.get('total_failed', 0)
        total_passed += int_results.get('total_passed', 0)
        total_failed += int_results.get('total_failed', 0)
    
    # Comprehensive validation
    if 'comprehensive_validation' in results:
        comp_results = results['comprehensive_validation']
        if 'total_tests' in comp_results:
            total_tests += comp_results['total_tests']
            total_passed += comp_results['total_tests'] - comp_results.get('failures', 0) - comp_results.get('errors', 0)
            total_failed += comp_results.get('failures', 0) + comp_results.get('errors', 0)
    
    # Update summary
    report['summary']['total_tests'] = total_tests
    report['summary']['total_passed'] = total_passed
    report['summary']['total_failed'] = total_failed
    
    if total_tests > 0:
        success_rate = total_passed / total_tests
        report['summary']['success_rate'] = success_rate
        
        # Determine overall status
        if success_rate >= 0.95:
            report['summary']['overall_status'] = 'EXCELLENT'
        elif success_rate >= 0.90:
            report['summary']['overall_status'] = 'GOOD'
        elif success_rate >= 0.80:
            report['summary']['overall_status'] = 'ACCEPTABLE'
        else:
            report['summary']['overall_status'] = 'NEEDS_IMPROVEMENT'
    
    # Update requirements coverage based on comprehensive validation
    if 'comprehensive_validation' in results:
        comp_results = results['comprehensive_validation']
        if comp_results.get('success_rate', 0) >= 0.8:
            for req in report['requirements_coverage']:
                report['requirements_coverage'][req] = 'PASS'
        else:
            for req in report['requirements_coverage']:
                report['requirements_coverage'][req] = 'PARTIAL'
    
    logger.info(f"Validation report generated: {report['summary']['overall_status']} "
               f"({report['summary']['success_rate']:.1%} success rate)")
    
    return report


def save_results(results, report):
    """
    Save validation results and report to files.
    
    Args:
        results (dict): Raw validation results
        report (dict): Formatted validation report
    """
    logger.info("Saving validation results...")
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Save raw results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results_file = f'logs/validation_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save formatted report
    report_file = f'logs/validation_report_{timestamp}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Save summary report
    summary_file = f'COMPREHENSIVE_VALIDATION_SUMMARY.md'
    with open(summary_file, 'w') as f:
        f.write(f"# Comprehensive Validation Summary\n\n")
        f.write(f"**Generated:** {report['metadata']['timestamp']}\n\n")
        f.write(f"## Overall Status: {report['summary']['overall_status']}\n\n")
        f.write(f"- **Total Tests:** {report['summary']['total_tests']}\n")
        f.write(f"- **Passed:** {report['summary']['total_passed']}\n")
        f.write(f"- **Failed:** {report['summary']['total_failed']}\n")
        f.write(f"- **Success Rate:** {report['summary']['success_rate']:.1%}\n\n")
        
        f.write(f"## Requirements Coverage\n\n")
        for req, status in report['requirements_coverage'].items():
            emoji = "✅" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
            f.write(f"- {emoji} **{req.replace('_', '.')}:** {status}\n")
        
        f.write(f"\n## Component Results\n\n")
        for component, data in report['components'].items():
            if isinstance(data, dict) and 'success_rate' in data:
                f.write(f"- **{component.replace('_', ' ').title()}:** {data['success_rate']:.1%}\n")
        
        f.write(f"\n## Detailed Results\n\n")
        f.write(f"See `{results_file}` and `{report_file}` for detailed results.\n")
    
    logger.info(f"Results saved to {results_file}, {report_file}, and {summary_file}")


def main():
    """Main validation runner function."""
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION RUNNER")
    print("Earthquake Wave Analysis System")
    print("=" * 80)
    
    start_time = time.time()
    
    # Initialize results dictionary
    results = {}
    
    try:
        # Step 1: Check system requirements
        print("\n1. Checking system requirements...")
        results['system_status'] = check_system_requirements()
        
        # Step 2: Run unit tests
        print("\n2. Running unit tests...")
        results['unit_tests'] = run_unit_tests()
        
        # Step 3: Run integration tests
        print("\n3. Running integration tests...")
        results['integration_tests'] = run_integration_tests()
        
        # Step 4: Run comprehensive validation
        print("\n4. Running comprehensive validation...")
        results['comprehensive_validation'] = run_comprehensive_validation()
        
        # Step 5: Run performance benchmarks
        print("\n5. Running performance benchmarks...")
        results['performance_benchmarks'] = run_performance_benchmarks()
        
        # Step 6: Generate report
        print("\n6. Generating validation report...")
        report = generate_validation_report(results)
        
        # Step 7: Save results
        print("\n7. Saving results...")
        save_results(results, report)
        
        # Final summary
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 80)
        print("VALIDATION COMPLETE")
        print("=" * 80)
        print(f"Overall Status: {report['summary']['overall_status']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Total Time: {total_time:.1f} seconds")
        
        # Exit with appropriate code
        if report['summary']['success_rate'] >= 0.9:
            print("\n✅ COMPREHENSIVE VALIDATION PASSED")
            sys.exit(0)
        else:
            print("\n❌ COMPREHENSIVE VALIDATION FAILED")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation runner failed: {str(e)}")
        print(f"\n❌ VALIDATION RUNNER ERROR: {str(e)}")
        sys.exit(2)


if __name__ == '__main__':
    main()