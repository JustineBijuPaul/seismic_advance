import sys
import os
import logging
import importlib.util
from pathlib import Path
import tempfile
import requests
import json
from typing import List, Dict, Any
import platform
import shutil  # Add missing import
import codecs  # For handling Unicode
import locale  # For system encoding

# Configure logging with UTF-8 encoding
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler('debug.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Update check marks to ASCII
SUCCESS_MARK = '[OK]'  # Instead of '✓'
FAIL_MARK = '[FAIL]'  # Instead of '×'

class SystemCheck:
    @staticmethod
    def check_python_version() -> bool:
        """Check Python version and implementation"""
        logger.info("\nRunning Python Environment check...")
        try:
            logger.info(f"Python Version: {sys.version}")
            logger.info(f"Python Implementation: {platform.python_implementation()}")
            logger.info(f"Platform: {platform.platform()}")
            
            if sys.version_info >= (3, 7):
                logger.info(f"{SUCCESS_MARK} Python Version check passed!")
                return True
            logger.error(f"{FAIL_MARK} Python Version must be 3.7 or higher!")
            return False
        except Exception as e:
            logger.error(f"{FAIL_MARK} Python version check failed: {e}")
            return False

    @staticmethod
    def check_disk_space(min_space_mb: int = 500) -> bool:
        """Check available disk space"""
        logger.info("\nRunning Disk Space check...")
        try:
            total, used, free = (os.statvfs(os.getcwd()) if os.name == 'posix' 
                               else shutil.disk_usage(os.getcwd()))
            free_mb = free // (1024 * 1024)
            logger.info(f"Available disk space: {free_mb}MB")
            
            if free_mb >= min_space_mb:
                logger.info(f"{SUCCESS_MARK} Sufficient disk space available")
                return True
            logger.error(f"{FAIL_MARK} Insufficient disk space. Need at least {min_space_mb}")
            return False
        except Exception as e:
            logger.error(f"{FAIL_MARK} Disk space check failed: {e}")
            return False

class PackageCheck:
    @staticmethod
    def check_required_packages() -> bool:
        """Check required packages with detailed version info"""
        logger.info("\nRunning Required Packages check...")
        required_packages = {
            'flask': 'flask',
            'flask_socketio': 'flask-socketio',
            'numpy': 'numpy',
            'librosa': 'librosa',
            'sklearn': 'scikit-learn',
            'joblib': 'joblib',
            'whitenoise': 'whitenoise',
            'pymongo': 'pymongo',
            'dotenv': 'python-dotenv',  # Changed from python_dotenv to dotenv
            'obspy': 'obspy',
            'soundfile': 'soundfile',
            'sounddevice': 'sounddevice',
            'geopy': 'geopy',
            'apscheduler': 'APScheduler',
            'pandas': 'pandas',
            'scipy': 'scipy',
            'matplotlib': 'matplotlib',
            'requests': 'requests',
            'pillow': 'Pillow'
        }
        
        # Remove built-in modules from check
        for pkg in ['sqlite3', 'json', 'xml']:
            if pkg in required_packages:
                del required_packages[pkg]
        
        missing_packages = []
        version_info = {}
        
        for import_name, pip_name in required_packages.items():
            try:
                if import_name == 'dotenv':
                    try:
                        from dotenv.main import __version__
                        version_info[pip_name] = __version__
                    except ImportError:
                        from dotenv import __version__
                        version_info[pip_name] = __version__
                    logger.info(f"[+] {pip_name} v{version_info[pip_name]} is installed")
                elif import_name == 'sklearn':
                    import sklearn
                    module = importlib.import_module(import_name)
                    version = getattr(module, '__version__', 'unknown')
                    version_info[pip_name] = version
                
                logger.info(f"[+] {pip_name} v{version_info[pip_name]} is installed")
            except ImportError as e:
                logger.error(f"[-] {pip_name} is missing: {e}")
                missing_packages.append(pip_name)
            except Exception as e:
                logger.error(f"[-] Error checking {pip_name}: {e}")
                missing_packages.append(pip_name)

        if missing_packages:
            logger.error("\nRequired Packages check failed!")
            logger.info("Install missing packages with:")
            logger.info(f"pip install {' '.join(missing_packages)}")
            return False
        
        logger.info("\nPackage versions:")
        for package, version in version_info.items():
            logger.info(f"{package}: {version}")
        
        logger.info(f"{SUCCESS_MARK} All required packages are installed!")
        return True

# Update FileSystemCheck to create missing directories
class FileSystemCheck:
    @staticmethod
    def check_required_files() -> bool:
        """Check required files and their permissions"""
        logger.info("\nRunning Required Files check...")
        
        # Create required directories first
        directories = ['templates', 'static']
        for directory in directories:
            path = Path(directory)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"{SUCCESS_MARK} Created directory: {directory}")
                except Exception as e:
                    logger.error(f"{FAIL_MARK} Failed to create directory {directory}: {e}")
                    return False
        
        required_files = {
            'app.py': {'type': 'file', 'permissions': ['read']},
            'earthquake_model.joblib': {'type': 'file', 'permissions': ['read']},
            'earthquake_scaler.joblib': {'type': 'file', 'permissions': ['read']},
            '.env': {'type': 'file', 'permissions': ['read']},
            'templates/realtime.html': {'type': 'file', 'permissions': ['read']},
            'templates/error.html': {'type': 'file', 'permissions': ['read']},
            'static/icon.png': {'type': 'file', 'permissions': ['read']}
        }
        
        missing_files = []
        permission_issues = []
        
        for filepath, requirements in required_files.items():
            path = Path(filepath)
            try:
                if not path.exists():
                    logger.error(f"[-] Missing: {filepath}")
                    missing_files.append(filepath)
                    continue
                
                # Check file type
                if requirements['type'] == 'directory' and not path.is_dir():
                    logger.error(f"[-] {filepath} should be a directory")
                    missing_files.append(filepath)
                elif requirements['type'] == 'file' and not path.is_file():
                    logger.error(f"[-] {filepath} should be a file")
                    missing_files.append(filepath)
                
                # Check permissions
                try:
                    if 'read' in requirements['permissions']:
                        with open(path, 'r') if path.is_file() else path.iterdir():
                            pass
                    if 'write' in requirements['permissions']:
                        if path.is_file():
                            with open(path, 'a'): pass
                        else:
                            with tempfile.NamedTemporaryFile(dir=path): pass
                    logger.info(f"[+] Found with correct permissions: {filepath}")
                except (IOError, PermissionError) as e:
                    logger.error(f"[-] Permission issue with {filepath}: {e}")
                    permission_issues.append(filepath)
                
            except Exception as e:
                logger.error(f"[-] Error checking {filepath}: {e}")
                missing_files.append(filepath)
        
        if missing_files or permission_issues:
            if missing_files:
                logger.error("\nMissing files/directories:")
                for file in missing_files:
                    logger.error(f"- {file}")
            if permission_issues:
                logger.error("\nPermission issues:")
                for file in permission_issues:
                    logger.error(f"- {file}")
            return False
        
        logger.info(f"{SUCCESS_MARK} All required files present with correct permissions!")
        return True

class NetworkCheck:
    @staticmethod
    def check_connectivity() -> bool:
        """Check internet connectivity and API access"""
        logger.info("\nRunning Network Connectivity check...")
        endpoints = {
            'USGS API': 'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&limit=1',
            'MongoDB Atlas': 'mongodb+srv',
            'OpenStreetMap': 'https://tile.openstreetmap.org/0/0/0.png'
        }
        
        failures = []
        for service, url in endpoints.items():
            try:
                if 'mongodb+srv' in url:
                    # MongoDB connection check handled separately
                    continue
                
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"[+] Successfully connected to {service}")
                else:
                    logger.error(f"[-] Failed to connect to {service}: Status {response.status_code}")
                    failures.append(service)
            except requests.exceptions.RequestException as e:
                logger.error(f"[-] Failed to connect to {service}: {e}")
                failures.append(service)
        
        if failures:
            logger.error("\nNetwork connectivity issues detected!")
            return False
            
        logger.info(f"{SUCCESS_MARK} Network connectivity check passed!")
        return True

class DatabaseCheck:
    @staticmethod
    def check_mongodb_connection() -> bool:
        """Comprehensive MongoDB connection and operations check"""
        logger.info("\nRunning MongoDB Connection check...")
        try:
            from pymongo import MongoClient
            from dotenv import load_dotenv
            load_dotenv()
            
            mongo_url = os.getenv('MONGO_URL')
            if not mongo_url:
                logger.error(f"{FAIL_MARK} MongoDB URL not found in environment variables")
                return False
            
            # Test connection
            client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
            client.server_info()
            
            # Test basic operations
            db = client['test_db']
            collection = db['test_collection']
            
            # Insert
            result = collection.insert_one({'test': 'data'})
            if not result.inserted_id:
                raise Exception("Failed to insert test document")
            
            # Query
            if not collection.find_one({'test': 'data'}):
                raise Exception("Failed to query test document")
            
            # Delete
            if not collection.delete_one({'test': 'data'}).deleted_count:
                raise Exception("Failed to delete test document")
            
            logger.info(f"{SUCCESS_MARK} MongoDB connection and operations successful!")
            return True
            
        except Exception as e:
            logger.error(f"{FAIL_MARK} MongoDB check failed: {e}")
            return False
        finally:
            if 'client' in locals():
                client.close()

def run_all_checks() -> bool:
    """Run all system checks"""
    logger.info("Starting comprehensive system checks...")
    
    checkers = [
        (SystemCheck.check_python_version, "Python Environment"),
        (SystemCheck.check_disk_space, "Disk Space"),
        (PackageCheck.check_required_packages, "Required Packages"),
        (FileSystemCheck.check_required_files, "File System"),
        (NetworkCheck.check_connectivity, "Network Connectivity"),
        (DatabaseCheck.check_mongodb_connection, "MongoDB Connection")
    ]
    
    failed_checks = []
    for check_func, check_name in checkers:
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            logger.error(f"Error during {check_name} check: {e}")
            failed_checks.append(check_name)
    
    if failed_checks:
        logger.error("\nThe following checks failed:")
        for check in failed_checks:
            logger.error(f"- {check}")
        return False
    
    logger.info(f"{SUCCESS_MARK} All system checks passed successfully!")
    return True

if __name__ == "__main__":
    try:
        if run_all_checks():
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Critical error during system checks: {e}")
        sys.exit(1)
