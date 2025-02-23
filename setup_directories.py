import os
from pathlib import Path

def setup_project_structure():
    """Create project directory structure"""
    directories = [
        'templates',
        'static',
        'static/css',
        'static/js',
        'static/images'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    setup_project_structure()
