import os
from pathlib import Path

def check_app_dependencies():
    required_files = {
        'app.py': 'Main application file',
        'earthquake_model.joblib': 'ML model file',
        'earthquake_scaler.joblib': 'Data scaler file',
        '.env': 'Environment variables file',
        'static/icon.png': 'Application icon',
        'templates/realtime.html': 'Realtime monitoring template',
        'templates/error.html': 'Error page template',
        'historic_analysis.py': 'Historic analysis module',
        'location_predictor.py': 'Location prediction module',
        'risk_assessment.py': 'Risk assessment module',
        'emergency_response.py': 'Emergency response module',
        'education_content.py': 'Educational content module',
        'seismic_utils.py': 'Seismic utilities module',
        'usgs_monitor.py': 'USGS monitoring module'
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not Path(file_path).exists():
            missing_files.append(f"Missing {description}: {file_path}")
    
    if missing_files:
        print("Missing required files:")
        for error in missing_files:
            print(f"- {error}")
        return False
    
    print("All required files present!")
    return True

if __name__ == "__main__":
    check_app_dependencies()
