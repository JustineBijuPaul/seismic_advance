import numpy as np
from scipy.interpolate import griddata
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

@dataclass
class BuildingType:
    category: str
    vulnerability_index: float
    height_factor: float
    age_factor: float
    material_factor: float

BUILDING_TYPES = {
    'RC_HIGH': BuildingType('Reinforced Concrete High-Rise', 0.7, 1.2, 1.0, 0.9),
    'RC_LOW': BuildingType('Reinforced Concrete Low-Rise', 0.6, 1.0, 1.0, 0.9),
    'MASONRY': BuildingType('Masonry', 0.8, 0.9, 1.2, 1.1),
    'WOOD': BuildingType('Wooden Structure', 0.5, 0.8, 1.1, 0.7),
    'STEEL': BuildingType('Steel Frame', 0.4, 1.1, 0.9, 0.8)
}

SOIL_TYPES = {
    'ROCK': {'amp_factor': 1.0, 'description': 'Hard Rock', 'vs30': 1500},
    'STIFF': {'amp_factor': 1.2, 'description': 'Stiff Soil', 'vs30': 760},
    'MEDIUM': {'amp_factor': 1.4, 'description': 'Medium Soil', 'vs30': 360},
    'SOFT': {'amp_factor': 1.6, 'description': 'Soft Soil', 'vs30': 180},
    'VERY_SOFT': {'amp_factor': 2.0, 'description': 'Very Soft Soil', 'vs30': 150}
}

def calculate_building_vulnerability(
    building_type: str,
    age: int,
    height: float,
    condition: str
) -> dict:
    """Calculate building vulnerability score"""
    if building_type not in BUILDING_TYPES:
        raise ValueError(f"Unknown building type: {building_type}")
    
    bt = BUILDING_TYPES[building_type]
    
    # Calculate age factor
    age_factor = bt.age_factor * (1 + (age / 100))
    
    # Adjust height factor
    height_factor = bt.height_factor * (1 + (height / 50))
    
    # Condition multiplier
    condition_multiplier = {
        'excellent': 0.8,
        'good': 1.0,
        'fair': 1.2,
        'poor': 1.5
    }.get(condition.lower(), 1.0)
    
    # Calculate final vulnerability score
    base_score = bt.vulnerability_index
    final_score = base_score * age_factor * height_factor * condition_multiplier
    
    # Normalize score to 0-100 range
    normalized_score = min(100, max(0, final_score * 100))
    
    return {
        'building_type': bt.category,
        'vulnerability_score': normalized_score,
        'risk_level': get_risk_level(normalized_score),
        'factors': {
            'age': age_factor,
            'height': height_factor,
            'condition': condition_multiplier
        }
    }

def analyze_soil_amplification(
    soil_type: str,
    vs30: float = None,
    water_table_depth: float = None
) -> dict:
    """Analyze soil amplification factors"""
    if soil_type not in SOIL_TYPES:
        raise ValueError(f"Unknown soil type: {soil_type}")
    
    soil_info = SOIL_TYPES[soil_type]
    
    # Use provided vs30 or default
    actual_vs30 = vs30 if vs30 is not None else soil_info['vs30']
    
    # Calculate amplification factor
    amp_factor = soil_info['amp_factor']
    
    # Adjust for water table if provided
    if water_table_depth is not None:
        if water_table_depth < 5:
            amp_factor *= 1.2
        elif water_table_depth < 10:
            amp_factor *= 1.1
    
    return {
        'soil_type': soil_info['description'],
        'amplification_factor': amp_factor,
        'vs30': actual_vs30,
        'hazard_increase': f"{((amp_factor - 1) * 100):.1f}%"
    }

def generate_hazard_map(
    locations: List[Tuple[float, float]],
    values: List[float],
    grid_size: int = 100
) -> dict:
    """Generate seismic hazard map data"""
    if not locations or not values:
        raise ValueError("Empty location or value data")
    
    # Create grid
    lats, lons = zip(*locations)
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    grid_lat = np.linspace(lat_min, lat_max, grid_size)
    grid_lon = np.linspace(lon_min, lon_max, grid_size)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Interpolate values
    grid_values = griddata(
        locations,
        values,
        (grid_lon, grid_lat),
        method='cubic',
        fill_value=0
    )
    
    return {
        'grid': {
            'latitudes': grid_lat.tolist(),
            'longitudes': grid_lon.tolist(),
            'values': grid_values.tolist()
        },
        'bounds': {
            'lat_min': lat_min,
            'lat_max': lat_max,
            'lon_min': lon_min,
            'lon_max': lon_max
        },
        'stats': {
            'min_value': float(np.min(values)),
            'max_value': float(np.max(values)),
            'mean_value': float(np.mean(values))
        }
    }

def get_risk_level(score: float) -> str:
    """Convert numerical score to risk level"""
    if score >= 80:
        return 'Very High'
    elif score >= 60:
        return 'High'
    elif score >= 40:
        return 'Moderate'
    elif score >= 20:
        return 'Low'
    return 'Very Low'

def get_mitigation_recommendations(risk_assessment: dict) -> List[str]:
    """Generate mitigation recommendations based on risk assessment"""
    score = risk_assessment['vulnerability_score']
    recommendations = []
    
    if score >= 80:
        recommendations.extend([
            "Immediate structural evaluation required",
            "Consider retrofit or relocation",
            "Develop emergency evacuation plan",
            "Install seismic monitoring systems"
        ])
    elif score >= 60:
        recommendations.extend([
            "Professional structural assessment recommended",
            "Strengthen critical structural elements",
            "Secure non-structural components",
            "Regular maintenance and inspections"
        ])
    elif score >= 40:
        recommendations.extend([
            "Periodic structural inspections",
            "Basic retrofitting considerations",
            "Develop basic emergency plan",
            "Secure heavy furniture and equipment"
        ])
    else:
        recommendations.extend([
            "Regular maintenance",
            "Basic emergency preparedness",
            "Document and monitor any structural changes",
            "Consider future upgrades"
        ])
    
    return recommendations
