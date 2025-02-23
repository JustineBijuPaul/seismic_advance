import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class SeismicCalculator:
    @staticmethod
    def calculate_impact_level(magnitude: float, distance: float) -> int:
        """Calculate impact level (1-5) based on magnitude and distance"""
        if magnitude >= 7.0:
            base_level = 5
        elif magnitude >= 6.0:
            base_level = 4
        elif magnitude >= 5.0:
            base_level = 3
        elif magnitude >= 4.0:
            base_level = 2
        else:
            base_level = 1
        
        # Adjust based on distance
        if distance <= 50:
            return base_level
        elif distance <= 100:
            return max(1, base_level - 1)
        elif distance <= 200:
            return max(1, base_level - 2)
        else:
            return max(1, base_level - 3)

    @staticmethod
    def process_earthquake_data(feature: Dict) -> Dict:
        """Process raw earthquake feature data"""
        props = feature.get('properties', {})
        coords = feature.get('geometry', {}).get('coordinates', [0, 0, 0])
        
        if not props or not coords or len(coords) < 3:
            return None
            
        return {
            'time': datetime.fromtimestamp(props.get('time', 0)/1000.0).strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': props.get('time', 0),
            'place': props.get('place', 'Unknown Location'),
            'magnitude': props.get('mag', 0.0),
            'depth': coords[2],
            'latitude': coords[1],
            'longitude': coords[0],
            'url': props.get('url', '#'),
            'id': feature.get('id'),
            'alert': props.get('alert', 'none'),
            'tsunami': props.get('tsunami', 0),
            'significance': props.get('sig', 0),
            'felt': props.get('felt', 0)
        }
