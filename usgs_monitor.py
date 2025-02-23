import requests
from typing import Dict, List
import time
from datetime import datetime
from requests.adapters import HTTPAdapter
from geopy.distance import geodesic
import logging
from seismic_utils import SeismicCalculator

logger = logging.getLogger(__name__)

class USGSMonitor:
    FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"
    FEEDS = {
        'hour': f"{FEED_URL}/all_hour.geojson",
        'day': f"{FEED_URL}/all_day.geojson",
        'week': f"{FEED_URL}/all_week.geojson"
    }

    def __init__(self):
        self.last_check_time = datetime.utcnow().timestamp() * 1000
        self.session = requests.Session()
        self.session.mount('https://', HTTPAdapter(max_retries=3))

    def check_feed(self, user_locations: Dict) -> List[Dict]:
        """Check USGS feed for new earthquakes"""
        try:
            # Add rate limiting
            time.sleep(1)  # Respect USGS rate limits
            
            current_time = datetime.utcnow().timestamp() * 1000
            response = self.session.get(self.FEEDS['hour'], timeout=10)
            response.raise_for_status()
            data = response.json()
            
            new_earthquakes = []
            for feature in data['features']:
                quake_time = feature['properties']['time']
                
                if quake_time > self.last_check_time:
                    quake_data = SeismicCalculator.process_earthquake_data(feature)
                    if quake_data:
                        new_earthquakes.append(quake_data)
            
            self.last_check_time = current_time
            
            if new_earthquakes:
                return self._process_nearby_earthquakes(new_earthquakes, user_locations)
            
            return []
            
        except requests.exceptions.RequestException as e:
            logger.error(f"USGS API Error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error checking USGS feed: {str(e)}")
            return []

    def _process_nearby_earthquakes(self, earthquakes: List[Dict], user_locations: Dict) -> List[Dict]:
        """Process earthquakes and find ones near users"""
        # Sort by significance
        earthquakes.sort(key=lambda x: x['significance'], reverse=True)
        
        nearby_events = []
        for user_id, location in user_locations.items():
            nearby_quakes = []
            for quake in earthquakes:
                distance = geodesic(
                    (location['lat'], location['lng']),
                    (quake['latitude'], quake['longitude'])
                ).kilometers
                
                if distance <= location['radius']:
                    quake_data = {
                        **quake,
                        'distance': round(distance, 2),
                        'impact_level': SeismicCalculator.calculate_impact_level(
                            quake['magnitude'], 
                            distance
                        )
                    }
                    nearby_quakes.append(quake_data)
            
            if nearby_quakes:
                nearby_events.extend([{
                    'user_id': user_id,
                    'quake': quake
                } for quake in nearby_quakes])
        
        return nearby_events
