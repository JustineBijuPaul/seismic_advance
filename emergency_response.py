from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from datetime import datetime
import requests
from geopy.distance import geodesic
import polyline
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmergencyContact:
    name: str
    relationship: str
    phone: str
    email: str
    address: str
    notify_priority: int

@dataclass
class EmergencyResource:
    name: str
    type: str
    address: str
    coordinates: tuple
    phone: str
    capacity: Optional[int] = None
    available: bool = True

class EmergencyResponseManager:
    def __init__(self):
        self.safety_checklist = {
            'immediate': [
                'Drop, Cover, and Hold On',
                'Stay away from windows',
                'If indoors, stay inside',
                'If outdoors, move to open area'
            ],
            'after_shock': [
                'Check for injuries',
                'Check for structural damage',
                'Turn off gas if leak suspected',
                'Monitor local news',
                'Be prepared for aftershocks'
            ],
            'evacuation': [
                'Grab emergency kit',
                'Follow designated evacuation route',
                'Help others if safe to do so',
                'Do not use elevators',
                'Meet at designated assembly point'
            ]
        }
        
        self.resources_db = {
            'hospitals': [],
            'shelters': [],
            'fire_stations': [],
            'police_stations': []
        }
        
        # Initialize contacts with a default empty list for each user
        self.contacts_db = defaultdict(list)
        logger.info("EmergencyResponseManager initialized")

    def add_emergency_contact(self, user_id: str, contact: EmergencyContact) -> bool:
        """Add or update emergency contact"""
        try:
            if not user_id:
                logger.error("No user_id provided")
                return False
            
            self.contacts_db[user_id].append(contact)
            logger.info(f"Added contact for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding contact: {str(e)}")
            return False

    def get_emergency_contacts(self, user_id: str) -> List[EmergencyContact]:
        """Get user's emergency contacts sorted by priority"""
        try:
            if not user_id:
                logger.warning("No user_id provided")
                return []
            
            contacts = self.contacts_db.get(user_id, [])
            return sorted(contacts, key=lambda x: x.notify_priority)
        except Exception as e:
            logger.error(f"Error retrieving contacts: {str(e)}")
            return []

    def add_resource(self, resource: EmergencyResource) -> bool:
        """Add emergency resource to database"""
        resource_type = resource.type.lower()
        if resource_type in self.resources_db:
            self.resources_db[resource_type].append(resource)
            return True
        return False

    def find_nearest_resources(
        self,
        lat: float,
        lng: float,
        resource_type: str,
        radius_km: float = 10,
        limit: int = 5
    ) -> List[Dict]:
        """Find nearest emergency resources"""
        if resource_type not in self.resources_db:
            return []

        resources = self.resources_db[resource_type]
        nearby = []

        for resource in resources:
            if not resource.available:
                continue

            distance = geodesic(
                (lat, lng),
                resource.coordinates
            ).kilometers

            if distance <= radius_km:
                nearby.append({
                    'name': resource.name,
                    'address': resource.address,
                    'phone': resource.phone,
                    'distance': round(distance, 2),
                    'coordinates': resource.coordinates,
                    'capacity': resource.capacity
                })

        return sorted(nearby, key=lambda x: x['distance'])[:limit]

    def get_evacuation_route(
        self,
        start_lat: float,
        start_lng: float,
        end_lat: float,
        end_lng: float,
        avoid_areas: List[Dict] = None
    ) -> Dict:
        """Calculate evacuation route avoiding dangerous areas"""
        # Use OSRM for routing
        base_url = "http://router.project-osrm.org/route/v1/driving"
        url = f"{base_url}/{start_lng},{start_lat};{end_lng},{end_lat}"
        
        params = {
            'alternatives': 'true',
            'steps': 'true',
            'annotations': 'true'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            route_data = response.json()
            
            if 'routes' not in route_data:
                return {'error': 'No route found'}
            
            # Get the best route
            best_route = route_data['routes'][0]
            
            # Decode route geometry
            geometry = polyline.decode(best_route['geometry'])
            
            return {
                'distance': best_route['distance'],
                'duration': best_route['duration'],
                'geometry': geometry,
                'steps': best_route['legs'][0]['steps']
            }
            
        except Exception as e:
            return {'error': str(e)}

    def get_safety_guidelines(self, phase: str = None) -> Dict:
        """Get safety guidelines for different phases"""
        if phase and phase in self.safety_checklist:
            return {phase: self.safety_checklist[phase]}
        return self.safety_checklist

def create_sample_resources():
    """Create sample emergency resources for testing"""
    return [
        EmergencyResource(
            name="Central Hospital",
            type="hospitals",
            address="123 Medical Dr",
            coordinates=(40.7128, -74.0060),
            phone="555-0123",
            capacity=200
        ),
        EmergencyResource(
            name="Community Shelter",
            type="shelters",
            address="456 Safe Haven St",
            coordinates=(40.7150, -74.0080),
            phone="555-0124",
            capacity=500
        ),
        EmergencyResource(
            name="Main Fire Station",
            type="fire_stations",
            address="789 Emergency Rd",
            coordinates=(40.7140, -74.0070),
            phone="555-0125"
        )
    ]
