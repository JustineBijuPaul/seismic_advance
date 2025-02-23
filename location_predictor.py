import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime, timedelta

class LocationPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.cluster_model = DBSCAN(eps=0.3, min_samples=3)
        
    def analyze_patterns(self, earthquakes: List[Dict]) -> List[Dict]:
        """Analyze earthquake patterns to identify high-risk zones"""
        if not earthquakes:
            return []
            
        df = pd.DataFrame(earthquakes)
        
        # Prepare features for clustering
        features = np.column_stack([
            df['latitude'].values,
            df['longitude'].values,
            df['magnitude'].values,
            df['depth'].values
        ])
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Perform clustering
        clusters = self.cluster_model.fit_predict(scaled_features)
        
        # Analyze each cluster
        risk_zones = []
        for cluster_id in set(clusters):
            if cluster_id == -1:  # Skip noise points
                continue
                
            mask = clusters == cluster_id
            cluster_quakes = df[mask]
            
            # Calculate cluster statistics
            center_lat = cluster_quakes['latitude'].mean()
            center_lon = cluster_quakes['longitude'].mean()
            avg_magnitude = cluster_quakes['magnitude'].mean()
            recent_activity = cluster_quakes['time'].max()
            
            # Calculate frequency of events
            time_span = (cluster_quakes['time'].max() - cluster_quakes['time'].min()).days
            frequency = len(cluster_quakes) / max(time_span, 1)  # Events per day
            
            # Calculate radius of influence
            distances = [
                geodesic((center_lat, center_lon), (lat, lon)).kilometers
                for lat, lon in zip(cluster_quakes['latitude'], cluster_quakes['longitude'])
            ]
            radius = max(distances) if distances else 0
            
            risk_zones.append({
                'center': (center_lat, center_lon),
                'radius': radius,
                'avg_magnitude': avg_magnitude,
                'frequency': frequency,
                'recent_activity': recent_activity,
                'num_events': len(cluster_quakes),
                'probability': self._calculate_probability(frequency, recent_activity)
            })
            
        return sorted(risk_zones, key=lambda x: x['probability'], reverse=True)
    
    def predict_next_location(self, risk_zones: List[Dict], time_horizon: int = 30) -> List[Dict]:
        """Predict potential earthquake locations for the next time period"""
        predictions = []
        
        for zone in risk_zones:
            # Calculate prediction confidence based on historical patterns
            confidence = min(
                zone['probability'] * 100,  # Base probability
                zone['frequency'] * time_horizon * 10,  # Frequency factor
                95  # Maximum confidence cap
            )
            
            # Calculate expected magnitude range
            magnitude_range = (
                max(zone['avg_magnitude'] - 0.5, 0),
                zone['avg_magnitude'] + 0.5
            )
            
            # Calculate time window for next event
            if zone['frequency'] > 0:
                days_until_next = max(1, int(1 / zone['frequency']))
                next_window = (
                    datetime.now(),
                    datetime.now() + timedelta(days=min(days_until_next, time_horizon))
                )
            else:
                next_window = (
                    datetime.now(),
                    datetime.now() + timedelta(days=time_horizon)
                )
            
            predictions.append({
                'location': {
                    'latitude': zone['center'][0],
                    'longitude': zone['center'][1],
                    'radius_km': zone['radius']
                },
                'magnitude_range': magnitude_range,
                'time_window': {
                    'start': next_window[0].strftime('%Y-%m-%d'),
                    'end': next_window[1].strftime('%Y-%m-%d')
                },
                'confidence': confidence,
                'basis': {
                    'past_events': zone['num_events'],
                    'frequency': f"{zone['frequency']:.2f} events/day",
                    'last_activity': zone['recent_activity'].strftime('%Y-%m-%d')
                }
            })
        
        return predictions
    
    def _calculate_probability(self, frequency: float, recent_activity: datetime) -> float:
        """Calculate probability of next event based on frequency and recency"""
        # Base probability from frequency
        prob = min(frequency * 0.5, 0.5)  # Up to 0.5 from frequency
        
        # Adjust based on recency
        days_since = (datetime.now() - recent_activity).days
        recency_factor = 1.0 / (1.0 + (days_since / 30))  # Decay over time
        
        return min(prob + (recency_factor * 0.5), 1.0)  # Up to 1.0 total
