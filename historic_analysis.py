import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple  # Added Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from scipy import stats

@dataclass
class TrendAnalysis:
    trend: str  # increasing, decreasing, stable
    confidence: float
    prediction_next_month: float
    seasonal_pattern: str

class HistoricDataAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()
    
    def analyze_trends(self, earthquakes: List[Dict]) -> Dict[str, Any]:
        """Analyze earthquake trends and patterns"""
        try:
            if not earthquakes:
                return {
                    'trend': 'stable',
                    'confidence': 0,
                    'prediction_next_month': 0,
                    'seasonal_pattern': 'No data available'
                }

            df = pd.DataFrame(earthquakes)
            df['time'] = pd.to_datetime(df['time'])
            
            # Calculate trend over time
            X = np.array((df['time'] - df['time'].min()).dt.total_seconds()).reshape(-1, 1)
            y = df['magnitude'].values
            model = LinearRegression()
            model.fit(X, y)
            
            slope = model.coef_[0]
            r2_score = model.score(X, y)
            confidence = min(r2_score * 100, 100)  # Convert R² to percentage
            
            # Determine trend direction
            if abs(slope) < 1e-6:
                trend = 'stable'
            elif slope > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            
            # Calculate next month prediction
            last_time = df['time'].max()
            next_month = last_time + timedelta(days=30)
            prediction_x = np.array((next_month - df['time'].min()).total_seconds()).reshape(1, -1)
            prediction = float(model.predict(prediction_x))
            
            # Analyze seasonal patterns
            seasonal_pattern = self._analyze_seasonal_pattern(df)
            
            return {
                'trend': trend,
                'confidence': round(confidence, 1),
                'prediction_next_month': round(max(0, prediction), 1),
                'seasonal_pattern': seasonal_pattern
            }
            
        except Exception as e:
            print(f"Error in trend analysis: {str(e)}")
            return {
                'trend': 'stable',
                'confidence': 0,
                'prediction_next_month': 0,
                'seasonal_pattern': 'Error analyzing patterns'
            }

    def calculate_statistics(self, earthquakes: List[Dict]) -> Dict[str, float]:
        """Calculate basic statistics from earthquake data"""
        try:
            if not earthquakes:
                return {
                    'total_events': 0,
                    'average_magnitude': 0,
                    'max_magnitude': 0,
                    'std_deviation': 0
                }
            
            df = pd.DataFrame(earthquakes)
            return {
                'total_events': len(df),
                'average_magnitude': round(df['magnitude'].mean(), 1),
                'max_magnitude': round(df['magnitude'].max(), 1),
                'std_deviation': round(df['magnitude'].std(), 2)
            }
        except Exception as e:
            print(f"Error calculating statistics: {str(e)}")
            return {
                'total_events': 0,
                'average_magnitude': 0,
                'max_magnitude': 0,
                'std_deviation': 0
            }

    def predict_future_events(self, earthquakes: List[Dict]) -> List[Dict]:
        """Predict future earthquake events"""
        try:
            if not earthquakes:
                return []
            
            df = pd.DataFrame(earthquakes)
            df['time'] = pd.to_datetime(df['time'])
            
            # Simple linear regression for prediction
            X = np.array((df['time'] - df['time'].min()).dt.total_seconds()).reshape(-1, 1)
            y = df['magnitude'].values
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate predictions for next 7 days
            predictions = []
            last_time = df['time'].max()
            
            for days in range(1, 8):
                future_time = last_time + timedelta(days=days)
                prediction_x = np.array((future_time - df['time'].min()).total_seconds()).reshape(1, -1)
                predicted_magnitude = float(model.predict(prediction_x))
                confidence = max(0, min(100, 100 - (days * 10)))  # Decrease confidence over time
                
                predictions.append({
                    'time': future_time.strftime('%Y-%m-%d'),
                    'predicted_magnitude': round(max(0, predicted_magnitude), 1),
                    'confidence': round(confidence, 1)
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error in future prediction: {str(e)}")
            return []

    def _analyze_seasonal_pattern(self, df: pd.DataFrame) -> str:
        """Analyze seasonal patterns in earthquake data"""
        try:
            if len(df) < 2:
                return "Insufficient data for pattern analysis"
            
            df['month'] = df['time'].dt.month
            monthly_avg = df.groupby('month')['magnitude'].mean()
            
            if monthly_avg.std() < 0.1:
                return "No significant seasonal pattern detected"
            
            peak_month = monthly_avg.idxmax()
            peak_magnitude = monthly_avg.max()
            
            months = ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            
            return f"Peak activity observed in {months[peak_month-1]} (avg. M{peak_magnitude:.1f})"
            
        except Exception as e:
            print(f"Error analyzing seasonal pattern: {str(e)}")
            return "Error analyzing seasonal patterns"
