function initPredictionMap(predictions) {
    // Initialize prediction map
    const map = L.map('prediction-map').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    
    const bounds = [];
    
    // Add prediction circles to map
    predictions.forEach(pred => {
        const circle = L.circle(
            [pred.location.latitude, pred.location.longitude],
            {
                color: 'red',
                fillColor: '#f03',
                fillOpacity: pred.confidence / 100,
                radius: pred.location.radius_km * 1000
            }
        ).addTo(map);
        
        circle.bindPopup(`
            <strong>Predicted Earthquake Zone</strong><br>
            Magnitude: M${pred.magnitude_range[0].toFixed(1)} - M${pred.magnitude_range[1].toFixed(1)}<br>
            Time Window: ${pred.time_window.start} to ${pred.time_window.end}<br>
            Confidence: ${pred.confidence.toFixed(1)}%
        `);
        
        bounds.push([pred.location.latitude, pred.location.longitude]);
    });
    
    // Fit map to show all predictions
    if (bounds.length > 0) {
        map.fitBounds(bounds);
    }
}
