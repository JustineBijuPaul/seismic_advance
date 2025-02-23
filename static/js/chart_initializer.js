function initializeCharts(rawData) {
    // Parse the raw data if it's in string format
    const data = {
        magnitudeBins: typeof rawData.magnitudeBins === 'string' ? JSON.parse(rawData.magnitudeBins) : rawData.magnitudeBins,
        magnitudeFreq: typeof rawData.magnitudeFreq === 'string' ? JSON.parse(rawData.magnitudeFreq) : rawData.magnitudeFreq,
        timelineLabels: typeof rawData.timelineLabels === 'string' ? JSON.parse(rawData.timelineLabels) : rawData.timelineLabels,
        timelineData: typeof rawData.timelineData === 'string' ? JSON.parse(rawData.timelineData) : rawData.timelineData,
        predictionLabels: typeof rawData.predictionLabels === 'string' ? JSON.parse(rawData.predictionLabels) : rawData.predictionLabels,
        predictionData: typeof rawData.predictionData === 'string' ? JSON.parse(rawData.predictionData) : rawData.predictionData
    };

    // Initialize charts
    const magnitudeCtx = document.getElementById('magnitudeChart').getContext('2d');
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    const predictionCtx = document.getElementById('predictionChart').getContext('2d');
    
    // Magnitude distribution chart
    new Chart(magnitudeCtx, {
        type: 'bar',
        data: {
            labels: data.magnitudeBins,
            datasets: [{
                label: 'Frequency',
                data: data.magnitudeFreq,
                backgroundColor: 'rgba(255, 68, 68, 0.5)',
                borderColor: 'rgba(255, 68, 68, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // Timeline chart
    new Chart(timelineCtx, {
        type: 'line',
        data: {
            labels: data.timelineLabels,
            datasets: [{
                label: 'Magnitude',
                data: data.timelineData,
                borderColor: 'rgba(255, 68, 68, 1)',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });

    // Prediction chart
    new Chart(predictionCtx, {
        type: 'line',
        data: {
            labels: data.predictionLabels,
            datasets: [{
                label: 'Predicted Magnitude',
                data: data.predictionData,
                borderColor: 'rgba(51, 181, 229, 1)',
                backgroundColor: 'rgba(51, 181, 229, 0.1)',
                fill: true
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: { beginAtZero: true }
            }
        }
    });
}
