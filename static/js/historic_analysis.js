function initCharts(magnitudeBins, magnitudeFreq, timelineLabels, timelineData, predictionLabels, predictionData) {
    // Initialize charts
    const magnitudeCtx = document.getElementById('magnitudeChart').getContext('2d');
    const timelineCtx = document.getElementById('timelineChart').getContext('2d');
    const predictionCtx = document.getElementById('predictionChart').getContext('2d');
    
    // Magnitude distribution chart
    new Chart(magnitudeCtx, {
        type: 'bar',
        data: {
            labels: magnitudeBins,
            datasets: [{
                label: 'Frequency',
                data: magnitudeFreq,
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
            labels: timelineLabels,
            datasets: [{
                label: 'Magnitude',
                data: timelineData,
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
            labels: predictionLabels,
            datasets: [{
                label: 'Predicted Magnitude',
                data: predictionData,
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

document.addEventListener('DOMContentLoaded', function() {
    // Form submission
    document.getElementById('analysisForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(e.target);
        try {
            const response = await fetch('/historic-analysis', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(Object.fromEntries(formData))
            });
            if (response.ok) {
                window.location.reload();
            }
        } catch (error) {
            console.error('Error:', error);
        }
    });
});
