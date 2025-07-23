/**
 * Wave Analysis Dashboard JavaScript
 * Handles interactive wave visualization and analysis controls
 */

class WaveAnalysisDashboard {
    constructor() {
        this.currentFileId = null;
        this.analysisData = null;
        this.currentChart = null;
        this.selectedWaveType = 'all';
        this.currentChartType = 'waveform';
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeChart();
        this.loadSampleData();
    }

    setupEventListeners() {
        // Wave type selection
        document.querySelectorAll('.wave-type-btn[data-wave-type]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectWaveType(e.target.closest('.wave-type-btn').dataset.waveType);
            });
        });

        // Chart type selection
        document.querySelectorAll('.chart-control-btn[data-chart]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.selectChartType(e.target.dataset.chart);
            });
        });

        // Educational content selection
        document.querySelectorAll('.wave-type-btn[data-education]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.showEducationalContent(e.target.closest('.wave-type-btn').dataset.education);
            });
        });

        // Parameter changes
        document.querySelectorAll('.parameter-input').forEach(input => {
            input.addEventListener('change', () => {
                this.updateAnalysisParameters();
            });
        });

        // Toolbar buttons
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.startAnalysis();
        });

        document.getElementById('exportBtn').addEventListener('click', () => {
            this.exportResults();
        });

        document.getElementById('reportBtn').addEventListener('click', () => {
            this.generateReport();
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetDashboard();
        });

        // File drop handling
        this.setupFileDropZone();
    }

    setupFileDropZone() {
        const dashboard = document.querySelector('.dashboard-container');
        
        dashboard.addEventListener('dragover', (e) => {
            e.preventDefault();
            dashboard.classList.add('drag-over');
        });

        dashboard.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dashboard.classList.remove('drag-over');
        });

        dashboard.addEventListener('drop', (e) => {
            e.preventDefault();
            dashboard.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileUpload(files[0]);
            }
        });
    }

    selectWaveType(waveType) {
        // Update UI
        document.querySelectorAll('.wave-type-btn[data-wave-type]').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-wave-type="${waveType}"]`).classList.add('active');
        
        this.selectedWaveType = waveType;
        this.updateVisualization();
        this.updateEducationalContent();
    }

    selectChartType(chartType) {
        // Update UI
        document.querySelectorAll('.chart-control-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-chart="${chartType}"]`).classList.add('active');
        
        this.currentChartType = chartType;
        this.updateChart();
    }

    showEducationalContent(contentType) {
        // Hide all educational content
        document.querySelectorAll('.educational-content').forEach(content => {
            content.classList.remove('active');
        });
        
        // Show selected content
        const targetContent = document.getElementById(`education-${contentType}`);
        if (targetContent) {
            targetContent.classList.add('active');
        }

        // Update button states
        document.querySelectorAll('.wave-type-btn[data-education]').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-education="${contentType}"]`).classList.add('active');
    }

    initializeChart() {
        const ctx = document.getElementById('mainChart');
        if (!ctx) return;

        this.currentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: 'rgba(255, 255, 255, 0.7)'
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: 'rgba(109, 93, 252, 0.5)',
                        borderWidth: 1
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    loadSampleData() {
        // Generate sample seismic data for demonstration
        const sampleData = this.generateSampleSeismicData();
        this.analysisData = sampleData;
        this.updateVisualization();
        this.updateResults(sampleData);
    }

    generateSampleSeismicData() {
        const duration = 60; // seconds
        const samplingRate = 100; // Hz
        const samples = duration * samplingRate;
        const timeAxis = Array.from({length: samples}, (_, i) => i / samplingRate);
        
        // Generate synthetic seismic signal with P, S, and surface waves
        const signal = new Array(samples).fill(0);
        
        // Add noise
        for (let i = 0; i < samples; i++) {
            signal[i] += (Math.random() - 0.5) * 0.1;
        }
        
        // Add P-wave (arrives at 10s)
        const pWaveStart = Math.floor(10 * samplingRate);
        for (let i = 0; i < 200; i++) {
            if (pWaveStart + i < samples) {
                signal[pWaveStart + i] += 0.5 * Math.sin(2 * Math.PI * 8 * i / samplingRate) * Math.exp(-i / 50);
            }
        }
        
        // Add S-wave (arrives at 18s)
        const sWaveStart = Math.floor(18 * samplingRate);
        for (let i = 0; i < 400; i++) {
            if (sWaveStart + i < samples) {
                signal[sWaveStart + i] += 0.8 * Math.sin(2 * Math.PI * 4 * i / samplingRate) * Math.exp(-i / 100);
            }
        }
        
        // Add surface waves (arrives at 25s)
        const surfaceWaveStart = Math.floor(25 * samplingRate);
        for (let i = 0; i < 1000; i++) {
            if (surfaceWaveStart + i < samples) {
                signal[surfaceWaveStart + i] += 0.6 * Math.sin(2 * Math.PI * 1 * i / samplingRate) * Math.exp(-i / 300);
            }
        }
        
        return {
            timeAxis: timeAxis,
            signal: signal,
            samplingRate: samplingRate,
            waveAnalysis: {
                pWaves: [{
                    arrivalTime: 10.0,
                    amplitude: 0.5,
                    frequency: 8.0,
                    duration: 2.0
                }],
                sWaves: [{
                    arrivalTime: 18.0,
                    amplitude: 0.8,
                    frequency: 4.0,
                    duration: 4.0
                }],
                surfaceWaves: [{
                    arrivalTime: 25.0,
                    amplitude: 0.6,
                    frequency: 1.0,
                    duration: 10.0
                }],
                magnitudeEstimates: [
                    { method: 'ML', magnitude: 4.2, confidence: 0.85 },
                    { method: 'Mb', magnitude: 4.1, confidence: 0.78 },
                    { method: 'Ms', magnitude: 4.3, confidence: 0.82 }
                ],
                qualityScore: 0.87
            }
        };
    }

    updateVisualization() {
        if (!this.analysisData) return;
        
        this.updateChart();
        this.updateSystemStatus('ready', 'Analysis Complete');
    }

    updateChart() {
        if (!this.currentChart || !this.analysisData) return;
        
        const { timeAxis, signal, waveAnalysis } = this.analysisData;
        
        let datasets = [];
        
        if (this.currentChartType === 'waveform') {
            // Main waveform
            if (this.selectedWaveType === 'all' || this.selectedWaveType === 'p-wave') {
                datasets.push({
                    label: 'Seismic Signal',
                    data: signal,
                    borderColor: 'rgba(255, 255, 255, 0.8)',
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0
                });
            }
            
            // Add wave markers
            if (this.selectedWaveType === 'all' || this.selectedWaveType === 'p-wave') {
                waveAnalysis.pWaves.forEach((wave, index) => {
                    const markerIndex = Math.floor(wave.arrivalTime * this.analysisData.samplingRate);
                    datasets.push({
                        label: `P-Wave ${index + 1}`,
                        data: [{x: wave.arrivalTime, y: signal[markerIndex]}],
                        borderColor: '#FF6B6B',
                        backgroundColor: '#FF6B6B',
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        showLine: false
                    });
                });
            }
            
            if (this.selectedWaveType === 'all' || this.selectedWaveType === 's-wave') {
                waveAnalysis.sWaves.forEach((wave, index) => {
                    const markerIndex = Math.floor(wave.arrivalTime * this.analysisData.samplingRate);
                    datasets.push({
                        label: `S-Wave ${index + 1}`,
                        data: [{x: wave.arrivalTime, y: signal[markerIndex]}],
                        borderColor: '#4ECDC4',
                        backgroundColor: '#4ECDC4',
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        showLine: false
                    });
                });
            }
            
            if (this.selectedWaveType === 'all' || this.selectedWaveType === 'surface') {
                waveAnalysis.surfaceWaves.forEach((wave, index) => {
                    const markerIndex = Math.floor(wave.arrivalTime * this.analysisData.samplingRate);
                    datasets.push({
                        label: `Surface Wave ${index + 1}`,
                        data: [{x: wave.arrivalTime, y: signal[markerIndex]}],
                        borderColor: '#45B7D1',
                        backgroundColor: '#45B7D1',
                        pointRadius: 8,
                        pointHoverRadius: 10,
                        showLine: false
                    });
                });
            }
        } else if (this.currentChartType === 'frequency') {
            // Generate frequency spectrum
            const spectrum = this.calculateFrequencySpectrum(signal);
            datasets.push({
                label: 'Frequency Spectrum',
                data: spectrum.magnitude,
                borderColor: 'rgba(109, 93, 252, 0.8)',
                backgroundColor: 'rgba(109, 93, 252, 0.2)',
                borderWidth: 2,
                pointRadius: 0,
                fill: true
            });
        } else if (this.currentChartType === 'spectrogram') {
            // For spectrogram, we'll show a simplified time-frequency representation
            datasets.push({
                label: 'Time-Frequency Analysis',
                data: this.generateSpectrogramData(signal),
                borderColor: 'rgba(255, 193, 7, 0.8)',
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                borderWidth: 1,
                pointRadius: 1
            });
        }
        
        this.currentChart.data.labels = timeAxis;
        this.currentChart.data.datasets = datasets;
        this.currentChart.update();
    }

    calculateFrequencySpectrum(signal) {
        // Simplified FFT calculation for demonstration
        const N = Math.min(signal.length, 1024);
        const frequencies = Array.from({length: N/2}, (_, i) => i * this.analysisData.samplingRate / N);
        const magnitude = Array.from({length: N/2}, (_, i) => {
            // Simplified magnitude calculation
            let sum = 0;
            for (let j = 0; j < N; j++) {
                sum += signal[j] * Math.cos(2 * Math.PI * i * j / N);
            }
            return Math.abs(sum) / N;
        });
        
        return { frequencies, magnitude };
    }

    generateSpectrogramData(signal) {
        // Generate simplified spectrogram data
        const windowSize = 256;
        const hopSize = 128;
        const spectrogramData = [];
        
        for (let i = 0; i < signal.length - windowSize; i += hopSize) {
            const window = signal.slice(i, i + windowSize);
            const spectrum = this.calculateFrequencySpectrum(window);
            const avgMagnitude = spectrum.magnitude.reduce((a, b) => a + b, 0) / spectrum.magnitude.length;
            spectrogramData.push(avgMagnitude);
        }
        
        return spectrogramData;
    }

    updateResults(data) {
        if (!data.waveAnalysis) return;
        
        const { waveAnalysis } = data;
        
        // Update wave counts
        document.getElementById('pWaveCount').textContent = waveAnalysis.pWaves.length;
        document.getElementById('sWaveCount').textContent = waveAnalysis.sWaves.length;
        document.getElementById('surfaceWaveCount').textContent = waveAnalysis.surfaceWaves.length;
        document.getElementById('qualityScore').textContent = `${(waveAnalysis.qualityScore * 100).toFixed(0)}%`;
        
        // Update arrival times
        if (waveAnalysis.pWaves.length > 0) {
            document.getElementById('pWaveArrival').textContent = `${waveAnalysis.pWaves[0].arrivalTime.toFixed(2)}s`;
        }
        if (waveAnalysis.sWaves.length > 0) {
            document.getElementById('sWaveArrival').textContent = `${waveAnalysis.sWaves[0].arrivalTime.toFixed(2)}s`;
        }
        if (waveAnalysis.pWaves.length > 0 && waveAnalysis.sWaves.length > 0) {
            const spDiff = waveAnalysis.sWaves[0].arrivalTime - waveAnalysis.pWaves[0].arrivalTime;
            document.getElementById('spTimeDiff').textContent = `${spDiff.toFixed(2)}s`;
        }
        if (waveAnalysis.surfaceWaves.length > 0) {
            document.getElementById('surfaceWaveArrival').textContent = `${waveAnalysis.surfaceWaves[0].arrivalTime.toFixed(2)}s`;
        }
        
        // Update magnitude estimates
        const magnitudeList = document.getElementById('magnitudeList');
        magnitudeList.innerHTML = '';
        waveAnalysis.magnitudeEstimates.forEach(est => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="characteristic-label">${est.method}</span>
                <span class="characteristic-value">${est.magnitude.toFixed(1)} (${(est.confidence * 100).toFixed(0)}%)</span>
            `;
            magnitudeList.appendChild(li);
        });
    }

    updateAnalysisParameters() {
        const parameters = {
            samplingRate: parseFloat(document.getElementById('samplingRate').value),
            minSnr: parseFloat(document.getElementById('minSnr').value),
            minConfidence: parseFloat(document.getElementById('minConfidence').value),
            filterFreq: parseFloat(document.getElementById('filterFreq').value)
        };
        
        console.log('Updated analysis parameters:', parameters);
        // In a real implementation, this would trigger re-analysis
    }

    updateSystemStatus(status, message) {
        const statusIndicator = document.getElementById('systemStatus');
        const statusClasses = ['ready', 'processing', 'error'];
        
        statusClasses.forEach(cls => statusIndicator.classList.remove(cls));
        statusIndicator.classList.add(status);
        statusIndicator.querySelector('span:last-child').textContent = message;
    }

    showLoading(message = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const messageEl = document.getElementById('loadingMessage');
        
        messageEl.textContent = message;
        overlay.style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    async handleFileUpload(file) {
        this.showLoading('Uploading file...');
        this.updateSystemStatus('processing', 'Uploading File');
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('enable_wave_analysis', 'true');
            formData.append('async_processing', 'false');
            
            const response = await axios.post('/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            
            if (response.data.wave_analysis) {
                this.processAnalysisResults(response.data);
            } else {
                this.showError('Wave analysis not available for this file');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showError('Failed to upload file');
        } finally {
            this.hideLoading();
        }
    }

    async startAnalysis() {
        if (!this.currentFileId) {
            this.showError('No file selected for analysis');
            return;
        }
        
        this.showLoading('Starting wave analysis...');
        this.updateSystemStatus('processing', 'Analyzing Waves');
        
        try {
            const parameters = {
                sampling_rate: parseFloat(document.getElementById('samplingRate').value),
                min_snr: parseFloat(document.getElementById('minSnr').value),
                min_detection_confidence: parseFloat(document.getElementById('minConfidence').value)
            };
            
            const response = await axios.post('/api/analyze_waves', {
                file_id: this.currentFileId,
                parameters: parameters
            });
            
            this.processAnalysisResults(response.data);
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Analysis failed');
        } finally {
            this.hideLoading();
        }
    }

    processAnalysisResults(data) {
        // Convert API response to internal format
        const analysisData = {
            timeAxis: data.time_labels || [],
            signal: data.amplitude_data || [],
            samplingRate: data.sampling_rate || 100,
            waveAnalysis: {
                pWaves: data.wave_analysis?.wave_separation?.p_waves_count ? 
                    [{arrivalTime: data.wave_analysis.arrival_times?.p_wave_arrival || 0}] : [],
                sWaves: data.wave_analysis?.wave_separation?.s_waves_count ? 
                    [{arrivalTime: data.wave_analysis.arrival_times?.s_wave_arrival || 0}] : [],
                surfaceWaves: data.wave_analysis?.wave_separation?.surface_waves_count ? 
                    [{arrivalTime: 30.0}] : [], // Placeholder
                magnitudeEstimates: data.wave_analysis?.magnitude_estimates || [],
                qualityScore: data.wave_analysis?.quality_score || 0
            }
        };
        
        this.analysisData = analysisData;
        this.updateVisualization();
        this.updateResults(analysisData);
        this.updateSystemStatus('ready', 'Analysis Complete');
    }

    exportResults() {
        if (!this.analysisData) {
            this.showError('No analysis results to export');
            return;
        }
        
        const exportData = {
            timestamp: new Date().toISOString(),
            waveType: this.selectedWaveType,
            parameters: {
                samplingRate: document.getElementById('samplingRate').value,
                minSnr: document.getElementById('minSnr').value,
                minConfidence: document.getElementById('minConfidence').value
            },
            results: this.analysisData.waveAnalysis
        };
        
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `wave_analysis_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    generateReport() {
        if (!this.analysisData) {
            this.showError('No analysis results to report');
            return;
        }
        
        // In a real implementation, this would generate a PDF report
        alert('Report generation feature would be implemented here');
    }

    resetDashboard() {
        this.currentFileId = null;
        this.analysisData = null;
        this.selectedWaveType = 'all';
        this.currentChartType = 'waveform';
        
        // Reset UI
        document.querySelectorAll('.wave-type-btn[data-wave-type]').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector('[data-wave-type="all"]').classList.add('active');
        
        document.querySelectorAll('.chart-control-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector('[data-chart="waveform"]').classList.add('active');
        
        // Reset parameters
        document.getElementById('samplingRate').value = '100';
        document.getElementById('minSnr').value = '2.0';
        document.getElementById('minConfidence').value = '0.3';
        document.getElementById('filterFreq').value = '20';
        
        // Clear results
        ['pWaveCount', 'sWaveCount', 'surfaceWaveCount', 'qualityScore'].forEach(id => {
            document.getElementById(id).textContent = '-';
        });
        
        ['pWaveArrival', 'sWaveArrival', 'spTimeDiff', 'surfaceWaveArrival'].forEach(id => {
            document.getElementById(id).textContent = '-';
        });
        
        document.getElementById('magnitudeList').innerHTML = '<li><span class="characteristic-label">No data available</span><span class="characteristic-value">-</span></li>';
        
        // Clear chart
        if (this.currentChart) {
            this.currentChart.data.labels = [];
            this.currentChart.data.datasets = [];
            this.currentChart.update();
        }
        
        this.updateSystemStatus('ready', 'System Ready');
        this.loadSampleData(); // Reload sample data
    }

    updateEducationalContent() {
        // Update educational content based on selected wave type
        const waveTypeMap = {
            'p-wave': 'p-wave',
            's-wave': 's-wave',
            'surface': 'surface',
            'all': 'analysis'
        };
        
        const contentType = waveTypeMap[this.selectedWaveType] || 'analysis';
        this.showEducationalContent(contentType);
    }

    showError(message) {
        this.updateSystemStatus('error', message);
        console.error(message);
    }

    // Real-time alert handling
    handleRealTimeAlert(alertData) {
        console.log('Received real-time alert:', alertData);
        
        // Update dashboard with alert information
        this.displayAlertOnDashboard(alertData);
        
        // If alert contains analysis data, update visualization
        if (alertData.analysis_data) {
            this.updateDashboardFromAlert(alertData);
        }
        
        // Flash the dashboard to indicate new alert
        this.flashDashboardAlert(alertData.severity);
    }

    displayAlertOnDashboard(alertData) {
        // Create or update alert banner on dashboard
        let alertBanner = document.getElementById('dashboardAlertBanner');
        if (!alertBanner) {
            alertBanner = document.createElement('div');
            alertBanner.id = 'dashboardAlertBanner';
            alertBanner.className = 'dashboard-alert-banner';
            
            const dashboard = document.querySelector('.dashboard-container');
            if (dashboard) {
                dashboard.insertBefore(alertBanner, dashboard.firstChild);
            }
        }
        
        alertBanner.className = `dashboard-alert-banner alert-${alertData.severity}`;
        alertBanner.innerHTML = `
            <div class="alert-banner-content">
                <div class="alert-banner-icon">⚠️</div>
                <div class="alert-banner-text">
                    <strong>${alertData.title}</strong>
                    <span>${alertData.message}</span>
                </div>
                <div class="alert-banner-time">${new Date(alertData.timestamp).toLocaleTimeString()}</div>
                <button class="alert-banner-close" onclick="this.parentElement.parentElement.style.display='none'">×</button>
            </div>
        `;
        
        // Auto-hide after delay (except critical)
        if (alertData.severity !== 'critical') {
            setTimeout(() => {
                if (alertBanner.style.display !== 'none') {
                    alertBanner.style.display = 'none';
                }
            }, 10000);
        }
    }

    updateDashboardFromAlert(alertData) {
        const analysisData = alertData.analysis_data;
        
        // Update magnitude display if available
        if (analysisData.magnitude) {
            const magnitudeDisplay = document.getElementById('currentMagnitude');
            if (magnitudeDisplay) {
                magnitudeDisplay.textContent = analysisData.magnitude.toFixed(1);
                magnitudeDisplay.className = `magnitude-display magnitude-${this.getMagnitudeClass(analysisData.magnitude)}`;
            }
        }
        
        // Update wave type indicators
        if (analysisData.wave_type) {
            this.highlightWaveType(analysisData.wave_type);
        }
        
        // Update frequency information
        if (analysisData.dominant_frequency) {
            const freqDisplay = document.getElementById('dominantFrequency');
            if (freqDisplay) {
                freqDisplay.textContent = `${analysisData.dominant_frequency.toFixed(2)} Hz`;
            }
        }
        
        // Update amplitude information
        if (analysisData.peak_amplitude) {
            const ampDisplay = document.getElementById('peakAmplitude');
            if (ampDisplay) {
                ampDisplay.textContent = analysisData.peak_amplitude.toFixed(2);
            }
        }
    }

    getMagnitudeClass(magnitude) {
        if (magnitude >= 7.0) return 'critical';
        if (magnitude >= 5.5) return 'high';
        if (magnitude >= 4.0) return 'medium';
        return 'low';
    }

    highlightWaveType(waveType) {
        // Remove existing highlights
        document.querySelectorAll('.wave-type-btn').forEach(btn => {
            btn.classList.remove('alert-highlight');
        });
        
        // Highlight the relevant wave type
        const waveTypeMap = {
            'P': 'p-wave',
            'S': 's-wave',
            'Love': 'surface',
            'Rayleigh': 'surface'
        };
        
        const mappedType = waveTypeMap[waveType] || waveType.toLowerCase();
        const targetBtn = document.querySelector(`[data-wave-type="${mappedType}"]`);
        if (targetBtn) {
            targetBtn.classList.add('alert-highlight');
            
            // Remove highlight after animation
            setTimeout(() => {
                targetBtn.classList.remove('alert-highlight');
            }, 3000);
        }
    }

    flashDashboardAlert(severity) {
        const dashboard = document.querySelector('.dashboard-container');
        if (!dashboard) return;
        
        const flashClass = `flash-alert-${severity}`;
        dashboard.classList.add(flashClass);
        
        setTimeout(() => {
            dashboard.classList.remove(flashClass);
        }, 1000);
    }

    // Method to subscribe to specific alert types
    subscribeToAlerts(alertTypes = ['magnitude_threshold', 'wave_amplitude']) {
        if (window.earthquakeAlerts && window.earthquakeAlerts.isConnectedToAlerts()) {
            window.earthquakeAlerts.socket.emit('subscribe_alerts', {
                alert_types: alertTypes,
                severity_levels: ['medium', 'high', 'critical']
            });
        }
    }

    // Method to add custom alert handler
    addAlertHandler(handler) {
        if (window.earthquakeAlerts) {
            window.earthquakeAlerts.addAlertHandler(handler);
        }
    }
}

// CSS styles for dashboard alert integration
const dashboardAlertStyles = `
<style>
.dashboard-alert-banner {
    margin-bottom: 15px;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    animation: slideDown 0.3s ease-out;
}

.dashboard-alert-banner.alert-low {
    background: linear-gradient(135deg, #28a745, #20c997);
}

.dashboard-alert-banner.alert-medium {
    background: linear-gradient(135deg, #ffc107, #fd7e14);
}

.dashboard-alert-banner.alert-high {
    background: linear-gradient(135deg, #fd7e14, #dc3545);
}

.dashboard-alert-banner.alert-critical {
    background: linear-gradient(135deg, #dc3545, #6f42c1);
    animation: pulse 1s infinite;
}

.alert-banner-content {
    display: flex;
    align-items: center;
    padding: 12px 15px;
    color: white;
}

.alert-banner-icon {
    font-size: 24px;
    margin-right: 12px;
}

.alert-banner-text {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.alert-banner-text strong {
    font-size: 16px;
    font-weight: bold;
}

.alert-banner-text span {
    font-size: 14px;
    opacity: 0.9;
}

.alert-banner-time {
    font-size: 12px;
    opacity: 0.8;
    margin-right: 10px;
}

.alert-banner-close {
    background: rgba(255, 255, 255, 0.2);
    border: none;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
}

.alert-banner-close:hover {
    background: rgba(255, 255, 255, 0.3);
}

.wave-type-btn.alert-highlight {
    animation: highlightPulse 3s ease-in-out;
    box-shadow: 0 0 20px rgba(255, 193, 7, 0.6);
}

.magnitude-display {
    font-size: 24px;
    font-weight: bold;
    padding: 8px 12px;
    border-radius: 6px;
    text-align: center;
    transition: all 0.3s ease;
}

.magnitude-display.magnitude-low {
    background: #28a745;
    color: white;
}

.magnitude-display.magnitude-medium {
    background: #ffc107;
    color: #000;
}

.magnitude-display.magnitude-high {
    background: #fd7e14;
    color: white;
}

.magnitude-display.magnitude-critical {
    background: #dc3545;
    color: white;
    animation: pulse 1s infinite;
}

.flash-alert-low {
    animation: flashGreen 1s ease-out;
}

.flash-alert-medium {
    animation: flashYellow 1s ease-out;
}

.flash-alert-high {
    animation: flashOrange 1s ease-out;
}

.flash-alert-critical {
    animation: flashRed 1s ease-out;
}

@keyframes slideDown {
    from {
        transform: translateY(-100%);
        opacity: 0;
    }
    to {
        transform: translateY(0);
        opacity: 1;
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.7;
    }
}

@keyframes highlightPulse {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 0 rgba(255, 193, 7, 0);
    }
    50% {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255, 193, 7, 0.6);
    }
}

@keyframes flashGreen {
    0%, 100% { box-shadow: none; }
    50% { box-shadow: 0 0 30px rgba(40, 167, 69, 0.5); }
}

@keyframes flashYellow {
    0%, 100% { box-shadow: none; }
    50% { box-shadow: 0 0 30px rgba(255, 193, 7, 0.5); }
}

@keyframes flashOrange {
    0%, 100% { box-shadow: none; }
    50% { box-shadow: 0 0 30px rgba(253, 126, 20, 0.5); }
}

@keyframes flashRed {
    0%, 100% { box-shadow: none; }
    50% { box-shadow: 0 0 30px rgba(220, 53, 69, 0.5); }
}
</style>
`;

// Inject dashboard alert styles
document.head.insertAdjacentHTML('beforeend', dashboardAlertStyles);

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.waveDashboard = new WaveAnalysisDashboard();
    
    // Subscribe to alerts once both systems are ready
    setTimeout(() => {
        if (window.waveDashboard && window.earthquakeAlerts) {
            window.waveDashboard.subscribeToAlerts();
            console.log('Wave dashboard subscribed to real-time alerts');
        }
    }, 1000);
});