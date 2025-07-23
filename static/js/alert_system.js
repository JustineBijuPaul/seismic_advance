/**
 * Real-time Alert System for Earthquake Events
 * Handles WebSocket connections and real-time alert notifications
 */

class EarthquakeAlertSystem {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.alertHistory = [];
        this.maxHistorySize = 100;
        this.alertHandlers = [];
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        
        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupUI();
        this.loadAlertHistory();
    }

    setupWebSocket() {
        try {
            // Initialize Socket.IO connection to alerts namespace
            this.socket = io('/alerts', {
                transports: ['websocket', 'polling'],
                timeout: 5000,
                reconnection: true,
                reconnectionAttempts: this.maxReconnectAttempts,
                reconnectionDelay: this.reconnectDelay
            });

            // Connection event handlers
            this.socket.on('connect', () => {
                console.log('Connected to alert system');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;
                this.updateConnectionStatus(true);
                this.subscribeToAlerts();
            });

            this.socket.on('disconnect', (reason) => {
                console.log('Disconnected from alert system:', reason);
                this.isConnected = false;
                this.updateConnectionStatus(false);
            });

            this.socket.on('connect_error', (error) => {
                console.error('Connection error:', error);
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.handleReconnection();
            });

            // Alert event handlers
            this.socket.on('earthquake_alert', (alertData) => {
                this.handleIncomingAlert(alertData);
            });

            this.socket.on('alert_update', (updateData) => {
                this.handleAlertUpdate(updateData);
            });

            this.socket.on('system_status', (statusData) => {
                this.handleSystemStatus(statusData);
            });

        } catch (error) {
            console.error('Failed to initialize WebSocket connection:', error);
            this.updateConnectionStatus(false);
        }
    }

    setupUI() {
        // Create alert notification container if it doesn't exist
        if (!document.getElementById('alertContainer')) {
            const alertContainer = document.createElement('div');
            alertContainer.id = 'alertContainer';
            alertContainer.className = 'alert-container';
            document.body.appendChild(alertContainer);
        }

        // Create alert history panel if it doesn't exist
        if (!document.getElementById('alertHistoryPanel')) {
            this.createAlertHistoryPanel();
        }

        // Setup alert controls
        this.setupAlertControls();
    }

    createAlertHistoryPanel() {
        const panel = document.createElement('div');
        panel.id = 'alertHistoryPanel';
        panel.className = 'alert-history-panel';
        panel.innerHTML = `
            <div class="alert-panel-header">
                <h3>Recent Alerts</h3>
                <div class="alert-controls">
                    <button id="clearAlertsBtn" class="btn btn-sm btn-outline">Clear</button>
                    <button id="toggleAlertPanel" class="btn btn-sm btn-primary">Hide</button>
                </div>
            </div>
            <div class="alert-panel-body">
                <div id="alertHistoryList" class="alert-history-list">
                    <div class="no-alerts">No recent alerts</div>
                </div>
            </div>
            <div class="alert-panel-footer">
                <div class="connection-status">
                    <span id="connectionIndicator" class="status-indicator disconnected"></span>
                    <span id="connectionText">Connecting...</span>
                </div>
            </div>
        `;

        // Position the panel
        panel.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            width: 350px;
            max-height: 500px;
            background: rgba(0, 0, 0, 0.9);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            z-index: 10000;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        `;

        document.body.appendChild(panel);
    }

    setupAlertControls() {
        // Clear alerts button
        const clearBtn = document.getElementById('clearAlertsBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearAlertHistory();
            });
        }

        // Toggle panel button
        const toggleBtn = document.getElementById('toggleAlertPanel');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                this.toggleAlertPanel();
            });
        }

        // Close alert notifications on click
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('alert-close')) {
                const alertElement = e.target.closest('.alert-notification');
                if (alertElement) {
                    this.dismissAlert(alertElement);
                }
            }
        });
    }

    subscribeToAlerts() {
        if (this.socket && this.isConnected) {
            // Subscribe to all alert types by default
            this.socket.emit('subscribe_alerts', {
                alert_types: ['magnitude_threshold', 'wave_amplitude', 'frequency_anomaly'],
                severity_levels: ['medium', 'high', 'critical']
            });
        }
    }

    handleIncomingAlert(alertData) {
        console.log('Received alert:', alertData);
        
        // Add to history
        this.addToHistory(alertData);
        
        // Show notification
        this.showAlertNotification(alertData);
        
        // Update dashboard if available
        this.updateDashboard(alertData);
        
        // Trigger custom handlers
        this.triggerAlertHandlers(alertData);
        
        // Play alert sound for high/critical alerts
        if (['high', 'critical'].includes(alertData.severity)) {
            this.playAlertSound(alertData.severity);
        }
    }

    handleAlertUpdate(updateData) {
        console.log('Alert update received:', updateData);
        // Handle alert updates (e.g., magnitude revisions)
        this.updateExistingAlert(updateData);
    }

    handleSystemStatus(statusData) {
        console.log('System status update:', statusData);
        // Update system status indicators
        this.updateSystemStatusIndicators(statusData);
    }

    showAlertNotification(alertData) {
        const container = document.getElementById('alertContainer');
        if (!container) return;

        const notification = document.createElement('div');
        notification.className = `alert-notification alert-${alertData.severity}`;
        notification.innerHTML = `
            <div class="alert-header">
                <div class="alert-severity ${alertData.severity}">${alertData.severity.toUpperCase()}</div>
                <div class="alert-time">${new Date(alertData.timestamp).toLocaleTimeString()}</div>
                <button class="alert-close">&times;</button>
            </div>
            <div class="alert-content">
                <h4 class="alert-title">${alertData.title}</h4>
                <p class="alert-message">${alertData.message}</p>
                ${this.formatAlertDetails(alertData)}
            </div>
        `;

        // Style the notification
        notification.style.cssText = `
            margin-bottom: 10px;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid ${this.getSeverityColor(alertData.severity)};
            background: rgba(0, 0, 0, 0.9);
            color: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
            animation: slideInRight 0.3s ease-out;
        `;

        container.appendChild(notification);

        // Auto-dismiss after delay (except for critical alerts)
        if (alertData.severity !== 'critical') {
            setTimeout(() => {
                this.dismissAlert(notification);
            }, this.getAlertDismissDelay(alertData.severity));
        }

        // Limit number of visible notifications
        this.limitVisibleNotifications(container);
    }

    formatAlertDetails(alertData) {
        let details = '<div class="alert-details">';
        
        if (alertData.analysis_data) {
            if (alertData.analysis_data.magnitude) {
                details += `<div class="alert-detail">Magnitude: <strong>${alertData.analysis_data.magnitude}</strong></div>`;
            }
            if (alertData.analysis_data.wave_type) {
                details += `<div class="alert-detail">Wave Type: <strong>${alertData.analysis_data.wave_type}</strong></div>`;
            }
            if (alertData.analysis_data.peak_amplitude) {
                details += `<div class="alert-detail">Peak Amplitude: <strong>${alertData.analysis_data.peak_amplitude.toFixed(2)}</strong></div>`;
            }
            if (alertData.analysis_data.dominant_frequency) {
                details += `<div class="alert-detail">Frequency: <strong>${alertData.analysis_data.dominant_frequency.toFixed(2)} Hz</strong></div>`;
            }
        }
        
        if (alertData.location_info) {
            details += `<div class="alert-detail">Location: <strong>${alertData.location_info.latitude?.toFixed(2)}, ${alertData.location_info.longitude?.toFixed(2)}</strong></div>`;
        }
        
        details += '</div>';
        return details;
    }

    getSeverityColor(severity) {
        const colors = {
            low: '#28a745',
            medium: '#ffc107',
            high: '#fd7e14',
            critical: '#dc3545'
        };
        return colors[severity] || '#6c757d';
    }

    getAlertDismissDelay(severity) {
        const delays = {
            low: 5000,
            medium: 8000,
            high: 12000,
            critical: 0 // Never auto-dismiss
        };
        return delays[severity] || 8000;
    }

    dismissAlert(alertElement) {
        alertElement.style.animation = 'slideOutRight 0.3s ease-in';
        setTimeout(() => {
            if (alertElement.parentNode) {
                alertElement.parentNode.removeChild(alertElement);
            }
        }, 300);
    }

    limitVisibleNotifications(container, maxVisible = 5) {
        const notifications = container.querySelectorAll('.alert-notification');
        if (notifications.length > maxVisible) {
            // Remove oldest notifications
            for (let i = 0; i < notifications.length - maxVisible; i++) {
                this.dismissAlert(notifications[i]);
            }
        }
    }

    addToHistory(alertData) {
        this.alertHistory.unshift(alertData);
        
        // Limit history size
        if (this.alertHistory.length > this.maxHistorySize) {
            this.alertHistory = this.alertHistory.slice(0, this.maxHistorySize);
        }
        
        this.updateAlertHistoryDisplay();
        this.saveAlertHistory();
    }

    updateAlertHistoryDisplay() {
        const historyList = document.getElementById('alertHistoryList');
        if (!historyList) return;

        if (this.alertHistory.length === 0) {
            historyList.innerHTML = '<div class="no-alerts">No recent alerts</div>';
            return;
        }

        historyList.innerHTML = this.alertHistory.map(alert => `
            <div class="alert-history-item alert-${alert.severity}">
                <div class="alert-history-header">
                    <span class="alert-history-severity">${alert.severity.toUpperCase()}</span>
                    <span class="alert-history-time">${new Date(alert.timestamp).toLocaleString()}</span>
                </div>
                <div class="alert-history-title">${alert.title}</div>
                <div class="alert-history-message">${alert.message}</div>
            </div>
        `).join('');
    }

    updateDashboard(alertData) {
        // Update wave dashboard if available
        if (window.waveDashboard) {
            window.waveDashboard.handleRealTimeAlert(alertData);
        }
        
        // Update any other dashboard components
        this.updateAlertCounters(alertData);
    }

    updateAlertCounters(alertData) {
        // Update alert counters in the UI
        const severityCounters = document.querySelectorAll(`[data-alert-counter="${alertData.severity}"]`);
        severityCounters.forEach(counter => {
            const currentCount = parseInt(counter.textContent) || 0;
            counter.textContent = currentCount + 1;
        });
    }

    playAlertSound(severity) {
        try {
            // Create audio context if not exists
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }

            // Play different tones based on severity
            const frequency = severity === 'critical' ? 800 : 600;
            const duration = severity === 'critical' ? 1000 : 500;
            
            this.playTone(frequency, duration);
        } catch (error) {
            console.warn('Could not play alert sound:', error);
        }
    }

    playTone(frequency, duration) {
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.frequency.value = frequency;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration / 1000);
        
        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + duration / 1000);
    }

    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connectionIndicator');
        const text = document.getElementById('connectionText');
        
        if (indicator && text) {
            if (connected) {
                indicator.className = 'status-indicator connected';
                text.textContent = 'Connected';
            } else {
                indicator.className = 'status-indicator disconnected';
                text.textContent = 'Disconnected';
            }
        }
    }

    handleReconnection() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            this.reconnectDelay *= 2; // Exponential backoff
            
            setTimeout(() => {
                console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
                this.setupWebSocket();
            }, this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
            this.updateConnectionStatus(false);
        }
    }

    clearAlertHistory() {
        this.alertHistory = [];
        this.updateAlertHistoryDisplay();
        this.saveAlertHistory();
        
        // Clear visible notifications
        const container = document.getElementById('alertContainer');
        if (container) {
            container.innerHTML = '';
        }
    }

    toggleAlertPanel() {
        const panel = document.getElementById('alertHistoryPanel');
        const toggleBtn = document.getElementById('toggleAlertPanel');
        
        if (panel && toggleBtn) {
            if (panel.style.display === 'none') {
                panel.style.display = 'block';
                toggleBtn.textContent = 'Hide';
            } else {
                panel.style.display = 'none';
                toggleBtn.textContent = 'Show';
            }
        }
    }

    saveAlertHistory() {
        try {
            localStorage.setItem('earthquake_alert_history', JSON.stringify(this.alertHistory));
        } catch (error) {
            console.warn('Could not save alert history:', error);
        }
    }

    loadAlertHistory() {
        try {
            const saved = localStorage.getItem('earthquake_alert_history');
            if (saved) {
                this.alertHistory = JSON.parse(saved);
                this.updateAlertHistoryDisplay();
            }
        } catch (error) {
            console.warn('Could not load alert history:', error);
        }
    }

    // Public API methods
    addAlertHandler(handler) {
        if (typeof handler === 'function') {
            this.alertHandlers.push(handler);
        }
    }

    removeAlertHandler(handler) {
        const index = this.alertHandlers.indexOf(handler);
        if (index > -1) {
            this.alertHandlers.splice(index, 1);
        }
    }

    triggerAlertHandlers(alertData) {
        this.alertHandlers.forEach(handler => {
            try {
                handler(alertData);
            } catch (error) {
                console.error('Alert handler error:', error);
            }
        });
    }

    getAlertHistory() {
        return [...this.alertHistory];
    }

    isConnectedToAlerts() {
        return this.isConnected;
    }

    // Test method for development
    simulateAlert(severity = 'medium') {
        const testAlert = {
            alert_id: `test_${Date.now()}`,
            timestamp: new Date().toISOString(),
            alert_type: 'magnitude_threshold',
            severity: severity,
            title: `Test ${severity.charAt(0).toUpperCase() + severity.slice(1)} Alert`,
            message: `This is a test alert with ${severity} severity level.`,
            analysis_data: {
                magnitude: 4.5 + Math.random() * 3,
                wave_type: 'P',
                confidence: 0.8 + Math.random() * 0.2
            },
            threshold_exceeded: {
                magnitude: Math.random() * 2
            }
        };
        
        this.handleIncomingAlert(testAlert);
    }
}

// CSS styles for alert system
const alertStyles = `
<style>
.alert-container {
    position: fixed;
    top: 20px;
    left: 20px;
    width: 400px;
    z-index: 9999;
    pointer-events: none;
}

.alert-notification {
    pointer-events: all;
    margin-bottom: 10px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    backdrop-filter: blur(10px);
}

.alert-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.alert-severity {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
}

.alert-severity.low { background: #28a745; }
.alert-severity.medium { background: #ffc107; color: #000; }
.alert-severity.high { background: #fd7e14; }
.alert-severity.critical { background: #dc3545; }

.alert-time {
    font-size: 12px;
    opacity: 0.8;
}

.alert-close {
    background: none;
    border: none;
    color: white;
    font-size: 20px;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.alert-close:hover {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
}

.alert-title {
    margin: 0 0 5px 0;
    font-size: 16px;
    font-weight: bold;
}

.alert-message {
    margin: 0 0 10px 0;
    font-size: 14px;
    line-height: 1.4;
}

.alert-details {
    font-size: 12px;
    opacity: 0.9;
}

.alert-detail {
    margin: 2px 0;
}

.alert-history-panel {
    font-size: 14px;
}

.alert-panel-header {
    padding: 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.alert-panel-header h3 {
    margin: 0;
    font-size: 16px;
}

.alert-controls {
    display: flex;
    gap: 8px;
}

.alert-controls .btn {
    padding: 4px 8px;
    font-size: 12px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    background: transparent;
    color: white;
    border-radius: 4px;
    cursor: pointer;
}

.alert-controls .btn:hover {
    background: rgba(255, 255, 255, 0.1);
}

.alert-panel-body {
    max-height: 350px;
    overflow-y: auto;
    padding: 10px;
}

.alert-history-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.alert-history-item {
    padding: 10px;
    border-radius: 6px;
    border-left: 3px solid;
    background: rgba(255, 255, 255, 0.05);
}

.alert-history-item.alert-low { border-left-color: #28a745; }
.alert-history-item.alert-medium { border-left-color: #ffc107; }
.alert-history-item.alert-high { border-left-color: #fd7e14; }
.alert-history-item.alert-critical { border-left-color: #dc3545; }

.alert-history-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
}

.alert-history-severity {
    font-size: 10px;
    font-weight: bold;
    padding: 1px 4px;
    border-radius: 2px;
    background: rgba(255, 255, 255, 0.2);
}

.alert-history-time {
    font-size: 10px;
    opacity: 0.7;
}

.alert-history-title {
    font-weight: bold;
    margin-bottom: 3px;
}

.alert-history-message {
    font-size: 12px;
    opacity: 0.8;
    line-height: 1.3;
}

.alert-panel-footer {
    padding: 10px 15px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.connection-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
}

.status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.status-indicator.connected {
    background: #28a745;
    box-shadow: 0 0 4px #28a745;
}

.status-indicator.disconnected {
    background: #dc3545;
    box-shadow: 0 0 4px #dc3545;
}

.no-alerts {
    text-align: center;
    opacity: 0.6;
    padding: 20px;
    font-style: italic;
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes slideOutRight {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}
</style>
`;

// Inject styles
document.head.insertAdjacentHTML('beforeend', alertStyles);

// Initialize alert system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize if Socket.IO is available
    if (typeof io !== 'undefined') {
        window.earthquakeAlerts = new EarthquakeAlertSystem();
        console.log('Earthquake Alert System initialized');
    } else {
        console.warn('Socket.IO not available - Alert system disabled');
    }
});