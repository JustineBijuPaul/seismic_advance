/**
 * Educational System for Wave Analysis
 * Provides tooltips, guided tours, and contextual help
 */

class EducationalSystem {
    constructor() {
        this.tooltips = new Map();
        this.tourSteps = [];
        this.currentTourStep = 0;
        this.tourActive = false;
        this.educationalContent = this.initializeEducationalContent();
        
        this.init();
    }

    init() {
        this.createTooltipContainer();
        this.setupTooltips();
        this.setupGuidedTour();
        this.setupEventListeners();
        this.loadUserPreferences();
    }

    initializeEducationalContent() {
        return {
            waveTypes: {
                'p-wave': {
                    title: 'P-Waves (Primary Waves)',
                    description: 'P-waves are the fastest seismic waves and arrive first at recording stations.',
                    characteristics: [
                        'Speed: 6-8 km/s in Earth\'s crust',
                        'Motion: Compressional (push-pull)',
                        'Can travel through solids and liquids',
                        'Smaller amplitude than S-waves',
                        'First to be detected on seismograms'
                    ],
                    detection: 'Detected using STA/LTA algorithm and characteristic function analysis',
                    importance: 'Critical for initial earthquake detection and location determination',
                    examples: [
                        'Typical frequency: 8-15 Hz',
                        'Duration: Usually 1-1.5 seconds',
                        'Amplitude: 0.1-1.0 relative units'
                    ]
                },
                's-wave': {
                    title: 'S-Waves (Secondary Waves)',
                    description: 'S-waves are slower than P-waves but cause more ground shaking.',
                    characteristics: [
                        'Speed: 3-4 km/s in Earth\'s crust',
                        'Motion: Shear (side-to-side)',
                        'Cannot travel through liquids',
                        'Larger amplitude than P-waves',
                        'Arrive after P-waves'
                    ],
                    detection: 'Identified through polarization analysis and particle motion studies',
                    importance: 'Essential for magnitude estimation and damage assessment',
                    examples: [
                        'Typical frequency: 1.5-5 Hz',
                        'Duration: Usually 2-6 seconds',
                        'Amplitude: 0.5-2.0 relative units'
                    ]
                },
                'surface': {
                    title: 'Surface Waves',
                    description: 'Surface waves travel along Earth\'s surface and cause the most damage.',
                    characteristics: [
                        'Speed: 1.5-2.8 km/s (slower than body waves)',
                        'Two types: Love waves and Rayleigh waves',
                        'Love waves: horizontal shearing motion',
                        'Rayleigh waves: elliptical rolling motion',
                        'Largest amplitude and longest duration'
                    ],
                    detection: 'Identified through frequency-time analysis and group velocity calculations',
                    importance: 'Dominant in earthquake damage, used for surface wave magnitude (Ms)',
                    examples: [
                        'Typical frequency: 0.05-1 Hz',
                        'Duration: 10-60 seconds',
                        'Amplitude: 1.0-5.0 relative units'
                    ]
                }
            }
        };
    }

    createTooltipContainer() {
        if (document.getElementById('educational-tooltip')) return;

        const tooltip = document.createElement('div');
        tooltip.id = 'educational-tooltip';
        tooltip.className = 'educational-tooltip';
        tooltip.innerHTML = `
            <div class="tooltip-header">
                <h4 class="tooltip-title"></h4>
                <button class="tooltip-close" onclick="educationalSystem.hideTooltip()">×</button>
            </div>
            <div class="tooltip-content">
                <div class="tooltip-description"></div>
                <div class="tooltip-details"></div>
                <div class="tooltip-examples"></div>
            </div>
            <div class="tooltip-footer">
                <button class="tooltip-learn-more" onclick="educationalSystem.showDetailedInfo()">Learn More</button>
                <button class="tooltip-dismiss" onclick="educationalSystem.dismissTooltip()">Got it</button>
            </div>
        `;
        document.body.appendChild(tooltip);

        // Add CSS styles
        this.addTooltipStyles();
    }

    addTooltipStyles() {
        if (document.getElementById('educational-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'educational-styles';
        styles.textContent = `
            .educational-tooltip {
                position: fixed;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 1px solid rgba(109, 93, 252, 0.3);
                border-radius: 12px;
                padding: 0;
                max-width: 400px;
                min-width: 300px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                z-index: 10000;
                opacity: 0;
                transform: translateY(-10px);
                transition: all 0.3s ease;
                pointer-events: none;
                color: white;
            }

            .educational-tooltip.visible {
                opacity: 1;
                transform: translateY(0);
                pointer-events: all;
            }

            .tooltip-header {
                background: linear-gradient(135deg, #6d5dfc 0%, #8b5cf6 100%);
                padding: 1rem;
                border-radius: 12px 12px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .tooltip-title {
                margin: 0;
                font-size: 1.1rem;
                font-weight: 600;
                color: white;
            }

            .tooltip-close {
                background: none;
                border: none;
                color: white;
                font-size: 1.5rem;
                cursor: pointer;
                padding: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                transition: background-color 0.2s;
            }

            .tooltip-close:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }

            .tooltip-content {
                padding: 1rem;
            }

            .tooltip-description {
                margin-bottom: 1rem;
                line-height: 1.5;
                color: rgba(255, 255, 255, 0.9);
            }

            .tooltip-details {
                margin-bottom: 1rem;
            }

            .tooltip-details ul {
                margin: 0.5rem 0;
                padding-left: 1.2rem;
            }

            .tooltip-details li {
                margin-bottom: 0.3rem;
                color: rgba(255, 255, 255, 0.8);
                font-size: 0.9rem;
            }

            .tooltip-examples {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 6px;
                padding: 0.8rem;
                margin-bottom: 1rem;
                font-size: 0.85rem;
                color: rgba(255, 255, 255, 0.7);
            }

            .tooltip-footer {
                padding: 1rem;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
                display: flex;
                gap: 0.5rem;
                justify-content: flex-end;
            }

            .tooltip-learn-more,
            .tooltip-dismiss {
                padding: 0.5rem 1rem;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.85rem;
                transition: all 0.2s;
            }

            .tooltip-learn-more {
                background: linear-gradient(135deg, #6d5dfc 0%, #8b5cf6 100%);
                color: white;
            }

            .tooltip-learn-more:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(109, 93, 252, 0.4);
            }

            .tooltip-dismiss {
                background: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.8);
            }

            .tooltip-dismiss:hover {
                background: rgba(255, 255, 255, 0.2);
            }

            .help-indicator {
                position: relative;
                display: inline-block;
                cursor: help;
            }

            .help-indicator::after {
                content: '?';
                position: absolute;
                top: -8px;
                right: -8px;
                width: 16px;
                height: 16px;
                background: #6d5dfc;
                color: white;
                border-radius: 50%;
                font-size: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }

            .help-indicator:hover::after {
                background: #8b5cf6;
            }
        `;
        document.head.appendChild(styles);
    }

    setupTooltips() {
        // Add help indicators to key elements
        this.addHelpIndicator('.wave-type-btn[data-wave-type="p-wave"]', 'p-wave');
        this.addHelpIndicator('.wave-type-btn[data-wave-type="s-wave"]', 's-wave');
        this.addHelpIndicator('.wave-type-btn[data-wave-type="surface"]', 'surface');
        
        // Add contextual tooltips to analysis parameters
        this.addParameterTooltips();
    }

    addHelpIndicator(selector, contentKey) {
        const element = document.querySelector(selector);
        if (!element) return;

        element.classList.add('help-indicator');
        element.addEventListener('mouseenter', (e) => {
            this.showTooltip(e.target, contentKey);
        });
        element.addEventListener('mouseleave', () => {
            this.hideTooltip();
        });
        element.addEventListener('click', (e) => {
            e.preventDefault();
            this.showDetailedTooltip(contentKey);
        });
    }

    addParameterTooltips() {
        const parameters = [
            { selector: '#samplingRate', key: 'sampling-rate' },
            { selector: '#minSnr', key: 'snr' },
            { selector: '#minConfidence', key: 'confidence' },
            { selector: '#filterFreq', key: 'frequency' }
        ];

        parameters.forEach(param => {
            const element = document.querySelector(param.selector);
            if (element) {
                const label = element.previousElementSibling;
                if (label && label.tagName === 'LABEL') {
                    label.classList.add('help-indicator');
                    label.addEventListener('click', () => {
                        this.showParameterHelp(param.key);
                    });
                }
            }
        });
    }

    showTooltip(element, contentKey) {
        const tooltip = document.getElementById('educational-tooltip');
        if (!tooltip) return;

        const content = this.getContentByKey(contentKey);
        if (!content) return;

        this.populateTooltip(content);
        this.positionTooltip(tooltip, element);
        
        tooltip.classList.add('visible');
    }

    hideTooltip() {
        const tooltip = document.getElementById('educational-tooltip');
        if (tooltip) {
            tooltip.classList.remove('visible');
        }
    }

    showDetailedTooltip(contentKey) {
        const content = this.getContentByKey(contentKey);
        if (!content) return;

        this.populateTooltip(content, true);
        
        const tooltip = document.getElementById('educational-tooltip');
        tooltip.style.position = 'fixed';
        tooltip.style.top = '50%';
        tooltip.style.left = '50%';
        tooltip.style.transform = 'translate(-50%, -50%)';
        tooltip.classList.add('visible');
    }

    populateTooltip(content, detailed = false) {
        const tooltip = document.getElementById('educational-tooltip');
        const title = tooltip.querySelector('.tooltip-title');
        const description = tooltip.querySelector('.tooltip-description');
        const details = tooltip.querySelector('.tooltip-details');
        const examples = tooltip.querySelector('.tooltip-examples');

        title.textContent = content.title;
        description.textContent = content.description || content.definition;

        if (content.characteristics) {
            details.innerHTML = `<ul>${content.characteristics.map(char => `<li>${char}</li>`).join('')}</ul>`;
        } else {
            details.innerHTML = '';
        }

        if (content.examples) {
            examples.innerHTML = `<strong>Examples:</strong><br>${content.examples.join('<br>')}`;
        } else {
            examples.innerHTML = '';
        }
    }

    positionTooltip(tooltip, element) {
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        let top = rect.bottom + 10;
        let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);

        // Adjust if tooltip goes off screen
        if (left < 10) left = 10;
        if (left + tooltipRect.width > window.innerWidth - 10) {
            left = window.innerWidth - tooltipRect.width - 10;
        }
        if (top + tooltipRect.height > window.innerHeight - 10) {
            top = rect.top - tooltipRect.height - 10;
        }

        tooltip.style.top = `${top}px`;
        tooltip.style.left = `${left}px`;
    }

    getContentByKey(key) {
        // Check wave types
        if (this.educationalContent.waveTypes[key]) {
            return this.educationalContent.waveTypes[key];
        }
        
        return null;
    }

    setupGuidedTour() {
        this.tourSteps = [
            {
                target: '.dashboard-header',
                title: 'Welcome to Wave Analysis',
                content: 'This dashboard helps you analyze seismic waves from earthquake data. Let\'s take a quick tour of the key features.',
                position: 'bottom'
            },
            {
                target: '.wave-type-selector',
                title: 'Wave Type Selection',
                content: 'Choose which types of seismic waves to analyze. P-waves arrive first, S-waves cause more shaking, and surface waves cause the most damage.',
                position: 'bottom'
            },
            {
                target: '.analysis-parameters',
                title: 'Analysis Parameters',
                content: 'Adjust these settings to fine-tune the wave detection algorithms. Higher SNR values improve accuracy but may miss weaker signals.',
                position: 'top'
            },
            {
                target: '#mainChart',
                title: 'Visualization Area',
                content: 'This chart displays your seismic data and detected wave arrivals. You can switch between waveform, frequency, and spectrogram views.',
                position: 'top'
            },
            {
                target: '.analysis-results',
                title: 'Analysis Results',
                content: 'View detailed information about detected waves, including arrival times, magnitudes, and quality metrics.',
                position: 'top'
            },
            {
                target: '.educational-panel',
                title: 'Educational Information',
                content: 'Learn about different wave types and analysis concepts. Click on the info buttons to get detailed explanations.',
                position: 'top'
            },
            {
                target: '.action-toolbar',
                title: 'Analysis Tools',
                content: 'Use these buttons to start analysis, export results, or generate reports. You can also drag and drop files directly onto the dashboard.',
                position: 'top'
            }
        ];

        // Add tour trigger button
        this.addTourButton();
    }

    addTourButton() {
        const tourButton = document.createElement('button');
        tourButton.id = 'start-tour-btn';
        tourButton.className = 'tour-trigger-btn';
        tourButton.innerHTML = '🎓 Take Tour';
        tourButton.onclick = () => this.startGuidedTour();

        // Add to dashboard header
        const header = document.querySelector('.dashboard-header');
        if (header) {
            tourButton.style.cssText = `
                position: absolute;
                top: 1rem;
                right: 1rem;
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                cursor: pointer;
                font-size: 0.85rem;
                transition: all 0.2s;
            `;
            header.style.position = 'relative';
            header.appendChild(tourButton);
        }
    }

    startGuidedTour() {
        this.tourActive = true;
        this.currentTourStep = 0;
        this.createTourOverlay();
        this.showTourStep();
    }

    createTourOverlay() {
        let overlay = document.getElementById('tour-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'tour-overlay';
            overlay.className = 'tour-overlay';
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                z-index: 9999;
                display: none;
            `;
            document.body.appendChild(overlay);
        }
        overlay.style.display = 'block';
    }

    showTourStep() {
        if (this.currentTourStep >= this.tourSteps.length) {
            this.endTour();
            return;
        }

        const step = this.tourSteps[this.currentTourStep];
        const target = document.querySelector(step.target);
        
        if (!target) {
            this.nextTourStep();
            return;
        }

        this.highlightElement(target);
        this.showTourPopup(step, target);
    }

    highlightElement(element) {
        // Remove previous highlights
        document.querySelectorAll('.tour-spotlight').forEach(el => el.remove());

        const rect = element.getBoundingClientRect();
        const spotlight = document.createElement('div');
        spotlight.className = 'tour-spotlight';
        spotlight.style.cssText = `
            position: absolute;
            border: 3px solid #6d5dfc;
            border-radius: 8px;
            box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.7);
            transition: all 0.3s ease;
            top: ${rect.top - 10}px;
            left: ${rect.left - 10}px;
            width: ${rect.width + 20}px;
            height: ${rect.height + 20}px;
        `;
        document.body.appendChild(spotlight);
    }

    showTourPopup(step, target) {
        let popup = document.getElementById('tour-popup');
        if (!popup) {
            popup = document.createElement('div');
            popup.id = 'tour-popup';
            popup.className = 'tour-popup';
            popup.style.cssText = `
                position: fixed;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border: 1px solid rgba(109, 93, 252, 0.3);
                border-radius: 12px;
                padding: 1.5rem;
                max-width: 350px;
                color: white;
                z-index: 10001;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            `;
            document.body.appendChild(popup);
        }

        popup.innerHTML = `
            <h4 style="margin: 0 0 1rem 0; color: #6d5dfc; font-size: 1.2rem;">${step.title}</h4>
            <p style="margin-bottom: 1rem; line-height: 1.5; color: rgba(255, 255, 255, 0.9);">${step.content}</p>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                <div style="font-size: 0.85rem; color: rgba(255, 255, 255, 0.7);">${this.currentTourStep + 1} of ${this.tourSteps.length}</div>
                <div style="display: flex; gap: 0.5rem;">
                    <button onclick="educationalSystem.endTour()" style="padding: 0.5rem 1rem; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; background: rgba(255, 255, 255, 0.1); color: rgba(255, 255, 255, 0.8);">Skip Tour</button>
                    ${this.currentTourStep > 0 ? '<button onclick="educationalSystem.previousTourStep()" style="padding: 0.5rem 1rem; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; background: rgba(255, 255, 255, 0.1); color: rgba(255, 255, 255, 0.8);">Previous</button>' : ''}
                    <button onclick="educationalSystem.nextTourStep()" style="padding: 0.5rem 1rem; border: none; border-radius: 6px; cursor: pointer; font-size: 0.85rem; background: linear-gradient(135deg, #6d5dfc 0%, #8b5cf6 100%); color: white;">${this.currentTourStep === this.tourSteps.length - 1 ? 'Finish' : 'Next'}</button>
                </div>
            </div>
        `;

        this.positionTourPopup(popup, target, step.position);
    }

    positionTourPopup(popup, target, position) {
        const rect = target.getBoundingClientRect();
        const popupRect = popup.getBoundingClientRect();
        
        let top, left;

        switch (position) {
            case 'top':
                top = rect.top - popupRect.height - 20;
                left = rect.left + (rect.width / 2) - (popupRect.width / 2);
                break;
            case 'bottom':
                top = rect.bottom + 20;
                left = rect.left + (rect.width / 2) - (popupRect.width / 2);
                break;
            default:
                top = rect.bottom + 20;
                left = rect.left + (rect.width / 2) - (popupRect.width / 2);
        }

        // Adjust if popup goes off screen
        if (left < 10) left = 10;
        if (left + popupRect.width > window.innerWidth - 10) {
            left = window.innerWidth - popupRect.width - 10;
        }
        if (top < 10) top = 10;
        if (top + popupRect.height > window.innerHeight - 10) {
            top = window.innerHeight - popupRect.height - 10;
        }

        popup.style.top = `${top}px`;
        popup.style.left = `${left}px`;
    }

    nextTourStep() {
        this.currentTourStep++;
        this.showTourStep();
    }

    previousTourStep() {
        if (this.currentTourStep > 0) {
            this.currentTourStep--;
            this.showTourStep();
        }
    }

    endTour() {
        this.tourActive = false;
        this.currentTourStep = 0;
        
        // Clean up tour elements
        const overlay = document.getElementById('tour-overlay');
        const popup = document.getElementById('tour-popup');
        const spotlights = document.querySelectorAll('.tour-spotlight');
        
        if (overlay) overlay.style.display = 'none';
        if (popup) popup.remove();
        spotlights.forEach(el => el.remove());

        // Save tour completion
        localStorage.setItem('waveAnalysisTourCompleted', 'true');
    }

    setupEventListeners() {
        // Hide tooltips when clicking elsewhere
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.educational-tooltip') && !e.target.closest('.help-indicator')) {
                this.hideTooltip();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideTooltip();
                if (this.tourActive) {
                    this.endTour();
                }
            }
            if (e.key === 'F1') {
                e.preventDefault();
                this.startGuidedTour();
            }
        });
    }

    showParameterHelp(paramKey) {
        const helpContent = {
            'sampling-rate': {
                title: 'Sampling Rate',
                description: 'The number of data points recorded per second. Higher rates capture more detail but require more processing power.',
                examples: ['Use 100 Hz for most earthquake analysis', '1000 Hz for detailed studies']
            },
            'snr': {
                title: 'Signal-to-Noise Ratio',
                description: 'Minimum ratio between signal strength and background noise for reliable detection.',
                examples: ['Use 2.0 for sensitive detection', '5.0 for high-confidence results']
            },
            'confidence': {
                title: 'Detection Confidence',
                description: 'Minimum confidence score required for wave detection algorithms.',
                examples: ['Use 0.3 for exploratory analysis', '0.7 for reliable results']
            },
            'frequency': {
                title: 'Filter Frequency',
                description: 'Maximum frequency to include in analysis. Filters out high-frequency noise.',
                examples: ['Use 20 Hz for regional earthquakes', '50 Hz for local events']
            }
        };

        const content = helpContent[paramKey];
        if (content) {
            this.populateTooltip(content);
            const tooltip = document.getElementById('educational-tooltip');
            tooltip.style.position = 'fixed';
            tooltip.style.top = '50%';
            tooltip.style.left = '50%';
            tooltip.style.transform = 'translate(-50%, -50%)';
            tooltip.classList.add('visible');
        }
    }

    loadUserPreferences() {
        // Check if user has completed tour
        const tourCompleted = localStorage.getItem('waveAnalysisTourCompleted');
        if (!tourCompleted) {
            // Show tour prompt after a delay
            setTimeout(() => {
                if (confirm('Would you like to take a guided tour of the wave analysis features?')) {
                    this.startGuidedTour();
                }
            }, 2000);
        }
    }

    // Public methods for external access
    showDetailedInfo() {
        console.log('Show detailed information');
    }

    dismissTooltip() {
        this.hideTooltip();
    }

    // Method to validate educational content accuracy
    validateContent() {
        const validationResults = {
            waveTypes: {}
        };

        // Validate wave type information
        Object.keys(this.educationalContent.waveTypes).forEach(key => {
            const content = this.educationalContent.waveTypes[key];
            validationResults.waveTypes[key] = {
                hasTitle: !!content.title,
                hasDescription: !!content.description,
                hasCharacteristics: Array.isArray(content.characteristics) && content.characteristics.length > 0,
                hasDetection: !!content.detection,
                hasImportance: !!content.importance,
                hasExamples: Array.isArray(content.examples) && content.examples.length > 0
            };
        });

        return validationResults;
    }
}

// Initialize educational system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.educationalSystem = new EducationalSystem();
});

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EducationalSystem;
}