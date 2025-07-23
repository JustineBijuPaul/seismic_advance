# Import required libraries
import os                          # For operating system operations
from flask import Flask, request, render_template, jsonify  # Core Flask functionality
from flask_socketio import SocketIO, emit  # WebSocket support
import numpy as np                 # For numerical operations
import librosa                     # For signal processing
import sklearn.preprocessing as preprocessing  # For data preprocessing
from sklearn.linear_model import LogisticRegression  # ML model
import joblib                      # For loading saved models
from datetime import datetime, timedelta  # For time operations
from whitenoise import WhiteNoise  # For serving static files
from pymongo import MongoClient    # MongoDB database connection
import gridfs                      # For storing large files in MongoDB
from dotenv import load_dotenv     # For loading environment variables
import tempfile                    # For temporary file operations
import csv                         # For CSV file operations
import obspy                       # For seismic data processing
from obspy.core import Trace, Stream  # For seismic data structures
from xml.etree.ElementTree import Element, SubElement, tostring  # For XML creation
from xml.dom.minidom import parseString  # For XML formatting
from flask import send_file        # For file downloads
import io                          # For in-memory file operations
import base64                      # For encoding/decoding base64 data
import requests
import threading
import time
import soundfile as sf
import warnings
import sounddevice as sd
import scipy.io.wavfile as wav
warnings.filterwarnings('ignore')

# Import wave analysis configuration and logging
from wave_analysis.config import config_manager
from wave_analysis.logging_config import wave_logger, health_monitor, performance_monitor

# Initialize Flask application
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
application = app  # For WSGI compatibility

# Configure Flask app with deployment settings
app.url_map.strict_slashes = False
app.config['SECRET_KEY'] = config_manager.deployment.secret_key
app.config['DEBUG'] = config_manager.deployment.debug

# Initialize SocketIO for real-time communication with CORS settings
cors_origins = config_manager.deployment.allowed_origins if not config_manager.deployment.enable_cors else "*"
socketio = SocketIO(app, cors_allowed_origins=cors_origins, async_mode='threading')

# Configure WhiteNoise for serving static files
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

# Get wave analysis configuration parameters
wave_config = config_manager.get_wave_analysis_params()

# Define constants for signal processing from configuration
SAMPLE_RATE = int(wave_config['sampling_rate'])
N_MELS = 128         # Number of Mel frequency bands
FMIN = 0             # Minimum frequency for analysis
FMAX = 19            # Maximum frequency for analysis
FRAME_SIZE = 512     # Size of each frame for spectrogram
HOP_LENGTH = 256     # Number of samples between frames

# Load pre-trained model and scaler
clf = joblib.load('earthquake_model.joblib')
scaler = joblib.load('earthquake_scaler.joblib')

# Load environment variables and setup MongoDB connection
load_dotenv()
MONGO_URI = os.getenv("MONGO_URL")

# Get database configuration
db_config = config_manager.get_database_params()
DB_NAME = 'seismic_quake'

# Initialize MongoDB client with configuration parameters
client = MongoClient(
    MONGO_URI,
    connectTimeoutMS=db_config['connection_timeout_ms'],
    socketTimeoutMS=db_config['socket_timeout_ms'],
    maxPoolSize=db_config['max_pool_size'],
    minPoolSize=db_config['min_pool_size'],
    maxIdleTimeMS=db_config['max_idle_time_ms']
)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

# Initialize application logger
app_logger = wave_logger.get_logger('app')
app_logger.info(f"Application starting in {config_manager.deployment.environment} environment")

# Initialize wave analysis repository
from wave_analysis.models import WaveAnalysisRepository
wave_analysis_repo = WaveAnalysisRepository(db, fs)

# Initialize analysis cache manager
from wave_analysis.services import AnalysisCacheManager
try:
    import redis
    redis_client = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
    # Test Redis connection
    redis_client.ping()
    app_logger.info("Redis cache backend connected successfully")
except Exception as e:
    app_logger.warning(f"Redis not available, using MongoDB-only caching: {e}")
    redis_client = None

analysis_cache_manager = AnalysisCacheManager(
    mongodb=db,
    redis_client=redis_client,
    default_ttl_hours=24,
    max_memory_cache_size=100
)

# Add global variable for continuous monitoring
monitoring_active = False

# Global variables for async processing
processing_tasks = {}  # Store task information
task_counter = 0

# Import wave analysis components
try:
    from wave_analysis import WaveAnalysisResult, DetailedAnalysis
    from wave_analysis.services import WaveAnalyzer
    from wave_analysis.services.cached_wave_analyzer import CachedWaveAnalyzer
    from wave_analysis.services.wave_separation_engine import WaveSeparationEngine, WaveSeparationParameters
    from wave_analysis.services.alert_system import (
        AlertSystem, WebSocketAlertHandler, LogAlertHandler
    )
    WAVE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Wave analysis components not available: {e}")
    WAVE_ANALYSIS_AVAILABLE = False

# Initialize Alert System
alert_system = None
if WAVE_ANALYSIS_AVAILABLE:
    alert_system = AlertSystem()
    # Add WebSocket handler for real-time alerts
    websocket_handler = WebSocketAlertHandler(socketio)
    alert_system.add_handler(websocket_handler)
    # Add logging handler for alert history
    log_handler = LogAlertHandler()
    alert_system.add_handler(log_handler)

@performance_monitor('extract_features')
def extract_features(file_id):
    """Extract features from stored file for prediction"""
    try:
        # Retrieve file from GridFS and save to temporary file
        with fs.get(file_id) as f:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(f.read())
                temp_file_path = temp_file.name

        try:
            # Try reading with scipy first (faster and more reliable for WAV)
            sr, y = wav.read(temp_file_path)
            y = y.astype(np.float32)
            if len(y.shape) > 1:  # Convert stereo to mono
                y = y.mean(axis=1)
        except:
            try:
                # Try soundfile next
                y, sr = sf.read(temp_file_path)
                if len(y.shape) > 1:  # Convert stereo to mono
                    y = y.mean(axis=1)
            except:
                # Fallback to librosa as last resort
                y, sr = librosa.load(temp_file_path, sr=SAMPLE_RATE, mono=True)

        # Ensure minimum length and resample if needed
        min_length = 1024  # Minimum length for processing
        if len(y) < min_length:
            y = np.pad(y, (0, min_length - len(y)))
        
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # Calculate adaptive window size
        n_fft = min(512, len(y))
        hop_length = n_fft // 4  # 75% overlap

        # Calculate Mel spectrogram with adjusted parameters
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=N_MELS, 
            fmin=FMIN, 
            fmax=min(FMAX, sr//2),
            n_fft=n_fft, 
            hop_length=hop_length,
            power=2.0
        )
        
        # Convert to log scale
        log_S = librosa.power_to_db(S, ref=np.max)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            S=log_S, 
            n_mfcc=13,
            dct_type=2,
            norm='ortho'
        )

        # Clean up temporary file
        os.remove(temp_file_path)
        
        return np.mean(mfcc, axis=1), y, sr
    
    except Exception as e:
        app.logger.error(f"Error in feature extraction: {str(e)}")
        raise

def predict(file_id):
    """Make prediction and identify earthquake events"""
    # Extract features and make prediction
    features, y, sr = extract_features(file_id)
    features = scaler.transform([features])
    prediction = clf.predict(features)
    print(prediction)

    # Identify potential earthquake events using threshold
    threshold = np.mean(y) + 3 * np.std(y)
    earthquake_indices = np.where(y > threshold)[0]

    return prediction[0], y, sr, earthquake_indices

def perform_wave_analysis(file_id, seismic_data, sampling_rate):
    """Perform wave analysis on seismic data"""
    if not WAVE_ANALYSIS_AVAILABLE:
        raise Exception("Wave analysis components not available")
    
    try:
        # Initialize wave separation engine with default parameters
        separation_params = WaveSeparationParameters(
            sampling_rate=sampling_rate,
            min_snr=2.0,
            min_detection_confidence=0.3
        )
        
        wave_engine = WaveSeparationEngine(separation_params)
        
        # Perform wave separation
        separation_result = wave_engine.separate_waves(seismic_data)
        
        # Initialize wave analyzer for detailed analysis with caching
        base_analyzer = WaveAnalyzer(sampling_rate)
        wave_analyzer = CachedWaveAnalyzer(
            wave_analyzer=base_analyzer,
            wave_separation_engine=wave_engine,
            cache_manager=analysis_cache_manager,
            enable_caching=True
        )
        
        # Perform comprehensive analysis
        detailed_analysis = wave_analyzer.analyze_waves(separation_result.wave_analysis_result)
        
        # Format simplified response for upload endpoint
        return {
            'wave_separation': {
                'p_waves_count': len(separation_result.wave_analysis_result.p_waves),
                's_waves_count': len(separation_result.wave_analysis_result.s_waves),
                'surface_waves_count': len(separation_result.wave_analysis_result.surface_waves)
            },
            'arrival_times': {
                'p_wave_arrival': detailed_analysis.arrival_times.p_wave_arrival,
                's_wave_arrival': detailed_analysis.arrival_times.s_wave_arrival,
                'sp_time_difference': detailed_analysis.arrival_times.sp_time_difference
            },
            'magnitude_estimates': [{
                'method': est.method,
                'magnitude': est.magnitude,
                'confidence': est.confidence
            } for est in detailed_analysis.magnitude_estimates[:3]],  # Limit to top 3
            'quality_score': separation_result.quality_metrics.analysis_quality_score
        }
        
    except Exception as e:
        app.logger.error(f"Wave analysis error: {str(e)}")
        raise

def start_async_analysis(file_id, enable_wave_analysis):
    """Start asynchronous analysis task"""
    global task_counter, processing_tasks
    
    task_counter += 1
    task_id = f"task_{task_counter}_{int(time.time())}"
    
    # Initialize task status
    processing_tasks[task_id] = {
        'status': 'queued',
        'file_id': file_id,
        'enable_wave_analysis': enable_wave_analysis,
        'created_at': datetime.utcnow(),
        'progress': 0,
        'message': 'Task queued for processing'
    }
    
    # Start processing in background thread
    def async_analysis_worker():
        try:
            # Update status to processing
            processing_tasks[task_id].update({
                'status': 'processing',
                'progress': 10,
                'message': 'Starting analysis...'
            })
            
            # Perform basic earthquake detection
            processing_tasks[task_id].update({
                'progress': 30,
                'message': 'Performing earthquake detection...'
            })
            
            prediction, y, sr, earthquake_indices = predict(file_id)
            time_labels = [str(timedelta(seconds=i / sr)) for i in range(len(y))]
            
            # Prepare basic results
            result_data = {
                'file_id': str(file_id),
                'prediction': 'Seismic Activity Detected' if prediction == 1 else 'No Seismic Activity Detected',
                'time_labels': time_labels,
                'amplitude_data': y.tolist(),
                'sampling_rate': sr,
                'async': True,
                'wave_analysis_enabled': enable_wave_analysis
            }
            
            if prediction == 1:
                result_data.update({
                    'time_indices': earthquake_indices.tolist(),
                    'amplitudes': [float(y[idx]) for idx in earthquake_indices]
                })
                
                # Perform wave analysis if enabled and seismic activity detected
                if enable_wave_analysis and WAVE_ANALYSIS_AVAILABLE:
                    processing_tasks[task_id].update({
                        'progress': 60,
                        'message': 'Performing wave analysis...'
                    })
                    
                    try:
                        wave_analysis_result = perform_wave_analysis(file_id, y, sr)
                        result_data['wave_analysis'] = wave_analysis_result
                    except Exception as e:
                        app.logger.warning(f"Wave analysis failed: {str(e)}")
                        result_data['wave_analysis_error'] = str(e)
            
            # Store results in database
            processing_tasks[task_id].update({
                'progress': 90,
                'message': 'Storing results...'
            })
            
            analysis_record = {
                'task_id': task_id,
                'file_id': file_id,
                'analysis_timestamp': datetime.utcnow(),
                'prediction': prediction,
                'wave_analysis_enabled': enable_wave_analysis,
                'results': result_data,
                'processing_time': (datetime.utcnow() - processing_tasks[task_id]['created_at']).total_seconds()
            }
            
            db.async_analyses.insert_one(analysis_record)
            
            # Update final status
            processing_tasks[task_id].update({
                'status': 'completed',
                'progress': 100,
                'message': 'Analysis completed successfully',
                'results': result_data,
                'completed_at': datetime.utcnow()
            })
            
        except Exception as e:
            app.logger.error(f"Async analysis failed for task {task_id}: {str(e)}")
            processing_tasks[task_id].update({
                'status': 'failed',
                'progress': 0,
                'message': f'Analysis failed: {str(e)}',
                'error': str(e),
                'failed_at': datetime.utcnow()
            })
    
    # Start the worker thread
    thread = threading.Thread(target=async_analysis_worker)
    thread.daemon = True
    thread.start()
    
    return task_id

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/documentation')
def documentation():
    """Render the documentation page"""
    return render_template('documentation.html')

@app.route('/wave_analysis_dashboard')
def wave_analysis_dashboard():
    """Render the wave analysis dashboard page"""
    return render_template('wave_analysis_dashboard.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and analysis with optional wave analysis"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            # Get analysis options from form data
            enable_wave_analysis = request.form.get('enable_wave_analysis', 'false').lower() == 'true'
            async_processing = request.form.get('async_processing', 'false').lower() == 'true'
            
            # Store file in GridFS
            file_id = fs.put(file, filename=file.filename)
            
            # Check file size for async processing decision
            file_size = len(file.read())
            file.seek(0)  # Reset file pointer
            large_file_threshold = 50 * 1024 * 1024  # 50MB threshold
            
            # Force async processing for large files
            if file_size > large_file_threshold:
                async_processing = True
            
            if async_processing:
                # Start asynchronous processing
                task_id = start_async_analysis(file_id, enable_wave_analysis)
                return jsonify({
                    'file_id': str(file_id),
                    'task_id': task_id,
                    'status': 'processing',
                    'message': 'File uploaded successfully. Analysis started in background.',
                    'async': True,
                    'wave_analysis_enabled': enable_wave_analysis
                })
            else:
                # Synchronous processing (existing behavior)
                try:
                    prediction, y, sr, earthquake_indices = predict(file_id)
                    time_labels = [str(timedelta(seconds=i / sr)) for i in range(len(y))]
                    
                    response_data = {
                        'file_id': str(file_id),
                        'prediction': 'Seismic Activity Detected' if prediction == 1 else 'No Seismic Activity Detected',
                        'time_labels': time_labels,
                        'amplitude_data': y.tolist(),
                        'sampling_rate': sr,
                        'async': False,
                        'wave_analysis_enabled': enable_wave_analysis
                    }
                    
                    if prediction == 1:
                        response_data.update({
                            'time_indices': earthquake_indices.tolist(),
                            'amplitudes': [float(y[idx]) for idx in earthquake_indices]
                        })
                        
                        # Trigger wave analysis if enabled and seismic activity detected
                        if enable_wave_analysis and WAVE_ANALYSIS_AVAILABLE:
                            try:
                                wave_analysis_result = perform_wave_analysis(file_id, y, sr)
                                response_data['wave_analysis'] = wave_analysis_result
                            except Exception as e:
                                app.logger.warning(f"Wave analysis failed: {str(e)}")
                                response_data['wave_analysis_error'] = str(e)
                    
                    return jsonify(response_data)
                    
                except Exception as e:
                    app.logger.error(f"Analysis failed: {str(e)}")
                    return jsonify({
                        'error': 'Analysis failed',
                        'message': str(e),
                        'file_id': str(file_id)
                    }), 500
                
    return render_template('upload.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start continuous monitoring"""
    global monitoring_active
    monitoring_active = True
    
    file_id = request.json.get('file_id')
    if not file_id:
        return jsonify({'error': 'No file ID provided'})

    def monitor_data():
        while monitoring_active:
            try:
                prediction, y, sr, earthquake_indices = predict(file_id)
                time_labels = [str(timedelta(seconds=i / sr)) for i in range(len(y))]
                
                # Store the latest result in the database
                db.monitoring_results.insert_one({
                    'timestamp': datetime.utcnow(),
                    'prediction': prediction,
                    'data': {
                        'time_labels': time_labels,
                        'amplitude_data': y.tolist(),
                        'earthquake_indices': earthquake_indices.tolist() if prediction == 1 else []
                    }
                })
            except Exception as e:
                print(f"Monitoring error: {e}")
            time.sleep(5)  # Check every 5 seconds

    # Start monitoring in a separate thread
    thread = threading.Thread(target=monitor_data)
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Monitoring started'})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop continuous monitoring"""
    global monitoring_active
    monitoring_active = False
    return jsonify({'status': 'Monitoring stopped'})

@app.route('/get_latest_results')
def get_latest_results():
    """Get the most recent monitoring results"""
    latest_result = db.monitoring_results.find_one(
        sort=[('timestamp', -1)]
    )
    if latest_result:
        latest_result['_id'] = str(latest_result['_id'])
        latest_result['timestamp'] = latest_result['timestamp'].isoformat()
        return jsonify(latest_result)
    return jsonify({'error': 'No results found'})

@app.route('/api/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get the status of an asynchronous analysis task"""
    if task_id not in processing_tasks:
        # Check database for completed tasks
        db_task = db.async_analyses.find_one({'task_id': task_id})
        if db_task:
            return jsonify({
                'task_id': task_id,
                'status': 'completed',
                'progress': 100,
                'message': 'Analysis completed',
                'results': db_task.get('results', {}),
                'processing_time': db_task.get('processing_time', 0)
            })
        else:
            return jsonify({'error': 'Task not found'}), 404
    
    task_info = processing_tasks[task_id].copy()
    
    # Convert datetime objects to ISO format for JSON serialization
    if 'created_at' in task_info:
        task_info['created_at'] = task_info['created_at'].isoformat()
    if 'completed_at' in task_info:
        task_info['completed_at'] = task_info['completed_at'].isoformat()
    if 'failed_at' in task_info:
        task_info['failed_at'] = task_info['failed_at'].isoformat()
    
    return jsonify(task_info)

@app.route('/api/task_results/<task_id>', methods=['GET'])
def get_task_results(task_id):
    """Get the results of a completed asynchronous analysis task"""
    # First check in-memory tasks
    if task_id in processing_tasks:
        task_info = processing_tasks[task_id]
        if task_info['status'] == 'completed':
            return jsonify(task_info.get('results', {}))
        elif task_info['status'] == 'failed':
            return jsonify({
                'error': 'Task failed',
                'message': task_info.get('message', 'Unknown error')
            }), 500
        else:
            return jsonify({
                'error': 'Task not completed',
                'status': task_info['status'],
                'progress': task_info.get('progress', 0)
            }), 202
    
    # Check database for completed tasks
    db_task = db.async_analyses.find_one({'task_id': task_id})
    if db_task:
        return jsonify(db_task.get('results', {}))
    
    return jsonify({'error': 'Task not found'}), 404

@app.route('/download_png', methods=['POST'])
def download_png():
    """Generate and send PNG visualization"""
    data = request.json
    image_base64 = data['image_base64']
    return send_file(
        io.BytesIO(base64.b64decode(image_base64.split(',')[1])),
        mimetype='image/png',
        as_attachment=True,
        download_name='waveform_chart.png'
    )

@app.route('/download_csv', methods=['POST'])
def download_csv():
    """Generate and send CSV data file"""
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']

    # Create CSV in memory
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Time', 'Amplitude'])
    cw.writerows(zip(time_labels, amplitude_data))

    # Prepare for download
    output = io.BytesIO()
    output.write(si.getvalue().encode())
    output.seek(0)
    si.close()

    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name='waveform_data.csv'
    )

@app.route('/download_mseed', methods=['POST'])
def download_mseed():
    """Generate and send MSEED format file"""
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']
    sampling_rate = data['sampling_rate']

    # Create MSEED format data
    trace = Trace(data=np.array(amplitude_data, dtype=np.float32), 
                 header={'sampling_rate': sampling_rate})
    stream = Stream([trace])

    # Prepare for download
    output = io.BytesIO()
    stream.write(output, format='MSEED')
    output.seek(0)

    return send_file(
        output,
        mimetype='application/octet-stream',
        as_attachment=True,
        download_name='waveform_data.mseed'
    )

@app.route('/download_xml', methods=['POST'])
def download_xml():
    """Generate and send XML format file"""
    data = request.json
    time_labels = data['time_labels']
    amplitude_data = data['amplitude_data']

    # Create XML structure
    root = Element('WaveformData')
    for time, amplitude in zip(time_labels, amplitude_data):
        entry = SubElement(root, 'Entry')
        time_elem = SubElement(entry, 'Time')
        time_elem.text = time
        amplitude_elem = SubElement(entry, 'Amplitude')
        amplitude_elem.text = str(amplitude)

    # Format XML with proper indentation
    xml_str = parseString(tostring(root)).toprettyxml(indent="  ")

    # Prepare for download
    output = io.BytesIO()
    output.write(xml_str.encode())
    output.seek(0)

    return send_file(
        output,
        mimetype='application/xml',
        as_attachment=True,
        download_name='waveform_data.xml'
    )

# Add a template filter for min/max functions
@app.template_filter('min')
def min_filter(value, other):
    return min(value, other)

@app.template_filter('max')
def max_filter(value, other):
    return max(value, other)

@app.route('/earthquake_history')
def earthquake_history():
    """Display historical earthquake data from USGS with pagination"""
    try:
        # Get query parameters with defaults
        days = max(1, request.args.get('days', default=30, type=int))
        min_magnitude = max(0.1, request.args.get('magnitude', default=0.1, type=float))
        sort_by = request.args.get('sort', default='time', type=str)
        order = request.args.get('order', default='desc', type=str)
        page = max(1, request.args.get('page', default=1, type=int))
        per_page = 25  # Number of items per page
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for USGS API
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # USGS API endpoint
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        
        # Parameters for the API request
        params = {
            'format': 'geojson',
            'starttime': start_str,
            'endtime': end_str,
            'minmagnitude': min_magnitude,
            'maxmagnitude': 10,
            'orderby': 'time',
            'limit': 1000  # Get more data for proper sorting
        }
        
        # Make API request with session
        session = requests.Session()
        session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
        earthquake_data = response.json()
        
        # Extract and process all earthquakes
        all_earthquakes = []
        for feature in earthquake_data.get('features', []):
            try:
                props = feature.get('properties', {})
                coords = feature.get('geometry', {}).get('coordinates', [0, 0, 0])
                
                if props and coords and len(coords) >= 3:
                    all_earthquakes.append({
                        'time': datetime.fromtimestamp(props.get('time', 0)/1000.0).strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp': props.get('time', 0),
                        'place': props.get('place', 'Unknown Location'),
                        'magnitude': props.get('mag', 0.0),
                        'depth': coords[2],
                        'latitude': coords[1],
                        'longitude': coords[0],
                        'url': props.get('url', '#')
                    })
            except (KeyError, IndexError, TypeError) as e:
                app.logger.warning(f"Error processing earthquake entry: {str(e)}")
                continue
        
        # Sort all earthquakes
        try:
            if sort_by == 'magnitude':
                all_earthquakes.sort(key=lambda x: float(x['magnitude']), reverse=(order == 'desc'))
            elif sort_by == 'depth':
                all_earthquakes.sort(key=lambda x: float(x['depth']), reverse=(order == 'desc'))
            elif sort_by == 'time':
                all_earthquakes.sort(key=lambda x: int(x['timestamp']), reverse=(order == 'desc'))
        except (KeyError, ValueError) as e:
            app.logger.error(f"Error sorting earthquakes: {str(e)}")
        
        # Calculate pagination values safely
        total_items = len(all_earthquakes)
        total_pages = max(1, (total_items + per_page - 1) // per_page)
        page = min(max(1, page), total_pages)  # Ensure page is within valid range
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        if end_idx > total_items:
            end_idx = total_items
        
        # Get current page's earthquakes
        current_earthquakes = all_earthquakes[start_idx:end_idx]
        
        return render_template('earthquake_history.html', 
                             earthquakes=current_earthquakes,
                             days=days,
                             min_magnitude=min_magnitude,
                             sort_by=sort_by,
                             order=order,
                             current_page=page,
                             total_pages=total_pages,
                             total_items=total_items,
                             per_page=per_page,
                             start_idx=start_idx,
                             end_idx=end_idx)
    
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching earthquake data: {str(e)}")
        return render_template('error.html', 
                             error_message="Unable to fetch earthquake data from USGS. Please try again later.")
    except Exception as e:
        app.logger.error(f"Unexpected error in earthquake_history: {str(e)}")
        return render_template('error.html', 
                             error_message="An unexpected error occurred. Please try again later.")

# WebSocket event handlers for real-time alerts
@socketio.on('connect', namespace='/alerts')
def handle_connect():
    """Handle client connection to alerts namespace"""
    emit('status', {'msg': 'Connected to earthquake alerts'})
    app.logger.info('Client connected to alerts namespace')

@socketio.on('disconnect', namespace='/alerts')
def handle_disconnect():
    """Handle client disconnection from alerts namespace"""
    app.logger.info('Client disconnected from alerts namespace')

@socketio.on('subscribe_alerts', namespace='/alerts')
def handle_subscribe_alerts(data):
    """Handle client subscription to specific alert types"""
    alert_types = data.get('alert_types', [])
    severities = data.get('severities', [])
    emit('subscription_confirmed', {
        'alert_types': alert_types,
        'severities': severities,
        'msg': 'Subscribed to earthquake alerts'
    })
    app.logger.info(f'Client subscribed to alerts: types={alert_types}, severities={severities}')

# Alert system API endpoints
@app.route('/api/alerts/recent', methods=['GET'])
def get_recent_alerts():
    """Get recent alerts from the alert system"""
    if not alert_system:
        return jsonify({'error': 'Alert system not available'}), 503
    
    try:
        limit = request.args.get('limit', default=50, type=int)
        limit = max(1, min(limit, 200))  # Limit between 1 and 200
        
        recent_alerts = alert_system.get_recent_alerts(limit)
        return jsonify({
            'alerts': recent_alerts,
            'count': len(recent_alerts)
        })
    except Exception as e:
        app.logger.error(f"Error getting recent alerts: {str(e)}")
        return jsonify({'error': 'Failed to retrieve alerts'}), 500

@app.route('/api/alerts/statistics', methods=['GET'])
def get_alert_statistics():
    """Get alert system statistics"""
    if not alert_system:
        return jsonify({'error': 'Alert system not available'}), 503
    
    try:
        stats = alert_system.get_alert_statistics()
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error getting alert statistics: {str(e)}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

@app.route('/api/alerts/thresholds', methods=['GET'])
def get_alert_thresholds():
    """Get current alert thresholds configuration"""
    if not alert_system:
        return jsonify({'error': 'Alert system not available'}), 503
    
    try:
        thresholds = []
        for threshold in alert_system.thresholds:
            thresholds.append({
                'alert_type': threshold.alert_type.value,
                'severity': threshold.severity.value,
                'threshold_value': threshold.threshold_value,
                'wave_types': threshold.wave_types,
                'enabled': threshold.enabled,
                'description': threshold.description
            })
        
        return jsonify({'thresholds': thresholds})
    except Exception as e:
        app.logger.error(f"Error getting alert thresholds: {str(e)}")
        return jsonify({'error': 'Failed to retrieve thresholds'}), 500

@app.route('/api/alerts/thresholds', methods=['POST'])
def update_alert_threshold():
    """Update an alert threshold"""
    if not alert_system:
        return jsonify({'error': 'Alert system not available'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        alert_type_str = data.get('alert_type')
        if not alert_type_str:
            return jsonify({'error': 'alert_type is required'}), 400
        
        # Convert string to AlertType enum
        from wave_analysis.services.alert_system import AlertType
        try:
            alert_type = AlertType(alert_type_str)
        except ValueError:
            return jsonify({'error': f'Invalid alert_type: {alert_type_str}'}), 400
        
        # Prepare update parameters
        update_params = {}
        if 'threshold_value' in data:
            update_params['threshold_value'] = float(data['threshold_value'])
        if 'enabled' in data:
            update_params['enabled'] = bool(data['enabled'])
        if 'description' in data:
            update_params['description'] = str(data['description'])
        
        # Update the threshold
        alert_system.update_threshold(alert_type, **update_params)
        
        return jsonify({'message': 'Threshold updated successfully'})
    except Exception as e:
        app.logger.error(f"Error updating alert threshold: {str(e)}")
        return jsonify({'error': 'Failed to update threshold'}), 500

@app.route('/api/alerts/clear_history', methods=['POST'])
def clear_alert_history():
    """Clear alert history"""
    if not alert_system:
        return jsonify({'error': 'Alert system not available'}), 503
    
    try:
        alert_system.clear_alert_history()
        return jsonify({'message': 'Alert history cleared successfully'})
    except Exception as e:
        app.logger.error(f"Error clearing alert history: {str(e)}")
        return jsonify({'error': 'Failed to clear alert history'}), 500

# Enhanced wave analysis endpoint with alert integration
@app.route('/api/analyze_waves', methods=['POST'])
def analyze_waves_api():
    """API endpoint for comprehensive wave analysis with alert checking"""
    if not WAVE_ANALYSIS_AVAILABLE:
        return jsonify({'error': 'Wave analysis not available'}), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        file_id = data.get('file_id')
        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400
        
        # Retrieve file and perform analysis
        try:
            # Extract features and get seismic data
            features, seismic_data, sampling_rate = extract_features(file_id)
            
            # Perform comprehensive wave analysis
            wave_analysis_result = perform_wave_analysis(file_id, seismic_data, sampling_rate)
            
            # If we have detailed analysis, check for alerts
            if alert_system and 'magnitude_estimates' in wave_analysis_result:
                # Create a mock DetailedAnalysis object for alert checking
                from wave_analysis.models import DetailedAnalysis, ArrivalTimes, MagnitudeEstimate
                
                # Convert magnitude estimates to proper objects
                magnitude_estimates = []
                for est_data in wave_analysis_result['magnitude_estimates']:
                    magnitude_estimates.append(MagnitudeEstimate(
                        method=est_data['method'],
                        magnitude=est_data['magnitude'],
                        confidence=est_data['confidence'],
                        wave_type_used='P'  # Default for now
                    ))
                
                # Create arrival times object
                arrival_times = ArrivalTimes(
                    p_wave_arrival=wave_analysis_result['arrival_times']['p_wave_arrival'],
                    s_wave_arrival=wave_analysis_result['arrival_times']['s_wave_arrival'],
                    sp_time_difference=wave_analysis_result['arrival_times']['sp_time_difference'],
                    surface_wave_arrival=0.0  # Default
                )
                
                # Create a simplified DetailedAnalysis for alert checking
                # Note: This is a simplified version - in a full implementation,
                # we would have the complete analysis object from the wave analyzer
                mock_analysis = type('MockAnalysis', (), {
                    'magnitude_estimates': magnitude_estimates,
                    'arrival_times': arrival_times,
                    'wave_result': type('MockWaveResult', (), {
                        'p_waves': [],
                        's_waves': [],
                        'surface_waves': []
                    })(),
                    'frequency_analysis': {}
                })()
                
                # Check for alerts
                triggered_alerts = alert_system.check_analysis_for_alerts(mock_analysis)
                
                # Add alert information to response
                wave_analysis_result['alerts'] = {
                    'triggered_count': len(triggered_alerts),
                    'alerts': [alert.to_dict() for alert in triggered_alerts]
                }
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'analysis': wave_analysis_result
            })
            
        except Exception as e:
            app.logger.error(f"Wave analysis failed: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
            
    except Exception as e:
        app.logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Use SocketIO's run method instead of app.run for WebSocket support
    socketio.run(app, debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

@app.route('/api/analyze_waves', methods=['POST'])
def analyze_waves():
    """
    API endpoint for comprehensive wave analysis.
    
    Expected JSON payload:
    {
        "file_id": "string",  # GridFS file ID
        "parameters": {       # Optional analysis parameters
            "sampling_rate": 100,
            "min_snr": 2.0,
            "min_detection_confidence": 0.3
        }
    }
    """
    if not WAVE_ANALYSIS_AVAILABLE:
        return jsonify({
            'error': 'Wave analysis components not available',
            'message': 'Advanced wave analysis features are not installed'
        }), 503
    
    try:
        # Parse request data
        try:
            data = request.get_json(force=True)
        except Exception as e:
            return jsonify({'error': 'Invalid JSON data', 'message': str(e)}), 400
        
        if data is None:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        file_id = data.get('file_id')
        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400
        
        # Get analysis parameters
        params = data.get('parameters', {})
        sampling_rate = params.get('sampling_rate', SAMPLE_RATE)
        min_snr = params.get('min_snr', 2.0)
        min_detection_confidence = params.get('min_detection_confidence', 0.3)
        
        # Retrieve and load seismic data from GridFS
        try:
            with fs.get(file_id) as f:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file.write(f.read())
                    temp_file_path = temp_file.name
            
            # Load audio data
            try:
                sr, seismic_data = wav.read(temp_file_path)
                seismic_data = seismic_data.astype(np.float32)
                if len(seismic_data.shape) > 1:  # Convert stereo to mono
                    seismic_data = seismic_data.mean(axis=1)
            except:
                try:
                    seismic_data, sr = sf.read(temp_file_path)
                    if len(seismic_data.shape) > 1:
                        seismic_data = seismic_data.mean(axis=1)
                except:
                    seismic_data, sr = librosa.load(temp_file_path, sr=sampling_rate, mono=True)
            
            # Clean up temporary file
            os.remove(temp_file_path)
            
        except Exception as e:
            return jsonify({
                'error': 'Failed to load seismic data',
                'message': str(e)
            }), 400
        
        # Resample if necessary
        if sr != sampling_rate:
            seismic_data = librosa.resample(seismic_data, orig_sr=sr, target_sr=sampling_rate)
            sr = sampling_rate
        
        # Initialize wave separation engine
        separation_params = WaveSeparationParameters(
            sampling_rate=sampling_rate,
            min_snr=min_snr,
            min_detection_confidence=min_detection_confidence
        )
        
        wave_engine = WaveSeparationEngine(separation_params)
        
        # Perform wave separation
        separation_result = wave_engine.separate_waves(seismic_data)
        
        # Initialize wave analyzer for detailed analysis with caching
        base_analyzer = WaveAnalyzer(sampling_rate)
        wave_analyzer = CachedWaveAnalyzer(
            wave_analyzer=base_analyzer,
            wave_separation_engine=wave_engine,
            cache_manager=analysis_cache_manager,
            enable_caching=True
        )
        
        # Perform comprehensive analysis
        detailed_analysis = wave_analyzer.analyze_waves(separation_result.wave_analysis_result)
        
        # Store analysis results in database
        analysis_id = db.wave_analyses.insert_one({
            'file_id': file_id,
            'analysis_timestamp': datetime.utcnow(),
            'parameters': params,
            'wave_separation': {
                'p_waves_count': len(separation_result.wave_analysis_result.p_waves),
                's_waves_count': len(separation_result.wave_analysis_result.s_waves),
                'surface_waves_count': len(separation_result.wave_analysis_result.surface_waves)
            },
            'quality_metrics': {
                'snr': separation_result.quality_metrics.signal_to_noise_ratio,
                'detection_confidence': separation_result.quality_metrics.detection_confidence,
                'analysis_quality_score': separation_result.quality_metrics.analysis_quality_score,
                'data_completeness': separation_result.quality_metrics.data_completeness
            },
            'processing_metadata': separation_result.processing_metadata
        }).inserted_id
        
        # Format response
        response_data = {
            'analysis_id': str(analysis_id),
            'file_id': file_id,
            'status': 'success',
            'wave_separation': {
                'p_waves': [{
                    'wave_type': wave.wave_type,
                    'start_time': wave.start_time,
                    'end_time': wave.end_time,
                    'arrival_time': wave.arrival_time,
                    'peak_amplitude': wave.peak_amplitude,
                    'dominant_frequency': wave.dominant_frequency,
                    'confidence': wave.confidence,
                    'duration': wave.duration
                } for wave in separation_result.wave_analysis_result.p_waves],
                's_waves': [{
                    'wave_type': wave.wave_type,
                    'start_time': wave.start_time,
                    'end_time': wave.end_time,
                    'arrival_time': wave.arrival_time,
                    'peak_amplitude': wave.peak_amplitude,
                    'dominant_frequency': wave.dominant_frequency,
                    'confidence': wave.confidence,
                    'duration': wave.duration
                } for wave in separation_result.wave_analysis_result.s_waves],
                'surface_waves': [{
                    'wave_type': wave.wave_type,
                    'start_time': wave.start_time,
                    'end_time': wave.end_time,
                    'arrival_time': wave.arrival_time,
                    'peak_amplitude': wave.peak_amplitude,
                    'dominant_frequency': wave.dominant_frequency,
                    'confidence': wave.confidence,
                    'duration': wave.duration
                } for wave in separation_result.wave_analysis_result.surface_waves]
            },
            'detailed_analysis': {
                'arrival_times': {
                    'p_wave_arrival': detailed_analysis.arrival_times.p_wave_arrival,
                    's_wave_arrival': detailed_analysis.arrival_times.s_wave_arrival,
                    'surface_wave_arrival': detailed_analysis.arrival_times.surface_wave_arrival,
                    'sp_time_difference': detailed_analysis.arrival_times.sp_time_difference
                },
                'magnitude_estimates': [{
                    'method': est.method,
                    'magnitude': est.magnitude,
                    'confidence': est.confidence,
                    'wave_type_used': est.wave_type_used
                } for est in detailed_analysis.magnitude_estimates],
                'epicenter_distance': detailed_analysis.epicenter_distance,
                'frequency_analysis': {
                    wave_type: {
                        'dominant_frequency': freq_data.dominant_frequency,
                        'frequency_range': freq_data.frequency_range,
                        'spectral_centroid': freq_data.spectral_centroid,
                        'bandwidth': freq_data.bandwidth
                    } for wave_type, freq_data in detailed_analysis.frequency_analysis.items()
                }
            },
            'quality_metrics': {
                'signal_to_noise_ratio': separation_result.quality_metrics.signal_to_noise_ratio,
                'detection_confidence': separation_result.quality_metrics.detection_confidence,
                'analysis_quality_score': separation_result.quality_metrics.analysis_quality_score,
                'data_completeness': separation_result.quality_metrics.data_completeness,
                'processing_warnings': separation_result.quality_metrics.processing_warnings
            },
            'processing_metadata': separation_result.processing_metadata,
            'warnings': separation_result.warnings,
            'errors': separation_result.errors
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Wave analysis error: {str(e)}")
        return jsonify({
            'error': 'Wave analysis failed',
            'message': str(e)
        }), 500

@app.route('/api/wave_results/<analysis_id>', methods=['GET'])
def get_wave_results(analysis_id):
    """
    API endpoint to retrieve wave analysis results by analysis ID.
    
    URL Parameters:
    - analysis_id: MongoDB ObjectId of the analysis result
    
    Query Parameters:
    - include_raw_data: boolean, whether to include raw wave data (default: false)
    - wave_types: comma-separated list of wave types to include (default: all)
    """
    if not WAVE_ANALYSIS_AVAILABLE:
        return jsonify({
            'error': 'Wave analysis components not available',
            'message': 'Advanced wave analysis features are not installed'
        }), 503
    
    try:
        from bson import ObjectId
        
        # Validate analysis_id format
        try:
            analysis_obj_id = ObjectId(analysis_id)
        except:
            return jsonify({'error': 'Invalid analysis_id format'}), 400
        
        # Get query parameters
        include_raw_data = request.args.get('include_raw_data', 'false').lower() == 'true'
        wave_types_param = request.args.get('wave_types', '')
        requested_wave_types = [wt.strip() for wt in wave_types_param.split(',') if wt.strip()] if wave_types_param else []
        
        # Retrieve analysis result using the repository
        detailed_analysis = wave_analysis_repo.get_wave_analysis(analysis_obj_id)
        
        if not detailed_analysis:
            return jsonify({'error': 'Analysis result not found'}), 404
        
        # If raw data is requested, we need to re-run the analysis or retrieve cached data
        if include_raw_data:
            try:
                # Retrieve original file and re-run analysis to get raw wave data
                file_id = analysis_result['file_id']
                
                with fs.get(file_id) as f:
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_file.write(f.read())
                        temp_file_path = temp_file.name
                
                # Load audio data
                try:
                    sr, seismic_data = wav.read(temp_file_path)
                    seismic_data = seismic_data.astype(np.float32)
                    if len(seismic_data.shape) > 1:
                        seismic_data = seismic_data.mean(axis=1)
                except:
                    try:
                        seismic_data, sr = sf.read(temp_file_path)
                        if len(seismic_data.shape) > 1:
                            seismic_data = seismic_data.mean(axis=1)
                    except:
                        seismic_data, sr = librosa.load(temp_file_path, sr=SAMPLE_RATE, mono=True)
                
                os.remove(temp_file_path)
                
                # Re-run wave separation to get raw data
                params = analysis_result.get('parameters', {})
                separation_params = WaveSeparationParameters(
                    sampling_rate=params.get('sampling_rate', SAMPLE_RATE),
                    min_snr=params.get('min_snr', 2.0),
                    min_detection_confidence=params.get('min_detection_confidence', 0.3)
                )
                
                wave_engine = WaveSeparationEngine(separation_params)
                separation_result = wave_engine.separate_waves(seismic_data)
                
                # Add raw wave data to response
                raw_wave_data = {}
                
                if not requested_wave_types or 'P' in requested_wave_types:
                    raw_wave_data['p_waves'] = [{
                        'wave_type': wave.wave_type,
                        'start_time': wave.start_time,
                        'end_time': wave.end_time,
                        'arrival_time': wave.arrival_time,
                        'peak_amplitude': wave.peak_amplitude,
                        'dominant_frequency': wave.dominant_frequency,
                        'confidence': wave.confidence,
                        'duration': wave.duration,
                        'raw_data': wave.data.tolist(),
                        'sampling_rate': wave.sampling_rate
                    } for wave in separation_result.wave_analysis_result.p_waves]
                
                if not requested_wave_types or 'S' in requested_wave_types:
                    raw_wave_data['s_waves'] = [{
                        'wave_type': wave.wave_type,
                        'start_time': wave.start_time,
                        'end_time': wave.end_time,
                        'arrival_time': wave.arrival_time,
                        'peak_amplitude': wave.peak_amplitude,
                        'dominant_frequency': wave.dominant_frequency,
                        'confidence': wave.confidence,
                        'duration': wave.duration,
                        'raw_data': wave.data.tolist(),
                        'sampling_rate': wave.sampling_rate
                    } for wave in separation_result.wave_analysis_result.s_waves]
                
                if not requested_wave_types or any(wt in requested_wave_types for wt in ['Love', 'Rayleigh']):
                    filtered_surface_waves = separation_result.wave_analysis_result.surface_waves
                    if requested_wave_types:
                        filtered_surface_waves = [w for w in filtered_surface_waves if w.wave_type in requested_wave_types]
                    
                    raw_wave_data['surface_waves'] = [{
                        'wave_type': wave.wave_type,
                        'start_time': wave.start_time,
                        'end_time': wave.end_time,
                        'arrival_time': wave.arrival_time,
                        'peak_amplitude': wave.peak_amplitude,
                        'dominant_frequency': wave.dominant_frequency,
                        'confidence': wave.confidence,
                        'duration': wave.duration,
                        'raw_data': wave.data.tolist(),
                        'sampling_rate': wave.sampling_rate
                    } for wave in filtered_surface_waves]
                
                analysis_result['raw_wave_data'] = raw_wave_data
                
            except Exception as e:
                app.logger.warning(f"Failed to retrieve raw wave data: {str(e)}")
                analysis_result['raw_data_error'] = str(e)
        
        # Format response
        response_data = {
            'analysis_id': str(analysis_result['_id']),
            'file_id': analysis_result['file_id'],
            'analysis_timestamp': analysis_result['analysis_timestamp'].isoformat(),
            'parameters': analysis_result.get('parameters', {}),
            'wave_separation': analysis_result.get('wave_separation', {}),
            'quality_metrics': analysis_result.get('quality_metrics', {}),
            'processing_metadata': analysis_result.get('processing_metadata', {})
        }
        
        # Add raw data if requested and available
        if include_raw_data and 'raw_wave_data' in analysis_result:
            response_data['raw_wave_data'] = analysis_result['raw_wave_data']
        elif include_raw_data and 'raw_data_error' in analysis_result:
            response_data['raw_data_error'] = analysis_result['raw_data_error']
        
        return jsonify(response_data)
        
    except Exception as e:
        app.logger.error(f"Error retrieving wave results: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve wave results',
            'message': str(e)
        }), 500


@app.route('/api/wave_analysis_stats', methods=['GET'])
def get_wave_analysis_stats():
    """
    API endpoint to retrieve wave analysis statistics.
    
    Returns statistics about stored wave analyses including:
    - Total number of analyses
    - Average quality score
    - Wave type distribution
    - Latest and oldest analysis timestamps
    """
    if not WAVE_ANALYSIS_AVAILABLE:
        return jsonify({
            'error': 'Wave analysis components not available',
            'message': 'Advanced wave analysis features are not installed'
        }), 503
    
    try:
        stats = wave_analysis_repo.get_analysis_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        app.logger.error(f"Error retrieving wave analysis statistics: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'message': str(e)
        }), 500


@app.route('/api/recent_wave_analyses', methods=['GET'])
def get_recent_wave_analyses():
    """
    API endpoint to retrieve recent wave analyses.
    
    Query Parameters:
    - limit: Maximum number of results (default: 20, max: 100)
    - min_quality_score: Minimum quality score filter (default: 0.0)
    """
    if not WAVE_ANALYSIS_AVAILABLE:
        return jsonify({
            'error': 'Wave analysis components not available',
            'message': 'Advanced wave analysis features are not installed'
        }), 503
    
    try:
        # Get query parameters
        limit = min(int(request.args.get('limit', 20)), 100)
        min_quality_score = float(request.args.get('min_quality_score', 0.0))
        
        # Get recent analyses
        recent_analyses = wave_analysis_repo.get_recent_analyses(
            limit=limit,
            min_quality_score=min_quality_score
        )
        
        return jsonify({
            'success': True,
            'analyses': recent_analyses,
            'count': len(recent_analyses)
        })
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid parameter value',
            'message': str(e)
        }), 400
    except Exception as e:
        app.logger.error(f"Error retrieving recent wave analyses: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve recent analyses',
            'message': str(e)
        }), 500


@app.route('/api/search_wave_analyses', methods=['GET'])
def search_wave_analyses():
    """
    API endpoint to search wave analyses by magnitude.
    
    Query Parameters:
    - min_magnitude: Minimum magnitude (required)
    - max_magnitude: Maximum magnitude (optional)
    """
    if not WAVE_ANALYSIS_AVAILABLE:
        return jsonify({
            'error': 'Wave analysis components not available',
            'message': 'Advanced wave analysis features are not installed'
        }), 503
    
    try:
        # Get query parameters
        min_magnitude = request.args.get('min_magnitude')
        max_magnitude = request.args.get('max_magnitude')
        
        if min_magnitude is None:
            return jsonify({
                'error': 'Missing required parameter',
                'message': 'min_magnitude parameter is required'
            }), 400
        
        min_magnitude = float(min_magnitude)
        max_magnitude = float(max_magnitude) if max_magnitude else None
        
        # Search analyses
        search_results = wave_analysis_repo.search_analyses_by_magnitude(
            min_magnitude=min_magnitude,
            max_magnitude=max_magnitude
        )
        
        return jsonify({
            'success': True,
            'analyses': search_results,
            'count': len(search_results),
            'search_criteria': {
                'min_magnitude': min_magnitude,
                'max_magnitude': max_magnitude
            }
        })
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid parameter value',
            'message': str(e)
        }), 400
    except Exception as e:
        app.logger.error(f"Error searching wave analyses: {str(e)}")
        return jsonify({
            'error': 'Failed to search analyses',
            'message': str(e)
        }), 500


@app.route('/api/wave_analyses_by_file/<file_id>', methods=['GET'])
def get_wave_analyses_by_file(file_id):
    """
    API endpoint to retrieve all wave analyses for a specific file.
    
    URL Parameters:
    - file_id: GridFS file ID
    """
    if not WAVE_ANALYSIS_AVAILABLE:
        return jsonify({
            'error': 'Wave analysis components not available',
            'message': 'Advanced wave analysis features are not installed'
        }), 503
    
    try:
        from bson import ObjectId
        
        # Validate file_id format
        try:
            file_obj_id = ObjectId(file_id)
        except:
            return jsonify({'error': 'Invalid file_id format'}), 400
        
        # Get analyses for the file
        analyses = wave_analysis_repo.get_analyses_by_file(file_obj_id)
        
        # Convert to serializable format
        analyses_data = []
        for analysis in analyses:
            analysis_summary = {
                'analysis_timestamp': analysis.analysis_timestamp.isoformat(),
                'total_waves_detected': analysis.wave_result.total_waves_detected,
                'wave_types_detected': analysis.wave_result.wave_types_detected,
                'quality_score': analysis.quality_metrics.analysis_quality_score if analysis.quality_metrics else None,
                'magnitude_estimates': [
                    {
                        'method': est.method,
                        'magnitude': est.magnitude,
                        'confidence': est.confidence
                    } for est in analysis.magnitude_estimates
                ],
                'arrival_times': {
                    'p_wave_arrival': analysis.arrival_times.p_wave_arrival,
                    's_wave_arrival': analysis.arrival_times.s_wave_arrival,
                    'sp_time_difference': analysis.arrival_times.sp_time_difference
                },
                'epicenter_distance': analysis.epicenter_distance
            }
            analyses_data.append(analysis_summary)
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'analyses': analyses_data,
            'count': len(analyses_data)
        })
        
    except Exception as e:
        app.logger.error(f"Error retrieving analyses for file {file_id}: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve file analyses',
            'message': str(e)
        }), 500


# Health check and monitoring endpoints
@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint."""
    try:
        health_status = health_monitor.check_system_health()
        
        # Return appropriate HTTP status based on health
        status_code = 200
        if health_status['overall_status'] == 'unhealthy':
            status_code = 503
        elif health_status['overall_status'] == 'warning':
            status_code = 200  # Still operational but with warnings
        
        return jsonify(health_status), status_code
        
    except Exception as e:
        app_logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'overall_status': 'unhealthy',
            'error': 'Health check system failure',
            'message': str(e)
        }), 503

@app.route('/metrics', methods=['GET'])
def performance_metrics():
    """Performance metrics endpoint."""
    try:
        metrics_summary = wave_logger.get_performance_summary()
        
        # Add system configuration info
        system_info = {
            'environment': config_manager.deployment.environment,
            'wave_analysis_config': {
                'sampling_rate': config_manager.wave_analysis.sampling_rate,
                'min_snr': config_manager.wave_analysis.min_snr,
                'max_concurrent_analyses': config_manager.wave_analysis.max_concurrent_analyses
            },
            'database_config': {
                'max_pool_size': config_manager.database.max_pool_size,
                'connection_timeout_ms': config_manager.database.connection_timeout_ms
            }
        }
        
        return jsonify({
            'system_info': system_info,
            'performance_metrics': metrics_summary,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        app_logger.error(f"Metrics endpoint failed: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve metrics',
            'message': str(e)
        }), 500

@app.route('/config', methods=['GET'])
def get_configuration():
    """Get current system configuration (non-sensitive parts)."""
    try:
        # Only return non-sensitive configuration information
        config_info = {
            'environment': config_manager.deployment.environment,
            'wave_analysis': {
                'sampling_rate': config_manager.wave_analysis.sampling_rate,
                'min_snr': config_manager.wave_analysis.min_snr,
                'min_detection_confidence': config_manager.wave_analysis.min_detection_confidence,
                'max_file_size_mb': config_manager.wave_analysis.max_file_size_mb,
                'processing_timeout_seconds': config_manager.wave_analysis.processing_timeout_seconds,
                'magnitude_estimation_methods': config_manager.wave_analysis.magnitude_estimation_methods
            },
            'logging': {
                'level': config_manager.logging.level,
                'enable_performance_logging': config_manager.logging.enable_performance_logging,
                'enable_alert_logging': config_manager.logging.enable_alert_logging
            },
            'features': {
                'wave_analysis_available': WAVE_ANALYSIS_AVAILABLE,
                'caching_enabled': config_manager.deployment.enable_caching,
                'metrics_enabled': config_manager.deployment.enable_metrics
            }
        }
        
        return jsonify(config_info)
        
    except Exception as e:
        app_logger.error(f"Configuration endpoint failed: {str(e)}")
        return jsonify({
            'error': 'Failed to retrieve configuration',
            'message': str(e)
        }), 500

@app.route('/config/validate', methods=['POST'])
def validate_configuration():
    """Validate current configuration."""
    try:
        validation_results = config_manager.validate_configuration()
        
        return jsonify({
            'valid': len(validation_results['errors']) == 0,
            'errors': validation_results['errors'],
            'warnings': validation_results['warnings'],
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        app_logger.error(f"Configuration validation failed: {str(e)}")
        return jsonify({
            'error': 'Configuration validation failed',
            'message': str(e)
        }), 500

# Add performance monitoring to key functions
predict = performance_monitor('predict')(predict)
perform_wave_analysis = performance_monitor('perform_wave_analysis')(perform_wave_analysis)

# Log application startup
app_logger.info("Flask application initialized successfully")
app_logger.info(f"Configuration: {config_manager.deployment.environment} environment")
app_logger.info(f"Wave analysis available: {WAVE_ANALYSIS_AVAILABLE}")

# Run the application
if __name__ == '__main__':
    # Log startup configuration
    app_logger.info("Starting Flask application")
    app_logger.info(f"Debug mode: {config_manager.deployment.debug}")
    app_logger.info(f"CORS enabled: {config_manager.deployment.enable_cors}")
    
    app.run(
        debug=config_manager.deployment.debug, 
        use_reloader=config_manager.deployment.debug
    )
