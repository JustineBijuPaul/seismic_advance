# Import required libraries
import os                          # For operating system operations
from flask import Flask, request, render_template, jsonify  # Core Flask functionality
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

# Initialize Flask application
app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')
application = app  # For WSGI compatibility

# Add this near the top of the file, after app initialization
app.url_map.strict_slashes = False

# Configure WhiteNoise for serving static files
app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

# Define constants for signal processing
SAMPLE_RATE = 100     # Sampling rate in Hz
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
print(f"MONGO_URI: {MONGO_URI}")

DB_NAME = 'seismic_quake'

# Initialize MongoDB client and GridFS
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
fs = gridfs.GridFS(db)

# Add global variable for continuous monitoring
monitoring_active = False

# Import wave analysis components
try:
    from wave_analysis import WaveAnalysisResult, DetailedAnalysis
    from wave_analysis.services import WaveAnalyzer
    from wave_analysis.services.wave_separation_engine import WaveSeparationEngine, WaveSeparationParameters
    WAVE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Wave analysis components not available: {e}")
    WAVE_ANALYSIS_AVAILABLE = False

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

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/documentation')
def documentation():
    """Render the documentation page"""
    return render_template('documentation.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload and analysis"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            # Store file and process it
            file_id = fs.put(file, filename=file.filename)
            prediction, y, sr, earthquake_indices = predict(file_id)
            time_labels = [str(timedelta(seconds=i / sr)) for i in range(len(y))]
            
            response_data = {
                'file_id': str(file_id),  # Include file_id in response
                'prediction': 'Seismic Activity Detected' if prediction == 1 else 'No Seismic Activity Detected',
                'time_labels': time_labels,
                'amplitude_data': y.tolist(),
                'sampling_rate': sr
            }
            
            if prediction == 1:
                response_data.update({
                    'time_indices': earthquake_indices.tolist(),
                    'amplitudes': [float(y[idx]) for idx in earthquake_indices]
                })
                
            return jsonify(response_data)
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
        app.logger.error(f"USGS API Error: {str(e)}")
        return render_template('error.html', 
                             error="Failed to fetch earthquake data. Please try with a shorter time range or higher magnitude threshold.",
                             details=str(e))
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return render_template('error.html',
                             error="An unexpected error occurred",
                             details=str(e))

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
        
        # Initialize wave analyzer for detailed analysis
        wave_analyzer = WaveAnalyzer(sampling_rate)
        
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
        
        # Retrieve analysis result from database
        analysis_result = db.wave_analyses.find_one({'_id': analysis_obj_id})
        
        if not analysis_result:
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

# Run the application
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
