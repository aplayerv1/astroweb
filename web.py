from flask import Flask, render_template, jsonify, Response
import os
from datetime import datetime
import logging
from threading import Thread
import queue
import aic
import numpy as np
from aic import process_continuous_stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a queue to store log messages
log_queue = queue.Queue()

# Custom handler to capture logs
class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

# Configure Flask
app = Flask(__name__)
logger.addHandler(QueueHandler())

@app.route('/')
def index():
    logger.debug("Index page accessed")
    return render_template('index.html')

@app.route('/api/plots')
def get_plot_data():
    output_dir = 'output'
    # logger.debug(f"Checking output directory: {output_dir}")
    data = {
        'plots': [],
        'spectrograms': [],
        'timestamp': datetime.now().isoformat()
    }
    
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for file in files:
            if file.startswith('signal_strength_plot'):
                data['plots'].append(file)
            elif file.startswith('spectrogram'):
                data['spectrograms'].append(file)
    
    # logger.debug(f"Returning data: {data}")
    return jsonify(data)



@app.route('/api/signal')
def get_signal_data():
    latest_data = app.config.get('LATEST_DATA', {})
    
    # Convert complex samples to magnitude for JSON serialization
    if 'processed_samples' in latest_data:
        latest_data['processed_samples'] = np.abs(latest_data['processed_samples']).tolist()
    
    if 'fft_data' in latest_data:
        latest_data['fft_data'] = np.abs(latest_data['fft_data']).tolist()
    
    return jsonify(latest_data)

@app.route('/logs')
def stream_logs():
    def generate():
        while True:
            try:
                log_message = log_queue.get_nowait()
                yield f"data: {log_message}\n\n"
            except queue.Empty:
                yield "data: \n\n"
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Start AIC processing in a separate thread
    aic_thread = Thread(target=aic.main)
    aic_thread.daemon = True
    aic_thread.start()
    
    # Start web server
    logger.info("Starting web server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
