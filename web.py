from flask import Flask, render_template, jsonify, Response
import os
from datetime import datetime
import logging
from threading import Thread
import queue
import aic
import numpy as np

# Configure logging
from logging_setup import configure_logging
logger = configure_logging()

# Queue for logs
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

logger.addHandler(QueueHandler())

app = Flask(__name__)

# Store latest data safely
from threading import Lock
# Provide a small placeholder so the UI does not repeatedly log warnings
# before the processing loop produces the first FFT.
latest_data = {
    'fft_data': {
        'frequency': list(np.linspace(0.0, 1.0, num=256)),
        'magnitude': [0.0] * 256,
        'power': [0.0] * 256,
        'phase': []
    },
    'fft_history': []
}
data_lock = Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/plots')
def get_plot_data():
    output_dir = 'output'
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
    return jsonify(data)

@app.route('/api/signal')
def get_signal():
    with data_lock:
        latest = latest_data.copy()

    logger.debug(f"API /signal: latest_data_keys={list(latest.keys())}")

    # Convert processed_samples to magnitude
    if 'processed_samples' in latest:
        arr = latest['processed_samples']
        if isinstance(arr, (list, np.ndarray)):
            latest['processed_samples'] = np.abs(np.array(arr)).tolist()
        else:
            logger.warning("processed_samples is not a list or ndarray")
            latest['processed_samples'] = []

    # FFT data processing
    fft_freq = []
    fft_magnitude = []
    fft_power_db_normalized = []

    if 'fft_data' in latest and isinstance(latest['fft_data'], dict):
        fft_entry = latest['fft_data']
        fft_freq = fft_entry.get('frequency', [])
        fft_magnitude = np.abs(np.array(fft_entry.get('magnitude', []))).tolist()
        power = np.array(fft_entry.get('power', []))
        # Normalize power
        if power.size > 0:
            power_normalized = (power - power.min()) / max(power.ptp(), 1e-12)
            fft_power_db_normalized = power_normalized.tolist()
        else:
            logger.warning("FFT power array is empty")

        # Keep FFT history
        if 'fft_history' not in latest:
            latest['fft_history'] = []
        latest['fft_history'].append(fft_power_db_normalized)
        latest['fft_history'] = latest['fft_history'][-100:]  # last 100
    else:
        logger.warning("fft_data missing or not a dict in latest_data — inserting placeholder data for UI")
        # Provide small placeholder FFT data so the frontend can render instead of empty plots
        try:
            freqs = list(np.linspace(0.0, 1.0, num=256))
            mags = [0.0] * 256
            power = np.array(mags)
            latest['fft_data'] = {'frequency': freqs, 'magnitude': mags, 'power': mags, 'phase': []}
            # Also set local variables so the rest of this function produces consistent output
            fft_freq = freqs
            fft_magnitude = mags
            latest['fft_history'] = [mags]
            fft_power_db_normalized = [mags]
        except Exception:
            # Fallback to empty lists if something goes wrong
            fft_freq = []
            fft_magnitude = []
            fft_power_db_normalized = []

    latest['fft_freq'] = fft_freq
    latest['fft_magnitude'] = fft_magnitude
    latest['fft_power_db_normalized'] = latest.get('fft_history', [])

    # Ensure detection/confidence keys exist for UI and are JSON-serializable
    try:
        latest['detection'] = bool(latest.get('detection', False))
    except Exception:
        latest['detection'] = False
    try:
        latest['confidence'] = float(latest.get('confidence', 0.0))
    except Exception:
        latest['confidence'] = 0.0

    logger.debug(f"API /signal - detection={latest['detection']}, confidence={latest['confidence']}")

    logger.debug(f"API /signal response: fft_freq_len={len(fft_freq)}, fft_magnitude_len={len(fft_magnitude)}, fft_history_len={len(latest['fft_power_db_normalized'])}")

    return jsonify(latest)
@app.route('/logs')
def stream_logs():
    def generate():
        while True:
            try:
                msg = log_queue.get(timeout=1)
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield "data: \n\n"
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    os.makedirs('output', exist_ok=True)

    # Start AIC thread
    aic_thread = Thread(target=aic.main)
    aic_thread.daemon = True
    aic_thread.start()

    # Run Flask app
    logger.info("Starting web server...")
    app.run(host='0.0.0.0', port=5000, debug=True)