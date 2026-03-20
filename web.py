"""web.py — Flask API for the AIC signal monitor.

Fixes vs original:
- fft_history now persists correctly (was mutating a shallow copy)
- /api/signal always returns valid JSON even before first SDR chunk
- SSE log stream uses a daemon-safe generator
- Added /api/status endpoint for health checks
- latest_data and data_lock are importable and rebindable from aic2
"""
from flask import Flask, render_template, jsonify, Response
import os
import numpy as np
from datetime import datetime
import logging
import queue
from threading import Lock, Thread

from logging_setup import configure_logging

logger = configure_logging()

# Suppress Werkzeug access log — hides IP addresses from the SSE log stream
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Queue for SSE log streaming
log_queue: queue.Queue = queue.Queue(maxsize=500)


class _QueueHandler(logging.Handler):
    def emit(self, record):
        try:
            log_queue.put_nowait(self.format(record))
        except queue.Full:
            pass  # drop old logs rather than blocking


_qh = _QueueHandler()
_qh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
logger.addHandler(_qh)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared state — aic2.__main__ rebinds these references after import
# ---------------------------------------------------------------------------
latest_data: dict = {
    'fft_data': {
        'frequency': list(np.linspace(1420.0, 1420.4, num=256)),
        'magnitude': [0.0] * 256,
        'power': [0.0] * 256,
        'phase': [0.0] * 256,
    },
    'fft_history': [],
    'signal_strength': 'N/A',
    'signal_strength_db': None,
    'confidence': 0.0,
    'detection': False,
    'stable_detection': False,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}
data_lock: Lock = Lock()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """Simple health-check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/plots')
def get_plot_data():
    output_dir = 'output'
    data = {'plots': [], 'spectrograms': [], 'timestamp': datetime.now().isoformat()}
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.startswith('signal_strength_plot'):
                data['plots'].append(f)
            elif f.startswith('spectrogram'):
                data['spectrograms'].append(f)
    return jsonify(data)


@app.route('/api/signal')
def get_signal():
    with data_lock:
        snapshot = {
            'timestamp':        latest_data.get('timestamp', ''),
            'frequency':        latest_data.get('frequency', ''),
            'signal_strength':  latest_data.get('signal_strength', 'N/A'),
            'signal_strength_db': latest_data.get('signal_strength_db'),
            'confidence':       float(latest_data.get('confidence', 0.0)),
            'detection':        bool(latest_data.get('detection', False)),
            'stable_detection': bool(latest_data.get('stable_detection', False)),
            'raw_model_output': float(latest_data.get('raw_model_output', 0.0)),
            'buffer_stats':     latest_data.get('buffer_stats', {}),
            'system_status':    latest_data.get('system_status', {}),
        }

        # --- FFT data ---
        # aic2.py stores it as latest_data['fft_data'] = {magnitude, frequency, power, phase}
        # Frontend reads d.fft_data.frequency and d.fft_data.magnitude — pass it through as-is
        # but ensure all values are plain Python lists (not numpy arrays).
        raw_fft = latest_data.get('fft_data', {})
        fft_freq      = _to_list(raw_fft.get('frequency', []))
        fft_magnitude = [abs(float(v)) for v in raw_fft.get('magnitude', [])]
        fft_power_raw = _to_list(raw_fft.get('power', []))
        fft_phase     = _to_list(raw_fft.get('phase', []))

        # Normalise power 0-1 for waterfall
        if fft_power_raw:
            p     = np.array(fft_power_raw, dtype=np.float64)
            p_min = float(p.min())
            p_rng = max(float(p.ptp()), 1e-12)
            p_norm = ((p - p_min) / p_rng).tolist()
        else:
            p_norm = []

        # Return fft_data as a nested dict so frontend d.fft_data.frequency works
        snapshot['fft_data'] = {
            'frequency': fft_freq,
            'magnitude': fft_magnitude,
            'power':     fft_power_raw,
            'phase':     fft_phase,
        }

        # Also keep flat aliases for backwards compat
        snapshot['fft_freq']      = fft_freq
        snapshot['fft_magnitude'] = fft_magnitude

        # Persist waterfall history inside the lock
        if 'fft_history' not in latest_data:
            latest_data['fft_history'] = []
        if p_norm:
            latest_data['fft_history'].append(p_norm)
            if len(latest_data['fft_history']) > 100:
                latest_data['fft_history'] = latest_data['fft_history'][-100:]

        snapshot['fft_power_db_normalized'] = list(latest_data['fft_history'])

    logger.debug(
        f"/api/signal: det={snapshot['detection']}, conf={snapshot['confidence']:.4f}, "
        f"fft_bins={len(fft_freq)}, history={len(snapshot['fft_power_db_normalized'])}"
    )
    return jsonify(snapshot)


def _to_list(val):
    """Convert numpy array or list to plain Python list of floats."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    try:
        return [float(v) for v in val]
    except Exception:
        return []


@app.route('/logs')
def stream_logs():
    """Server-sent events stream for live log output."""
    def generate():
        while True:
            try:
                msg = log_queue.get(timeout=1.0)
                # Escape newlines so the SSE frame stays valid
                safe = msg.replace('\n', ' ').replace('\r', '')
                yield f'data: {safe}\n\n'
            except queue.Empty:
                yield 'data: \n\n'  # keepalive

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ---------------------------------------------------------------------------
# Entry point (standalone — normally started from aic2.__main__)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import aic2 as aic_module  # avoid circular import at module level

    os.makedirs('output', exist_ok=True)

    # Bind shared state
    aic_module.latest_data = latest_data
    aic_module.data_lock = data_lock

    aic_thread = Thread(target=aic_module.main, daemon=True)
    aic_thread.start()

    logger.info('Starting web server on http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)