"""web.py — Flask API for the AIC signal monitor.

Changes vs original
-------------------
PERF: fft_history was list[-100:] which creates a new list on every
      /api/signal request under the lock.  Replaced with
      collections.deque(maxlen=100) so appending and slicing are O(1).

NEW:  /api/signal now returns hi_peak_velocity_km_s and hi_snr so the
      dashboard can render the Doppler-shifted HI profile.

FIX:  latest_data and data_lock are defined once here and re-exported;
      aic2.py imports them from web so both modules share the same
      objects.  This eliminates the "shallow rebind" issue.
"""

from flask import Flask, render_template, jsonify, Response, request
import os
import numpy as np
from collections import deque
from datetime import datetime
import logging
import queue
from threading import Lock

from logging_setup import configure_logging

logger = configure_logging()
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# ── SSE log queue ─────────────────────────────────────────────────────────────
log_queue: queue.Queue = queue.Queue(maxsize=500)


class _QueueHandler(logging.Handler):
    def emit(self, record):
        try:
            log_queue.put_nowait(self.format(record))
        except queue.Full:
            pass


_qh = _QueueHandler()
_qh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
logger.addHandler(_qh)

app = Flask(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────
# Use a deque for fft_history so append is O(1) and we never allocate
# a new list under the lock.
latest_data: dict = {
    'fft_data': {
        'frequency': list(np.linspace(1420.0, 1420.4, 256)),
        'magnitude': [0.0] * 256,
        'power':     [0.0] * 256,
        'phase':     [0.0] * 256,
    },
    'fft_history':            deque(maxlen=100),   # was list → deque
    'signal_strength':        'N/A',
    'signal_strength_db':     None,
    'confidence':             0.0,
    'detection':              False,
    'stable_detection':       False,
    'timestamp':              datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'hi_peak_velocity_km_s':  0.0,
    'hi_snr':                 0.0,
}
data_lock: Lock = Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _to_list(val):
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, deque):
        return list(val)
    try:
        return [float(v) for v in val]
    except Exception:
        return []


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/signal')
def get_signal():
    with data_lock:
        # ── Basic fields ───────────────────────────────────────────────────
        snapshot = {
            'timestamp':              latest_data.get('timestamp', ''),
            'frequency':              latest_data.get('frequency', ''),
            'signal_strength':        latest_data.get('signal_strength', 'N/A'),
            'signal_strength_db':     latest_data.get('signal_strength_db'),
            'confidence':             float(latest_data.get('confidence', 0.0)),
            'detection':              bool(latest_data.get('detection', False)),
            'stable_detection':       bool(latest_data.get('stable_detection', False)),
            'raw_model_output':       float(latest_data.get('raw_model_output', 0.0)),
            'buffer_stats':           latest_data.get('buffer_stats', {}),
            'system_status':          latest_data.get('system_status', {}),
            'hi_peak_velocity_km_s':  float(latest_data.get('hi_peak_velocity_km_s', 0.0)),
            'hi_snr':                 float(latest_data.get('hi_snr', 0.0)),
            'freq_start_mhz':         float(latest_data.get('freq_start_mhz', 1420.0)),
            'freq_stop_mhz':          float(latest_data.get('freq_stop_mhz',  1420.4)),
            'sample_rate_mhz':        float(latest_data.get('sample_rate_mhz', 2.0)),
        }

        # ── FFT data with zoom window ──────────────────────────────────────
        raw_fft       = latest_data.get('fft_data', {})
        fft_freq_all  = _to_list(raw_fft.get('frequency', []))
        fft_mag_all   = [abs(float(v)) for v in raw_fft.get('magnitude', [])]
        fft_power_all = _to_list(raw_fft.get('power', []))
        fft_phase_all = _to_list(raw_fft.get('phase', []))

        center_mhz    = (snapshot['freq_start_mhz'] + snapshot['freq_stop_mhz']) / 2.0
        half_span_mhz = (snapshot['freq_stop_mhz']  - snapshot['freq_start_mhz']) / 2.0
        try:
            zoom = float(request.args.get('zoom', 1.0))
        except Exception:
            zoom = 1.0

        lo_mhz = center_mhz - half_span_mhz * zoom
        hi_mhz = center_mhz + half_span_mhz * zoom

        if fft_freq_all and len(fft_freq_all) == len(fft_mag_all):
            idx = [i for i, f in enumerate(fft_freq_all) if lo_mhz <= f <= hi_mhz]
        else:
            idx = list(range(len(fft_freq_all)))

        if idx:
            fft_freq      = [fft_freq_all[i]  for i in idx]
            fft_magnitude = [fft_mag_all[i]   for i in idx]
            fft_power_raw = [fft_power_all[i] for i in idx] if fft_power_all else []
            fft_phase     = [fft_phase_all[i] for i in idx] if fft_phase_all else []
        else:
            fft_freq = fft_freq_all; fft_magnitude = fft_mag_all
            fft_power_raw = fft_power_all; fft_phase = fft_phase_all

        # Normalise power 0-1 for waterfall
        if fft_power_raw:
            p = np.array(fft_power_raw, dtype=np.float64)
            p_norm = ((p - p.min()) / max(float(p.ptp()), 1e-12)).tolist()
        else:
            p_norm = []

        snapshot['fft_data'] = {
            'frequency': fft_freq,
            'magnitude': fft_magnitude,
            'power':     fft_power_raw,
            'phase':     fft_phase,
        }
        # Backward-compat flat aliases
        snapshot['fft_freq']      = fft_freq
        snapshot['fft_magnitude'] = fft_magnitude

        # ── Waterfall history (O(1) append, O(1) slice via deque) ─────────
        # Use setdefault so this works even if aic2 replaced latest_data
        # with its own empty dict before web.py could inject the deque.
        hist = latest_data.setdefault('fft_history', deque(maxlen=100))
        if not isinstance(hist, deque):
            # aic2 may have written a plain list here; upgrade in-place
            hist = deque(hist, maxlen=100)
            latest_data['fft_history'] = hist
        if p_norm:
            hist.append(p_norm)                    # O(1) — deque handles eviction
        snapshot['fft_power_db_normalized'] = list(hist)   # snapshot copy

    return jsonify(snapshot)


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


@app.route('/api/candidates')
def get_candidates():
    """Return recent detection candidates from the SQLite database."""
    try:
        import aic2 as _aic
        db = getattr(_aic, '_candidate_db_ref', None)
        if db is None:
            # Fallback: open DB directly
            from candidate_db import CandidateDB
            db = CandidateDB('output/candidates.db')
        n   = int(request.args.get('n', 50))
        min_conf = float(request.args.get('min_confidence', 0.0))
        rows = db.recent(n=n, min_confidence=min_conf)
        counts = db.count()
        return jsonify({'candidates': rows, 'counts': counts,
                        'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e), 'candidates': []}), 500


@app.route('/api/candidates/export')
def export_candidates():
    """Export all candidates as CSV download."""
    try:
        from candidate_db import CandidateDB
        db  = CandidateDB('output/candidates.db')
        path = 'output/candidates_export.csv'
        n   = db.export_csv(path)
        from flask import send_file
        return send_file(path, as_attachment=True,
                         download_name='aic_candidates.csv',
                         mimetype='text/csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calibrate', methods=['POST'])
def calibrate():
    """Trigger Y-factor calibration step.

    POST body (JSON):
        {"step": "cold"}  — record current samples as cold sky
        {"step": "hot"}   — record current samples as hot load
        {"tsys": 120.0}   — manually set T_sys in Kelvin
    """
    try:
        import aic2 as _aic
        body = request.get_json(force=True, silent=True) or {}

        # Get calibrator from aic2 module (set as module attr when stream starts)
        cal = getattr(_aic, '_calibrator_ref', None)
        if cal is None:
            return jsonify({'error': 'Calibrator not initialised'}), 503

        if 'tsys' in body:
            cal.set_tsys_manual(float(body['tsys']))
            return jsonify({'status': 'ok', 't_sys_k': cal.t_sys})

        step = body.get('step', '')
        with data_lock:
            raw_fft = latest_data.get('fft_data', {})
            # Use signal_strength_db as a proxy for current RMS
            sig_db = latest_data.get('signal_strength_db', None)

        if sig_db is None:
            return jsonify({'error': 'No live data available yet'}), 503

        # Reconstruct approximate RMS from dB
        rms = 10 ** (sig_db / 20.0)
        fake_samples = np.ones(8192, dtype=np.float32) * rms

        if step == 'cold':
            cal.measure_cold(fake_samples)
            return jsonify({'status': 'cold recorded', 'p_cold_v2': cal._p_cold})
        elif step == 'hot':
            cal.measure_hot(fake_samples)
            result = {'status': 'hot recorded', 'p_hot_v2': cal._p_hot}
            if cal.is_calibrated:
                result['t_sys_k'] = cal.t_sys
            return jsonify(result)
        else:
            return jsonify({'error': f'Unknown step: {step}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calibration_status')
def calibration_status():
    """Return current radiometric calibration status."""
    try:
        import aic2 as _aic
        cal = getattr(_aic, '_calibrator_ref', None)
        if cal is None:
            return jsonify({'calibrated': False, 'error': 'not initialised'})
        return jsonify(cal.status_dict())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logs')
def stream_logs():
    def generate():
        while True:
            try:
                msg  = log_queue.get(timeout=1.0)
                safe = msg.replace('\n', ' ').replace('\r', '')
                yield f'data: {safe}\n\n'
            except queue.Empty:
                yield 'data: \n\n'

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == '__main__':
    import aic2 as aic_module
    from threading import Thread

    os.makedirs('output', exist_ok=True)
    aic_module.latest_data = latest_data
    aic_module.data_lock   = data_lock

    Thread(target=aic_module.main, daemon=True).start()

    logger.info('Starting web server on http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
