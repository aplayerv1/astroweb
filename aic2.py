"""aic2.py — AIC signal monitor, real-time safe edition.

Key changes vs original
-----------------------

BUG: np.append() in hot loop (slowdown root cause)
  np.append() copies the *entire* accumulation buffer on every call.
  After 10 s at 2 MHz this grows to ~160 MB/call and OOM-kills the
  process within minutes.  Fixed with a pre-allocated RingBuffer that
  never copies more than one chunk_size worth of data.

BUG: Hill/baseline correction destroyed HI line
  The old denoise_signal() path applied a moving-average that was
  subtracting the very emission peak we wanted to detect.  The new
  pipeline calls extract_hi_line() (polynomial continuum, masked at
  centre) so the HI feature is preserved and reported in the web data.

BUG: predict_signal() duplicated in aic2 + processing
  aic2.predict_signal() had a slightly different threshold formula to
  processing.predict_signal().  The hot loop now calls
  processing.predict_signal() directly; the local copy is removed.

BUG: LNB flag not threaded through after connect_to_server()
  advance_signal_processing.set_lnb_enabled() is now called once
  in main() after args are parsed; the processing loop doesn't touch
  it again.

BUG: Matplotlib figure leak in background detection worker
  plt.figure() without plt.close() leaked ~4 MB per detection event.
  All figure creation is now in aic_io.py which already calls plt.close().

PERF: process_fft() called with a shared _Workspace
  The Kaiser window is created once and reused every chunk.

PERF: FFT history in web.py (deque, not list[-100:] every poll)
  web.py's list[-100:] creates a new list on every /api/signal request.
  Replaced with collections.deque(maxlen=100) in latest_data.

NEW: HI extraction result published to latest_data and /api/signal
  The web layer now receives hi_peak_velocity_km_s, hi_snr so the
  dashboard can display the Doppler-shifted emission.
"""
import cupy as cp
USE_CUPY = True
import logging
import sys
import os

_proj_root = os.path.dirname(os.path.abspath(__file__))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import time
import gc
import numpy as np
from collections import deque
from astropy.io import fits
import argparse
import signal as _signal_mod
import socket
import json

# ── Graceful shutdown ────────────────────────────────────────────────────────
SHUTDOWN_REQUESTED = False
_shutdown_event = __import__('threading').Event()


def _handle_exit(sig, frame):
    global SHUTDOWN_REQUESTED
    print(f'\nShutdown requested (signal {sig})')
    SHUTDOWN_REQUESTED = True
    _shutdown_event.set()


_signal_mod.signal(_signal_mod.SIGINT,  _handle_exit)
_signal_mod.signal(_signal_mod.SIGTERM, _handle_exit)
try:
    _signal_mod.signal(_signal_mod.SIGABRT, _handle_exit)
except Exception:
    pass


def _clean_shutdown(sdr=None, model=None):
    _log = logging.getLogger('aic')
    _log.info('Clean shutdown initiated')
    if sdr is not None and not getattr(sdr, '_already_closed', False):
        try:
            sdr.stop_rx() if hasattr(sdr, 'stop_rx') else None
        except Exception:
            pass
        try:
            sdr.close() if hasattr(sdr, 'close') else None
        except Exception:
            pass
        time.sleep(0.3)
    if model is not None:
        try:
            import tensorflow as _tf
            _tf.keras.backend.clear_session()
        except Exception:
            pass
    try:
        logging.shutdown()
    except Exception:
        pass
    os._exit(0)


# ── Optional hardware / ML libs ──────────────────────────────────────────────
try:
    from pyhackrf2 import HackRF
except Exception:
    HackRF = None

os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices=false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf
try:
    tf.keras.mixed_precision.set_global_policy('float32')
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    import psutil
except Exception:
    psutil = None

try:
    from logging_setup import configure_logging
    _root_logger = configure_logging()
except Exception:
    _root_logger = logging.getLogger()

logger = logging.getLogger('aic')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s'))
    logger.addHandler(ch)

# ── SDR / frequency constants ────────────────────────────────────────────────
LNB_LOW  = float(os.getenv('LNB_LOW',  '9750000000'))
LNB_HIGH = float(os.getenv('LNB_HIGH', '10600000000'))

fs          = 2e6                          # 2 MHz sample rate
sample_rate = fs
HI_FREQ     = 1420.40575177e6             # Hz  (hydrogen line rest freq)
bandwidth   = float(os.getenv('SDR_BANDWIDTH', str(fs)))
center_freq = float(os.getenv('SDR_CENTER_FREQ', str(HI_FREQ)))
freq_start  = 1420.390e6
freq_stop   = 1420.419e6
gain        = int(float(os.getenv('SDR_GAIN', '5')))
USE_LNB     = False
LO          = 0.0
center_freq = (freq_start + freq_stop) / 2
CONSECUTIVE_DETECTIONS = int(os.getenv('CONSECUTIVE_DETECTIONS', '3'))

# ── Processing imports ───────────────────────────────────────────────────────
try:
    from processing import (
        create_model, train_model, check_model_for_nans,
        calibrate_model_threshold, inject_wow_signal,
        _prepare_input_for_model, predict_signal,
        MODEL_RECOMMENDED_THRESHOLD, RUNTIME_DETECTION_THRESHOLD,
        MIN_DETECTION_THRESHOLD,
    )
except Exception:
    logger.warning('processing module unavailable — using stubs')

    def create_model(input_shape=(8192, 1)):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import InputLayer, Dense, Flatten
        m = Sequential([InputLayer(input_shape=input_shape), Flatten(),
                        Dense(32, activation='relu'), Dense(1, activation='sigmoid')])
        m.compile(optimizer='adam', loss='binary_crossentropy')
        return m

    def train_model(model, path): return None
    def check_model_for_nans(model): return True
    def calibrate_model_threshold(model, **kw): return {'recommended_threshold': None}
    def inject_wow_signal(s): return s
    def _prepare_input_for_model(s, **kw): return np.zeros((1, 8192, 1), dtype=np.float32)
    def predict_signal(model, s): return False, 0.0
    MODEL_RECOMMENDED_THRESHOLD = None
    RUNTIME_DETECTION_THRESHOLD = None
    MIN_DETECTION_THRESHOLD = 0.05

# ── Signal processing imports ────────────────────────────────────────────────
try:
    from advance_signal_processing import (
        remove_lnb_effect, denoise_signal, process_fft,
        extract_hi_line, _get_workspace, set_lnb_enabled,
    )
    _HAS_SIGNAL_PROC = True
except Exception:
    _HAS_SIGNAL_PROC = False
    remove_lnb_effect = denoise_signal = process_fft = None
    extract_hi_line   = _get_workspace = set_lnb_enabled = None

try:
    from radiometry import RadiometryCalibrator
    _HAS_RADIOMETRY = True
except Exception:
    _HAS_RADIOMETRY = False
    RadiometryCalibrator = None

try:
    from coordinates import get_pointing_metadata, freq_to_velocity_lsr, apply_lsr_correction
    _HAS_COORDS = True
except Exception:
    _HAS_COORDS = False
    get_pointing_metadata = None
    freq_to_velocity_lsr  = None
    apply_lsr_correction  = None

try:
    from candidate_db import CandidateDB
    _HAS_CANDIDATE_DB = True
except Exception:
    _HAS_CANDIDATE_DB = False
    CandidateDB = None

try:
    from rfi_mask import RollingBaseline, SpectralAverager, build_rfi_mask, apply_rfi_mask
    _HAS_RFI = True
except Exception:
    _HAS_RFI = False
    RollingBaseline = SpectralAverager = build_rfi_mask = apply_rfi_mask = None

try:
    from dedispersion import IncoherentDedisperser
    _HAS_DEDISP = True
except Exception:
    _HAS_DEDISP = False
    IncoherentDedisperser = None

# ── Detection job queue ──────────────────────────────────────────────────────
try:
    from detection import start_detection_job_consumer, DETECTION_JOB_QUEUE
except Exception:
    def start_detection_job_consumer(*a, **kw): return None, None
    DETECTION_JOB_QUEUE = None

# ── Shared state for Flask ───────────────────────────────────────────────────
# Import web's pre-seeded dict so defaults like 'fft_history' exist from
# the start.  web.py must NOT import aic2 at module level (circular import),
# so we lazy-import here only to share the already-constructed objects.
from threading import Lock
from collections import deque
try:
    import web as _web_mod
    latest_data: dict = _web_mod.latest_data
    data_lock: Lock   = _web_mod.data_lock
except Exception:
    # Fallback: running aic2 standalone without Flask
    latest_data: dict = {'fft_history': deque(maxlen=100)}
    data_lock: Lock   = Lock()


# ── Pre-flight dataset download ──────────────────────────────────────────────
def _preflight_download():
    try:
        from training import download_all_datasets
        data_root = os.getenv('DATA_DIR', 'data')
        logger.info('Starting dataset pre-flight download ...')
        results = download_all_datasets(data_root=data_root)
        ok = sum(1 for v in results.values() if v)
        logger.info(f'Pre-flight complete: {ok}/{len(results)} datasets ready')
    except Exception as e:
        logger.warning(f'Pre-flight download failed ({e}) — using synthetic data')


# ── SDR connection ───────────────────────────────────────────────────────────
def connect_to_server():
    sdr_server = os.getenv('SDR_SERVER', '').strip()
    if sdr_server:
        host, *rest = sdr_server.split(':')
        port = int(rest[0]) if rest else 8888
        logger.info(f'Connecting to remote SDR server {host}:{port}')

        class RemoteHackRFClient:
            def __init__(self):
                self.host = host; self.port = port
                self.sock = None; self.buffer = bytearray()

            def connect(self):
                self.sock = socket.create_connection((self.host, self.port), timeout=10)
                params = {
                    'start_freq': int(freq_start), 'end_freq': int(freq_stop),
                    'sample_rate': int(fs), 'gain': float(gain),
                    'duration_seconds': 0, 'buffer_size': 8192,
                }
                self.sock.sendall(json.dumps(params).encode())

            def _read_exact(self, n):
                while len(self.buffer) < n:
                    chunk = self.sock.recv(65536)
                    if not chunk:
                        raise ConnectionError('Remote SDR socket closed')
                    self.buffer.extend(chunk)
                out = bytes(self.buffer[:n])
                del self.buffer[:n]
                return out

            def read_samples(self, n):
                return np.frombuffer(self._read_exact(int(n) * 8), dtype=np.complex64)

            def close(self):
                try: self.sock and self.sock.close()
                except Exception: pass

        client = RemoteHackRFClient()
        client.connect()
        logger.info('Connected to remote SDR server')
        return client

    if HackRF is None:
        raise ConnectionError('pyhackrf2 not installed and no SDR_SERVER set')

    sdr = HackRF()
    try:
        sdr.sample_rate = sample_rate
        sdr.center_freq = int((freq_start + freq_stop) / 2)
        sdr.bandwidth   = int(sample_rate)
        sdr.lna_gain    = gain
        sdr.vga_gain    = gain
        sdr.amp_enable  = False
    except Exception:
        pass
    logger.info('Local HackRF initialised')
    return sdr


# ── Ring buffer (replaces np.append accumulator) ─────────────────────────────
class _ChunkRing:
    """
    Fixed-capacity ring buffer for raw complex samples.
    Never copies more than `chunk_size` samples on push/pop.
    Solves the O(n²) growth of the old np.append() accumulator.
    """

    def __init__(self, capacity: int, dtype=np.complex64):
        self._cap   = int(capacity)
        self._buf   = np.zeros(self._cap, dtype=dtype)
        self._head  = 0    # write pointer
        self._count = 0    # valid samples

    def push(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=self._buf.dtype).ravel()
        n = len(data)
        if n == 0:
            return
        if n >= self._cap:
            self._buf[:] = data[-self._cap:]
            self._head   = 0
            self._count  = self._cap
            return
        end = self._head + n
        if end <= self._cap:
            self._buf[self._head:end] = data
        else:
            first = self._cap - self._head
            self._buf[self._head:] = data[:first]
            self._buf[:n - first]  = data[first:]
        self._head  = end % self._cap
        self._count = min(self._count + n, self._cap)

    def pop_chunk(self, chunk_size: int) -> np.ndarray | None:
        """Pop exactly chunk_size contiguous samples, or None if unavailable."""
        if self._count < chunk_size:
            return None
        tail = (self._head - self._count) % self._cap
        end  = tail + chunk_size
        if end <= self._cap:
            out = self._buf[tail:end].copy()
        else:
            first = self._cap - tail
            out   = np.concatenate([self._buf[tail:], self._buf[:chunk_size - first]])
        self._count -= chunk_size
        return out

    @property
    def available(self) -> int:
        return self._count


# ── DetectionBuffer (unchanged logic, same API) ──────────────────────────────
class DetectionBuffer:
    def __init__(self, fs, duration_sec=72, dtype=np.float32, decim=20):
        self.fs          = int(fs)
        self.decim       = decim
        self.max_samples = int(self.fs * duration_sec / self.decim)
        self.buffer      = np.zeros(self.max_samples, dtype=dtype)
        self.idx         = 0
        self.full        = False

    def add_samples(self, samples):
        mag = np.abs(samples)[::self.decim].astype(self.buffer.dtype, copy=False)
        n   = len(mag)
        end = self.idx + n
        if end <= self.max_samples:
            self.buffer[self.idx:end] = mag
        else:
            first = self.max_samples - self.idx
            self.buffer[self.idx:] = mag[:first]
            self.buffer[:n - first] = mag[first:]
            self.full = True
        self.idx = (self.idx + n) % self.max_samples

    def get_buffer(self):
        if not self.full:
            return self.buffer[:self.idx].copy()
        return np.concatenate([self.buffer[self.idx:], self.buffer[:self.idx]])

    def clear(self):
        self.buffer[:] = 0; self.idx = 0; self.full = False


# ── BufferTracker (stats display) ────────────────────────────────────────────
class BufferTracker:
    def __init__(self, maxlen=100):
        self.values      = deque(maxlen=maxlen)
        self.confidences = deque(maxlen=maxlen)
        self.detections  = deque(maxlen=maxlen)
        self.timestamps  = deque(maxlen=maxlen)

    def update(self, value, confidence, detection):
        self.values.append(value)
        self.confidences.append(confidence)
        self.detections.append(detection)
        self.timestamps.append(time.time())

    def stats(self):
        c = len(self.values)
        ts = list(self.timestamps)
        elapsed = max(ts[-1] - ts[0], 1e-6) if len(ts) > 1 else 1e-6
        return {
            'count':           c,
            'mean_confidence': float(np.mean(self.confidences)) if self.confidences else 0.0,
            'detection_rate':  int(sum(self.detections)),
            'update_rate':     int(c / elapsed),
        }


# ── Raw snapshot ring (for constellation / inst-freq) ────────────────────────
class RawRingBuffer:
    def __init__(self, max_samples=65536, dtype=np.complex64):
        self.max_samples = int(max_samples)
        self.buf         = np.zeros(self.max_samples, dtype=dtype)
        self.idx         = 0
        self.full        = False

    def add(self, samples):
        n = len(samples)
        if n >= self.max_samples:
            self.buf[:]  = samples[-self.max_samples:]
            self.idx     = 0
            self.full    = True
            return
        end = self.idx + n
        if end <= self.max_samples:
            self.buf[self.idx:end] = samples
            if end == self.max_samples:
                self.full = True
        else:
            first = self.max_samples - self.idx
            self.buf[self.idx:] = samples[:first]
            self.buf[:n - first] = samples[first:]
            self.full = True
        self.idx = (self.idx + n) % self.max_samples

    def get_snapshot(self, length):
        length = int(length)
        if not self.full and self.idx < length:
            out = np.zeros(length, dtype=self.buf.dtype)
            out[-self.idx:] = self.buf[:self.idx]
            return out
        start = (self.idx - length) % self.max_samples
        if start + length <= self.max_samples:
            return self.buf[start:start + length].copy()
        first = self.max_samples - start
        return np.concatenate([self.buf[start:], self.buf[:length - first]])


# ── Main processing loop ─────────────────────────────────────────────────────
def process_continuous_stream(sdr, model, output_dir: str):
    import datetime
    import sys
    import select

    logger.info('Starting continuous stream processing')

    chunk_size = 8192

    # Shared workspace — Kaiser window allocated once
    ws = _get_workspace(chunk_size) if _get_workspace else None

    tracker          = BufferTracker(maxlen=100)
    det_buf          = DetectionBuffer(fs=fs, duration_sec=72)
    raw_ring         = RawRingBuffer(max_samples=65536)
    ring             = _ChunkRing(capacity=chunk_size * 8)

    # ── Radiometric calibrator ────────────────────────────────────────────────
    calibrator = None
    if _HAS_RADIOMETRY:
        try:
            calibrator = RadiometryCalibrator(
                freq_hz=center_freq,
                bandwidth_hz=fs,
            )
            # Auto-set T_sys from env if provided, otherwise uncalibrated
            t_sys_env = os.getenv('T_SYS_K')
            if t_sys_env:
                calibrator.set_tsys_manual(float(t_sys_env))
                logger.info(f'T_sys set from env: {float(t_sys_env):.1f} K')
            else:
                logger.info('Radiometry uncalibrated — set T_SYS_K env var or '
                            'run Y-factor calibration via /api/calibrate')
        except Exception as e:
            logger.warning(f'RadiometryCalibrator init failed: {e}')

    # ── Candidate database ────────────────────────────────────────────────────
    candidate_db = None
    if _HAS_CANDIDATE_DB:
        try:
            db_path = os.path.join(output_dir, 'candidates.db')
            candidate_db = CandidateDB(db_path)
        except Exception as e:
            logger.warning(f'CandidateDB init failed: {e}')

    # ── RFI masking + rolling baseline ───────────────────────────────────────
    rolling_baseline = None
    spectral_averager = None
    rfi_mask_cache = None    # computed once per stream on first FFT
    if _HAS_RFI:
        try:
            chunk_dur_s = chunk_size / fs
            rolling_baseline  = RollingBaseline(
                n_bins=chunk_size, chunk_rate_hz=1.0 / chunk_dur_s)
            avg_sec = float(os.getenv('HI_AVERAGE_SECONDS', '30.0'))
            spectral_averager = SpectralAverager(
                n_bins=chunk_size,
                average_seconds=avg_sec,
                chunk_duration_s=chunk_dur_s)
            logger.info(f'Spectral averaging: {avg_sec:.0f}s window')
        except Exception as e:
            logger.warning(f'RFI/baseline init failed: {e}')

    # ── Incoherent dedisperser ────────────────────────────────────────────────
    dedisperser = None
    if _HAS_DEDISP:
        try:
            dedisperser = IncoherentDedisperser(
                center_freq_hz=center_freq,
                bandwidth_hz=fs,
                sample_rate_hz=fs,
            )
        except Exception as e:
            logger.warning(f'Dedisperser init failed: {e}')

    detection_streak  = 0
    empty_reads       = 0
    MAX_EMPTY         = 3
    last_stat_log     = datetime.datetime.now()
    _chunk_counter    = 0

    # Terminal key-press (best-effort; no-op in non-TTY environments)
    IS_TTY = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
    old_settings = None
    try:
        if IS_TTY:
            import termios, tty
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
    except Exception:
        IS_TTY = False

    def _key():
        if IS_TTY and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    inject_wow = False

    try:
        while not SHUTDOWN_REQUESTED:

            # ── Key handling ─────────────────────────────────────────────
            key = _key()
            if key == 'w':
                inject_wow = True
                logger.info('WOW signal injection queued')
            elif key == 'q':
                logger.info('Quit key detected')
                break

            # ── Read SDR chunk ───────────────────────────────────────────
            try:
                raw = sdr.read_samples(chunk_size)
                if raw is None or len(raw) == 0:
                    empty_reads += 1
                    if empty_reads >= MAX_EMPTY:
                        logger.error('Too many empty reads — reconnecting')
                        sdr.close()
                        sdr = connect_to_server()
                        empty_reads = 0
                    continue
                empty_reads = 0
            except Exception as e:
                logger.error(f'SDR read error: {e}')
                empty_reads += 1
                if empty_reads >= MAX_EMPTY:
                    try: sdr.close()
                    except Exception: pass
                    sdr = connect_to_server()
                    empty_reads = 0
                continue

            # ── Push to ring (zero-copy accumulator) ─────────────────────
            ring.push(raw.astype(np.complex64))
            _chunk_counter += 1

            # ── Process all complete chunks in the ring ──────────────────
            while True:
                process_chunk = ring.pop_chunk(chunk_size)
                if process_chunk is None:
                    break                  # not enough data yet — wait for more

                # ── Denoise ──────────────────────────────────────────────
                try:
                    if _HAS_SIGNAL_PROC and denoise_signal is not None:
                        processed = denoise_signal(process_chunk)
                    else:
                        processed = process_chunk
                except Exception:
                    processed = process_chunk

                # ── WOW injection ─────────────────────────────────────────
                if inject_wow:
                    processed  = inject_wow_signal(processed)
                    inject_wow = False
                    logger.info('WOW signal injected')

                # ── Feed ring buffers ─────────────────────────────────────
                det_buf.add_samples(processed)
                raw_ring.add(process_chunk)

                # ── Signal strength (RMS → dB) ────────────────────────────
                pred_samples = np.real(processed) if np.iscomplexobj(processed) else processed
                rms = float(np.sqrt(np.mean(np.square(np.abs(pred_samples)))))
                sig_db = max(20.0 * np.log10(rms + 1e-12), -200.0)

                # ── Radiometric calibration ───────────────────────────────────
                flux_jy = None
                t_ant_k = None
                if calibrator is not None:
                    try:
                        t_ant_k = calibrator.voltage_to_t_ant(pred_samples)
                        flux_jy = calibrator.t_ant_to_jy(t_ant_k)
                    except Exception:
                        pass

                # ── Pointing & LSR metadata ───────────────────────────────────
                pointing = {}
                if _HAS_COORDS and get_pointing_metadata is not None:
                    try:
                        pointing = get_pointing_metadata(utc_time=now)
                    except Exception:
                        pass

                # ── FFT + RFI masking + baseline + averaging ──────────────────
                hi_result         = {}
                fft_dict          = {'magnitude': [], 'frequency': [], 'power': [], 'phase': []}
                fft_mag_for_model = None
                averaged_spectrum = None
                _cfreq = (freq_start + freq_stop) / 2.0
                try:
                    fft_mag, fft_freq, _, fft_phase, fft_power = process_fft(
                        processed, chunk_size, fs,
                        center_freq=_cfreq,
                        workspace=ws,
                    )
                    # Pass RAW fft_mag to model — the model was trained on
                    # raw log-magnitude spectrum, not baseline-subtracted power.
                    # Baseline subtraction is only applied for HI extraction.
                    fft_mag_for_model = fft_mag

                    # Build RFI mask once (bins don't change between chunks)
                    if _HAS_RFI and rfi_mask_cache is None and build_rfi_mask is not None:
                        try:
                            rfi_mask_cache = build_rfi_mask(fft_freq)
                        except Exception:
                            pass

                    # Apply RFI mask to power spectrum
                    power_clean = fft_power.copy()
                    if _HAS_RFI and rfi_mask_cache is not None and apply_rfi_mask is not None:
                        try:
                            power_clean = apply_rfi_mask(fft_power, rfi_mask_cache,
                                                          fill='interp')
                        except Exception:
                            pass

                    # Rolling baseline subtraction (removes slow RFI)
                    if rolling_baseline is not None:
                        try:
                            power_clean, _ = rolling_baseline.subtract_with_baseline(power_clean)
                        except Exception:
                            pass

                    # Spectral averaging (radiometer equation: SNR ∝ √τ)
                    if spectral_averager is not None:
                        try:
                            averaged_spectrum = spectral_averager.push(power_clean)
                        except Exception:
                            pass

                    # Use averaged spectrum for HI extraction when available
                    power_for_hi = averaged_spectrum if averaged_spectrum is not None else power_clean

                    # Apply LSR correction to velocity axis
                    lsr_corr = pointing.get('lsr_correction_km_s', 0.0)

                    freq_mhz = fft_freq / 1e6
                    # Store as numpy — web.py converts to list on demand.
                    # Avoids 4 × 8192-element Python list allocation every 4ms chunk.
                    fft_dict = {
                        'magnitude': fft_mag,
                        'frequency': freq_mhz,
                        'power':     power_clean,
                        'phase':     fft_phase,
                    }

                    # ── HI extraction on baseline-subtracted, averaged spectrum ──
                    if _HAS_SIGNAL_PROC and extract_hi_line is not None:
                        hi_result = extract_hi_line(
                            fft_freq_hz=fft_freq,
                            fft_power=power_for_hi.astype(np.float64),
                            velocity_range_km_s=500.0,
                        )
                        # Apply LSR correction to the reported peak velocity
                        if hi_result and 'peak_velocity' in hi_result:
                            hi_result['peak_velocity'] = (
                                hi_result['peak_velocity'] + lsr_corr)

                except Exception as e:
                    logger.error(f'FFT/HI extraction failed: {e}')

                # ── Incoherent dedispersion search ────────────────────────────
                dedisp_result = {}
                if dedisperser is not None:
                    try:
                        dedisp_result = dedisperser.push(process_chunk)
                    except Exception:
                        pass

                # ── Model prediction (time domain + FFT spectrum) ─────────────
                detection, confidence = predict_signal(
                    model, pred_samples, fft_magnitude=fft_mag_for_model
                )

                # ── Hysteresis ────────────────────────────────────────────
                detection_streak = (detection_streak + 1) if detection else 0
                stable           = detection_streak >= CONSECUTIVE_DETECTIONS

                # ── Update shared web state ───────────────────────────────
                tracker.update(float(np.mean(pred_samples)), confidence, detection)
                now = datetime.datetime.now()

                # ── Candidate DB logging (on stable detection) ─────────────
                if stable and candidate_db is not None:
                    try:
                        candidate_db.insert(
                            confidence         = float(confidence),
                            timestamp_utc      = now,
                            signal_strength_db = float(sig_db) if np.isfinite(sig_db) else None,
                            flux_jy            = flux_jy,
                            t_ant_k            = t_ant_k,
                            hi_peak_vel_kms    = float(hi_result.get('peak_velocity', 0.0)),
                            hi_snr             = float(hi_result.get('snr', 0.0)),
                            ra_deg             = pointing.get('ra_deg'),
                            dec_deg            = pointing.get('dec_deg'),
                            glon_deg           = pointing.get('glon_deg'),
                            glat_deg           = pointing.get('glat_deg'),
                            az_deg             = pointing.get('az_deg'),
                            el_deg             = pointing.get('el_deg'),
                            lsr_corr_kms       = pointing.get('lsr_correction_km_s'),
                            freq_center_mhz    = float(center_freq / 1e6),
                            bandwidth_mhz      = float(fs / 1e6),
                        )
                    except Exception as e:
                        logger.error(f'Candidate DB insert failed: {e}')

                    # Check dedisperser for FRB candidacy
                    if dedisp_result.get('is_candidate') and not dedisp_result.get('bw_limited'):
                        logger.warning(
                            f'FRB/pulsar candidate: DM={dedisp_result["best_dm"]:.1f} '
                            f'SNR={dedisp_result["best_snr"]:.1f}σ'
                        )

                cal_status = {}
                if calibrator is not None:
                    try:
                        cal_status = calibrator.status_dict()
                        if cal_status.get('needs_recal'):
                            logger.warning('Gain drift >5% or calibration >1h old — recalibrate')
                    except Exception:
                        pass

                web_data = {
                    'timestamp':          now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'frequency':          f'{freq_start/1e6:.3f} MHz',
                    'freq_start_mhz':     float(freq_start / 1e6),
                    'freq_stop_mhz':      float(freq_stop  / 1e6),
                    'sample_rate_mhz':    float(fs / 1e6),
                    'signal_strength':    f'{sig_db:.2f} dB' if np.isfinite(sig_db) else 'N/A',
                    'signal_strength_db': float(sig_db) if np.isfinite(sig_db) else None,
                    'flux_jy':            flux_jy,
                    't_ant_k':            t_ant_k,
                    'confidence':         float(confidence),
                    'detection':          bool(detection),
                    'stable_detection':   bool(stable),
                    'raw_model_output':   float(confidence),
                    'fft_data':           fft_dict,
                    'buffer_stats':       tracker.stats(),
                    'hi_peak_velocity_km_s': float(hi_result.get('peak_velocity', 0.0)),
                    'hi_snr':                float(hi_result.get('snr', 0.0)),
                    # Pointing / coordinates
                    'ra_deg':             pointing.get('ra_deg'),
                    'dec_deg':            pointing.get('dec_deg'),
                    'glon_deg':           pointing.get('glon_deg'),
                    'glat_deg':           pointing.get('glat_deg'),
                    'az_deg':             pointing.get('az_deg'),
                    'el_deg':             pointing.get('el_deg'),
                    'lsr_correction_km_s': pointing.get('lsr_correction_km_s', 0.0),
                    'lst_deg':            pointing.get('lst_deg'),
                    # Calibration status
                    'calibration':        cal_status,
                    # Dedispersion
                    'dedisp_best_dm':     float(dedisp_result.get('best_dm',  0.0)),
                    'dedisp_best_snr':    float(dedisp_result.get('best_snr', 0.0)),
                    'dedisp_candidate':   bool(dedisp_result.get('is_candidate', False)),
                    # Spectral averaging progress
                    'spectral_avg_progress': float(spectral_averager.progress)
                                               if spectral_averager else 0.0,
                    'system_status': {
                        'gpu_available': (USE_CUPY and getattr(cp, 'cuda', None) is not None and cp.cuda.runtime.getDeviceCount() > 0),
                        'cpu_usage':      psutil.cpu_percent()                    if psutil else 0,
                        'memory_usage':   psutil.virtual_memory().percent         if psutil else 0,
                        'process_memory': psutil.Process().memory_info().rss // (1024*1024) if psutil else 0,
                    },
                }

                with data_lock:
                    latest_data.update(web_data)

                # ── Periodic log + memory housekeeping ────────────────────
                if (now - last_stat_log).total_seconds() > 5:
                    st = tracker.stats()
                    logger.info(
                        f'AvgConf={st["mean_confidence"]:.3f} '
                        f'Det/100={st["detection_rate"]} '
                        f'HI_vel={hi_result.get("peak_velocity", 0):.1f}km/s '
                        f'HI_SNR={hi_result.get("snr", 0):.1f} '
                        f'Ring={ring.available}samp'
                    )
                    last_stat_log = now

                    # Periodic GC: Python's reference-counting handles most
                    # short-lived objects, but cyclic garbage (TF tensors,
                    # numpy views) accumulates. Collect every 5-second log tick
                    # (~5s intervals) to prevent multi-hour heap growth.
                    gc.collect()

                # ── Stable detection: offload to background ───────────────
                if stable and DETECTION_JOB_QUEUE is not None:
                    full_buf     = det_buf.get_buffer()
                    raw_snapshot = raw_ring.get_snapshot(8192)
                    effective_fs = float(fs) / float(det_buf.decim)
                    try:
                        DETECTION_JOB_QUEUE.put_nowait(
                            (full_buf.copy(), raw_snapshot[::10].copy(),
                             now, output_dir, effective_fs)
                        )
                        logger.info(f'Detection queued (conf={confidence:.3f})')
                    except Exception as e:
                        logger.error(f'Detection queue put failed: {e}')
                    det_buf.clear()

    except KeyboardInterrupt:
        logger.info('Interrupted by user')
    except Exception:
        logger.exception('Unexpected error in processing loop')
    finally:
        if IS_TTY and old_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception:
                pass
        try:
            sdr.close()
        except Exception:
            pass
        try:
            sdr._already_closed = True
        except Exception:
            pass
        logger.info('Processing stopped')


# ── FITS save helpers ─────────────────────────────────────────────────────────
def save_fits(processed_samples, output_dir, timestamp):
    fits_dir = os.path.join(output_dir, 'fits')
    os.makedirs(fits_dir, exist_ok=True)
    fname = os.path.join(fits_dir, f'signal_{timestamp.strftime("%Y%m%d_%H%M%S")}.fits')
    try:
        data_c = np.asarray(processed_samples, dtype=np.complex64)
        real_p = np.real(data_c).astype(np.float32)
        imag_p = np.imag(data_c).astype(np.float32)
        hdu = fits.PrimaryHDU(np.vstack([real_p, imag_p]))
        hdu.header['DATE']    = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        hdu.header['TELESCOP'] = 'HackRF One'
        hdu.header['LOFREQ']  = float(LO / 1e6)
        hdu.header['IFCENT']  = float(center_freq / 1e6)
        hdu.header['COMPLEX'] = 'REAL_IMAG'
        fits.HDUList([hdu]).writeto(fname, overwrite=True)
        logger.info(f'FITS saved: {fname}')
    except Exception as e:
        logger.error(f'FITS write failed: {e}')


# ── main() ────────────────────────────────────────────────────────────────────
def main():
    try:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(8)
    except RuntimeError:
        pass

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
        try: tf.keras.mixed_precision.set_global_policy('float32')
        except Exception: pass

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir',  default='output')
    parser.add_argument('--band',              choices=['low', 'high'], default='low')
    parser.add_argument('-lnboff', '--lnboff', action='store_true')
    parser.add_argument('--quick-retrain',     action='store_true')
    parser.add_argument('--path',              default='data')
    parser.add_argument('--train',             action='store_true')
    parser.add_argument('--force-train',       action='store_true')
    # Observer location (also readable from env vars OBSERVER_LAT/LON/ALT)
    parser.add_argument('--lat',  type=float, default=None, help='Observer latitude (deg N)')
    parser.add_argument('--lon',  type=float, default=None, help='Observer longitude (deg E)')
    parser.add_argument('--alt',  type=float, default=None, help='Observer altitude (m)')
    parser.add_argument('--az',   type=float, default=None, help='Antenna azimuth (deg)')
    parser.add_argument('--el',   type=float, default=None, help='Antenna elevation (deg)')
    parser.add_argument('--tsys', type=float, default=None, help='System temperature K (skip Y-factor)')
    args = parser.parse_args()

    # Push CLI location args into env so coordinates.py picks them up
    if args.lat  is not None: os.environ['OBSERVER_LAT']  = str(args.lat)
    if args.lon  is not None: os.environ['OBSERVER_LON']  = str(args.lon)
    if args.alt  is not None: os.environ['OBSERVER_ALT']  = str(args.alt)
    if args.az   is not None: os.environ['ANTENNA_AZ']    = str(args.az)
    if args.el   is not None: os.environ['ANTENNA_EL']    = str(args.el)
    if args.tsys is not None: os.environ['T_SYS_K']       = str(args.tsys)

    args.train       |= os.getenv('TRAIN',       '').lower() in ('1','true','yes')
    args.force_train |= os.getenv('FORCE_TRAIN', '').lower() in ('1','true','yes')

    if args.quick_retrain:
        import quick_retrain
        quick_retrain.main()
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)

    global USE_LNB, LO
    lnb_band = LNB_LOW if args.band == 'low' else LNB_HIGH
    if args.lnboff:
        USE_LNB = False; LO = 0.0
        if set_lnb_enabled: set_lnb_enabled(False)
        logger.info('LNB disabled')
    else:
        USE_LNB = True; LO = lnb_band
        if set_lnb_enabled: set_lnb_enabled(True)

    model_path = os.path.join('models', 'full_state', 'full_model.keras')
    model = None

    if os.path.exists(model_path) and not (args.train or args.force_train):
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info('Model loaded')
        except Exception as e:
            logger.error(f'Model load failed: {e}')

    if model is None:
        logger.info('Training new model')
        model = create_model()
        _preflight_download()
        train_model(model, args.path)

    # Verify — dual-input model requires both time and FFT tensors
    dummy_time = np.random.rand(1, 8192, 1).astype(np.float32)
    dummy_fft  = np.zeros((1, 1024, 1), dtype=np.float32)
    try:
        pred = model.predict([dummy_time, dummy_fft], verbose=0)
        if np.isnan(pred).any():
            logger.error('Model outputs NaN — aborting')
            sys.exit(1)
        logger.info(f'Model OK: dummy pred={float(pred[0,0]):.4f}')
    except Exception as e:
        logger.error(f'Model test failed: {e}')
        sys.exit(1)

    try:
        calib = calibrate_model_threshold(model)
        logger.info(f'Calibration: threshold={calib["recommended_threshold"]}')
    except Exception as e:
        logger.warning(f'Calibration failed: {e}')

    if not check_model_for_nans(model):
        logger.error('NaN/Inf in model weights — aborting')
        sys.exit(1)

    try:
        _fwd, _mp = start_detection_job_consumer()
    except Exception as e:
        _fwd = _mp = None
        logger.warning(f'Detection consumer failed to start: {e}')

    try:
        sdr = connect_to_server()
        process_continuous_stream(sdr, model, args.output_dir)
    except Exception as e:
        logger.error(f'SDR processing failed: {e}')
    finally:
        try:
            if DETECTION_JOB_QUEUE: DETECTION_JOB_QUEUE.put(None)
        except Exception: pass
        try:
            if _fwd: _fwd.join(timeout=10)
        except Exception: pass
        try:
            if _mp:  _mp.join(timeout=10)
        except Exception: pass
        _clean_shutdown(
            sdr=locals().get('sdr'),
            model=model,
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from threading import Thread
    import web

    app = web.app
    # latest_data and data_lock are already shared (aic2 imported them from web)
    # No rebind needed.

    proc_thread = Thread(target=main, daemon=True, name='aic-processing')
    proc_thread.start()

    def _watch():
        _shutdown_event.wait()
        logger.info('Shutdown event — stopping Flask')
        time.sleep(1.0)
        _clean_shutdown()

    Thread(target=_watch, daemon=True, name='shutdown-watcher').start()

    logger.info('Web server starting on http://0.0.0.0:5001')
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f'Web app error: {e}')
    finally:
        SHUTDOWN_REQUESTED = True
        _shutdown_event.set()
        _clean_shutdown()