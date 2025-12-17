from asyncio import sleep
import logging
import sys
import os
# Ensure project root is on sys.path so local modules resolve when started
# as a background process (prevents ModuleNotFoundError for local imports).
_proj_root = os.path.dirname(os.path.abspath(__file__))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)
import time
import numpy as np
from scipy.signal import lfilter, butter
from astropy.io import fits
import argparse
import os
try:
    from pyhackrf2 import HackRF
    # HackRF import succeeded
except Exception:
    HackRF = None
    # pyhackrf2 not available; local HackRF device will not be available.
    # connect_to_server() will raise ConnectionError if no remote server is configured.
import subprocess
# Prefer to disable XLA/JIT on CPU at process start to avoid runtime JIT symbol
# resolution issues on CPU-only environments (prevents __extendhfsf2 / __truncsfhf2 errors).
# Set environment flags before importing TensorFlow so they take effect early.
os.environ.setdefault('TF_XLA_FLAGS', '--tf_xla_enable_xla_devices=false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
import socket
import json
try:
    import psutil
except Exception:
    psutil = None

# Try to programmatically turn off XLA JIT at runtime to avoid CPU JIT custom-call
# symbol resolution problems on systems without full XLA support.
try:
    tf.config.optimizer.set_jit(False)
    logger = logging.getLogger('aic')
    logger.debug('tf.config.optimizer.set_jit(False) called to disable XLA JIT')
except Exception:
    pass

# Initialize logging for this module (uses logging_setup.configure_logging)
try:
    from logging_setup import configure_logging
    _root_logger = configure_logging()
except Exception:
    _root_logger = logging.getLogger()

logger = logging.getLogger('aic')
logger.setLevel(logging.DEBUG)
# Ensure at least one handler so debug logs appear on console when logging_setup isn't present
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.debug('Console logging handler attached')
logger.debug('aic2 module imported and logger configured')

# Import common constants from legacy module when available, otherwise use safe defaults
try:
    from aic import LNB_LOW, LNB_HIGH, fs, freq_start, freq_stop, sample_rate, gain
except Exception:
    LNB_LOW = 9.75e9
    LNB_HIGH = 10.6e9
    fs = 20e6
    freq_start = 1420e6
    freq_stop = 1420.4e6
    sample_rate = fs
    gain = 5

# Default LNB usage flags
USE_LNB = False
LO = 0.0

# Center frequency used for FITS headers (Hz)
center_freq = (freq_start + freq_stop) / 2

# Hysteresis / detection stability defaults
CONSECUTIVE_DETECTIONS = int(os.getenv('CONSECUTIVE_DETECTIONS', '3'))

# Import processing helpers from refactored module
try:
    from processing import (
        create_model,
        train_model,
        check_model_for_nans,
        calibrate_model_threshold,
        inject_wow_signal,
        _prepare_input_for_model,
        MODEL_RECOMMENDED_THRESHOLD,
        RUNTIME_DETECTION_THRESHOLD,
        MIN_DETECTION_THRESHOLD
    )
except Exception:
    # If processing module not available, provide safe fallbacks so runtime
    # continues (these are conservative, minimal implementations).
    logger.debug('Could not import processing helpers; installing fallback stubs')

    def create_model(input_shape=(8192, 1)):
        try:
            from tensorflow.keras import Sequential
            from tensorflow.keras.layers import InputLayer, Dense, Flatten
            m = Sequential([InputLayer(input_shape=input_shape), Flatten(), Dense(32, activation='relu'), Dense(1, activation='sigmoid')])
            m.compile(optimizer='adam', loss='binary_crossentropy')
            return m
        except Exception:
            return None

    def train_model(model, path):
        logger.warning('train_model stub called; no training performed (processing module missing)')
        return None

    def check_model_for_nans(model):
        try:
            if model is None:
                return False
            for w in model.weights:
                arr = w.numpy()
                if np.isnan(arr).any() or np.isinf(arr).any():
                    return False
            return True
        except Exception:
            return True

    def calibrate_model_threshold(model, samples=64):
        return {'recommended_threshold': None, 'pos_mean': 0.0, 'neg_mean': 0.0}

    def inject_wow_signal(samples):
        try:
            s = np.array(samples, copy=True)
            n = len(s)
            if n > 0:
                i = n // 2
                s[i:i+8] += 5.0 * np.hanning(min(8, n - i))
            return s
        except Exception:
            return samples

    def _prepare_input_for_model(samples, expected_len=8192, return_normalized=False):
        arr = np.asarray(samples)
        if np.iscomplexobj(arr):
            arr = np.real(arr)
        # pad or crop
        if arr.size < expected_len:
            pad = expected_len - arr.size
            arr = np.pad(arr, (0, pad), mode='constant')
        else:
            arr = arr[:expected_len]
        # normalize
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-6:
            norm = arr - mean
        else:
            norm = (arr - mean) / std
        norm = np.clip(norm, -5, 5)
        tensor = np.asarray(norm, dtype=np.float32).reshape(1, expected_len, 1)
        if return_normalized:
            return tensor, norm
        return tensor

    MODEL_RECOMMENDED_THRESHOLD = None
    RUNTIME_DETECTION_THRESHOLD = None
    MIN_DETECTION_THRESHOLD = 0.01

# Shared state for web UI (module-level, always defined)
from threading import Lock
latest_data = {}
data_lock = Lock()

# Import lower-level signal utilities from advance_signal_processing when available
try:
    from advance_signal_processing import remove_lnb_effect, denoise_signal, process_fft
    logger.debug('Imported signal utilities from advance_signal_processing')
except Exception:
    remove_lnb_effect = None
    denoise_signal = None
    process_fft = None
    logger.debug('advance_signal_processing utilities not available; some processing steps will be skipped')

    # Shared state for web UI and thread-safe update
    from threading import Lock
    latest_data = {}
    data_lock = Lock()

# Expose detection job API expected by tests (real implementation in detection.py)
try:
    from detection import start_detection_job_consumer, DETECTION_JOB_QUEUE
except Exception:
    def start_detection_job_consumer(*args, **kwargs):
        logger.debug('start_detection_job_consumer stub called (detection module missing)')
    DETECTION_JOB_QUEUE = None

# Attempt a late import to bind the real detection queue if available
if DETECTION_JOB_QUEUE is None:
    try:
        import detection as _det
        DETECTION_JOB_QUEUE = getattr(_det, 'DETECTION_JOB_QUEUE', DETECTION_JOB_QUEUE)
        start_detection_job_consumer = getattr(_det, 'start_detection_job_consumer', start_detection_job_consumer)
        logger.debug('Bound detection job queue from detection module')
    except Exception as _e:
        logger.debug(f'Late import of detection module failed: {_e}')

def set_cpu_affinity(cores):
   logger.debug(f"Setting CPU affinity to cores: {cores}")
   p = psutil.Process(os.getpid())
   p.cpu_affinity(cores)

def connect_to_server():
    """Connect to an SDR source.

    Behavior:
    - If the environment variable `SDR_SERVER` is set (format `host:port`), connect to
      that remote server (compatible with `serverHRF.py`) and return a client object
      exposing `read_samples(n)` and `close()`.
    - Otherwise fall back to a local `HackRF()` device.
    """
    logger.debug('connect_to_server() called')
    sdr_server = os.getenv('SDR_SERVER', '').strip()
    if sdr_server:
        # Remote server connection
        try:
            host, port = (sdr_server.split(':') + [ '8888' ])[:2]
            port = int(port)
        except Exception:
            host = sdr_server
            port = 8888

        logger.info(f"Connecting to remote SDR server at {host}:{port}")

        class RemoteHackRFClient:
            def __init__(self, host, port, start_freq=None, end_freq=None, sample_rate=fs, gain_val=gain, buffer_size=8192):
                self.host = host
                self.port = port
                self.sock = None
                self.buffer = bytearray()
                self.sample_rate = int(sample_rate)
                self.start_freq = int(start_freq) if start_freq is not None else int(freq_start)
                self.end_freq = int(end_freq) if end_freq is not None else int(freq_stop)
                self.gain = float(gain_val)
                self.buffer_size = int(buffer_size)

            def connect(self):
                self.sock = socket.create_connection((self.host, self.port), timeout=10)
                params = {
                    'start_freq': int(self.start_freq),
                    'end_freq': int(self.end_freq),
                    'sample_rate': int(self.sample_rate),
                    'gain': float(self.gain),
                    'duration_seconds': 0,
                    'buffer_size': int(self.buffer_size)
                }
                # send tuning parameters as JSON
                self.sock.sendall(json.dumps(params).encode('utf-8'))

            def read_exact(self, nbytes):
                # ensure buffer has at least nbytes
                while len(self.buffer) < nbytes:
                    chunk = self.sock.recv(65536)
                    if not chunk:
                        raise ConnectionError('Remote SDR socket closed')
                    self.buffer.extend(chunk)
                out = bytes(self.buffer[:nbytes])
                del self.buffer[:nbytes]
                return out

            def read_samples(self, n):
                # server sends complex64 (2x float32) samples -> 8 bytes per sample
                nbytes = int(n) * 8
                data = self.read_exact(nbytes)
                arr = np.frombuffer(data, dtype=np.complex64)
                return arr

            def close(self):
                try:
                    if self.sock:
                        self.sock.close()
                except Exception:
                    pass

        client = RemoteHackRFClient(host, port, start_freq=freq_start, end_freq=freq_stop, sample_rate=fs, gain_val=gain, buffer_size=8192)
        try:
            client.connect()
            logger.info("Connected to remote SDR server")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to remote SDR server: {e}")
            raise ConnectionError("Remote SDR connection failed")

    # Fallback: local HackRF (if available) otherwise use a DummySDR for testing
    logger.debug("Attempting to connect to local HackRF device via USB...")
    try:
        sdr = HackRF()
        # Derive reasonable tuning parameters from available constants
        try:
            center = int((freq_start + freq_stop) / 2)
            bw = int(sample_rate)
            sdr.sample_rate = sample_rate
            # Many HackRF wrappers expect integer center frequency
            try:
                sdr.center_freq = center
            except Exception:
                pass
            try:
                sdr.bandwidth = bw
            except Exception:
                pass
            try:
                sdr.lna_gain = gain
                sdr.vga_gain = gain
            except Exception:
                pass
            try:
                sdr.amp_enable = gain > 0
                sdr.amplifier_on = True
            except Exception:
                pass
        except Exception:
            pass

        logging.info(f"Configured SDR with sample_rate={sample_rate}, gain={gain}, center_freq={(freq_start+freq_stop)/2}, bandwidth={sample_rate}")
        logger.info("HackRF device initialized successfully")
        return sdr
    except Exception as e:
        logger.error(f"Could not initialize HackRF: {e}; no fallback available")
        raise ConnectionError(f"Could not initialize HackRF: {e}")


def predict_signal(model, samples):
    try:
        logger.debug(f"Raw input samples type: {type(samples)}, dtype: {getattr(samples, 'dtype', 'N/A')}, shape: {np.shape(samples)}")

        # Use unified input preparation (handles pad/crop, real-part conversion, normalization)
        input_tensor, samples_normalized = _prepare_input_for_model(samples, expected_len=8192, return_normalized=True)
        logger.debug(f"Input tensor prepared - dtype: {input_tensor.dtype}, shape: {input_tensor.shape}")

        # If the normalized array is degenerate, bail out
        if samples_normalized is None or np.std(samples_normalized) < 1e-6:
            logger.warning("Near-zero standard deviation in input signal (after prep)")
            return (False, 0.0)

        confidence = model(input_tensor, training=False)

        # Log raw model output (safe conversion for tf.Tensor or numpy)
        try:
            raw_out = confidence
            if hasattr(raw_out, 'numpy'):
                raw_np = raw_out.numpy()
            else:
                raw_np = np.array(raw_out)
            raw_dtype = getattr(raw_np, 'dtype', type(raw_np)).__str__()
            raw_shape = getattr(raw_np, 'shape', None)
            try:
                raw_min = float(np.min(raw_np))
                raw_max = float(np.max(raw_np))
            except Exception:
                raw_min = None
                raw_max = None
            logger.debug(f"Raw model output - dtype: {raw_dtype}, shape: {raw_shape}, min: {raw_min}, max: {raw_max}")
        except Exception as e:
            logger.debug(f"Could not inspect raw model output: {e}")

        # handle models that return logits or arrays with extra dims
        try:
            confidence_value = float(confidence[0][0])
        except Exception:
            try:
                confidence_value = float(np.ravel(confidence)[0])
            except Exception:
                logger.debug("Failed to parse raw model output into float; defaulting to 0.0")
                confidence_value = 0.0

        # Use normalized statistics to derive a conservative threshold
        norm_std = float(np.std(samples_normalized)) if samples_normalized.size else 0.0
        norm_max = float(np.max(np.abs(samples_normalized))) if samples_normalized.size else 0.0
        alpha = min(1.0, norm_std if norm_std > 0 else 1.0)

        # Make default threshold more conservative to reduce false positives
        threshold = 0.7 - 0.15 * alpha
        threshold = float(np.clip(threshold, 0.4, 0.95))

        # If calibration produced a recommended threshold, prefer the stricter (higher) value
        used_threshold = threshold
        try:
            if MODEL_RECOMMENDED_THRESHOLD is not None and MODEL_RECOMMENDED_THRESHOLD > 1e-6:
                used_threshold = float(max(threshold, float(MODEL_RECOMMENDED_THRESHOLD)))
        except Exception:
            used_threshold = threshold

        # Runtime override from environment takes highest precedence
        try:
            if RUNTIME_DETECTION_THRESHOLD is not None:
                used_threshold = float(RUNTIME_DETECTION_THRESHOLD)
        except Exception:
            pass

        # Enforce a reasonable floor so tiny but non-zero model outputs are ignored
        floor_applied = False
        try:
            if MIN_DETECTION_THRESHOLD is not None and used_threshold < MIN_DETECTION_THRESHOLD:
                used_threshold = float(MIN_DETECTION_THRESHOLD)
                floor_applied = True
        except Exception:
            pass

        detection = confidence_value >= used_threshold

        logger.debug(
            f"Prediction - Confidence: {confidence_value:.6f}, NormStd: {norm_std:.4f}, NormMax: {norm_max:.4f}, "
            f"Threshold: {threshold:.4f}, UsedThreshold: {used_threshold:.6f}, FloorApplied: {floor_applied}, Detection: {detection}"
        )

        return (detection, confidence_value)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return (False, 0.0)



def process_continuous_stream(sdr, model, output_dir):
    logger.debug('Entered process_continuous_stream()')
    import sys
    import select
    import termios
    import tty
    from scipy import signal
    import datetime
    from collections import deque

    # Optional GPU array library (CuPy) — keep cp defined even if import fails
    try:
        import cupy as cp
        USE_CUPY = True
    except Exception:
        cp = None
        USE_CUPY = False
    class BufferTracker:
        def __init__(self, maxlen=100):
            self.maxlen = maxlen
            self.values = deque(maxlen=maxlen)
            self.confidences = deque(maxlen=maxlen)
            self.detections = deque(maxlen=maxlen)
            self.timestamps = deque(maxlen=maxlen)

        def update(self, value, confidence, detection):
            self.values.append(value)
            self.confidences.append(confidence)
            self.detections.append(detection)
            self.timestamps.append(time.time())

        def stats(self):
            count = len(self.values)
            mean_confidence = float(np.mean(self.confidences)) if self.confidences else 0.0
            detection_rate = int(np.sum(self.detections)) if self.detections else 0
            update_rate = int(count / max((self.timestamps[-1] - self.timestamps[0]), 1e-6)) if count > 1 else 0
            last_change = self.timestamps[-1] - self.timestamps[0] if count > 1 else 0
            return {
                'count': count,
                'capacity': self.maxlen,
                'mean_confidence': mean_confidence,
                'detection_rate': detection_rate,
                'update_rate': update_rate,
                'last_change': last_change
            }
    class DetectionBuffer:
        def __init__(self, fs, duration_sec=72, dtype=np.float32, decim=20):
            """
            fs: original SDR sampling rate in Hz
            duration_sec: how many seconds of samples to store
            decim: decimation factor to reduce storage size
            """
            self.fs = int(fs)
            self.decim = decim
            self.max_samples = int(self.fs * duration_sec / self.decim)
            self.buffer = np.zeros(self.max_samples, dtype=dtype)
            self.idx = 0
            self.full = False

        def add_samples(self, samples):
            mag = np.abs(samples)[::self.decim].astype(self.buffer.dtype, copy=False)
            n = len(mag)
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
            self.buffer[:] = 0
            self.idx = 0
            self.full = False


    class RawRingBuffer:
        """Store a short rolling buffer of raw complex samples (no decimation).

        Allows extracting a recent high-resolution snapshot for constellation/inst-freq/HR-FFT.
        """
        def __init__(self, max_samples=65536, dtype=np.complex64):
            self.max_samples = int(max_samples)
            self.dtype = dtype
            self.buf = np.zeros(self.max_samples, dtype=self.dtype)
            self.idx = 0
            self.full = False

        def add(self, samples):
            n = len(samples)
            try:
                mag = None
                # compute a small summary for logging (avoid expensive ops)
                if n > 0:
                    mag = np.abs(samples).astype(np.float32)
                    s_min = float(np.min(mag))
                    s_max = float(np.max(mag))
                    s_std = float(np.std(mag))
                else:
                    s_min = s_max = s_std = 0.0
            except Exception:
                s_min = s_max = s_std = 0.0
            logger.debug(f"RawRingBuffer.add: writing n={n} samples, idx_before={self.idx}, full_before={self.full}, mag_min={s_min:.6f}, mag_max={s_max:.6f}, mag_std={s_std:.6f}")
            if n >= self.max_samples:
                # keep only the last max_samples
                self.buf[:] = samples[-self.max_samples:]
                self.idx = 0
                self.full = True
                logger.debug(f"RawRingBuffer.add: replaced entire buffer (n >= max_samples). idx_after={self.idx}, full={self.full}")
                return
            end = self.idx + n
            # If the write fits entirely before the buffer end
            if end < self.max_samples:
                self.buf[self.idx:end] = samples
            elif end == self.max_samples:
                # Exactly fills to the end: write and mark full
                self.buf[self.idx:end] = samples
                self.full = True
            else:
                # Wrap-around write
                first = self.max_samples - self.idx
                self.buf[self.idx:] = samples[:first]
                self.buf[:n - first] = samples[first:]
                self.full = True
            self.idx = (self.idx + n) % self.max_samples
            logger.debug(f"RawRingBuffer.add: write complete idx_after={self.idx}, full_after={self.full}")

        def get_snapshot(self, length):
            length = int(length)
            if length >= self.max_samples:
                # return a copy
                return np.copy(self.buf)
            if not self.full:
                # no samples available
                if self.idx == 0:
                    # No samples have been written yet — return zeros and log debug info
                    logger.debug(f"RawRingBuffer.get_snapshot: idx=0, full={self.full}, returning zeros for length={length}")
                    return np.zeros(length, dtype=self.dtype)
                # not enough samples yet; return what's available, zero-padded front
                if self.idx < length:
                    out = np.zeros(length, dtype=self.dtype)
                    out[-self.idx:] = self.buf[:self.idx]
                    logger.debug(f"RawRingBuffer.get_snapshot: partial buffer (idx={self.idx}) returned, length_requested={length}")
                    return out
            # return last `length` samples in order
            start = (self.idx - length) % self.max_samples
            if start + length <= self.max_samples:
                out = np.copy(self.buf[start:start+length])
                logger.debug(f"RawRingBuffer.get_snapshot: returning contiguous slice start={start} length={length}")
                return out
            else:
                first = self.max_samples - start
                out = np.concatenate([self.buf[start:], self.buf[:length-first]]).copy()
                logger.debug(f"RawRingBuffer.get_snapshot: returning wrapped slice start={start} first_part={first} second_part={length-first}")
                return out


    def is_key_pressed():
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key
        return None
    
    def validate_web_data(data):
        """Ensure all values are JSON-serializable"""
        validated = {}
        for key, value in data.items():
            if isinstance(value, (np.ndarray, cp.ndarray)):
                validated[key] = value.tolist()
            elif isinstance(value, (np.generic)):
                validated[key] = float(value)
            else:
                validated[key] = value
        return validated

    old_settings = None
    # Terminal availability flags (guarded for non-interactive environments)
    try:
        import termios as _termios
        import tty as _tty
        TERM_AVAILABLE = True
    except Exception:
        TERM_AVAILABLE = False
    try:
        IS_TTY = sys.stdin.isatty()
    except Exception:
        IS_TTY = False

    if TERM_AVAILABLE and IS_TTY:
        try:
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception as e:
            logger.warning(f"Could not set terminal cbreak mode: {e}")

    try:
        logger.info("🚀 Starting enhanced continuous stream processing")
        buffer = BufferTracker(maxlen=100)
        detection_buffer = DetectionBuffer(fs=fs, duration_sec=72)
        raw_ring = RawRingBuffer(max_samples=65536)
        data_buffer = np.array([], dtype=np.complex64)
        chunk_size = 8192
        inject_wow = False
        # Running count of consecutive positive chunk detections (hysteresis)
        detection_streak = 0
        consecutive_empty_reads = 0
        max_empty_reads = 3
        last_stat_log = datetime.datetime.now()

        # Test model with synthetic signal
        test_signal = np.random.normal(0, 0.5, chunk_size)
        test_detection, test_confidence = predict_signal(model, test_signal)
        logger.info(f"✅ Model test - Confidence: {test_confidence:.4f} (should be > 0)")

        while True:
            try:
                # Check for key presses
                key = is_key_pressed()
                if key == 'w':
                    logger.info("🔧 WOW Signal injection triggered!")
                    inject_wow = True
                elif key == 'd':
                    logger.info("🔧 Debug signal injection triggered")
                    debug_signal = np.random.normal(0, 0.5, chunk_size)
                    debug_detection, debug_confidence = predict_signal(model, debug_signal)
                    logger.info(f"Debug test - Confidence: {debug_confidence:.4f}")
                elif key == 'q':
                    logger.info("🛑 Quit key detected. Shutting down gracefully...")
                    break
                    
                # Read from SDR
                try:
                    chunk = sdr.read_samples(chunk_size)
                    
                    if chunk is None or len(chunk) == 0:
                        consecutive_empty_reads += 1
                        logger.warning(f"⚠️ No samples received. Empty read count: {consecutive_empty_reads}")
                        if consecutive_empty_reads >= max_empty_reads:
                            logger.error("❌ Maximum empty reads reached, reconnecting SDR...")
                            sdr.close()
                            sdr = connect_to_server()
                            consecutive_empty_reads = 0
                        continue
                    else:
                        consecutive_empty_reads = 0
                        logger.debug(f"Read {len(chunk)} samples from SDR")

                except Exception as e:
                    logger.error(f"❌ Error reading from SDR: {e}")
                    consecutive_empty_reads += 1
                    if consecutive_empty_reads >= max_empty_reads:
                        logger.error("Reconnecting after multiple SDR read failures...")
                        sdr.close()
                        sdr = connect_to_server()
                        consecutive_empty_reads = 0
                    continue

                # Process the new chunk
                samples = chunk.astype(np.complex64)

                # If LNB is used, attempt to remove its effect (translation/notch)
                if USE_LNB:
                    try:
                        samples = remove_lnb_effect(samples, fs, notch_freq=0.0, notch_width=0.0, lnb_band=LO)
                        logger.debug("Applied LNB removal to incoming samples")
                    except Exception as e:
                        logger.warning(f"LNB removal failed: {e}")

                data_buffer = np.append(data_buffer, samples)

                while len(data_buffer) >= chunk_size:
                    process_chunk = data_buffer[:chunk_size]
                    
                    # Signal processing pipeline
                    try:
                        if USE_CUPY and getattr(cp, 'cuda', None) is not None and cp.cuda.runtime.getDeviceCount() > 0:
                            logger.debug("Running GPU denoise")
                            process_chunk_gpu = cp.asarray(process_chunk)
                            processed_samples_gpu = denoise_signal(process_chunk_gpu)
                            processed_samples = cp.asnumpy(processed_samples_gpu)
                        else:
                            logger.debug("Running CPU denoise")
                            processed_samples = denoise_signal(process_chunk)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Processing failed, using raw samples: {e}")
                        processed_samples = process_chunk

                    # Optional WOW signal injection
                    if inject_wow:
                        processed_samples = inject_wow_signal(processed_samples)
                        inject_wow = False
                        logger.info("✨ Injected WOW signal into current chunk")

                    logger.debug(
                        f"Samples stats - mean: {np.mean(samples):.4f}, "
                        f"std: {np.std(samples):.4f}, "
                        f"min: {np.min(samples):.4f}, "
                        f"max: {np.max(samples):.4f}"
                    )

                    # Prepare for prediction: use real-valued time-series (match training)
                    if np.iscomplexobj(processed_samples):
                        prediction_samples = np.real(processed_samples)
                    else:
                        prediction_samples = processed_samples

                    # Compute signal strength using RMS (power-based) which is
                    # physically meaningful for dB and always non-negative.
                    # Clamp very low values to a floor (e.g. -200 dB) to keep
                    # charts stable and avoid -inf from log10(0).
                    try:
                        rms = float(np.sqrt(np.mean(np.square(np.abs(prediction_samples)))))
                        signal_strength = 20 * np.log10(rms + 1e-12)
                        if np.isfinite(signal_strength):
                            # clamp to a reasonable floor for display
                            signal_strength = max(signal_strength, -200.0)
                        else:
                            signal_strength = float('nan')
                    except Exception:
                        signal_strength = float('nan')

                    # Add processed samples into the 72s buffer (decimated magnitude)
                    try:
                        detection_buffer.add_samples(processed_samples)
                    except Exception:
                        logger.debug("detection_buffer.add_samples failed", exc_info=True)
                    # Also append raw complex samples into the raw ring buffer
                    try:
                        raw_ring.add(process_chunk.astype(np.complex64))
                    except Exception:
                        logger.debug("raw_ring.add failed", exc_info=True)

                    # Model prediction on current chunk
                    detection, confidence_value = predict_signal(model, prediction_samples)
                    logger.debug(f"Chunk detection={detection}, confidence={confidence_value:.4f}, strength={signal_strength:.2f} dB")
                    
                    # Update buffer tracker
                    buffer.update(
                        value=np.mean(prediction_samples),
                        confidence=confidence_value,
                        detection=detection
                    )
                    
                    # Periodic logging (every 5 seconds)
                    current_time = datetime.datetime.now()
                    if (current_time - last_stat_log).total_seconds() > 5:
                        stats = buffer.stats()
                        logger.info(
                            f"📊 Buffer: {stats['count']}/{stats['capacity']} | "
                            f"AvgConf={stats['mean_confidence']:.4f} | "
                            f"DetRate={stats['detection_rate']} | "
                            f"Rate={stats['update_rate']} | "
                            f"Last={stats['last_change']}"
                        )
                        last_stat_log = current_time
                    
                    # FFT processing with proper data conversion
                    try:
                        if USE_CUPY and getattr(cp, 'cuda', None) is not None and cp.cuda.runtime.getDeviceCount() > 0:
                            fft_magnitude, fft_freq, _, fft_phase, fft_power = process_fft(
                                processed_samples_gpu,
                                chunk_size,
                                fs
                            )
                            fft_magnitude = cp.asnumpy(fft_magnitude)
                            fft_freq = cp.asnumpy(fft_freq)
                            fft_power = cp.asnumpy(fft_power)
                        else:
                            fft_magnitude, fft_freq, _, fft_phase, fft_power = process_fft(
                                processed_samples,
                                chunk_size,
                                fs
                            )
                        
                        # fft_freq is in Hz; convert to MHz and offset by start frequency in MHz
                        try:
                            freq_mhz = (np.array(fft_freq) / 1e6) + (freq_start / 1e6)
                        except Exception:
                            freq_mhz = np.array([])
                        fft_data_dict = {
                            'magnitude': np.abs(fft_magnitude).tolist(),
                            'frequency': freq_mhz.tolist(),
                            'power': np.abs(fft_power).tolist(),
                            'phase': fft_phase.tolist() if isinstance(fft_phase, np.ndarray) else []
                        }
                        logger.debug("FFT processed successfully")
                    except Exception as e:
                        logger.error(f"❌ FFT processing failed: {e}")
                        fft_data_dict = {'magnitude': [], 'frequency': [], 'power': [], 'phase': []}
                    
                    # Prepare complete web data package
                    # Determine stable detection using consecutive-hit hysteresis
                    if detection:
                        detection_streak += 1
                    else:
                        detection_streak = 0
                    stable_detection = detection_streak >= CONSECUTIVE_DETECTIONS
                    if detection_streak > 0:
                        logger.debug(f"Detection streak: {detection_streak}/{CONSECUTIVE_DETECTIONS}")
                    # Create both numeric and formatted signal strength
                    try:
                        if np.isfinite(signal_strength):
                            signal_strength_db = float(signal_strength)
                            signal_strength_display = f"{signal_strength_db:.2f} dB"
                        else:
                            signal_strength_db = None
                            signal_strength_display = "N/A"
                    except Exception:
                        signal_strength_db = None
                        signal_strength_display = "N/A"

                    web_data = {
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        'frequency': f"{freq_start/1e6:.2f} MHz",
                        'signal_strength': signal_strength_display,
                        'signal_strength_db': signal_strength_db,
                        'confidence': float(confidence_value),
                        'detection': bool(detection),
                        'stable_detection': bool(stable_detection),
                        'raw_model_output': float(confidence_value),
                        'buffer_stats': buffer.stats(),
                        'fft_data': fft_data_dict,
                        'system_status': {
                            'gpu_available': (USE_CUPY and getattr(cp, 'cuda', None) is not None and cp.cuda.runtime.getDeviceCount() > 0),
                            'cpu_usage': psutil.cpu_percent() if psutil is not None else 0,
                            'memory_usage': psutil.virtual_memory().percent if psutil is not None else 0,
                            'process_memory': (psutil.Process().memory_info().rss / 1024 / 1024) if psutil is not None else 0  # MB
                        }
                    }
                    # Update shared Flask data (ensure detection/confidence keys exist)
                    with data_lock:
                        validated = validate_web_data(web_data)
                        try:
                            validated['detection'] = bool(web_data.get('detection', False))
                        except Exception:
                            validated['detection'] = False
                        # also expose stable detection for UI/logic
                        try:
                            validated['stable_detection'] = bool(web_data.get('stable_detection', False))
                        except Exception:
                            validated['stable_detection'] = False
                        try:
                            validated['confidence'] = float(web_data.get('confidence', 0.0))
                        except Exception:
                            validated['confidence'] = 0.0
                        latest_data.update(validated)
                        logger.debug(f"Updated latest_data - detection={validated['detection']}, confidence={validated['confidence']}")
                    # 🔥 If stable detection (hysteresis) triggered, run full-buffer analysis
                    if stable_detection:

                        # Full (decimated magnitude) buffer for overview plots
                        full_buffer = detection_buffer.get_buffer()
                        # High-resolution raw snapshot (complex) for constellation/inst-freq/HR-FFT
                        raw_snapshot = raw_ring.get_snapshot(8192)

                        # Defensive checks: ensure buffers have expected shapes
                        try:
                            fb_len = len(full_buffer) if full_buffer is not None else 0
                        except Exception:
                            fb_len = 0
                        rs_len = len(raw_snapshot) if raw_snapshot is not None else 0
                        logger.info(f"🚨 Detection! Running full-buffer analysis ({fb_len} decimated samples), raw snapshot {rs_len} samples")

                        # If full_buffer is empty, try to derive an overview from the raw snapshot
                        if fb_len == 0:
                            try:
                                if rs_len > 0:
                                    # decimate raw snapshot to approximate full_buffer representation
                                    dec = detection_buffer.decim if getattr(detection_buffer, 'decim', None) else 20
                                    raw_abs = np.abs(raw_snapshot)
                                    try:
                                        logger.debug(
                                            f"Raw-snapshot pre-decimate stats - len={len(raw_abs)}, dtype={raw_abs.dtype}, min={np.min(raw_abs):.6f}, max={np.max(raw_abs):.6f}, mean={np.mean(raw_abs):.6f}, std={np.std(raw_abs):.6f}"
                                        )
                                    except Exception:
                                        logger.debug("Raw-snapshot pre-decimate stats unavailable")

                                    approx = raw_abs[::dec]

                                    # If decimation produced near-zero values, try using the current processing chunk
                                    if np.std(approx) < 1e-6:
                                        try:
                                            if 'samples' in locals() and samples is not None:
                                                approx = np.abs(samples)[::dec]
                                                logger.debug("Used current chunk for decimation fallback (raw_snapshot produced near-zero approximation)")
                                        except Exception:
                                            pass

                                    full_buffer = approx.astype(np.float32)
                                    fb_len = len(full_buffer)
                                    logger.info(f"Full buffer was empty; substituted from raw snapshot (len={fb_len})")
                                else:
                                    # final fallback: zero array
                                    full_buffer = np.zeros(1024, dtype=np.float32)
                                    fb_len = len(full_buffer)
                                    logger.info('Full buffer and raw snapshot empty; using zero-filled fallback for overview plots')
                            except Exception as _e:
                                full_buffer = np.zeros(1024, dtype=np.float32)
                                fb_len = len(full_buffer)
                                logger.info(f'Could not build substitute full_buffer: {_e}; using zeros')

                        # Inspect and log full-buffer statistics and normalized stats
                        try:
                            # Keep magnitude for logging/diagnostics but ensure the model
                            # always receives the real part (training used real-part).
                            full_abs = np.abs(full_buffer)
                            full_real = np.real(full_buffer)

                            # Prepare normalized input for inspection (use real part)
                            try:
                                input_t, samples_norm = _prepare_input_for_model(full_real, expected_len=8192, return_normalized=True)
                                logger.info(
                                    f"Full-buffer raw stats - len={len(full_buffer)}, dtype={full_buffer.dtype}, "
                                    f"min={np.min(full_buffer):.6f}, max={np.max(full_buffer):.6f}, mean={np.mean(full_buffer):.6f}, std={np.std(full_buffer):.6f}"
                                )
                                logger.info(
                                    f"Full-buffer magnitude stats - min={np.min(full_abs):.6f}, max={np.max(full_abs):.6f}, "
                                    f"mean={np.mean(full_abs):.6f}, std={np.std(full_abs):.6f}"
                                )
                                logger.info(
                                    f"Full-buffer normalized (real-part) stats - min={np.min(samples_norm):.6f}, max={np.max(samples_norm):.6f}, "
                                    f"mean={np.mean(samples_norm):.6f}, std={np.std(samples_norm):.6f}"
                                )
                                if np.std(full_abs) < 1e-6:
                                    logger.warning("Full-buffer magnitude is nearly constant; ensure real-part is used for model input.")
                            except Exception as e:
                                logger.error(f"Failed to prepare/inspect full-buffer normalized data: {e}")
                        except Exception as e:
                            logger.error(f"Failed to compute absolute/real of full buffer: {e}")

                        # Run a quick prediction synchronously on the real part to get an overall confidence
                        full_detection, full_conf = predict_signal(model, full_real)

                        timestamp = datetime.datetime.now()

                        # Compute effective sampling rate for plots (detection_buffer.decim may exist)
                        try:
                            effective_fs = float(fs) / float(detection_buffer.decim)
                        except Exception:
                            effective_fs = float(fs)

                        # Offload expensive IO/plotting to background worker so main loop continues
                        try:
                            # Log shapes for diagnostics
                            try:
                                logger.debug(f"Enqueueing detection job - full_buffer.shape={getattr(full_buffer,'shape',None)} len={len(full_buffer) if full_buffer is not None else 'None'} raw_snapshot.shape={getattr(raw_snapshot,'shape',None)} len={len(raw_snapshot) if raw_snapshot is not None else 'None'}")
                            except Exception:
                                pass
                            buf_copy = np.copy(full_buffer)
                            raw_copy = np.copy(raw_snapshot)
                            DETECTION_JOB_QUEUE.put((buf_copy, raw_copy, timestamp, output_dir, effective_fs))
                            logger.info("Enqueued background detection job for save/plots (full + raw)")
                        except Exception as e:
                            logger.error(f"Failed to enqueue background detection job: {e}")

                        logger.info(f"📦 Full-buffer detection scheduled. Detection={full_detection}, Confidence={full_conf:.4f}")

                        # Clear the 72s buffer immediately so we continue collecting new data
                        detection_buffer.clear()

                    # Move to next chunk
                    data_buffer = data_buffer[chunk_size:]

            except (ConnectionError, socket.error) as e:
                logger.error(f"❌ Connection error: {e}")
                break
            except KeyboardInterrupt:
                logger.info("🛑 Processing interrupted by user")
                break
            except Exception as e:
                logger.exception("❌ Unexpected error in processing loop")
                break
    
    finally:
        if TERM_AVAILABLE and IS_TTY and old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            except Exception as e:
                logger.warning(f"Failed to restore terminal settings: {e}")
        try:
            sdr.close()
        except Exception:
            pass
        logger.info("✅ Processing stopped and SDR closed")

def save_fits(processed_samples, output_dir, timestamp):
    fits_dir = os.path.join(output_dir, 'fits')
    os.makedirs(fits_dir, exist_ok=True)

    filename = os.path.join(fits_dir, f'signal_{timestamp.strftime("%Y%m%d_%H%M%S")}.fits')
    try:
        # Convert input to complex array
        data_c = np.array(processed_samples, dtype=np.complex64)

        # FITS does not support complex dtypes directly. Store as 2 x N float32: [real; imag]
        real_part = np.ascontiguousarray(np.real(data_c).astype(np.float32))
        imag_part = np.ascontiguousarray(np.imag(data_c).astype(np.float32))
        stacked = np.vstack([real_part, imag_part])

        hdu = fits.PrimaryHDU(stacked)
        hdr = hdu.header
        hdr['DATE']    = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        hdr['TELESCOP'] = 'HackRF One'
        hdr['OBSERVER'] = 'Automated System'
        hdr['FREQ-ST'] = freq_start / 1e6
        hdr['FREQ-EN'] = freq_stop  / 1e6
        hdr['FRQUNIT'] = 'MHz'
        hdr['LOFREQ']  = (LO/1e6) if USE_LNB else 0.0
        hdr['IFCENT']  = (center_freq/1e6)
        hdr['COMPLEX'] = 'REAL_IMAG'
        hdr['CTYPE1']  = 'REAL'  # first row
        hdr['CTYPE2']  = 'IMAG'  # second row

        hdul = fits.HDUList([hdu])
        hdul.writeto(filename, overwrite=True)
        logger.info(f"Saved FITS (real/imag rows): {filename}")
    except Exception as e:
        logger.error(f"Failed to write raw FITS: {e}")

def main():
    logger.debug('Entered main()')
    # Set threading and precision config
    try:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(8)
    except RuntimeError as e:
        logger.warning(f"Could not set TensorFlow config: {e}")

    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found GPU: {gpus}")
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            logger.error(f"Could not set GPU memory growth: {e}")
        try:
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Enabled mixed_float16 policy for GPU")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")
    else:
        logger.warning("No GPU found, falling back to CPU")

    # Parse CLI args and env variables
    logger.debug('Parsing command-line arguments')
    parser = argparse.ArgumentParser(description='Process continuous SDR stream.', 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output-dir', type=str, default='output')
    parser.add_argument('--band', choices=['low', 'high'], default='low')
    parser.add_argument('-lnboff', '--lnboff', action='store_true', help='Disable LNB handling (treat LNB as unpowered)')
    parser.add_argument('--quick-retrain', action='store_true', help='Run quick retrain and exit')
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--fast-train', action='store_true')
    parser.add_argument('--force-train', action='store_true')
    args = parser.parse_args()
    logger.debug(f'CLI args: {args}')

    env_train = os.getenv('TRAIN', '').lower() in ('1','true','yes','on')
    env_fast_train = os.getenv('FAST_TRAIN', '').lower() in ('1','true','yes','on')
    env_force_train = os.getenv('FORCE_TRAIN', '').lower() in ('1','true','yes','on')

    args.train = args.train or env_train
    args.fast_train = args.fast_train or env_fast_train
    args.force_train = args.force_train or env_force_train

    # Quick retrain on demand: run and exit before heavy initialization
    if args.quick_retrain:
        try:
            import quick_retrain
            quick_retrain.main()
            logger.info('Quick retrain completed; exiting as requested')
            sys.exit(0)
        except Exception as e:
            logger.error(f'Quick retrain failed: {e}', exc_info=True)
            sys.exit(1)

    if args.train:
        logger.info("🔄 Training enabled")
    if args.fast_train:
        logger.info("⚡ Fast training enabled")
    if args.force_train:
        logger.info("💥 Force training enabled")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.debug(f'Output directory ensured: {args.output_dir}')

    lnb_band = LNB_LOW if args.band == 'low' else LNB_HIGH
    logger.debug(f"LNB band selected: {lnb_band}")

    # Enable or disable LNB removal depending on CLI flag.
    global USE_LNB, LO
    if getattr(args, 'lnboff', False):
        USE_LNB = False
        LO = 0.0
        logger.info('LNB handling disabled via -lnboff; will not apply LNB removal')
    else:
        USE_LNB = True
        LO = lnb_band

    model_dir = os.path.join('models', 'full_state')
    model_path = os.path.join(model_dir, 'full_model.keras')
    bn_path = os.path.join(model_dir, 'bn_states.json')

    model = None
    logger.debug('Beginning model load/train decision')

    # Decide on training vs loading existing model
    if os.path.exists(model_path) and not (args.train or args.force_train):
        logger.info("Loading existing model...")
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if args.train or args.force_train:
                logger.info("Will train new model due to load failure")
                model = create_model()
                train_model(model, args.path)
            else:
                logger.error("No model available and training not requested, exiting")
                sys.exit(1)

    else:
        logger.info("Training new model from scratch...")
        model = create_model()
        train_model(model, args.path)

    # Final quick test prediction after training/loading
    try:
        dummy_input = np.random.rand(1, 8192, 1).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        logger.info(f"Model quick test prediction: {prediction[0][0]:.4f}")
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        sys.exit(1)

    # Run small calibration to estimate threshold and log raw outputs
    try:
        calib = calibrate_model_threshold(model)
        logger.info(f"Model calibration completed: recommended_threshold={calib['recommended_threshold']:.6f}")
    except Exception as e:
        logger.warning(f"Model calibration failed: {e}")

    if not check_model_for_nans(model):
        logger.error("Aborting due to NaNs/Inf in model weights.")
        sys.exit(1)

    # Dummy test prediction
    dummy_input = np.random.rand(1, 8192, 1).astype(np.float32)
    try:
        pred = model.predict(dummy_input, verbose=0)
        if np.isnan(pred).any() or np.isinf(pred).any():
            logger.error("Model prediction on dummy input produces NaN or Inf!")
            sys.exit(1)
        else:
            logger.info(f"Model dummy prediction OK: {pred[0][0]:.4f}")
    except Exception as e:
        logger.error(f"Model dummy prediction failed: {e}")
        sys.exit(1)
    # Start background detection job consumer to serialize detection IO/plotting
    try:
        logger.debug('Starting detection job consumer')
        # capture returned forwarder thread and mp process so we can shut them down
        try:
            forwarder_thread, mp_process = start_detection_job_consumer()
        except Exception:
            # Some implementations may return None or not return both handles
            res = start_detection_job_consumer()
            if isinstance(res, tuple) and len(res) >= 2:
                forwarder_thread, mp_process = res[0], res[1]
            else:
                forwarder_thread = None
                mp_process = None
        logger.debug('Detection job consumer started (or attempted)')
    except Exception as e:
        logger.warning(f"Could not start detection job consumer: {e}")
    # Disable startup tests by default to avoid child-process environment issues
    # (can be re-enabled by explicitly setting DISABLE_ALWAYS_RUN=0 in env)
    os.environ.setdefault('DISABLE_ALWAYS_RUN', '1')
    logger.info('Startup tasks disabled by default via DISABLE_ALWAYS_RUN')
    # Proceed to continuous SDR processing
    # Run startup tasks (unit tests + quick retrain) by default unless disabled
    def run_startup_tasks():
        try:
            if os.getenv('DISABLE_ALWAYS_RUN', '').lower() in ('1', 'true', 'yes'):
                logger.info('Startup tasks disabled via DISABLE_ALWAYS_RUN')
                return

            logger.info('Running startup tasks: unit tests and quick retrain')

            # Run unit tests (prefer pytest if available)
            try:
                # Ensure child processes can import local modules by setting PYTHONPATH
                child_env = os.environ.copy()
                child_env['PYTHONPATH'] = _proj_root + os.pathsep + child_env.get('PYTHONPATH', '')
                ret = subprocess.run(['pytest', '-q'], cwd=os.getcwd(), check=False, env=child_env)
                if ret.returncode != 0:
                    logger.warning('pytest returned non-zero exit; falling back to direct test run')
                    raise FileNotFoundError
            except Exception:
                # Fallback: run test file directly
                test_file = os.path.join('tests', 'test_processing.py')
                if os.path.exists(test_file):
                    logger.info(f'Running tests directly: {test_file}')
                    subprocess.run([sys.executable, test_file], check=False, env=child_env)
                else:
                    logger.warning('No tests found to run')

            # Quick retrain is not run automatically here; keep tests only.

        except Exception as e:
            logger.error(f'run_startup_tasks encountered an error: {e}')

    run_startup_tasks()
    logger.debug('Completed run_startup_tasks (if any)')
    try:
        logger.debug('Connecting to SDR source')
        sdr = connect_to_server()
        logger.debug('Connected to SDR; entering processing loop')
        logger.debug('About to call process_continuous_stream')
        process_continuous_stream(sdr, model, args.output_dir)
    except Exception as e:
        logger.error(f"SDR processing failed: {e}")
    finally:
        if 'sdr' in locals():
            sdr.close()
            logger.info("HackRF device closed")
        # Signal detection forwarder/worker to stop gracefully
        try:
            if DETECTION_JOB_QUEUE is not None:
                try:
                    DETECTION_JOB_QUEUE.put(None)
                    logger.debug('Signalled detection forwarder to shutdown')
                except Exception:
                    logger.exception('Failed to put shutdown sentinel into DETECTION_JOB_QUEUE')
        except Exception:
            pass

        try:
            if 'forwarder_thread' in locals() and forwarder_thread is not None:
                forwarder_thread.join(timeout=5)
                logger.debug('Forwarder thread join attempted')
        except Exception:
            logger.exception('Error while joining forwarder thread')

        try:
            if 'mp_process' in locals() and mp_process is not None:
                mp_process.join(timeout=5)
                logger.debug('MP worker process join attempted')
        except Exception:
            logger.exception('Error while joining mp worker process')
            
if __name__ == "__main__":
    from threading import Thread
    # Import the web module (do not import aic here to avoid circular imports)
    import web

    # Bind the Flask app and its shared state to this runtime's copies so that
    # the web UI observes updates performed inside this process.
    app = web.app
    web.latest_data = latest_data
    web.data_lock = data_lock

    logger.debug('__main__ entry: starting processing thread')
    processing_thread = Thread(target=main)
    processing_thread.start()
    logger.debug('Processing thread started; launching web app')
    try:
        app.run(host='0.0.0.0', port=5001)
    except Exception as e:
        logger.error(f'Web app failed to start: {e}', exc_info=True)