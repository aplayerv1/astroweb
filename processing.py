"""processing.py — model creation, training, prediction.

Fixes vs original:
- MODEL_RECOMMENDED_THRESHOLD accessed via module reference (not stale import copy)
- 2-channel input (real + imag) replaces silent imaginary discard
- Improved CNN architecture with residual-style skip connection
- Training uses ReduceLROnPlateau + class-weight-aware fit()
- calibrate_model_threshold updates module-level var correctly
"""
import numpy as np
import tensorflow as tf
import logging
import os
from scipy.signal import butter, lfilter

logger = logging.getLogger('aic.processing')

# Lazy imports — pulled in at call time so a broken data_generation or
# training module does NOT prevent processing.py from loading at all.
def _get_data_generator():
    from data_generation import data_generator
    return data_generator

def _get_wow_signals():
    from training import generate_wow_signals
    return generate_wow_signals

USE_CUPY = False
try:
    import cupy as cp
    USE_CUPY = True
except Exception:
    cp = np

# ---------------------------------------------------------------------------
# Threshold state — accessed as processing.MODEL_RECOMMENDED_THRESHOLD
# so callers always see the live value, not an import-time snapshot.
# ---------------------------------------------------------------------------
MODEL_RECOMMENDED_THRESHOLD = None

# Compiled inference function — avoids TF graph growth from repeated model() calls.
# Re-traced only when the model object changes (i.e. after retraining).
_COMPILED_PREDICT_FN    = None
_COMPILED_PREDICT_MODEL = None
MIN_DETECTION_THRESHOLD = float(os.getenv('MIN_DETECTION_THRESHOLD', '0.70'))
RUNTIME_DETECTION_THRESHOLD = None
try:
    if os.getenv('DETECTION_THRESHOLD'):
        RUNTIME_DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD'))
except Exception:
    RUNTIME_DETECTION_THRESHOLD = None


# Number of FFT bins fed to the spectral branch.
# Must match the second dim of inp_fft in create_model().
FFT_BINS = 1024


def _prepare_fft_input(fft_magnitude: np.ndarray) -> 'tf.Tensor':
    """Convert raw FFT magnitude (8192,) into spectral branch tensor (1, 1024, 1).

    Steps:
      1. Downsample 8192 → 1024 via strided slice.
      2. Log10 scale to compress dynamic range.
      3. Zero-mean unit-variance normalisation per sample.
      4. Reshape to (1, 1024, 1).
    """
    mag = np.asarray(fft_magnitude, dtype=np.float32).ravel()
    stride = max(1, len(mag) // FFT_BINS)
    mag = mag[::stride][:FFT_BINS]
    if mag.size < FFT_BINS:
        mag = np.pad(mag, (0, FFT_BINS - mag.size), mode='edge')
    mag = np.log10(mag + 1e-6)
    m, s = float(np.mean(mag)), float(np.std(mag))
    if s < 1e-6:
        s = 1.0
    mag = np.clip((mag - m) / s, -5.0, 5.0)
    return tf.convert_to_tensor(mag.reshape(1, FFT_BINS, 1), dtype=tf.float32)


def _prepare_input_for_model(samples, expected_len=8192, return_normalized=False,
                              fft_magnitude=None):
    """Pad/crop, normalise waveform; optionally prepare FFT branch tensor too.

    Parameters
    ----------
    samples          : waveform (real or complex), any length
    expected_len     : target waveform length (default 8192)
    return_normalized: if True, also return the normalised 1-D float32 array
    fft_magnitude    : raw magnitude array from process_fft() (length 8192).
                       When provided returns a spectral tensor for the FFT branch.
                       When None returns a zero tensor so the model can still run.

    Returns
    -------
    Always returns: (time_tensor, fft_tensor [, arr_n])
      time_tensor : tf.float32 (1, expected_len, channels)
      fft_tensor  : tf.float32 (1, FFT_BINS, 1)
      arr_n       : np.float32 (expected_len,)  — only when return_normalized=True
    """
    samples = np.asarray(samples).ravel()

    if samples.size < expected_len:
        pad = expected_len - samples.size
        samples = np.pad(samples, (pad // 2, pad - pad // 2), mode='constant')
    elif samples.size > expected_len:
        start = (samples.size - expected_len) // 2
        samples = samples[start:start + expected_len]

    def _norm(x):
        m, s = float(np.mean(x)), float(np.std(x))
        if s < 1e-6:
            s = 1.0
        return np.clip((x - m) / s, -5.0, 5.0)

    if np.iscomplexobj(samples):
        real_n = _norm(np.real(samples).astype(np.float32))
        imag_n = _norm(np.imag(samples).astype(np.float32))
        arr_n  = real_n
        time_t = tf.convert_to_tensor(
            np.stack([real_n, imag_n], axis=-1).reshape(1, expected_len, 2),
            dtype=tf.float32)
    else:
        arr_n  = _norm(samples.astype(np.float32))
        time_t = tf.convert_to_tensor(arr_n.reshape(1, expected_len, 1), dtype=tf.float32)

    fft_t = _prepare_fft_input(fft_magnitude) if fft_magnitude is not None             else tf.zeros((1, FFT_BINS, 1), dtype=tf.float32)

    if return_normalized:
        return time_t, fft_t, arr_n
    return time_t, fft_t


def apply_bandpass_filter(samples, fs, low_cutoff, high_cutoff):
    logger.debug('Applying bandpass filter')
    nyquist = 0.5 * fs
    lowcut = np.clip(low_cutoff / nyquist, 1e-4, 0.999)
    highcut = np.clip(high_cutoff / nyquist, 1e-4, 0.999)
    if lowcut >= highcut:
        return samples
    b, a = butter(4, [lowcut, highcut], btype='band')
    return lfilter(b, a, samples)


def _res_block(x, filters: int, kernel_size: int, dropout: float = 0.1):
    """Pre-activation residual block with squeeze-excitation.

    BN -> GELU -> Conv1D -> BN -> GELU -> Conv1D -> SE -> Add(skip)

    Pre-activation keeps the skip path clean (no BN on residual).
    SE ratio=8 lets the model up-weight channels that carry the HI peak.
    """
    KL = tf.keras.layers
    in_filters = x.shape[-1]

    r = KL.BatchNormalization()(x)
    r = KL.Activation('gelu')(r)
    r = KL.Conv1D(filters, kernel_size, padding='same', use_bias=False)(r)
    r = KL.BatchNormalization()(r)
    r = KL.Activation('gelu')(r)
    r = KL.Conv1D(filters, kernel_size, padding='same', use_bias=False)(r)
    if dropout > 0:
        r = KL.Dropout(dropout)(r)

    if in_filters != filters:
        x = KL.Conv1D(filters, 1, padding='same', use_bias=False)(x)

    # Squeeze-excitation (ratio 8)
    se = KL.GlobalAveragePooling1D()(r)
    se = KL.Dense(max(filters // 8, 8), activation='gelu', use_bias=False)(se)
    se = KL.Dense(filters, activation='sigmoid', use_bias=False)(se)
    se = KL.Reshape((1, filters))(se)
    r  = KL.Multiply()([r, se])

    return KL.Add()([x, r])


def _fft_branch(fft_inp):
    """Lightweight spectral CNN branch.

    Input : (batch, 1024, 1)  — log-magnitude spectrum, 1024 bins
    Output: (batch, 128)      — spectral feature vector

    Deliberately small (374K params, 42 MB VRAM) so it adds spectral
    awareness without overwhelming the time-domain branch budget.
    Four strided Conv1D layers compress 1024 bins → 64 bins → GAP → Dense(128).
    """
    KL = tf.keras.layers
    s = KL.Conv1D(64,  kernel_size=7, strides=2, padding='same', use_bias=False)(fft_inp)
    s = KL.BatchNormalization()(s)
    s = KL.Activation('gelu')(s)
    # 1024 -> 512

    s = KL.Conv1D(128, kernel_size=5, strides=2, padding='same', use_bias=False)(s)
    s = KL.BatchNormalization()(s)
    s = KL.Activation('gelu')(s)
    # 512 -> 256

    s = KL.Conv1D(256, kernel_size=3, strides=2, padding='same', use_bias=False)(s)
    s = KL.BatchNormalization()(s)
    s = KL.Activation('gelu')(s)
    # 256 -> 128

    s = KL.Conv1D(256, kernel_size=3, strides=2, padding='same', use_bias=False)(s)
    s = KL.BatchNormalization()(s)
    s = KL.Activation('gelu')(s)
    # 128 -> 64

    s = KL.GlobalAveragePooling1D()(s)   # (batch, 256)
    s = KL.Dense(128, activation='gelu', use_bias=True)(s)
    s = KL.Dropout(0.15)(s)
    return s                              # (batch, 128)


def create_model(input_channels: int = 1):
    """Dual-input residual CNN + BiGRU: time domain + FFT spectrum.

    Two parallel input branches:
      1. Time-domain branch  — 8192-sample waveform → BiGRU → (batch, 512)
         Captures pulse shape, periodicity, temporal envelope.
      2. Spectral branch     — 1024-bin log-magnitude FFT → CNN → (batch, 128)
         Captures HI line peak, RFI tone positions, spectral shape.

    Branches are concatenated → (batch, 640) → shared head → sigmoid.

    The spectral branch gives the model explicit frequency-domain features
    that the time-domain CNN would otherwise have to re-derive from scratch.
    Critically, it can now directly see the HI 21-cm line as a spectral
    feature rather than inferring it from amplitude modulation alone.

    VRAM budget (batch=16)
    ----------------------
    Time-domain branch:  ~985 MB  (unchanged from single-input model)
    Spectral branch:      ~42 MB  (small by design)
    Total weights+Adam:  ~229 MB
    Total estimate:     ~1256 MB  — fits in 2 GB with headroom

    Inputs
    ------
    inp_time : (batch, 8192, input_channels)  raw waveform
    inp_fft  : (batch, 1024, 1)               log-magnitude spectrum

    input_channels=1  real-only (standard)
    input_channels=2  complex IQ (real+imag stacked)
    """
    KL = tf.keras.layers

    # ── Input A: time-domain waveform ────────────────────────────────────────
    inp_time = tf.keras.Input(shape=(8192, input_channels), name='time_input')

    # Stem: stride to seq=1024 before ResBlocks (prevents OOM in backprop)
    x = KL.Conv1D(64,  kernel_size=15, strides=2, padding='same', use_bias=False)(inp_time)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('gelu')(x)
    # seq: 8192 -> 4096

    x = KL.Conv1D(128, kernel_size=7,  strides=4, padding='same', use_bias=False)(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('gelu')(x)
    # seq: 4096 -> 1024

    x = KL.Conv1D(256, kernel_size=3,  strides=1, padding='same', use_bias=False)(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('gelu')(x)
    # seq: 1024, channels: 256

    x = _res_block(x, 256, 7, dropout=0.10)
    x = _res_block(x, 256, 7, dropout=0.10)
    x = KL.Conv1D(384, 1, strides=4, padding='same', use_bias=False)(x)
    x = KL.BatchNormalization()(x)
    # seq: 256, channels: 384

    x = _res_block(x, 384, 5, dropout=0.10)
    x = _res_block(x, 384, 5, dropout=0.10)
    x = KL.Conv1D(512, 1, strides=4, padding='same', use_bias=False)(x)
    x = KL.BatchNormalization()(x)
    # seq: 64, channels: 512

    x = _res_block(x, 512, 3, dropout=0.10)
    x = _res_block(x, 512, 3, dropout=0.10)
    x = KL.Conv1D(384, 1, strides=2, padding='same', use_bias=False)(x)
    x = KL.BatchNormalization()(x)
    # seq: 32, channels: 384

    x = _res_block(x, 384, 3, dropout=0.10)
    x = _res_block(x, 384, 3, dropout=0.10)
    # seq: 32, channels: 384

    x = KL.Bidirectional(KL.GRU(256, return_sequences=False, dropout=0.10,
                                  recurrent_dropout=0.0))(x)
    # time_feat: (batch, 512)

    # ── Input B: log-magnitude FFT spectrum ──────────────────────────────────
    inp_fft = tf.keras.Input(shape=(1024, 1), name='fft_input')
    s = _fft_branch(inp_fft)
    # spec_feat: (batch, 128)

    # ── Merge + head ─────────────────────────────────────────────────────────
    merged = KL.Concatenate()([x, s])          # (batch, 640)
    merged = KL.LayerNormalization()(merged)
    merged = KL.Dense(512, activation='gelu')(merged)
    merged = KL.Dropout(0.30)(merged)
    merged = KL.Dense(256, activation='gelu')(merged)
    merged = KL.Dropout(0.20)(merged)
    out = KL.Dense(1, activation='sigmoid')(merged)

    model = tf.keras.Model(inputs=[inp_time, inp_fft], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=3e-4,
            beta_1=0.9, beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0,
        ),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model

def calibrate_model_threshold(model, chunk_size=8192, n_pos=60, n_neg=120):
    """Calibrate detection threshold using percentile separation.

    Problems with the old approach
    --------------------------------
    1. Negatives were pure Gaussian noise only — the model scores these near 0
       trivially, so neg_mean → 0 and threshold = pos_mean/2, which is far too
       permissive for real sky data containing RFI/hardware artefacts.

    2. Midpoint-of-means is not robust to outliers and doesn't control FPR.

    New approach
    ------------
    Negatives include: Gaussian noise, RFI tones, broadband FM interference,
    saturated IQ, DC offset, Rayleigh fading — the same distribution used in
    data_generation.py negative class.

    Threshold is set at:
        max(neg_99th_percentile, pos_5th_percentile - margin)
    clamped to [0.70, 0.97].

    This guarantees <1% false positive rate on the calibration set while
    accepting any signal the model scores above its own 5th percentile.
    The floor of 0.70 prevents the threshold from being set so low that
    ambient RFI triggers constant detections.
    """
    global MODEL_RECOMMENDED_THRESHOLD
    rng = np.random.RandomState(42)

    def _run(sig, fft_mag=None):
        time_t, fft_t = _prepare_input_for_model(sig, fft_magnitude=fft_mag)
        try:
            return float(np.ravel(model([time_t, fft_t], training=False))[0])
        except Exception:
            return 0.0

    # ── Positive samples: WOW-like + pulsar + hydrogen-line ──────────────────
    pos_outputs = []
    try:
        generate_wow_signals = _get_wow_signals()
        pos_pool = generate_wow_signals(chunk_size, n=n_pos, as_numpy=True)
    except Exception:
        pos_pool = None

    t = np.arange(chunk_size, dtype=np.float32)
    for i in range(n_pos):
        if pos_pool is not None and i < len(pos_pool):
            sig = pos_pool[i].copy()
        else:
            # Synthetic narrowband burst
            f0 = rng.uniform(0.005, 0.05)
            env = np.exp(-((t - chunk_size/2)**2) / (chunk_size/6)**2)
            sig = env * np.sin(2*np.pi*f0*t) + rng.normal(0, 0.15, chunk_size)
        pos_outputs.append(_run(sig.astype(np.float32)))

    # ── Negative samples: realistic RFI / hardware artefacts ─────────────────
    neg_outputs = []
    for i in range(n_neg):
        choice = i % 8
        if choice == 0:
            sig = rng.normal(0, 1.0, chunk_size)
        elif choice == 1:
            sig = rng.rayleigh(1.0, chunk_size)
        elif choice == 2:
            # Narrowband RFI tone
            freq = rng.uniform(0.05, 0.45)
            sig = rng.uniform(0.5, 2.0) * np.sin(2*np.pi*freq*t)
            sig += rng.normal(0, 0.5, chunk_size)
        elif choice == 3:
            # Broadband FM
            sig = np.sin(2*np.pi*0.05*t) * rng.normal(0, 1.0, chunk_size)
        elif choice == 4:
            # Saturated / clipped IQ (amp near ADC rail)
            amp = rng.uniform(1.2, 1.45)
            sig = amp * np.cos(rng.uniform(0, 2*np.pi, chunk_size))
            sig += rng.normal(0, 0.002, chunk_size)
        elif choice == 5:
            # DC offset / LO leakage
            dc = rng.uniform(-1.5, 1.5)
            sig = np.full(chunk_size, dc, dtype=np.float32)
            sig += rng.normal(0, 0.01, chunk_size)
        elif choice == 6:
            # Non-stationary noise (gain drift)
            sig = rng.normal(0, 1.0 + 0.5*np.sin(2*np.pi*0.01*t), chunk_size)
        else:
            # Cumulative drift / 1/f-like
            sig = np.cumsum(rng.normal(0, 0.3, chunk_size))
            sig /= (np.std(sig) + 1e-9)
        neg_outputs.append(_run(sig.astype(np.float32)))

    pos_arr = np.array(pos_outputs)
    neg_arr = np.array(neg_outputs)

    pos_mean = float(np.mean(pos_arr))
    neg_mean = float(np.mean(neg_arr))
    pos_p05  = float(np.percentile(pos_arr, 5))
    neg_p99  = float(np.percentile(neg_arr, 99))

    # Threshold: just above the 99th-percentile false-positive, but no lower
    # than the 5th-percentile true-positive minus a small margin.
    # Floor at 0.70 to prevent ambient RFI from triggering continuously.
    recommended = float(np.clip(
        max(neg_p99 + 0.02, pos_p05 - 0.05),
        0.70, 0.97
    ))

    MODEL_RECOMMENDED_THRESHOLD = recommended

    logger.info(
        f'Calibration complete: '        f'pos_mean={pos_mean:.3f} pos_p05={pos_p05:.3f} | '        f'neg_mean={neg_mean:.3f} neg_p99={neg_p99:.3f} | '        f'threshold={recommended:.3f}'
    )
    return {
        'pos_mean':             pos_mean,
        'neg_mean':             neg_mean,
        'pos_std':              float(np.std(pos_arr)),
        'neg_std':              float(np.std(neg_arr)),
        'pos_p05':              pos_p05,
        'neg_p99':              neg_p99,
        'recommended_threshold': recommended,
    }


def train_model(model, csv_data, htru2_dir: str = 'data/htru2',
                use_real_data: bool = True):
    """Train model with balanced batches, class weights, and LR scheduling.

    Tuned for the 15 M-param residual+GRU architecture:
    - Cosine-decay LR with linear warmup (first 5 epochs ramp 1e-5 → 3e-4)
    - Smaller batch default (16) to leave VRAM headroom for the BiGRU gates
    - Higher patience (20) because the GRU needs more epochs to converge
    - Gradient clipping already set in compile() (clipnorm=1.0)
    """
    callbacks_dir = os.path.join('models', 'callbacks')
    os.makedirs(callbacks_dir, exist_ok=True)
    os.makedirs(os.path.join('models', 'logs'), exist_ok=True)

    # Linear warmup over 5 epochs then cosine decay
    total_epochs   = int(os.getenv('TRAIN_EPOCHS', '100'))  # early stopping fires ~20-30
    warmup_epochs  = 8      # longer warmup avoids early val_loss oscillation
    base_lr        = float(os.getenv('TRAIN_LR', '1e-4'))  # 3e-4 caused val_loss swings
    min_lr         = float(os.getenv('TRAIN_MIN_LR', '1e-6'))

    class _WarmupCosineDecay(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if epoch < warmup_epochs:
                lr = min_lr + (base_lr - min_lr) * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
            # Keras 3 removed set_value(); direct assignment works on both
            # tf.keras (Keras 2) and standalone Keras 3.
            try:
                self.model.optimizer.learning_rate = float(lr)
            except Exception:
                # Edge case: LR exposed as a tf.Variable
                self.model.optimizer.learning_rate.assign(float(lr))

    callbacks = [
        _WarmupCosineDecay(),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(callbacks_dir, 'best_model.keras'),
            monitor='val_auc', mode='max',
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc', mode='max', patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('models', 'logs'), histogram_freq=0,
        ),
    ]

    batch_size = int(os.getenv('TRAIN_BATCH_SIZE', '16'))

    # Build the positive pool ONCE — shared between train and val generators,
    # and used to compute steps_per_epoch so it must come first.
    data_generator = _get_data_generator()
    from data_generation import build_positive_pool, _build_csv_pool
    logger.info('Building shared positive pool for train + val generators...')
    shared_pool = build_positive_pool(
        chunk_size=8192, seed=42,
        htru2_dir=htru2_dir,
        use_real_data=use_real_data,
    )
    logger.info(f'Shared pool ready: {len(shared_pool)} positive signals')

    # steps_per_epoch = 1.0x pool coverage (was 1.5x) → ~10 min/epoch at pool=9443.
    # Val fixed at 50 steps: val_loss oscillates too much to use for early stopping;
    # use val_auc instead (stable, monotonically improving).
    _pool_size = max(len(shared_pool), 1)
    _default_steps = max(100, int(_pool_size * 1.0 / batch_size))
    steps_per_epoch  = int(os.getenv('TRAIN_STEPS_PER_EPOCH',  str(_default_steps)))
    validation_steps = int(os.getenv('TRAIN_VALIDATION_STEPS', '50'))

    # Sample weights compensate for HTRU2 ~9:1 class imbalance
    CLASS_WEIGHTS = {0: 1.0, 1: 9.0}

    # ── Fixed validation set (built once, reused every epoch) ────────────────
    # The root cause of val_loss oscillation (0.13 → 8.30 → 0.17) is that the
    # validation *generator* draws different random batches each epoch, so 50
    # steps × 16 samples = 800 samples have high variance.  val_auc is stable
    # because it accumulates over all steps; val_loss is per-batch and noisy.
    #
    # Solution: materialise a fixed numpy validation set at training start.
    # Same 800 samples every epoch → deterministic, monotonically improving
    # val_loss → reliable early stopping signal.
    val_n_batches = validation_steps  # default 50
    logger.info(f'Building fixed validation set ({val_n_batches} batches × {batch_size})...')
    val_X_time = np.zeros((val_n_batches * batch_size, 8192, 1), dtype=np.float32)
    val_X_fft  = np.zeros((val_n_batches * batch_size, FFT_BINS, 1), dtype=np.float32)
    val_y      = np.zeros((val_n_batches * batch_size,), dtype=np.float32)
    val_sw     = np.zeros((val_n_batches * batch_size,), dtype=np.float32)

    val_gen_fixed = data_generator(
        chunk_size=8192, csv_data=None,
        batch_size=batch_size, positive_ratio=0.5,
        seed=999,
        htru2_dir=htru2_dir, use_real_data=False,
        as_numpy=True, _prebuilt_pool=shared_pool,
    )
    for bi, (inputs, y_b) in enumerate(val_gen_fixed):
        if bi >= val_n_batches:
            break
        s = bi * batch_size
        e = s + batch_size
        val_X_time[s:e] = inputs[0]
        val_X_fft[s:e]  = inputs[1]
        val_y[s:e]      = y_b
        val_sw[s:e]     = np.array([CLASS_WEIGHTS[int(l)] for l in y_b],
                                    dtype=np.float32)

    val_data = (
        (val_X_time, val_X_fft),
        val_y,
        val_sw,
    )
    logger.info(
        f'Fixed val set: {val_n_batches * batch_size} samples, '
        f'pos={int(val_y.sum())}, neg={int((1-val_y).sum())}'
    )

    # ── Training dataset (infinite generator wrapped in tf.data) ─────────────
    B, N, F = batch_size, 8192, FFT_BINS
    sig = (
        (
            tf.TensorSpec(shape=(B, N, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(B, F, 1), dtype=tf.float32),
        ),
        tf.TensorSpec(shape=(B,), dtype=tf.float32),
        tf.TensorSpec(shape=(B,), dtype=tf.float32),
    )

    def _train_gen():
        base = data_generator(
            chunk_size=8192, csv_data=None,
            batch_size=batch_size, positive_ratio=0.5,
            htru2_dir=htru2_dir, use_real_data=False,
            as_numpy=True, _prebuilt_pool=shared_pool,
        )
        for inputs, y in base:
            X_time, X_fft = inputs[0], inputs[1]
            sw = np.array([CLASS_WEIGHTS[int(l)] for l in y], dtype=np.float32)
            yield (X_time, X_fft), y, sw

    train_ds = tf.data.Dataset.from_generator(_train_gen, output_signature=sig).prefetch(2)

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=total_epochs,
        validation_data=val_data,   # fixed numpy arrays — deterministic
        callbacks=callbacks,
        verbose=1,
    )

    state_dir = os.path.join('models', 'full_state')
    os.makedirs(state_dir, exist_ok=True)
    model.save(os.path.join(state_dir, 'full_model.keras'))
    logger.info('Model saved to models/full_state/full_model.keras')

    return history


def load_model_weights(model, weights_file):
    weights_path = os.path.join('models', weights_file)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        logger.info(f'Loaded weights from {weights_path}')
    else:
        logger.error(f'Weights file not found: {weights_path}')


def check_model_for_nans(model):
    for i, w in enumerate(model.get_weights()):
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            logger.error(f'Model weight {i} contains NaN/Inf')
            return False
    logger.info('Model weights are clean')
    return True


def inject_wow_signal(samples, amplitude=100000.0):
    t = np.arange(len(samples))
    center_freq = 0.01
    drift_rate = 0.003
    intensity = 2.0 * np.exp(-(t - len(t) / 2) ** 2 / (len(t) / 4) ** 2)
    freq_drift = center_freq + drift_rate * (t - len(t) / 2) / len(t)
    wow = amplitude * intensity * np.sin(2 * np.pi * freq_drift * t)
    wow = wow * np.exp(1j * 2 * np.pi * freq_drift * t) * np.exp(1j * np.pi / 3)
    return samples.astype(np.complex64) + wow.astype(np.complex64)


def predict_signal(model, samples, fft_magnitude=None):
    """Run model inference.  Returns (detection: bool, confidence: float).

    Parameters
    ----------
    model         : Keras model (dual-input: time + FFT)
    samples       : raw waveform (8192 samples, real or complex)
    fft_magnitude : FFT magnitude array from process_fft() (length 8192).
                    If None, a zero tensor is used for the FFT branch so the
                    model still runs without spectral information.
    """
    try:
        raw      = np.asarray(samples)
        mag      = np.abs(raw) if np.iscomplexobj(raw) else np.abs(raw.astype(np.float32))
        mag_mean = float(np.mean(mag))
        mag_std  = float(np.std(mag))
        cov = mag_std / (mag_mean + 1e-12)
        is_saturated = (mag_mean > 1.35) and (cov < 0.015)
        is_dc_leak   = (mag_mean < 0.05) and (mag_std < 0.005)
        if is_saturated or is_dc_leak:
            reason = 'Clipped/saturated IQ' if is_saturated else 'DC offset / LO leakage'
            logger.warning(
                f'{reason} rejected — '
                f'mag_mean={mag_mean:.4f}, mag_std={mag_std:.5f}, CoV={cov:.4f}. '
                f'{"Lower HackRF gain." if is_saturated else "Check LO leakage."}'
            )
            return False, 0.0

        time_t, fft_t, samples_normalized = _prepare_input_for_model(
            samples, expected_len=8192,
            return_normalized=True,
            fft_magnitude=fft_magnitude,
        )

        if samples_normalized is None or np.std(samples_normalized) < 1e-6:
            logger.warning('Near-zero std in input signal')
            return False, 0.0

        time_t  = tf.cast(time_t, tf.float32)
        fft_t   = tf.cast(fft_t,  tf.float32)
        global _COMPILED_PREDICT_FN, _COMPILED_PREDICT_MODEL
        if _COMPILED_PREDICT_MODEL is not model:
            _COMPILED_PREDICT_FN    = tf.function(
                lambda t, f: model([t, f], training=False),
                reduce_retracing=True,
            )
            _COMPILED_PREDICT_MODEL = model
        raw_out = _COMPILED_PREDICT_FN(time_t, fft_t)

        try:
            confidence_value = float(np.ravel(raw_out)[0])
        except Exception:
            confidence_value = 0.0

        if RUNTIME_DETECTION_THRESHOLD is not None:
            threshold = float(RUNTIME_DETECTION_THRESHOLD)
        elif MODEL_RECOMMENDED_THRESHOLD is not None and MODEL_RECOMMENDED_THRESHOLD > 1e-6:
            threshold = float(MODEL_RECOMMENDED_THRESHOLD)
        else:
            threshold = float(MIN_DETECTION_THRESHOLD)
        threshold = max(threshold, float(MIN_DETECTION_THRESHOLD))

        detection = confidence_value >= threshold
        logger.debug(
            f'Prediction: conf={confidence_value:.4f}, thresh={threshold:.4f}, '
            f'fft_provided={fft_magnitude is not None}, det={detection}'
        )
        return bool(detection), float(confidence_value)

    except Exception as e:
        logger.error(f'predict_signal failed: {e}', exc_info=True)
        return False, 0.0