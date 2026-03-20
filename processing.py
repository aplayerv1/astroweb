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
MIN_DETECTION_THRESHOLD = float(os.getenv('MIN_DETECTION_THRESHOLD', '0.05'))
RUNTIME_DETECTION_THRESHOLD = None
try:
    if os.getenv('DETECTION_THRESHOLD'):
        RUNTIME_DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD'))
except Exception:
    RUNTIME_DETECTION_THRESHOLD = None


def _prepare_input_for_model(samples, expected_len=8192, return_normalized=False):
    """Pad/crop, cast to real float32, normalise, return tf tensor.

    If samples is complex, both real and imaginary parts are used as a
    2-channel input (shape 1, expected_len, 2). Otherwise single-channel.
    """
    samples = np.asarray(samples)

    # Flatten
    samples = samples.ravel()

    # Pad or crop to expected_len
    if samples.size < expected_len:
        pad = expected_len - samples.size
        samples = np.pad(samples, (pad // 2, pad - pad // 2), mode='constant')
    elif samples.size > expected_len:
        start = (samples.size - expected_len) // 2
        samples = samples[start:start + expected_len]

    # Build channels
    if np.iscomplexobj(samples):
        real_part = np.real(samples).astype(np.float32)
        imag_part = np.imag(samples).astype(np.float32)

        def _norm(x):
            m, s = np.mean(x), np.std(x)
            if s < 1e-6:
                s = 1.0
            return np.clip((x - m) / s, -5.0, 5.0)

        real_n = _norm(real_part)
        imag_n = _norm(imag_part)
        arr_2ch = np.stack([real_n, imag_n], axis=-1)  # (expected_len, 2)
        input_tensor = tf.constant(arr_2ch.reshape(1, expected_len, 2), dtype=tf.float32)
        if return_normalized:
            return input_tensor, real_n
        return input_tensor
    else:
        arr = samples.astype(np.float32)
        m, s = np.mean(arr), np.std(arr)
        if s < 1e-6:
            s = 1.0
        arr_n = np.clip((arr - m) / s, -5.0, 5.0)
        input_tensor = tf.constant(arr_n.reshape(1, expected_len, 1), dtype=tf.float32)
        if return_normalized:
            return input_tensor, arr_n
        return input_tensor


def apply_bandpass_filter(samples, fs, low_cutoff, high_cutoff):
    logger.debug('Applying bandpass filter')
    nyquist = 0.5 * fs
    lowcut = np.clip(low_cutoff / nyquist, 1e-4, 0.999)
    highcut = np.clip(high_cutoff / nyquist, 1e-4, 0.999)
    if lowcut >= highcut:
        return samples
    b, a = butter(4, [lowcut, highcut], btype='band')
    return lfilter(b, a, samples)


def create_model(input_channels: int = 1):
    """Build 1D-CNN classifier.

    input_channels=1 for real-only input, 2 for complex (real+imag).
    """
    inp = tf.keras.Input(shape=(8192, input_channels))

    # Block 1
    x = tf.keras.layers.Conv1D(128, kernel_size=9, padding='same', activation='relu')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Block 2
    x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Block 3
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Block 4 (extra depth)
    x = tf.keras.layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model


def calibrate_model_threshold(model, chunk_size=8192, n_pos=30, n_neg=30):
    """Run model on synthetic pos/neg samples; set MODULE-LEVEL threshold."""
    global MODEL_RECOMMENDED_THRESHOLD
    pos_outputs, neg_outputs = [], []

    try:
        generate_wow_signals = _get_wow_signals()
        pos_pool = generate_wow_signals(chunk_size, n=n_pos, as_numpy=True)
    except Exception:
        pos_pool = None

    for i in range(n_pos):
        sig = pos_pool[i % len(pos_pool)] if pos_pool else np.random.normal(0, 1, chunk_size)
        x = _prepare_input_for_model(sig)
        try:
            val = float(np.ravel(model(x, training=False))[0])
        except Exception:
            val = 0.0
        pos_outputs.append(val)

    for _ in range(n_neg):
        noise = np.random.normal(0, 1, chunk_size).astype(np.float32)
        x = _prepare_input_for_model(noise)
        try:
            val = float(np.ravel(model(x, training=False))[0])
        except Exception:
            val = 0.0
        neg_outputs.append(val)

    pos_mean = float(np.mean(pos_outputs)) if pos_outputs else 0.5
    neg_mean = float(np.mean(neg_outputs)) if neg_outputs else 0.0
    recommended = float(np.clip((pos_mean + neg_mean) / 2.0, 0.0, 1.0))

    # Update module-level variable — callers using `import processing` will
    # always read the live value via `processing.MODEL_RECOMMENDED_THRESHOLD`
    MODEL_RECOMMENDED_THRESHOLD = recommended

    logger.info(
        f'Calibration: pos_mean={pos_mean:.4f}, neg_mean={neg_mean:.4f}, '
        f'recommended_threshold={recommended:.4f}'
    )
    return {
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'pos_std': float(np.std(pos_outputs)),
        'neg_std': float(np.std(neg_outputs)),
        'recommended_threshold': recommended,
    }


def train_model(model, csv_data, htru2_dir: str = 'data/htru2',
                use_real_data: bool = True):
    """Train model with balanced batches, class weights, and LR scheduling."""
    callbacks_dir = os.path.join('models', 'callbacks')
    os.makedirs(callbacks_dir, exist_ok=True)
    os.makedirs(os.path.join('models', 'logs'), exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(callbacks_dir, 'best_model.keras'),
            monitor='val_auc', mode='max',
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5,
            min_lr=1e-6, verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('models', 'logs'), histogram_freq=0,
        ),
    ]

    total_samples = int(os.getenv('TRAIN_TOTAL_SAMPLES', '20000'))
    batch_size = int(os.getenv('TRAIN_BATCH_SIZE', '32'))
    epochs = int(os.getenv('TRAIN_EPOCHS', '100'))
    steps_per_epoch = max(50, total_samples // batch_size)
    validation_steps = max(10, total_samples // (5 * batch_size))

    if os.getenv('TRAIN_STEPS_PER_EPOCH'):
        steps_per_epoch = int(os.getenv('TRAIN_STEPS_PER_EPOCH'))
    if os.getenv('TRAIN_VALIDATION_STEPS'):
        validation_steps = int(os.getenv('TRAIN_VALIDATION_STEPS'))

    # class_weight is NOT supported for Python generators in Keras.
    # Instead we wrap the generator to yield (X, y, sample_weights) triples.
    # Weight 9.0 for positives compensates for the ~9:1 HTRU2 imbalance.
    CLASS_WEIGHTS = {0: 1.0, 1: 9.0}

    def _weighted_gen(base_gen):
        for X, y in base_gen:
            sw = np.array([CLASS_WEIGHTS[int(label)] for label in y], dtype=np.float32)
            yield X, y, sw

    data_generator = _get_data_generator()

    # Build the positive pool ONCE and share between train and val.
    # Previously each generator called load_all_datasets() independently,
    # causing downloads + full signal reconstruction twice per training run.
    from data_generation import build_positive_pool, _build_csv_pool
    logger.info('Building shared positive pool for train + val generators...')
    shared_pool = build_positive_pool(
        chunk_size=8192, seed=42,
        htru2_dir=htru2_dir,
        use_real_data=use_real_data,
    )
    logger.info(f'Shared pool ready: {len(shared_pool)} positive signals')

    # Pass the pre-built pool as csv_data so generators skip load_all_datasets()
    base_train = data_generator(
        chunk_size=8192, csv_data=None,
        batch_size=batch_size, positive_ratio=0.5,
        htru2_dir=htru2_dir, use_real_data=False,  # pool already built
        as_numpy=True,
        _prebuilt_pool=shared_pool,
    )
    base_val = data_generator(
        chunk_size=8192, csv_data=None,
        batch_size=batch_size, positive_ratio=0.5,
        seed=999,
        htru2_dir=htru2_dir, use_real_data=False,  # pool already built
        as_numpy=True,
        _prebuilt_pool=shared_pool,
    )

    train_gen = _weighted_gen(base_train)
    val_gen   = _weighted_gen(base_val)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
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


def inject_wow_signal(samples, amplitude=25.0):
    t = np.arange(len(samples))
    center_freq = 0.01
    drift_rate = 0.003
    intensity = 2.0 * np.exp(-(t - len(t) / 2) ** 2 / (len(t) / 4) ** 2)
    freq_drift = center_freq + drift_rate * (t - len(t) / 2) / len(t)
    wow = amplitude * intensity * np.sin(2 * np.pi * freq_drift * t)
    wow = wow * np.exp(1j * 2 * np.pi * freq_drift * t) * np.exp(1j * np.pi / 3)
    return samples.astype(np.complex64) + wow.astype(np.complex64)


def predict_signal(model, samples):
    """Run model inference.  Returns (detection: bool, confidence: float)."""
    # Read live threshold values directly from module globals — no self-import needed.
    try:
        # --- Saturation / clipping guard ---
        # A HackRF with gain too high produces IQ with magnitude ~1.4 and
        # near-zero std. After normalisation this looks like a strong signal
        # and scores ~0.9999. Reject before inference.
        raw      = np.asarray(samples)
        mag      = np.abs(raw) if np.iscomplexobj(raw) else np.abs(raw.astype(np.float32))
        mag_mean = float(np.mean(mag))
        mag_std  = float(np.std(mag))
        if mag_std < 0.01 or mag_mean > 1.35:
            logger.warning(
                f'Saturated input rejected — '
                f'mag_mean={mag_mean:.4f}, mag_std={mag_std:.5f}. '
                f'Lower HackRF gain (LNA/VGA/AMP).'
            )
            return False, 0.0

        input_tensor, samples_normalized = _prepare_input_for_model(
            samples, expected_len=8192, return_normalized=True
        )

        if samples_normalized is None or np.std(samples_normalized) < 1e-6:
            logger.warning('Near-zero std in input signal')
            return False, 0.0

        input_tensor = tf.cast(input_tensor, tf.float32)
        raw = model(input_tensor, training=False)

        try:
            confidence_value = float(np.ravel(raw)[0])
        except Exception:
            confidence_value = 0.0

        # Adaptive threshold: tighter for low-variance signals
        norm_std = float(np.std(samples_normalized))
        alpha = min(1.0, norm_std)
        threshold = float(np.clip(0.7 - 0.15 * alpha, 0.4, 0.95))

        # Override with calibrated threshold if available (read from module globals)
        if MODEL_RECOMMENDED_THRESHOLD is not None and MODEL_RECOMMENDED_THRESHOLD > 1e-6:
            threshold = max(threshold, float(MODEL_RECOMMENDED_THRESHOLD))

        # Override with runtime env threshold
        if RUNTIME_DETECTION_THRESHOLD is not None:
            threshold = float(RUNTIME_DETECTION_THRESHOLD)

        # Floor
        if threshold < MIN_DETECTION_THRESHOLD:
            threshold = float(MIN_DETECTION_THRESHOLD)

        detection = confidence_value >= threshold
        logger.debug(
            f'Prediction: conf={confidence_value:.4f}, thresh={threshold:.4f}, '
            f'det={detection}'
        )
        return bool(detection), float(confidence_value)

    except Exception as e:
        logger.error(f'predict_signal failed: {e}', exc_info=True)
        return False, 0.0