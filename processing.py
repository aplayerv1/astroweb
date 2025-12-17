import numpy as np
import tensorflow as tf
import logging
from scipy.signal import butter, lfilter
from training import generate_wow_signals
from data_generation import data_generator
import os

logger = logging.getLogger('aic.processing')

USE_CUPY = False
try:
    import cupy as cp
    USE_CUPY = True
except Exception:
    cp = np

MODEL_RECOMMENDED_THRESHOLD = None
MIN_DETECTION_THRESHOLD = float(os.getenv('MIN_DETECTION_THRESHOLD', '0.05'))
RUNTIME_DETECTION_THRESHOLD = None
try:
    if os.getenv('DETECTION_THRESHOLD'):
        RUNTIME_DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD'))
except Exception:
    RUNTIME_DETECTION_THRESHOLD = None


def _prepare_input_for_model(samples, expected_len=8192, return_normalized=False):
    samples = np.asarray(samples)

    if samples.size != expected_len:
        try:
            if samples.size < expected_len:
                pad = expected_len - samples.size
                left = pad // 2
                right = pad - left
                samples = np.concatenate([np.zeros(left, dtype=samples.dtype), samples.ravel(), np.zeros(right, dtype=samples.dtype)])
            else:
                start = (samples.size - expected_len) // 2
                samples = samples.ravel()[start:start+expected_len]
        except Exception:
            samples = np.resize(samples, expected_len)

    if np.iscomplexobj(samples):
        samples = np.real(samples)

    mean = np.mean(samples)
    std = np.std(samples)
    if std < 1e-6:
        std = 1.0
    samples_normalized = (samples - mean) / std
    samples_normalized = np.clip(samples_normalized, -5.0, 5.0)
    input_tensor = tf.reshape(samples_normalized.astype(np.float32), (1, expected_len, 1))
    if return_normalized:
        return input_tensor, samples_normalized
    return input_tensor


def apply_bandpass_filter(samples, fs, low_cutoff, high_cutoff):
   logger.debug("Applying bandpass filter")
   nyquist = 0.5 * fs
   lowcut = low_cutoff / nyquist
   highcut = high_cutoff / nyquist

   b, a = butter(4, [lowcut, highcut], btype='band')
   filtered_samples = lfilter(b, a, samples)
   return filtered_samples


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(128, kernel_size=9, activation='relu', input_shape=(8192, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )
    
    return model


def calibrate_model_threshold(model, chunk_size=8192, n_pos=20, n_neg=20):
    global MODEL_RECOMMENDED_THRESHOLD
    pos_outputs = []
    neg_outputs = []
    try:
        pos_pool = generate_wow_signals(chunk_size, as_numpy=True)
    except Exception:
        pos_pool = None

    for i in range(n_pos):
        if pos_pool is not None and len(pos_pool) > 0:
            sig = pos_pool[i % len(pos_pool)]
        else:
            sig = np.random.normal(0, 1.0, chunk_size).astype(np.float32)
        x = _prepare_input_for_model(sig)
        out = model(x, training=False)
        try:
            val = float(out[0][0])
        except Exception:
            val = float(np.ravel(out)[0])
        pos_outputs.append(val)

    for i in range(n_neg):
        noise = np.random.normal(0, 1.0, chunk_size).astype(np.float32)
        x = _prepare_input_for_model(noise)
        out = model(x, training=False)
        try:
            val = float(out[0][0])
        except Exception:
            val = float(np.ravel(out)[0])
        neg_outputs.append(val)

    pos_mean = float(np.mean(pos_outputs)) if pos_outputs else 0.0
    neg_mean = float(np.mean(neg_outputs)) if neg_outputs else 0.0
    pos_std = float(np.std(pos_outputs)) if pos_outputs else 0.0
    neg_std = float(np.std(neg_outputs)) if neg_outputs else 0.0

    recommended = float(np.clip((pos_mean + neg_mean) / 2.0, 0.0, 1.0))
    MODEL_RECOMMENDED_THRESHOLD = recommended
    logger.info(f"Calibration: pos_mean={pos_mean:.6f}, neg_mean={neg_mean:.6f}, recommended_threshold={recommended:.6f}")
    return {
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'pos_std': pos_std,
        'neg_std': neg_std,
        'recommended_threshold': recommended
    }


def train_model(model, csv_data):
    callbacks_dir = os.path.join('models', 'callbacks')
    os.makedirs(callbacks_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(callbacks_dir, 'best_model.keras'),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join('models', 'logs'),
            histogram_freq=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
    ]

    base_gen = data_generator(chunk_size=8192, csv_data=csv_data)

    def balanced_data_generator(base_gen):
        while True:
            X, y = next(base_gen)
            class_counts = np.bincount(y.astype(int))
            if len(class_counts) < 2:
                continue
            class_weights = {0: 1., 1: class_counts[0] / class_counts[1]}
            sample_weights = np.array([class_weights[int(label)] for label in y])
            yield X, y, sample_weights

    val_gen = data_generator(chunk_size=8192, csv_data=csv_data)

    total_samples = int(os.getenv('TRAIN_TOTAL_SAMPLES', '20000'))
    batch_size = int(os.getenv('TRAIN_BATCH_SIZE', '32'))
    epochs = int(os.getenv('TRAIN_EPOCHS', '200'))

    if os.getenv('TRAIN_STEPS_PER_EPOCH'):
        steps_per_epoch = int(os.getenv('TRAIN_STEPS_PER_EPOCH'))
    else:
        steps_per_epoch = max(50, total_samples // batch_size)

    if os.getenv('TRAIN_VALIDATION_STEPS'):
        validation_steps = int(os.getenv('TRAIN_VALIDATION_STEPS'))
    else:
        validation_steps = max(10, total_samples // (5 * batch_size))

    history = model.fit(
        balanced_data_generator(base_gen),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    state_dir = os.path.join('models', 'full_state')
    os.makedirs(state_dir, exist_ok=True)
    model.save(os.path.join(state_dir, 'full_model.keras'))

    return history


def load_model_weights(model, weights_file):
   weights_path = os.path.join('models', weights_file)
   if os.path.exists(weights_path):
       model.load_weights(weights_path)
       logger.info(f"Successfully loaded weights from {weights_path}")
   else:
       logger.error(f"Model weights file {weights_path} does not exist")


def check_model_for_nans(model):
    weights = model.get_weights()
    for i, w in enumerate(weights):
        if np.any(np.isnan(w)) or np.any(np.isinf(w)):
            logger.error(f"Model weight {i} contains NaN or Inf!")
            return False
    logger.info("Model weights are clean (no NaN or Inf).")
    return True


def inject_wow_signal(samples, amplitude=25.0):
    t = np.arange(len(samples))
    center_freq = 0.01
    drift_rate = 0.003
    intensity = 2.0 * np.exp(-(t - len(t)/2)**2 / (len(t)/4)**2)
    freq_drift = center_freq + drift_rate * (t - len(t)/2) / len(t)
    wow_signal = amplitude * intensity * np.sin(2 * np.pi * freq_drift * t)
    wow_signal = wow_signal * np.exp(1j * 2 * np.pi * freq_drift * t)
    phase_coherence = np.exp(1j * np.pi/3)
    wow_signal = wow_signal * phase_coherence
    return samples.astype(np.complex64) + wow_signal.astype(np.complex64)


def predict_signal(model, samples):
    try:
        logger.debug(f"Raw input samples type: {type(samples)}, dtype: {getattr(samples, 'dtype', 'N/A')}, shape: {np.shape(samples)}")
        input_tensor, samples_normalized = _prepare_input_for_model(samples, expected_len=8192, return_normalized=True)
        logger.debug(f"Input tensor prepared - dtype: {input_tensor.dtype}, shape: {input_tensor.shape}")
        if samples_normalized is None or np.std(samples_normalized) < 1e-6:
            logger.warning("Near-zero standard deviation in input signal (after prep)")
            return (False, 0.0)
        confidence = model(input_tensor, training=False)
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
        try:
            confidence_value = float(confidence[0][0])
        except Exception:
            try:
                confidence_value = float(np.ravel(confidence)[0])
            except Exception:
                logger.debug("Failed to parse raw model output into float; defaulting to 0.0")
                confidence_value = 0.0
        norm_std = float(np.std(samples_normalized)) if samples_normalized.size else 0.0
        norm_max = float(np.max(np.abs(samples_normalized))) if samples_normalized.size else 0.0
        alpha = min(1.0, norm_std if norm_std > 0 else 1.0)
        threshold = 0.7 - 0.15 * alpha
        threshold = float(np.clip(threshold, 0.4, 0.95))
        used_threshold = threshold
        try:
            if MODEL_RECOMMENDED_THRESHOLD is not None and MODEL_RECOMMENDED_THRESHOLD > 1e-6:
                used_threshold = float(max(threshold, float(MODEL_RECOMMENDED_THRESHOLD)))
        except Exception:
            used_threshold = threshold
        try:
            if RUNTIME_DETECTION_THRESHOLD is not None:
                used_threshold = float(RUNTIME_DETECTION_THRESHOLD)
        except Exception:
            pass
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
