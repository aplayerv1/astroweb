from asyncio import sleep
import logging
import sys
import numpy as np
from scipy.signal import lfilter, butter
import argparse
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import psutil
from astropy.io import fits
import h5py
import scipy.signal
import datetime
import socket
import json
from advance_signal_processing import denoise_signal, remove_dc_offset, remove_lnb_effect, process_fft
from training import generate_wow_signals, generate_pulsar_signals, generate_frb_signals, generate_hydrogen_line, load_training_data_from_folder
import cupy as cp
from pyhackrf2 import HackRF
from collections import deque

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# Constants
fs = 20e6          # Sampling frequency in Hz
notch_freq = 100e3 # Notch frequency in Hz
notch_width = 50e3 # Notch width in Hz
LNB_LOW = 9.75e9   # Low band LNB frequency (9.75 GHz)
LNB_HIGH = 10.6e9  # High band LNB frequency (10.6 GHz)
freq_start = 1420e6
freq_stop = 1420.40e6
sample_rate = fs
center_freq = freq_stop - freq_start
bandwidth = fs
gain = 5
lnb_band = LNB_HIGH
signal_buffer = deque(maxlen=100)

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

    # Custom optimizer configuration
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

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

def data_generator(chunk_size=8192, csv_data=None, debug=False):
    if isinstance(csv_data, str):
        csv_data = load_training_data_from_folder(csv_data, chunk_size)
    
    while True:
        t = cp.arange(chunk_size, dtype=cp.float32)

        # Generate WOW signals and apply boosted amplitude
        wow_signals = cp.array(generate_wow_signals(chunk_size))
        wow_signals = wow_signals.reshape(-1, chunk_size)
        wow_signals *= cp.random.uniform(0.5, 1.5, size=(wow_signals.shape[0], 1))  # boosted range
        wow_signals += cp.random.normal(0, 0.3, size=wow_signals.shape)

        # Interference signals
        interference = cp.stack([
            cp.random.normal(size=chunk_size) * (1 + 0.5 * cp.sin(2 * cp.pi * 0.03 * t)),
            cp.random.rayleigh(size=chunk_size) * cp.random.uniform(0.2, 1.8),
            cp.cumsum(cp.random.normal(size=chunk_size)) * cp.exp(-t/chunk_size) * cp.random.uniform(0.3, 1.7),
            cp.random.normal(0, cp.exp(-t/chunk_size)) * cp.random.uniform(0.4, 1.6)
        ])

        # Natural signals
        natural_signals = cp.stack([
            cp.random.normal(size=chunk_size) * (1 + 0.3 * cp.sin(2 * cp.pi * 0.01 * t)),
            cp.random.rayleigh(size=chunk_size) + cp.random.normal(size=chunk_size) * 0.2,
            cp.cumsum(cp.random.normal(size=chunk_size)) * cp.exp(-t/chunk_size),
            cp.random.normal(0, cp.exp(-t/chunk_size)) + cp.random.poisson(1.0, size=chunk_size),
            cp.random.chisquare(df=2, size=chunk_size) * cp.random.uniform(0.5, 1.5),
            cp.sin(2 * cp.pi * 0.1 * t) * cp.random.normal(size=chunk_size),
            cp.random.exponential(scale=1.0, size=chunk_size),
            cp.random.gamma(2.0, 2.0, size=chunk_size),
            cp.random.normal(0, 1 + 0.5 * cp.sin(2 * cp.pi * 0.01 * t)),
            cp.ones(chunk_size) * cp.random.normal(),
            cp.linspace(-1, 1, chunk_size) * cp.random.normal(),
            cp.sin(2 * cp.pi * 0.001 * t) * cp.random.normal(),
            cp.random.normal() + cp.random.normal(size=chunk_size) * 0.1
        ])

        # Mix interference into natural signals
        natural_signals = cp.stack([
            signal + interference[i % len(interference)] * cp.random.uniform(0.1, 0.4)
            for i, signal in enumerate(natural_signals)
        ])

        # Positive samples = WOW + optional CSV
        all_positive = [wow_signals]
        if csv_data is not None:
            csv_data = cp.array(csv_data).reshape(-1, chunk_size)
            all_positive.append(csv_data)
        positive = cp.vstack(all_positive)
        negative = natural_signals

        # Balance samples
        min_len = min(len(positive), len(negative))
        positive = positive[:min_len]
        negative = negative[:min_len]

        # Stack and label
        X = cp.vstack([positive, negative])
        y = cp.hstack([
            cp.ones(len(positive)),
            cp.zeros(len(negative))
        ])

        # Reshape for Conv1D
        X = cp.reshape(X, (-1, chunk_size, 1))
        y = cp.reshape(y, (-1,))

        # Shuffle
        indices = cp.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        # Debug: visualize first signal
        if debug:
            import matplotlib.pyplot as plt
            plt.plot(cp.asnumpy(X[0]).squeeze())
            plt.title(f"Debug Signal Plot - Label: {int(y[0])}")
            plt.show()

        yield cp.asnumpy(X), cp.asnumpy(y)

       
class PerformanceEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, acc_threshold=0.99, loss_threshold=0.01, auc_threshold=0.99):
        super().__init__()
        self.acc_threshold = acc_threshold
        self.loss_threshold = loss_threshold
        self.auc_threshold = auc_threshold

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get('accuracy', 0)
        auc = logs.get('auc', 0)
        loss = logs.get('loss', 1)

        if acc >= self.acc_threshold and auc >= self.auc_threshold and loss <= self.loss_threshold:
            print(f"\n✅ Early stopping: acc={acc:.4f}, auc={auc:.4f}, loss={loss:.4f}")
            self.model.stop_training = True
            


def train_model(model, csv_data):
    # Create callbacks directory
    callbacks_dir = os.path.join('models', 'callbacks')
    os.makedirs(callbacks_dir, exist_ok=True)

    # Define callbacks
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
        PerformanceEarlyStopping(
            acc_threshold=0.95,
            auc_threshold=0.98,
            loss_threshold=0.1
        )
    ]

    # Setup data generator
    base_gen = data_generator(chunk_size=8192, csv_data=csv_data)

    # Balanced training generator
    def balanced_data_generator(base_gen):
        while True:
            X, y = next(base_gen)

            # Optional debugging
            preds = model.predict(X, verbose=0)
            for i in range(len(preds)):
                print(f"Prediction: {preds[i][0]:.4f} - Label: {y[i]}")
            print("Shapes:", X.shape, y.shape)
            print("Label distribution:", np.bincount(y.astype(int)))

            class_counts = np.bincount(y.astype(int))
            if len(class_counts) < 2:
                continue  # Skip batch if it's only one class

            class_weights = {0: 1., 1: class_counts[0] / class_counts[1]}
            sample_weights = np.array([class_weights[int(label)] for label in y])
            yield X, y, sample_weights

    # Validation generator
    val_gen = data_generator(chunk_size=8192, csv_data=csv_data)

    # Training
    total_samples = 10000  # Adjust as needed
    batch_size = 32
    steps_per_epoch = total_samples // batch_size
    validation_steps = total_samples // (5 * batch_size)

    history = model.fit(
        balanced_data_generator(base_gen),
        steps_per_epoch=steps_per_epoch,
        epochs=100,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # Save model
    state_dir = os.path.join('models', 'full_state')
    os.makedirs(state_dir, exist_ok=True)
    model.save(os.path.join(state_dir, 'full_model.keras'))

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history.get('auc', []), label='Training AUC')
    plt.plot(history.history.get('val_auc', []), label='Validation AUC')
    plt.title('Training History (AUC)')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.savefig(os.path.join(state_dir, 'training_history.png'))
    plt.close()

    # Save batch norm states
    bn_states = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            mean = layer.moving_mean.numpy()
            variance = layer.moving_variance.numpy()
            bn_states[layer.name] = {
                'mean': mean.tolist(),
                'variance': variance.tolist()
            }
            print(f"BatchNorm Layer '{layer.name}':")
            print(f"  moving_mean (first 5): {mean[:5]}")
            print(f"  moving_variance (first 5): {variance[:5]}")

    with open(os.path.join(state_dir, 'bn_states.json'), 'w') as f:
        json.dump(bn_states, f)

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


def plot_signal_strength(signal_strength, output_dir, timestamp):
   logger.debug("Plotting signal strength")
   plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
   os.makedirs(plot_dir, exist_ok=True)
   
   filename = os.path.join(plot_dir, 'signal_strength.png')
   plt.figure(figsize=(12, 6))
   plt.plot(signal_strength)
   plt.title('Signal Strength')
   plt.xlabel('Samples')
   plt.ylabel('Strength')
   plt.grid(True)
   plt.savefig(filename, dpi=300, bbox_inches='tight')
   plt.close()
   logger.info(f"Signal strength plot saved to {filename}")

def plot_spectrogram(signal, sample_rate, nperseg, output_dir, timestamp, title='Spectrogram'):
   logger.debug("Plotting spectrogram")
   plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
   os.makedirs(plot_dir, exist_ok=True)
   
   filename = os.path.join(plot_dir, 'spectrogram.png')
   noverlap = nperseg // 2
   
   f, t, Sxx = scipy.signal.spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
   
   fig = plt.figure(figsize=(12, 8))
   plt.imshow(10 * np.log10(Sxx), aspect='auto', origin='lower',
              extent=[f.min()/1e6, f.max()/1e6, t.min(), t.max()])
   plt.xlabel('Frequency (MHz)')
   plt.ylabel('Time (s)')
   plt.colorbar(label='Power (dB)')
   plt.title(title)
   plt.tight_layout()
   
   fig.savefig(filename, dpi=300, bbox_inches='tight')
   plt.close(fig)
   logger.info(f"Spectrogram saved to {filename}")

def plot_signal_analysis(signal, sample_rate, output_dir, timestamp):
   # Create plots directory
   plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
   os.makedirs(plot_dir, exist_ok=True)
   
   # Time domain plot
   plt.figure(figsize=(12, 6))
   t = np.arange(len(signal)) / sample_rate
   plt.plot(t, np.abs(signal))
   plt.title('Signal Amplitude vs Time')
   plt.xlabel('Time (s)')
   plt.ylabel('Amplitude')
   plt.savefig(os.path.join(plot_dir, 'time_domain.png'))
   plt.close()
   
   # Frequency domain plot
   plt.figure(figsize=(12, 6))
   freq = np.fft.fftfreq(len(signal), 1/sample_rate)
   fft_data = np.abs(np.fft.fft(signal))
   plt.plot(freq/1e6, fft_data)
   plt.title('Signal Frequency Spectrum')
   plt.xlabel('Frequency (MHz)')
   plt.ylabel('Magnitude')
   plt.grid(True)
   plt.savefig(os.path.join(plot_dir, 'frequency_spectrum.png'))
   plt.close()

def set_cpu_affinity(cores):
   logger.debug(f"Setting CPU affinity to cores: {cores}")
   p = psutil.Process(os.getpid())
   p.cpu_affinity(cores)

def connect_to_server():
    logger.debug("Connecting to HackRF device via USB...")
    
    try:
        sdr = HackRF()
        sdr.sample_rate = sample_rate
        sdr.center_freq = center_freq
        sdr.bandwidth = bandwidth
        sdr.lna_gain = gain
        sdr.vga_gain = gain
        sdr.amp_enable = gain > 0
        sdr.amplifier_on = True
        
        logging.info(f"Configured SDR with sample_rate={sample_rate}, gain={gain}, center_freq={center_freq}, bandwidth={bandwidth}")
        logger.info("HackRF device initialized successfully")
        return sdr
    except Exception as e:
        logger.error(f"Failed to initialize HackRF: {e}")
        raise ConnectionError("HackRF USB connection failed")


def predict_signal(model, samples):
    try:
        logger.debug(f"Raw input samples type: {type(samples)}, dtype: {getattr(samples, 'dtype', 'N/A')}, shape: {np.shape(samples)}")

        samples = np.asarray(samples)
        logger.debug(f"After np.asarray - dtype: {samples.dtype}, shape: {samples.shape}")

        # If complex, convert to magnitude
        if np.iscomplexobj(samples):
            samples_real = np.abs(samples)
            logger.debug(f"Converted complex samples to magnitude - dtype: {samples_real.dtype}, shape: {samples_real.shape}")
        else:
            samples_real = samples
            logger.debug(f"Samples are real-valued")

        signal_mean = np.mean(samples_real)
        signal_std = np.std(samples_real)
        signal_min = np.min(samples_real)
        signal_max = np.max(samples_real)
        logger.debug(f"Samples stats - mean: {signal_mean}, std: {signal_std}, min: {signal_min}, max: {signal_max}")

        if signal_std < 1e-6:
            logger.warning("Near-zero standard deviation in input signal")
            return (False, 0.0)

        samples_normalized = (samples_real - signal_mean) / signal_std
        samples_normalized = np.clip(samples_normalized, -5.0, 5.0)
        logger.debug(f"Samples normalized - dtype: {samples_normalized.dtype}, shape: {samples_normalized.shape}, min: {np.min(samples_normalized)}, max: {np.max(samples_normalized)}")

        input_tensor = tf.reshape(samples_normalized.astype(np.float32), (1, 8192, 1))
        logger.debug(f"Input tensor prepared - dtype: {input_tensor.dtype}, shape: {input_tensor.shape}")

        confidence = model(input_tensor, training=False)
        confidence_value = float(confidence[0][0])

        threshold = 0.5 - (0.2 * (signal_std / np.max(np.abs(samples_normalized))))
        detection = confidence_value >= max(0.1, threshold)

        logger.debug(f"Prediction - Confidence: {confidence_value:.6f}, Threshold: {threshold:.6f}, Detection: {detection}")

        return (detection, confidence_value)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return (False, 0.0)


def process_continuous_stream(sdr, model, output_dir):
    import sys
    import select
    import termios
    import tty
    from scipy import signal

    class BufferTracker:
        def __init__(self, maxlen=100):
            self.values = deque(maxlen=maxlen)
            self.confidences = deque(maxlen=maxlen)
            self.timestamps = deque(maxlen=maxlen)
            self.detections = deque(maxlen=maxlen)
            
        def update(self, value, confidence, detection):
            self.values.append(value)
            self.confidences.append(confidence)
            self.timestamps.append(datetime.datetime.now())
            self.detections.append(detection)
            
        def stats(self):
            if not self.values:
                return {
                    'count': 0,
                    'capacity': self.values.maxlen,
                    'mean_value': 0.0,
                    'mean_confidence': 0.0,
                    'detection_rate': "0%",
                    'update_rate': "0 Hz",
                    'last_change': "Never"
                }
                
            time_diff = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            rate = len(self.values)/max(0.1, time_diff)
            
            return {
                'count': len(self.values),
                'capacity': self.values.maxlen,
                'mean_value': float(np.mean(self.values)),
                'mean_confidence': float(np.mean(self.confidences)),
                'detection_rate': f"{100 * np.mean(self.detections):.1f}%",
                'update_rate': f"{rate:.1f} Hz",
                'last_change': self.timestamps[-1].strftime("%H:%M:%S.%f")[:-3]
            }

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

    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        logger.info("Starting enhanced continuous stream processing")
        buffer = BufferTracker(maxlen=100)
        data_buffer = np.array([], dtype=np.complex64)
        chunk_size = 8192
        inject_wow = False
        consecutive_empty_reads = 0
        max_empty_reads = 3
        last_stat_log = datetime.datetime.now()
        
        # Test model with synthetic signal
        test_signal = np.random.normal(0, 0.5, chunk_size)
        test_detection, test_confidence = predict_signal(model, test_signal)
        logger.info(f"Model test - Confidence: {test_confidence:.4f} (should be > 0)")
        
        while True:
            try:
                # Check for key presses
                key = is_key_pressed()
                if key == 'w':
                    logger.info("WOW Signal injection triggered!")
                    inject_wow = True
                elif key == 'd':
                    logger.info("Debug signal injection")
                    debug_signal = np.random.normal(0, 0.5, chunk_size)
                    debug_detection, debug_confidence = predict_signal(model, debug_signal)
                    logger.info(f"Debug test - Confidence: {debug_confidence:.4f}")
                elif key == 'q':
                    logger.info("Quit key detected. Shutting down gracefully...")
                    break
                    
                # Read from SDR
                try:
                    chunk = sdr.read_samples(chunk_size)
                    
                    if chunk is None or len(chunk) == 0:
                        consecutive_empty_reads += 1
                        logger.warning(f"No samples received. Empty read count: {consecutive_empty_reads}")
                        if consecutive_empty_reads >= max_empty_reads:
                            logger.error("Maximum empty reads reached, reconnecting...")
                            sdr.close()
                            sdr = connect_to_server()
                            consecutive_empty_reads = 0
                        continue
                    else:
                        consecutive_empty_reads = 0

                except Exception as e:
                    logger.error(f"Error reading from HackRF: {e}")
                    consecutive_empty_reads += 1
                    if consecutive_empty_reads >= max_empty_reads:
                        logger.error("Reconnecting after multiple failures...")
                        sdr.close()
                        sdr = connect_to_server()
                        consecutive_empty_reads = 0
                    continue

                # Process the new chunk
                samples = chunk.astype(np.complex64)
                data_buffer = np.append(data_buffer, samples)

                while len(data_buffer) >= chunk_size:
                    process_chunk = data_buffer[:chunk_size]
                    
                    # Signal processing pipeline
                    try:
                        if cp.cuda.is_available():
                            process_chunk_gpu = cp.asarray(process_chunk)
                            processed_samples_gpu = denoise_signal(process_chunk_gpu)
                            processed_samples = cp.asnumpy(processed_samples_gpu)
                        else:
                            processed_samples = denoise_signal(process_chunk)
                            
                    except Exception as e:
                        logger.warning(f"Processing failed, using raw samples: {e}")
                        processed_samples = process_chunk

                    # Optional WOW signal injection
                    if inject_wow:
                        processed_samples = inject_wow_signal(processed_samples)
                        inject_wow = False
                        logger.info("Injected WOW signal into current chunk")
                    logger.debug(f"Samples stats - mean: {np.mean(samples):.4f}, std: {np.std(samples):.4f}, min: {np.min(samples):.4f}, max: {np.max(samples):.4f}")
                    # Prepare for prediction
                    prediction_samples = np.abs(processed_samples)
                    signal_strength = 20 * np.log10(np.mean(prediction_samples) + 1e-10)
                    
                    # Model prediction
                    detection, confidence_value = predict_signal(model, prediction_samples)
                    
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
                            f"Buffer Status: {stats['count']}/{stats['capacity']} | "
                            f"Avg Confidence: {stats['mean_confidence']:.4f} | "
                            f"Detections: {stats['detection_rate']} | "
                            f"Rate: {stats['update_rate']} | "
                            f"Last: {stats['last_change']}"
                        )
                        last_stat_log = current_time
                    
                    # FFT processing with proper data conversion
                    try:
                        if cp.cuda.is_available():
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
                        
                        # Create properly structured FFT data
                        fft_data_dict = {
                            'magnitude': np.abs(fft_magnitude).tolist(),
                            'frequency': (fft_freq + freq_start/1e6).tolist(),
                            'power': np.abs(fft_power).tolist(),
                            'phase': fft_phase.tolist() if isinstance(fft_phase, np.ndarray) else []
                        }
                    except Exception as e:
                        logger.error(f"FFT processing failed: {e}")
                        fft_data_dict = {
                            'magnitude': [],
                            'frequency': [],
                            'power': [],
                            'phase': []
                        }
                    
                    # Prepare complete web data package
                    web_data = {
                        'timestamp': current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        'frequency': f"{freq_start/1e6:.2f} MHz",
                        'signal_strength': f"{signal_strength:.2f} dB",
                        'confidence': float(confidence_value),
                        'detection': bool(detection),
                        'buffer_stats': buffer.stats(),
                        'fft_data': fft_data_dict,
                        'system_status': {
                            'gpu_available': cp.cuda.is_available(),
                            'cpu_usage': psutil.cpu_percent(),
                            'memory_usage': psutil.virtual_memory().percent,
                            'process_memory': psutil.Process().memory_info().rss / 1024 / 1024  # MB
                        }
                    }
                    
                    # Store validated data
                    app.config['LATEST_DATA'] = validate_web_data(web_data)
                    
                    # Handle detections
                    if detection:
                        timestamp = datetime.datetime.now()
                        logger.info(f"🚨 Detection! Confidence: {confidence_value:.4f}")
                        save_detected_signal(processed_samples, timestamp, output_dir)
                        plot_signal_strength(processed_samples, output_dir, timestamp)
                        plot_spectrogram(processed_samples, fs, chunk_size, output_dir, timestamp)

                    # Move to next chunk
                    data_buffer = data_buffer[chunk_size:]

            except (ConnectionError, socket.error) as e:
                logger.error(f"Connection error: {e}")
                break
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
    
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        sdr.close()
        logger.info("Processing stopped and SDR closed")

def save_detected_signal(processed_samples, timestamp, output_dir):
    detection_path = os.path.join(output_dir, 'detections', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(detection_path, exist_ok=True)
    
    fits_path = os.path.join(detection_path, f'signal_detection_{timestamp.strftime("%Y%m%d_%H%M%S")}.fits')
    logger.debug(f"Writing FITS file to: {fits_path}")
    
    # Create detailed FITS header
    hdr = fits.Header()
    hdr['DATE-OBS'] = timestamp.strftime("%Y-%m-%d")
    hdr['TIME-OBS'] = timestamp.strftime("%H:%M:%S")
    hdr['FREQ-ST'] = f"{freq_start/1e6:.2f}"
    hdr['FREQ-END'] = f"{freq_start/1e6 + (LNB_HIGH - LNB_LOW):.2f}"
    hdr['TELESCOP'] = 'HackRF'
    hdr['INSTRUME'] = 'USB'
    hdr['LOCATION'] = 'Your Observatory Location'
    hdr['OBSERVER'] = 'Your Name'
    hdr['SIGNAL'] = np.mean(np.abs(processed_samples))
    hdr['SAMPRATE'] = f"{fs}"
    hdr['BANDWIDT'] = f"{(LNB_HIGH - LNB_LOW):e}"
    
    # Convert and save data
    processed_samples = np.abs(processed_samples).astype(np.float64)
    primary_hdu = fits.PrimaryHDU(processed_samples, header=hdr)
    fft_data = np.abs(np.fft.fft(processed_samples))
    fft_hdu = fits.ImageHDU(fft_data, name='FFT')
    
    # Create and write FITS file
    hdul = fits.HDUList([primary_hdu, fft_hdu])
    hdul.writeto(fits_path, overwrite=True)
    hdul.close()
    
    if os.path.exists(fits_path):
        logger.debug(f"FITS file successfully written: {os.path.getsize(fits_path)} bytes")
        logger.info(f"Signal data saved to FITS file: {fits_path}")

def inject_wow_signal(samples, amplitude=25.0):
    """Inject the classic 1977 WOW signal pattern"""
    t = np.arange(len(samples))
    
    # Enhanced WOW signal characteristics
    center_freq = 0.01
    drift_rate = 0.003
    
    # Stronger bell-shaped intensity curve
    intensity = 2.0 * np.exp(-(t - len(t)/2)**2 / (len(t)/4)**2)
    
    # Sharper frequency drift
    freq_drift = center_freq + drift_rate * (t - len(t)/2) / len(t)
    wow_signal = amplitude * intensity * np.sin(2 * np.pi * freq_drift * t)
    
    # Enhanced narrowband characteristics
    wow_signal = wow_signal * np.exp(1j * 2 * np.pi * freq_drift * t)
    
    # Add phase coherence
    phase_coherence = np.exp(1j * np.pi/3)
    wow_signal = wow_signal * phase_coherence
    
    return samples.astype(np.complex64) + wow_signal.astype(np.complex64)

def main():
    # Set threading and precision config
    try:
        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
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
    else:
        logger.warning("No GPU found, falling back to CPU")

    # Parse CLI args and env variables
    parser = argparse.ArgumentParser(description='Process continuous SDR stream.', 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output-dir', type=str, default='output')
    parser.add_argument('--band', choices=['low', 'high'], default='low')
    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--fast-train', action='store_true')
    parser.add_argument('--force-train', action='store_true')
    args = parser.parse_args()

    env_train = os.getenv('TRAIN', '').lower() in ('1','true','yes','on')
    env_fast_train = os.getenv('FAST_TRAIN', '').lower() in ('1','true','yes','on')
    env_force_train = os.getenv('FORCE_TRAIN', '').lower() in ('1','true','yes','on')

    args.train = args.train or env_train
    args.fast_train = args.fast_train or env_fast_train
    args.force_train = args.force_train or env_force_train

    if args.train:
        logger.info("🔄 Training enabled")
    if args.fast_train:
        logger.info("⚡ Fast training enabled")
    if args.force_train:
        logger.info("💥 Force training enabled")

    os.makedirs(args.output_dir, exist_ok=True)

    lnb_band = LNB_LOW if args.band == 'low' else LNB_HIGH
    logger.debug(f"LNB band selected: {lnb_band}")

    model_dir = os.path.join('models', 'full_state')
    model_path = os.path.join(model_dir, 'full_model.keras')
    bn_path = os.path.join(model_dir, 'bn_states.json')

    model = None

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
    # Proceed to continuous SDR processing
    try:
        sdr = connect_to_server()
        process_continuous_stream(sdr, model, args.output_dir)
    except Exception as e:
        logger.error(f"SDR processing failed: {e}")
    finally:
        if 'sdr' in locals():
            sdr.close()
            logger.info("HackRF device closed")
            
if __name__ == "__main__":
    from threading import Thread
    from web import app
    
    processing_thread = Thread(target=main)
    processing_thread.start()
    
    app.run(host='0.0.0.0', port=5001)