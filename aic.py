import logging
from time import sleep
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

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(-1.0))
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    return model

def data_generator(chunk_size=8192, csv_data=None, batch_size=32, wow_perc=0.5):
    """
    Yields (X, y) with shapes (batch_size, chunk_size, 1) and (batch_size, 1).
    Keeps class balance roughly wow_perc : (1 - wow_perc).
    """
    if isinstance(csv_data, str):
        csv_data = load_training_data_from_folder(csv_data, chunk_size)

    # Preload CSV once on GPU (if any)
    csv_pool = None
    if csv_data is not None and len(csv_data) > 0:
        csv_pool = cp.asarray(csv_data, dtype=cp.float32).reshape(-1, chunk_size)

    nat_count = 13  # we generate 13 natural patterns per call (see below)

    while True:
        # Pools for this step
        t = cp.arange(chunk_size, dtype=cp.float32)

        # WOW pool
        wow = cp.asarray(generate_wow_signals(chunk_size), dtype=cp.float32).reshape(-1, chunk_size)
        wow = wow * cp.random.uniform(0.1, 1.0, size=(wow.shape[0], 1))
        wow += cp.random.normal(0, 0.3, size=wow.shape)

        # Interference (used to mix into naturals)
        interference = cp.stack([
            cp.random.normal(size=chunk_size) * (1 + 0.5 * cp.sin(2 * cp.pi * 0.03 * t)),
            cp.random.rayleigh(size=chunk_size) * cp.random.uniform(0.2, 1.8),
            cp.cumsum(cp.random.normal(size=chunk_size)) * cp.exp(-t/chunk_size) * cp.random.uniform(0.3, 1.7),
            cp.random.normal(0, cp.exp(-t/chunk_size)) * cp.random.uniform(0.4, 1.6)
        ])

        naturals_raw = [
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
        ]
        naturals = cp.stack([
            s + interference[i % len(interference)] * cp.random.uniform(0.1, 0.4)
            for i, s in enumerate(naturals_raw)
        ])  # shape (13, chunk_size)

        # Build a single fixed-size batch
        n_wow = int(batch_size * wow_perc)
        n_nat = batch_size - n_wow

        # Sample rows for WOW / CSV positive class
        pos_pool = wow
        if csv_pool is not None:
            # concatenate CSV and WOW as positives
            pos_pool = cp.concatenate([pos_pool, csv_pool], axis=0)

        pos_idx = cp.random.randint(0, pos_pool.shape[0], size=n_wow)
        X_pos = pos_pool[pos_idx]

        # Sample rows for natural (negative) class
        nat_idx = cp.random.randint(0, nat_count, size=n_nat)
        X_neg = naturals[nat_idx]

        X = cp.concatenate([X_pos, X_neg], axis=0).astype(cp.float32)
        y = cp.concatenate([cp.ones((n_wow, 1), dtype=cp.float32),
                            cp.zeros((n_nat, 1), dtype=cp.float32)], axis=0)

        # Per-sample z-score + clip to match training/inference
        mean_val = cp.mean(X, axis=1, keepdims=True)
        std_val = cp.std(X, axis=1, keepdims=True) + 1e-6
        X = (X - mean_val) / std_val
        X = cp.clip(X, -1.0, 1.0)

        # Final shapes
        X = X.reshape(batch_size, chunk_size, 1)

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
            
def train_model(model, csv_data, batch_size=32, steps_per_epoch=20, epochs=100):
    # Disable XLA / mixed precision while debugging
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

    # Compile (float32) to keep numerics simple
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    gen = data_generator(chunk_size=8192, csv_data=csv_data, batch_size=batch_size, wow_perc=0.5)

    history = model.fit(
        gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1
    )

    # Save model + BN states
    state_dir = os.path.join('models', 'full_state')
    os.makedirs(state_dir, exist_ok=True)
    model.save(os.path.join(state_dir, 'full_model.keras'))

    bn_states = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_states[layer.name] = {
                'mean': layer.moving_mean.numpy().tolist(),
                'variance': layer.moving_variance.numpy().tolist()
            }
    with open(os.path.join(state_dir, 'bn_states.json'), 'w') as f:
        json.dump(bn_states, f)

    X_test, y_test = next(data_generator(batch_size=32))
    preds = model.predict(X_test)
    print("pos mean:", preds[y_test[:,0]==1].mean(), "neg mean:", preds[y_test[:,0]==0].mean())
    logger.info(f"Sanity check — mean(pred|pos)={preds[y_test[:,0]==1].mean():.3f}, mean(pred|neg)={preds[y_test[:,0]==0].mean():.3f}")
    logger.info(f"Sanity check — mean(pred|pos)={preds[y_test.reshape(-1) > 0.5].mean():.3f}, mean(pred|neg)={preds[y_test.reshape(-1) < 0.5].mean():.3f}")
    pos_mean = float(preds[y_test.reshape(-1) > 0.5].mean()) if np.any(y_test > 0.5) else float('nan')
    neg_mean = float(preds[y_test.reshape(-1) < 0.5].mean()) if np.any(y_test < 0.5) else float('nan')
    logger.info(f"Sanity check — mean(pred|pos)={pos_mean:.3f}, mean(pred|neg)={neg_mean:.3f}")

    return history


def load_model_weights(model, weights_file):
   weights_path = os.path.join('models', weights_file)
   if os.path.exists(weights_path):
       model.load_weights(weights_path)
       logger.info(f"Successfully loaded weights from {weights_path}")
   else:
       logger.error(f"Model weights file {weights_path} does not exist")

def predict_signal(model, samples, chunk_size=8192):
    logger.debug("Starting predict_signal function")

    # Ensure 1D numpy array
    if hasattr(samples, 'get'):  # CuPy
        samples = samples.get()
    samples = np.asarray(samples).astype(np.float32).reshape(-1)

    # Pad or truncate to exact chunk_size
    if samples.shape[0] < chunk_size:
        pad = np.zeros((chunk_size - samples.shape[0],), dtype=np.float32)
        samples = np.concatenate([samples, pad], axis=0)
    elif samples.shape[0] > chunk_size:
        samples = samples[:chunk_size]

    # Magnitude if complex
    if np.iscomplexobj(samples):
        samples = np.abs(samples).astype(np.float32)

    # Per-sample z-score + clip to [-1, 1] to match training
    mean_val = np.mean(samples)
    std_val = np.std(samples) + 1e-6
    samples = (samples - mean_val) / std_val
    samples = np.clip(samples, -1.0, 1.0)

    x = samples.reshape(1, chunk_size, 1)

    # Inference in float32
    tf.keras.mixed_precision.set_global_policy('float32')

    logits = model(x, training=True)  # (1, 1)
    confidence = float(logits.numpy().reshape(()))

    logger.debug(f"Model confidence: {confidence:.6f}")
    result = (confidence >= 0.75, confidence)
    logger.debug(f"Detection result: {result}")
    return result



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

def process_continuous_stream(sdr, model, output_dir):
    import sys
    import select
    import termios
    import tty
    from scipy import signal

    
    def is_key_pressed():
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key
        return None
    
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        logger.debug("Starting continuous stream processing using USB connection")
        data_buffer = np.array([], dtype=np.complex64)
        chunk_size = 8192
        inject_wow = False
        consecutive_empty_reads = 0
        max_empty_reads = 3
        
        while True:
            try:
                key = is_key_pressed()
                if key == 'w':
                    logger.info("WOW Signal injection triggered!")
                    inject_wow = True
                elif key == 'q':
                    logger.info("Quit key detected. Shutting down gracefully...")
                    break
                try:
                    chunk = sdr.read_samples(chunk_size)
                    
                    if chunk is None or len(chunk) == 0:
                        consecutive_empty_reads += 1
                        logger.warning(f"No samples received. Empty read count: {consecutive_empty_reads}")
                    else:
                        consecutive_empty_reads = 0

                    if consecutive_empty_reads >= max_empty_reads:
                        logger.info("Connection appears dead, initiating reconnection...")
                        sdr.close()  # Clean up old instance
                        sdr = connect_to_server()  # Reconnect to HackRF
                        consecutive_empty_reads = 0
                        continue  # Retry loop

                except Exception as e:
                    logger.error(f"Error while reading from HackRF: {e}")
                    consecutive_empty_reads += 1
                    if consecutive_empty_reads >= max_empty_reads:
                        logger.warning("Too many failed reads. Attempting to reconnect...")
                        sdr.close()
                        sdr = connect_to_server()
                        consecutive_empty_reads = 0
                    continue  # Try again

                consecutive_empty_reads = 0
                samples = chunk.astype(np.complex64)
                data_buffer = np.append(data_buffer, samples)

                while len(data_buffer) >= chunk_size:
                    process_chunk = data_buffer[:chunk_size]
                    processed_samples = process_chunk
                    # Convert to CuPy for GPU-accelerated signal processing
                    try:
                        if cp.cuda.is_available():
                            process_chunk_gpu = cp.asarray(process_chunk)
                            logger.info("🚀 Using GPU acceleration for signal processing pipeline")
                            # GPU-accelerated signal processing
                            processed_samples_gpu = process_chunk_gpu
                            # processed_samples_gpu = remove_dc_offset(processed_samples_gpu)
                            processed_samples_gpu = denoise_signal(processed_samples_gpu)
                        else:
                            logger.debug("GPU not available, using CPU for signal processing")
                            processed_samples = remove_dc_offset(process_chunk)
                            processed_samples = denoise_signal(processed_samples)
                            processed_samples_gpu = cp.asarray(processed_samples) if cp.cuda.is_available() else processed_samples
                    except Exception as e:
                        logger.warning(f"GPU processing failed ({e}), falling back to CPU")
                        processed_samples = remove_dc_offset(process_chunk)
                        processed_samples = denoise_signal(processed_samples)
                        processed_samples_gpu = processed_samples

                    if inject_wow:
                        processed_samples = inject_wow_signal(processed_samples)
                        inject_wow = False
                    
                    
                    prediction_samples = np.abs(processed_samples_gpu)
                    signal_strength = 20 * np.log10(np.mean(np.abs(processed_samples_gpu)) + 1e-10)
                    
                    logger.debug("before fft")
                    # GPU-accelerated FFT processing
                    fft_magnitude, fft_freq, fft_data, fft_phase, fft_power = process_fft(processed_samples_gpu, chunk_size, fs)

                    # GPU-optimized model prediction
                    detection, confidence_value = predict_signal(model, prediction_samples)

                    web_data = {
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'frequency': f"{freq_start/1e6:.2f} MHz",
                        'signal_strength': f"{signal_strength:.2f} dB",
                        'status': f"Active - Detection Confidence: {confidence_value*100:.2f}%",
                        'processed_samples': processed_samples_gpu.tolist(),
                        'fft_data': fft_data.tolist(),
                        'fft_freq': (fft_freq + freq_start/1e6).tolist(),
                        'fft_magnitude': fft_magnitude.tolist(),
                        'fft_phase': fft_phase.tolist(),
                        'fft_power': fft_power.tolist(),
                        'fft_power_db': (10 * np.log10(fft_power + 1e-10)).tolist(),
                        'fft_power_db_normalized': (10 * np.log10((fft_power + 1e-10) / np.max(fft_power + 1e-10))).tolist()
                    }

                    app.config['LATEST_DATA'] = web_data
                    
                    if detection:
                        logger.info("WOW Signal detected!")
                        timestamp = datetime.datetime.now()
                        save_detected_signal(processed_samples, timestamp, output_dir)
                        plot_signal_strength(processed_samples, output_dir, timestamp)
                        plot_signal_analysis(processed_samples, fs, output_dir, timestamp)
                        plot_spectrogram(processed_samples, fs, chunk_size, output_dir, timestamp)
                        
                    data_buffer = data_buffer[chunk_size:]

            except (ConnectionError, socket.error) as e:
                logger.error(f"Stream error: {e}")
                break  # Break inner loop to reinitialize HackRF
            except KeyboardInterrupt:
                logger.info("Stream processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                break
    
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        sdr.close()  # Assuming there's a close method for the HackRF device
        logger.info("HackRF device closed.")

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
    # Threading and mixed precision
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

    # Command-line arguments
    parser = argparse.ArgumentParser(
        description='Process continuous SDR stream.',
        epilog='''Environment Variables:
  TRAIN=1           Force model training (same as --train)
  FAST_TRAIN=1      Use faster training settings
Examples:
  python aic.py --train
  TRAIN=1 python aic.py
  FAST_TRAIN=1 python aic.py
  TRAIN=1 FAST_TRAIN=1 python aic.py''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-o', '--output-dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--host', type=str, default='localhost', help='Server host address')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--band', choices=['low', 'high'], default='low', help='LNB band selection')
    parser.add_argument('--path', type=str, default='data', help='Path to CSV file')
    parser.add_argument('--train', action='store_true', help='Force model training')
    parser.add_argument('--fast-train', action='store_true', help='Use faster training settings')
    args = parser.parse_args()

    # Check environment variables
    env_train = os.getenv('TRAIN', '').lower() in ('1', 'true', 'yes', 'on')


    # Combine CLI args with environment variables
    args.train = args.train or env_train

    if env_train:
        logger.info("🔄 Training enabled via TRAIN environment variable")


    os.makedirs(args.output_dir, exist_ok=True)

    lnb_band = LNB_LOW if args.band == 'low' else LNB_HIGH
    logger.debug(f"Selected LNB band: {lnb_band}")

    state_dir = os.path.join('models', 'full_state')
    model_path = os.path.join(state_dir, 'full_model.keras')
    bn_path = os.path.join(state_dir, 'bn_states.json')

    model = None

    if os.path.exists(model_path):
        logger.info("Loading existing model...")
        try:
            model = tf.keras.models.load_model(model_path)
            logger.debug("Model successfully loaded")
            logger.debug(f"Model summary:\n{model.summary(print_fn=lambda x: logger.debug(x))}")
        except Exception as e:
            logger.error(f"❌ Failed to load model from {model_path}: {e}")
            raise

        # Load batch norm states if available
        if os.path.exists(bn_path):
            logger.debug(f"Batch norm state file found: {bn_path}")
            try:
                with open(bn_path, 'r') as f:
                    bn_states = json.load(f)
                for layer in model.layers:
                    if isinstance(layer, tf.keras.layers.BatchNormalization) and layer.name in bn_states:
                        mean_val = np.array(bn_states[layer.name]['mean'])
                        var_val = np.array(bn_states[layer.name]['variance'])
                        layer.moving_mean.assign(mean_val)
                        layer.moving_variance.assign(var_val)
                        logger.debug(f"Restored BN state for {layer.name}")
            except Exception as e:
                logger.error(f"❌ Failed to load BN states from {bn_path}: {e}")

        # Retrain if requested
        if args.train:
            logger.info("🔄 Retraining existing model as --train was specified")
            train_model(model, args.path)

    else:
        # No model exists
        if args.train:
            logger.info("Training new model from scratch...")
            model = create_model()
            train_model(model, args.path)
        else:
            logger.error("❌ Model file not found and training was not requested.")
            logger.error("Use --train or set TRAIN=1 to train a new model.")
            return

    # Save the model after training/retraining
    if args.train:
        os.makedirs(state_dir, exist_ok=True)
        model.save(model_path)
        logger.info(f"✅ Model saved to {model_path}")

        # Save batch norm states
        bn_states = {}
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                bn_states[layer.name] = {
                    'mean': layer.moving_mean.numpy().tolist(),
                    'variance': layer.moving_variance.numpy().tolist()
                }
        with open(bn_path, 'w') as f:
            json.dump(bn_states, f)
        logger.info(f"✅ Batch norm states saved to {bn_path}")
        
    logger.info("🚀 Main function completed.")

    sdr = connect_to_server()  # Connect to HackRF device via USB
    process_continuous_stream(sdr, model, args.output_dir)

if __name__ == "__main__":
    from threading import Thread
    from web import app
    
    processing_thread = Thread(target=main)
    processing_thread.start()
    
    app.run(host='0.0.0.0', port=5001)
