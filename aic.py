import logging
import numpy as np
from scipy.signal import lfilter, butter
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import os
import psutil
from astropy.io import fits
import h5py
import scipy.signal
import datetime
import socket
import json
from astropy.io import fits
from advance_signal_processing import denoise_signal, remove_dc_offset, remove_lnb_effect, process_fft
from training import generate_wow_signals, generate_pulsar_signals, generate_frb_signals, generate_hydrogen_line, load_training_data_from_folder
import cupy as cp
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
        # Input processing block
        tf.keras.layers.Conv1D(256, kernel_size=9, activation='relu', input_shape=(8192, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Dropout(0.2),
        
        # Feature extraction block
        tf.keras.layers.Conv1D(128, kernel_size=7, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Dropout(0.2),
        
        # Pattern recognition block
        tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(0.2),
        
        # Final feature processing
        tf.keras.layers.GlobalAveragePooling1D(),
        
        # Classification layers
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(-1.0))


    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    return model



def data_generator(chunk_size=8192, csv_data=None):
    if isinstance(csv_data, str):
        csv_data = load_training_data_from_folder(csv_data, chunk_size)
        
    while True:
        t = cp.arange(chunk_size, dtype=cp.float32)
        
        # Mix WOW signals with interference
        wow_signals = cp.array(generate_wow_signals(chunk_size))
        wow_signals = wow_signals.reshape(-1, chunk_size) * cp.random.uniform(0.1, 1.0, size=(wow_signals.shape[0], 1))
        wow_signals += cp.random.normal(0, 0.3, size=wow_signals.shape)
        
        # Create complex interference patterns
        interference = cp.stack([
            cp.random.normal(size=chunk_size) * (1 + 0.5 * cp.sin(2 * cp.pi * 0.03 * t)),
            cp.random.rayleigh(size=chunk_size) * cp.random.uniform(0.2, 1.8),
            cp.cumsum(cp.random.normal(size=chunk_size)) * cp.exp(-t/chunk_size) * cp.random.uniform(0.3, 1.7),
            cp.random.normal(0, cp.exp(-t/chunk_size)) * cp.random.uniform(0.4, 1.6)
        ])
        
        # Complex natural signals with mixed interference
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
        
        signals = [wow_signals]
        if csv_data is not None:
            csv_data = cp.array(csv_data).reshape(-1, chunk_size)
            signals.append(csv_data)
        signals.append(natural_signals)
        
        X = cp.vstack(signals).astype(cp.complex64).real.astype(cp.float32)
        y = cp.hstack([
            cp.ones(wow_signals.shape[0] + (csv_data.shape[0] if csv_data is not None else 0)),  # WOW signals + CSV data are 1
            cp.zeros(natural_signals.shape[0])  # Natural signals/noise are 0
        ])
        
        X = cp.reshape(X, (-1, chunk_size, 1))
        y = cp.reshape(y, (-1,))
        
        yield cp.asnumpy(X), cp.asnumpy(y)


def train_model(model, csv_data):
    history = model.fit(
        data_generator(chunk_size=8192, csv_data=csv_data),
        steps_per_epoch=72,
        epochs=1000,
        batch_size=4  # Increased from 4
    )
    state_dir = os.path.join('models', 'full_state')
    os.makedirs(state_dir, exist_ok=True)
    model.save(os.path.join(state_dir, 'full_model.keras'))
    
    web_data = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'status': 'Training in progress...',
    }
    app.config['LATEST_DATA'] = web_data
    
    bn_states = {}
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_states[layer.name] = {
                'mean': layer.moving_mean.numpy().tolist(),
                'variance': layer.moving_variance.numpy().tolist()
            }
    
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


def predict_signal(model, samples):
    logger.debug("Predicting signal")
    reshaped_samples = np.reshape(samples, (1, 8192, 1))
    confidence = model.predict(reshaped_samples, verbose=0)
    confidence_value = float(confidence[0][0])
    logger.debug(f"Detection confidence: {confidence_value}")
    return (confidence_value >= 0.75, confidence_value)

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

def retry_connection(func):
    """Decorator for retrying connections with exponential backoff"""
    def wrapper(*args, **kwargs):
        import time
        max_attempts = 10
        base_delay = 1
        
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except socket.timeout:
                logger.error(f"Connection timed out (Attempt {attempt + 1}/{max_attempts})")
            except socket.gaierror:
                logger.error(f"DNS lookup failed (Attempt {attempt + 1}/{max_attempts})")
            except ConnectionRefusedError:
                logger.error(f"Connection refused (Attempt {attempt + 1}/{max_attempts})")
            except ConnectionResetError:
                logger.error(f"Connection reset by peer (Attempt {attempt + 1}/{max_attempts})")
            except ConnectionAbortedError:
                logger.error(f"Connection aborted (Attempt {attempt + 1}/{max_attempts})")
            except OSError as e:
                logger.error(f"OS error: {e} (Attempt {attempt + 1}/{max_attempts})")
            except socket.error as e:
                logger.error(f"Socket error: {e} (Attempt {attempt + 1}/{max_attempts})")
            except Exception as e:
                logger.error(f"Unexpected error: {e} (Attempt {attempt + 1}/{max_attempts})")
            
            if attempt < max_attempts - 1:
                delay = min(300, base_delay * (2 ** attempt))
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
        
        raise ConnectionError("Maximum retry attempts reached")    
    return wrapper

@retry_connection
def connect_to_server(host, port):
    """Connect to server while maintaining CLI argument values"""
    logger.debug(f"Connecting to server at {host}:{port}")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(30)
    client_socket.connect((host, port))
    
    tuning_parameters = {
        'start_freq': 1420e6,
        'end_freq': 1420.4e6,
        'sample_rate': fs,
        'gain': 20,
        'duration_seconds': 0
    }
    
    client_socket.sendall(json.dumps(tuning_parameters).encode())
    logger.info(f"Successfully connected to server at {host}:{port}")
    return client_socket

def process_continuous_stream(client_socket, model, output_dir, lnb_band=LNB_HIGH,host=None,port=None):
    import sys
    import select
    import termios
    import tty
    from scipy import signal
    connection_host = host or client_socket.getpeername()[0]
    connection_port = port or client_socket.getpeername()[1]
    def is_key_pressed():
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key
        return None
    
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        logger.debug(f"Starting continuous stream processing using LNB band: {lnb_band}")
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
                
                chunk = client_socket.recv(chunk_size * 4)
                if not chunk:
                    consecutive_empty_reads += 1
                    logger.warning(f"No samples received. Empty read count: {consecutive_empty_reads}")
                    
                if consecutive_empty_reads >= max_empty_reads:
                    logger.info("Connection appears dead, initiating reconnection...")
                    client_socket.close()
                    client_socket = connect_to_server(connection_host, connection_port)
                    consecutive_empty_reads = 0
                    continue

                consecutive_empty_reads = 0
                samples = np.frombuffer(chunk, dtype=np.complex64)
                data_buffer = np.append(data_buffer, samples)

                while len(data_buffer) >= chunk_size:
                    process_chunk = data_buffer[:chunk_size]
                    processed_samples = remove_dc_offset(process_chunk)
                    processed_samples = remove_lnb_effect(process_chunk, fs, notch_freq, notch_width, lnb_band)
                    processed_samples = denoise_signal(processed_samples)

                    if inject_wow:
                        processed_samples = inject_wow_signal(processed_samples)
                        inject_wow = False
                    
                    prediction_samples = np.abs(processed_samples)
                    signal_strength = 20 * np.log10(np.mean(np.abs(processed_samples)) + 1e-10)

                    fft_magnitude, fft_freq, fft_data, fft_phase, fft_power = process_fft(cp.asarray(processed_samples), chunk_size, fs)

                    detection, confidence_value = predict_signal(model, prediction_samples)
                    
                    web_data = {
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'frequency': f"{freq_start/1e6:.2f} MHz",
                        'signal_strength': f"{signal_strength:.2f} dB",
                        'status': f"Active - Detection Confidence: {confidence_value*100:.2f}%",
                        'processed_samples': processed_samples.tolist(),
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
                client_socket.close()
                client_socket = connect_to_server(connection_host, connection_port)
                continue
            except KeyboardInterrupt:
                logger.info("Stream processing interrupted by user")
                break
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                client_socket.close()
                client_socket = connect_to_server(connection_host, connection_port)
                continue
    
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        client_socket.close()





def save_detected_signal(processed_samples, timestamp, output_dir, freq_start=1420e6, freq_end=1420.4e6):
    detection_path = os.path.join(output_dir, 'detections', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(detection_path, exist_ok=True)
    
    fits_path = os.path.join(detection_path, f'signal_detection_{timestamp.strftime("%Y%m%d_%H%M%S")}.fits')
    logger.debug(f"Writing FITS file to: {fits_path}")
    
    # Create detailed FITS header
    hdr = fits.Header()
    hdr['DATE-OBS'] = timestamp.strftime("%Y-%m-%d")
    hdr['TIME-OBS'] = timestamp.strftime("%H:%M:%S")
    hdr['FREQ-ST'] = f"{freq_start/1e6:.2f}"
    hdr['FREQ-END'] = f"{freq_end/1e6:.2f}"
    hdr['TELESCOP'] = 'HackRF'
    hdr['INSTRUME'] = 'LNB-High'
    hdr['LOCATION'] = 'Your Observatory Location'
    hdr['OBSERVER'] = 'Your Name'
    hdr['SIGNAL'] = np.mean(np.abs(processed_samples))
    hdr['SAMPRATE'] = f"{fs}"
    hdr['BANDWIDT'] = f"{freq_end - freq_start}"
    
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
        # Set threading configuration at startup
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    

    parser = argparse.ArgumentParser(description='Process continuous SDR stream.')
    parser.add_argument('-o', '--output-dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--host', type=str, default='localhost', help='Server host address')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--band', choices=['low', 'high'], default='low', help='LNB band selection')
    parser.add_argument('--path', type=str, default='data', help='Path to csv file')
    args = parser.parse_args()
    HOST_H = args.host
    PORT_P = args.port
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    lnb_band = LNB_LOW if args.band == 'low' else LNB_HIGH
    logger.debug(f"Selected LNB band: {lnb_band}")
    
    state_dir = os.path.join('models', 'full_state')
    if os.path.exists(os.path.join(state_dir, 'full_model.keras')):
        model = tf.keras.models.load_model(os.path.join(state_dir, 'full_model.keras'))
        with open(os.path.join(state_dir, 'bn_states.json'), 'r') as f:
            bn_states = json.load(f)
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.moving_mean.assign(np.array(bn_states[layer.name]['mean']))
                layer.moving_variance.assign(np.array(bn_states[layer.name]['variance']))
    else:
        model = create_model()
        train_model(model, args.path)
    
    client_socket = connect_to_server(HOST_H, PORT_P)
    process_continuous_stream(client_socket, model, args.output_dir, lnb_band,HOST_H,PORT_P)


if __name__ == "__main__":
    from threading import Thread
    from web import app
    
    processing_thread = Thread(target=main)
    processing_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
