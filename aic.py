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

def remove_lnb_effect(samples, fs, notch_freq, notch_width, lnb_band=LNB_LOW):
    """Remove LNB effects and interference from samples"""
    logger.debug(f"Removing LNB effect using band {lnb_band}")
    nyquist = 0.5 * fs
    lowcut = (notch_freq - notch_width / 2) / nyquist
    highcut = (notch_freq + notch_width / 2) / nyquist

    # Apply bandstop filter to remove interference
    b, a = butter(4, [lowcut, highcut], btype='bandstop')
    filtered_samples = lfilter(b, a, samples)

    # Apply frequency translation based on LNB band
    t = np.arange(len(filtered_samples)) / fs
    freq_offset = lnb_band - 1420e6
    translated_samples = filtered_samples * np.exp(2j * np.pi * freq_offset * t)

    return translated_samples


def apply_bandpass_filter(samples, fs, low_cutoff, high_cutoff):
    logger.debug("Applying bandpass filter")
    nyquist = 0.5 * fs
    lowcut = low_cutoff / nyquist
    highcut = high_cutoff / nyquist

    b, a = butter(4, [lowcut, highcut], btype='band')
    filtered_samples = lfilter(b, a, samples)
    return filtered_samples

def create_model():
    logger.debug("Creating model")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(8192, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def data_generator(chunk_size=8192):
    while True:
        # Base phenomena patterns
        wow_signal = np.sin(np.linspace(0, 2*np.pi, chunk_size))
        pulsar = np.zeros(chunk_size)
        frb = np.zeros(chunk_size)
        
        # Create pulsar signal with varying periods
        pulse_periods = [chunk_size//8, chunk_size//4, chunk_size//6]
        for period in pulse_periods:
            pulse_locations = np.arange(0, chunk_size-50, period)
            for loc in pulse_locations:
                pulsar[loc:loc+50] = np.exp(-np.linspace(0, 3, 50))
        
        # Create Fast Radio Burst with random location
        burst_loc = np.random.randint(0, chunk_size-100)
        frb[burst_loc:burst_loc+100] = 2.0 * np.exp(-np.linspace(0, 5, 100))
        
        # Create frequency-shifting signal
        t = np.arange(chunk_size)
        freq_shift = np.sin(2*np.pi*0.001*t) * np.sin(2*np.pi*0.01*t)
        
        # Generate noise with different characteristics
        gaussian_noise = np.random.normal(size=(8, chunk_size))
        colored_noise = np.cumsum(np.random.normal(size=(8, chunk_size)), axis=1)
        
        # Combine signals with variations
        signals = [
            wow_signal,
            pulsar,
            frb,
            freq_shift,
            wow_signal * (1 + 0.3*np.sin(2*np.pi*0.005*t)),
            pulsar + 0.5*freq_shift,
            frb * (1 + 0.2*np.sin(2*np.pi*0.002*t))
        ]
        
        X = np.vstack([signals, gaussian_noise, colored_noise])
        y = np.hstack([np.ones(len(signals)), np.zeros(16)])
        X = np.reshape(X, (-1, chunk_size, 1))
        
        yield X, y



def train_model(model):
    logger.debug("Training model with generator")
    chunk_size = 8192  # 8K samples per chunk
    steps_per_epoch = 72  # To represent 72 seconds worth of data
    
    history = model.fit(
        data_generator(chunk_size),
        steps_per_epoch=steps_per_epoch,
        epochs=100,
        batch_size=8
    )
    weights_path = os.path.join('models', 'signal_classifier.weights.h5')
    os.makedirs('models', exist_ok=True)
    model.save_weights(weights_path)
    logger.info(f"Model weights saved to {weights_path}")
    return history


def load_model_weights(model, weights_file):
    weights_path = os.path.join('models', weights_file)
    logger.debug(f"Loading model weights from {weights_path}")
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        logger.info(f"Successfully loaded weights from {weights_path}")
    else:
        logger.error(f"Model weights file {weights_path} does not exist")

def predict_signal(model, samples):
    logger.debug("Predicting signal")
    reshaped_samples = np.reshape(samples, (1, 8192, 1))
    confidence = model.predict(reshaped_samples, verbose=1)
    logger.debug(f"Detection confidence: {confidence[0][0]}")
    return confidence[0][0] >= 0.85  # Lower threshold for better detection

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

def connect_to_server(host='localhost', port=8888, max_retries=None):
    """Connect to SDR server with constant retry"""
    import time
    retry_count = 0
    while max_retries is None or retry_count < max_retries:
        try:
            logger.debug(f"Attempting connection to server at {host}:{port}")
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            
            # Configure continuous streaming
            tuning_parameters = {
                'start_freq': 1420e6,
                'end_freq': 1420.4e6,
                'sample_rate': fs,
                'gain': 20,
                'duration_seconds': 0
            }
            
            client_socket.sendall(json.dumps(tuning_parameters).encode())
            logger.info("Successfully connected to server")
            return client_socket
            
        except (ConnectionRefusedError, socket.error) as e:
            retry_count += 1
            logger.warning(f"Connection attempt {retry_count} failed: {e}")
            time.sleep(5)  # Wait 5 seconds between retries
            
    logger.error("Max retries reached, could not connect to server")
    raise ConnectionError("Failed to establish connection after maximum retries")


def process_continuous_stream(client_socket, model, output_dir, lnb_band=LNB_HIGH):
    logger.debug(f"Starting continuous stream processing using LNB band: {lnb_band}")
    data_buffer = np.array([], dtype=np.complex64)
    chunk_size = 8192
    
    while True:
        try:
            chunk = client_socket.recv(chunk_size * 4)
            if not chunk:
                logger.info("No more data received, ending stream processing")
                break

            samples = np.frombuffer(chunk, dtype=np.complex64)
            data_buffer = np.append(data_buffer, samples)

            while len(data_buffer) >= chunk_size:
                process_chunk = data_buffer[:chunk_size]
                processed_samples = remove_lnb_effect(process_chunk, fs, notch_freq, notch_width, lnb_band)
                
                # processed_samples = inject_wow_signal(processed_samples)
                
                prediction_samples = np.abs(processed_samples)
                signal_strength = np.mean(np.abs(processed_samples))
                signal_no_dc = processed_samples - np.mean(processed_samples)
                fft_magnitude = np.abs(np.fft.fft(signal_no_dc))
                
                web_data = {
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'frequency': f"{freq_start/1e6:.2f} MHz",
                    'signal_strength': f"{signal_strength:.2f} dB",
                    'status': 'Active',
                    'processed_samples': processed_samples.tolist(),
                    'fft_data': np.abs(np.fft.fft(processed_samples)).tolist(),
                    'fft_freq': np.fft.fftfreq(chunk_size, 1/fs).tolist(),
                    'fft_magnitude': fft_magnitude.tolist(),
                    'fft_phase': np.angle(np.fft.fft(processed_samples)).tolist(),
                    'fft_power': (np.abs(np.fft.fft(processed_samples))**2).tolist(),
                    'fft_power_db': (10 * np.log10(np.abs(np.fft.fft(processed_samples))**2)).tolist(),
                    'fft_power_db_normalized': (10 * np.log10(np.abs(np.fft.fft(processed_samples))**2 / np.max(np.abs(np.fft.fft(processed_samples))**2))).tolist()
                }

                
                app.config['LATEST_DATA'] = web_data
                
                
                if predict_signal(model, prediction_samples):
                    logger.info("WOW Signal detected!")
                    timestamp = datetime.datetime.now()
                    save_detected_signal(processed_samples, timestamp, output_dir)
                    plot_signal_strength(processed_samples, output_dir,timestamp)
                    plot_signal_analysis(processed_samples, fs, output_dir, timestamp)
                    plot_spectrogram(processed_samples, fs, chunk_size, output_dir, timestamp)

                
                data_buffer = data_buffer[chunk_size:]

        except KeyboardInterrupt:
            logger.info("Stream processing interrupted by user")
            break
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
            break
    
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



def inject_wow_signal(samples, amplitude=4.0):
    """Inject a wow signal pattern with correct data type"""
    t = np.arange(len(samples))
    base_signal = amplitude * np.sin(2 * np.pi * 0.01 * t)
    modulation = np.sin(2 * np.pi * 0.001 * t)
    wow_signal = base_signal * (1 + 0.5 * modulation)
    
    return samples.astype(np.complex64) + wow_signal.astype(np.complex64)


def main():
    parser = argparse.ArgumentParser(description='Process continuous SDR stream.')
    parser.add_argument('-o', '--output-dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--host', type=str, default='localhost', help='Server host address')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--band', choices=['low', 'high'], default='low', help='LNB band selection')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    lnb_band = LNB_LOW if args.band == 'low' else LNB_HIGH
    logger.debug(f"Selected LNB band: {lnb_band}")
    
    weights_path = os.path.join('models', 'signal_classifier.weights.h5')
    model = create_model()
    
    if os.path.exists(weights_path):
        load_model_weights(model, weights_path)
    else:
        train_model(model)
    
    client_socket = connect_to_server(args.host, args.port)
    process_continuous_stream(client_socket, model, args.output_dir, lnb_band)

if __name__ == "__main__":
    from threading import Thread
    from web import app
    
    processing_thread = Thread(target=main)
    processing_thread.start()
    
    app.run(host='0.0.0.0', port=5000)
