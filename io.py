import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
from astropy.io import fits
import logging

logger = logging.getLogger('aic.io')

# Try to read defaults from environment; fall back to reasonable values
FREQ_START = float(os.getenv('FREQ_START', '1420e6'))
FREQ_STOP = float(os.getenv('FREQ_STOP', '1420.4e6'))
LO = float(os.getenv('LNB_LO', '10600000000'))
FS = float(os.getenv('FS', '20000000'))


def save_fits(processed_samples, output_dir, timestamp):
    fits_dir = os.path.join(output_dir, 'fits')
    os.makedirs(fits_dir, exist_ok=True)

    filename = os.path.join(fits_dir, f'signal_{timestamp.strftime("%Y%m%d_%H%M%S")}.fits')

    data = np.array(processed_samples, dtype=np.complex64)
    hdu = fits.PrimaryHDU(data)
    hdr = hdu.header
    hdr['DATE']    = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
    hdr['TELESCOP'] = 'HackRF One'
    hdr['OBSERVER'] = 'Automated System'
    hdr['FREQ-ST'] = FREQ_START / 1e6
    hdr['FREQ-EN'] = FREQ_STOP  / 1e6
    hdr['FRQUNIT'] = 'MHz'
    hdr['LOFREQ']  = (LO/1e6)
    hdr['IFCENT']  = (FREQ_START/1e6)

    hdul = fits.HDUList([hdu])
    hdul.writeto(filename, overwrite=True)
    logger.info(f"Saved FITS: {filename}")


def save_detected_signal(processed_samples, timestamp, output_dir):
    detection_path = os.path.join(output_dir, 'detections', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(detection_path, exist_ok=True)
    
    fits_path = os.path.join(detection_path, f'signal_detection_{timestamp.strftime("%Y%m%d_%H%M%S")}.fits')
    logger.debug(f"Writing FITS file to: {fits_path}")
    
    hdr = fits.Header()
    hdr['DATE-OBS'] = timestamp.strftime("%Y-%m-%d")
    hdr['TIME-OBS'] = timestamp.strftime("%H:%M:%S")
    hdr['FREQ-ST'] = f"{FREQ_START/1e6:.2f}"
    hdr['FREQ-END'] = f"{FREQ_START/1e6 + (FREQ_STOP-FREQ_START)/1e6:.2f}"
    hdr['TELESCOP'] = 'HackRF'
    hdr['INSTRUME'] = 'USB'
    hdr['LOCATION'] = 'Automated System'
    hdr['OBSERVER'] = 'Automated System'
    hdr['SIGNAL'] = np.mean(np.abs(processed_samples))
    hdr['SAMPRATE'] = f"{FS}"
    hdr['BANDWIDT'] = f"{(FREQ_STOP - FREQ_START):e}"
    
    processed_samples = np.abs(processed_samples).astype(np.float64)
    primary_hdu = fits.PrimaryHDU(processed_samples, header=hdr)
    fft_data = np.abs(np.fft.fft(processed_samples))
    fft_hdu = fits.ImageHDU(fft_data, name='FFT')
    
    hdul = fits.HDUList([primary_hdu, fft_hdu])
    hdul.writeto(fits_path, overwrite=True)
    hdul.close()
    
    if os.path.exists(fits_path):
        logger.debug(f"FITS file successfully written: {os.path.getsize(fits_path)} bytes")
        logger.info(f"Signal data saved to FITS file: {fits_path}")


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
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'spectrogram.png')

    nperseg = int(nperseg)
    nperseg = max(256, min(4096, nperseg))
    if nperseg > len(signal):
        nperseg = max(256, len(signal))
    noverlap = int(nperseg // 2)

    f, t, Sxx = scipy.signal.spectrogram(signal, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)

    plt.figure(figsize=(12, 8))
    f_mhz = f / 1e6
    extent = [t.min() if t.size else 0, t.max() if t.size else 0, f_mhz.min() if f_mhz.size else 0, f_mhz.max() if f_mhz.size else 0]
    plt.imshow(Sxx_db, aspect='auto', origin='lower', extent=extent)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (MHz)')
    plt.colorbar(label='Power (dB)')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_signal_analysis(signal, sample_rate, output_dir, timestamp):
   plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
   os.makedirs(plot_dir, exist_ok=True)
   
   plt.figure(figsize=(12, 6))
   t = np.arange(len(signal)) / sample_rate
   plt.plot(t, np.abs(signal))
   plt.title('Signal Amplitude vs Time')
   plt.xlabel('Time (s)')
   plt.ylabel('Amplitude')
   plt.savefig(os.path.join(plot_dir, 'time_domain.png'))
   plt.close()


def plot_psd(signal, fs, output_dir, timestamp):
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    f, Pxx = scipy.signal.welch(signal, fs=fs, nperseg=4096)
    plt.figure(figsize=(10, 4))
    plt.semilogy(f/1e6, Pxx)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('PSD')
    plt.title('Power Spectral Density (Welch)')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'psd.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_autocorrelation(signal, fs, output_dir, timestamp):
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    sig = np.asarray(signal)
    n = len(sig)
    corr = np.correlate(sig - np.mean(sig), sig - np.mean(sig), mode='full')
    corr = corr[corr.size//2:]
    lags = np.arange(len(corr)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(lags, corr / np.max(np.abs(corr)))
    plt.xlabel('Lag (s)')
    plt.ylabel('Normalized Autocorrelation')
    plt.title('Autocorrelation')
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'autocorrelation.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_instantaneous_frequency(signal, fs, output_dir, timestamp):
    from scipy.signal import hilbert
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    sig = np.asarray(signal)
    try:
        analytic = hilbert(sig)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        t = np.arange(len(inst_freq)) / fs
        plt.figure(figsize=(10, 4))
        plt.plot(t, inst_freq)
        plt.xlabel('Time (s)')
        plt.ylabel('Instantaneous Frequency (Hz)')
        plt.title('Instantaneous Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'instantaneous_frequency.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception:
        return
