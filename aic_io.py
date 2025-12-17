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

    # Support either a single array or a tuple (full_decimated, raw_complex)
    if isinstance(processed_samples, (list, tuple)) and len(processed_samples) == 2:
        full_buf, raw_buf = processed_samples
    else:
        full_buf = processed_samples
        raw_buf = None

    # Write full (decimated magnitude) FITS
    fits_path_full = os.path.join(detection_path, f'signal_full_{timestamp.strftime("%Y%m%d_%H%M%S")}.fits')
    logger.debug(f"Writing full FITS file to: {fits_path_full}")
    try:
        hdr = fits.Header()
        hdr['DATE-OBS'] = timestamp.strftime("%Y-%m-%d")
        hdr['TIME-OBS'] = timestamp.strftime("%H:%M:%S")
        hdr['FREQ-ST'] = f"{FREQ_START/1e6:.2f}"
        hdr['FREQ-END'] = f"{FREQ_START/1e6 + (FREQ_STOP-FREQ_START)/1e6:.2f}"
        hdr['TELESCOP'] = 'HackRF'
        hdr['INSTRUME'] = 'USB'
        hdr['LOCATION'] = 'Automated System'
        hdr['OBSERVER'] = 'Automated System'
        hdr['SIGNAL'] = float(np.mean(np.abs(full_buf))) if full_buf is not None else 0.0
        hdr['SAMPRATE'] = f"{FS}"
        hdr['BANDWIDT'] = f"{(FREQ_STOP - FREQ_START):e}"

        full_arr = np.abs(full_buf).astype(np.float64) if full_buf is not None else np.array([], dtype=np.float64)
        primary_hdu = fits.PrimaryHDU(full_arr, header=hdr)
        fft_data = np.abs(np.fft.fft(full_arr)) if full_arr.size else np.array([])
        fft_hdu = fits.ImageHDU(fft_data, name='FFT')
        hdul = fits.HDUList([primary_hdu, fft_hdu])
        hdul.writeto(fits_path_full, overwrite=True)
        hdul.close()
        logger.info(f"Signal full data saved to FITS file: {fits_path_full}")
    except Exception as e:
        logger.error(f"Failed to write full FITS: {e}")

    # Write raw complex snapshot FITS if available
    if raw_buf is not None:
        fits_path_raw = os.path.join(detection_path, f'signal_raw_{timestamp.strftime("%Y%m%d_%H%M%S")}.fits')
        try:
            raw_arr = np.asarray(raw_buf, dtype=np.complex64)
            hdu = fits.PrimaryHDU(raw_arr)
            hdu.header['NOTE'] = 'Raw complex snapshot (complex64)'
            hdu.header['SAMPRATE'] = f"{FS}"
            hdu.writeto(fits_path_raw, overwrite=True)
            logger.info(f"Signal raw snapshot saved to FITS file: {fits_path_raw}")
        except Exception as e:
            logger.error(f"Failed to write raw FITS: {e}")


# Modular plot registry --------------------------------------------------
PLOTTERS = []

def register_plot(name=None):
    def _decorator(func):
        PLOTTERS.append((name or func.__name__, func))
        return func
    return _decorator

def get_plotters():
    return list(PLOTTERS)

def plot_all(processed_samples, fs, output_dir, timestamp):
    """Call all registered plotters with a standard signature.

    Each plotter must accept (processed_samples, fs, output_dir, timestamp).
    """
    for name, fn in get_plotters():
        try:
            logger.debug(f"Running plotter: {name}")
            fn(processed_samples, fs, output_dir, timestamp)
        except Exception as e:
            logger.error(f"Plotter {name} failed: {e}")


def plot_from_list(processed_samples, fs, output_dir, timestamp, names):
    """Call only the plotters whose names are in `names` (list of strings)."""
    names_set = set(names or [])
    for name, fn in get_plotters():
        if name not in names_set:
            continue
        try:
            logger.debug(f"Running plotter: {name}")
            fn(processed_samples, fs, output_dir, timestamp)
        except Exception as e:
            logger.error(f"Plotter {name} failed: {e}")


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
    return filename


def plot_spectrogram(signal, sample_rate, nperseg, output_dir, timestamp, title='Spectrogram'):
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'spectrogram.png')

    nperseg = int(nperseg)
    nperseg = max(256, min(4096, nperseg))
    if nperseg > len(signal):
        nperseg = max(256, len(signal))
    noverlap = int(nperseg // 2)

    # Prefer working with real-valued power for spectrograms to avoid complex warnings
    sig_for_spec = np.abs(signal) if np.iscomplexobj(signal) else signal
    f, t, Sxx = scipy.signal.spectrogram(sig_for_spec, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
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
    return filename


def plot_signal_analysis(signal, sample_rate, output_dir, timestamp):
    # Placeholder for higher-level combined analysis
    try:
        plot_time_waveform(signal, sample_rate, output_dir, timestamp)
        plot_spectrogram(signal, sample_rate, min(4096, max(256, len(signal)//10)), output_dir, timestamp)
        plot_psd(signal, sample_rate, output_dir, timestamp)
    except Exception as e:
        logger.error(f"plot_signal_analysis failed: {e}")
    return None


def plot_psd(signal, fs, output_dir, timestamp):
    logger.debug("Plotting PSD")
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'psd.png')
    try:
        sig_for_psd = np.abs(signal) if np.iscomplexobj(signal) else signal
        sig_len = len(sig_for_psd) if sig_for_psd is not None else 0
        if sig_len < 16:
            # Not enough data to compute a meaningful PSD — emit placeholder image
            plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, f'Insufficient data for PSD (len={sig_len})', ha='center', va='center')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"PSD placeholder saved to {filename} (insufficient data)")
            return filename

        # choose nperseg no greater than available samples to avoid scipy warning
        nperseg = min(4096, sig_len)
        noverlap = nperseg // 2
        f, Pxx = scipy.signal.welch(sig_for_psd, fs=fs, nperseg=nperseg, noverlap=noverlap)
        plt.figure(figsize=(10, 6))
        plt.semilogy(f, Pxx)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.title('Power Spectral Density')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"PSD plot saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to plot PSD: {e}")


def plot_autocorrelation(signal, fs, output_dir, timestamp):
    logger.debug("Plotting autocorrelation")
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'autocorrelation.png')
    try:
        # Use magnitude (real-valued) autocorrelation to avoid complex plotting warnings
        s = np.abs(signal) if np.iscomplexobj(signal) else np.real(signal)
        n = len(s)
        corr = np.correlate(s, s, mode='full')
        corr = corr[corr.size//2:]
        lags = np.arange(0, corr.size) / fs
        plt.figure(figsize=(10, 6))
        plt.plot(lags, corr)
        plt.xlabel('Lag (s)')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Autocorrelation plot saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to plot autocorrelation: {e}")


def plot_instantaneous_frequency(signal, fs, output_dir, timestamp):
    logger.debug("Plotting instantaneous frequency")
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'inst_freq.png')
    try:
        if np.iscomplexobj(signal):
            # For complex input, derive phase directly from the complex samples
            inst_phase = np.unwrap(np.angle(signal))
        else:
            analytic = scipy.signal.hilbert(signal)
            inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase) * fs / (2.0 * np.pi)
        t = np.arange(len(inst_freq)) / fs
        plt.figure(figsize=(10, 6))
        plt.plot(t, inst_freq)
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Instantaneous Frequency')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Instantaneous frequency plot saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to plot instantaneous frequency: {e}")
        return None


@register_plot('time_waveform')
def plot_time_waveform(signal, fs, output_dir, timestamp):
    logger.debug('Plotting time-domain waveform')
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'time_waveform.png')
    try:
        sig = np.real(signal)
        t = np.arange(len(sig)) / float(fs)
        plt.figure(figsize=(12, 4))
        plt.plot(t, sig)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Time-domain Waveform (real part)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'Time waveform saved to {filename}')
        return filename
    except Exception as e:
        logger.error(f'Failed to plot time waveform: {e}')
        return None


@register_plot('constellation')
def plot_constellation(signal, fs, output_dir, timestamp, num_points=2048):
    logger.debug('Plotting IQ constellation')
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'constellation.png')
    try:
        s = signal
        if np.iscomplexobj(s):
            vals = s[:num_points]
            plt.figure(figsize=(6, 6))
            plt.scatter(np.real(vals), np.imag(vals), s=1)
            plt.xlabel('I')
            plt.ylabel('Q')
            plt.title('IQ Constellation')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f'Constellation saved to {filename}')
            return filename
        else:
            logger.warning('Constellation plot skipped: signal not complex')
            return None
    except Exception as e:
        logger.error(f'Failed to plot constellation: {e}')
        return None


@register_plot('dynamic_spectrum')
def plot_dynamic_spectrum(signal, fs, output_dir, timestamp):
    logger.debug('Plotting dynamic spectrum')
    plot_dir = os.path.join(output_dir, 'plots', timestamp.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'dynamic_spectrum.png')
    try:
        # compute spectrogram with short windows to show time evolution
        nperseg = 512
        noverlap = nperseg // 2
        f, t, Sxx = scipy.signal.spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        plt.figure(figsize=(12, 6))
        plt.imshow(Sxx_db, aspect='auto', origin='lower', extent=[t.min() if t.size else 0, t.max() if t.size else 0, f.min()/1e6 if f.size else 0, f.max()/1e6 if f.size else 0])
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (MHz)')
        plt.title('Dynamic Spectrum')
        plt.colorbar(label='Power (dB)')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f'Dynamic spectrum saved to {filename}')
        return filename
    except Exception as e:
        logger.error(f'Failed to plot dynamic spectrum: {e}')
        return None


# Wrapper plotters to adapt existing functions to the standard signature
@register_plot('spectrogram')
def _wrapper_spectrogram(sig, fs, output_dir, timestamp):
    nperseg = min(4096, max(256, len(sig)//10))
    return plot_spectrogram(sig, fs, nperseg, output_dir, timestamp)


@register_plot('psd')
def _wrapper_psd(sig, fs, output_dir, timestamp):
    return plot_psd(sig, fs, output_dir, timestamp)


@register_plot('autocorrelation')
def _wrapper_autocorr(sig, fs, output_dir, timestamp):
    return plot_autocorrelation(sig, fs, output_dir, timestamp)


@register_plot('inst_freq')
def _wrapper_inst_freq(sig, fs, output_dir, timestamp):
    # pass complex signal through so plot_instantaneous_frequency can use phase when available
    return plot_instantaneous_frequency(sig, fs, output_dir, timestamp)


@register_plot('signal_strength')
def _wrapper_signal_strength(sig, fs, output_dir, timestamp):
    # compute simple envelope/RMS over short windows for plotting
    try:
        window = 256
        if len(sig) < window:
            arr = np.abs(sig)
        else:
            arr = np.sqrt(np.convolve(np.abs(sig)**2, np.ones(window)/window, mode='valid'))
        return plot_signal_strength(arr, output_dir, timestamp)
    except Exception as e:
        logger.error(f"signal_strength wrapper failed: {e}")
        return None
