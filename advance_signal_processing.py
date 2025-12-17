import numpy as np
try:
    import cupy as cp
    USE_CUPY = True
except Exception:
    cp = np
    USE_CUPY = False
import os
import pywt
from scipy import signal
from scipy.signal import lfilter, butter, filtfilt, sosfiltfilt
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
# Module-level LNB handling flag. Can be toggled at runtime by callers.
LNB_ENABLED = os.getenv('LNB_OFF', '').strip().lower() not in ('1', 'true', 'yes')

def set_lnb_enabled(enabled: bool):
    """Enable or disable LNB-specific processing (frequency translation / notch).

    Callers (e.g. `aic2.py`) can use this to disable LNB handling when the LNB
    is unpowered. When disabled, `remove_lnb_effect()` returns the input unchanged.
    """
    global LNB_ENABLED
    LNB_ENABLED = bool(enabled)
    logger.info(f"advance_signal_processing: LNB handling enabled={LNB_ENABLED}")

def is_lnb_enabled():
    return LNB_ENABLED
# Constants
def denoise_signal(data, wavelet='db4', level=3):
    """
    Apply wavelet denoising to the input signal using a band-pass filter.
    
    Parameters:
        data (np.ndarray or cp.ndarray): The input signal.
        wavelet (str): The wavelet type for denoising. Default is 'db4'.
        level (int): The decomposition level. Default is 3.
        
    Returns:
        np.ndarray or cp.ndarray: Denoised signal.
    """
    using_gpu = isinstance(data, cp.ndarray)
    if using_gpu:
        data_cpu = cp.asnumpy(data)
    else:
        data_cpu = data

    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data_cpu, wavelet, level=level)
    # Estimate noise sigma from the finest scale detail coefficients
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745 if len(detail_coeffs) > 0 else 0.0
    # Universal threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(data_cpu))) if sigma > 0 else 0.0
    # Soft threshold detail coefficients (leave approximation coeffs untouched)
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    # Ensure length matches original
    denoised_signal = denoised_signal[:len(data_cpu)]
    
    if using_gpu:
        return cp.asarray(denoised_signal)
    return denoised_signal

def denoise_preserve_spikes(data, wavelet='db4', level=3, spike_threshold=5):
    """
    Denoise the signal while preserving narrowband transient spikes (e.g., WOW! signals).

    Parameters:
        data (np.ndarray or cp.ndarray): Input signal.
        wavelet (str): Wavelet type for denoising.
        level (int): Decomposition level for wavelet.
        spike_threshold (float): Threshold in std deviations to protect spikes.

    Returns:
        np.ndarray or cp.ndarray: Denoised signal with spikes preserved.
    """
    using_gpu = isinstance(data, cp.ndarray)
    if using_gpu:
        data_cpu = cp.asnumpy(data)
    else:
        data_cpu = data

    # Compute moving standard deviation efficiently using convolution
    n = len(data_cpu)
    window_size = max(3, n // 1000)
    if window_size >= n:
        window_size = max(3, n // 10)
    # moving mean and mean of squares
    kernel = np.ones(window_size, dtype=float) / float(window_size)
    mean = np.convolve(np.abs(data_cpu), kernel, mode='same')
    mean_sq = np.convolve(np.abs(data_cpu)**2, kernel, mode='same')
    var = np.maximum(0.0, mean_sq - mean**2)
    rolling_std = np.sqrt(var)

    # Identify potential spikes (guard small std)
    guard = np.maximum(rolling_std, 1e-12)
    spikes_mask = (np.abs(data_cpu) > spike_threshold * guard).astype(float)

    # Wavelet denoising on the whole signal
    coeffs = pywt.wavedec(data_cpu, wavelet, level=level)
    detail = coeffs[-1] if len(coeffs) > 1 else np.array([])
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(len(data_cpu))) if sigma > 0 else 0.0
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    denoised_signal = denoised_signal[:len(data_cpu)]

    # Restore spikes to original values
    denoised_signal[spikes_mask > 0] = data_cpu[spikes_mask > 0]

    if using_gpu:
        return cp.asarray(denoised_signal)
    return denoised_signal



def remove_dc_offset(signal_data):
    """
    Remove the DC offset from the signal using a high-pass filter.
    
    Parameters:
        signal_data (np.ndarray or cp.ndarray): The input signal.
        
    Returns:
        np.ndarray or cp.ndarray: Signal with removed DC offset.
    """
    # Allow caller to override sample rate or cutoff by passing attributes on the array
    fs = getattr(signal_data, 'fs', 20e6)
    cutoff = 1000  # default cutoff 1 kHz
    nyquist = fs / 2
    # Use second-order-sections for numerical stability
    sos = signal.butter(8, cutoff / nyquist, btype='high', output='sos')

    if isinstance(signal_data, cp.ndarray):
        arr = cp.asnumpy(signal_data)
        arr = np.nan_to_num(arr)
        # Apply filter separately to real and imag parts for complex signals
        if np.iscomplexobj(arr):
            real = sosfiltfilt(sos, np.real(arr))
            imag = sosfiltfilt(sos, np.imag(arr))
            out = real + 1j * imag
        else:
            out = sosfiltfilt(sos, arr)
        return cp.asarray(out)
    else:
        arr = np.nan_to_num(signal_data)
        if np.iscomplexobj(arr):
            real = sosfiltfilt(sos, np.real(arr))
            imag = sosfiltfilt(sos, np.imag(arr))
            return real + 1j * imag
        return sosfiltfilt(sos, arr)

def remove_lnb_effect(samples, fs, notch_freq, notch_width, lnb_band):
    """
    Remove LNB effects and interference from samples using a band-stop filter followed by frequency translation.
    
    Parameters:
        samples (np.ndarray or cp.ndarray): The input signal.
        fs (float): Sampling frequency.
        notch_freq (float): Center frequency of the notch filter.
        notch_width (float): Width of the notch filter.
        lnb_band (float): LNB band center frequency.
        
    Returns:
        np.ndarray or cp.ndarray: Signal with LNB effects removed.
    """
    # If LNB handling is disabled globally, return samples unchanged
    if not LNB_ENABLED:
        logger.debug('remove_lnb_effect: LNB handling disabled; returning original samples')
        return samples

    # Keep track of whether input was a CuPy array so we can return the same type
    using_gpu = isinstance(samples, cp.ndarray)
    try:
        xp = cp if using_gpu else np
        arr = xp.asnumpy(samples) if using_gpu else np.asarray(samples)

        filtered = arr
        nyquist = 0.5 * fs
        if notch_width and notch_width > 0 and notch_freq and 0.0 < notch_freq < nyquist:
            lowcut = (notch_freq - notch_width / 2) / nyquist
            highcut = (notch_freq + notch_width / 2) / nyquist
            lowcut = max(1e-6, min(0.999, lowcut))
            highcut = max(lowcut + 1e-6, min(0.999, highcut))
            sos = signal.butter(4, [lowcut, highcut], btype='bandstop', output='sos')
            try:
                filtered = sosfiltfilt(sos, arr)
            except Exception:
                # fallback to filtfilt on SOS if available
                b, a = signal.butter(4, [lowcut, highcut], btype='bandstop')
                filtered = filtfilt(b, a, arr)

        # Optional frequency translation using xp arrays for GPU support
        translated = filtered
        try:
            if lnb_band is not None and lnb_band != 0:
                freq_offset = float(lnb_band) - 1420e6
                t = xp.arange(len(filtered)) / float(fs)
                translated = filtered * xp.exp(-2j * xp.pi * freq_offset * t)
        except Exception:
            translated = filtered

        return cp.asarray(translated) if using_gpu else translated
    except Exception as e:
        logger.warning(f"remove_lnb_effect failed ({e}), returning original samples")
        return samples

def process_fft(samples, chunk_size, fs):
    """
    Compute FFT magnitude, frequency, phase, and normalized power robustly.
    Supports CPU (numpy) and GPU (cupy).
    Eliminates central black line caused by hard-zeroing DC bins.
    """
    import numpy as np
    try:
        import cupy as cp
    except Exception:
        cp = None

    logger.debug(f"Processing FFT: samples_shape={getattr(samples,'shape',None)}, chunk_size={chunk_size}, fs={fs}")

    # Normalize input to numpy/cupy array of length chunk_size (pad or trim)
    is_gpu = cp is not None and isinstance(samples, cp.ndarray)
    xp = cp if is_gpu else np

    # Ensure 1D
    try:
        arr = samples
        if is_gpu:
            if arr.ndim != 1:
                arr = arr.ravel()
            if arr.size < chunk_size:
                pad = chunk_size - arr.size
                arr = xp.concatenate([arr, xp.zeros(pad, dtype=arr.dtype)])
            elif arr.size > chunk_size:
                arr = arr[:chunk_size]
        else:
            arr = np.asarray(samples).ravel()
            if arr.size < chunk_size:
                arr = np.pad(arr, (0, chunk_size - arr.size), mode='constant')
            elif arr.size > chunk_size:
                arr = arr[:chunk_size]
    except Exception as e:
        logger.error(f"process_fft input normalization failed: {e}")
        arr = np.zeros(chunk_size)
        xp = np
        is_gpu = False

    if arr.size == 0:
        logger.error("Empty samples array passed to process_fft after normalization")
        empty = np.zeros(chunk_size)
        return empty, empty, empty, empty, empty

    # Window and center
    window = xp.kaiser(chunk_size, beta=14)
    centered = arr - xp.mean(arr)
    windowed = centered * window

    # FFT (complex)
    fft_signal = xp.fft.fft(windowed)

    # Softly reduce edge bins to avoid artifacts
    n_remove = max(1, int(chunk_size * 0.005))
    max_val = float(xp.max(xp.abs(fft_signal))) if fft_signal.size else 0.0
    if max_val > 1e-12:
        fft_signal[0:n_remove] *= 0.2
        fft_signal[-n_remove:] *= 0.2

    # Magnitude and power
    fft_magnitude = xp.abs(fft_signal) + 1e-12
    denom = float(xp.max(fft_magnitude)) if fft_magnitude.size else 1.0
    if denom == 0:
        denom = 1.0
    fft_magnitude = fft_magnitude / denom
    fft_power = fft_magnitude**2

    # Frequency and phase (shifted for visualization)
    fft_freq = xp.fft.fftshift(xp.fft.fftfreq(chunk_size, 1.0/fs))
    fft_phase = xp.fft.fftshift(xp.angle(fft_signal))

    # Shift magnitude & power for visualization
    fft_magnitude = xp.fft.fftshift(fft_magnitude)
    fft_power = xp.fft.fftshift(fft_power)

    # Convert back to NumPy for consumers
    if is_gpu:
        fft_magnitude = cp.asnumpy(fft_magnitude)
        fft_freq = cp.asnumpy(fft_freq)
        fft_signal_out = cp.asnumpy(fft_signal)
        fft_phase = cp.asnumpy(fft_phase)
        fft_power = cp.asnumpy(fft_power)
    else:
        fft_magnitude = np.asarray(fft_magnitude)
        fft_freq = np.asarray(fft_freq)
        fft_signal_out = np.asarray(fft_signal)
        fft_phase = np.asarray(fft_phase)
        fft_power = np.asarray(fft_power)

    logger.debug(f"FFT computed: freq_len={len(fft_freq)}, mag_len={len(fft_magnitude)}, power_len={len(fft_power)}")
    # Return magnitude, frequency, complex FFT, phase, and power
    return fft_magnitude, fft_freq, fft_signal_out, fft_phase, fft_power
