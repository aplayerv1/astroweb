import numpy as np
import cupy as cp
import pywt
from scipy import signal
from scipy.signal import lfilter, butter, filtfilt

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

    # Perform wavelet decomposition and reconstruction
    coeffs = pywt.wavedec(data_cpu, wavelet, level=level)
    denoised_signal = pywt.waverec(coeffs, wavelet)
    
    # Ensure length matches original
    denoised_signal = denoised_signal[:len(data_cpu)]
    
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
    fs = 20e6
    cutoff = 1000  # Lowered cutoff from 10 kHz to 1 kHz
    nyquist = fs / 2
    b, a = butter(8, cutoff / nyquist, btype='high')

    if isinstance(signal_data, cp.ndarray):
        signal_cpu = cp.asnumpy(signal_data)
        signal_cpu = np.nan_to_num(signal_cpu)
        filtered = filtfilt(b, a, signal_cpu)
        return cp.asarray(filtered)
    else:
        signal_data = np.nan_to_num(signal_data)
        return filtfilt(b, a, signal_data)

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
    # Convert to CPU if using GPU
    if isinstance(samples, cp.ndarray):
        samples = cp.asnumpy(samples)
    
    nyquist = 0.5 * fs
    lowcut = (notch_freq - notch_width / 2) / nyquist
    highcut = (notch_freq + notch_width / 2) / nyquist

    b, a = signal.butter(4, [lowcut, highcut], btype='bandstop')
    filtered_samples = lfilter(b, a, samples)

    t = np.arange(len(filtered_samples)) / fs
    freq_offset = lnb_band - 1420e6
    translated_samples = filtered_samples * np.exp(2j * np.pi * freq_offset * t)

    return translated_samples

def process_fft(samples, chunk_size, fs):
    """
    Perform FFT on the input signal using GPU acceleration if available.
    
    Parameters:
        samples (np.ndarray or cp.ndarray): The input signal.
        chunk_size (int): Size of the chunk for processing.
        fs (float): Sampling frequency.
        
    Returns:
        tuple: FFT magnitude, frequency, data, phase, and power arrays.
    """
    # Handle GPU processing
    if isinstance(samples, cp.ndarray):
        window = cp.kaiser(chunk_size, beta=14)
        centered_samples = samples - cp.mean(samples)
        windowed_samples = centered_samples * window

        fft_signal = cp.fft.fft(windowed_samples)
        
        # Remove DC and high frequency components
        n_remove = max(1, int(chunk_size * 0.005))  # 0.5%
        fft_signal[0:n_remove] = 0
        fft_signal[-n_remove:] = 0

        # Calculate FFT components
        fft_magnitude = cp.abs(fft_signal)
        fft_magnitude = fft_magnitude / cp.max(fft_magnitude)  # Normalize
        fft_freq = cp.fft.fftshift(cp.fft.fftfreq(chunk_size, 1/fs))
        fft_data = cp.fft.fftshift(cp.abs(fft_signal))
        fft_phase = cp.fft.fftshift(cp.angle(fft_signal))
        
        # Calculate power spectrum
        power = cp.abs(fft_signal)**2
        normalized_power = power / cp.max(power)
        fft_power = cp.fft.fftshift(normalized_power)

        # Convert to NumPy for consistent output
        return (
            cp.asnumpy(fft_magnitude), 
            cp.asnumpy(fft_freq), 
            cp.asnumpy(fft_data), 
            cp.asnumpy(fft_phase), 
            cp.asnumpy(fft_power)
    )
    # Handle CPU processing
    else:
        window = np.kaiser(chunk_size, beta=14)
        centered_samples = samples - np.mean(samples)
        windowed_samples = centered_samples * window

        fft_signal = np.fft.fft(windowed_samples)
        
        # Remove DC and high frequency components
        n_remove = max(1, int(chunk_size * 0.005))  # 0.5%
        fft_signal[0:n_remove] = 0
        fft_signal[-n_remove:] = 0

        # Calculate FFT components
        fft_magnitude = np.abs(fft_signal)
        fft_magnitude = fft_magnitude / np.max(fft_magnitude)  # Normalize
        fft_freq = np.fft.fftshift(np.fft.fftfreq(chunk_size, 1/fs))
        fft_data = np.fft.fftshift(np.abs(fft_signal))
        fft_phase = np.fft.fftshift(np.angle(fft_signal))
        
        # Calculate power spectrum
        power = np.abs(fft_signal)**2
        normalized_power = power / np.max(power)
        fft_power = np.fft.fftshift(normalized_power)

        return fft_magnitude, fft_freq, fft_data, fft_phase, fft_power