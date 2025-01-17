import numpy as np
import cupy as cp
import multiprocessing as mp
from multiprocessing import Queue
import logging
from numba import cuda
import numba
from scipy import signal, ndimage
from scipy.signal import lfilter, medfilt, butter, filtfilt
import pywt
from PyEMD import EMD
from sklearn.decomposition import FastICA
# from py_kernel.kernels import phase_locked_loop_gpu
from cupyx.scipy import signal as cp_signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


def denoise_signal(data, wavelet='db1', level=1):
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    
    if len(data.shape) != 1:
        raise ValueError("Input data must be a 1D array.")

    coeffs = pywt.wavedec(data, wavelet, level=level)

    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))

    new_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    denoised_signal = pywt.waverec(new_coeffs, wavelet)

    return cp.asarray(denoised_signal) if isinstance(data, cp.ndarray) else denoised_signal

def remove_dc_offset(signal):
    if isinstance(signal, cp.ndarray):
        signal = cp.nan_to_num(signal)
        fs = 20e6  # Sample rate
        cutoff = 10000  # Increased cutoff to 10kHz
        nyquist = fs/2
        b, a = butter(8, cutoff/nyquist, btype='high')  # Increased order to 8
        filtered_signal = filtfilt(b, a, cp.asnumpy(signal))
        return cp.asarray(filtered_signal)
    else:
        signal = np.nan_to_num(signal)
        fs = 20e6
        cutoff = 10000
        nyquist = fs/2
        b, a = butter(8, cutoff/nyquist, btype='high')
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal

    
def remove_lnb_effect(samples, fs, notch_freq, notch_width, lnb_band):
    """Remove LNB effects and interference from samples"""
    logging.debug(f"Removing LNB effect using band {lnb_band}")
    nyquist = 0.5 * fs
    lowcut = (notch_freq - notch_width / 2) / nyquist
    highcut = (notch_freq + notch_width / 2) / nyquist

    # Apply bandstop filter to remove interference
    b, a = signal.butter(4, [lowcut, highcut], btype='bandstop')
    filtered_samples = lfilter(b, a, samples)

    # Apply frequency translation based on LNB band
    t = np.arange(len(filtered_samples)) / fs
    freq_offset = lnb_band - 1420e6
    translated_samples = filtered_samples * np.exp(2j * np.pi * freq_offset * t)

    return translated_samples

def process_fft(samples, chunk_size, fs):
    if isinstance(samples, cp.ndarray):
        # GPU processing
        window = cp.kaiser(chunk_size, beta=14)
        centered_samples = samples - cp.mean(samples)
        windowed_samples = centered_samples * window

        fft_signal = cp.fft.fft(windowed_samples)
        
        # Keep only non-zero frequency components
        start_idx = int(chunk_size * 0.01)  # Adjust this value to control cutoff
        end_idx = chunk_size - start_idx
        fft_signal = fft_signal[start_idx:end_idx]
        
        # Calculate FFT components for non-zero region
        fft_magnitude = cp.abs(fft_signal) / cp.max(cp.abs(fft_signal))
        fft_freq = cp.fft.fftshift(cp.fft.fftfreq(len(fft_signal), 1/fs))
        fft_data = cp.fft.fftshift(cp.abs(fft_signal))
        fft_phase = cp.fft.fftshift(cp.angle(fft_signal))
        
        power = cp.abs(fft_signal)**2
        normalized_power = power / cp.max(power)
        fft_power = cp.fft.fftshift(normalized_power)

        return cp.asnumpy(fft_magnitude), cp.asnumpy(fft_freq), cp.asnumpy(fft_data), cp.asnumpy(fft_phase), cp.asnumpy(fft_power)


def process_fft_cpu(samples, chunk_size,fs):
    window = np.kaiser(chunk_size, beta=14)
    centered_samples = samples - np.mean(samples)
    windowed_samples = centered_samples * window

    fft_signal = np.fft.fft(windowed_samples)
    fft_signal[0:int(chunk_size*0.005)] = 0
    fft_signal[-int(chunk_size*0.005):] = 0

    fft_magnitude = np.abs(fft_signal) / np.max(np.abs(fft_signal))
    fft_freq = np.fft.fftshift(np.fft.fftfreq(chunk_size, 1/fs))
    fft_data = np.fft.fftshift(np.abs(fft_signal))
    fft_phase = np.fft.fftshift(np.angle(fft_signal))
    
    power = np.abs(fft_signal)**2
    normalized_power = power / np.max(power)
    fft_power = np.fft.fftshift(normalized_power)

    return fft_magnitude, fft_freq, fft_data, fft_phase, fft_power
