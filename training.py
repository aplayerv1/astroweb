import numpy as np
import cupy as cp
import pandas as pd
import os

def generate_hydrogen_line(chunk_size=8192):
    t = cp.arange(chunk_size, dtype=cp.float32)
    velocity_shifts = [-100, -50, 0, 50, 100]  # km/s
    signals = []
    
    for velocity in velocity_shifts:
        freq_shift = 1420.4 * (1 + velocity/3e5)  # MHz
        density = 1.0 + 0.3 * cp.sin(2 * cp.pi * 0.001 * t)
        signal = density * cp.exp(-(t - len(t)/2)**2 / (len(t)/8)**2)
        signal = signal * cp.exp(2j * cp.pi * freq_shift * t/len(t))
        signals.append(signal)
    
    return cp.array(signals)

def generate_pulsar_signals(chunk_size=8192):
    t = cp.arange(chunk_size, dtype=cp.float32)
    signals = []
    
    period_1919 = 1.337
    pulse_width_1919 = int(chunk_size * 0.05)
    signal_1919 = cp.zeros_like(t)
    pulse_locations = cp.arange(0, chunk_size, int(period_1919 * chunk_size/8))
    for loc in pulse_locations:
        if loc + pulse_width_1919 <= chunk_size:
            signal_1919[loc:loc+pulse_width_1919] = cp.exp(-cp.linspace(0, 5, pulse_width_1919))
    
    period_vela = 0.089
    signal_vela = cp.zeros_like(t)
    pulse_width_vela = int(chunk_size * 0.02)
    pulse_locations = cp.arange(0, chunk_size, int(period_vela * chunk_size/8))
    for loc in pulse_locations:
        if loc + pulse_width_vela <= chunk_size:
            signal_vela[loc:loc+pulse_width_vela] = 1.5 * cp.exp(-cp.linspace(0, 4, pulse_width_vela))
    
    signals.extend([signal_1919, signal_vela])
    return cp.array(signals)

def generate_frb_signals(chunk_size=8192):
    t = cp.arange(chunk_size, dtype=cp.float32)
    signals = []
    
    burst_width = int(chunk_size * 0.01)
    burst_loc = cp.random.randint(0, chunk_size-burst_width)
    single_frb = cp.zeros_like(t)
    single_frb[burst_loc:burst_loc+burst_width] = 3.0 * cp.exp(-cp.linspace(0, 6, burst_width))
    
    repeating_frb = cp.zeros_like(t)
    burst_locations = cp.random.choice(chunk_size-burst_width, 3)
    for loc in burst_locations:
        repeating_frb[loc:loc+burst_width] = 2.5 * cp.exp(-cp.linspace(0, 5, burst_width))
    
    drift_rate = -0.002
    freq_drift = cp.exp(2j * cp.pi * drift_rate * t * t)
    
    signals.extend([single_frb * freq_drift, repeating_frb * freq_drift])
    return cp.array(signals)


def generate_wow_signals(chunk_size=8192):
    t = cp.arange(chunk_size, dtype=cp.float32)
    wow_signals = []
    
    for _ in range(4):
        # Gaussian envelope
        intensity = 2.0 * cp.exp(-((t - chunk_size / 2) ** 2) / (chunk_size / 4) ** 2)
        
        # Frequency drift
        drift_rate = 0.003
        base_freq = 0.01
        freq_drift = base_freq + drift_rate * (t - chunk_size / 2) / chunk_size
        
        # Complex Wow! signal
        phase = 2 * cp.pi * freq_drift * t
        signal = intensity * cp.exp(1j * phase)
        
        # Return absolute value (real power envelope), or real part if needed
        wow_signals.append(cp.asnumpy(cp.abs(signal)).astype(np.float32))
    
    return wow_signals

def load_training_data_from_folder(folder_path=None, chunk_size=8192):
    training_data = []
    if folder_path and os.path.exists(folder_path):
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        if not csv_files:
            return None
            
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            data = cp.array(pd.read_csv(file_path).values)
            num_samples = len(data)
            if num_samples > chunk_size:
                data = data[:chunk_size]
            elif num_samples < chunk_size:
                padding = cp.zeros(chunk_size - num_samples)
                data = cp.concatenate([data, padding])
            training_data.append(data)
        return cp.stack(training_data)
    return None