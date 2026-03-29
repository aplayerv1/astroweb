"""data_generation.py — deterministic, batch-aware data generator.

Fixes vs original:
- Imports HTRU2 real-world data via training.load_htru2()
- Balances real pulsar examples with synthetic ones
- Consistent float32 normalisation
- Seed-reproducible RNG
"""
from typing import Optional
import numpy as np
import os
import logging

logger = logging.getLogger('aic.data_generation')

try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = np
    _HAS_CUPY = False

from training import (
    generate_wow_signals,
    generate_pulsar_signals,
    generate_frb_signals,
    generate_hydrogen_line,
    generate_doppler_shifted_hi,
    load_training_data_from_folder,
    load_all_datasets,
)


def _build_csv_pool(csv_data, chunk_size):
    """Normalise csv_data into a list of 1-D float32 arrays of length chunk_size."""
    if csv_data is None:
        return []
    if isinstance(csv_data, str):
        try:
            pool = load_training_data_from_folder(csv_data, chunk_size, as_numpy=True)
            return pool
        except Exception as e:
            logger.warning(f'_build_csv_pool folder load failed: {e}')
            return []
    arr = np.asarray(csv_data)
    out = []
    try:
        arr2 = arr.reshape(-1, chunk_size)
        for r in arr2:
            out.append(r.astype(np.float32))
    except Exception:
        if arr.ndim == 1 and arr.size == chunk_size:
            out.append(arr.astype(np.float32))
    return out


def build_positive_pool(chunk_size: int, seed: Optional[int], htru2_dir: str = 'data/htru2',
                         use_real_data: bool = True):
    """Build a pool of positive (signal) examples from all datasets."""
    all_pos = []

    # 1. Synthetic generators
    for gen_fn in (generate_wow_signals, generate_pulsar_signals,
                   generate_frb_signals, generate_hydrogen_line,
                   generate_doppler_shifted_hi):
        try:
            lst = gen_fn(chunk_size, seed=seed, as_numpy=True)
            for s in lst:
                arr = np.real(np.asarray(s)).ravel()
                if arr.size == chunk_size:
                    all_pos.append(arr.astype(np.float32))
        except Exception as e:
            logger.debug(f'Synthetic generator {gen_fn.__name__} failed: {e}')

    # 2. All real-world datasets via load_all_datasets
    if use_real_data:
        try:
            data_root = os.path.dirname(htru2_dir)  # e.g. 'data'
            X_all, y_all = load_all_datasets(chunk_size=chunk_size,
                                              data_root=data_root,
                                              auto_download=True)
            if X_all.shape[0] > 0:
                pos_mask = y_all == 1
                pulsar_sigs = X_all[pos_mask]
                logger.info(f'All datasets: adding {len(pulsar_sigs)} positive signals to pool')
                for s in pulsar_sigs:
                    all_pos.append(s.astype(np.float32))
        except Exception as e:
            logger.warning(f'load_all_datasets failed: {e}')

    if len(all_pos) == 0:
        logger.warning('Positive pool is empty — using random signals as fallback')
        rng = np.random.RandomState(seed)
        for _ in range(32):
            t = np.arange(chunk_size, dtype=np.float32)
            s = np.sin(2 * np.pi * 0.01 * t) + rng.normal(0, 0.3, chunk_size)
            all_pos.append(s.astype(np.float32))

    return np.vstack(all_pos)


def data_generator(chunk_size=8192, csv_data=None, debug=False, seed: Optional[int] = None,
                   batch_size=32, positive_ratio=0.5, as_numpy=True,
                   htru2_dir: str = 'data/htru2', use_real_data: bool = True,
                   _prebuilt_pool=None):
    """Deterministic, batch-aware infinite generator.

    Yields (X, y) where:
      X shape: (batch_size, chunk_size, 1)  float32
      y shape: (batch_size,)                float32  {0, 1}
    """
    rng = np.random.RandomState(seed)

    # Use pre-built pool if provided — skips load_all_datasets() entirely
    csv_pool = _build_csv_pool(csv_data, chunk_size)
    if _prebuilt_pool is not None:
        wow_pool = _prebuilt_pool
        logger.info(f'data_generator: using pre-built pool ({len(wow_pool)} signals)')
    else:
        wow_pool = build_positive_pool(chunk_size, seed=seed, htru2_dir=htru2_dir,
                                       use_real_data=use_real_data)
        logger.info(f'data_generator: positive pool size={len(wow_pool)}, csv_pool size={len(csv_pool)}')

    pos_per_batch = max(1, int(batch_size * float(positive_ratio)))

    t = np.arange(chunk_size, dtype=np.float32)

    while True:
        X_batch = np.zeros((batch_size, chunk_size, 1), dtype=np.float32)
        F_batch = np.zeros((batch_size, 1024, 1), dtype=np.float32)   # FFT branch
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            if i < pos_per_batch:
                # --- Positive sample ---
                signal = None
                if csv_pool and rng.rand() < 0.3:
                    signal = csv_pool[rng.randint(len(csv_pool))].copy()
                elif len(wow_pool) > 0:
                    signal = wow_pool[rng.randint(len(wow_pool))].copy()

                if signal is None:
                    signal = 0.8 * np.sin(2 * np.pi * 0.01 * t * rng.uniform(0.8, 1.2))
                    signal += rng.normal(0, 0.3, size=chunk_size).astype(np.float32)

                signal = signal * rng.uniform(0.7, 1.5)
                signal += rng.normal(0, 0.1, size=chunk_size).astype(np.float32)
                label = 1.0

            else:
                # --- Negative sample: RFI / noise / hardware artefacts ---
                # 22 types weighted toward the highest false-positive risk for
                # a BiGRU + SE-attention model:
                #   • Periodic structures (impulse trains, harmonic stacks)
                #   • Gaussian-enveloped bursts (look like FRB/WOW)
                #   • Spectrally shaped noise (triggers channel attention)
                #   • Hardware artefacts (IQ imbalance, clipping, DC)
                #
                # Weights are intentionally non-uniform: the high-FP-risk types
                # appear 2-3× more often than simple noise so the model is
                # forced to discriminate them during every training epoch.
                NEG_WEIGHTS = np.array([
                    1.0,  # 0  white Gaussian
                    1.0,  # 1  Rayleigh
                    1.0,  # 2  cumulative drift
                    1.0,  # 3  non-stationary Gaussian + Poisson
                    2.0,  # 4  narrowband RFI tone          ← high FP risk
                    1.5,  # 5  broadband FM
                    1.0,  # 6  gain-varying Gaussian
                    1.5,  # 7  saturated / clipped IQ
                    1.0,  # 8  DC offset / LO leakage
                    3.0,  # 9  periodic impulse train        ← highest FP risk
                    3.0,  # 10 Gaussian-enveloped burst      ← highest FP risk
                    2.5,  # 11 shaped thermal noise          ← high FP risk
                    2.5,  # 12 harmonic stack                ← high FP risk
                    2.0,  # 13 chirp / swept tone            ← high FP risk
                    2.0,  # 14 multi-tone RFI
                    1.5,  # 15 band-limited (bandpass) noise
                    2.0,  # 16 1/f pink noise
                    1.5,  # 17 IQ imbalance artefact
                    2.5,  # 18 bursty RFI (on/off)          ← high FP risk
                    1.0,  # 19 quantisation noise floor
                    1.5,  # 20 wideband impulse + ringing
                    1.5,  # 21 frequency hop
                ], dtype=np.float64)
                NEG_WEIGHTS /= NEG_WEIGHTS.sum()
                choice = rng.choice(len(NEG_WEIGHTS), p=NEG_WEIGHTS)

                if choice == 0:
                    # White Gaussian noise
                    signal = rng.normal(0, 1.0, size=chunk_size)

                elif choice == 1:
                    # Rayleigh fading envelope
                    signal = rng.rayleigh(scale=1.0, size=chunk_size)

                elif choice == 2:
                    # Cumulative drift with exponential decay
                    signal = np.cumsum(rng.normal(0, 0.5, size=chunk_size)) * np.exp(-t / float(chunk_size))

                elif choice == 3:
                    # Non-stationary Gaussian + Poisson shot noise
                    signal = rng.normal(0, np.exp(-t / float(chunk_size))) + rng.poisson(1.0, size=chunk_size)

                elif choice == 4:
                    # Narrowband RFI tone — single CW carrier with noise
                    freq_rfi = rng.uniform(0.05, 0.45)
                    signal = rng.uniform(0.3, 2.0) * np.sin(2 * np.pi * freq_rfi * t)
                    signal += rng.normal(0, 0.5, size=chunk_size)

                elif choice == 5:
                    # Broadband FM interference (AM-modulated noise)
                    carrier = rng.uniform(0.03, 0.1)
                    signal = np.sin(2 * np.pi * carrier * t) * rng.normal(0, 1.0, size=chunk_size)

                elif choice == 6:
                    # Gain-varying Gaussian (slow amplitude modulation)
                    gain = 1.0 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.005, 0.02) * t)
                    signal = rng.normal(0, gain, size=chunk_size)

                elif choice == 7:
                    # Saturated / clipped IQ — overdriven HackRF (amp near ADC rail)
                    amp = rng.uniform(1.2, 1.45)
                    phase = rng.uniform(0, 2 * np.pi, size=chunk_size)
                    signal = amp * np.cos(phase).astype(np.float32)
                    signal += rng.normal(0, 0.002, size=chunk_size)

                elif choice == 8:
                    # DC offset / LO leakage
                    dc = rng.uniform(-1.5, 1.5)
                    signal = np.full(chunk_size, dc, dtype=np.float32)
                    signal += rng.normal(0, 0.005, size=chunk_size)

                elif choice == 9:
                    # ★ Periodic impulse train — mimics pulsar periodicity
                    # This is the #1 false-positive risk for a BiGRU model.
                    # Man-made sources: power-line harmonics, switching supplies,
                    # motor controllers, USB polling, clock jitter.
                    period = rng.randint(max(2, chunk_size // 200), chunk_size // 4)
                    signal = np.zeros(chunk_size, dtype=np.float32)
                    pulse_indices = np.arange(0, chunk_size, period)
                    width = rng.randint(1, max(2, period // 8))
                    amp = rng.uniform(0.5, 3.0)
                    for idx in pulse_indices:
                        end = min(idx + width, chunk_size)
                        signal[idx:end] = amp * rng.uniform(0.8, 1.2)
                    signal += rng.normal(0, 0.2, size=chunk_size)

                elif choice == 10:
                    # ★ Gaussian-enveloped burst — looks like FRB or WOW signal
                    # The envelope shape is identical to what signal generators
                    # produce; only the underlying carrier differs.
                    center = rng.randint(chunk_size // 4, 3 * chunk_size // 4)
                    width_frac = rng.uniform(0.05, 0.25)
                    width_s = max(10, int(width_frac * chunk_size))
                    env = np.exp(-0.5 * ((t - center) / width_s) ** 2).astype(np.float32)
                    # Carrier is either noise or a tone at an unusual frequency
                    if rng.rand() < 0.5:
                        carrier = rng.normal(0, 1.0, size=chunk_size).astype(np.float32)
                    else:
                        freq_c = rng.uniform(0.1, 0.45)
                        carrier = np.sin(2 * np.pi * freq_c * t).astype(np.float32)
                    signal = rng.uniform(0.5, 2.0) * env * carrier
                    signal += rng.normal(0, 0.1, size=chunk_size)

                elif choice == 11:
                    # ★ Shaped thermal noise — noise convolved with a Gaussian
                    # window so it has a smooth amplitude envelope that resembles
                    # a signal profile but contains no coherent carrier.
                    raw = rng.normal(0, 1.0, size=chunk_size).astype(np.float32)
                    win_size = rng.randint(chunk_size // 20, chunk_size // 4)
                    window = np.exp(-0.5 * (np.arange(win_size) - win_size/2)**2 / (win_size/4)**2)
                    window /= window.sum()
                    signal = np.convolve(raw, window, mode='same')

                elif choice == 12:
                    # ★ Harmonic stack — fundamental + N harmonics
                    # Mimics power-line (50/60 Hz aliases), switching regulators.
                    # SE channel attention flags the structured spectrum.
                    f0 = rng.uniform(0.01, 0.05)
                    n_harmonics = rng.randint(3, 8)
                    signal = np.zeros(chunk_size, dtype=np.float32)
                    for h in range(1, n_harmonics + 1):
                        amp_h = rng.uniform(0.2, 1.0) / h  # 1/h amplitude rolloff
                        phase_h = rng.uniform(0, 2 * np.pi)
                        signal += amp_h * np.sin(2 * np.pi * f0 * h * t + phase_h)
                    signal += rng.normal(0, 0.15, size=chunk_size)

                elif choice == 13:
                    # ★ Chirp / swept tone — linear frequency sweep
                    # Mimics Doppler-shifted HI line to naive temporal models.
                    f_start = rng.uniform(0.02, 0.2)
                    f_end   = rng.uniform(0.2,  0.48)
                    if rng.rand() < 0.5:
                        f_start, f_end = f_end, f_start  # descending chirp
                    instantaneous_phase = 2 * np.pi * (f_start * t + (f_end - f_start) / (2 * chunk_size) * t**2)
                    signal = np.sin(instantaneous_phase).astype(np.float32)
                    signal += rng.normal(0, 0.3, size=chunk_size)

                elif choice == 14:
                    # Multi-tone RFI — 2-4 simultaneous CW carriers
                    # Models intermodulation products from nearby transmitters.
                    n_tones = rng.randint(2, 5)
                    signal = np.zeros(chunk_size, dtype=np.float32)
                    used_freqs = []
                    for _ in range(n_tones):
                        # Keep tones separated by at least 0.05 to avoid overlap
                        for attempt in range(20):
                            f = rng.uniform(0.05, 0.45)
                            if all(abs(f - uf) > 0.05 for uf in used_freqs):
                                break
                        used_freqs.append(f)
                        amp_t = rng.uniform(0.3, 1.5)
                        signal += amp_t * np.sin(2 * np.pi * f * t + rng.uniform(0, 2*np.pi))
                    signal += rng.normal(0, 0.3, size=chunk_size)

                elif choice == 15:
                    # Band-limited (bandpass) noise — spectrally structured
                    # Passes noise through a narrow bandpass; looks like a weak
                    # signal in the FFT but has no coherent phase.
                    from scipy.signal import butter, sosfilt
                    f_center = rng.uniform(0.1, 0.4)
                    bw = rng.uniform(0.02, 0.08)
                    lo = max(0.01, f_center - bw/2)
                    hi = min(0.49, f_center + bw/2)
                    if hi > lo + 0.01:
                        sos = butter(4, [lo, hi], btype='band', output='sos')
                        noise = rng.normal(0, 1.0, size=chunk_size + 512)
                        filtered = sosfilt(sos, noise)[512:]
                        std = np.std(filtered)
                        signal = (filtered / (std + 1e-9)).astype(np.float32)
                    else:
                        signal = rng.normal(0, 1.0, size=chunk_size)

                elif choice == 16:
                    # 1/f pink noise — power-law PSD, common in analog electronics
                    # Generate via inverse FFT of shaped spectrum
                    freqs = np.fft.rfftfreq(chunk_size)
                    freqs[0] = 1e-6  # avoid divide-by-zero at DC
                    spectrum = rng.normal(0, 1.0, freqs.size) + 1j * rng.normal(0, 1.0, freqs.size)
                    spectrum /= np.sqrt(freqs)
                    signal = np.fft.irfft(spectrum, n=chunk_size).astype(np.float32)

                elif choice == 17:
                    # IQ imbalance artefact — amplitude or phase mismatch between
                    # I and Q channels creates a mirror image in the spectrum.
                    amp_imbalance  = rng.uniform(0.8, 1.2)
                    phase_imbalance = rng.uniform(-0.2, 0.2)  # radians
                    freq_c = rng.uniform(0.05, 0.3)
                    I = np.cos(2 * np.pi * freq_c * t + rng.uniform(0, 2*np.pi))
                    Q = amp_imbalance * np.sin(2 * np.pi * freq_c * t + rng.uniform(0, 2*np.pi) + phase_imbalance)
                    signal = (I + Q).astype(np.float32)  # combined real output
                    signal += rng.normal(0, 0.2, size=chunk_size)

                elif choice == 18:
                    # ★ Bursty RFI — short on/off bursts that look like FRB
                    # Duty cycle 10-40%, random burst positions, varying amplitude.
                    duty = rng.uniform(0.1, 0.4)
                    burst_len = max(1, int(duty * chunk_size / rng.randint(2, 8)))
                    signal = rng.normal(0, 0.15, size=chunk_size).astype(np.float32)
                    pos = 0
                    on = rng.rand() < 0.5
                    while pos < chunk_size:
                        seg_len = rng.randint(max(1, burst_len//2), burst_len*2)
                        end = min(pos + seg_len, chunk_size)
                        if on:
                            freq_b = rng.uniform(0.05, 0.4)
                            amp_b  = rng.uniform(0.5, 2.0)
                            signal[pos:end] += amp_b * np.sin(2 * np.pi * freq_b * t[pos:end])
                        pos = end
                        on = not on

                elif choice == 19:
                    # Quantisation noise floor — low-amplitude signal with coarse
                    # ADC steps, simulates a very weak or distant source that sits
                    # at the noise floor.
                    n_bits = rng.randint(4, 8)
                    levels = 2 ** n_bits
                    raw = rng.uniform(-1, 1, size=chunk_size)
                    signal = (np.round(raw * levels) / levels).astype(np.float32)
                    signal += rng.normal(0, 0.5 / levels, size=chunk_size)

                elif choice == 20:
                    # Wideband impulse + ringing — single large spike that rings
                    # through the filter bank, creating spurious spectral structure.
                    signal = rng.normal(0, 0.1, size=chunk_size).astype(np.float32)
                    n_impulses = rng.randint(1, 4)
                    for _ in range(n_impulses):
                        idx = rng.randint(0, chunk_size)
                        amp_imp = rng.uniform(3.0, 8.0)
                        signal[idx] += amp_imp
                    # Simulate filter ringing with exponentially decaying sinusoid
                    ring_len = min(256, chunk_size // 4)
                    ring_t = np.arange(ring_len)
                    ring = np.exp(-ring_t / rng.uniform(10, 50)) * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * ring_t)
                    signal = np.convolve(signal, ring, mode='same').astype(np.float32)

                else:  # choice == 21
                    # Frequency hop — discontinuous jumps between carriers
                    # Triggers detectors that look for temporal structure.
                    n_hops = rng.randint(4, 12)
                    hop_freqs = rng.uniform(0.05, 0.45, size=n_hops)
                    hop_len = chunk_size // n_hops
                    signal = np.zeros(chunk_size, dtype=np.float32)
                    for h, freq_h in enumerate(hop_freqs):
                        start = h * hop_len
                        end = min(start + hop_len, chunk_size)
                        amp_h = rng.uniform(0.5, 1.5)
                        signal[start:end] = amp_h * np.sin(2 * np.pi * freq_h * t[start:end])
                    signal += rng.normal(0, 0.2, size=chunk_size)

                # Add a small random interference term to all negatives so the
                # model cannot rely on the absence of any background noise.
                interfere = rng.normal(0, 0.15, size=chunk_size) * (1 + 0.3 * np.sin(2 * np.pi * 0.02 * t))
                signal = signal + interfere
                label = 0.0

            # Per-signal normalisation
            s_mean = np.mean(signal)
            s_std = np.std(signal)
            if s_std < 1e-6:
                s_std = 1.0
            signal = np.clip((signal - s_mean) / s_std, -5.0, 5.0).astype(np.float32)

            X_batch[i, :, 0] = signal
            y_batch[i] = label

            # Compute log-magnitude FFT for the spectral branch.
            # Use the normalised signal so the FFT reflects what the model sees.
            raw_fft = np.abs(np.fft.fft(signal * np.kaiser(chunk_size, 14))) + 1e-12
            raw_fft = raw_fft / (np.max(raw_fft) + 1e-12)
            # Downsample 8192 -> FFT_BINS=1024
            stride = chunk_size // 1024
            fft_ds = raw_fft[::stride][:1024]
            if fft_ds.size < 1024:
                fft_ds = np.pad(fft_ds, (0, 1024 - fft_ds.size), mode='edge')
            fft_ds = np.log10(fft_ds + 1e-6)
            fm, fs = np.mean(fft_ds), np.std(fft_ds)
            if fs < 1e-6: fs = 1.0
            fft_ds = np.clip((fft_ds - fm) / fs, -5.0, 5.0).astype(np.float32)
            F_batch[i, :, 0] = fft_ds

        # Shuffle batch
        perm = rng.permutation(batch_size)
        X_batch = X_batch[perm]
        F_batch = F_batch[perm]
        y_batch = y_batch[perm]

        if debug:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 3))
                plt.plot(X_batch[0].squeeze())
                plt.title(f'Debug Signal — Label: {int(y_batch[0])}')
                plt.tight_layout()
                plt.show()
            except Exception:
                pass

        if as_numpy:
            yield [X_batch, F_batch], y_batch
        else:
            if _HAS_CUPY:
                yield [cp.asarray(X_batch), cp.asarray(F_batch)], cp.asarray(y_batch)
            else:
                yield [X_batch, F_batch], y_batch