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
                choice = rng.randint(0, 9)
                if choice == 0:
                    signal = rng.normal(0, 1.0, size=chunk_size)
                elif choice == 1:
                    signal = rng.rayleigh(scale=1.0, size=chunk_size)
                elif choice == 2:
                    signal = np.cumsum(rng.normal(0, 0.5, size=chunk_size)) * np.exp(-t / float(chunk_size))
                elif choice == 3:
                    signal = rng.normal(0, np.exp(-t / float(chunk_size))) + rng.poisson(1.0, size=chunk_size)
                elif choice == 4:
                    # Narrowband RFI tone
                    freq_rfi = rng.uniform(0.05, 0.4)
                    signal = rng.uniform(0.3, 2.0) * np.sin(2 * np.pi * freq_rfi * t)
                    signal += rng.normal(0, 0.5, size=chunk_size)
                elif choice == 5:
                    # Broadband FM interference
                    signal = np.sin(2 * np.pi * 0.05 * t) * rng.normal(0, 1.0, size=chunk_size)
                elif choice == 6:
                    signal = rng.normal(0, 1.0 + 0.3 * np.sin(2 * np.pi * 0.01 * t), size=chunk_size)
                elif choice == 7:
                    # Saturated / clipped IQ — constant amplitude ~1.4, what an
                    # overdriven HackRF looks like after complex-to-real conversion.
                    # The model must learn to reject this as a false positive.
                    amp = rng.uniform(1.2, 1.45)
                    phase = rng.uniform(0, 2 * np.pi, size=chunk_size)
                    signal = amp * np.cos(phase).astype(np.float32)
                    signal += rng.normal(0, 0.002, size=chunk_size)  # tiny ADC jitter
                else:
                    # DC offset / LO leakage — near-zero std, constant value
                    dc = rng.uniform(-1.5, 1.5)
                    signal = np.full(chunk_size, dc, dtype=np.float32)
                    signal += rng.normal(0, 0.005, size=chunk_size)

                interfere = rng.normal(0, 0.3, size=chunk_size) * (1 + 0.5 * np.sin(2 * np.pi * 0.02 * t))
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

        # Shuffle batch
        perm = rng.permutation(batch_size)
        X_batch = X_batch[perm]
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
            yield X_batch, y_batch
        else:
            if _HAS_CUPY:
                yield cp.asarray(X_batch), cp.asarray(y_batch)
            else:
                yield X_batch, y_batch