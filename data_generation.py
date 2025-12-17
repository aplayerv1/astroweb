"""Data generation helpers moved out of aic2.py.

Provides a deterministic, seedable `data_generator` that prefers NumPy
but can return CuPy arrays when requested and available.
"""
from typing import Optional, Iterable
import numpy as np
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
    load_training_data_from_folder,
)


def _build_csv_pool(csv_data, chunk_size):
    """Normalize csv_data into a list of 1-D numpy float32 arrays of length chunk_size.

    csv_data may be:
    - None
    - a path (str) handled by `load_training_data_from_folder`
    - an array-like of samples
    """
    if csv_data is None:
        return []

    if isinstance(csv_data, str):
        try:
            pool = load_training_data_from_folder(csv_data, chunk_size, as_numpy=True)
            arr = np.asarray(pool)
        except Exception:
            return []
    else:
        arr = np.asarray(csv_data)

    # Try reshape into (-1, chunk_size)
    out = []
    try:
        arr2 = arr.reshape(-1, chunk_size)
        for r in arr2:
            out.append(r.astype(np.float32))
    except Exception:
        if arr.ndim == 1 and arr.size == chunk_size:
            out.append(arr.astype(np.float32))
    return out


def data_generator(chunk_size=8192, csv_data=None, debug=False, seed: Optional[int]=None,
                   batch_size=32, positive_ratio=0.5, as_numpy=True):
    """Deterministic, batch-aware data generator.

    Yields (X, y) pairs where X shape is (batch_size, chunk_size, 1) and y shape is (batch_size,).
    - Uses NumPy RNG by default; if `seed` provided the generator is deterministic.
    - `csv_data` may be a path (str) or an array-like of signals (will be coerced to numpy).
    - `positive_ratio` controls fraction of positives in each batch.
    - By default returns NumPy arrays; set `as_numpy=False` to return CuPy arrays when available.
    """
    # Build csv pool
    csv_pool = _build_csv_pool(csv_data, chunk_size)

    # Deterministic RNG
    rng = np.random.RandomState(seed)

    # Pre-generate a small pool of synthetic positives (WOW/pulsar/FRB/hydrogen)
    try:
        wow_list = generate_wow_signals(chunk_size, seed=seed, as_numpy=True)
        pulsar_list = generate_pulsar_signals(chunk_size, seed=seed, as_numpy=True)
        frb_list = generate_frb_signals(chunk_size, seed=seed, as_numpy=True)
        hydro_list = generate_hydrogen_line(chunk_size, seed=seed, as_numpy=True)

        all_pos = []
        for l in (wow_list, pulsar_list, frb_list, hydro_list):
            if l:
                for s in l:
                    arr = np.asarray(s).ravel()
                    if arr.size == chunk_size:
                        all_pos.append(arr.astype(np.float32))
        if len(all_pos) > 0:
            wow_pool = np.vstack(all_pos)
        else:
            wow_pool = np.empty((0, chunk_size), dtype=np.float32)
    except Exception:
        wow_pool = np.empty((0, chunk_size), dtype=np.float32)

    pos_per_batch = max(1, int(batch_size * float(positive_ratio)))
    neg_per_batch = batch_size - pos_per_batch

    while True:
        X_batch = np.zeros((batch_size, chunk_size, 1), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        for i in range(batch_size):
            if i < pos_per_batch:
                # Positive: prefer CSV examples, then WOW pool, else synthesize
                signal = None
                if csv_pool and rng.rand() < 0.45:
                    signal = csv_pool[rng.randint(len(csv_pool))]
                elif len(wow_pool) > 0 and rng.rand() < 0.8:
                    signal = wow_pool[rng.randint(len(wow_pool))]

                if signal is None:
                    t = np.arange(chunk_size, dtype=np.float32)
                    signal = 0.8 * np.sin(2 * np.pi * 0.01 * t * rng.uniform(0.8, 1.2))
                    signal = signal + rng.normal(0, 0.5, size=chunk_size)

                amp = rng.uniform(0.8, 2.0)
                sig = signal * amp + rng.normal(0, 0.2, size=chunk_size)
                label = 1.0
            else:
                # Negative / natural: variety of noise/interference types
                t = np.arange(chunk_size, dtype=np.float32)
                choice = rng.randint(0, 6)
                if choice == 0:
                    sig = rng.normal(0, 1.0, size=chunk_size)
                elif choice == 1:
                    sig = rng.rayleigh(scale=1.0, size=chunk_size)
                elif choice == 2:
                    sig = np.cumsum(rng.normal(0, 0.5, size=chunk_size)) * np.exp(-t / float(chunk_size))
                elif choice == 3:
                    sig = rng.normal(0, np.exp(-t / float(chunk_size))) + rng.poisson(1.0, size=chunk_size)
                elif choice == 4:
                    sig = np.sin(2 * np.pi * 0.05 * t) * rng.normal(0, 1.0, size=chunk_size)
                else:
                    sig = rng.normal(0, 1.0 + 0.3 * np.sin(2 * np.pi * 0.01 * t), size=chunk_size)

                interfere = rng.normal(0, 0.3, size=chunk_size) * (1 + 0.5 * np.sin(2 * np.pi * 0.02 * t))
                sig = sig + interfere
                label = 0.0

            # Per-signal normalization (stable variance)
            s_mean = np.mean(sig)
            s_std = np.std(sig)
            if s_std < 1e-6:
                s_std = 1.0
            sig = ((sig - s_mean) / s_std).astype(np.float32)

            X_batch[i, :, 0] = sig
            y_batch[i] = label

        # Shuffle the batch
        perm = rng.permutation(batch_size)
        X_batch = X_batch[perm]
        y_batch = y_batch[perm]

        if debug:
            try:
                import matplotlib.pyplot as plt
                plt.plot(X_batch[0].squeeze())
                plt.title(f"Debug Signal Plot - Label: {int(y_batch[0])}")
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
