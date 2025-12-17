# This module contains training helpers including signal generators and data loader.

import numpy as np
import os
from typing import List, Optional


def _seed_rng(seed: Optional[int]):
	return np.random.RandomState(seed)


def generate_wow_signals(chunk_size: int, n: int = 16, seed: Optional[int] = None, as_numpy: bool = True) -> List[np.ndarray]:
	rng = _seed_rng(seed)
	out = []
	t = np.arange(chunk_size, dtype=np.float32)
	for i in range(n):
		center = rng.uniform(0.005, 0.02)
		drift = rng.uniform(-0.001, 0.001)
		amp = rng.uniform(0.5, 3.0)
		envelope = np.exp(-((t - chunk_size/2)**2) / (0.08 * chunk_size)**2)
		freq = center + drift * (t - chunk_size/2) / float(chunk_size)
		sig = amp * envelope * np.exp(2j * np.pi * freq * t)
		out.append(sig.astype(np.complex64) if as_numpy else sig)
	return out


def generate_pulsar_signals(chunk_size: int, n: int = 16, seed: Optional[int] = None, as_numpy: bool = True) -> List[np.ndarray]:
	rng = _seed_rng(seed)
	out = []
	t = np.arange(chunk_size, dtype=np.float32)
	for i in range(n):
		period = rng.uniform(50, 400)
		pulse_width = rng.uniform(1, 8)
		amp = rng.uniform(0.5, 2.0)
		pulse_train = np.zeros(chunk_size, dtype=np.float32)
		phases = rng.randint(0, int(period))
		for p in range(phases, chunk_size, int(period)):
			start = int(max(0, p - pulse_width//2))
			end = int(min(chunk_size, p + pulse_width//2))
			pulse_train[start:end] += amp * np.hanning(end - start)
		sig = pulse_train + rng.normal(0, 0.2, size=chunk_size)
		out.append(sig.astype(np.float32) if as_numpy else sig)
	return out


def generate_frb_signals(chunk_size: int, n: int = 16, seed: Optional[int] = None, as_numpy: bool = True) -> List[np.ndarray]:
	rng = _seed_rng(seed)
	out = []
	t = np.arange(chunk_size, dtype=np.float32)
	for i in range(n):
		peak = rng.randint(int(chunk_size*0.1), int(chunk_size*0.9))
		width = rng.uniform(2, 20)
		amp = rng.uniform(1.0, 4.0)
		sig = amp * np.exp(-0.5 * ((t - peak)/width)**2)
		sig = sig + rng.normal(0, 0.3, size=chunk_size)
		out.append(sig.astype(np.float32) if as_numpy else sig)
	return out


def generate_hydrogen_line(chunk_size: int, n: int = 8, seed: Optional[int] = None, as_numpy: bool = True) -> List[np.ndarray]:
	rng = _seed_rng(seed)
	out = []
	t = np.arange(chunk_size, dtype=np.float32)
	for i in range(n):
		freq = rng.uniform(0.009, 0.011)
		amp = rng.uniform(0.05, 0.6)
		sig = amp * np.sin(2 * np.pi * freq * t)
		sig = sig + rng.normal(0, 0.2, size=chunk_size)
		out.append(sig.astype(np.float32) if as_numpy else sig)
	return out


def load_training_data_from_folder(path: str, chunk_size: int, as_numpy: bool = True):
	"""Load CSV or NPY files from a folder as a pool of signals.

	This function is permissive: it will attempt to read '.npy' and '.csv' files
	and will reshape/trim/pad to `chunk_size`. Returns a list of 1-D arrays.
	"""
	pool = []
	if not os.path.isdir(path):
		return pool
	for fname in os.listdir(path):
		fpath = os.path.join(path, fname)
		try:
			if fname.lower().endswith('.npy'):
				arr = np.load(fpath)
			elif fname.lower().endswith('.csv'):
				arr = np.loadtxt(fpath, delimiter=',')
			else:
				continue
			arr = np.asarray(arr).ravel()
			if arr.size < chunk_size:
				# pad
				pad = np.zeros(chunk_size - arr.size, dtype=arr.dtype)
				arr = np.concatenate([arr, pad])
			elif arr.size > chunk_size:
				arr = arr[:chunk_size]
			pool.append(arr.astype(np.float32) if as_numpy else arr)
		except Exception:
			continue
	return pool


__all__ = [
	'generate_wow_signals',
	'generate_pulsar_signals',
	'generate_frb_signals',
	'generate_hydrogen_line',
	'load_training_data_from_folder'
]