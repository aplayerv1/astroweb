"""rfi_mask.py — RFI frequency masking and rolling spectral baseline subtraction.

Two responsibilities:
  1. RFI mask: zero out known interference frequencies before inference
     and HI extraction so they don't bias the spectrum or fool the model.

  2. Rolling baseline: maintain a median spectrum over the last N minutes
     and subtract it per-chunk. Removes slow RFI that looks like spectral
     structure (e.g. a warm amplifier producing a broad bump, or a CW
     carrier that drifts slowly across the band).

Environment variables
---------------------
    RFI_FREQS_MHZ  : comma-separated list of known RFI centre frequencies
                     in MHz (e.g. "1420.0,1418.5,1421.3")
    RFI_WIDTH_KHZ  : width of each RFI notch in kHz (default 10.0)
    BASELINE_MINUTES: rolling baseline window in minutes (default 10.0)
    BASELINE_MIN_FRAMES: minimum FFT frames before baseline subtraction
                         starts (default 60)
"""

import os
import logging
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

logger = logging.getLogger('aic.rfi_mask')


# ── RFI mask ──────────────────────────────────────────────────────────────────

def _parse_rfi_freqs() -> List[float]:
    """Parse RFI_FREQS_MHZ environment variable into a list of Hz values."""
    raw = os.getenv('RFI_FREQS_MHZ', '').strip()
    if not raw:
        return []
    freqs = []
    for tok in raw.split(','):
        tok = tok.strip()
        if tok:
            try:
                freqs.append(float(tok) * 1e6)
            except ValueError:
                logger.warning(f'Invalid RFI frequency: {tok}')
    return freqs


# Default known interference sources at 1420 MHz band
# (common USB clock harmonics, LTE bands, WiFi harmonics, GPS L1 alias)
_DEFAULT_RFI_HZ = [
    1415.0e6,   # Common USB 3.0 harmonic
    1422.7e6,   # GPS L1 sideband
    1427.0e6,   # LTE Band 23 uplink edge
    1418.0e6,   # Some WiFi 5 GHz harmonic (alias)
]


def build_rfi_mask(fft_freq_hz: np.ndarray,
                   extra_freqs_hz: Optional[List[float]] = None,
                   notch_width_hz: Optional[float] = None) -> np.ndarray:
    """Build a boolean mask for RFI-contaminated FFT bins.

    Parameters
    ----------
    fft_freq_hz    : absolute frequency axis in Hz (from process_fft)
    extra_freqs_hz : additional RFI frequencies in Hz to mask
    notch_width_hz : width of each notch in Hz (default from RFI_WIDTH_KHZ env)

    Returns
    -------
    mask : bool array, True = clean bin, False = RFI-contaminated
    """
    if notch_width_hz is None:
        notch_width_hz = float(os.getenv('RFI_WIDTH_KHZ', '10.0')) * 1e3

    rfi_freqs = _parse_rfi_freqs() + _DEFAULT_RFI_HZ
    if extra_freqs_hz:
        rfi_freqs.extend(extra_freqs_hz)

    mask = np.ones(len(fft_freq_hz), dtype=bool)
    half = notch_width_hz / 2.0
    for f_rfi in rfi_freqs:
        mask &= ~((fft_freq_hz >= f_rfi - half) & (fft_freq_hz <= f_rfi + half))

    n_flagged = int(np.sum(~mask))
    if n_flagged > 0:
        logger.debug(f'RFI mask: {n_flagged}/{len(mask)} bins flagged')

    return mask


def apply_rfi_mask(power_spectrum: np.ndarray,
                   mask: np.ndarray,
                   fill: str = 'median') -> np.ndarray:
    """Zero or interpolate RFI-flagged bins in a power spectrum.

    Parameters
    ----------
    power_spectrum : 1-D power array
    mask           : True = clean, False = flagged (from build_rfi_mask)
    fill           : 'median' — fill with local median of clean bins
                     'zero'   — fill with 0
                     'interp' — linear interpolation across flagged regions

    Returns
    -------
    cleaned spectrum (copy, does not modify input)
    """
    out = power_spectrum.copy()
    flagged = ~mask

    if not np.any(flagged):
        return out

    if fill == 'zero':
        out[flagged] = 0.0

    elif fill == 'median':
        fill_val = float(np.median(power_spectrum[mask])) if mask.any() else 0.0
        out[flagged] = fill_val

    elif fill == 'interp':
        idx = np.arange(len(out))
        clean_idx = idx[mask]
        if len(clean_idx) >= 2:
            out[flagged] = np.interp(idx[flagged], clean_idx,
                                      power_spectrum[mask])
        else:
            out[flagged] = 0.0

    return out


# ── Rolling spectral baseline ─────────────────────────────────────────────────

class RollingBaseline:
    """Accumulate FFT power spectra and subtract a rolling median baseline.

    Usage
    -----
        baseline = RollingBaseline(n_bins=8192)
        for chunk in stream:
            fft_mag, fft_freq, ... = process_fft(chunk, ...)
            cleaned = baseline.subtract(fft_mag**2)
            # cleaned is baseline-subtracted power, ready for HI extraction
    """

    def __init__(self,
                 n_bins: int = 8192,
                 window_minutes: Optional[float] = None,
                 chunk_rate_hz: float = 1.0,
                 min_frames: Optional[int] = None):
        """
        Parameters
        ----------
        n_bins         : number of FFT bins (must match fft_magnitude length)
        window_minutes : rolling window duration in minutes
        chunk_rate_hz  : approximate processing rate in chunks/second
                         Used to convert window_minutes → frame count.
        min_frames     : minimum frames before subtraction is applied.
                         Before this, returns the raw spectrum unchanged.
        """
        window_min  = window_minutes or float(os.getenv('BASELINE_MINUTES', '10.0'))
        self._min_f = min_frames or int(os.getenv('BASELINE_MIN_FRAMES', '60'))

        max_frames  = max(self._min_f, int(window_min * 60 * chunk_rate_hz))
        self._buf   = deque(maxlen=max_frames)
        self._n     = n_bins
        self._cache: Optional[np.ndarray] = None
        self._cache_dirty = True
        self._frame_count = 0

        logger.info(
            f'RollingBaseline: {n_bins} bins, window={window_min:.1f} min '
            f'({max_frames} frames), min_frames={self._min_f}'
        )

    def push(self, power_spectrum: np.ndarray) -> None:
        """Add a new power spectrum frame to the rolling window."""
        ps = np.asarray(power_spectrum, dtype=np.float64)
        if ps.size != self._n:
            # Resize if needed (e.g. after chunk_size change)
            ps = np.interp(np.linspace(0, 1, self._n),
                           np.linspace(0, 1, ps.size), ps)
        self._buf.append(ps)
        self._cache_dirty = True
        self._frame_count += 1

    @property
    def baseline(self) -> Optional[np.ndarray]:
        """Current baseline estimate (median over rolling window), or None."""
        if len(self._buf) < self._min_f:
            return None
        if self._cache_dirty:
            self._cache = np.median(np.array(self._buf), axis=0)
            self._cache_dirty = False
        return self._cache

    def subtract(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Push spectrum and return baseline-subtracted version.

        Before min_frames have accumulated, returns the raw spectrum.
        After, returns max(raw - baseline, 0) so noise floor is ≥ 0.
        """
        self.push(power_spectrum)
        bl = self.baseline
        if bl is None:
            return np.asarray(power_spectrum, dtype=np.float32)
        residual = np.asarray(power_spectrum, dtype=np.float64) - bl
        return np.maximum(residual, 0.0).astype(np.float32)

    def subtract_with_baseline(self,
                                power_spectrum: np.ndarray
                                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Like subtract() but also returns the baseline for diagnostics."""
        self.push(power_spectrum)
        bl = self.baseline
        if bl is None:
            return np.asarray(power_spectrum, dtype=np.float32), None
        residual = np.maximum(
            np.asarray(power_spectrum, dtype=np.float64) - bl, 0.0
        ).astype(np.float32)
        return residual, bl.astype(np.float32)

    @property
    def ready(self) -> bool:
        """True once enough frames have accumulated for a reliable baseline."""
        return len(self._buf) >= self._min_f

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self) -> None:
        self._buf.clear()
        self._cache = None
        self._cache_dirty = True
        self._frame_count = 0
        logger.info('RollingBaseline reset')


# ── Spectral averaging accumulator ────────────────────────────────────────────

class SpectralAverager:
    """Accumulate FFT power frames and return time-averaged spectrum.

    The radiometer equation: ΔT_min ∝ 1/sqrt(Δf × τ)
    Averaging N frames of duration dt gives τ = N × dt, reducing noise by √N.

    At 2 MHz bandwidth and 4ms per chunk, 60s integration = 15,000 chunks.
    In practice use 10-60s (250-15000 chunks) for HI emission detection.
    """

    def __init__(self,
                 n_bins: int = 8192,
                 average_seconds: float = 30.0,
                 chunk_duration_s: float = 4.096e-3):
        self._n       = n_bins
        self._n_avg   = max(1, int(average_seconds / chunk_duration_s))
        self._accum   = np.zeros(n_bins, dtype=np.float64)
        self._count   = 0
        logger.info(
            f'SpectralAverager: {n_bins} bins, '
            f'{average_seconds:.0f}s window = {self._n_avg} frames'
        )

    def push(self, power_spectrum: np.ndarray) -> Optional[np.ndarray]:
        """Add a frame. Returns averaged spectrum when window is full, else None."""
        ps = np.asarray(power_spectrum, dtype=np.float64)
        if ps.size != self._n:
            ps = np.interp(np.linspace(0, 1, self._n),
                           np.linspace(0, 1, ps.size), ps)
        self._accum += ps
        self._count += 1
        if self._count >= self._n_avg:
            avg = (self._accum / self._count).astype(np.float32)
            self._accum[:] = 0.0
            self._count = 0
            return avg
        return None

    @property
    def progress(self) -> float:
        """Fraction of averaging window filled (0.0 → 1.0)."""
        return self._count / self._n_avg

    def reset(self) -> None:
        self._accum[:] = 0.0
        self._count = 0
