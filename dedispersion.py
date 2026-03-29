"""dedispersion.py — Incoherent dedispersion for FRB and pulsar detection.

Dispersion measure (DM) causes lower radio frequencies to arrive later:
    Δt = 4.148808 ms × DM × (f_low⁻² - f_high⁻²)   [f in GHz, DM in pc/cm³]

At 1420 MHz with 2 MHz bandwidth:
    DM=10:  Δt ≈ 0.41 ms  (negligible vs 4ms chunk resolution)
    DM=100: Δt ≈ 4.1 ms   (one chunk — borderline)
    DM=500: Δt ≈ 20 ms    (5 chunks — smears out FRB completely)

Incoherent dedispersion:
  - Split the band into N sub-bands
  - Shift each sub-band in time by the expected DM delay
  - Sum to form a dedispersed time series
  - Search for impulsive peaks across multiple DM trials

This module implements a lightweight DM search optimised for the
HackRF's narrow 2 MHz bandwidth, where the maximum useful DM is ~200 pc/cm³.

For a 2 MHz band at 1420 MHz:
  - 8 sub-bands of 250 kHz each
  - DM trials: 0, 5, 10, 20, 50, 100, 150, 200 pc/cm³
  - Each trial shifts sub-bands by 0-5 samples

Environment variables
---------------------
    DM_TRIALS      : comma-separated DM values to trial (default: 0,5,10,25,50,100,200)
    DM_N_SUBBANDS  : number of frequency sub-bands (default: 8)
    DM_SNR_THRESH  : SNR threshold for FRB candidate flagging (default: 6.0)
"""

import os
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict

logger = logging.getLogger('aic.dedispersion')

# Dispersion constant (exact)
_K_DM = 4.148808e3   # ms × GHz² × pc⁻¹ × cm³


def dm_delay_samples(dm: float,
                     f_high_hz: float, f_low_hz: float,
                     sample_rate_hz: float) -> int:
    """Compute DM delay in samples between f_high and f_low.

    Parameters
    ----------
    dm            : dispersion measure in pc/cm³
    f_high_hz     : highest frequency in Hz
    f_low_hz      : lowest frequency in Hz
    sample_rate_hz: SDR sample rate in Hz

    Returns
    -------
    delay in samples (always ≥ 0)
    """
    f_hi_ghz = f_high_hz / 1e9
    f_lo_ghz = f_low_hz  / 1e9
    delay_ms = _K_DM * dm * (f_lo_ghz**-2 - f_hi_ghz**-2)
    delay_s  = delay_ms * 1e-3
    delay_samp = int(round(delay_s * sample_rate_hz))
    return max(0, delay_samp)


def _parse_dm_trials() -> List[float]:
    raw = os.getenv('DM_TRIALS', '0,5,10,25,50,100,200')
    try:
        return [float(x.strip()) for x in raw.split(',') if x.strip()]
    except Exception:
        return [0, 5, 10, 25, 50, 100, 200]


class IncoherentDedisperser:
    """Incoherent dedispersion search over DM trials.

    Usage
    -----
        dd = IncoherentDedisperser(center_freq_hz=1420.4e6,
                                   bandwidth_hz=2e6,
                                   sample_rate_hz=2e6)
        # Feed complex IQ samples one chunk at a time
        result = dd.push(iq_chunk)
        if result['best_snr'] > 6.0:
            print(f"FRB candidate! DM={result['best_dm']:.1f}")
    """

    def __init__(self,
                 center_freq_hz:  float = 1420.405e6,
                 bandwidth_hz:    float = 2e6,
                 sample_rate_hz:  float = 2e6,
                 n_subbands:      int   = None,
                 dm_trials:       Optional[List[float]] = None,
                 snr_threshold:   float = None):

        self.fs          = float(sample_rate_hz)
        self.f_center    = float(center_freq_hz)
        self.bw          = float(bandwidth_hz)
        self.f_high      = center_freq_hz + bandwidth_hz / 2
        self.f_low       = center_freq_hz - bandwidth_hz / 2

        self.n_sub       = int(n_subbands or os.getenv('DM_N_SUBBANDS', '8'))
        self.dm_trials   = dm_trials or _parse_dm_trials()
        self.snr_thresh  = snr_threshold or float(os.getenv('DM_SNR_THRESH', '6.0'))

        # Sub-band centre frequencies
        self.sub_freqs   = np.linspace(self.f_low, self.f_high, self.n_sub + 1)
        self.sub_centers = (self.sub_freqs[:-1] + self.sub_freqs[1:]) / 2

        # Pre-compute delay table: dm_delays[dm_idx, sub_idx] = samples
        self.delay_table = np.zeros((len(self.dm_trials), self.n_sub), dtype=int)
        for i, dm in enumerate(self.dm_trials):
            for j, f_sub in enumerate(self.sub_centers):
                # Delay relative to highest sub-band
                self.delay_table[i, j] = dm_delay_samples(
                    dm, self.f_high, f_sub, self.fs)

        max_delay = int(np.max(self.delay_table))
        self._buf_len = 0   # set on first chunk
        self._buf     = None
        self._max_delay = max_delay

        logger.info(
            f'Dedisperser: {self.n_sub} sub-bands, '
            f'{len(self.dm_trials)} DM trials [{min(self.dm_trials):.0f}'
            f'–{max(self.dm_trials):.0f} pc/cm³], '
            f'max_delay={max_delay} samples'
        )

    def _init_buf(self, chunk_size: int) -> None:
        # Buffer holds 2 chunks so we can shift across chunk boundaries
        self._buf_len = chunk_size
        self._buf = np.zeros(chunk_size * 2, dtype=np.complex64)

    def _subband_power(self, iq: np.ndarray) -> np.ndarray:
        """Split IQ into sub-band power time series.

        Returns array of shape (n_sub, n_samples) — power in each sub-band.
        Uses FFT-based channelisation: FFT → select bins → IFFT → |.|²
        """
        n       = len(iq)
        spec    = np.fft.fft(iq)
        n_per   = n // self.n_sub
        power   = np.zeros((self.n_sub, n), dtype=np.float32)

        for j in range(self.n_sub):
            # Select bins for this sub-band
            lo = j * n_per
            hi = lo + n_per
            sub_spec = np.zeros(n, dtype=np.complex64)
            sub_spec[lo:hi] = spec[lo:hi]
            sub_time = np.fft.ifft(sub_spec)
            power[j] = np.abs(sub_time).astype(np.float32) ** 2

        return power

    def push(self, iq_chunk: np.ndarray) -> Dict:
        """Process one IQ chunk and return dedispersion search results.

        IMPORTANT — bandwidth limitation:
        At 2 MHz bandwidth centred on 1420 MHz, the dispersion delay for
        DM=10 pc/cm³ is ~116,000 samples — far larger than any practical
        chunk size. All sub-band delays exceed the buffer length, so the
        alignment reads zeros and the SNR formula produces nonsense values.

        The method detects this condition and returns is_candidate=False with
        snr=0 when max_delay > chunk_size // 4.  Dedispersion only becomes
        meaningful at bandwidths of ~20 MHz or more at 1420 MHz.

        Returns
        -------
        dict with keys:
            best_dm     : float — DM of best SNR trial
            best_snr    : float — peak SNR across all DM trials
            dm_snr      : list  — (dm, snr) for each trial
            is_candidate: bool  — True if best_snr > snr_threshold
            bw_limited  : bool  — True if bandwidth too narrow for DM search
        """
        iq = np.asarray(iq_chunk, dtype=np.complex64).ravel()
        n  = len(iq)

        if self._buf is None:
            self._init_buf(n)

        # Bandwidth guard: if all delays >> chunk_size the search is meaningless.
        # At 2 MHz BW / 1420 MHz, DM=10 → ~116k sample delay vs 8192 chunk.
        # Return early rather than producing infinite SNR from zero-filled arrays.
        if self._max_delay > n // 4:
            null = [(dm, 0.0) for dm in self.dm_trials]
            return {
                'best_dm':      0.0,
                'best_snr':     0.0,
                'dm_snr':       null,
                'is_candidate': False,
                'bw_limited':   True,
                'n_trials':     len(self.dm_trials),
            }

        # Roll buffer: drop old data, append new chunk
        self._buf = np.roll(self._buf, -n)
        self._buf[-n:] = iq

        # Compute sub-band power for the full buffer
        power = self._subband_power(self._buf)   # (n_sub, 2*n)
        focus = power[:, n:]                      # newest chunk (n_sub, n)

        results = []
        best_dm  = 0.0
        best_snr = 0.0
        best_ts  = None

        for i, dm in enumerate(self.dm_trials):
            aligned = np.zeros((self.n_sub, n), dtype=np.float32)
            for j in range(self.n_sub):
                d = self.delay_table[i, j]
                if d == 0:
                    aligned[j] = focus[j]
                elif d < n:
                    aligned[j, :n-d] = focus[j, d:]
                    aligned[j, n-d:] = power[j, n-d:n]
                # else: delay > chunk — leave zeros (correctly handled by guard above)

            dedispersed = aligned.sum(axis=0)

            # SNR: use IQR-based robust estimator instead of MAD.
            # MAD on a near-flat array has catastrophic cancellation → +inf SNR.
            # IQR is more stable: noise floor ~ IQR/1.349 (Gaussian equiv.)
            p25, p75 = float(np.percentile(dedispersed, 25)),                        float(np.percentile(dedispersed, 75))
            noise_iqr = (p75 - p25) / 1.349 + 1e-6   # robust noise estimate
            peak      = float(np.max(dedispersed))
            baseline  = float(np.median(dedispersed))
            snr       = (peak - baseline) / noise_iqr
            results.append((dm, snr))

            if snr > best_snr:
                best_snr = snr
                best_dm  = dm
                best_ts  = dedispersed

        is_candidate = best_snr >= self.snr_thresh
        if is_candidate:
            logger.info(
                f'FRB/pulsar candidate: DM={best_dm:.1f} pc/cm³, '
                f'SNR={best_snr:.1f}σ'
            )

        return {
            'best_dm':      best_dm,
            'best_snr':     best_snr,
            'dm_snr':       results,
            'is_candidate': is_candidate,
            'bw_limited':   False,
            'timeseries':   best_ts,
            'n_trials':     len(self.dm_trials),
        }

    def dm_vs_snr_array(self, result: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract DM and SNR arrays from a push() result for plotting."""
        dms  = np.array([x[0] for x in result['dm_snr']])
        snrs = np.array([x[1] for x in result['dm_snr']])
        return dms, snrs