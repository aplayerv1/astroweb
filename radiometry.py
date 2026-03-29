"""radiometry.py — Physical calibration for the HackRF radio telescope.

Converts raw ADC voltage readings into physically meaningful units:
  - System temperature (T_sys) via Y-factor method
  - Antenna temperature (T_A) in Kelvin
  - Flux density in Jansky (1 Jy = 1e-26 W/m²/Hz)
  - Gain drift tracking over time

Usage
-----
    from radiometry import RadiometryCalibrator
    cal = RadiometryCalibrator()

    # Y-factor calibration (point at cold sky, then absorber)
    cal.measure_cold(cold_samples)   # blank sky
    cal.measure_hot(hot_samples)     # 50Ω terminator at room temp (~290 K)
    print(cal.t_sys)                 # system temperature in K

    # Per-chunk conversion
    t_ant = cal.voltage_to_t_ant(rms_voltage)
    flux_jy = cal.t_ant_to_jy(t_ant)

Environment variables
---------------------
    OBSERVER_LAT     : degrees N  (default 0.0)
    OBSERVER_LON     : degrees E  (default 0.0)
    OBSERVER_ALT     : metres above sea level (default 0.0)
    ANTENNA_GAIN_DBI : antenna gain in dBi (default 8.0 for typical dish)
    T_HOT            : hot load physical temperature K (default 290.0)
    T_COLD_SKY       : expected cold sky temp K (default 10.0 — blank sky at 1420 MHz)
"""

import os
import time
import logging
import numpy as np
from collections import deque
from typing import Optional

logger = logging.getLogger('aic.radiometry')

# Physical constants
K_B       = 1.380649e-23   # Boltzmann constant, J/K
C_LIGHT   = 299792458.0    # m/s
HI_FREQ   = 1420.405751786e6  # Hz


class RadiometryCalibrator:
    """Y-factor system temperature calibration and flux conversion.

    The Y-factor method:
        Y = P_hot / P_cold
        T_sys = (T_hot - Y * T_cold) / (Y - 1)

    where P is measured noise power (proportional to RMS²),
    T_hot is the physical temperature of a matched load (room temp ≈ 290 K),
    T_cold is the expected sky temperature at the observed frequency.

    For 1420 MHz blank sky: T_cold ≈ 5-15 K (CMB + galactic synchrotron).
    """

    def __init__(self,
                 freq_hz:        float = HI_FREQ,
                 bandwidth_hz:   float = 2e6,
                 antenna_gain_dbi: float = None,
                 t_hot:          float = None,
                 t_cold_sky:     float = None):

        self.freq_hz      = freq_hz
        self.bandwidth_hz = bandwidth_hz

        # Antenna effective area from gain: A_eff = G * λ² / (4π)
        gain_dbi = antenna_gain_dbi or float(os.getenv('ANTENNA_GAIN_DBI', '8.0'))
        self.antenna_gain_linear = 10 ** (gain_dbi / 10.0)
        wavelength = C_LIGHT / freq_hz
        self.a_eff = self.antenna_gain_linear * wavelength**2 / (4 * np.pi)
        logger.info(f'Antenna: gain={gain_dbi:.1f} dBi, A_eff={self.a_eff:.4f} m²')

        self.t_hot      = t_hot      or float(os.getenv('T_HOT',       '290.0'))
        self.t_cold_sky = t_cold_sky or float(os.getenv('T_COLD_SKY',  '10.0'))

        # Calibration state
        self.t_sys:      Optional[float] = None
        self.gain_k_per_v2: Optional[float] = None  # K / (V_rms²)

        self._p_cold:    Optional[float] = None   # cold sky power (V²)
        self._p_hot:     Optional[float] = None   # hot load power (V²)

        # Gain drift tracking: rolling buffer of (timestamp, rms²) pairs
        self._gain_history: deque = deque(maxlen=3600)   # ~1 hr at 1 Hz
        self._cal_time:  Optional[float] = None

    # ── Calibration measurements ──────────────────────────────────────────────

    def measure_cold(self, samples: np.ndarray) -> float:
        """Measure cold sky power. Point antenna at blank sky patch."""
        rms2 = float(np.mean(np.abs(samples.ravel()).astype(np.float64) ** 2))
        self._p_cold = rms2
        logger.info(f'Cold sky measurement: P_cold={rms2:.6e} V²')
        self._try_compute_tsys()
        return rms2

    def measure_hot(self, samples: np.ndarray) -> float:
        """Measure hot load power. Connect 50Ω terminator at room temperature."""
        rms2 = float(np.mean(np.abs(samples.ravel()).astype(np.float64) ** 2))
        self._p_hot = rms2
        logger.info(f'Hot load measurement: P_hot={rms2:.6e} V²')
        self._try_compute_tsys()
        return rms2

    def _try_compute_tsys(self) -> None:
        """Compute T_sys once both hot and cold measurements are available."""
        if self._p_cold is None or self._p_hot is None:
            return
        if self._p_cold <= 0 or self._p_hot <= 0:
            logger.warning('Invalid power measurements for Y-factor')
            return

        y = self._p_hot / self._p_cold
        if y <= 1.0:
            logger.warning(f'Y-factor={y:.3f} ≤ 1 — hot load not warmer than cold sky? '
                           f'Check connections.')
            return

        t_sys = (self.t_hot - y * self.t_cold_sky) / (y - 1.0)
        if t_sys < 0:
            logger.warning(f'T_sys={t_sys:.1f} K is negative — calibration invalid')
            return

        self.t_sys = t_sys
        # Gain constant: T = gain_k_per_v2 * P_measured
        # From T_cold: T_sys + T_cold_sky = gain_k_per_v2 * P_cold
        self.gain_k_per_v2 = (t_sys + self.t_cold_sky) / self._p_cold
        self._cal_time = time.time()

        logger.info(
            f'Calibration: Y={y:.3f}, T_sys={t_sys:.1f} K, '
            f'T_hot={self.t_hot:.0f} K, T_cold={self.t_cold_sky:.1f} K, '
            f'gain={self.gain_k_per_v2:.3e} K/V²'
        )

    def set_tsys_manual(self, t_sys_k: float) -> None:
        """Manually set T_sys when Y-factor measurement isn't possible."""
        self.t_sys = float(t_sys_k)
        if self._p_cold is not None and self._p_cold > 0:
            self.gain_k_per_v2 = (t_sys_k + self.t_cold_sky) / self._p_cold
        self._cal_time = time.time()
        logger.info(f'T_sys manually set to {t_sys_k:.1f} K')

    # ── Per-chunk conversion ──────────────────────────────────────────────────

    def voltage_to_t_ant(self, samples_or_rms: np.ndarray | float) -> float:
        """Convert samples (or pre-computed RMS) to antenna temperature in K.

        Returns T_ant = T_total - T_sys, clipped at 0.
        T_total > T_sys means the antenna is seeing emission above the noise floor.
        """
        if isinstance(samples_or_rms, np.ndarray):
            rms2 = float(np.mean(np.abs(samples_or_rms.ravel()).astype(np.float64) ** 2))
        else:
            # Assume RMS voltage was passed; square it
            rms2 = float(samples_or_rms) ** 2

        # Track gain drift
        self._gain_history.append((time.time(), rms2))

        if self.gain_k_per_v2 is None or self.t_sys is None:
            return 0.0   # not calibrated yet

        t_total = self.gain_k_per_v2 * rms2
        t_ant   = max(0.0, t_total - self.t_sys)
        return t_ant

    def t_ant_to_jy(self, t_ant_k: float) -> float:
        """Convert antenna temperature to flux density in Jansky.

        S = 2 * k_B * T_ant / A_eff   (single polarisation)
        1 Jy = 1e-26 W/m²/Hz
        """
        if self.a_eff <= 0:
            return 0.0
        flux_w = 2.0 * K_B * t_ant_k / self.a_eff   # W/Hz
        flux_jy = flux_w / 1e-26
        return float(flux_jy)

    def rms_to_jy(self, samples_or_rms) -> float:
        """One-call convenience: raw samples → Jy."""
        return self.t_ant_to_jy(self.voltage_to_t_ant(samples_or_rms))

    def sensitivity_jy(self, integration_seconds: float) -> float:
        """Theoretical 1-sigma sensitivity (radiometer equation) in Jy.

        ΔS = 2 * k_B * T_sys / (A_eff * sqrt(Δf * τ))
        """
        if self.t_sys is None or self.a_eff <= 0:
            return float('inf')
        delta_s = 2.0 * K_B * self.t_sys / (
            self.a_eff * np.sqrt(self.bandwidth_hz * integration_seconds))
        return float(delta_s / 1e-26)   # Jy

    # ── Gain drift monitoring ─────────────────────────────────────────────────

    def gain_drift_fraction(self, window_seconds: float = 300.0) -> Optional[float]:
        """Return fractional gain drift over the last `window_seconds`.

        > 0.05 (5%) suggests significant thermal drift — recalibrate.
        Returns None if insufficient history.
        """
        now = time.time()
        recent = [(t, p) for t, p in self._gain_history if now - t < window_seconds]
        if len(recent) < 10:
            return None
        powers = [p for _, p in recent]
        return float(np.std(powers) / (np.mean(powers) + 1e-30))

    @property
    def is_calibrated(self) -> bool:
        return self.t_sys is not None and self.gain_k_per_v2 is not None

    def calibration_age_seconds(self) -> Optional[float]:
        if self._cal_time is None:
            return None
        return time.time() - self._cal_time

    def status_dict(self) -> dict:
        drift = self.gain_drift_fraction()
        age   = self.calibration_age_seconds()
        return {
            'calibrated':          self.is_calibrated,
            't_sys_k':             self.t_sys,
            'a_eff_m2':            self.a_eff,
            'gain_k_per_v2':       self.gain_k_per_v2,
            'sensitivity_1s_jy':   self.sensitivity_jy(1.0) if self.is_calibrated else None,
            'sensitivity_60s_jy':  self.sensitivity_jy(60.0) if self.is_calibrated else None,
            'gain_drift_5min':     drift,
            'cal_age_seconds':     age,
            'needs_recal':         (drift is not None and drift > 0.05)
                                   or (age is not None and age > 3600),
        }
