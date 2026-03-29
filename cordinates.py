"""coordinates.py — Astrometric utilities for the radio telescope.

Provides:
  - LSR (Local Standard of Rest) Doppler velocity correction
  - Observer location from environment variables
  - Sky coordinate conversions (Az/El → RA/Dec → Galactic)
  - Pointing metadata dict for FITS headers and candidate DB
  - Beam transit profile generator for realistic training signals

Environment variables
---------------------
    OBSERVER_LAT   : geodetic latitude, degrees N  (default 0.0)
    OBSERVER_LON   : geodetic longitude, degrees E  (default 0.0)
    OBSERVER_ALT   : altitude above WGS84 ellipsoid, metres (default 0.0)
    ANTENNA_AZ     : fixed antenna azimuth, degrees (default 180.0 = South)
    ANTENNA_EL     : fixed antenna elevation, degrees (default 45.0)
    BEAM_FWHM_DEG  : antenna beam FWHM, degrees (default 5.0)

All are optional — if not set the code degrades gracefully, returning
zeros for corrections and NaN for coordinates, so the rest of the
pipeline continues without crashing.
"""

import os
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger('aic.coordinates')

# LSR solar motion (IAU 1985 standard)
# V_sun = 20 km/s toward α=18h, δ=+30° (1900.0)
_LSR_RA_DEG  = 270.0   # 18h in degrees
_LSR_DEC_DEG = 30.0
_LSR_V_KMS   = 20.0

HI_REST_FREQ_HZ = 1420.405751786e6
C_LIGHT_KMS     = 299792.458


# ── Observer location ─────────────────────────────────────────────────────────

def get_observer_location():
    """Return (lat_deg, lon_deg, alt_m) from env vars, defaulting to 0,0,0."""
    lat = float(os.getenv('OBSERVER_LAT', '0.0'))
    lon = float(os.getenv('OBSERVER_LON', '0.0'))
    alt = float(os.getenv('OBSERVER_ALT', '0.0'))
    return lat, lon, alt


def get_antenna_pointing():
    """Return (az_deg, el_deg, beam_fwhm_deg) from env vars."""
    az   = float(os.getenv('ANTENNA_AZ',    '180.0'))
    el   = float(os.getenv('ANTENNA_EL',    '45.0'))
    fwhm = float(os.getenv('BEAM_FWHM_DEG', '5.0'))
    return az, el, fwhm


# ── Coordinate transforms (no astropy dependency — pure numpy) ────────────────

def azel_to_radec(az_deg: float, el_deg: float,
                  lat_deg: float, lst_deg: float) -> Tuple[float, float]:
    """Convert (Az, El) to (RA, Dec) given observer latitude and LST.

    Az: degrees, measured East from North
    El: degrees above horizon
    lst_deg: Local Sidereal Time in degrees (0-360)
    Returns (ra_deg, dec_deg) in J2000 approximately.
    """
    az  = np.radians(az_deg)
    el  = np.radians(el_deg)
    lat = np.radians(lat_deg)

    sin_dec = (np.sin(el) * np.sin(lat) +
               np.cos(el) * np.cos(lat) * np.cos(az))
    dec = np.arcsin(np.clip(sin_dec, -1.0, 1.0))

    cos_ha_num = np.sin(el) - np.sin(dec) * np.sin(lat)
    cos_ha_den = np.cos(dec) * np.cos(lat)
    if abs(cos_ha_den) < 1e-10:
        ha = 0.0
    else:
        cos_ha = np.clip(cos_ha_num / cos_ha_den, -1.0, 1.0)
        ha = np.arccos(cos_ha)
        if np.sin(az) > 0:
            ha = 2 * np.pi - ha

    ra = np.radians(lst_deg) - ha
    ra_deg = np.degrees(ra) % 360.0
    dec_deg = np.degrees(dec)
    return ra_deg, dec_deg


def radec_to_galactic(ra_deg: float, dec_deg: float) -> Tuple[float, float]:
    """Convert equatorial (J2000) to Galactic (l, b) in degrees.

    Uses the IAU 1958 definition:
        North Galactic Pole: RA=192.85948°, Dec=+27.12825° (J2000)
        Galactic Centre:     RA=266.40499°, Dec=-28.93617° (J2000)
        Position angle at NGP: 122.93192°
    """
    ra  = np.radians(ra_deg)
    dec = np.radians(dec_deg)

    ra_ngp  = np.radians(192.85948)
    dec_ngp = np.radians(27.12825)
    l_ncp   = np.radians(122.93192)

    sin_b = (np.sin(dec) * np.sin(dec_ngp) +
             np.cos(dec) * np.cos(dec_ngp) * np.cos(ra - ra_ngp))
    b = np.arcsin(np.clip(sin_b, -1.0, 1.0))

    x = np.cos(dec) * np.sin(ra - ra_ngp)
    y = np.sin(dec) * np.cos(dec_ngp) - np.cos(dec) * np.sin(dec_ngp) * np.cos(ra - ra_ngp)
    l = l_ncp - np.arctan2(x, y)
    l_deg = np.degrees(l) % 360.0
    b_deg = np.degrees(b)
    return l_deg, b_deg


def compute_lst(utc_datetime: datetime, lon_deg: float) -> float:
    """Compute Local Sidereal Time in degrees.

    Accurate to ~0.1° for dates within a few decades of J2000.
    """
    # Julian Date
    dt = utc_datetime.replace(tzinfo=timezone.utc)
    jd = (dt - datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 2451545.0
    # Greenwich Mean Sidereal Time in degrees
    T  = (jd - 2451545.0) / 36525.0
    gmst_deg = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
                + 0.000387933 * T**2 - T**3 / 38710000.0) % 360.0
    lst_deg = (gmst_deg + lon_deg) % 360.0
    return lst_deg


# ── LSR Doppler correction ────────────────────────────────────────────────────

def doppler_correction_kms(ra_deg: float, dec_deg: float,
                            utc_time: datetime,
                            lat_deg: float, lon_deg: float,
                            alt_m: float) -> float:
    """Compute velocity correction from topocentric to LSR frame, in km/s.

    v_lsr = v_observed + v_correction
    where v_correction accounts for:
      1. Earth's orbital motion (heliocentric correction, ±30 km/s)
      2. Earth's rotation (diurnal aberration, ±0.5 km/s)
      3. Solar motion toward LSR apex (±20 km/s)

    This is an analytic approximation accurate to ~0.5 km/s, sufficient
    for most amateur HI work. For publication-grade accuracy use astropy.

    Returns correction in km/s to ADD to observed velocity to get v_LSR.
    """
    try:
        ra  = np.radians(ra_deg)
        dec = np.radians(dec_deg)

        # Unit vector toward source
        src = np.array([
            np.cos(dec) * np.cos(ra),
            np.cos(dec) * np.sin(ra),
            np.sin(dec)
        ])

        # ── 1. Earth orbital velocity (heliocentric) ──────────────────────
        # Approximate Earth velocity vector in equatorial frame (km/s)
        # Based on mean orbital speed 29.78 km/s, direction perpendicular
        # to Earth-Sun vector
        dt   = utc_time.replace(tzinfo=timezone.utc)
        jd   = ((dt - datetime(2000, 1, 1, 12, tzinfo=timezone.utc)).total_seconds()
                / 86400.0 + 2451545.0)
        # Mean longitude of Sun (degrees)
        L_sun = (280.460 + 0.9856474 * (jd - 2451545.0)) % 360.0
        # Ecliptic longitude of Earth (opposite Sun)
        lam = np.radians((L_sun + 180.0) % 360.0)
        eps = np.radians(23.439)   # obliquity
        # Earth velocity vector (ecliptic → equatorial)
        v_orb_kms = 29.78
        v_earth = v_orb_kms * np.array([
            -np.sin(lam),
             np.cos(lam) * np.cos(eps),
             np.cos(lam) * np.sin(eps)
        ])
        v_helio = float(np.dot(v_earth, src))

        # ── 2. Earth rotation (diurnal, ~0.5 km/s max) ───────────────────
        lst_deg  = compute_lst(utc_time, lon_deg)
        lat_r    = np.radians(lat_deg)
        lst_r    = np.radians(lst_deg)
        R_earth  = 6378.137   # km
        omega    = 7.2921150e-5  # rad/s
        v_rot_kms = omega * R_earth * np.cos(lat_r) * 1e-3 * \
                    np.cos(dec) * np.sin(lst_r - ra)
        # Note: sign convention — positive toward source = blueshift

        # ── 3. Solar motion toward LSR apex ──────────────────────────────
        ra_apex  = np.radians(_LSR_RA_DEG)
        dec_apex = np.radians(_LSR_DEC_DEG)
        apex = np.array([
            np.cos(dec_apex) * np.cos(ra_apex),
            np.cos(dec_apex) * np.sin(ra_apex),
            np.sin(dec_apex)
        ])
        v_lsr_correction = _LSR_V_KMS * float(np.dot(apex, src))

        total_correction = v_helio + v_rot_kms + v_lsr_correction
        logger.debug(
            f'Doppler correction: helio={v_helio:.2f} rot={v_rot_kms:.3f} '
            f'lsr={v_lsr_correction:.2f} total={total_correction:.2f} km/s'
        )
        return float(total_correction)

    except Exception as e:
        logger.warning(f'Doppler correction failed: {e}')
        return 0.0


def apply_lsr_correction(velocity_obs_km_s: np.ndarray,
                         ra_deg: float, dec_deg: float,
                         utc_time: datetime,
                         lat_deg: float = 0.0,
                         lon_deg: float = 0.0,
                         alt_m:   float = 0.0) -> np.ndarray:
    """Apply LSR correction to an observed velocity array.

    velocity_obs_km_s: array of observed velocities relative to observatory
    Returns: array of LSR velocities (physically meaningful, comparable
             across sessions and with HI surveys).
    """
    corr = doppler_correction_kms(ra_deg, dec_deg, utc_time, lat_deg, lon_deg, alt_m)
    return np.asarray(velocity_obs_km_s) + corr


def freq_to_velocity_lsr(freq_hz: np.ndarray,
                         ra_deg: float, dec_deg: float,
                         utc_time: datetime,
                         lat_deg: float = 0.0,
                         lon_deg: float = 0.0,
                         alt_m:   float = 0.0) -> np.ndarray:
    """Convert frequency axis (Hz) to LSR velocity (km/s).

    Uses relativistic formula: v = c * (f0² - f²) / (f0² + f²)
    Then applies LSR correction.
    """
    f  = np.asarray(freq_hz, dtype=np.float64)
    f0 = HI_REST_FREQ_HZ
    v_obs = C_LIGHT_KMS * (f0**2 - f**2) / (f0**2 + f**2)
    return apply_lsr_correction(v_obs, ra_deg, dec_deg, utc_time,
                                lat_deg, lon_deg, alt_m)


# ── Pointing metadata ─────────────────────────────────────────────────────────

def get_pointing_metadata(utc_time: Optional[datetime] = None) -> dict:
    """Return a dict of pointing and observer metadata for FITS headers / DB.

    Always succeeds — returns NaN for unknown values rather than raising.
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)

    lat_deg, lon_deg, alt_m = get_observer_location()
    az_deg,  el_deg, fwhm   = get_antenna_pointing()

    try:
        lst_deg = compute_lst(utc_time, lon_deg)
    except Exception:
        lst_deg = float('nan')

    try:
        ra_deg, dec_deg = azel_to_radec(az_deg, el_deg, lat_deg, lst_deg)
    except Exception:
        ra_deg = dec_deg = float('nan')

    try:
        glon_deg, glat_deg = radec_to_galactic(ra_deg, dec_deg)
    except Exception:
        glon_deg = glat_deg = float('nan')

    try:
        doppler_corr = doppler_correction_kms(
            ra_deg, dec_deg, utc_time, lat_deg, lon_deg, alt_m)
    except Exception:
        doppler_corr = 0.0

    return {
        # Observer
        'obs_lat_deg':      lat_deg,
        'obs_lon_deg':      lon_deg,
        'obs_alt_m':        alt_m,
        # Pointing
        'az_deg':           az_deg,
        'el_deg':           el_deg,
        'beam_fwhm_deg':    fwhm,
        # Celestial coordinates
        'ra_deg':           ra_deg,
        'dec_deg':          dec_deg,
        'glon_deg':         glon_deg,
        'glat_deg':         glat_deg,
        # Time
        'lst_deg':          lst_deg,
        'utc_iso':          utc_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
        # Velocity correction
        'lsr_correction_km_s': doppler_corr,
    }


def pointing_to_fits_header(meta: dict) -> dict:
    """Convert pointing metadata to FITS keyword/value pairs (8-char keys)."""
    return {
        'OBS-LAT':  (meta['obs_lat_deg'],    '[deg] Observer geodetic latitude'),
        'OBS-LON':  (meta['obs_lon_deg'],    '[deg] Observer geodetic longitude'),
        'OBS-ALT':  (meta['obs_alt_m'],      '[m] Observer altitude'),
        'AZIMUTH':  (meta['az_deg'],         '[deg] Antenna azimuth (N=0, E=90)'),
        'ELEVATIO':  (meta['el_deg'],        '[deg] Antenna elevation'),
        'BMFWHM':   (meta['beam_fwhm_deg'],  '[deg] Beam FWHM'),
        'RA':       (meta['ra_deg'],         '[deg] Right Ascension J2000'),
        'DEC':      (meta['dec_deg'],        '[deg] Declination J2000'),
        'GLON':     (meta['glon_deg'],       '[deg] Galactic longitude'),
        'GLAT':     (meta['glat_deg'],       '[deg] Galactic latitude'),
        'LST':      (meta['lst_deg'],        '[deg] Local Sidereal Time'),
        'DATE-OBS': (meta['utc_iso'],        'UTC observation start'),
        'VHELIO':   (meta['lsr_correction_km_s'], '[km/s] Helio+LSR correction'),
    }


# ── Beam transit profile (for training signal generation) ────────────────────

def beam_transit_profile(n_samples: int,
                         beam_fwhm_deg: float = 5.0,
                         sidereal_rate_deg_per_s: float = 0.00417807,
                         sample_rate_hz: float = 2e6) -> np.ndarray:
    """Generate a realistic antenna beam transit envelope.

    A point source transits through a Gaussian beam at the sidereal rate.
    The envelope is sinc² for a rectangular aperture or Gaussian for a
    circular dish — we use Gaussian as it's more common for dishes.

    Parameters
    ----------
    n_samples              : number of samples in the output
    beam_fwhm_deg          : beam FWHM in degrees (typical: 5° for 1m dish at 1420 MHz)
    sidereal_rate_deg_per_s: sky rotation rate at the equator (0.00417807°/s default)
                             Multiply by cos(dec) for off-equatorial sources.
    sample_rate_hz         : SDR sample rate

    Returns
    -------
    envelope : float32 array of shape (n_samples,), peak = 1.0
    """
    duration_s   = n_samples / sample_rate_hz
    t_s          = np.linspace(-duration_s / 2, duration_s / 2, n_samples)
    # Angular offset as source transits
    angle_deg    = sidereal_rate_deg_per_s * t_s
    # Gaussian beam: FWHM → sigma
    sigma_deg    = beam_fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    envelope     = np.exp(-0.5 * (angle_deg / sigma_deg) ** 2).astype(np.float32)
    return envelope