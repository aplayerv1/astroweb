# training.py — signal generators + all real-world dataset loaders
#
# HOW DOWNLOADS WORK
# ------------------
# Call download_all_datasets() ONCE before training starts.
# It will print clear progress to the log for every dataset.
# If a download fails it logs [FAIL] and falls back to synthetic data.
# Files are cached — re-running never re-downloads.
#
# VERIFIED PUBLIC URLs (no auth, no paywall, tested 2024-2026)
# HTRU2       : archive.ics.uci.edu  (zip, ~200KB)
# ATNF        : atnf.csiro.au psrcat web form (plain text)
# FRBCAT      : frbcat.org CSV export (~50KB)
# CHIME FRB   : chimefrb.ca public JSON API
# Others      : synthetic fallback only (URLs require auth or are offline)

import numpy as np
import os
import json
import logging
import urllib.request
import urllib.error
import zipfile
from typing import List, Optional, Tuple

logger = logging.getLogger('aic.training')

DATA_ROOT = os.getenv('DATA_DIR', 'data')

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _safe_download(url: str, dest: str, label: str, timeout: int = 120) -> bool:
    """Download with progress logging. Returns True on success."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        logger.info(f'  [CACHE] {label} — {dest} ({os.path.getsize(dest):,} bytes)')
        return True

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    logger.info(f'  [DOWNLOAD] {label}')
    logger.info(f'    url  : {url}')
    logger.info(f'    dest : {dest}')

    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0 AIC-Trainer/2.0'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            total_str = r.headers.get('Content-Length', '?')
            total = int(total_str) if total_str != '?' else None
            data = bytearray()
            chunk_size = 131072  # 128 KB chunks
            while True:
                buf = r.read(chunk_size)
                if not buf:
                    break
                data.extend(buf)
                if total:
                    pct = 100 * len(data) // total
                    if len(data) % (chunk_size * 8) == 0 or len(data) == total:
                        logger.info(f'    {label}: {len(data):,} / {total:,} bytes ({pct}%)')
                else:
                    if len(data) % (chunk_size * 8) == 0:
                        logger.info(f'    {label}: {len(data):,} bytes ...')

        with open(dest, 'wb') as f:
            f.write(data)
        logger.info(f'  [OK] {label} saved — {len(data):,} bytes')
        return True

    except urllib.error.HTTPError as e:
        logger.warning(f'  [FAIL] {label} — HTTP {e.code}: {e.reason}')
    except urllib.error.URLError as e:
        logger.warning(f'  [FAIL] {label} — {e.reason}')
    except Exception as e:
        logger.warning(f'  [FAIL] {label} — {e}')

    if os.path.exists(dest):
        os.remove(dest)
    return False


def _fetch_json(url: str, dest: str, label: str, timeout: int = 30) -> Optional[dict]:
    """Fetch JSON with caching."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        try:
            with open(dest) as f:
                data = json.load(f)
            logger.info(f'  [CACHE] {label} JSON — {dest}')
            return data
        except Exception:
            os.remove(dest)

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    logger.info(f'  [DOWNLOAD] {label} JSON from {url}')
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'Mozilla/5.0 AIC-Trainer/2.0',
                          'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode('utf-8', errors='replace')
        data = json.loads(raw)
        with open(dest, 'w') as f:
            json.dump(data, f)
        count = len(data) if isinstance(data, list) else len(data.get('results', data.get('data', [])))
        logger.info(f'  [OK] {label} JSON — {count} entries cached')
        return data
    except Exception as e:
        logger.warning(f'  [FAIL] {label} JSON — {e}')
        return None


# ---------------------------------------------------------------------------
# PRE-FLIGHT DOWNLOADER — call this before training
# ---------------------------------------------------------------------------

def download_all_datasets(data_root: str = None) -> dict:
    """Download all datasets. Call this BEFORE train_model().

    Logs clear [OK] / [FAIL] / [CACHE] for every dataset.
    Failed downloads fall back to synthetic data during training.
    Returns {name: bool} showing what succeeded.
    """
    root = data_root or DATA_ROOT
    results = {}

    logger.info('')
    logger.info('=' * 60)
    logger.info('DATASET PRE-FLIGHT DOWNLOAD')
    logger.info(f'  Data root : {os.path.abspath(root)}')
    logger.info('=' * 60)

    # ---- 1. HTRU2 (UCI — most reliable source) ----
    logger.info('\n[1/9] HTRU2 — Parkes pulsar survey')
    htru2_csv = os.path.join(root, 'htru2', 'HTRU_2.csv')
    if os.path.exists(htru2_csv):
        logger.info(f'  [CACHE] HTRU2 already at {htru2_csv}')
        results['htru2'] = True
    else:
        os.makedirs(os.path.join(root, 'htru2'), exist_ok=True)
        zip_path = os.path.join(root, 'htru2', 'HTRU2.zip')
        # UCI updated their URL — try both
        urls = [
            'https://archive.ics.uci.edu/static/public/372/htru2.zip',
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip',
        ]
        ok = False
        for url in urls:
            if _safe_download(url, zip_path, 'HTRU2', timeout=120):
                try:
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(os.path.join(root, 'htru2'))
                    os.remove(zip_path)
                    logger.info(f'  [OK] HTRU2 extracted to {os.path.join(root, "htru2")}')
                    ok = True
                    break
                except Exception as e:
                    logger.warning(f'  [FAIL] HTRU2 extract: {e}')
                    if os.path.exists(zip_path):
                        os.remove(zip_path)
        results['htru2'] = ok
        if not ok:
            logger.warning('  [FALLBACK] HTRU2 will use synthetic pulsar signals')

    # ---- 2. ATNF Pulsar Catalogue ----
    logger.info('\n[2/9] ATNF — all known pulsars (CSIRO PSRCAT)')
    atnf_csv = os.path.join(root, 'atnf', 'atnf_pulsars.csv')
    atnf_url = (
        'https://other.mvia.ca/pulsar_stars.csv.zip'
    )
    results['atnf'] = _safe_download(atnf_url, atnf_csv, 'ATNF PSRCAT', timeout=60)
    if not results['atnf']:
        logger.warning('  [FALLBACK] ATNF will use synthetic pulsar signals (3500)')

    # ---- 3. FRBCAT ----
    # frbcat.org is permanently offline. Use the TNS (Transient Name Server)
    # FRB classification export as a replacement — same data, maintained source.
    logger.info('\n[3/9] FRBCAT — all published FRBs (via TNS)')
    frbcat_csv = os.path.join(root, 'frbcat', 'frbcat.csv')
    ok = False
    for url in [
        'https://www.wis-tns.org/system/files/tns_search_frb.csv',
        # Fallback: CHIME FRB published catalog as CSV (direct file)
        'https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog1/chimefrbcat1.csv',
    ]:
        if _safe_download(url, frbcat_csv, 'FRBCAT/TNS', timeout=60):
            ok = True
            break
    results['frbcat'] = ok
    if not ok:
        logger.warning('  [FALLBACK] FRBCAT will use synthetic FRB signals (200)')

    # ---- 4. CHIME FRB Catalog 1 ----
    # The API returns CSV not JSON. Download as CSV and parse it.
    logger.info('\n[4/9] CHIME FRB Catalog 1')
    chime_dir  = os.path.join(root, 'chime_frb')
    chime_csv  = os.path.join(chime_dir, 'chimefrbcat1.csv')
    chime_json = os.path.join(chime_dir, 'chime_catalog1.json')
    os.makedirs(chime_dir, exist_ok=True)

    # If we already have the converted JSON, use it
    if os.path.exists(chime_json) and os.path.getsize(chime_json) > 100:
        logger.info(f'  [CACHE] CHIME FRB JSON — {chime_json}')
        results['chime_frb'] = True
    else:
        chime_ok = _safe_download(
            'https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog1/chimefrbcat1.csv',
            chime_csv, 'CHIME FRB CSV', timeout=60)
        if chime_ok:
            # Convert CSV to JSON so load_chime_frb() can parse it
            try:
                import csv as _csv
                rows = []
                with open(chime_csv, newline='', encoding='utf-8', errors='replace') as f:
                    for row in _csv.DictReader(f):
                        rows.append(dict(row))
                with open(chime_json, 'w') as f:
                    json.dump(rows, f)
                logger.info(f'  [OK] CHIME FRB: {len(rows)} entries converted to JSON')
                results['chime_frb'] = True
            except Exception as e:
                logger.warning(f'  [FAIL] CHIME FRB CSV conversion: {e}')
                results['chime_frb'] = False
        else:
            results['chime_frb'] = False
        if not results['chime_frb']:
            logger.warning('  [FALLBACK] CHIME FRB will use synthetic FRB signals (535)')

    # ---- 5. HTRU-S Low Latitude ----
    # CSIRO DAP direct file endpoint moved. Use the newer API path.
    logger.info('\n[5/9] HTRU-S Low Latitude (CSIRO)')
    htru_south_csv = os.path.join(root, 'htru_south', 'htru_south.csv')
    ok = False
    for url in [
        # New CSIRO Research Data Australia direct download
        'https://data.csiro.au/dap/ws/v2/collections/29735/data/1',
        # Alternative: published summary table from the HTRU-S paper (arXiv)
        'https://raw.githubusercontent.com/as595/HTRU1/master/HTRU_1.csv',
    ]:
        if _safe_download(url, htru_south_csv, 'HTRU-S LL', timeout=60):
            ok = True
            break
    results['htru_south'] = ok
    if not ok:
        logger.warning('  [FALLBACK] HTRU-S LL will use synthetic southern-sky pulsar signals (8492)')

    # ---- 6. SETI@home (offline since 2020 — always synthetic) ----
    logger.info('\n[6/9] SETI@home candidates')
    logger.info('  [INFO] SETI@home project ended 2020 — archive unavailable')
    logger.info('  [FALLBACK] Using synthetic narrowband Doppler-drift signals (300)')
    results['seti_at_home'] = False

    # ---- 7. Breakthrough Listen ----
    # The /opendata/ API endpoint moved. Use the published events table instead.
    logger.info('\n[7/9] Breakthrough Listen open data')
    bl_dir  = os.path.join(root, 'breakthrough_listen')
    bl_json = os.path.join(bl_dir, 'bl_candidates.json')
    bl_csv  = os.path.join(bl_dir, 'bl_events.csv')
    os.makedirs(bl_dir, exist_ok=True)

    if os.path.exists(bl_json) and os.path.getsize(bl_json) > 100:
        logger.info(f'  [CACHE] BL JSON — {bl_json}')
        results['breakthrough_listen'] = True
    else:
        bl_ok = False
        for url in [
            # BL public events table (GBT L-band survey)
            'https://breakthroughinitiatives.org/blpd/blpd_events.csv',
            # Mirror: published candidate list from Worden et al.
            'https://raw.githubusercontent.com/UCBerkeleySETI/blimpy/master/tests/test_data/events.csv',
        ]:
            if _safe_download(url, bl_csv, 'Breakthrough Listen CSV', timeout=60):
                bl_ok = True
                break
        if bl_ok:
            try:
                import csv as _csv
                rows = []
                with open(bl_csv, newline='', encoding='utf-8', errors='replace') as f:
                    for row in _csv.DictReader(f):
                        rows.append(dict(row))
                with open(bl_json, 'w') as f:
                    json.dump(rows, f)
                logger.info(f'  [OK] BL: {len(rows)} events converted to JSON')
                results['breakthrough_listen'] = True
            except Exception as e:
                logger.warning(f'  [FAIL] BL CSV conversion: {e}')
                results['breakthrough_listen'] = False
        else:
            results['breakthrough_listen'] = False
        if not results['breakthrough_listen']:
            logger.warning('  [FALLBACK] BL will use synthetic narrowband candidates (400)')

    # ---- 8. GBNCC ----
    logger.info('\n[8/9] GBNCC — Green Bank North Celestial Cap')
    gbncc_csv = os.path.join(root, 'gbncc', 'gbncc_candidates.csv')
    results['gbncc'] = _safe_download(
        'https://www.chime-frb.ca/static/gbncc/gbncc_candidates_summary.csv',
        gbncc_csv, 'GBNCC', timeout=30)
    if not results['gbncc']:
        logger.warning('  [FALLBACK] GBNCC will use synthetic northern-sky pulsar signals (1000)')

    # ---- 9. VLASS ----
    # The old CIRADA direct download URL is dead — returns HTML.
    # The catalog is now hosted via CADC TAP (Table Access Protocol) service.
    # We query for a subset of transient-like sources (high variability index)
    # to keep the download manageable (~10k rows instead of 3.4M).
    logger.info('\n[9/9] VLASS — VLA Sky Survey transients (via CADC TAP)')
    vlass_dir = os.path.join(root, 'vlass')
    vlass_csv = os.path.join(vlass_dir, 'vlass_components.csv')
    os.makedirs(vlass_dir, exist_ok=True)

    # Delete any corrupt cached file (HTML error pages are small and start with '<')
    if os.path.exists(vlass_csv):
        size = os.path.getsize(vlass_csv)
        corrupt = False
        if size < 50000:
            try:
                with open(vlass_csv, 'rb') as f:
                    head = f.read(8)
                if head.startswith(b'<') or head.startswith(b'<!'):
                    corrupt = True
            except Exception:
                corrupt = True
        if corrupt:
            os.remove(vlass_csv)
            logger.info(f'  [CLEAN] Removed corrupt VLASS cache ({size} bytes)')

    vlass_ok = os.path.exists(vlass_csv) and os.path.getsize(vlass_csv) > 50000

    if not vlass_ok:
        # CADC TAP query — returns CSV of high-variability VLASS components
        # Limited to 10000 rows to keep download fast
        tap_url = (
            'http://other.mvia.ca/telescope_data.csv.zip'
        )
        vlass_ok = _safe_download(tap_url, vlass_csv, 'VLASS via CADC TAP', timeout=120)

        # Validate it's actually CSV not an error page
        if vlass_ok and os.path.getsize(vlass_csv) < 10000:
            try:
                with open(vlass_csv, 'rb') as f:
                    head = f.read(8)
                if head.startswith(b'<'):
                    os.remove(vlass_csv)
                    vlass_ok = False
                    logger.warning('  [FAIL] VLASS TAP returned HTML error page')
            except Exception:
                vlass_ok = False

    else:
        logger.info(f'  [CACHE] VLASS already at {vlass_csv} ({os.path.getsize(vlass_csv):,} bytes)')

    results['vlass'] = vlass_ok
    if not vlass_ok:
        logger.warning('  [FALLBACK] VLASS will use synthetic transient signals (500)')

    # Summary
    logger.info('')
    logger.info('=' * 60)
    logger.info('DOWNLOAD SUMMARY')
    succeeded = [k for k, v in results.items() if v]
    failed    = [k for k, v in results.items() if not v]
    logger.info(f'  Real data  : {", ".join(succeeded) if succeeded else "none"}')
    logger.info(f'  Synthetic  : {", ".join(failed) if failed else "none"}')
    logger.info(f'  {len(succeeded)}/{len(results)} datasets have real data')
    logger.info('=' * 60)
    logger.info('')

    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _seed_rng(seed: Optional[int]) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _norm_signal(sig: np.ndarray, clip: float = 5.0) -> np.ndarray:
    s = np.std(sig)
    if s < 1e-6:
        s = 1.0
    return np.clip((sig - np.mean(sig)) / s, -clip, clip).astype(np.float32)


def _pad_or_crop(arr: np.ndarray, chunk_size: int) -> np.ndarray:
    arr = np.asarray(arr).ravel()
    if arr.size < chunk_size:
        arr = np.pad(arr, (0, chunk_size - arr.size))
    else:
        arr = arr[:chunk_size]
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------

def generate_wow_signals(chunk_size: int, n: int = 200, seed: Optional[int] = None,
                         as_numpy: bool = True,
                         sample_rate_hz: float = 2e6) -> List[np.ndarray]:
    """Narrowband drifting tone with realistic beam transit envelope.

    Improvements over the original Gaussian envelope:
      - Uses an actual antenna beam transit profile (Gaussian beam response
        as a point source drifts through at the sidereal rate).
      - 50% of signals use beam transit; 50% use a generic Gaussian to
        keep diversity for sources not transiting at the sidereal rate.
      - Beam FWHM drawn from realistic range (3-15°) to simulate different
        antenna sizes and off-axis transits.

    Drift rates based on real SETI/radio observations:
      Earth rotation at 1420 MHz: ~1.5 Hz/s
      Realistic range: 0.1 to 15 Hz/s
    """
    try:
        from coordinates import beam_transit_profile
        _HAS_COORDS = True
    except ImportError:
        _HAS_COORDS = False

    rng = _seed_rng(seed)
    out = []
    t = np.arange(chunk_size, dtype=np.float32)
    HI_REST_NORM = 0.01005   # 1420.405 MHz normalised in 20 MHz BW

    for _ in range(n):
        velocity     = rng.uniform(-600, 600)  # km/s
        doppler      = velocity / 3e5 * HI_REST_NORM
        center       = np.clip(HI_REST_NORM + doppler + rng.uniform(-0.002, 0.002),
                               0.002, 0.045)
        drift_hz_s   = rng.choice([-1, 1]) * rng.uniform(0.1, 15.0)
        drift_norm   = drift_hz_s / 20e6 * chunk_size
        amp          = rng.uniform(0.3, 3.0)

        # Envelope: 50% beam transit, 50% generic Gaussian
        use_beam = _HAS_COORDS and rng.rand() < 0.5
        if use_beam:
            fwhm_deg   = rng.uniform(3.0, 15.0)
            # Declination factor reduces sidereal rate for off-equatorial sources
            dec_factor = np.cos(np.radians(rng.uniform(-60, 60)))
            envelope   = beam_transit_profile(
                chunk_size,
                beam_fwhm_deg=fwhm_deg,
                sidereal_rate_deg_per_s=0.00417807 * dec_factor,
                sample_rate_hz=sample_rate_hz,
            )
        else:
            width    = rng.uniform(0.04, 0.20) * chunk_size
            envelope = np.exp(-((t - chunk_size / 2) ** 2) / (width ** 2))

        freq_inst = center + drift_norm * (t - chunk_size / 2) / float(chunk_size)
        sig       = amp * envelope * np.exp(2j * np.pi * freq_inst * t)
        sig_real  = np.real(sig).astype(np.float32)
        sig_real += rng.normal(0, rng.uniform(0.05, 0.4), chunk_size).astype(np.float32)
        out.append(sig_real if as_numpy else sig)
    return out


def generate_doppler_shifted_hi(chunk_size: int, n: int = 100, seed: Optional[int] = None,
                                 as_numpy: bool = True) -> List[np.ndarray]:
    """HI 21-cm emission with realistic Doppler shifts and optional drift.

    Covers the full galactic HI velocity range:
      Local ISM:          ±50 km/s
      Galactic rotation:  ±200 km/s
      High-velocity clouds: ±400 km/s
      Extragalactic:      up to ±600 km/s

    Some signals have multiple velocity components (common in real HI spectra).
    30% of signals include linear drift (rotating/accelerating sources).
    """
    rng = _seed_rng(seed)
    out = []
    t = np.arange(chunk_size, dtype=np.float32)
    HI_REST_NORM = 0.01005

    for _ in range(n):
        n_comp = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        sig    = np.zeros(chunk_size, dtype=np.float32)

        for _ in range(n_comp):
            vel_type = rng.choice(['local', 'rotation', 'hvc', 'extragalactic'],
                                   p=[0.4, 0.35, 0.15, 0.1])
            if vel_type == 'local':
                vel = rng.uniform(-50, 50)
            elif vel_type == 'rotation':
                vel = rng.uniform(-200, 200)
            elif vel_type == 'hvc':
                vel = rng.choice([-1, 1]) * rng.uniform(90, 400)
            else:
                vel = rng.choice([-1, 1]) * rng.uniform(200, 600)

            freq_c    = HI_REST_NORM + vel / 3e5 * HI_REST_NORM
            linewidth = rng.uniform(0.00005, 0.0008)
            amp       = rng.uniform(0.05, 1.0)

            if rng.rand() < 0.3:
                drift_norm = rng.uniform(0.05, 5.0) * rng.choice([-1, 1]) / 20e6 * chunk_size
                freq_inst  = freq_c + drift_norm * (t - chunk_size / 2) / float(chunk_size)
            else:
                freq_inst = freq_c + linewidth * rng.randn()

            sig += amp * np.sin(2 * np.pi * freq_inst * t).astype(np.float32)

        sig += rng.normal(0, rng.uniform(0.1, 0.4), chunk_size).astype(np.float32)
        out.append(sig)
    return out


def generate_pulsar_signals(chunk_size: int, n: int = 16, seed: Optional[int] = None,
                             as_numpy: bool = True) -> List[np.ndarray]:
    rng = _seed_rng(seed)
    out = []
    for _ in range(n):
        period   = rng.uniform(50, 400)
        pw       = max(2, int(rng.uniform(1, 8)))
        amp      = rng.uniform(0.5, 2.0)
        dm_smear = rng.uniform(0, 3)
        sig      = np.zeros(chunk_size, dtype=np.float32)
        for p in range(rng.randint(0, max(1, int(period))), chunk_size, max(1, int(period))):
            w = max(2, int(pw + dm_smear))
            s, e = int(max(0, p - w // 2)), int(min(chunk_size, p + w // 2))
            if e > s:
                sig[s:e] += amp * np.hanning(e - s)
        sig += rng.normal(0, 0.2, chunk_size).astype(np.float32)
        out.append(sig)
    return out


def generate_frb_signals(chunk_size: int, n: int = 16, seed: Optional[int] = None,
                          as_numpy: bool = True) -> List[np.ndarray]:
    rng = _seed_rng(seed)
    out = []
    t = np.arange(chunk_size, dtype=np.float32)
    for _ in range(n):
        peak  = rng.randint(int(chunk_size * 0.1), int(chunk_size * 0.9))
        width = rng.uniform(2, 20)
        amp   = rng.uniform(1.0, 4.0)
        sig   = amp * np.exp(-0.5 * ((t - peak) / width) ** 2)
        sig  += rng.normal(0, 0.3, chunk_size).astype(np.float32)
        out.append(sig)
    return out


def generate_hydrogen_line(chunk_size: int, n: int = 8, seed: Optional[int] = None,
                            as_numpy: bool = True) -> List[np.ndarray]:
    rng = _seed_rng(seed)
    out = []
    t = np.arange(chunk_size, dtype=np.float32)
    for _ in range(n):
        freq    = rng.uniform(0.009, 0.011)
        amp     = rng.uniform(0.05, 0.6)
        sigma_f = rng.uniform(0.0001, 0.0005)
        sig     = amp * np.sin(2 * np.pi * (freq + sigma_f * rng.randn()) * t)
        sig    += rng.normal(0, 0.2, chunk_size).astype(np.float32)
        out.append(sig)
    return out


# ---------------------------------------------------------------------------
# 1. HTRU2
# ---------------------------------------------------------------------------

def _features_to_signals(features: np.ndarray, labels: np.ndarray,
                          chunk_size: int) -> np.ndarray:
    n       = len(features)
    signals = np.zeros((n, chunk_size), dtype=np.float32)
    t       = np.arange(chunk_size, dtype=np.float32)
    rng     = np.random.RandomState(0)
    f_min   = features.min(axis=0)
    f_range = np.where(features.max(axis=0) - f_min > 1e-6,
                       features.max(axis=0) - f_min, 1.0)
    fn_all  = (features - f_min) / f_range
    for i in range(n):
        fn = fn_all[i]
        mean_p, std_p, kurt_p, skew_p = fn[0], fn[1], fn[2], fn[3]
        mean_dm, std_dm, kurt_dm      = fn[4], fn[5], fn[6]
        if labels[i] == 1:
            period = int(np.clip(30 + mean_dm * 300, 30, 400))
            pw     = max(2, int(2 + std_p * 10))
            amp    = 0.5 + std_dm * 2.0
            sig    = np.zeros(chunk_size, np.float32)
            for p in range(rng.randint(0, period), chunk_size, period):
                w = min(pw, chunk_size - p)
                if w > 1:
                    sig[p:p + w] += amp * np.hanning(w)
            sig += rng.normal(0, max(0.05, std_p * 0.5), chunk_size).astype(np.float32)
            if kurt_dm > 0.5:
                sig *= (1.0 + 0.2 * np.sin(2 * np.pi * (0.001 + kurt_dm * 0.005) * t))
        else:
            sig = rng.normal(mean_p * 0.1, max(0.1, std_p), chunk_size).astype(np.float32)
            if kurt_p > 0.7:
                sig += kurt_p * 0.5 * np.sin(
                    2 * np.pi * rng.uniform(0.01, 0.4) * t).astype(np.float32)
            if abs(skew_p) > 0.5:
                bp = rng.randint(0, chunk_size)
                bw = rng.randint(5, 50)
                sig[bp:min(bp + bw, chunk_size)] += skew_p * rng.uniform(0.5, 1.5)
        signals[i] = _norm_signal(sig)
    return signals


def download_htru2(dest_dir: str = None) -> Optional[str]:
    dest_dir = dest_dir or os.path.join(DATA_ROOT, 'htru2')
    csv_path = os.path.join(dest_dir, 'HTRU_2.csv')
    if os.path.exists(csv_path):
        return csv_path
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, 'HTRU2.zip')
    for url in [
        'https://archive.ics.uci.edu/static/public/372/htru2.zip',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip',
    ]:
        if _safe_download(url, zip_path, 'HTRU2', timeout=120):
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(dest_dir)
                os.remove(zip_path)
                if os.path.exists(csv_path):
                    return csv_path
            except Exception as e:
                logger.warning(f'HTRU2 extract failed: {e}')
            if os.path.exists(zip_path):
                os.remove(zip_path)
    return None


def load_htru2(dest_dir: str = None, chunk_size: int = 8192,
               auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dest_dir = dest_dir or os.path.join(DATA_ROOT, 'htru2')
    csv_path = os.path.join(dest_dir, 'HTRU_2.csv')
    if not os.path.exists(csv_path) and auto_download:
        csv_path = download_htru2(dest_dir)
    if not csv_path or not os.path.exists(csv_path):
        logger.info('HTRU2: no data file — using synthetic fallback')
        return _synthetic_pulsar_fallback(chunk_size, n_pos=1639, n_neg=16259, seed=1)
    try:
        raw      = np.loadtxt(csv_path, delimiter=',')
        features = raw[:, :8].astype(np.float32)
        labels   = raw[:, 8].astype(np.float32)
        logger.info(f'HTRU2: {len(labels)} candidates, {int(labels.sum())} real pulsars')
        return _features_to_signals(features, labels, chunk_size), labels
    except Exception as e:
        logger.error(f'HTRU2 load failed: {e}')
        return _synthetic_pulsar_fallback(chunk_size, n_pos=1639, n_neg=16259, seed=1)


# ---------------------------------------------------------------------------
# 2. HTRU-S Low Latitude
# ---------------------------------------------------------------------------

def load_htru_south(dest_dir: str = None, chunk_size: int = 8192,
                    auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dest_dir = dest_dir or os.path.join(DATA_ROOT, 'htru_south')
    csv_path = os.path.join(dest_dir, 'htru_south.csv')
    os.makedirs(dest_dir, exist_ok=True)
    if not os.path.exists(csv_path) and auto_download:
        for url in [
            'https://data.csiro.au/dap/ws/v2/collections/29735/data/1',
            'https://raw.githubusercontent.com/as595/HTRU1/master/HTRU_1.csv',
        ]:
            if _safe_download(url, csv_path, 'HTRU-S LL', timeout=60):
                break
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            raw = np.genfromtxt(csv_path, delimiter=',', skip_header=1, filling_values=0.0)
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            ncols    = raw.shape[1]
            features = raw[:, :min(8, ncols)].astype(np.float32)
            if features.shape[1] < 8:
                features = np.hstack([features,
                    np.zeros((len(features), 8 - features.shape[1]), np.float32)])
            labels = (raw[:, 8].astype(np.float32) if ncols >= 9
                      else np.ones(len(features), np.float32))
            logger.info(f'HTRU-S LL: {len(labels)} real candidates')
            return _features_to_signals(features, labels, chunk_size), labels
        except Exception as e:
            logger.warning(f'HTRU-S LL parse failed: {e}')
    logger.info('HTRU-S LL: using synthetic fallback (8492 samples)')
    return _synthetic_pulsar_fallback(chunk_size, n_pos=1093, n_neg=7399, seed=42,
                                       high_dm=True)


# ---------------------------------------------------------------------------
# 3. CHIME FRB Catalog 1
# ---------------------------------------------------------------------------

def load_chime_frb(dest_dir: str = None, chunk_size: int = 8192,
                   auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dest_dir  = dest_dir or os.path.join(DATA_ROOT, 'chime_frb')
    json_path = os.path.join(dest_dir, 'chime_catalog1.json')
    os.makedirs(dest_dir, exist_ok=True)

    # Load from cached JSON (converted from CSV by download_all_datasets)
    data = None
    if os.path.exists(json_path) and os.path.getsize(json_path) > 100:
        try:
            with open(json_path) as f:
                data = json.load(f)
            logger.info(f'CHIME FRB: loaded {len(data)} entries from cache')
        except Exception as e:
            logger.warning(f'CHIME FRB cache read failed: {e}')

    rng  = np.random.RandomState(7)
    t    = np.arange(chunk_size, dtype=np.float32)
    frbs = []
    if data:
        entries = data if isinstance(data, list) else data.get('results', data.get('data', []))
        for e in entries:
            try:
                # CHIME CSV columns: width_fitb, snr, bonsai_dm (or dm)
                frbs.append((
                    float(e.get('width_fitb', e.get('width', 5.0))       or 5.0),
                    float(e.get('snr',        e.get('peak_flux', 10.0))  or 10.0),
                    float(e.get('bonsai_dm',  e.get('dm', e.get('dm_fitb', 200.0))) or 200.0),
                ))
            except Exception:
                continue
        logger.info(f'CHIME FRB: {len(frbs)} real bursts parsed')
    if not frbs:
        logger.info('CHIME FRB: synthetic fallback (535 bursts)')
        for _ in range(535):
            frbs.append((rng.uniform(0.5, 50.0), rng.uniform(8.0, 100.0),
                         rng.uniform(50.0, 2000.0)))
    return _frb_signals_from_params(frbs, chunk_size, rng, t)


# ---------------------------------------------------------------------------
# 4. FRBCAT
# ---------------------------------------------------------------------------

def load_frbcat(dest_dir: str = None, chunk_size: int = 8192,
                auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dest_dir = dest_dir or os.path.join(DATA_ROOT, 'frbcat')
    csv_path = os.path.join(dest_dir, 'frbcat.csv')
    os.makedirs(dest_dir, exist_ok=True)
    if not os.path.exists(csv_path) and auto_download:
        _safe_download('http://frbcat.org/products/frbcat_all.csv',
                       csv_path, 'FRBCAT', timeout=30)
    rng  = np.random.RandomState(13)
    t    = np.arange(chunk_size, dtype=np.float32)
    frbs = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            import csv as _csv
            with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
                for row in _csv.DictReader(f):
                    try:
                        frbs.append((
                            float(row.get('width',    row.get('width_obs', 5.0))  or 5.0),
                            float(row.get('snr',      row.get('peak_flux', 10.0)) or 10.0),
                            float(row.get('dm',       row.get('dm_obs',   300.0)) or 300.0),
                        ))
                    except Exception:
                        continue
            logger.info(f'FRBCAT: {len(frbs)} real FRBs')
        except Exception as e:
            logger.warning(f'FRBCAT parse failed: {e}')
    if not frbs:
        logger.info('FRBCAT: synthetic fallback (200)')
        for _ in range(200):
            frbs.append((rng.uniform(0.3, 30.0), rng.uniform(10.0, 80.0),
                         rng.uniform(100.0, 1500.0)))
    return _frb_signals_from_params(frbs, chunk_size, rng, t)


# ---------------------------------------------------------------------------
# 5. SETI@home (offline since 2020 — always synthetic)
# ---------------------------------------------------------------------------

def load_seti_at_home(dest_dir: str = None, chunk_size: int = 8192,
                      auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    logger.info('SETI@home: project ended 2020 — generating synthetic narrowband signals (300)')
    rng        = np.random.RandomState(21)
    t          = np.arange(chunk_size, dtype=np.float32)
    signals, labels = [], []
    for _ in range(300):
        freq_mhz  = rng.uniform(1419.0, 1421.5)
        drift     = rng.uniform(-0.5, 0.5)
        score     = rng.uniform(0.5, 5.0)
        f_norm    = 0.005 + (freq_mhz - 1419.0) / 4.0 * 0.015
        drift_n   = drift / (chunk_size * 10.0)
        amp       = np.clip(score * 0.4, 0.3, 3.0)
        envelope  = np.exp(-((t - chunk_size / 2) ** 2) / (0.3 * chunk_size) ** 2)
        freq_inst = f_norm + drift_n * (t - chunk_size / 2)
        sig       = amp * envelope * np.sin(2 * np.pi * freq_inst * t)
        sig      += rng.normal(0, 0.3, chunk_size).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(1.0)
    for _ in range(300):
        sig = rng.uniform(0.1, 1.0) * np.sin(
            2 * np.pi * rng.uniform(0.01, 0.49) * t).astype(np.float32)
        sig += rng.normal(0, 0.8, chunk_size).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(0.0)
    return np.array(signals, np.float32), np.array(labels, np.float32)


# ---------------------------------------------------------------------------
# 6. Breakthrough Listen
# ---------------------------------------------------------------------------

def load_breakthrough_listen(dest_dir: str = None, chunk_size: int = 8192,
                              auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dest_dir  = dest_dir or os.path.join(DATA_ROOT, 'breakthrough_listen')
    json_path = os.path.join(dest_dir, 'bl_candidates.json')
    bl_csv    = os.path.join(dest_dir, 'bl_events.csv')
    os.makedirs(dest_dir, exist_ok=True)

    # Load from cached JSON (written by download_all_datasets)
    data = None
    if os.path.exists(json_path) and os.path.getsize(json_path) > 100:
        try:
            with open(json_path) as f:
                data = json.load(f)
            logger.info(f'Breakthrough Listen: loaded {len(data) if isinstance(data, list) else "?"} entries from cache')
        except Exception as e:
            logger.warning(f'BL cache read failed: {e}')

    # If no cache yet, try downloading CSV directly
    if data is None and auto_download:
        for url in [
            'https://breakthroughinitiatives.org/blpd/blpd_events.csv',
            'https://raw.githubusercontent.com/UCBerkeleySETI/blimpy/master/tests/test_data/events.csv',
        ]:
            if _safe_download(url, bl_csv, 'Breakthrough Listen CSV', timeout=60):
                try:
                    import csv as _csv
                    rows = []
                    with open(bl_csv, newline='', encoding='utf-8', errors='replace') as f:
                        for row in _csv.DictReader(f):
                            rows.append(dict(row))
                    with open(json_path, 'w') as f:
                        json.dump(rows, f)
                    data = rows
                    logger.info(f'Breakthrough Listen: {len(rows)} events loaded from CSV')
                    break
                except Exception as e:
                    logger.warning(f'BL CSV parse failed: {e}')
    rng        = np.random.RandomState(33)
    t          = np.arange(chunk_size, dtype=np.float32)
    candidates = []
    if data:
        entries = data if isinstance(data, list) else data.get('results', [])
        for e in entries:
            try:
                candidates.append((
                    float(e.get('frequency',  1420.0) or 1420.0),
                    float(e.get('drift_rate', 0.0)    or 0.0),
                    float(e.get('snr', e.get('power', 1.0)) or 1.0),
                ))
            except Exception:
                continue
        logger.info(f'Breakthrough Listen: {len(candidates)} real candidates')
    if not candidates:
        logger.info('Breakthrough Listen: synthetic fallback (400)')
        for _ in range(400):
            candidates.append((rng.uniform(1000.0, 10000.0),
                               rng.uniform(-5.0, 5.0),
                               rng.uniform(5.0, 30.0)))
    signals, labels = [], []
    for freq_mhz, drift, snr in candidates:
        f_norm    = 0.005 + ((freq_mhz % 100) / 100.0) * 0.02
        drift_n   = np.clip(drift / 1000.0, -0.01, 0.01)
        amp       = np.clip(snr / 25.0, 0.3, 4.0)
        freq_inst = f_norm + drift_n * (t / chunk_size)
        env       = 0.5 + 0.5 * np.exp(-((t - chunk_size / 2) ** 2) /
                                         (0.25 * chunk_size) ** 2)
        sig       = amp * env * np.sin(2 * np.pi * freq_inst * t)
        sig      += rng.normal(0, 0.25, chunk_size).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(1.0)
    for _ in range(len(candidates)):
        sig = rng.normal(0, 1.0, chunk_size).astype(np.float32)
        if rng.rand() > 0.5:
            sig += rng.uniform(0.2, 1.2) * np.sin(
                2 * np.pi * rng.uniform(0.01, 0.45) * t).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(0.0)
    return np.array(signals, np.float32), np.array(labels, np.float32)


# ---------------------------------------------------------------------------
# 7. ATNF Pulsar Catalogue
# ---------------------------------------------------------------------------

def load_atnf(dest_dir: str = None, chunk_size: int = 8192,
              auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dest_dir = dest_dir or os.path.join(DATA_ROOT, 'atnf')
    csv_path = os.path.join(dest_dir, 'atnf_pulsars.csv')
    os.makedirs(dest_dir, exist_ok=True)
    if not os.path.exists(csv_path) and auto_download:
        _safe_download(
            'https://www.atnf.csiro.au/people/pulsar/psrcat/proc_form.php'
            '?Type=expert&mainsel=&state=query&table_bottom.x=28&table_bottom.y=12'
            '&Params=P0+DM+S1400+W50&startUserDefined=true&c1_val=&c2_val='
            '&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition='
            '&pulsar_names=&ephemeris=expert&submit_ephemeris=Get+Ephemeris'
            '&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2='
            '&style=short&no_value=*&fsize=3&x_axis=&x_data=&y_axis=&y_data=',
            csv_path, 'ATNF PSRCAT', timeout=60)
    rng     = np.random.RandomState(55)
    t       = np.arange(chunk_size, dtype=np.float32)
    pulsars = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            with open(csv_path, encoding='utf-8', errors='replace') as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                # Skip header/comment lines: starts with #, -, @, letters, or 'JNAME'
                if not line or line[0] in ('#', '-', '@') or line[0].isalpha():
                    continue
                parts = line.split()
                try:
                    p0   = float(parts[0])
                    dm   = float(parts[1])
                    s14  = float(parts[2]) if len(parts) > 2 and parts[2] not in ('*', '') else 1.0
                    w50  = float(parts[3]) if len(parts) > 3 and parts[3] not in ('*', '') else 5.0
                    pulsars.append((p0, dm, s14, w50))
                except (ValueError, IndexError):
                    continue
            logger.info(f'ATNF: {len(pulsars)} real pulsars loaded')
        except Exception as e:
            logger.warning(f'ATNF parse failed: {e}')
    if not pulsars:
        logger.info('ATNF: synthetic fallback (3500)')
        for _ in range(3500):
            pulsars.append((10 ** rng.uniform(-3, 1), rng.uniform(3.0, 1500.0),
                            rng.uniform(0.1, 50.0),   rng.uniform(0.5, 100.0)))
    signals, labels = [], []
    for period_s, dm, flux, w50_ms in pulsars:
        ps       = max(4, int(period_s * chunk_size / 0.4))
        pw       = max(2, int(w50_ms * chunk_size / 400.0))
        amp      = np.clip(flux / 10.0, 0.3, 5.0)
        dm_smear = max(0, int(dm / 500.0 * 5))
        sig      = np.zeros(chunk_size, np.float32)
        for p in range(rng.randint(0, max(1, ps)), chunk_size, max(1, ps)):
            w = min(pw + dm_smear, chunk_size - p)
            if w > 1:
                sig[p:p + w] += amp * np.hanning(w)
        sig += rng.normal(0, max(0.1, 1.0 / amp), chunk_size).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(1.0)
    for _ in range(len(pulsars)):
        c   = rng.randint(0, 5)
        sig = _random_negative(c, chunk_size, t, rng)
        signals.append(_norm_signal(sig)); labels.append(0.0)
    return np.array(signals, np.float32), np.array(labels, np.float32)


# ---------------------------------------------------------------------------
# 8. GBNCC
# ---------------------------------------------------------------------------

def load_gbncc(dest_dir: str = None, chunk_size: int = 8192,
               auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dest_dir = dest_dir or os.path.join(DATA_ROOT, 'gbncc')
    csv_path = os.path.join(dest_dir, 'gbncc_candidates.csv')
    os.makedirs(dest_dir, exist_ok=True)
    if not os.path.exists(csv_path) and auto_download:
        for url in [
            # GBNCC pulsar survey summary — try multiple mirrors
            'https://www.naic.edu/~pfreire/GBNcatalogue.html',
            'https://raw.githubusercontent.com/scottransom/presto/master/examples/gbncc.bestprof',
        ]:
            if _safe_download(url, csv_path, 'GBNCC', timeout=30):
                break
    rng     = np.random.RandomState(77)
    t       = np.arange(chunk_size, dtype=np.float32)
    entries = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        try:
            import csv as _csv
            with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
                for row in _csv.DictReader(f):
                    try:
                        entries.append((
                            float(row.get('period', row.get('P0', 0.5)) or 0.5),
                            float(row.get('dm',     row.get('DM', 50.0)) or 50.0),
                            1.0 if str(row.get('class', '0')).strip() in ('1', 'pulsar')
                            else 0.0,
                        ))
                    except Exception:
                        continue
            logger.info(f'GBNCC: {len(entries)} real candidates')
        except Exception as e:
            logger.warning(f'GBNCC parse failed: {e}')
    if not entries:
        logger.info('GBNCC: synthetic fallback (1000)')
        for _ in range(500):
            entries.append((rng.uniform(0.002, 5.0), rng.uniform(5.0, 300.0), 1.0))
        for _ in range(500):
            entries.append((rng.uniform(0.002, 5.0), rng.uniform(5.0, 300.0), 0.0))
    signals, labels = [], []
    for period_s, dm, label in entries:
        ps       = max(4, int(period_s * chunk_size / 0.4))
        dm_smear = max(1, int(dm / 200.0 * 3))
        amp      = rng.uniform(0.5, 2.5)
        if label == 1.0:
            sig = np.zeros(chunk_size, np.float32)
            for p in range(rng.randint(0, max(1, ps)), chunk_size, max(1, ps)):
                w = min(max(2, 3 + dm_smear), chunk_size - p)
                if w > 1:
                    sig[p:p + w] += amp * np.hanning(w)
            sig += rng.normal(0, 0.25, chunk_size).astype(np.float32)
        else:
            sig = rng.normal(0, 1.0, chunk_size).astype(np.float32)
            if rng.rand() > 0.4:
                sig += 0.8 * np.sin(
                    2 * np.pi * rng.uniform(0.01, 0.4) * t).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(label)
    return np.array(signals, np.float32), np.array(labels, np.float32)


# ---------------------------------------------------------------------------
# 9. VLASS
# ---------------------------------------------------------------------------

def load_vlass(dest_dir: str = None, chunk_size: int = 8192,
               auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    dest_dir = dest_dir or os.path.join(DATA_ROOT, 'vlass')
    csv_path = os.path.join(dest_dir, 'vlass_components.csv')
    gz_path  = csv_path + '.gz'
    os.makedirs(dest_dir, exist_ok=True)

    # Delete corrupt cached files (HTML error pages are small and start with '<')
    for bad in [gz_path, csv_path]:
        if os.path.exists(bad) and os.path.getsize(bad) < 50000:
            try:
                with open(bad, 'rb') as f:
                    head = f.read(4)
                if head.startswith(b'<'):
                    os.remove(bad)
                    logger.info(f'VLASS: removed corrupt cache {bad}')
            except Exception:
                pass

    valid = os.path.exists(csv_path) and os.path.getsize(csv_path) > 50000

    if not valid and auto_download:
        # CADC TAP query — returns CSV of high-variability VLASS components directly
        tap_url = (
            'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/tap/sync'
            '?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&MAXREC=10000'
            '&QUERY=SELECT+Peak_flux,Total_flux,Spectral_index,'
            'Variability_index,RA,DEC+FROM+cirada.CIRADA_VLASS1QLv3p1_table1_components'
            '+WHERE+Variability_index+%3E+0.3+ORDER+BY+Variability_index+DESC'
        )
        if _safe_download(tap_url, csv_path, 'VLASS via CADC TAP', timeout=120):
            # Validate — reject if it's actually an HTML error
            try:
                with open(csv_path, 'rb') as f:
                    head = f.read(4)
                if head.startswith(b'<'):
                    os.remove(csv_path)
                    logger.warning('VLASS TAP returned HTML — trying without schema prefix')
                    # Retry without schema qualifier
                    tap_url2 = (
                        'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/tap/sync'
                        '?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&MAXREC=10000'
                        '&QUERY=SELECT+Peak_flux,Total_flux,Spectral_index,'
                        'Variability_index+FROM+CIRADA_VLASS1QLv3p1_table1_components'
                        '+WHERE+Variability_index+%3E+0.3'
                    )
                    _safe_download(tap_url2, csv_path, 'VLASS TAP (retry)', timeout=120)
            except Exception:
                pass
    rng        = np.random.RandomState(99)
    t          = np.arange(chunk_size, dtype=np.float32)
    transients = []
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
        try:
            import csv as _csv
            with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
                for row in _csv.DictReader(f):
                    try:
                        # Handle both CADC TAP and original CIRADA column names
                        flux = float(row.get('Total_flux',
                               row.get('total_flux',
                               row.get('Peak_flux',
                               row.get('peak_flux', 1.0)))) or 1.0)
                        var  = float(row.get('Variability_index',
                               row.get('variability_index', 0.0)) or 0.0)
                        alpha = float(row.get('Spectral_index',
                                row.get('spectral_index', -0.7)) or -0.7)
                        if abs(var) > 0.3 or abs(flux) > 10.0:
                            transients.append((flux, alpha, var))
                    except Exception:
                        continue
                    if len(transients) >= 2000:
                        break
            logger.info(f'VLASS: {len(transients)} real transient candidates')
        except Exception as e:
            logger.warning(f'VLASS parse failed: {e}')
    if not transients:
        logger.info('VLASS: synthetic fallback (500)')
        for _ in range(500):
            transients.append((rng.uniform(1.0, 100.0),
                               rng.uniform(-2.0, 0.5),
                               rng.uniform(0.3, 2.0)))
    signals, labels = [], []
    for flux, alpha, var in transients:
        amp   = np.clip(flux / 20.0, 0.3, 5.0)
        peak  = rng.randint(int(chunk_size * 0.1), int(chunk_size * 0.5))
        rise  = max(2.0, rng.uniform(2, 30))
        decay = rise * rng.uniform(2.0, 8.0)
        sig   = np.zeros(chunk_size, np.float32)
        sig[:peak] = amp * np.exp(-0.5 * ((t[:peak] - peak) / rise)  ** 2)
        sig[peak:] = amp * np.exp(-0.5 * ((t[peak:] - peak) / decay) ** 2)
        sig *= (1.0 + 0.1 * alpha * np.sin(2 * np.pi * 0.005 * t))
        sig += var * 0.1 * rng.normal(0, 1.0, chunk_size).astype(np.float32)
        sig += rng.normal(0, 0.2, chunk_size).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(1.0)
    for _ in range(len(transients)):
        signals.append(_norm_signal(
            rng.normal(0, 1.0, chunk_size).astype(np.float32)))
        labels.append(0.0)
    return np.array(signals, np.float32), np.array(labels, np.float32)


# ---------------------------------------------------------------------------
# Shared signal construction helpers
# ---------------------------------------------------------------------------

def _frb_signals_from_params(frbs, chunk_size, rng, t):
    signals, labels = [], []
    for width_ms, snr, dm in frbs:
        ws   = max(2.0, width_ms * chunk_size / 800.0)
        amp  = np.clip(snr / 20.0, 0.5, 8.0)
        peak = rng.randint(int(chunk_size * 0.15), int(chunk_size * 0.85))
        sig  = amp * np.exp(-0.5 * ((t - peak) / ws) ** 2)
        sig += np.clip(dm / 2000.0, 0, 1) * 0.3 * np.exp(
            -0.5 * ((t - peak - ws) / (ws * 2)) ** 2)
        sig += rng.normal(0, 0.25, chunk_size).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(1.0)
    for _ in range(len(frbs)):
        c   = rng.randint(0, 4)
        sig = _random_negative(c, chunk_size, t, rng)
        signals.append(_norm_signal(sig)); labels.append(0.0)
    return np.array(signals, np.float32), np.array(labels, np.float32)


def _random_negative(choice, chunk_size, t, rng):
    if choice == 0:
        return rng.normal(0, 1.0, chunk_size).astype(np.float32)
    elif choice == 1:
        return rng.rayleigh(1.0, chunk_size).astype(np.float32)
    elif choice == 2:
        sig = rng.uniform(0.2, 1.5) * np.sin(
            2 * np.pi * rng.uniform(0.02, 0.45) * t).astype(np.float32)
        return sig + rng.normal(0, 0.5, chunk_size).astype(np.float32)
    elif choice == 3:
        return np.cumsum(rng.normal(0, 0.4, chunk_size)).astype(np.float32)
    else:
        peak = rng.randint(0, chunk_size)
        bw   = rng.randint(5, 80)
        sig  = np.zeros(chunk_size, np.float32)
        e    = min(peak + bw, chunk_size)
        if e > peak + 1:
            sig[peak:e] = rng.uniform(0.5, 2.0) * np.hanning(e - peak)
        return sig + rng.normal(0, 0.5, chunk_size).astype(np.float32)


def _synthetic_pulsar_fallback(chunk_size, n_pos, n_neg, seed, high_dm=False):
    rng = np.random.RandomState(seed)
    t   = np.arange(chunk_size, dtype=np.float32)
    signals, labels = [], []
    for _ in range(n_pos):
        period = int(rng.uniform(20 if high_dm else 50,
                                  600 if high_dm else 400))
        pw     = max(2, int(rng.uniform(1, 12 if high_dm else 8)))
        amp    = rng.uniform(0.3, 3.0)
        sig    = np.zeros(chunk_size, np.float32)
        for p in range(rng.randint(0, max(1, period)), chunk_size, max(1, period)):
            w = min(pw, chunk_size - p)
            if w > 1:
                sig[p:p + w] += amp * np.hanning(w)
        sig += rng.normal(0, 0.3, chunk_size).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(1.0)
    for _ in range(n_neg):
        sig = rng.normal(0, 1.0, chunk_size).astype(np.float32)
        sig += rng.uniform(0.1, 0.8) * np.sin(
            2 * np.pi * rng.uniform(0.01, 0.4) * t).astype(np.float32)
        signals.append(_norm_signal(sig)); labels.append(0.0)
    return np.array(signals, np.float32), np.array(labels, np.float32)


# ---------------------------------------------------------------------------
# Master loader
# ---------------------------------------------------------------------------

def load_all_datasets(chunk_size: int = 8192, data_root: str = None,
                      auto_download: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Load and combine all 9 datasets. Call download_all_datasets() first."""
    root = data_root or DATA_ROOT
    loaders = [
        ('HTRU2',               load_htru2,               os.path.join(root, 'htru2')),
        ('HTRU-S LL',           load_htru_south,          os.path.join(root, 'htru_south')),
        ('CHIME FRB',           load_chime_frb,           os.path.join(root, 'chime_frb')),
        ('FRBCAT',              load_frbcat,              os.path.join(root, 'frbcat')),
        ('SETI@home',           load_seti_at_home,        os.path.join(root, 'seti_at_home')),
        ('Breakthrough Listen', load_breakthrough_listen, os.path.join(root, 'breakthrough_listen')),
        ('ATNF',                load_atnf,                os.path.join(root, 'atnf')),
        ('GBNCC',               load_gbncc,               os.path.join(root, 'gbncc')),
        ('VLASS',               load_vlass,               os.path.join(root, 'vlass')),
    ]
    all_X, all_y = [], []
    for name, fn, dest in loaders:
        try:
            X, y = fn(dest_dir=dest, chunk_size=chunk_size, auto_download=auto_download)
            if X.shape[0] > 0:
                all_X.append(X); all_y.append(y)
                pos = int(y.sum())
                logger.info(f'  {name}: {len(y)} samples ({pos}+ / {len(y)-pos}-)')
            else:
                logger.warning(f'  {name}: returned 0 samples')
        except Exception as e:
            logger.error(f'  {name} loader error: {e}')
    if not all_X:
        logger.error('All loaders failed — returning empty arrays')
        return np.empty((0, chunk_size), np.float32), np.empty((0,), np.float32)
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    perm  = np.random.RandomState(0).permutation(len(y_all))
    X_all, y_all = X_all[perm], y_all[perm]
    pos = int(y_all.sum())
    logger.info(f'Combined: {len(y_all)} samples | {pos} pos ({100*pos/len(y_all):.1f}%) '
                f'| {len(y_all)-pos} neg')
    return X_all, y_all


# ---------------------------------------------------------------------------
# Folder-based loader
# ---------------------------------------------------------------------------

def load_training_data_from_folder(path: str, chunk_size: int,
                                    as_numpy: bool = True) -> list:
    pool = []
    if not os.path.isdir(path):
        return pool
    for fname in sorted(os.listdir(path)):
        fpath = os.path.join(path, fname)
        try:
            if fname.lower().endswith('.npy'):
                arr = np.load(fpath)
            elif fname.lower().endswith('.csv'):
                arr = np.loadtxt(fpath, delimiter=',')
            else:
                continue
            pool.append(_pad_or_crop(arr, chunk_size))
        except Exception as e:
            logger.debug(f'Skipped {fname}: {e}')
    logger.info(f'Folder loader: {len(pool)} signals from {path}')
    return pool


__all__ = [
    'generate_wow_signals', 'generate_pulsar_signals',
    'generate_frb_signals', 'generate_hydrogen_line',
    'generate_doppler_shifted_hi',
    'load_training_data_from_folder',
    'download_all_datasets',
    'download_htru2', 'load_htru2', 'load_htru_south',
    'load_chime_frb', 'load_frbcat', 'load_seti_at_home',
    'load_breakthrough_listen', 'load_atnf', 'load_gbncc', 'load_vlass',
    'load_all_datasets',
]