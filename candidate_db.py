"""candidate_db.py — SQLite candidate event database.

Every detection above threshold is logged here with full metadata so
detections are queryable, exportable, and comparable across sessions.

Schema
------
candidates table:
    id              INTEGER PRIMARY KEY
    timestamp_utc   TEXT     ISO-8601 UTC
    confidence      REAL     model output [0-1]
    signal_strength_db REAL  RMS power in dB
    flux_jy         REAL     estimated flux density (Jy), NULL if uncalibrated
    t_ant_k         REAL     antenna temperature (K), NULL if uncalibrated
    hi_peak_vel_kms REAL     HI peak velocity in LSR frame (km/s)
    hi_snr          REAL     HI line SNR
    ra_deg          REAL     Right Ascension J2000
    dec_deg         REAL     Declination J2000
    glon_deg        REAL     Galactic longitude
    glat_deg        REAL     Galactic latitude
    az_deg          REAL     Antenna azimuth
    el_deg          REAL     Antenna elevation
    lsr_corr_kms    REAL     LSR velocity correction applied
    freq_center_mhz REAL     SDR centre frequency
    bandwidth_mhz   REAL     SDR bandwidth
    fits_path       TEXT     path to saved FITS file (NULL if not saved)
    notes           TEXT     free text
    flagged         INTEGER  0=candidate, 1=confirmed, -1=RFI/rejected

Usage
-----
    from candidate_db import CandidateDB
    db = CandidateDB('output/candidates.db')
    db.insert(confidence=0.92, signal_strength_db=-45.3, ...)
    rows = db.recent(n=10)
    db.export_csv('candidates.csv')
"""

import os
import sqlite3
import logging
import csv
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

logger = logging.getLogger('aic.candidate_db')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc       TEXT    NOT NULL,
    confidence          REAL    NOT NULL,
    signal_strength_db  REAL,
    flux_jy             REAL,
    t_ant_k             REAL,
    hi_peak_vel_kms     REAL,
    hi_snr              REAL,
    ra_deg              REAL,
    dec_deg             REAL,
    glon_deg            REAL,
    glat_deg            REAL,
    az_deg              REAL,
    el_deg              REAL,
    lsr_corr_kms        REAL,
    freq_center_mhz     REAL,
    bandwidth_mhz       REAL,
    fits_path           TEXT,
    notes               TEXT,
    flagged             INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_timestamp ON candidates(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_confidence ON candidates(confidence);
CREATE INDEX IF NOT EXISTS idx_flagged ON candidates(flagged);
"""


class CandidateDB:
    """Thread-safe SQLite candidate event store."""

    def __init__(self, db_path: str = 'output/candidates.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        # Persistent connection — avoids open/close overhead on every insert.
        # check_same_thread=False is safe because all writes come from the
        # single processing thread; reads from the web thread use a separate
        # short-lived connection via _read_connect().
        self._write_conn: sqlite3.Connection = sqlite3.connect(
            self.db_path, timeout=10, check_same_thread=False)
        self._write_conn.row_factory = sqlite3.Row
        self._write_conn.execute('PRAGMA journal_mode=WAL')
        self._write_conn.execute('PRAGMA synchronous=NORMAL')
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Return the persistent write connection."""
        return self._write_conn

    def _read_connect(self) -> sqlite3.Connection:
        """Short-lived read connection for web thread queries."""
        conn = sqlite3.connect(self.db_path, timeout=5, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def _init_db(self) -> None:
        self._write_conn.executescript(_SCHEMA)
        self._write_conn.commit()
        logger.info(f'Candidate DB initialised: {self.db_path}')

    def insert(self,
               confidence:          float,
               timestamp_utc:       Optional[datetime] = None,
               signal_strength_db:  Optional[float] = None,
               flux_jy:             Optional[float] = None,
               t_ant_k:             Optional[float] = None,
               hi_peak_vel_kms:     Optional[float] = None,
               hi_snr:              Optional[float] = None,
               ra_deg:              Optional[float] = None,
               dec_deg:             Optional[float] = None,
               glon_deg:            Optional[float] = None,
               glat_deg:            Optional[float] = None,
               az_deg:              Optional[float] = None,
               el_deg:              Optional[float] = None,
               lsr_corr_kms:        Optional[float] = None,
               freq_center_mhz:     Optional[float] = None,
               bandwidth_mhz:       Optional[float] = None,
               fits_path:           Optional[str]   = None,
               notes:               Optional[str]   = None) -> int:
        """Insert a detection event. Returns the new row id."""
        ts = (timestamp_utc or datetime.now(timezone.utc)).strftime(
            '%Y-%m-%dT%H:%M:%S.%f')[:-3]
        row = (ts, confidence, signal_strength_db, flux_jy, t_ant_k,
               hi_peak_vel_kms, hi_snr, ra_deg, dec_deg,
               glon_deg, glat_deg, az_deg, el_deg, lsr_corr_kms,
               freq_center_mhz, bandwidth_mhz, fits_path, notes, 0)
        try:
            with self._connect() as conn:
                cur = conn.execute("""
                    INSERT INTO candidates (
                        timestamp_utc, confidence, signal_strength_db,
                        flux_jy, t_ant_k, hi_peak_vel_kms, hi_snr,
                        ra_deg, dec_deg, glon_deg, glat_deg,
                        az_deg, el_deg, lsr_corr_kms,
                        freq_center_mhz, bandwidth_mhz,
                        fits_path, notes, flagged
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, row)
                rowid = cur.lastrowid
            logger.info(
                f'Candidate #{rowid} logged: conf={confidence:.3f} '
                f'HI_vel={hi_peak_vel_kms:.1f}km/s SNR={hi_snr:.1f}'
            )
            return rowid
        except Exception as e:
            logger.error(f'Failed to insert candidate: {e}')
            return -1

    def flag(self, row_id: int, status: int, notes: Optional[str] = None) -> None:
        """Update flagged status: 1=confirmed, -1=rejected, 0=candidate."""
        try:
            with self._connect() as conn:
                if notes:
                    conn.execute(
                        'UPDATE candidates SET flagged=?, notes=? WHERE id=?',
                        (status, notes, row_id))
                else:
                    conn.execute(
                        'UPDATE candidates SET flagged=? WHERE id=?',
                        (status, row_id))
        except Exception as e:
            logger.error(f'Failed to flag candidate {row_id}: {e}')

    def recent(self, n: int = 20, min_confidence: float = 0.0,
               flagged: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return the N most recent candidates as list of dicts."""
        try:
            with self._read_connect() as conn:
                where = 'WHERE confidence >= ?'
                params: list = [min_confidence]
                if flagged is not None:
                    where += ' AND flagged = ?'
                    params.append(flagged)
                rows = conn.execute(
                    f'SELECT * FROM candidates {where} '
                    f'ORDER BY timestamp_utc DESC LIMIT ?',
                    params + [n]
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f'DB query failed: {e}')
            return []

    def count(self) -> Dict[str, int]:
        """Return counts by flag status."""
        try:
            with self._read_connect() as conn:
                rows = conn.execute(
                    'SELECT flagged, COUNT(*) as n FROM candidates GROUP BY flagged'
                ).fetchall()
            counts = {'candidate': 0, 'confirmed': 0, 'rejected': 0, 'total': 0}
            for r in rows:
                f, n = r['flagged'], r['n']
                counts['total'] += n
                if f == 0:  counts['candidate'] += n
                elif f == 1: counts['confirmed'] += n
                elif f == -1: counts['rejected'] += n
            return counts
        except Exception as e:
            logger.error(f'DB count failed: {e}')
            return {}

    def export_csv(self, path: str) -> int:
        """Export all candidates to CSV. Returns number of rows written."""
        try:
            with self._read_connect() as conn:
                rows = conn.execute(
                    'SELECT * FROM candidates ORDER BY timestamp_utc'
                ).fetchall()
            if not rows:
                return 0
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows([dict(r) for r in rows])
            logger.info(f'Exported {len(rows)} candidates to {path}')
            return len(rows)
        except Exception as e:
            logger.error(f'CSV export failed: {e}')
            return 0

    def hi_velocity_histogram(self, bins: int = 50,
                               v_range: tuple = (-300, 300)) -> Dict[str, list]:
        """Return velocity histogram of all HI detections for analysis."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    'SELECT hi_peak_vel_kms FROM candidates '
                    'WHERE hi_peak_vel_kms IS NOT NULL AND flagged >= 0'
                ).fetchall()
            vels = [r['hi_peak_vel_kms'] for r in rows]
            if not vels:
                return {'bins': [], 'counts': []}
            counts, edges = np.histogram(vels, bins=bins, range=v_range)
            import numpy as np
            centers = ((edges[:-1] + edges[1:]) / 2).tolist()
            return {'bins': centers, 'counts': counts.tolist()}
        except Exception as e:
            logger.error(f'Histogram failed: {e}')
            return {'bins': [], 'counts': []}
