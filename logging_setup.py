import logging
import os
from logging.handlers import RotatingFileHandler


def configure_logging(logfile=None, level=logging.DEBUG, max_bytes=100 * 1024 * 1024, backup_count=5):
    """Configure root logger with a rotating file handler and console handler.

    - logfile: path to log file (defaults to 'logs/aic.log')
    - max_bytes: rotate after this many bytes (default 100MB)
    - backup_count: number of rotated files to keep
    This function is idempotent and safe to call multiple times.
    """
    if logfile is None:
        logfile = os.path.join('logs', 'aic.log')

    # Ensure directory exists
    logdir = os.path.dirname(logfile)
    if logdir and not os.path.exists(logdir):
        try:
            os.makedirs(logdir, exist_ok=True)
        except Exception:
            pass

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers
    has_file = any(isinstance(h, RotatingFileHandler) and getattr(h, 'baseFilename', '') == os.path.abspath(logfile) for h in root.handlers)
    if not has_file:
        try:
            fh = RotatingFileHandler(logfile, maxBytes=max_bytes, backupCount=backup_count)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
            root.addHandler(fh)
        except Exception:
            # If file handler cannot be created, continue with console only
            pass

    # Ensure a console handler exists for interactive logs
    has_console = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        root.addHandler(ch)

    return root
