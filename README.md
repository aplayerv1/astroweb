# AIC — Real-time Signal Processing Web Interface

Brief: a lightweight Flask-based interface and continuous-processing service for ingesting SDR samples, running a trained model on short windows of data, and producing detection artifacts (plots, FITS, logs).

Status: intended for production use with an SDR backend (local HackRF or remote SDR server). This README focuses on quick setup, runtime options and environment variables used by `aic2.py`.

Prerequisites
------------
- Python 3.8+ with required packages (see `requirements.txt`).
- Optional: HackRF hardware or a compatible remote SDR server.
- Optional: a GPU-enabled TensorFlow build for better performance.

Quick setup
-----------
1. Create and activate a virtual environment:

   python -m venv env
   source env/bin/activate

2. Install dependencies:

   pip install -r requirements.txt

3. Start the web + processing stack (example):

   python aic2.py         # headless processing + web UI thread
   python web.py          # alternate: start web UI only

Runtime options (CLI & env)
---------------------------
The primary entrypoint is `aic2.py`. It accepts the following CLI arguments and also reads several environment variables:

CLI arguments (from `aic2.py`)
- `-o, --output-dir <path>`: Output directory (default: `output`).
- `--band <low|high>`: Select LNB band (default: `low`).
- `-lnboff, --lnboff`: Disable LNB handling (treat LNB as unpowered).
- `--quick-retrain`: Run quick retrain and exit.
- `--path <path>`: Data path used for training/loading (default: `data`).
- `--train`: Enable training (can also be enabled via `TRAIN` env var).
- `--fast-train`: Enable fast training (can also be enabled via `FAST_TRAIN`).
- `--force-train`: Force training (can also be enabled via `FORCE_TRAIN`).

Environment variables
- `TF_XLA_FLAGS`: XLA configuration (aic2.py defaults this to `--tf_xla_enable_xla_devices=false` when unset).
- `TF_CPP_MIN_LOG_LEVEL`: TensorFlow logging level (default `2`).
- `CONSECUTIVE_DETECTIONS`: Integer hysteresis for stable detection reporting (default `3`).
- `SDR_SERVER`: Remote SDR server address in `host:port` format (when set, `aic2.py` connects remotely instead of local HackRF).
- `TRAIN`, `FAST_TRAIN`, `FORCE_TRAIN`: Boolean-like flags (`1`, `true`, `yes`, `on`) that map to their corresponding CLI options.
- `DISABLE_ALWAYS_RUN`: Controls automatic startup tasks (unit tests, etc.). `aic2.py` sets this to `1` by default to avoid running tests in child processes; set to `0`/`false` to enable startup tasks.
- `LNB` / LNB config: `aic2.py` uses legacy module constants (`LNB_LOW`, `LNB_HIGH`) and CLI `--band`/`-lnboff` to control LNB handling; set these via the code or supply the CLI flags.

Usage examples
--------------
- Run processing and expose the web UI on all interfaces, port 8080:

  HOST=0.0.0.0 PORT=8080 python aic2.py

- Enable training by environment (equivalent to `--train`):

  TRAIN=1 python aic2.py --path data

API Endpoints
-------------
The running web UI exposes a small JSON API used by the frontend and integrations:
- `/api/plots` — list generated plots and spectrograms.
- `/api/signal` — current/latest signal processing values (confidence, strength, detection state).
- `/logs` — server-sent events endpoint streaming runtime logs.

Cleaning up disk usage
----------------------
- Output and intermediate artifacts are produced under `output/` and `temp_output/`.
- Model artifacts are stored under `models/` (e.g., `models/full_state/full_model.keras`). Remove or archive large model files if you need disk space.

Development notes
-----------------
- Startup tasks (unit tests, quick retrain) are intentionally disabled by default to avoid unexpected child-process work. See `DISABLE_ALWAYS_RUN` to control this.
- Check `aic2.py` top-of-file comments and argument parser for the most authoritative list of CLI flags and behavior.

Contributing
------------
- Fork, implement changes and open a PR. Keep changes scoped and include tests where appropriate.

License
-------
See the `LICENSE` file in this repository.


Remote SDR server (serverHRF.py)
--------------------------------
The repository includes `serverHRF.py`, a simple HackRF server that accepts a TCP client connection, receives JSON tuning parameters, configures a local HackRF, and streams complex64 samples back to the client.

Quick start (server):

  python serverHRF.py --server-address 0.0.0.0 --server-port 8888

Default options: `server-address=localhost`, `server-port=8888`, `gain=20`.

Client notes (used by `aic2.py`):
- `aic2.py` can connect to a remote SDR server by setting `SDR_SERVER=host:port` in the environment. When set, `aic2.py` sends a JSON object describing tuning parameters and then reads complex64 samples from the socket.

Example JSON tuning parameters sent by the client:

{
  "start_freq": 1420000000,
  "end_freq": 1420400000,
  "sample_rate": 20000000,
  "gain": 20,
  "duration_seconds": 0,      # 0 => continuous streaming
  "buffer_size": 8192
}

Notes & troubleshooting:
- `duration_seconds=0` requests continuous streaming.
- The server logs to `serverHRF.log` (rotating file) and stdout.
- Ensure the HackRF device is attached and accessible by the user running `serverHRF.py`.

