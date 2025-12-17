import threading
import queue
import logging
import multiprocessing as mp
import numpy as np
from collections import deque
from aic_io import save_detected_signal, plot_all

logger = logging.getLogger('aic.detection')

# Threading queue used by producers (non-blocking for SDR loop)
DETECTION_JOB_QUEUE = queue.Queue()

# Multiprocessing queue used by the heavy I/O/plot worker to avoid GIL
MP_QUEUE_MAXSIZE = 4
MP_BATCH_MAX_ITEMS = 8
MP_DETECTION_QUEUE = mp.Queue(maxsize=MP_QUEUE_MAXSIZE)
MP_ACK_QUEUE = mp.Queue()

# Limit total bytes used by queued/staged detection jobs (default 256 MiB)
MAX_DETECTION_JOB_RAM = 256 * 1024 * 1024


def _detection_worker(full_buffer, raw_snapshot, timestamp, output_dir, effective_fs):
    # Save both representations (full decimated magnitude and raw complex snapshot)
    try:
        save_detected_signal((full_buffer, raw_snapshot), timestamp, output_dir)
    except Exception as e:
        logger.error(f"Background save_detected_signal failed: {e}")

    # Plot overview products from the decimated full buffer
    try:
        from aic_io import plot_from_list
        overview_names = ['spectrogram', 'psd', 'autocorrelation', 'signal_strength', 'dynamic_spectrum']
        plot_from_list(full_buffer, effective_fs, output_dir, timestamp, overview_names)
    except Exception as e:
        logger.error(f"Background overview plotting failed: {e}")

    # Plot high-resolution products from raw complex snapshot
    try:
        highres_names = ['time_waveform', 'constellation', 'inst_freq', 'spectrogram']
        plot_from_list(raw_snapshot, effective_fs, output_dir, timestamp, highres_names)
    except Exception as e:
        logger.error(f"Background high-res plotting failed: {e}")

    logger.info("Background detection worker complete")


def _mp_detection_worker(mp_queue):
    """Process that runs heavy I/O and plotting from the multiprocessing queue."""
    logger.info("MP detection worker (process) started")
    while True:
        try:
            item = mp_queue.get()
            if item is None:
                logger.info("MP detection worker received shutdown signal")
                break

            # item is expected to be a wrapper dict: {'jobs': [...], 'bytes': N}
            if isinstance(item, dict) and 'jobs' in item:
                jobs = item['jobs']
                total_bytes = int(item.get('bytes', 0))
            else:
                # fallback: single legacy job
                jobs = [item]
                total_bytes = 0

            for j in jobs:
                try:
                    if len(j) == 4:
                        buf_copy, timestamp, output_dir, effective_fs = j
                        raw_snapshot = None
                        _detection_worker(buf_copy, raw_snapshot, timestamp, output_dir, effective_fs)
                    else:
                        full_buf, raw_snap, timestamp, output_dir, effective_fs = j
                        _detection_worker(full_buf, raw_snap, timestamp, output_dir, effective_fs)
                except Exception as e:
                    logger.error(f"Error while executing detection job in MP worker: {e}")

            # Acknowledge bytes processed back to the forwarder so it can decrement accounting
            try:
                if total_bytes > 0:
                    MP_ACK_QUEUE.put_nowait(total_bytes)
            except Exception:
                logger.exception("Failed to send ack from MP worker")
        except Exception as e:
            logger.error(f"MP detection worker encountered an error: {e}")


def _queue_forwarder(mp_queue, batch_max_items=MP_BATCH_MAX_ITEMS):
    """Thread that forwards jobs from the threading queue to the multiprocessing queue.

    When the MP queue is full we accumulate a small batch and try to push it as one item
    to reduce overhead and limit RAM growth.
    """
    logger.info("Detection queue forwarder started")
    batch = deque()
    pending_bytes = 0
    def compute_job_bytes(job):
        # Sum nbytes of any numpy arrays in the job tuple/list
        b = 0
        try:
            if isinstance(job, dict) and 'bytes' in job:
                return int(job['bytes'])
            # job is expected to be a tuple/list with arrays and other metadata
            for part in job:
                if isinstance(part, np.ndarray):
                    b += int(part.nbytes)
                elif isinstance(part, (list, tuple)):
                    for sub in part:
                        if isinstance(sub, np.ndarray):
                            b += int(sub.nbytes)
        except Exception:
            logger.exception("Failed to compute job bytes, assuming 0")
        return b

    while True:
        # drain ack queue first to keep pending_bytes accurate
        try:
            while True:
                ack = MP_ACK_QUEUE.get_nowait()
                pending_bytes = max(0, pending_bytes - int(ack))
        except Exception:
            pass
        try:
            job = DETECTION_JOB_QUEUE.get()
            if job is None:
                logger.info("Forwarder received shutdown signal")
                # flush batch if present, then signal MP worker to stop
                if batch:
                    try:
                        mp_queue.put(batch)
                    except Exception:
                        logger.exception("Failed to push final batch to MP queue")
                try:
                    mp_queue.put(None)
                except Exception:
                    logger.exception("Failed to send shutdown to MP queue")
                break

            try:
                job_bytes = compute_job_bytes(job)
                # Enforce RAM limit: if adding this job would exceed max, drop oldest in batch
                if pending_bytes + job_bytes > MAX_DETECTION_JOB_RAM:
                    logger.warning("Memory limit exceeded for detection jobs; attempting to drop oldest batch item")
                    if batch:
                        old_job, old_bytes = batch.popleft()
                        pending_bytes = max(0, pending_bytes - old_bytes)
                        logger.warning(f"Dropped oldest staged job of size {old_bytes} bytes")
                    else:
                        # No batch to drop — drop the new job instead
                        logger.error("Dropping incoming detection job due to memory limit")
                        DETECTION_JOB_QUEUE.task_done()
                        continue

                # Wrap job with size metadata so MP worker can ack bytes when done
                wrapper = {'jobs': [job], 'bytes': int(job_bytes)}

                try:
                    mp_queue.put_nowait(wrapper)
                    pending_bytes += job_bytes
                except queue.Full:
                    # MP queue is full; accumulate into batch (store job and bytes)
                    batch.append((job, job_bytes))
                    pending_bytes += job_bytes
                    if len(batch) >= batch_max_items:
                        # flush the batch as one wrapper
                        jobs = [j for (j, b) in batch]
                        total_b = sum(int(b) for (j, b) in batch)
                        batch_wrapper = {'jobs': jobs, 'bytes': int(total_b)}
                        try:
                            mp_queue.put(batch_wrapper)
                            # pending_bytes already includes these bytes
                            batch.clear()
                        except Exception:
                            logger.exception("Failed to flush batch to MP queue; will retry later")
            finally:
                DETECTION_JOB_QUEUE.task_done()
        except Exception as e:
            logger.error(f"Queue forwarder encountered an error: {e}")


def start_detection_job_consumer():
    """Start the forwarder thread and the multiprocessing detection worker.

    Returns (forwarder_thread, mp_process)
    """
    mp_proc = mp.Process(target=_mp_detection_worker, args=(MP_DETECTION_QUEUE,), daemon=True)
    mp_proc.start()

    forwarder = threading.Thread(target=_queue_forwarder, args=(MP_DETECTION_QUEUE,), daemon=True)
    forwarder.start()

    logger.info("Detection job consumer (forwarder + MP worker) started")
    return forwarder, mp_proc
