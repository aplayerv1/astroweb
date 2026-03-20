"""quick_retrain.py — fast retrain harness.

Usage: python quick_retrain.py
       QUICK_TRAIN_EPOCHS=10 python quick_retrain.py

Changes:
- Uses real HTRU2 data when available
- Correct create_model import from processing (not aic2)
- Saves with timestamp to avoid overwriting production model
"""
import os
import numpy as np
from datetime import datetime
from data_generation import data_generator
from processing import create_model, check_model_for_nans
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('quick_retrain')


def main():
    state_dir = os.path.join('models', 'full_state')
    os.makedirs(state_dir, exist_ok=True)

    batch_size  = int(os.getenv('QUICK_TRAIN_BATCH_SIZE', '16'))
    chunk_size  = int(os.getenv('QUICK_TRAIN_CHUNK_SIZE', '8192'))
    steps       = int(os.getenv('QUICK_TRAIN_STEPS',      '64'))
    epochs      = int(os.getenv('QUICK_TRAIN_EPOCHS',     '5'))
    use_real    = os.getenv('QUICK_USE_REAL_DATA', '1').lower() not in ('0', 'false', 'no')
    htru2_dir   = os.getenv('HTRU2_DIR', 'data/htru2')

    logger.info(f'Quick retrain: batch={batch_size}, steps={steps}, epochs={epochs}, '
                f'real_data={use_real}')

    gen = data_generator(
        chunk_size=chunk_size,
        csv_data=None,
        seed=42,
        batch_size=batch_size,
        positive_ratio=0.5,
        htru2_dir=htru2_dir,
        use_real_data=use_real,
        as_numpy=True,
    )

    model = create_model(input_channels=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    model.fit(gen, steps_per_epoch=steps, epochs=epochs, verbose=1)

    if not check_model_for_nans(model):
        logger.error('Model has NaN weights — not saving')
        return

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(state_dir, f'quick_model_{ts}.keras')
    model.save(save_path)
    logger.info(f'Saved quick model to {save_path}')

    # Also update the canonical model if it doesn't exist yet
    canonical = os.path.join(state_dir, 'full_model.keras')
    if not os.path.exists(canonical):
        model.save(canonical)
        logger.info(f'Saved as canonical model: {canonical}')


if __name__ == '__main__':
    main()
