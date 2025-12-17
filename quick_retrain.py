"""Quick retrain harness: small, fast retrain to validate improved data pipeline.

Run with: `conda activate tf_gpu && python quick_retrain.py`
"""
import os
import numpy as np
from data_generation import data_generator
from aic2 import create_model
import tensorflow as tf


def main():
    os.makedirs(os.path.join('models', 'full_state'), exist_ok=True)
    batch_size = int(os.getenv('QUICK_TRAIN_BATCH_SIZE', '16'))
    chunk_size = int(os.getenv('QUICK_TRAIN_CHUNK_SIZE', '8192'))
    gen = data_generator(chunk_size=chunk_size, csv_data=None, seed=42, batch_size=batch_size, positive_ratio=0.5, as_numpy=True)

    model = create_model()

    # small compile for quick run
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    steps_per_epoch = int(os.getenv('QUICK_TRAIN_STEPS', '32'))
    epochs = int(os.getenv('QUICK_TRAIN_EPOCHS', '5'))

    model.fit(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1)

    save_path = os.path.join('models', 'full_state', 'quick_model.keras')
    model.save(save_path)
    print('Saved quick model to', save_path)


if __name__ == '__main__':
    main()
