import os
import sys
import numpy as np
import tensorflow as tf
# Ensure repo root is on sys.path for local imports
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from processing import _prepare_input_for_model, predict_signal, create_model
from advance_signal_processing import denoise_signal, denoise_preserve_spikes

MODEL_PATH = os.path.join('models', 'full_state', 'full_model.keras')

print('Loading model...')
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print('Model loaded from', MODEL_PATH)
else:
    print('Model file not found, creating new model')
    model = create_model()


def run_sample(name, samples):
    print('\n---', name, '---')
    print('raw dtype', samples.dtype, 'shape', samples.shape, 'min/max/std', np.min(samples), np.max(samples), np.std(samples))
    # Denoise
    try:
        den = denoise_signal(samples)
    except Exception as e:
        print('denoise_signal failed:', e)
        den = samples
    print('denoised min/max/std', np.min(den), np.max(den), np.std(den))
    # preserve spikes variant
    try:
        denp = denoise_preserve_spikes(samples)
        print('denoise_preserve_spikes min/max/std', np.min(denp), np.max(denp), np.std(denp))
    except Exception as e:
        print('denoise_preserve_spikes failed:', e)

    # Prepare for model
    inp, norm = _prepare_input_for_model(den, expected_len=8192, return_normalized=True)
    print('prepared input dtype', inp.dtype, 'shape', inp.shape)
    print('normalized min/max/std', np.min(norm), np.max(norm), np.std(norm))

    # Model forward
    out = model(inp, training=False)
    try:
        out_np = out.numpy() if hasattr(out, 'numpy') else np.array(out)
    except Exception:
        out_np = np.array(out)
    print('model out dtype/shape/min/max', getattr(out_np, 'dtype', None), getattr(out_np, 'shape', None), np.min(out_np), np.max(out_np))
    try:
        val = float(np.ravel(out_np)[0])
    except Exception:
        val = 0.0
    print('confidence', val)
    det, conf = predict_signal(model, den)
    print('predict_signal -> detection, confidence:', det, conf)


# Noise
noise = np.random.normal(0, 1.0, 8192).astype(np.float32)
run_sample('Noise', noise)

# Low-amplitude noise
noise_small = (np.random.normal(0, 1e-4, 8192)).astype(np.float32)
run_sample('Small-noise', noise_small)

# Synthetic WOW-like: use inject function if available, else simple chirp
from processing import inject_wow_signal
wow = np.zeros(8192, dtype=np.float32)
wow = inject_wow_signal(wow, amplitude=25.0)
run_sample('Injected WOW', np.real(wow).astype(np.float32))

print('\nDone')
