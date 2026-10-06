"""Load the small public replay, without checkpoints, pickle, or model code."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def validate(arrays, metadata):
    n, neurons = metadata['sample_count'], metadata['neuron_count']
    shapes = {'neuron_ids': (neurons,), 'activity': (n, neurons),
              'camera': (n, 160, 160), 'position': (n, 3),
              'heading': (n,), 'lap': (n,), 'time_s': (n,)}
    if set(arrays) != set(shapes):
        raise ValueError('Unexpected recording arrays')
    for key, shape in shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f'Invalid shape or non-finite values: {key}')
    ids = arrays['neuron_ids']
    if ids.dtype.kind not in 'iu' or len(np.unique(ids)) != neurons or np.any(ids <= 0):
        raise ValueError('Neuron IDs must be unique positive integers')
    if arrays['camera'].dtype != np.uint8:
        raise ValueError('Camera must contain uint8 pixels')
    if np.any((arrays['activity'] < 0) | (arrays['activity'] > 1)):
        raise ValueError('Activity must use the fixed 0–1 rate scale')
    if metadata['fps'] != 20 or metadata['dt_s'] != 0.05:
        raise ValueError('This example uses the original 20 Hz sample rate')
    if not np.allclose(arrays['time_s'], np.arange(n)/20, rtol=0, atol=1e-12):
        raise ValueError('Non-contiguous recording timestamps')
    if n != 879 or not np.array_equal(arrays['lap'], np.repeat([1, 2, 3], 293)):
        raise ValueError('Expected three complete consecutive recorded laps')


def load_recording():
    meta_path = ROOT/'data/recording.json'
    metadata = json.loads(meta_path.read_text())
    if metadata['schema_version'] != 1:
        raise ValueError('Unsupported recording schema')
    path = ROOT/'data/replay.npz'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != metadata['array_sha256']:
        raise ValueError('Replay checksum mismatch; restore data/replay.npz')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    validate(arrays, metadata)
    return dict(ids=arrays['neuron_ids'], rates=arrays['activity'],
                frames=arrays['camera'][:, None], position=arrays['position'],
                heading=arrays['heading'],
                segment=np.array([f'lap{x-1}' for x in arrays['lap']]),
                sample_count=metadata['sample_count'], meta=metadata,
                source_sha256=digest)
