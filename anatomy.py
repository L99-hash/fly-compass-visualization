"""Download and read versioned public MaleCNS geometry, verifying every file."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
import hashlib
import io
import json
import os
import struct
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest():
    return json.loads((ROOT/'data/anatomy_manifest.json').read_text())


def files(manifest):
    for rec in manifest['skeletons']:
        yield rec['file'], rec['url'], rec['sha256'], rec['bytes']
    for rec in manifest['meshes']:
        yield from zip(rec['files'], rec['urls'], rec['sha256'], rec['bytes'])


def fetch_one(cache, entry, offline=False):
    name, url, expected, size = entry
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError('Invalid anatomy cache path')
    if not url.startswith('https://storage.googleapis.com/flyem-male-cns/'):
        raise ValueError('Unexpected anatomy source')
    path = cache.joinpath(*relative.parts)
    if path.is_file() and path.stat().st_size == size and sha(path) == expected:
        return False
    if offline:
        raise ValueError(f'Missing or corrupted anatomy: {name}. Run without --offline to download it.')
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temporary = None
        try:
            request = Request(url, headers={'User-Agent': 'fly-compass-visualization/0.1'})
            with urlopen(request, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as dest:
                    temporary = Path(dest.name)
                    digest, total = hashlib.sha256(), 0
                    while block := response.read(1024*1024):
                        total += len(block)
                        if total > size:
                            raise ValueError(f'Unexpected download size: {name}')
                        digest.update(block)
                        dest.write(block)
            if total != size or digest.hexdigest() != expected:
                raise ValueError(f'Anatomy checksum mismatch: {name}')
            os.replace(temporary, path)
            return True
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt+1))


def ensure_anatomy(cache, manifest, offline=False):
    entries = list(files(manifest))
    downloaded = 0
    print(f'Checking {len(entries)} public anatomy files ({sum(e[3] for e in entries)/1e6:.1f} MB)', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one, cache, entry, offline) for entry in entries]
        for n, future in enumerate(as_completed(futures), 1):
            downloaded += int(future.result())
            if n % 100 == 0 or n == len(entries):
                print(f'Anatomy {n}/{len(entries)} verified; {downloaded} downloaded', flush=True)


def read_mesh(path, cell=1.4):
    raw = path.read_bytes()
    n = struct.unpack_from('<I', raw)[0]
    vertices = np.frombuffer(raw, '<f4', count=3*n, offset=4).reshape(-1, 3).astype(float)*.001
    faces = np.frombuffer(raw, '<u4', offset=4+12*n).reshape(-1, 3)
    # Display-only surface simplification. Neuron skeletons remain unchanged.
    _, inv = np.unique(np.floor(vertices/cell).astype(np.int32), axis=0, return_inverse=True)
    counts = np.bincount(inv)
    reduced = np.column_stack([np.bincount(inv, weights=vertices[:, k])/counts for k in range(3)])
    f = inv[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    return reduced, f


def read_swc(path):
    tab = np.loadtxt(io.StringIO(path.read_text()), comments='#', ndmin=2)
    ids = {int(v): i for i, v in enumerate(tab[:, 0])}
    children = np.asarray([i for i in np.flatnonzero(tab[:, 6] >= 0) if int(tab[i, 6]) in ids], dtype=int)
    parents = np.asarray([ids[int(tab[i, 6])] for i in children], dtype=int)
    xyz = tab[:, 2:5]*.008
    if not np.isfinite(xyz).all():
        raise ValueError('Non-finite SWC geometry')
    return xyz, np.stack([xyz[children], xyz[parents]], axis=1)


def view_matrix(tilt=18., yaw=0.):
    a, b = np.radians([tilt, yaw])
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    return rx @ rz
