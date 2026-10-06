"""One-command renderer for the public, fixed-replay visualization."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('OMP_NUM_THREADS', '1')

import imageio_ffmpeg
import numpy as np
from PIL import Image
from scipy import sparse
import anatomy
import drawing as d
from recording import load_recording
from activity_pca import LABELS, Replay

SUMMARY_SECONDS = 12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plates(cache, manifest, recording):
    render_cache = cache.parent/'render'
    render_cache.mkdir(parents=True, exist_ok=True)
    inputs = {name: sha(ROOT/name) for name in ('drawing.py', 'anatomy.py', 'data/anatomy_manifest.json')}
    inputs['neuron_ids'] = hashlib.sha256(recording['ids'].tobytes()).hexdigest()
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    metadata_path = render_cache/'meta.json'
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        if meta.get('fingerprint') == key:
            try:
                for name, digest in meta['cache_sha256'].items():
                    if sha(render_cache/name) != digest:
                        raise ValueError('Cached rendering checksum mismatch')
                plate = Image.open(render_cache/'detail.png').convert('RGB')
                locator = Image.open(render_cache/'locator.png').convert('RGB')
                with np.load(render_cache/'masks.npz', allow_pickle=False) as a:
                    masks = (a['active'], a['alpha'], sparse.load_npz(render_cache/'weights.npz'))
                print('Using verified anatomical rendering cache', flush=True)
                return plate, locator, masks, meta
            except (OSError, ValueError, KeyError):
                print('Rebuilding incomplete rendering cache', flush=True)
    print('Projecting measured anatomy; this can take a few minutes on CPU', flush=True)
    tri, context, neurons, full_limits, detail_limits, hashes = d.load_geometry(recording['ids'], cache, manifest)
    plate, _ = d.static_plate(tri, context, neurons, d.PLATE, detail_limits)
    locator, transform = d.static_plate(tri, context, neurons, (320, 180), full_limits, 3.)
    masks = d.activity_masks(neurons, d.PLATE, detail_limits)
    plate.save(render_cache/'detail.png')
    locator.save(render_cache/'locator.png')
    np.savez_compressed(render_cache/'masks.npz', active=masks[0], alpha=masks[1])
    sparse.save_npz(render_cache/'weights.npz', masks[2])
    meta = dict(fingerprint=key, limits=[full_limits, detail_limits], locator_transform=transform,
                geometry_sha256=hashes,
                cache_sha256={name: sha(render_cache/name) for name in ('detail.png', 'locator.png', 'masks.npz', 'weights.npz')})
    metadata_path.write_text(json.dumps(meta, indent=2)+'\n')
    return plate, locator, masks, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'.cache/anatomy', help='Verified anatomy download directory')
    parser.add_argument('--output', type=Path, default=ROOT/'output')
    parser.add_argument('--offline', action='store_true', help='Require the cached anatomy; never download')
    parser.add_argument('--download-only', action='store_true', help='Fetch and verify anatomy without rendering')
    parser.add_argument('--preview-only', action='store_true', help='Render stills, PCA and the grid without encoding a video')
    args = parser.parse_args()
    recording = load_recording()
    manifest = anatomy.load_manifest()
    required = set(recording['ids'].tolist())
    supplied = {r['bodyId'] for r in manifest['skeletons']}
    if not required <= supplied:
        raise ValueError('Anatomy manifest does not contain every recorded neuron')
    anatomy.ensure_anatomy(args.cache, manifest, args.offline)
    if args.download_only:
        return
    plate, locator, masks, meta = plates(args.cache, manifest, recording)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    replay = Replay(recording, plate, masks, locator, meta['limits'], meta['locator_transform'])
    frame = replay.frame
    np.savez_compressed(out/'pca.npz', **replay.pca, neuron_ids=recording['ids'])
    replay.figure().save(out/'path_pca.png')
    (out/'pca.json').write_text(json.dumps({
        'method': 'PCA via SVD of centred float64 model rates; loading signs fixed by largest-magnitude entry',
        'samples': recording['sample_count'], 'neurons': len(recording['ids']),
        'fit_scope': 'All recorded samples; activity only, no heading, position, lap or time labels',
        'explained_variance_ratio': replay.pca['explained_variance_ratio'].tolist(),
        'recording_sha256': recording['source_sha256'],
        'interpretation': 'Population activity coordinates, not a spatial map. Heading and position covary on this route.'}, indent=2)+'\n')

    grid, selection = d.heading_summary(recording, plate, masks)
    grid.save(out/'heading_summary.png')
    summary_frames = round(SUMMARY_SECONDS*d.FPS)
    total_frames = recording['sample_count']+summary_frames
    frame(686).save(out/'cover.jpg', quality=96)
    for k in (100, 393, 686):
        frame(k).save(out/f'sample_{k:03d}.png')
    if args.preview_only:
        print(f'Stills ready: {out}', flush=True)
        return
    video = out/'compass_activity.mp4'
    temporary = out/'compass_activity.partial.mp4'
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    command = [encoder, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', '1920x1080', '-r', str(d.FPS), '-i', '-', '-an', '-map_metadata', '-1',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(temporary)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for k in range(recording['sample_count']):
            proc.stdin.write(frame(k).tobytes())
            if k % 200 == 0:
                print(f'Encoded sample {k}/{recording["sample_count"]}', flush=True)
        grid_bytes = grid.tobytes()
        for _ in range(summary_frames):
            proc.stdin.write(grid_bytes)
        print(f'Appended {SUMMARY_SECONDS}s lap-by-heading comparison', flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError('Video encoding failed')
        temporary.replace(video)
    except BaseException:
        proc.kill()
        proc.wait()
        temporary.unlink(missing_ok=True)
        raise
    receipt = {
        'schema_version': 1,
        'recording_sha256': recording['source_sha256'],
        'anatomy_manifest_sha256': sha(ROOT/'data/anatomy_manifest.json'),
        'code_sha256': {p: sha(ROOT/p) for p in ('render.py', 'drawing.py', 'anatomy.py', 'recording.py', 'activity_pca.py')},
        'output_sha256': sha(video), 'frames': total_frames, 'fps': d.FPS,
        'duration_s': total_frames/d.FPS, 'replay_frames': recording['sample_count'],
        'dimensions': [1920, 1080], 'summary_s': SUMMARY_SECONDS,
        'summary_start_s': recording['sample_count']/d.FPS, 'standalone_heading_summary': selection,
        'video_labels': list(LABELS)+['Lap', 'Heading'],
        'pca_metadata': json.loads((out/'pca.json').read_text()),
        'activity_range': [0, 1], 'per_frame_normalisation': False,
        'python': sys.version.split()[0],
        'dependencies': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'matplotlib', 'Pillow', 'imageio-ffmpeg')},
        'ffmpeg_version': imageio_ffmpeg.get_ffmpeg_version(),
        'scope': 'Visualization of saved simulated activity; no model is trained or run',
        'anatomical_credit': manifest['credit'], 'anatomical_license': manifest['license'],
        'data_source': manifest['source']}
    (out/'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fly compass visualization</title>
<style>body{font:18px system-ui;background:#0b151e;color:#eef4f4;max-width:1280px;margin:32px auto;padding:0 20px}video,img{width:100%}a{color:#63c9ce}p{line-height:1.6}</style>
<h1>Inside a simulated compass</h1>
<video controls playsinline preload="metadata" poster="cover.jpg" src="compass_activity.mp4"></video>
<p>Saved simulated activity on reconstructed fly anatomy. Learning is off; a separate controller drives the route.</p>
<p>The final 12 seconds compare three laps (rows) at six simulator headings (columns).</p>
<p><a href="compass_activity.mp4">Download video</a> · <a href="path_pca.png">Path and PCA</a> · <a href="heading_summary.png">Heading grid</a> · <a href="provenance.json">Provenance</a></p>
<img src="path_pca.png" alt="Physical path and two principal components of recorded neural activity">
<p>PCA uses the activity of 46 EPG neurons, fitted once over the whole recording. Path and PCA panels share the same sample and lap colours. Heading and position covary on this route; a recurring loop does not establish a spatial map or place recognition.</p>
<p>Anatomy: MaleCNS v1.0, FlyEM / HHMI Janelia and collaborators, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. <a href="https://male-cns.janelia.org/download/">Dataset</a>. Selected skeletons and display surfaces are shown with model activity colours.</p>''')
    print(f'Ready: {video}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
