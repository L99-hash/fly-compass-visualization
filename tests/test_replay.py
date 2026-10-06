import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import numpy as np
from PIL import Image
from scipy import sparse
import anatomy
import drawing
import recording
from activity_pca import compute_pca, project


class ReplayChecks(unittest.TestCase):
    def setUp(self):
        self.meta = json.loads((ROOT/'data/recording.json').read_text())
        with np.load(ROOT/'data/replay.npz', allow_pickle=False) as a:
            self.arrays = {k: a[k].copy() for k in a.files}

    def test_recording_identities_exist_exactly_once_in_anatomy(self):
        data = recording.load_recording()
        names = [r['bodyId'] for r in anatomy.load_manifest()['skeletons']]
        for body_id in data['ids']:
            self.assertEqual(names.count(body_id), 1)

    def test_duplicate_neuron_id_rejected(self):
        self.arrays['neuron_ids'][1] = self.arrays['neuron_ids'][0]
        with self.assertRaisesRegex(ValueError, 'unique'):
            recording.validate(self.arrays, self.meta)

    def test_missing_sample_time_rejected(self):
        self.arrays['time_s'][300:] += 0.05
        with self.assertRaisesRegex(ValueError, 'timestamps'):
            recording.validate(self.arrays, self.meta)

    def test_activity_scale_is_not_silently_normalised(self):
        self.arrays['activity'][0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, 'scale'):
            recording.validate(self.arrays, self.meta)

    def test_swc_coordinates_use_eight_nanometre_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'neuron.swc'
            path.write_text('1 0 1000 0 0 1 -1\n2 0 1000 500 0 1 1\n')
            xyz, segments = anatomy.read_swc(path)
            np.testing.assert_array_equal(xyz, [[8., 0., 0.], [8., 4., 0.]])
            np.testing.assert_array_equal(segments[0], [xyz[1], xyz[0]])

    def test_offline_cache_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp); (cache/'one.swc').write_bytes(b'bad!')
            entry = ('one.swc', 'https://storage.googleapis.com/flyem-male-cns/example',
                     hashlib.sha256(b'good').hexdigest(), 4)
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                anatomy.fetch_one(cache, entry, offline=True)
            (cache/'one.swc').write_bytes(b'good')
            self.assertFalse(anatomy.fetch_one(cache, entry, offline=True))

    def test_grid_selection_uses_true_heading_and_is_activity_independent(self):
        data = recording.load_recording()
        plate = Image.new('RGB', (2, 2))
        masks = (np.array([0, 1, 2, 3]), None, None)
        with patch.object(drawing, 'paint', return_value=plate):
            _, first = drawing.heading_summary(data, plate, masks)
            data['rates'] = np.zeros_like(data['rates'])
            _, second = drawing.heading_summary(data, plate, masks)
        self.assertEqual(first, second)
        self.assertEqual(len(first['cells']), 18)
        self.assertLess(max(c['heading_mismatch_deg'] for c in first['cells']), 1.1)
        for lap in range(1, 4):
            cells = [c for c in first['cells'] if c['lap'] == lap]
            self.assertEqual([c['column_heading_deg'] for c in cells], list(range(0, 360, 60)))
            self.assertLess(cells[0]['sample_index'], (lap-1)*293+20)
            self.assertEqual([c['sample_index'] for c in cells], sorted(c['sample_index'] for c in cells))

    def test_rendered_colour_uses_fixed_scale(self):
        base = Image.new('RGB', (2, 1), 'black')
        masks = (np.array([0, 1]), np.ones((2, 1)), sparse.eye(2, format='csr'))
        first = np.asarray(drawing.paint(base, masks, np.array([.3, .6])))
        second = np.asarray(drawing.paint(base, masks, np.array([.3, 1.])))
        np.testing.assert_array_equal(first[0, 0], second[0, 0])
        self.assertFalse(np.array_equal(first[0, 1], second[0, 1]))

    def test_pca_reconstructs_rank_two_activity_with_fixed_loading_signs(self):
        t = np.linspace(0, 2*np.pi, 120, endpoint=False)
        latent = np.column_stack([np.cos(t), .4*np.sin(t)])
        mixing = np.array([[.1,.2,.15,.05], [.07,-.1,.02,.1]])
        activity = .5 + latent@mixing
        result = compute_pca(activity)
        np.testing.assert_allclose(result['scores']@result['loadings'].T+result['mean'],activity,atol=1e-12)
        self.assertAlmostEqual(result['explained_variance_ratio'].sum(),1.)
        for column in result['loadings'].T:
            self.assertGreaterEqual(column[np.argmax(np.abs(column))],0)
        constant = compute_pca(np.ones((10,4)))
        np.testing.assert_array_equal(constant['explained_variance_ratio'],[0.,0.])
        np.testing.assert_array_equal(constant['scores'],np.zeros((10,2)))

    def test_pca_activity_geometry_is_preserved_under_sample_permutation(self):
        activity = self.arrays['activity']
        reference = compute_pca(activity)
        order = np.random.default_rng(42).permutation(len(activity))
        permuted = compute_pca(activity[order])
        np.testing.assert_allclose(permuted['scores'][np.argsort(order)],reference['scores'],atol=1e-10)
        np.testing.assert_allclose(permuted['loadings'],reference['loadings'],atol=1e-10)

    def test_panel_projection_preserves_equal_coordinate_scales(self):
        xy = np.array([[0.,0.],[1.,0.],[0.,1.]])
        points = project(xy,(0,0,900,200))
        lengths = np.linalg.norm(points[1:]-points[0],axis=1)
        self.assertAlmostEqual(lengths[0],lengths[1])


if __name__ == '__main__':
    unittest.main()
