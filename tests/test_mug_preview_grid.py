import unittest

import numpy as np

from app.routers.render import _compute_adaptive_axis_edges


class MugPreviewGridTests(unittest.TestCase):
    def test_mesh_density_strength_one_keeps_uniform_edges(self) -> None:
        edges = _compute_adaptive_axis_edges(4, density_strength=1.0)
        np.testing.assert_allclose(edges, np.linspace(0.0, 1.0, 5, dtype=np.float32), atol=1e-6)

    def test_mesh_density_strength_pushes_columns_toward_edges(self) -> None:
        edges = _compute_adaptive_axis_edges(4, density_strength=2.0)

        self.assertAlmostEqual(float(edges[0]), 0.0, places=6)
        self.assertAlmostEqual(float(edges[-1]), 1.0, places=6)
        self.assertLess(float(edges[1]), 0.25)
        self.assertGreater(float(edges[-2]), 0.75)
        self.assertTrue(np.all(np.diff(edges) >= 0.0))


if __name__ == "__main__":
    unittest.main()
