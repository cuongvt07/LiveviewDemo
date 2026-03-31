import unittest

import numpy as np

from app.pipeline.shared.structured_mesh_warp import apply_structured_mesh_warp


class StructuredMeshWarpTests(unittest.TestCase):
    def test_identity_mesh_is_noop(self) -> None:
        image = np.zeros((32, 32, 4), dtype=np.uint8)
        image[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :]
        image[:, :, 1] = np.arange(32, dtype=np.uint8)[:, None]
        image[:, :, 3] = 255

        xs = np.linspace(4, 28, 4, dtype=np.float32)
        ys = np.linspace(4, 28, 4, dtype=np.float32)
        pts = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)

        warped = apply_structured_mesh_warp(image, pts, pts, output_size=(32, 32))
        np.testing.assert_allclose(warped, image, atol=1)

    def test_moving_internal_node_changes_result(self) -> None:
        image = np.zeros((48, 48, 4), dtype=np.uint8)
        image[:, :, 0] = np.arange(48, dtype=np.uint8)[None, :]
        image[:, :, 1] = np.arange(48, dtype=np.uint8)[:, None]
        image[:, :, 3] = 255

        xs = np.linspace(6, 42, 4, dtype=np.float32)
        ys = np.linspace(6, 42, 4, dtype=np.float32)
        src = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
        dst = src.copy()
        dst[5, 0] += 6.0
        dst[5, 1] += 4.0

        warped = apply_structured_mesh_warp(image, src, dst, output_size=(48, 48))
        self.assertGreater(float(np.mean(np.abs(warped.astype(np.float32) - image.astype(np.float32)))), 0.1)

    def test_shifted_mesh_clears_stale_source_pixels(self) -> None:
        image = np.zeros((48, 48, 4), dtype=np.uint8)
        image[:, :, 0] = 180
        image[:, :, 3] = 255

        xs = np.linspace(6, 42, 4, dtype=np.float32)
        ys = np.linspace(6, 42, 4, dtype=np.float32)
        src = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
        dst = src.copy()
        dst[:, 0] += 4.0

        warped = apply_structured_mesh_warp(image, src, dst, output_size=(48, 48))

        self.assertEqual(int(warped[24, 7, 3]), 0)
        self.assertGreater(int(warped[24, 12, 3]), 0)


if __name__ == "__main__":
    unittest.main()
