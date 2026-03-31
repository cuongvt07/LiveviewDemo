import unittest

import numpy as np

from app.pipeline.shared.design_transform import apply_design_transform


class DesignTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = np.zeros((100, 50, 4), dtype=np.uint8)
        self.design[:, :, :3] = [255, 0, 0]
        self.design[:, :, 3] = 255

    def test_cover_fills_target_canvas(self) -> None:
        transformed = apply_design_transform(
            self.design,
            fit_mode="cover",
            target_width=200,
            target_height=200,
        )

        self.assertEqual(transformed.shape, (200, 200, 4))
        self.assertGreater(float((transformed[:, :, 3] > 200).mean()), 0.95)
        self.assertGreaterEqual(int(transformed[:, 0, 3].max()), 200)

    def test_contain_preserves_full_artwork_with_transparent_side_padding(self) -> None:
        transformed = apply_design_transform(
            self.design,
            fit_mode="contain",
            target_width=200,
            target_height=200,
        )

        self.assertEqual(transformed.shape, (200, 200, 4))
        self.assertEqual(int(transformed[:, 0, 3].max()), 0)
        self.assertEqual(int(transformed[:, -1, 3].max()), 0)
        self.assertGreaterEqual(int(transformed[:, 100, 3].max()), 200)
        self.assertLess(float((transformed[:, :, 3] > 200).mean()), 0.75)


if __name__ == "__main__":
    unittest.main()
