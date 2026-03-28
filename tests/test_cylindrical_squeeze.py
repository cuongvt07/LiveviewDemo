import unittest

import numpy as np

from app.pipeline.mugs.cylinder_math import apply_horizontal_squeeze, compute_uv_cylindrical


class HorizontalSqueezeTests(unittest.TestCase):
    def test_zero_edge_and_width_is_noop(self) -> None:
        theta_max = np.radians(52.0)
        theta = np.linspace(-theta_max, theta_max, 9, dtype=np.float32)

        squeezed = apply_horizontal_squeeze(
            theta,
            theta_max,
            edge_squeeze=0.0,
            squeeze_power=2.0,
            center_focus_width=0.0,
        )

        np.testing.assert_allclose(squeezed, np.clip(theta / theta_max, -1.0, 1.0), atol=1e-6)

    def test_center_and_edges_remain_anchored(self) -> None:
        theta_max = np.radians(52.0)
        theta = np.array([-theta_max, 0.0, theta_max], dtype=np.float32)

        squeezed = apply_horizontal_squeeze(
            theta,
            theta_max,
            edge_squeeze=1.0,
            squeeze_power=2.0,
            center_focus_width=1.0,
        )

        np.testing.assert_allclose(squeezed, np.array([-1.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)

    def test_edge_squeeze_rolls_outer_band_without_touching_middle(self) -> None:
        theta_max = np.radians(52.0)
        theta = np.array([0.2 * theta_max, 0.3 * theta_max, 0.8 * theta_max], dtype=np.float32)

        squeezed = apply_horizontal_squeeze(
            theta,
            theta_max,
            edge_squeeze=1.0,
            squeeze_power=2.0,
            center_focus_width=0.0,
        )

        self.assertAlmostEqual(float(squeezed[0]), 0.2, places=6)
        self.assertAlmostEqual(float(squeezed[1]), 0.3, places=6)
        self.assertGreater(float(squeezed[2]), 0.8)

    def test_center_width_only_expands_middle_band(self) -> None:
        theta_max = np.radians(52.0)
        widened = apply_horizontal_squeeze(
            np.array([0.2 * theta_max, 0.8 * theta_max], dtype=np.float32),
            theta_max,
            edge_squeeze=0.0,
            squeeze_power=2.0,
            center_focus_width=1.0,
        )

        self.assertAlmostEqual(float(widened[0]), 0.46666667, places=6)
        self.assertAlmostEqual(float(widened[1]), 0.91428571, places=6)
        self.assertGreater(float(widened[0]) - 0.2, float(widened[1]) - 0.8)

    def test_width_and_edge_are_visibly_distinct_profiles(self) -> None:
        theta_max = np.radians(52.0)
        theta = np.array([0.3 * theta_max, 0.6 * theta_max], dtype=np.float32)

        width_only = apply_horizontal_squeeze(
            theta,
            theta_max,
            edge_squeeze=0.0,
            squeeze_power=2.0,
            center_focus_width=1.0,
        )
        edge_only = apply_horizontal_squeeze(
            theta,
            theta_max,
            edge_squeeze=1.0,
            squeeze_power=2.0,
            center_focus_width=0.0,
        )

        self.assertGreater(float(width_only[0]), float(edge_only[0]))
        self.assertAlmostEqual(float(edge_only[0]), 0.3, places=6)
        self.assertGreater(float(edge_only[1]), 0.6)

    def test_zero_edge_and_width_is_noop_in_uv(self) -> None:
        x_proj = np.linspace(-1.0, 1.0, 17, dtype=np.float32)[None, :]
        y_proj = np.zeros_like(x_proj)

        u_base, v_base = compute_uv_cylindrical(
            x_proj,
            y_proj,
            theta_max_deg=52.0,
            curve_top=0.0,
            curve_bottom=0.0,
        )
        u_same, v_same = compute_uv_cylindrical(
            x_proj,
            y_proj,
            theta_max_deg=52.0,
            curve_top=0.0,
            curve_bottom=0.0,
            edge_squeeze=0.0,
            squeeze_power=2.0,
            center_focus_width=0.0,
        )

        np.testing.assert_allclose(u_same, u_base, atol=1e-6)
        np.testing.assert_allclose(v_same, v_base, atol=1e-6)

    def test_width_only_changes_horizontal_mapping(self) -> None:
        x = np.linspace(-1.0, 1.0, 33, dtype=np.float32)[None, :]
        y = np.linspace(-1.0, 1.0, 9, dtype=np.float32)[:, None]
        x_proj = np.broadcast_to(x, (9, 33))
        y_proj = np.broadcast_to(y, (9, 33))

        u_base, v_base = compute_uv_cylindrical(
            x_proj,
            y_proj,
            theta_max_deg=52.0,
            curve_top=12.0,
            curve_bottom=-8.0,
        )
        u_width, v_width = compute_uv_cylindrical(
            x_proj,
            y_proj,
            theta_max_deg=52.0,
            curve_top=12.0,
            curve_bottom=-8.0,
            edge_squeeze=0.0,
            squeeze_power=2.0,
            center_focus_width=1.0,
        )

        self.assertGreater(np.max(np.abs(u_width - u_base)), 1e-5)
        np.testing.assert_allclose(v_width, v_base, atol=1e-6)

    def test_edge_changes_only_horizontal_mapping(self) -> None:
        x = np.linspace(-1.0, 1.0, 33, dtype=np.float32)[None, :]
        y = np.linspace(-1.0, 1.0, 9, dtype=np.float32)[:, None]
        x_proj = np.broadcast_to(x, (9, 33))
        y_proj = np.broadcast_to(y, (9, 33))

        u_base, v_base = compute_uv_cylindrical(
            x_proj,
            y_proj,
            theta_max_deg=52.0,
            curve_top=12.0,
            curve_bottom=-8.0,
        )
        u_squeezed, v_squeezed = compute_uv_cylindrical(
            x_proj,
            y_proj,
            theta_max_deg=52.0,
            curve_top=12.0,
            curve_bottom=-8.0,
            edge_squeeze=0.35,
            squeeze_power=2.0,
            center_focus_width=0.0,
        )

        self.assertGreater(np.max(np.abs(u_squeezed - u_base)), 1e-5)
        np.testing.assert_allclose(v_squeezed, v_base, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
