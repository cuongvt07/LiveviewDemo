import unittest

import numpy as np

from app.pipeline.shared.liveview_cache import (
    get_cylindrical_map_cache,
    make_cylindrical_map_cache_key,
    set_cylindrical_map_cache,
)
from app.routers.render import _build_preview_design_canvas


class LiveviewCacheTests(unittest.TestCase):
    def test_preview_design_canvas_reuses_cached_instance(self) -> None:
        canvas_1 = _build_preview_design_canvas(240, 180, mesh_density_strength=2.0)
        canvas_2 = _build_preview_design_canvas(240, 180, mesh_density_strength=2.0)

        self.assertIs(canvas_1, canvas_2)
        self.assertEqual(canvas_1.shape, (180, 240, 4))

    def test_cylindrical_map_cache_key_is_stable_with_small_float_drift(self) -> None:
        print_area = {
            "top_left": [10, 20],
            "top_right": [210, 20],
            "bottom_right": [210, 220],
            "bottom_left": [10, 220],
        }
        key_a = make_cylindrical_map_cache_key(
            print_area=print_area,
            output_size=(960, 960),
            design_size=(512, 512),
            theta_max_deg=52.00001,
            pitch=0.00003,
            smile_base=0.149999,
            curve_top=16.00001,
            curve_bottom=6.00002,
            edge_squeeze=0.150001,
            squeeze_power=2.00001,
            center_focus_width=0.00002,
        )
        key_b = make_cylindrical_map_cache_key(
            print_area=print_area,
            output_size=(960, 960),
            design_size=(512, 512),
            theta_max_deg=52.00002,
            pitch=0.00004,
            smile_base=0.150000,
            curve_top=16.00002,
            curve_bottom=6.00001,
            edge_squeeze=0.150002,
            squeeze_power=2.00002,
            center_focus_width=0.00001,
        )

        self.assertEqual(key_a, key_b)

    def test_cylindrical_map_cache_stores_bundle(self) -> None:
        key = make_cylindrical_map_cache_key(
            print_area={
                "top_left": [0, 0],
                "top_right": [100, 0],
                "bottom_right": [100, 100],
                "bottom_left": [0, 100],
            },
            output_size=(256, 256),
            design_size=(128, 128),
            theta_max_deg=52.0,
            pitch=0.0,
            smile_base=0.15,
            curve_top=16.0,
            curve_bottom=6.0,
            edge_squeeze=0.1,
            squeeze_power=2.0,
            center_focus_width=0.0,
        )
        bundle = {
            "map_x": np.zeros((8, 8), dtype=np.float32),
            "map_y": np.ones((8, 8), dtype=np.float32),
            "curved_mask": np.full((8, 8), 255, dtype=np.uint8),
        }
        set_cylindrical_map_cache(key, bundle)
        cached = get_cylindrical_map_cache(key)

        self.assertIsNotNone(cached)
        self.assertIs(cached["map_x"], bundle["map_x"])
        self.assertIs(cached["map_y"], bundle["map_y"])
        self.assertIs(cached["curved_mask"], bundle["curved_mask"])


if __name__ == "__main__":
    unittest.main()
