import json
import unittest

from app.routers.render import _build_default_adhoc_config, _merge_adhoc_user_config


class RenderAdhocConfigTests(unittest.TestCase):
    def test_cylinder_mesh_keeps_mug_pipeline(self) -> None:
        config = _build_default_adhoc_config(1200, 900)
        merged = _merge_adhoc_user_config(
            config,
            json.dumps(
                {
                    "print_area": {
                        "product_type": "cylinder_ceramic",
                        "mesh_control_src": [[100, 100], [300, 100], [300, 300], [100, 300]],
                        "mesh_control_dst": [[100, 100], [310, 105], [305, 300], [95, 300]],
                    },
                    "warp": {
                        "warp_type": "cylinder",
                    },
                }
            ),
        )

        self.assertEqual(merged["product_type"], "mug")
        self.assertEqual(merged["mesh"]["control_src"], merged["print_area"]["mesh_control_src"])
        self.assertEqual(merged["mesh"]["control_dst"], merged["print_area"]["mesh_control_dst"])

    def test_tps_mesh_routes_to_clothes_pipeline(self) -> None:
        config = _build_default_adhoc_config(1200, 900)
        merged = _merge_adhoc_user_config(
            config,
            json.dumps(
                {
                    "print_area": {
                        "mesh_control_src": [[100, 100], [300, 100], [300, 300], [100, 300]],
                        "mesh_control_dst": [[100, 100], [320, 110], [310, 300], [90, 300]],
                    },
                    "warp": {
                        "warp_type": "tps",
                    },
                }
            ),
        )

        self.assertEqual(merged["product_type"], "clothes")

    def test_light_direction_controls_merge_into_lighting_config(self) -> None:
        config = _build_default_adhoc_config(1200, 900)
        merged = _merge_adhoc_user_config(
            config,
            json.dumps(
                {
                    "lighting": {
                        "light_pos_x": 0.78,
                        "light_pos_y": 0.24,
                        "light_height": 72,
                        "light_contrast": 64,
                        "light_highlight": 81,
                        "light_softness": 36,
                    }
                }
            ),
        )

        self.assertEqual(merged["lighting"]["light_pos_x"], 0.78)
        self.assertEqual(merged["lighting"]["light_pos_y"], 0.24)
        self.assertEqual(merged["lighting"]["light_height"], 72.0)
        self.assertEqual(merged["lighting"]["light_contrast"], 64.0)
        self.assertEqual(merged["lighting"]["light_highlight"], 81.0)
        self.assertEqual(merged["lighting"]["light_softness"], 36.0)

    def test_design_fit_mode_merges_into_design_transform(self) -> None:
        config = _build_default_adhoc_config(1200, 900)
        merged = _merge_adhoc_user_config(
            config,
            json.dumps(
                {
                    "warp": {
                        "design_scale": 1.15,
                        "design_offset_x": -0.2,
                        "design_offset_y": 0.3,
                        "design_fit_mode": "contain",
                    }
                }
            ),
        )

        self.assertEqual(merged["design_transform"]["fit_mode"], "contain")
        self.assertEqual(merged["design_transform"]["scale"], 1.15)
        self.assertEqual(merged["design_transform"]["offset_x"], -0.2)
        self.assertEqual(merged["design_transform"]["offset_y"], 0.3)

    def test_invalid_design_fit_mode_falls_back_to_cover(self) -> None:
        config = _build_default_adhoc_config(1200, 900)
        merged = _merge_adhoc_user_config(
            config,
            json.dumps(
                {
                    "warp": {
                        "design_fit_mode": "stretch",
                    }
                }
            ),
        )

        self.assertEqual(merged["design_transform"]["fit_mode"], "cover")


if __name__ == "__main__":
    unittest.main()
