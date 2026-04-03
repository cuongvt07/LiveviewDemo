import unittest

import numpy as np

from app.pipeline.mugs.lighting_extract import (
    build_cylinder_surface_maps,
    build_light_direction,
    build_light_field,
    build_cylinder_geometry_maps,
    compute_directional_diffuse,
    derive_highlight_map,
    extract_masked_highlight_detail,
    extract_masked_lighting,
)
from app.pipeline.mugs.shadow_overlay import apply_shadow_overlay


class MugLightingPipelineTests(unittest.TestCase):
    def test_extract_masked_lighting_normalizes_inside_mask_only(self) -> None:
        mockup = np.zeros((4, 4, 3), dtype=np.uint8)
        mockup[:, :2] = [40, 40, 40]
        mockup[:, 2:] = [220, 220, 220]

        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:, 1:4] = 255

        lighting = extract_masked_lighting(mockup, mask, blur_kernel=1)

        self.assertTrue(np.allclose(lighting[:, 0], 0.0))
        self.assertLess(float(lighting[0, 1]), float(lighting[0, 3]))
        self.assertGreaterEqual(float(lighting[0, 3]), 0.99)

    def test_derive_highlight_map_respects_threshold(self) -> None:
        lighting = np.array(
            [[0.2, 0.7, 0.85, 1.0]],
            dtype=np.float32,
        )

        highlight = derive_highlight_map(lighting, threshold=0.7, blur_kernel=1)

        self.assertEqual(float(highlight[0, 0]), 0.0)
        self.assertEqual(float(highlight[0, 1]), 0.0)
        self.assertGreater(float(highlight[0, 2]), 0.0)
        self.assertGreater(float(highlight[0, 3]), float(highlight[0, 2]))

    def test_shadow_overlay_multiplies_against_lighting(self) -> None:
        warped = np.zeros((1, 2, 4), dtype=np.uint8)
        warped[:, :, :3] = 200
        warped[:, :, 3] = 255

        shadow_map = np.array([[0.0, 1.0]], dtype=np.float32)
        result = apply_shadow_overlay(warped, shadow_map, strength=1.0)

        self.assertLess(int(result[0, 0, 0]), int(result[0, 1, 0]))
        self.assertEqual(int(result[0, 1, 0]), 200)

    def test_extract_masked_highlight_detail_finds_bright_stripe(self) -> None:
        mockup = np.full((9, 9, 3), 120, dtype=np.uint8)
        mockup[:, 4:6] = 230
        mask = np.ones((9, 9), dtype=np.uint8) * 255

        detail = extract_masked_highlight_detail(
            mockup,
            mask,
            diffuse_blur_kernel=5,
            detail_blur_kernel=1,
        )

        self.assertGreater(float(detail[:, 4:6].mean()), float(detail[:, :2].mean()))

    def test_build_light_direction_maps_ui_height_to_nonzero_z(self) -> None:
        light_dir = build_light_direction(light_pos_x=1.0, light_pos_y=0.5, light_height=0.0)

        self.assertAlmostEqual(float(np.linalg.norm(light_dir)), 1.0, places=5)
        self.assertGreater(float(light_dir[2]), 0.0)
        self.assertGreater(float(light_dir[0]), 0.0)

    def test_directional_diffuse_biases_toward_light_side(self) -> None:
        mask = np.ones((9, 9), dtype=np.uint8) * 255
        print_area = {
            "top_left": [1, 1],
            "top_right": [7, 1],
            "bottom_right": [7, 7],
            "bottom_left": [1, 7],
        }

        surface_maps = build_cylinder_surface_maps(
            print_area=print_area,
            output_size=(9, 9),
            mask=mask,
            theta_max_deg=52.0,
        )
        diffuse = compute_directional_diffuse(
            surface_maps["normals"],
            build_light_direction(light_pos_x=0.82, light_pos_y=0.32, light_height=55.0),
            valid_mask=surface_maps["valid_mask"],
        )

        self.assertGreater(float(diffuse[4, 6]), float(diffuse[4, 2]))

    def test_light_field_peaks_near_requested_region(self) -> None:
        mask = np.ones((9, 9), dtype=np.uint8) * 255
        print_area = {
            "top_left": [1, 1],
            "top_right": [7, 1],
            "bottom_right": [7, 7],
            "bottom_left": [1, 7],
        }

        surface_maps = build_cylinder_surface_maps(
            print_area=print_area,
            output_size=(9, 9),
            mask=mask,
            theta_max_deg=52.0,
        )
        field = build_light_field(
            surface_maps["surface_u"],
            surface_maps["surface_v"],
            surface_maps["valid_mask"],
            light_pos_x=0.82,
            light_pos_y=0.25,
            softness=30.0,
            contrast=70.0,
        )

        self.assertGreater(float(field[2, 6]), float(field[2, 2]))

    def test_cylinder_geometry_maps_brighten_center_and_add_specular_line(self) -> None:
        mask = np.ones((9, 9), dtype=np.uint8) * 255
        print_area = {
            "top_left": [1, 1],
            "top_right": [7, 1],
            "bottom_right": [7, 7],
            "bottom_left": [1, 7],
        }

        cylinder_map, specular_line = build_cylinder_geometry_maps(
            print_area=print_area,
            output_size=(9, 9),
            mask=mask,
            theta_max_deg=52.0,
            cylinder_strength=0.4,
            edge_darkening_strength=0.2,
            specular_position=0.2,
            specular_sigma=0.18,
            specular_blur_kernel=1,
        )

        self.assertGreater(float(cylinder_map[4, 4]), float(cylinder_map[4, 1]))
        self.assertGreater(float(specular_line[4, 5]), float(specular_line[4, 1]))


if __name__ == "__main__":
    unittest.main()
