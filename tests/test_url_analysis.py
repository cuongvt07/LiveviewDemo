import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.url_analysis import (
    analyze_and_ingest_url,
    build_url_lookup_context,
    list_available_mockup_views,
    parse_printerval_liveview_url,
    resolve_local_asset_path,
    resolve_public_mockup_url,
)


class UrlAnalysisTests(unittest.TestCase):
    def test_parse_four_part_slug_with_resolution(self) -> None:
        parsed = parse_printerval_liveview_url(
            "https://cdn.printerval.com/image/960x960/mugs-15oz,white,print-seller-2026-03-30_small_2034r-decorated-with-beautiful-colorful-flowers-56aa34d4f04559230f214e0c56298d3d,ffffff.jpeg"
        )

        self.assertEqual(parsed.resolution, "960x960")
        self.assertEqual(parsed.template, "mugs-15oz")
        self.assertEqual(parsed.color_slug, "white")
        self.assertEqual(parsed.section, "print")
        self.assertEqual(parsed.color_hex, "ffffff")

    def test_parse_three_part_slug_without_resolution(self) -> None:
        parsed = parse_printerval_liveview_url(
            "https://cdn.printerval.com/image/mugs-11oz,print-seller-2025-09-13_funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525,2d2d2d.jpeg"
        )

        self.assertIsNone(parsed.resolution)
        self.assertEqual(parsed.template, "mugs-11oz")
        self.assertIsNone(parsed.color_slug)
        self.assertEqual(parsed.color_hex, "2d2d2d")

    def test_resolve_public_mockup_prefers_matching_color_and_size(self) -> None:
        parsed = parse_printerval_liveview_url(
            "https://cdn.printerval.com/image/mugs-11oz,black,print-demo,2d2d2d.jpeg"
        )

        mockup_url = resolve_public_mockup_url(parsed)
        self.assertEqual(mockup_url, "/static/mockups/mug/mug-black-left-11oz.png")

    def test_resolve_local_asset_path_supports_mockup_static_prefix(self) -> None:
        resolved = resolve_local_asset_path("/static/mockups/mug/mug-white-right-11oz.png")
        self.assertTrue(str(resolved).replace("\\", "/").endswith("public/mockups/mug/mug-white-right-11oz.png"))

    @patch(
        "app.services.url_analysis._detect_print_area_preset",
        return_value=(
            {
                "top_left": [10, 20],
                "top_right": [110, 20],
                "bottom_right": [110, 220],
                "bottom_left": [10, 220],
                "quad": [[10, 20], [110, 20], [110, 220], [10, 220]],
                "base_points_raw": [[10, 20], [110, 20], [110, 220], [10, 220]],
                "mask_points": [[10, 20], [110, 20], [110, 220], [10, 220]],
                "product_type": "cylinder_ceramic",
            },
            0.91,
            "auto_detect",
        ),
    )
    @patch(
        "app.services.url_analysis.download_remote_asset",
        return_value=Path("inputs/artworks/mugs-11oz-print-demo-mocked.png"),
    )
    def test_analyze_url_returns_mockup_policy_and_presets(self, *_mocks) -> None:
        result = analyze_and_ingest_url(
            "https://cdn.printerval.com/image/mugs-11oz,black,print-demo,2d2d2d.jpeg"
        )

        self.assertEqual(result["mockup_policy_key"], "11oz:black")
        self.assertEqual(result["mockup_url"], "/static/mockups/mug/mug-black-left-11oz.png")
        self.assertEqual(result["mockup_view"], "left")
        self.assertEqual(result["warp_config_preset"]["product_type"], "cylinder_ceramic")
        self.assertEqual(result["print_area_preset_source"], "auto_detect")
        self.assertAlmostEqual(result["print_area_preset_confidence"], 0.91)
        self.assertEqual(result["design_url"], "/static/artworks/mugs-11oz-print-demo-mocked.png")
        self.assertIn("top_left", result["print_area_preset"])
        self.assertIn("bottom_right", result["print_area_preset"])

    def test_lookup_context_and_view_order_are_exposed(self) -> None:
        context = build_url_lookup_context(
            "https://cdn.printerval.com/image/mugs-11oz,white,print-demo,ffffff.jpeg"
        )
        parsed = context["parsed"]
        available_views = [item["view"] for item in list_available_mockup_views(parsed)]

        self.assertEqual(context["mockup_family_key"], "11oz:white")
        self.assertEqual(context["design_lookup_key"], "mugs-11oz|white|print-demo")
        self.assertEqual(available_views, ["left", "right", "center"])

    @patch(
        "app.services.url_analysis._detect_print_area_preset",
        return_value=(
            {
                "top_left": [10, 20],
                "top_right": [110, 20],
                "bottom_right": [110, 220],
                "bottom_left": [10, 220],
                "quad": [[10, 20], [110, 20], [110, 220], [10, 220]],
                "base_points_raw": [[10, 20], [110, 20], [110, 220], [10, 220]],
                "mask_points": [[10, 20], [110, 20], [110, 220], [10, 220]],
                "product_type": "cylinder_ceramic",
            },
            0.91,
            "auto_detect",
        ),
    )
    def test_analyze_url_uses_builtin_print_rule_when_env_missing(self, *_mocks) -> None:
        with patch(
            "app.services.url_analysis.download_remote_asset",
            return_value=Path("inputs/artworks/builtin-source.png"),
        ) as mocked_download:
            result = analyze_and_ingest_url(
                "https://cdn.printerval.com/image/mugs-11oz,black,print-seller-2025-09-13_funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525,2d2d2d.jpeg"
            )

        first_download_url = mocked_download.call_args_list[0].args[0]
        self.assertEqual(result["design_source_mode"], "builtin_section_rule")
        self.assertEqual(
            first_download_url,
            "https://asset.prtvstatic.com/seller/2025/09/13/funny-baby-huey-cartoon-classic-t-shirt-1-054bcb6a0659c9e13a5f0fd3bfc7c525.png",
        )

    @patch(
        "app.services.url_analysis._detect_print_area_preset",
        return_value=(
            {
                "top_left": [1, 1],
                "top_right": [2, 1],
                "bottom_right": [2, 2],
                "bottom_left": [1, 2],
                "quad": [[1, 1], [2, 1], [2, 2], [1, 2]],
                "base_points_raw": [[1, 1], [2, 1], [2, 2], [1, 2]],
                "mask_points": [[1, 1], [2, 1], [2, 2], [1, 2]],
                "product_type": "cylinder_ceramic",
            },
            0.5,
            "auto_detect",
        ),
    )
    def test_analyze_url_prefers_configured_section_template(self, *_mocks) -> None:
        with patch.dict(
            "os.environ",
            {
                "PRINTERVAL_SECTION_URL_TEMPLATES": (
                    '{"print":"https://assets.example.com/{section}/{folder_path}/{name}.png"}'
                )
            },
            clear=False,
        ):
            with patch(
                "app.services.url_analysis.download_remote_asset",
                return_value=Path("inputs/artworks/configured-source.png"),
            ) as mocked_download:
                result = analyze_and_ingest_url(
                    "https://cdn.printerval.com/image/960x960/mugs-15oz,white,print-seller-2026-03-30_small_2034r-decorated-with-beautiful-colorful-flowers-56aa34d4f04559230f214e0c56298d3d,ffffff.jpeg"
                )

        first_download_url = mocked_download.call_args_list[0].args[0]
        self.assertEqual(result["design_source_mode"], "configured_template")
        self.assertEqual(
            first_download_url,
            "https://assets.example.com/print/seller/2026/03/30/small_2034r-decorated-with-beautiful-colorful-flowers-56aa34d4f04559230f214e0c56298d3d.png",
        )


if __name__ == "__main__":
    unittest.main()
