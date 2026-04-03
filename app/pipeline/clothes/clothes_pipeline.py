from dataclasses import dataclass

import cv2
import numpy as np

from ..shared.color_match import apply_color_match
from ..shared.composite import composite
from ..shared.decode import decode_design, decode_design_preview
from ..shared.design_transform import apply_design_transform, estimate_print_area_canvas_size
from .tps_warp import tps_warp_design
from .wrinkle_lighting import apply_fabric_multiply


@dataclass
class ClothesAssets:
    mockup: np.ndarray
    wrinkle_map: np.ndarray
    shadow_map: np.ndarray
    mask: np.ndarray
    config: dict
    slug: str = "adhoc"


def run_clothes_pipeline(
    design_bytes: bytes,
    assets: ClothesAssets,
) -> np.ndarray:
    """
    Pipeline for clothes:
    1. TPS warp (manual mesh if provided, otherwise auto/cached mesh)
    2. Color match
    3. Fabric multiply lighting
    4. Composite
    """
    cfg = assets.config
    l = cfg.get("lighting", {})
    color_cfg = cfg.get("color", {})
    W, H = assets.mockup.shape[1], assets.mockup.shape[0]
    is_preview = bool(cfg.get("is_preview", False))
    preview_design_max_dim = max(256, int(cfg.get("preview_design_max_dim", 960)))

    design = (
        decode_design_preview(design_bytes, max_dim=preview_design_max_dim)
        if is_preview
        else decode_design(design_bytes)
    )
    dt = cfg.get("design_transform", {})
    canvas_w, canvas_h = estimate_print_area_canvas_size(
        cfg.get("print_area"),
        fallback_width=design.shape[1],
        fallback_height=design.shape[0],
    )
    design = apply_design_transform(
        design,
        scale=dt.get("scale", 1.0),
        offset_x=dt.get("offset_x", 0.0),
        offset_y=dt.get("offset_y", 0.0),
        fit_mode=dt.get("fit_mode", "cover"),
        target_width=canvas_w,
        target_height=canvas_h,
    )

    mesh = cfg.get("mesh", {})
    warped = None

    manual_src = mesh.get("control_src") or cfg.get("print_area", {}).get("mesh_control_src")
    manual_dst = mesh.get("control_dst") or cfg.get("print_area", {}).get("mesh_control_dst")
    print_area_cfg = cfg.get("print_area", {})
    if (
        isinstance(manual_src, list)
        and isinstance(manual_dst, list)
        and len(manual_src) >= 3
        and len(manual_src) == len(manual_dst)
        and all(k in print_area_cfg for k in ("top_left", "top_right", "bottom_right", "bottom_left"))
    ):
        # 1) Project design to calibrated quad
        dh, dw = design.shape[:2]
        src_rect = np.float32([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]])
        quad = np.float32(
            [
                print_area_cfg["top_left"],
                print_area_cfg["top_right"],
                print_area_cfg["bottom_right"],
                print_area_cfg["bottom_left"],
            ]
        )
        interpolation = cv2.INTER_LANCZOS4
        if W <= 1000:
            interpolation = cv2.INTER_LINEAR
        elif W <= 2000:
            interpolation = cv2.INTER_CUBIC

        base = cv2.warpPerspective(
            design,
            cv2.getPerspectiveTransform(src_rect, quad),
            (W, H),
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        # 2) Apply TPS in output pixel space (1:1 with editor mesh)
        warped = tps_warp_design(
            base,
            src_pts=np.float32(manual_src),
            dst_pts=np.float32(manual_dst),
            output_size=(W, H),
        )

    if warped is None:
        # Ưu tiên mapping deterministic 1-1 theo quad nếu chưa có mesh tay
        if all(k in print_area_cfg for k in ("top_left", "top_right", "bottom_right", "bottom_left")):
            dh, dw = design.shape[:2]
            src_rect = np.float32([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]])
            quad = np.float32(
                [
                    print_area_cfg["top_left"],
                    print_area_cfg["top_right"],
                    print_area_cfg["bottom_right"],
                    print_area_cfg["bottom_left"],
                ]
            )
            interpolation = cv2.INTER_LANCZOS4
            if W <= 1000:
                interpolation = cv2.INTER_LINEAR
            elif W <= 2000:
                interpolation = cv2.INTER_CUBIC

            warped = cv2.warpPerspective(
                design,
                cv2.getPerspectiveTransform(src_rect, quad),
                (W, H),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
        else:
            if assets.slug == "adhoc":
                from .harris_control_points import extract_wrinkle_control_points

                src_pts, dst_pts = extract_wrinkle_control_points(
                    assets.mockup,
                    assets.mask,
                    n_points=mesh.get("n_points", 80),
                    strength=mesh.get("displacement_strength", 6.0),
                )
            else:
                from app.services.template_registry import get_cached_mesh

                src_pts, dst_pts = get_cached_mesh(assets.slug, assets.mockup, assets.mask, cfg)

            warped = tps_warp_design(
                design,
                src_pts=np.float32(src_pts),
                dst_pts=np.float32(dst_pts),
                output_size=(W, H),
            )

    if color_cfg.get("enable_color_match", True):
        warped = apply_color_match(
            warped,
            assets.mockup,
            assets.mask,
            strength=color_cfg.get("match_strength", 0.25),
        )

    warped = apply_fabric_multiply(
        warped,
        shirt_img=assets.mockup,
        shadow_strength=l.get("shadow_strength", 0.40),
        specular_strength=l.get("specular_strength", 0.08),
    )

    result = composite(
        assets.mockup,
        warped,
        assets.mask,
        feather_px=cfg.get("edge", {}).get("feather_px", 10),
    )

    return result
