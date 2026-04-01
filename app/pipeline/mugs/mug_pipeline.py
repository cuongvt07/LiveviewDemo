from dataclasses import dataclass

import numpy as np

from ..shared.color_match import apply_color_match
from ..shared.composite import composite
from ..shared.decode import decode_design
from ..shared.design_transform import apply_design_transform, estimate_print_area_canvas_size
from ..shared.smooth_mesh_warp import apply_smooth_mesh_warp
from .cylindrical_warp import cylindrical_warp
from .renderer import render_mug_final, render_mug_preview
from .mesh_config import MugMeshConfig
from .lighting_extract import (
    build_cylinder_surface_maps,
    build_light_direction,
    build_light_field,
    compute_directional_diffuse,
    compute_directional_specular,
    derive_highlight_map,
    extract_masked_highlight_detail,
    extract_masked_lighting,
    normalize_masked_field,
)
from .shadow_overlay import apply_shadow_overlay
from .specular_gloss import apply_specular_gloss, apply_mug_lighting


@dataclass
class MugAssets:
    mockup: np.ndarray
    shadow_map: np.ndarray
    normal_map: np.ndarray
    mask: np.ndarray
    specular_map: np.ndarray
    config: dict


def run_mug_pipeline(
    design_bytes: bytes,
    assets: MugAssets,
) -> np.ndarray:
    """
    Mug render flow:
    [1] Cylindrical warp + optional mesh correction
    [2] Optional color match
    [3] Extract masked lighting from mockup LAB L channel
    [4] Apply shadow by multiply
    [5] Composite
    [6] Apply highlight by screen
    """
    cfg = assets.config
    lighting_cfg = cfg.get("lighting", {})
    color_cfg = cfg.get("color", {})
    render_cfg = cfg.get("render", {})
    preserve_original_color = bool(render_cfg.get("preserve_original_color", False))
    out_w, out_h = assets.mockup.shape[1], assets.mockup.shape[0]

    design = decode_design(design_bytes)
    design_transform = cfg.get("design_transform", {})
    canvas_w, canvas_h = estimate_print_area_canvas_size(
        cfg.get("print_area"),
        fallback_width=design.shape[1],
        fallback_height=design.shape[0],
    )
    design = apply_design_transform(
        design,
        scale=design_transform.get("scale", 1.0),
        offset_x=design_transform.get("offset_x", 0.0),
        offset_y=design_transform.get("offset_y", 0.0),
        fit_mode=design_transform.get("fit_mode", "cover"),
        target_width=canvas_w,
        target_height=canvas_h,
    )

    # If template provides a dense mesh config, prefer mesh-based warp renderer.
    mesh_cfg = cfg.get("mesh", {})
    if isinstance(mesh_cfg, dict) and mesh_cfg.get("enabled", False):
        try:
            mcfg = MugMeshConfig(
                cols=int(mesh_cfg.get("cols", 12)),
                rows=int(mesh_cfg.get("rows", 8)),
                spacing=str(mesh_cfg.get("spacing", "cosine")),
                tension=float(mesh_cfg.get("tension", 0.35)),
                corner_blend_radius=float(mesh_cfg.get("corner_blend_radius", 0.08)),
                symmetry_lock=bool(mesh_cfg.get("symmetry_lock", True)),
                max_displacement=float(mesh_cfg.get("max_displacement", 0.15)),
            )
            # If points provided, override
            pts = mesh_cfg.get("points")
            if isinstance(pts, list) and len(pts) == (mcfg.rows + 1) * (mcfg.cols + 1):
                from .mesh_config import MeshPoint
                mcfg.points = [
                    MeshPoint(u=float(p[0]), v=float(p[1]), locked=bool(p[2]) if len(p) > 2 else False)
                    for p in pts
                ]

            # Use full renderer for final output
            warped = render_mug_final(design, mcfg, out_w=out_w, out_h=out_h)
        except Exception:
            # fallback to cylindrical
            cyl = cfg.get("cylinder", {})
            warped = cylindrical_warp(
                design,
                print_area=cfg["print_area"],
                output_size=(out_w, out_h),
                reference_mockup=assets.mockup,
                theta_max_deg=cyl.get("theta_max_deg", 52.0),
                pitch=cyl.get("pitch", 0.0),
                smile_base=cyl.get("smile_base", 0.08),
                curve_top=cyl.get("curve_top"),
                curve_bottom=cyl.get("curve_bottom"),
                edge_squeeze=cyl.get("edge_squeeze", 0.0),
                squeeze_power=cyl.get("squeeze_power", 2.0),
                center_focus_width=cyl.get("center_focus_width", 0.0),
                curve_correction_alpha=cyl.get("curve_correction_alpha", 0.0),
                curve_snap_threshold_px=cyl.get("curve_snap_threshold_px", 2.0),
            )
    else:
        cyl = cfg.get("cylinder", {})
        warped = cylindrical_warp(
            design,
            print_area=cfg["print_area"],
            output_size=(out_w, out_h),
            reference_mockup=assets.mockup,
            theta_max_deg=cyl.get("theta_max_deg", 52.0),
            pitch=cyl.get("pitch", 0.0),
            smile_base=cyl.get("smile_base", 0.08),
            curve_top=cyl.get("curve_top"),
            curve_bottom=cyl.get("curve_bottom"),
            edge_squeeze=cyl.get("edge_squeeze", 0.0),
            squeeze_power=cyl.get("squeeze_power", 2.0),
            center_focus_width=cyl.get("center_focus_width", 0.0),
            curve_correction_alpha=cyl.get("curve_correction_alpha", 0.0),
            curve_snap_threshold_px=cyl.get("curve_snap_threshold_px", 2.0),
        )

    mesh_src = cfg.get("print_area", {}).get("mesh_control_src")
    mesh_dst = cfg.get("print_area", {}).get("mesh_control_dst")
    if (
        isinstance(mesh_src, list)
        and isinstance(mesh_dst, list)
        and len(mesh_src) >= 4
        and len(mesh_src) == len(mesh_dst)
    ):
        warped = apply_smooth_mesh_warp(
            warped,
            src_pts=np.float32(mesh_src),
            dst_pts=np.float32(mesh_dst),
            output_size=(out_w, out_h),
        )

    if (not preserve_original_color) and color_cfg.get("enable_color_match", True):
        warped = apply_color_match(
            warped,
            assets.mockup,
            assets.mask,
            strength=color_cfg.get("match_strength", 0.40),
        )

    lighting_mask = warped[:, :, 3].astype(np.float32) / 255.0
    base_mask = assets.mask.astype(np.float32)
    if base_mask.max() > 1.0:
        base_mask /= 255.0
    lighting_mask = np.clip(lighting_mask * base_mask, 0.0, 1.0)

    extracted_lighting = extract_masked_lighting(
        assets.mockup,
        lighting_mask,
        blur_kernel=int(lighting_cfg.get("lighting_blur_kernel", 21)),
    )
    surface_maps = build_cylinder_surface_maps(
        print_area=cfg["print_area"],
        output_size=(out_w, out_h),
        mask=lighting_mask,
        theta_max_deg=cyl.get("theta_max_deg", 52.0),
        edge_squeeze=cyl.get("edge_squeeze", 0.0),
        squeeze_power=cyl.get("squeeze_power", 2.0),
        center_focus_width=cyl.get("center_focus_width", 0.0),
    )
    light_dir = build_light_direction(
        light_pos_x=float(lighting_cfg.get("light_pos_x", 0.62)),
        light_pos_y=float(lighting_cfg.get("light_pos_y", 0.32)),
        light_height=float(lighting_cfg.get("light_height", 55.0)),
    )
    geometry_diffuse = compute_directional_diffuse(
        surface_maps["normals"],
        light_dir,
        valid_mask=surface_maps["valid_mask"],
    )
    geometry_diffuse = normalize_masked_field(
        geometry_diffuse,
        surface_maps["valid_mask"],
        empty_fill=1.0,
        outside_fill=0.0,
        low_percentile=2.0,
        high_percentile=98.0,
    )
    geometry_diffuse = np.where(
        surface_maps["valid_mask"],
        0.35 + 0.65 * geometry_diffuse,
        0.0,
    ).astype(np.float32)

    light_field = build_light_field(
        surface_maps["surface_u"],
        surface_maps["surface_v"],
        surface_maps["valid_mask"],
        light_pos_x=float(lighting_cfg.get("light_pos_x", 0.62)),
        light_pos_y=float(lighting_cfg.get("light_pos_y", 0.32)),
        softness=float(lighting_cfg.get("light_softness", 55.0)),
        contrast=float(lighting_cfg.get("light_contrast", 50.0)),
        theta_max_deg=cyl.get("theta_max_deg", 52.0),
        edge_squeeze=cyl.get("edge_squeeze", 0.0),
        squeeze_power=cyl.get("squeeze_power", 2.0),
        center_focus_width=cyl.get("center_focus_width", 0.0),
    )
    geometry_lighting = np.clip(geometry_diffuse * light_field, 0.0, 1.0)
    del geometry_diffuse, light_field

    photo_lighting = np.where(
        surface_maps["valid_mask"],
        0.60 + 0.40 * extracted_lighting,
        0.0,
    ).astype(np.float32)
    lighting_map = normalize_masked_field(
        photo_lighting * geometry_lighting,
        surface_maps["valid_mask"],
        empty_fill=1.0,
        outside_fill=0.0,
        low_percentile=2.0,
        high_percentile=98.0,
    )
    lighting_map = np.where(
        surface_maps["valid_mask"],
        0.28 + 0.72 * lighting_map,
        0.0,
    ).astype(np.float32)

    # Precompute specular/highlight maps early so we can apply them to the warped
    # design before compositing. This separates diffuse (multiply) and specular
    # (additive in linear space).
    specular_strength = float(lighting_cfg.get("specular_strength", 0.0))
    highlight_map = None
    if (not preserve_original_color) and specular_strength > 0:
        # Reuse extracted_lighting to build diffuse highlight map
        light_highlight = float(lighting_cfg.get("light_highlight", 60.0))
        highlight_t = np.clip(light_highlight / 100.0, 0.0, 1.0)
        diffuse_highlight = derive_highlight_map(
            extracted_lighting,
            threshold=float(lighting_cfg.get("specular_threshold", 180)),
            blur_kernel=int(lighting_cfg.get("highlight_blur_kernel", 9)),
        )
        detail_highlight = extract_masked_highlight_detail(
            assets.mockup,
            lighting_mask,
            diffuse_blur_kernel=int(lighting_cfg.get("highlight_extract_blur_kernel", 41)),
            detail_blur_kernel=int(lighting_cfg.get("highlight_detail_blur_kernel", 9)),
        )
        geometry_specular = compute_directional_specular(
            surface_maps["normals"],
            light_dir,
            valid_mask=surface_maps["valid_mask"],
            shininess=8.0 + highlight_t * 40.0,
            blur_kernel=int(3 + round(float(lighting_cfg.get("light_softness", 55.0)) / 8.0) * 2),
        )
        highlight_map = np.clip(
            geometry_specular * (0.75 + highlight_t * 0.55)
            + diffuse_highlight * float(lighting_cfg.get("diffuse_highlight_strength", 0.25))
            + detail_highlight * float(lighting_cfg.get("highlight_detail_strength", 0.35)),
            0.0,
            1.0,
        )

    shadow_strength = float(lighting_cfg.get("shadow_strength", 0.45))
    if (not preserve_original_color) and shadow_strength > 0:
        warped = apply_shadow_overlay(
            warped,
            lighting_map,
            strength=shadow_strength,
        )

    # Apply separated mug lighting (diffuse * lighting_map + additive specular)
    if (not preserve_original_color) and (highlight_map is not None):
        warped = apply_mug_lighting(
            warped,
            light_map=lighting_map,
            specular_mask=highlight_map,
            specular_strength=specular_strength,
            shininess=float(lighting_cfg.get("shininess", 20.0)),
        )

    result = composite(
        assets.mockup,
        warped,
        assets.mask,
        feather_px=cfg.get("edge", {}).get("feather_px", 6),
    )

    return result
