from dataclasses import dataclass

import logging
import time
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import cv2

from ..shared.color_match import apply_color_match
from ..shared.composite import composite
from ..shared.decode import decode_design, decode_design_preview
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


@lru_cache(maxsize=64)
def _get_cached_surface_maps(
    w: int, h: int, theta_max_deg: float, edge_squeeze: float,
    squeeze_power: float, center_focus_width: float, print_area_str: str
):
    print_area = json.loads(print_area_str)
    return build_cylinder_surface_maps(
        print_area, (w, h), None, theta_max_deg=theta_max_deg,
        edge_squeeze=edge_squeeze, squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
    )


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
    logger = logging.getLogger('mockup_service')
    t_total = time.perf_counter()
    lighting_cfg = cfg.get("lighting", {})
    # Ensure specular_strength is always defined to avoid NameError
    specular_strength = float(lighting_cfg.get("specular_strength", 0.0))
    color_cfg = cfg.get("color", {})
    render_cfg = cfg.get("render", {})
    preserve_original_color = bool(render_cfg.get("preserve_original_color", False))
    out_w, out_h = assets.mockup.shape[1], assets.mockup.shape[0]
    is_preview = bool(cfg.get("is_preview", False))
    preview_design_max_dim = max(256, int(cfg.get("preview_design_max_dim", 960)))

    design = (
        decode_design_preview(design_bytes, max_dim=preview_design_max_dim)
        if is_preview
        else decode_design(design_bytes)
    )
    logger.info('[PERF]   decode_design: %dms', int((time.perf_counter() - t_total) * 1000))
    design_transform = cfg.get("design_transform", {})
    canvas_w, canvas_h = estimate_print_area_canvas_size(
        cfg.get("print_area"),
        fallback_width=design.shape[1],
        fallback_height=design.shape[0],
    )
    t_dt = time.perf_counter()
    design = apply_design_transform(
        design,
        scale=design_transform.get("scale", 1.0),
        offset_x=design_transform.get("offset_x", 0.0),
        offset_y=design_transform.get("offset_y", 0.0),
        fit_mode=design_transform.get("fit_mode", "cover"),
        target_width=canvas_w,
        target_height=canvas_h,
    )
    logger.info('[PERF]   design_transform: %dms', int((time.perf_counter() - t_dt) * 1000))

    # If template provides a dense mesh config, prefer mesh-based warp renderer.
    t_warp = time.perf_counter()
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
            is_preview = cfg.get("is_preview", False)
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
                is_preview=is_preview,
            )
    else:
        is_preview = cfg.get("is_preview", False)
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
            is_preview=is_preview,
        )
    logger.info('[PERF]   cylindrical_warp: %dms', int((time.perf_counter() - t_warp) * 1000))

    mesh_src = cfg.get("print_area", {}).get("mesh_control_src")
    mesh_dst = cfg.get("print_area", {}).get("mesh_control_dst")
    if (
        isinstance(mesh_src, list)
        and isinstance(mesh_dst, list)
        and len(mesh_src) >= 4
        and len(mesh_src) == len(mesh_dst)
    ):
        is_preview = cfg.get("is_preview", False)
        warped = apply_smooth_mesh_warp(
            warped,
            src_pts=np.float32(mesh_src),
            dst_pts=np.float32(mesh_dst),
            output_size=(out_w, out_h),
            is_preview=is_preview,
        )
        logger.info('[PERF]   smooth_mesh_warp: %dms', int((time.perf_counter() - t_warp) * 1000))

    is_preview = cfg.get("is_preview", False)

    if is_preview:
        t_comp = time.perf_counter()
        result = composite(
            assets.mockup,
            warped,
            assets.mask,
            feather_px=cfg.get("edge", {}).get("feather_px", 6),
        )
        logger.info('[PERF]   composite: %dms', int((time.perf_counter() - t_comp) * 1000))
        logger.info('[PERF]   ===== TOTAL mug_pipeline (Preview): %dms =====', int((time.perf_counter() - t_total) * 1000))
        return result


    if (not preserve_original_color) and color_cfg.get("enable_color_match", True):
        t_cm = time.perf_counter()
        warped = apply_color_match(
            warped,
            assets.mockup,
            assets.mask,
            strength=color_cfg.get("match_strength", 0.40),
        )
        logger.info('[PERF]   color_match: %dms', int((time.perf_counter() - t_cm) * 1000))

    lighting_mask = warped[:, :, 3].astype(np.float32) / 255.0
    base_mask = assets.mask.astype(np.float32)
    if base_mask.max() > 1.0:
        base_mask /= 255.0
    lighting_mask = np.clip(lighting_mask * base_mask, 0.0, 1.0)

    t_lighting_phase = time.perf_counter()
    logger = logging.getLogger('mockup_service')
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        def _extract():
            t0 = time.perf_counter()
            res = extract_masked_lighting(
                assets.mockup,
                lighting_mask,
                blur_kernel=int(lighting_cfg.get("lighting_blur_kernel", 21))
            )
            logger.info('[PERF]   extract_masked_lighting: %dms', int((time.perf_counter() - t0) * 1000))
            return res
            
        fut_extracted = executor.submit(_extract)
        
        def compute_scaled_geometry():
            SCALE = 0.25
            sw, sh = max(1, int(out_w * SCALE)), max(1, int(out_h * SCALE))
            small_lighting_mask = cv2.resize(lighting_mask, (sw, sh), interpolation=cv2.INTER_AREA)
            
            p_area = cfg["print_area"]
            p_scaled = {}
            for k, v in p_area.items():
                # Scale numeric coordinate pairs
                if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v[:2]):
                    p_scaled[k] = [v[0] * SCALE, v[1] * SCALE]
                # Scale list of coordinate pairs (e.g., mesh_control_src / mesh_control_dst)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (list, tuple)):
                    scaled_list = []
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) >= 2 and all(isinstance(x, (int, float)) for x in item[:2]):
                            scaled_list.append([item[0] * SCALE, item[1] * SCALE])
                        else:
                            scaled_list.append(item)
                    p_scaled[k] = scaled_list
                else:
                    p_scaled[k] = v
            
            t_surface = time.perf_counter()
            s_maps = _get_cached_surface_maps(
                sw, sh,
                float(cyl.get("theta_max_deg", 52.0)),
                float(cyl.get("edge_squeeze", 0.0)),
                float(cyl.get("squeeze_power", 2.0)),
                float(cyl.get("center_focus_width", 0.0)),
                json.dumps(p_scaled, sort_keys=True)
            )
            logger.info('[PERF]   build_cylinder_surface_maps (scaled/cached): %dms', int((time.perf_counter() - t_surface) * 1000))
            
            v_mask = s_maps["valid_mask"] & (small_lighting_mask > 1e-3)
            t_fields = time.perf_counter()

            ldir = build_light_direction(
                light_pos_x=float(lighting_cfg.get("light_pos_x", 0.62)),
                light_pos_y=float(lighting_cfg.get("light_pos_y", 0.32)),
                light_height=float(lighting_cfg.get("light_height", 55.0)),
            )

            g_diff = compute_directional_diffuse(s_maps["normals"], ldir, valid_mask=v_mask)
            g_diff = normalize_masked_field(g_diff, v_mask, empty_fill=1.0, outside_fill=0.0, low_percentile=2.0, high_percentile=98.0)
            g_diff = np.where(v_mask, 0.35 + 0.65 * g_diff, 0.0).astype(np.float32)

            lf = build_light_field(
                s_maps["surface_u"], s_maps["surface_v"], v_mask,
                light_pos_x=float(lighting_cfg.get("light_pos_x", 0.62)),
                light_pos_y=float(lighting_cfg.get("light_pos_y", 0.32)),
                softness=float(lighting_cfg.get("light_softness", 55.0)),
                contrast=float(lighting_cfg.get("light_contrast", 50.0)),
                theta_max_deg=cyl.get("theta_max_deg", 52.0),
                edge_squeeze=cyl.get("edge_squeeze", 0.0),
                squeeze_power=cyl.get("squeeze_power", 2.0),
                center_focus_width=cyl.get("center_focus_width", 0.0),
            )
            g_light = np.clip(g_diff * lf, 0.0, 1.0)
            
            specular_strength = float(lighting_cfg.get("specular_strength", 0.0))
            g_spec = None
            if (not preserve_original_color) and specular_strength > 0:
                h_t = np.clip(float(lighting_cfg.get("light_highlight", 60.0)) / 100.0, 0.0, 1.0)
                g_spec = compute_directional_specular(
                    s_maps["normals"], ldir, valid_mask=v_mask,
                    shininess=8.0 + h_t * 40.0,
                    blur_kernel=max(1, int(3 + round(float(lighting_cfg.get("light_softness", 55.0)) / 8.0) * 2) // 4)
                ) * (0.75 + h_t * 0.55)
            
            logger.info('[PERF]   light_field+diffuse+specular_geometry (scaled): %dms', int((time.perf_counter() - t_fields) * 1000))
            return g_light, g_spec
            
        fut_geometry = executor.submit(compute_scaled_geometry)
        
        specular_strength = float(lighting_cfg.get("specular_strength", 0.0))
        fut_detail = None
        if (not preserve_original_color) and specular_strength > 0:
            def _detail():
                t0 = time.perf_counter()
                res = extract_masked_highlight_detail(
                    assets.mockup,
                    lighting_mask,
                    int(lighting_cfg.get("highlight_extract_blur_kernel", 41)),
                    int(lighting_cfg.get("highlight_detail_blur_kernel", 9))
                )
                logger.info('[PERF]   extract_masked_highlight_detail: %dms', int((time.perf_counter() - t0) * 1000))
                return res
            fut_detail = executor.submit(_detail)

        extracted_lighting = fut_extracted.result()
        geometry_lighting_small, geometry_specular_small = fut_geometry.result()
        detail_highlight = fut_detail.result() if fut_detail else None

    full_valid_mask = lighting_mask > 1e-3
    geometry_lighting = cv2.resize(geometry_lighting_small, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    
    photo_lighting = np.where(full_valid_mask, 0.60 + 0.40 * extracted_lighting, 0.0).astype(np.float32)
    lighting_map = normalize_masked_field(
        photo_lighting * geometry_lighting, full_valid_mask, empty_fill=1.0, outside_fill=0.0,
        low_percentile=2.0, high_percentile=98.0
    )
    lighting_map = np.where(full_valid_mask, 0.28 + 0.72 * lighting_map, 0.0).astype(np.float32)

    highlight_map = None
    if (not preserve_original_color) and specular_strength > 0:
        t_high = time.perf_counter()
        geometry_specular = cv2.resize(geometry_specular_small, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        diffuse_highlight = derive_highlight_map(
            extracted_lighting,
            threshold=float(lighting_cfg.get("specular_threshold", 180)),
            blur_kernel=int(lighting_cfg.get("highlight_blur_kernel", 9)),
        )
        highlight_map = np.clip(
            geometry_specular
            + diffuse_highlight * float(lighting_cfg.get("diffuse_highlight_strength", 0.25))
            + detail_highlight * float(lighting_cfg.get("highlight_detail_strength", 0.35)),
            0.0, 1.0
        )
        logger.info('[PERF]   highlight_map computation: %dms', int((time.perf_counter() - t_high) * 1000))

    logger.info('[PERF]   parallel_lighting_phase: %dms', int((time.perf_counter() - t_lighting_phase) * 1000))

    shadow_strength = float(lighting_cfg.get("shadow_strength", 0.45))
    if (not preserve_original_color) and shadow_strength > 0:
        t_shadow = time.perf_counter()
        warped = apply_shadow_overlay(
            warped,
            lighting_map,
            strength=shadow_strength,
        )
        logger.info('[PERF]   apply_shadow_overlay: %dms', int((time.perf_counter() - t_shadow) * 1000))

    # Apply separated mug lighting (diffuse * lighting_map + additive specular)
    if (not preserve_original_color) and (highlight_map is not None):
        t_mug_light = time.perf_counter()
        warped = apply_mug_lighting(
            warped,
            light_map=lighting_map,
            specular_mask=highlight_map,
            specular_strength=specular_strength,
            shininess=float(lighting_cfg.get("shininess", 20.0)),
        )
        logger.info('[PERF]   apply_mug_lighting: %dms', int((time.perf_counter() - t_mug_light) * 1000))

    t_comp = time.perf_counter()
    result = composite(
        assets.mockup,
        warped,
        assets.mask,
        feather_px=cfg.get("edge", {}).get("feather_px", 6),
    )
    logger.info('[PERF]   composite: %dms', int((time.perf_counter() - t_comp) * 1000))
    logger.info('[PERF]   ===== TOTAL mug_pipeline: %dms =====', int((time.perf_counter() - t_total) * 1000))

    return result
