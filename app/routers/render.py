# app/routers/render.py

import asyncio
import copy
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import cv2
import numpy as np
import base64
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select

from app.config import getRenderDevice, isGpuEnabled, setRenderDevice
from app.db.database import async_session
from app.db.models import Template
from app.pipeline.clothes.clothes_pipeline import ClothesAssets
from app.pipeline.mugs.cylinder_math import summarize_horizontal_squeeze
from app.pipeline.mugs.mug_pipeline import MugAssets
from app.pipeline.mugs.specular_gloss import extract_specular_from_mockup
from app.pipeline.pipeline import run_pipeline
from app.pipeline.shared.decode import decode_design, decode_design_preview
from app.pipeline.shared.design_transform import apply_design_transform, estimate_print_area_canvas_size
from app.pipeline.mugs.cylindrical_warp import cylindrical_warp, compute_and_cache_cylindrical_map
from app.pipeline.shared.liveview_cache import (
    get_cylindrical_map_cache,
    make_cylindrical_map_cache_key,
    set_cylindrical_map_cache,
    get_preview_canvas_cache,
    make_preview_canvas_cache_key,
    set_preview_canvas_cache,
)
from app.pipeline.shared.smooth_mesh_warp import apply_smooth_mesh_warp
from app.services import template_registry
from app.services.render_persist import persistRenderResult
from app.services.url_analysis import INPUTS_DIR, resolve_or_download_remote_asset
from app.services.vision import auto_detect_print_area_v2, bake_normal_map, create_soft_mask
from app.services.render_queue import enqueue_adhoc_render, get_job

router = APIRouter(tags=['render'])

GRID_CELL_PX = 90
MIN_GRID_DIVS = 4
MAX_GRID_DIVS = 24
DEFAULT_MESH_DENSITY_STRENGTH = 2.0
DEFAULT_CURVE_CORRECTION_ALPHA = 0.7
DEFAULT_CURVE_SNAP_THRESHOLD_PX = 2.0
PREVIEW_PNG_COMPRESSION = 1


def _force_jpeg_output_format(requested_format: str | None) -> str:
    normalized = str(requested_format or "").strip().lower()
    if normalized not in {"jpg", "jpeg"}:
        logging.getLogger("mockup_service").info(
            "Coerce output_format '%s' -> 'jpg' for lightweight output",
            requested_format,
        )
    return "jpg"


def _to_finite_float(val: object, default: float) -> float:
    try:
        f = float(val) if val is not None else float(default)
        if np.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return float(default)


def _to_finite_int(val: object, default: int) -> int:
    try:
        return int(val) if val is not None else int(default)
    except (TypeError, ValueError):
        pass
    return int(default)


def _parse_config_json(config_json: str | None, default_factory=dict) -> dict:
    if not config_json:
        return default_factory()
    try:
        return json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail={"error": "invalid_json", "message": f"Malformed config JSON: {exc}"})


async def _resolve_adhoc_inputs(
    mockup_image: Optional[UploadFile],
    design_image: Optional[UploadFile],
    mockup_url: Optional[str],
    design_url: Optional[str],
    request_id: str,
) -> tuple[bytes, bytes]:
    """Resolve mockup and design bytes from file upload or URL.

    Returns (mockup_bytes, design_bytes). Raises HTTPException on missing input.
    """
    if mockup_image:
        mockup_bytes = await mockup_image.read()
    elif mockup_url:
        mockup_bytes, _ = await _read_asset_bytes_from_url(
            mockup_url,
            asset_kind='mockup',
            request_id=request_id,
        )
    else:
        raise HTTPException(status_code=400, detail='Missing mockup_image or mockup_url')

    if design_image:
        design_bytes = await design_image.read()
    elif design_url:
        design_bytes, _ = await _read_asset_bytes_from_url(
            design_url,
            asset_kind='design',
            request_id=request_id,
        )
    else:
        raise HTTPException(status_code=400, detail='Missing design_image or design_url')

    return mockup_bytes, design_bytes


def _decode_and_resize_mockup(
    mockup_bytes: bytes,
    is_preview: bool = False,
) -> tuple[np.ndarray, float]:
    """Decode mockup from bytes and resize to max_dim cap.

    Returns (mockup_bgr, scale_factor). Raises ValueError on invalid image.
    """
    buf = np.frombuffer(mockup_bytes, dtype=np.uint8)
    mockup = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if mockup is None:
        raise ValueError('invalid_mockup')

    h, w = mockup.shape[:2]
    max_dim = 512 if is_preview else 3000
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        mockup = cv2.resize(
            mockup,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
    return mockup, scale


def _build_adhoc_pipeline_config(
    mockup: np.ndarray,
    config_json: str | None,
    scale: float,
    is_preview: bool,
) -> dict:
    """Build default adhoc config and merge user overrides.

    Returns the fully merged pipeline config dict.
    """
    h, w = mockup.shape[:2]
    config = _build_default_adhoc_config(w, h)
    config = _merge_adhoc_user_config(config, config_json, scale)
    config['is_preview'] = is_preview
    return config


def _target_dir_for_remote_asset(asset_kind: str) -> Path:
    normalized_kind = str(asset_kind or "").strip().lower()
    if normalized_kind == "mockup":
        return (INPUTS_DIR / "bases").resolve()
    return (INPUTS_DIR / "artworks").resolve()


async def _read_asset_bytes_from_url(
    raw_url: str,
    *,
    asset_kind: str,
    request_id: str,
) -> tuple[bytes, Path]:
    normalized_url = str(raw_url or "").strip()
    if not normalized_url:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"missing_{asset_kind}",
                "message": f"Missing {asset_kind} URL",
                "request_id": request_id,
            },
        )

    target_dir = _target_dir_for_remote_asset(asset_kind)
    try:
        local_path = await resolve_or_download_remote_asset(
            normalized_url,
            target_dir=target_dir,
            reuse_local=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"{asset_kind}_resolve_failed",
                "message": f"Cannot resolve {asset_kind} URL {normalized_url}: {exc}",
                "request_id": request_id,
            },
        ) from exc

    if not local_path.exists():
        raise HTTPException(
            status_code=400,
            detail={
                "error": f"{asset_kind}_not_found",
                "message": f"{asset_kind.capitalize()} URL {normalized_url} not found",
                "request_id": request_id,
            },
        )

    try:
        return local_path.read_bytes(), local_path
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"{asset_kind}_read_failed",
                "message": f"Failed to read local {asset_kind} file {local_path}: {exc}",
                "request_id": request_id,
            },
        ) from exc


def _log_horizontal_squeeze_debug(
    source: str,
    theta_max_deg: float,
    edge_squeeze: float,
    squeeze_power: float,
    center_focus_width: float,
) -> None:
    logger = logging.getLogger("mockup_service")
    debug = summarize_horizontal_squeeze(
        theta_max_deg=theta_max_deg,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
    )
    logger.info(
        "%s horizontal squeeze debug: edge_squeeze=%s squeeze_power=%s center_focus_width=%s active=%s mode=%s inactive_reason=%s sample_in=%s sample_out=%s sample_delta=%s",
        source,
        edge_squeeze,
        squeeze_power,
        center_focus_width,
        debug["active"],
        debug["mode"],
        debug["inactive_reason"],
        debug["sample_in"],
        debug["sample_out"],
        debug["sample_delta"],
    )


import gc


class WarpPreviewRequest(BaseModel):
    mockup_width: int
    mockup_height: int
    print_area: dict
    warp_type: str = "cylinder"
    theta_max_deg: float = 52.0
    curve: float = 0.0
    curve_top: Optional[float] = None
    curve_bottom: Optional[float] = None
    edge_squeeze: float = 0.0
    squeeze_power: float = 2.0
    center_focus_width: float = 0.0
    mesh_density_strength: float = DEFAULT_MESH_DENSITY_STRENGTH
    curve_correction_alpha: float = DEFAULT_CURVE_CORRECTION_ALPHA
    curve_snap_threshold_px: float = DEFAULT_CURVE_SNAP_THRESHOLD_PX
    design_scale: float = 1.0
    design_offset_x: float = 0.0
    design_offset_y: float = 0.0
    design_fit_mode: str = "cover"
    mask_points: Optional[list[list[int]]] = None
    template_id: Optional[str] = None
    # New inputs: either a URL to a design image, or a base64-encoded design image
    design_url: Optional[str] = None
    design_base64: Optional[str] = None




class RenderDeviceRequest(BaseModel):
    device: str


def _resolve_product_type(raw_product_type: Optional[str], warp_type: str, has_manual_mesh: bool) -> str:
    if isinstance(raw_product_type, str):
        lowered = raw_product_type.lower()
        if lowered.startswith("apparel"):
            return "clothes"
        if lowered in {"clothes", "tshirt", "hoodie", "tote"}:
            return "clothes"
        if lowered in {"mug", "cylinder_ceramic", "cylinder_glass", "cylinder_travel"}:
            return "mug"
    if warp_type == "tps":
        return "clothes"
    return "mug"


def _build_default_adhoc_config(width: int, height: int) -> dict:
    mx, my = int(width * 0.22), int(height * 0.14)
    return {
        "product_type": "mug",
        "cylinder": {
            "theta_max_deg": 52.0,
            "smile_base": 0.08,
            "pitch": 0.0,
            "curve_top": None,
            "curve_bottom": None,
            "edge_squeeze": 0.0,
            "squeeze_power": 2.0,
            "center_focus_width": 0.0,
            "mesh_density_strength": DEFAULT_MESH_DENSITY_STRENGTH,
            "curve_correction_alpha": DEFAULT_CURVE_CORRECTION_ALPHA,
            "curve_snap_threshold_px": DEFAULT_CURVE_SNAP_THRESHOLD_PX,
        },
        "mesh": {
            "n_points": 80,
            "displacement_strength": 6.0,
        },
        "print_area": {
            "top_left": [mx, my],
            "top_right": [width - mx, my],
            "bottom_right": [width - mx, height - my],
            "bottom_left": [mx, height - my],
        },
        "design_transform": {
            "scale": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
            "fit_mode": "cover",
        },
        "lighting": {
            # Ad-hoc mug defaults aim for realism instead of a flat raw overlay.
            "shadow_strength": 0.45,
            "displacement_strength": 0.0,
            "specular_strength": 0.30,
            "specular_threshold": 180,
            "light_pos_x": 0.62,
            "light_pos_y": 0.32,
            "light_height": 55.0,
            "light_contrast": 50.0,
            "light_highlight": 60.0,
            "light_softness": 55.0,
            "cylinder_shading_strength": 0.28,
            "edge_darkening_strength": 0.12,
            "specular_line_strength": 0.65,
            "specular_line_position": 0.18,
            "specular_line_sigma": 0.12,
            "specular_line_blur_kernel": 11,
            "diffuse_highlight_strength": 0.25,
            "highlight_detail_strength": 0.35,
            "lighting_blur_kernel": 21,
            "highlight_blur_kernel": 9,
            "highlight_extract_blur_kernel": 41,
            "highlight_detail_blur_kernel": 9,
        },
        "color": {
            "enable_color_match": False,
            "match_strength": 0.0,
        },
        "edge": {
            "feather_px": 0,
        },
        "render": {
            "preserve_original_color": False,
        },
        "output": {
            "jpeg_quality": 90,
        },
    }


def _merge_adhoc_user_config(config: dict, config_json: Optional[str], scale_factor: float = 1.0) -> dict:
    if not config_json:
        return config

    try:
        user_config = _parse_config_json(config_json)
    except HTTPException:
        return config

    def _normalize_design_fit_mode(value: object) -> str:
        lowered = str(value or "cover").strip().lower()
        return lowered if lowered in {"cover", "contain"} else "cover"

    user_pa = user_config.get("print_area", {})
    if isinstance(user_pa, dict):
        # Scale specific coordinates
        if scale_factor != 1.0:
            for pt_key in ["top_left", "top_right", "bottom_right", "bottom_left"]:
                if pt_key in user_pa and isinstance(user_pa[pt_key], list) and len(user_pa[pt_key]) >= 2:
                    user_pa[pt_key] = [user_pa[pt_key][0] * scale_factor, user_pa[pt_key][1] * scale_factor]
            
            for list_key in ["mask_points", "mesh_control_src", "mesh_control_dst"]:
                if list_key in user_pa and isinstance(user_pa[list_key], list):
                    user_pa[list_key] = [
                        [pt[0] * scale_factor, pt[1] * scale_factor] if isinstance(pt, list) and len(pt) >= 2 else pt
                        for pt in user_pa[list_key]
                    ]

        for key in [
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left",
            "mask_points",
            "camera_elevation",
            "mesh_control_src",
            "mesh_control_dst",
            "product_type",
        ]:
            if key in user_pa:
                config["print_area"][key] = user_pa[key]

    user_warp = user_config.get("warp", {})
    if not isinstance(user_warp, dict):
        user_warp = {}

    mesh_src = config["print_area"].get("mesh_control_src")
    mesh_dst = config["print_area"].get("mesh_control_dst")
    has_manual_mesh = (
        isinstance(mesh_src, list)
        and isinstance(mesh_dst, list)
        and len(mesh_src) >= 3
        and len(mesh_src) == len(mesh_dst)
    )

    warp_type = str(user_warp.get("warp_type", "")).lower()
    raw_product_type = (
        user_warp.get("product_type")
        or user_config.get("product_type")
        or config["print_area"].get("product_type")
    )
    config["product_type"] = _resolve_product_type(raw_product_type, warp_type, has_manual_mesh)

    config["cylinder"]["theta_max_deg"] = _to_finite_float(
        user_warp.get("theta_max_deg"), config["cylinder"]["theta_max_deg"]
    )
    config["cylinder"]["smile_base"] = _to_finite_float(
        user_warp.get("curve"), config["cylinder"]["smile_base"]
    )
    if "curve_top" in user_warp and "curve_bottom" in user_warp:
        if user_warp.get("curve_top") is not None and user_warp.get("curve_bottom") is not None:
            config["cylinder"]["curve_top"] = _to_finite_float(user_warp.get("curve_top"), 0.0)
            config["cylinder"]["curve_bottom"] = _to_finite_float(user_warp.get("curve_bottom"), 0.0)
        else:
            config["cylinder"]["curve_top"] = None
            config["cylinder"]["curve_bottom"] = None
    else:
        config["cylinder"]["curve_top"] = None
        config["cylinder"]["curve_bottom"] = None

    config["cylinder"]["pitch"] = _to_finite_float(
        config["print_area"].get(
            "camera_elevation", user_warp.get("camera_elevation", config["cylinder"]["pitch"])
        ), config["cylinder"]["pitch"]
    )
    config["cylinder"]["edge_squeeze"] = _to_finite_float(
        user_warp.get("edge_squeeze"), config["cylinder"]["edge_squeeze"]
    )
    config["cylinder"]["squeeze_power"] = _to_finite_float(
        user_warp.get("squeeze_power"), config["cylinder"]["squeeze_power"]
    )
    config["cylinder"]["center_focus_width"] = _to_finite_float(
        user_warp.get("center_focus_width"), config["cylinder"]["center_focus_width"]
    )
    config["cylinder"]["mesh_density_strength"] = _to_finite_float(
        user_warp.get("mesh_density_strength"), config["cylinder"]["mesh_density_strength"]
    )
    config["cylinder"]["curve_correction_alpha"] = _to_finite_float(
        user_warp.get("curve_correction_alpha"), config["cylinder"]["curve_correction_alpha"]
    )
    config["cylinder"]["curve_snap_threshold_px"] = _to_finite_float(
        user_warp.get("curve_snap_threshold_px"), config["cylinder"]["curve_snap_threshold_px"]
    )

    if "feather_radius" in user_warp:
        config["edge"]["feather_px"] = _to_finite_int(user_warp.get("feather_radius"), config["edge"]["feather_px"])

    user_edge = user_config.get("edge", {})
    if isinstance(user_edge, dict) and "feather_px" in user_edge:
        config["edge"]["feather_px"] = _to_finite_int(user_edge.get("feather_px"), config["edge"]["feather_px"])

    user_color = user_config.get("color", {})
    if isinstance(user_color, dict):
        if "enable_color_match" in user_color:
            config["color"]["enable_color_match"] = bool(user_color.get("enable_color_match"))
        if "match_strength" in user_color:
            config["color"]["match_strength"] = _to_finite_float(user_color.get("match_strength"), config["color"]["match_strength"])

    user_lighting = user_config.get("lighting", {})
    if not isinstance(user_lighting, dict):
        user_lighting = {}

    # Helper to pick up value from either lighting block or warp block
    def _get_lighting_val(key, default):
        val = user_lighting.get(key)
        if val is None:
            val = user_warp.get(key)
        return _to_finite_float(val, default) if key != "specular_threshold" and not key.endswith("kernel") else _to_finite_int(val, default)

    if user_lighting or user_warp:
        if "shadow_strength" in user_lighting or "shadow_strength" in user_warp:
            config["lighting"]["shadow_strength"] = _get_lighting_val("shadow_strength", config["lighting"]["shadow_strength"])
        if "displacement_strength" in user_lighting or "displacement_strength" in user_warp:
            config["lighting"]["displacement_strength"] = _get_lighting_val("displacement_strength", config["lighting"]["displacement_strength"])
        if "specular_strength" in user_lighting or "specular_strength" in user_warp:
            config["lighting"]["specular_strength"] = _get_lighting_val("specular_strength", config["lighting"]["specular_strength"])
        if "specular_threshold" in user_lighting or "specular_threshold" in user_warp:
            config["lighting"]["specular_threshold"] = _get_lighting_val("specular_threshold", config["lighting"]["specular_threshold"])
        if "light_pos_x" in user_lighting or "light_pos_x" in user_warp:
            config["lighting"]["light_pos_x"] = _get_lighting_val("light_pos_x", config["lighting"]["light_pos_x"])
        if "light_pos_y" in user_lighting or "light_pos_y" in user_warp:
            config["lighting"]["light_pos_y"] = _get_lighting_val("light_pos_y", config["lighting"]["light_pos_y"])
        if "light_height" in user_lighting or "light_height" in user_warp:
            config["lighting"]["light_height"] = _get_lighting_val("light_height", config["lighting"]["light_height"])
        if "light_contrast" in user_lighting or "light_contrast" in user_warp:
            config["lighting"]["light_contrast"] = _get_lighting_val("light_contrast", config["lighting"]["light_contrast"])
        if "light_highlight" in user_lighting or "light_highlight" in user_warp:
            config["lighting"]["light_highlight"] = _get_lighting_val("light_highlight", config["lighting"]["light_highlight"])
        if "light_softness" in user_lighting or "light_softness" in user_warp:
            config["lighting"]["light_softness"] = _get_lighting_val("light_softness", config["lighting"]["light_softness"])
        if "cylinder_shading_strength" in user_lighting or "cylinder_shading_strength" in user_warp:
            config["lighting"]["cylinder_shading_strength"] = _get_lighting_val("cylinder_shading_strength", config["lighting"]["cylinder_shading_strength"])
        if "edge_darkening_strength" in user_lighting or "edge_darkening_strength" in user_warp:
            config["lighting"]["edge_darkening_strength"] = _get_lighting_val("edge_darkening_strength", config["lighting"]["edge_darkening_strength"])
        if "specular_line_strength" in user_lighting or "specular_line_strength" in user_warp:
            config["lighting"]["specular_line_strength"] = _get_lighting_val("specular_line_strength", config["lighting"]["specular_line_strength"])
        if "specular_line_position" in user_lighting or "specular_line_position" in user_warp:
            config["lighting"]["specular_line_position"] = _get_lighting_val("specular_line_position", config["lighting"]["specular_line_position"])
        if "specular_line_sigma" in user_lighting or "specular_line_sigma" in user_warp:
            config["lighting"]["specular_line_sigma"] = _get_lighting_val("specular_line_sigma", config["lighting"]["specular_line_sigma"])
        if "specular_line_blur_kernel" in user_lighting or "specular_line_blur_kernel" in user_warp:
            config["lighting"]["specular_line_blur_kernel"] = _get_lighting_val("specular_line_blur_kernel", config["lighting"]["specular_line_blur_kernel"])
        if "diffuse_highlight_strength" in user_lighting or "diffuse_highlight_strength" in user_warp:
            config["lighting"]["diffuse_highlight_strength"] = _get_lighting_val("diffuse_highlight_strength", config["lighting"]["diffuse_highlight_strength"])
        if "highlight_detail_strength" in user_lighting or "highlight_detail_strength" in user_warp:
            config["lighting"]["highlight_detail_strength"] = _get_lighting_val("highlight_detail_strength", config["lighting"]["highlight_detail_strength"])
        if "lighting_blur_kernel" in user_lighting or "lighting_blur_kernel" in user_warp:
            config["lighting"]["lighting_blur_kernel"] = _get_lighting_val("lighting_blur_kernel", config["lighting"]["lighting_blur_kernel"])
        if "highlight_blur_kernel" in user_lighting or "highlight_blur_kernel" in user_warp:
            config["lighting"]["highlight_blur_kernel"] = _get_lighting_val("highlight_blur_kernel", config["lighting"]["highlight_blur_kernel"])
        if "highlight_extract_blur_kernel" in user_lighting or "highlight_extract_blur_kernel" in user_warp:
            config["lighting"]["highlight_extract_blur_kernel"] = _get_lighting_val("highlight_extract_blur_kernel", config["lighting"]["highlight_extract_blur_kernel"])
        if "highlight_detail_blur_kernel" in user_lighting or "highlight_detail_blur_kernel" in user_warp:
            config["lighting"]["highlight_detail_blur_kernel"] = _get_lighting_val("highlight_detail_blur_kernel", config["lighting"]["highlight_detail_blur_kernel"])

    user_render = user_config.get("render", {})
    if isinstance(user_render, dict) and "preserve_original_color" in user_render:
        config.setdefault("render", {})
        config["render"]["preserve_original_color"] = bool(user_render.get("preserve_original_color"))

    # Keep ad-hoc mug renders color-faithful by default, but preserve richer
    # clothes defaults unless caller explicitly overrides color/lighting keys.
    if config["product_type"] == "clothes":
        if not isinstance(user_color, dict) or "enable_color_match" not in user_color:
            config["color"]["enable_color_match"] = True
        if not isinstance(user_color, dict) or "match_strength" not in user_color:
            config["color"]["match_strength"] = 0.25

        if not isinstance(user_lighting, dict) or "shadow_strength" not in user_lighting:
            config["lighting"]["shadow_strength"] = 0.40
        if not isinstance(user_lighting, dict) or "displacement_strength" not in user_lighting:
            config["lighting"]["displacement_strength"] = 0.08
        if not isinstance(user_lighting, dict) or "specular_strength" not in user_lighting:
            config["lighting"]["specular_strength"] = 0.08
        if not isinstance(user_edge, dict) or "feather_px" not in user_edge:
            config["edge"]["feather_px"] = 10
        if not isinstance(user_render, dict) or "preserve_original_color" not in user_render:
            config.setdefault("render", {})
            config["render"]["preserve_original_color"] = False

    config["design_transform"] = {
        "scale": _to_finite_float(user_warp.get("design_scale"), config["design_transform"]["scale"]),
        "offset_x": _to_finite_float(user_warp.get("design_offset_x"), config["design_transform"]["offset_x"]),
        "offset_y": _to_finite_float(user_warp.get("design_offset_y"), config["design_transform"]["offset_y"]),
        "fit_mode": _normalize_design_fit_mode(
            user_warp.get("design_fit_mode", config["design_transform"].get("fit_mode", "cover"))
        ),
    }

    if has_manual_mesh:
        config["mesh"]["control_src"] = mesh_src
        config["mesh"]["control_dst"] = mesh_dst

    return config


def _clone_assets_with_config(assets: MugAssets | ClothesAssets, config: dict) -> MugAssets | ClothesAssets:
    if isinstance(assets, MugAssets):
        return MugAssets(
            mockup=assets.mockup,
            shadow_map=assets.shadow_map,
            normal_map=assets.normal_map,
            mask=assets.mask,
            specular_map=assets.specular_map,
            config=config,
        )

    if isinstance(assets, ClothesAssets):
        return ClothesAssets(
            mockup=assets.mockup,
            wrinkle_map=assets.wrinkle_map,
            shadow_map=assets.shadow_map,
            mask=assets.mask,
            config=config,
            slug=assets.slug,
        )

    return assets


def _scale_point_pair(value: object, scale: float) -> object:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return value
    if not all(isinstance(v, (int, float)) for v in value[:2]):
        return value
    scaled = [float(value[0]) * scale, float(value[1]) * scale]
    if len(value) > 2:
        scaled.extend(value[2:])
    return scaled


def _scale_point_list(values: object, scale: float) -> object:
    if not isinstance(values, list):
        return values
    scaled_points = []
    for item in values:
        scaled_points.append(_scale_point_pair(item, scale))
    return scaled_points


def _scale_print_area_config(print_area: dict, scale: float) -> dict:
    scaled_print_area = copy.deepcopy(print_area)
    for key in ("top_left", "top_right", "bottom_right", "bottom_left"):
        if key in scaled_print_area:
            scaled_print_area[key] = _scale_point_pair(scaled_print_area[key], scale)
    for key in ("mask_points", "mesh_control_src", "mesh_control_dst"):
        if key in scaled_print_area:
            scaled_print_area[key] = _scale_point_list(scaled_print_area[key], scale)
    return scaled_print_area


def _build_preview_assets(
    assets: MugAssets | ClothesAssets,
    max_dim: int = 512,
) -> MugAssets | ClothesAssets:
    safe_max_dim = max(128, min(2048, int(max_dim)))
    src_h, src_w = assets.mockup.shape[:2]
    src_max_dim = max(src_w, src_h)

    preview_config = copy.deepcopy(assets.config)
    preview_config["is_preview"] = True
    preview_config["preview_design_max_dim"] = max(512, safe_max_dim * 2)

    if src_max_dim <= safe_max_dim:
        return _clone_assets_with_config(assets, preview_config)

    scale = safe_max_dim / float(src_max_dim)
    dst_w = max(1, int(round(src_w * scale)))
    dst_h = max(1, int(round(src_h * scale)))

    if isinstance(preview_config.get("print_area"), dict):
        preview_config["print_area"] = _scale_print_area_config(preview_config["print_area"], scale)
    if isinstance(preview_config.get("mesh"), dict):
        mesh_cfg = preview_config["mesh"]
        if "control_src" in mesh_cfg:
            mesh_cfg["control_src"] = _scale_point_list(mesh_cfg.get("control_src"), scale)
        if "control_dst" in mesh_cfg:
            mesh_cfg["control_dst"] = _scale_point_list(mesh_cfg.get("control_dst"), scale)

    def _resize(arr: np.ndarray | None, interpolation: int) -> np.ndarray | None:
        if arr is None:
            return None
        return cv2.resize(arr, (dst_w, dst_h), interpolation=interpolation)

    if isinstance(assets, MugAssets):
        return MugAssets(
            mockup=_resize(assets.mockup, cv2.INTER_AREA),
            shadow_map=_resize(assets.shadow_map, cv2.INTER_AREA),
            normal_map=_resize(assets.normal_map, cv2.INTER_LINEAR),
            mask=_resize(assets.mask, cv2.INTER_NEAREST),
            specular_map=_resize(assets.specular_map, cv2.INTER_LINEAR),
            config=preview_config,
        )

    if isinstance(assets, ClothesAssets):
        return ClothesAssets(
            mockup=_resize(assets.mockup, cv2.INTER_AREA),
            wrinkle_map=_resize(assets.wrinkle_map, cv2.INTER_AREA),
            shadow_map=_resize(assets.shadow_map, cv2.INTER_AREA),
            mask=_resize(assets.mask, cv2.INTER_NEAREST),
            config=preview_config,
            slug=assets.slug,
        )

    return _clone_assets_with_config(assets, preview_config)


def _compute_adaptive_grid(width: int, height: int, cell_px: int = GRID_CELL_PX) -> tuple[int, int]:
    safe_w = max(1, int(width))
    safe_h = max(1, int(height))
    cols = int(np.clip(np.round(safe_w / max(cell_px, 1)), MIN_GRID_DIVS, MAX_GRID_DIVS))
    rows = int(np.clip(np.round(safe_h / max(cell_px, 1)), MIN_GRID_DIVS, MAX_GRID_DIVS))
    return cols, rows


def _compute_adaptive_axis_edges(divisions: int, density_strength: float = DEFAULT_MESH_DENSITY_STRENGTH) -> np.ndarray:
    divs = max(1, int(divisions))
    samples = np.linspace(0.0, 1.0, divs + 1, dtype=np.float32)
    power = max(1.0, float(density_strength))
    if power <= 1.0 + 1e-8:
        return samples

    edges = np.empty_like(samples)
    left_mask = samples <= 0.5
    left_samples = samples[left_mask] / 0.5
    edges[left_mask] = 0.5 * np.power(left_samples, power)

    right_samples = (1.0 - samples[~left_mask]) / 0.5
    edges[~left_mask] = 1.0 - 0.5 * np.power(right_samples, power)

    edges = np.clip(edges, 0.0, 1.0)
    edges[0] = 0.0
    edges[-1] = 1.0
    return np.maximum.accumulate(edges)


def _build_preview_design_canvas(
    width: int,
    height: int,
    cell_px: int = GRID_CELL_PX,
    mesh_density_strength: float = DEFAULT_MESH_DENSITY_STRENGTH,
) -> np.ndarray:
    safe_w = max(1, int(width))
    safe_h = max(1, int(height))
    cache_key = make_preview_canvas_cache_key(
        width=safe_w,
        height=safe_h,
        mesh_density_strength=mesh_density_strength,
    )
    cached_canvas = get_preview_canvas_cache(cache_key)
    if cached_canvas is not None:
        return cached_canvas

    cols, rows = _compute_adaptive_grid(safe_w, safe_h, cell_px=cell_px)
    canvas = np.zeros((safe_h, safe_w, 4), dtype=np.uint8)

    x_edges = np.rint(_compute_adaptive_axis_edges(cols, mesh_density_strength) * safe_w).astype(np.int32)
    y_edges = np.linspace(0, safe_h, rows + 1, dtype=np.int32)

    for r in range(rows):
        y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
        for c in range(cols):
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            even = (r + c) % 2 == 0
            color = 240 if even else 160
            alpha = 100 if even else 55
            canvas[y0:y1, x0:x1, :3] = [color, color, color]
            canvas[y0:y1, x0:x1, 3] = alpha

    for x in x_edges:
        xi = int(np.clip(x, 0, safe_w - 1))
        cv2.line(canvas, (xi, 0), (xi, safe_h - 1), (255, 255, 255, 170), 1, cv2.LINE_AA)
    for y in y_edges:
        yi = int(np.clip(y, 0, safe_h - 1))
        cv2.line(canvas, (0, yi), (safe_w - 1, yi), (255, 255, 255, 170), 1, cv2.LINE_AA)
    return set_preview_canvas_cache(cache_key, canvas)


def _encode_preview_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(
        ".png",
        image,
        [int(cv2.IMWRITE_PNG_COMPRESSION), PREVIEW_PNG_COMPRESSION],
    )
    if not ok:
        raise HTTPException(status_code=500, detail="encode_failed")
    return bytes(buf)


def _render_warp_preview_image(
    req: WarpPreviewRequest,
    design_canvas: np.ndarray,
    reference_mockup: np.ndarray | None = None,
) -> np.ndarray:
    from app.pipeline.clothes.tps_warp import tps_warp_design
    from app.pipeline.mugs.cylindrical_warp import cylindrical_warp

    w, h = req.mockup_width, req.mockup_height
    pa = {
        "top_left": req.print_area["top_left"],
        "top_right": req.print_area["top_right"],
        "bottom_right": req.print_area["bottom_right"],
        "bottom_left": req.print_area["bottom_left"],
    }
    canvas_w, canvas_h = estimate_print_area_canvas_size(
        pa,
        fallback_width=design_canvas.shape[1],
        fallback_height=design_canvas.shape[0],
    )

    design_canvas = apply_design_transform(
        design_canvas,
        scale=req.design_scale,
        offset_x=req.design_offset_x,
        offset_y=req.design_offset_y,
        fit_mode=req.design_fit_mode,
        target_width=canvas_w,
        target_height=canvas_h,
    )

    if req.warp_type == "cylinder":
        smile_val = float(req.curve)
        pitch_val = float(req.print_area.get("camera_elevation", 0))
        _log_horizontal_squeeze_debug(
            source="warp-preview",
            theta_max_deg=req.theta_max_deg,
            edge_squeeze=req.edge_squeeze,
            squeeze_power=req.squeeze_power,
            center_focus_width=req.center_focus_width,
        )
        warped = cylindrical_warp(
            design_canvas,
            pa,
            (w, h),
            reference_mockup=reference_mockup,
            theta_max_deg=req.theta_max_deg,
            pitch=pitch_val,
            smile_base=smile_val,
            curve_top=req.curve_top,
            curve_bottom=req.curve_bottom,
            edge_squeeze=req.edge_squeeze,
            squeeze_power=req.squeeze_power,
            center_focus_width=req.center_focus_width,
            curve_correction_alpha=req.curve_correction_alpha,
            curve_snap_threshold_px=req.curve_snap_threshold_px,
            is_preview=True,
        )
        mesh_src = req.print_area.get("mesh_control_src", [])
        mesh_dst = req.print_area.get("mesh_control_dst", [])
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
                output_size=(w, h),
                is_preview=True,
            )
    elif req.warp_type == "tps":
        src_pts_px = np.float32(req.print_area.get("mesh_control_src", []))
        dst_pts = np.float32(req.print_area.get("mesh_control_dst", []))
        dh, dw = design_canvas.shape[:2]
        if len(src_pts_px) >= 3 and len(src_pts_px) == len(dst_pts):
            base = cv2.warpPerspective(
                design_canvas,
                cv2.getPerspectiveTransform(
                    np.float32([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]]),
                    np.float32([pa["top_left"], pa["top_right"], pa["bottom_right"], pa["bottom_left"]]),
                ),
                (w, h),
            )
            warped = tps_warp_design(base, src_pts_px, dst_pts, (w, h))
        else:
            warped = cv2.warpPerspective(
                design_canvas,
                cv2.getPerspectiveTransform(
                    np.float32([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]]),
                    np.float32([pa["top_left"], pa["top_right"], pa["bottom_right"], pa["bottom_left"]]),
                ),
                (w, h),
            )
    else:
        dh, dw = design_canvas.shape[:2]
        warped = cv2.warpPerspective(
            design_canvas,
            cv2.getPerspectiveTransform(
                np.float32([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]]),
                np.float32([pa["top_left"], pa["top_right"], pa["bottom_right"], pa["bottom_left"]]),
            ),
            (w, h),
        )

    if req.mask_points and warped.shape[2] == 4:
        mask_to_use = create_soft_mask(req.mask_points, (h, w), feather_radius=2)
        warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], mask_to_use)

    return warped



@router.post("/mockup/render")
async def renderMockup(
    design_image: Optional[UploadFile] = File(None),
    design_url: Optional[str] = Form(None),
    template_id: str = Form(...),
    output_format: str = Form("jpg"),
    jpeg_quality: int = Form(None),
    config_json: str = Form(None),
    is_preview: bool = Form(False),
    preview_max_dim: int = Form(512),
):
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    effective_output_format = _force_jpeg_output_format(output_format)
    persist_source_url: Optional[str] = design_url.strip() if isinstance(design_url, str) else None
    assets = template_registry.get(template_id)

    if assets is None:
        async with async_session() as session:
            stmt = select(Template).where(Template.slug == template_id, Template.status == "active")
            result = await session.execute(stmt)
            template = result.scalar_one_or_none()

        if template is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "template_not_found",
                    "message": f"Template '{template_id}' not found or inactive",
                    "request_id": request_id,
                },
            )

        record = {
            "mockup_path": template.mockup_path,
            "shadow_map_path": template.shadow_map_path,
            "normal_map_path": template.normal_map_path,
            "specular_path": template.specular_path,
            "mask_path": template.mask_path,
            "config": template.config,
            "output_width": template.output_width,
            "output_height": template.output_height,
        }
        try:
            assets = template_registry.load_template(template_id, record)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": "render_failed", "message": str(exc), "request_id": request_id},
            ) from exc

    if design_image is not None:
        design_bytes = await design_image.read()
    elif design_url:
        design_bytes, _ = await _read_asset_bytes_from_url(
            design_url,
            asset_kind="design",
            request_id=request_id,
        )
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "missing_design",
                "message": "Missing design_image or design_url",
                "request_id": request_id,
            },
        )

    effective_assets = assets
    if config_json:
        try:
            effective_config = _merge_adhoc_user_config(copy.deepcopy(assets.config), config_json)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_config_override",
                    "message": str(exc),
                    "request_id": request_id,
                },
            ) from exc
        effective_assets = _clone_assets_with_config(assets, effective_config)
    if is_preview:
        effective_assets = _build_preview_assets(effective_assets, max_dim=preview_max_dim)

    quality = (
        jpeg_quality
        if jpeg_quality is not None
        else effective_assets.config.get("output", {}).get("jpeg_quality", 90)
    )

    try:
        image_bytes, meta = await asyncio.to_thread(
            run_pipeline,
            design_bytes,
            effective_assets,
            effective_output_format,
            quality,
        )
    except ValueError as exc:
        error_code = str(exc)
        if error_code == "image_too_large":
            msg = "File > 10MB"
        elif error_code == "invalid_image":
            msg = "Invalid image data"
        else:
            msg = error_code
        raise HTTPException(
            status_code=400,
            detail={"error": error_code, "message": msg, "request_id": request_id},
        ) from exc
    except Exception as exc:
        import traceback

        logger = logging.getLogger("mockup_service")
        logger.error(f"Render failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={"error": "render_failed", "message": str(exc), "request_id": request_id},
        ) from exc

    # Persist result in background if the design came from a URL
    if persist_source_url and not is_preview:
        asyncio.create_task(
            persistRenderResult(persist_source_url, image_bytes, effective_output_format)
        )

    return Response(
        content=image_bytes,
        media_type=meta['content_type'],
        headers={
            'X-Processing-Time-Ms': str(meta['processing_time_ms']),
            'X-Template-Id': template_id,
            'X-Request-Id': request_id,
            'X-Preview-Mode': '1' if is_preview else '0',
        },
    )


@router.post("/mockup/render-preview")
async def renderMockupPreview(
    design_image: Optional[UploadFile] = File(None),
    design_url: Optional[str] = Form(None),
    template_id: str = Form(...),
    jpeg_quality: int = Form(None),
    config_json: str = Form(None),
    preview_max_dim: int = Form(512),
):
    return await renderMockup(
        design_image=design_image,
        design_url=design_url,
        template_id=template_id,
        output_format="jpg",
        jpeg_quality=jpeg_quality,
        config_json=config_json,
        is_preview=True,
        preview_max_dim=preview_max_dim,
    )


@router.post("/mockup/render-adhoc")
async def renderAdhoc(
    mockup_image: Optional[UploadFile] = File(None),
    design_image: Optional[UploadFile] = File(None),
    mockup_url: Optional[str] = Form(None),
    design_url: Optional[str] = Form(None),
    output_format: str = Form("jpg"),
    jpeg_quality: int = Form(None),
    config_json: str = Form(None),
    is_preview: bool = Form(False),
):
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    effective_output_format = _force_jpeg_output_format(output_format)
    persist_source_url: Optional[str] = design_url.strip() if isinstance(design_url, str) else None

    try:
        logger = logging.getLogger('mockup_service')

        # 1. Resolve inputs
        mockup_bytes, design_bytes = await _resolve_adhoc_inputs(
            mockup_image, design_image, mockup_url, design_url, request_id,
        )

        # 2–4. Decode mockup, synthesize maps, build config, create assets
        # Offload all CPU-intensive work to a thread so the event loop stays free.
        def _prepare_adhoc_assets():
            t_adhoc_start = time.perf_counter()
            mockup, scale = _decode_and_resize_mockup(mockup_bytes, is_preview)
            h, w = mockup.shape[:2]
            logger.info('[PERF] adhoc: mockup decode+resize: %dms (size=%dx%d)', int((time.perf_counter() - t_adhoc_start) * 1000), w, h)

            # Synthesize maps
            mask = np.full((h, w), 255, dtype=np.uint8)

            theta_max_rad = np.deg2rad(52.0)
            cols_arr = np.arange(w, dtype=np.float32)
            theta = (cols_arr / w - 0.5) * 2.0 * theta_max_rad
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            nx = np.broadcast_to(sin_t[None, :], (h, w))
            nz = np.broadcast_to(cos_t[None, :], (h, w))
            ny = np.zeros((h, w), dtype=np.float32)
            normal_map = np.dstack(
                [
                    ((nx * 0.5 + 0.5) * 255).astype(np.uint8),
                    ((ny * 0.5 + 0.5) * 255).astype(np.uint8),
                    ((nz * 0.5 + 0.5) * 255).astype(np.uint8),
                ]
            )

            shadow_intensity = np.broadcast_to(cos_t[None, :], (h, w))
            shadow_map = np.clip(shadow_intensity * 0.6 + 0.4, 0, 1)
            shadow_map = (shadow_map * 255).astype(np.uint8)

            # Build config
            t_config = time.perf_counter()
            config = _build_adhoc_pipeline_config(mockup, config_json, scale, is_preview)
            logger.info('[PERF] adhoc: config merge: %dms', int((time.perf_counter() - t_config) * 1000))
            lighting_cfg = config.get('lighting', {})
            spec_strength = float(lighting_cfg.get('specular_strength', 0.0))
            if spec_strength > 0:
                specular_threshold = int(lighting_cfg.get('specular_threshold', 220))
                specular_map = extract_specular_from_mockup(mockup, threshold=specular_threshold)
            else:
                specular_map = np.zeros((h, w), dtype=np.float32)
            logger.info(
                'render-adhoc effective warp: theta_max_deg=%s curve_top=%s curve_bottom=%s pitch=%s edge_squeeze=%s squeeze_power=%s center_focus_width=%s',
                config.get('cylinder', {}).get('theta_max_deg'),
                config.get('cylinder', {}).get('curve_top'),
                config.get('cylinder', {}).get('curve_bottom'),
                config.get('cylinder', {}).get('pitch'),
                config.get('cylinder', {}).get('edge_squeeze'),
                config.get('cylinder', {}).get('squeeze_power'),
                config.get('cylinder', {}).get('center_focus_width'),
            )
            _log_horizontal_squeeze_debug(
                source='render-adhoc',
                theta_max_deg=float(config.get('cylinder', {}).get('theta_max_deg', 52.0)),
                edge_squeeze=float(config.get('cylinder', {}).get('edge_squeeze', 0.0)),
                squeeze_power=float(config.get('cylinder', {}).get('squeeze_power', 2.0)),
                center_focus_width=float(config.get('cylinder', {}).get('center_focus_width', 0.0)),
            )

            product_type = config.get('product_type', 'mug')
            if product_type == 'mug':
                built_assets = MugAssets(
                    mockup=mockup,
                    shadow_map=shadow_map,
                    normal_map=normal_map,
                    mask=mask,
                    specular_map=specular_map,
                    config=config,
                )
            else:
                built_assets = ClothesAssets(
                    mockup=mockup,
                    wrinkle_map=normal_map,
                    shadow_map=shadow_map,
                    mask=mask,
                    config=config,
                )
            return built_assets, config

        assets, config = await asyncio.to_thread(_prepare_adhoc_assets)

        quality = (
            max(1, min(100, int(jpeg_quality)))
            if jpeg_quality is not None
            else int(config.get('output', {}).get('jpeg_quality', 90))
        )

        image_bytes, meta = await asyncio.to_thread(
            run_pipeline,
            design_bytes,
            assets,
            effective_output_format,
            quality,
            False,
        )

        # Persist result in background if the design came from a URL
        if persist_source_url and not is_preview:
            asyncio.create_task(
                persistRenderResult(persist_source_url, image_bytes, effective_output_format)
            )

        return Response(
            content=image_bytes,
            media_type=meta['content_type'],
            headers={
                'X-Processing-Time-Ms': str(meta['processing_time_ms']),
                'X-Template-Id': 'adhoc',
                'X-Request-Id': request_id,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        logger = logging.getLogger('mockup_service')
        logger.error(f'Render-adhoc failed: {traceback.format_exc()}')
        raise HTTPException(
            status_code=400,
            detail={'error': 'render_failed', 'message': str(exc), 'request_id': request_id},
        ) from exc


@router.post("/mockup/warp-preview")
async def renderWarpPreview(req: WarpPreviewRequest, background_tasks: BackgroundTasks):
    """Extended warp-preview: accepts `design_url` or `design_base64` in JSON payload.

    Renders preview, saves PNG under `static/renders/` and returns JSON with `preview_url` and `expires_at`.
    """
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    # determine design canvas
    design_canvas = None
    if req.design_base64:
        b64 = req.design_base64
        if b64.startswith('data:'):
            parts = b64.split(',', 1)
            if len(parts) == 2:
                b64 = parts[1]
        try:
            import base64 as _base64

            design_bytes = _base64.b64decode(b64)
            design_canvas = decode_design_preview(design_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"error": "invalid_base64", "message": str(exc), "request_id": request_id})
    elif req.design_url:
        design_bytes, _ = await _read_asset_bytes_from_url(req.design_url, asset_kind="design", request_id=request_id)
        try:
            design_canvas = decode_design_preview(design_bytes)
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"error": "invalid_design", "message": str(exc), "request_id": request_id})

    if design_canvas is None:
        canvas_w, canvas_h = estimate_print_area_canvas_size(
            req.print_area,
            fallback_width=req.mockup_width,
            fallback_height=req.mockup_height,
        )
        design_canvas = _build_preview_design_canvas(
            canvas_w,
            canvas_h,
            mesh_density_strength=req.mesh_density_strength,
        )

    try:
        warped = await asyncio.to_thread(_render_warp_preview_image, req, design_canvas)
        png_bytes = await asyncio.to_thread(_encode_preview_png, warped)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail={"error": "render_failed", "message": str(exc), "request_id": request_id})

    # Save PNG to static/renders with TTL
    renders_dir = Path('static') / 'renders'
    renders_dir.mkdir(parents=True, exist_ok=True)
    fname = f"preview-{uuid.uuid4().hex}.png"
    out_path = renders_dir / fname
    out_path.write_bytes(png_bytes)

    PREVIEW_TTL_SECONDS = 300

    async def _delete_later(p: str, delay: int):
        await asyncio.sleep(delay)
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    asyncio.create_task(_delete_later(str(out_path), PREVIEW_TTL_SECONDS))
    expires_at = (datetime.utcnow() + timedelta(seconds=PREVIEW_TTL_SECONDS)).isoformat() + 'Z'
    preview_url = f"/static/renders/{fname}"

    return {"preview_url": preview_url, "expires_at": expires_at}


@router.post('/mockup/render-async-adhoc')
async def render_adhoc_async(
    mockup_image: Optional[UploadFile] = File(None),
    design_image: Optional[UploadFile] = File(None),
    mockup_url: Optional[str] = Form(None),
    design_url: Optional[str] = Form(None),
    config_json: str = Form(None),
    output_format: str = Form('jpg'),
    is_preview: bool = Form(False),
):
    request_id = f"req_{uuid.uuid4().hex[:8]}"
    effective_output_format = _force_jpeg_output_format(output_format)
    try:
        # 1. Resolve inputs
        mockup_bytes, design_bytes = await _resolve_adhoc_inputs(
            mockup_image, design_image, mockup_url, design_url, request_id,
        )

        # 2. Decode + resize mockup (offload to thread)
        mockup, scale = await asyncio.to_thread(_decode_and_resize_mockup, mockup_bytes, is_preview)

        # 3. Build config (identical to synchronous /render-adhoc)
        effective_cfg = _build_adhoc_pipeline_config(mockup, config_json, scale, is_preview)

        ok, res_buf = cv2.imencode('.png', mockup)
        job_id = enqueue_adhoc_render(
            design_bytes,
            res_buf.tobytes(),
            effective_cfg,
            output_format=effective_output_format,
        )
        return {"job_id": job_id, "request_id": request_id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail={'error': 'enqueue_failed', 'message': str(exc), 'request_id': request_id}) from exc


@router.get('/mockup/render-status/{job_id}')
async def render_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={'error': 'not_found'})
    res = {
        'id': job['id'],
        'status': job.get('status'),
        'phase': job.get('phase'),
        'created_at': job.get('created_at'),
        'updated_at': job.get('updated_at'),
        'error': job.get('error'),
        'result': {},
    }
    if job.get('result'):
        final = job['result'].get('final_path')
        preview = job['result'].get('preview_path')
        if preview:
            res['result']['preview_url'] = f"/static/renders/{Path(preview).name}"
        if final:
            res['result']['final_url'] = f"/static/renders/{Path(final).name}"
        res['result']['content_type'] = job['result'].get('content_type')
        res['result']['processing_time_ms'] = job['result'].get('processing_time_ms')
    return res


@router.post("/mockup/warp-preview-file")
async def renderWarpPreviewFile(
    config_json: str = Form(...),
    design_image: UploadFile = File(...),
):
    try:
        payload = _parse_config_json(config_json)
        req = WarpPreviewRequest(**payload)
        design_bytes = await design_image.read()
        design_canvas = await asyncio.to_thread(decode_design_preview, design_bytes)
        warped = await asyncio.to_thread(_render_warp_preview_image, req, design_canvas)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    png_bytes = await asyncio.to_thread(_encode_preview_png, warped)
    return Response(content=png_bytes, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.post('/mockup/warp-preview-base64')
async def renderWarpPreviewBase64(
    config_json: str = Form(...),
    design_base64: str = Form(...),
):
    """Render warp preview from a base64-encoded design image provided in the form.

    Allows client to send design blob inline (data URL or raw base64) so preview
    can run in parallel with async upload.
    """
    try:
        payload = json.loads(config_json)
        req = WarpPreviewRequest(**payload)

        b64 = design_base64
        if b64.startswith('data:'):
            parts = b64.split(',', 1)
            if len(parts) == 2:
                b64 = parts[1]

        try:
            design_bytes = base64.b64decode(b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f'invalid_base64: {exc}')

        design_canvas = await asyncio.to_thread(decode_design_preview, design_bytes)
        warped = await asyncio.to_thread(_render_warp_preview_image, req, design_canvas)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    png_bytes = await asyncio.to_thread(_encode_preview_png, warped)
    return Response(content=png_bytes, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.post("/mockup/warp-prefetch", status_code=202)
async def renderWarpPrefetch(
    background_tasks: BackgroundTasks,
    config_json: str = Form(...),
):
    try:
        payload = _parse_config_json(config_json)
        req = WarpPreviewRequest(**payload)

        canvas_w, canvas_h = estimate_print_area_canvas_size(
            req.print_area,
            fallback_width=req.mockup_width,
            fallback_height=req.mockup_height,
        )

        def run_cache():
            from app.pipeline.mugs.cylindrical_warp import compute_and_cache_cylindrical_map
            compute_and_cache_cylindrical_map(
                print_area={
                    "top_left": req.print_area["top_left"],
                    "top_right": req.print_area["top_right"],
                    "bottom_right": req.print_area["bottom_right"],
                    "bottom_left": req.print_area["bottom_left"],
                },
                output_size=(req.mockup_width, req.mockup_height),
                design_size=(canvas_w, canvas_h),
                theta_max_deg=req.theta_max_deg,
                pitch=float(req.print_area.get("camera_elevation", 0)),
                smile_base=req.curve,
                curve_top=req.curve_top,
                curve_bottom=req.curve_bottom,
                edge_squeeze=req.edge_squeeze,
                squeeze_power=req.squeeze_power,
                center_focus_width=req.center_focus_width,
                curve_correction_alpha=req.curve_correction_alpha,
                curve_snap_threshold_px=req.curve_snap_threshold_px,
            )

        background_tasks.add_task(run_cache)
        return Response(status_code=202)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc



@router.post("/mockup/warp-preview-adhoc")
async def renderWarpPreviewAdhoc(
    config_json: str = Form(...),
    mockup_image: UploadFile = File(...),
    design_image: UploadFile | None = File(None),
):
    try:
        payload = _parse_config_json(config_json)
        req = WarpPreviewRequest(**payload)

        mockup_bytes = await mockup_image.read()
        mockup_buf = np.frombuffer(mockup_bytes, dtype=np.uint8)
        mockup = await asyncio.to_thread(cv2.imdecode, mockup_buf, cv2.IMREAD_COLOR)
        if mockup is None:
            raise HTTPException(status_code=400, detail="invalid_mockup")

        if design_image is not None:
            design_bytes = await design_image.read()
            design_canvas = await asyncio.to_thread(decode_design_preview, design_bytes)
        else:
            canvas_w, canvas_h = estimate_print_area_canvas_size(
                req.print_area,
                fallback_width=req.mockup_width,
                fallback_height=req.mockup_height,
            )
            design_canvas = _build_preview_design_canvas(
                canvas_w,
                canvas_h,
                mesh_density_strength=req.mesh_density_strength,
            )

        warped = await asyncio.to_thread(_render_warp_preview_image, req, design_canvas, mockup)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    png_bytes = await asyncio.to_thread(_encode_preview_png, warped)
    del warped
    gc.collect()
    return Response(content=png_bytes, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.post("/mockup/detect-region")
async def detectRegion(mockup_image: UploadFile = File(...)):
    content = await mockup_image.read()

    def _detect(raw_bytes: bytes):
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_small = cv2.resize(img, (int(w * scale), int(h * scale)))
            result = auto_detect_print_area_v2(img_small)
            for key in ["quad", "clip_mask"]:
                if result.get(key):
                    result[key] = [[int(p[0] / scale), int(p[1] / scale)] for p in result[key]]
            return result
        return auto_detect_print_area_v2(img)

    result = await asyncio.to_thread(_detect, content)
    if result is None:
        raise HTTPException(status_code=400, detail="invalid_mockup")
    return result


@router.post("/mockup/bake-normal")
async def bakeNormal(
    mockup_image: UploadFile = File(...),
    fold_map: UploadFile = File(None),
):
    content = await mockup_image.read()
    fold_content = None
    if fold_map:
        fold_content = await fold_map.read()

    def _bake(raw_bytes: bytes, fold_bytes: bytes | None):
        buf = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        f_map = None
        if fold_bytes:
            f_buf = np.frombuffer(fold_bytes, dtype=np.uint8)
            raw = cv2.imdecode(f_buf, cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                f_map = (raw.astype(np.float32) - 128.0) / 128.0
        normal_map = bake_normal_map(img, f_map)
        ok, encoded = cv2.imencode(".png", normal_map)
        if not ok:
            return False
        return bytes(encoded)

    result = await asyncio.to_thread(_bake, content, fold_content)
    if result is None:
        raise HTTPException(status_code=400, detail="invalid_mockup")
    if result is False:
        raise HTTPException(status_code=500, detail="encode_failed")
    return Response(content=result, media_type="image/png")


@router.get("/templates")
async def listTemplates(request: Request):
    asset_base_dir = Path(os.getenv("ASSET_BASE_DIR", "./templates"))

    async with async_session() as session:
        stmt = select(Template).where(Template.status == "active")
        result = await session.execute(stmt)
        templates = result.scalars().all()

    def _preview_url(slug: str) -> str:
        template_dir = asset_base_dir / slug
        if (template_dir / "preview.jpg").exists():
            return f"/static/templates/{slug}/preview.jpg"
        if (template_dir / "preview.png").exists():
            return f"/static/templates/{slug}/preview.png"
        return f"/static/templates/{slug}/mockup.jpg"

    return {
        "templates": [
            {
                "id": t.slug,
                "name": t.name,
                "preview_url": _preview_url(t.slug),
                "preview_url_full": f"{str(request.base_url).rstrip('/')}{_preview_url(t.slug)}",
                "output_size": [t.output_width, t.output_height],
            }
            for t in templates
        ]
    }


@router.get("/render/device")
async def getRenderDeviceState():
    return {
        "device": getRenderDevice(),
        "gpu_active": isGpuEnabled(),
        "opencl_available": bool(cv2.ocl.haveOpenCL()),
    }


@router.post("/render/device")
async def setRenderDeviceState(req: RenderDeviceRequest):
    try:
        gpu_active = setRenderDevice(req.device)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc

    return {
        "device": getRenderDevice(),
        "gpu_active": gpu_active,
        "opencl_available": bool(cv2.ocl.haveOpenCL()),
    }
