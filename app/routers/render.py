# app/routers/render.py

import asyncio
import json
import logging
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select

from app.config import getRenderDevice, isGpuEnabled, setRenderDevice
from app.db.database import async_session
from app.db.models import Template
from app.pipeline.clothes.clothes_pipeline import ClothesAssets
from app.pipeline.mugs.mug_pipeline import MugAssets
from app.pipeline.mugs.specular_gloss import extract_specular_from_mockup
from app.pipeline.pipeline import run_pipeline
from app.pipeline.shared.decode import decode_design
from app.pipeline.shared.design_transform import apply_design_transform, estimate_print_area_canvas_size
from app.services import template_registry
from app.services.vision import auto_detect_print_area_v2, bake_normal_map, create_soft_mask

router = APIRouter(tags=["render"])


class WarpPreviewRequest(BaseModel):
    mockup_width: int
    mockup_height: int
    print_area: dict
    warp_type: str = "cylinder"
    theta_max_deg: float = 52.0
    curve: float = 0.0
    curve_top: Optional[float] = None
    curve_bottom: Optional[float] = None
    design_scale: float = 1.0
    design_offset_x: float = 0.0
    design_offset_y: float = 0.0
    mask_points: Optional[list[list[int]]] = None
    template_id: Optional[str] = None


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
    if warp_type == "tps" or has_manual_mesh:
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
        },
        "lighting": {
            # Ad-hoc profile defaults to color fidelity over synthetic lighting.
            "shadow_strength": 0.0,
            "displacement_strength": 0.0,
            "specular_strength": 0.0,
            "specular_threshold": 245,
        },
        "color": {
            "enable_color_match": False,
            "match_strength": 0.0,
        },
        "edge": {
            "feather_px": 0,
        },
        "render": {
            "preserve_original_color": True,
        },
        "output": {
            "jpeg_quality": 90,
        },
    }


def _merge_adhoc_user_config(config: dict, config_json: Optional[str]) -> dict:
    if not config_json:
        return config

    import json

    try:
        user_config = json.loads(config_json)
    except Exception:
        return config

    user_pa = user_config.get("print_area", {})
    if isinstance(user_pa, dict):
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

    config["cylinder"]["theta_max_deg"] = float(
        user_warp.get("theta_max_deg", config["cylinder"]["theta_max_deg"])
    )
    config["cylinder"]["smile_base"] = float(
        user_warp.get("curve", config["cylinder"]["smile_base"])
    )
    if "curve_top" in user_warp and "curve_bottom" in user_warp:
        try:
            config["cylinder"]["curve_top"] = float(user_warp.get("curve_top"))
            config["cylinder"]["curve_bottom"] = float(user_warp.get("curve_bottom"))
        except (TypeError, ValueError):
            config["cylinder"]["curve_top"] = None
            config["cylinder"]["curve_bottom"] = None
    else:
        config["cylinder"]["curve_top"] = None
        config["cylinder"]["curve_bottom"] = None

    config["cylinder"]["pitch"] = float(
        config["print_area"].get(
            "camera_elevation", user_warp.get("camera_elevation", config["cylinder"]["pitch"])
        )
    )

    if "feather_radius" in user_warp:
        config["edge"]["feather_px"] = int(user_warp.get("feather_radius", config["edge"]["feather_px"]))

    user_edge = user_config.get("edge", {})
    if isinstance(user_edge, dict) and "feather_px" in user_edge:
        config["edge"]["feather_px"] = int(user_edge.get("feather_px", config["edge"]["feather_px"]))

    user_color = user_config.get("color", {})
    if isinstance(user_color, dict):
        if "enable_color_match" in user_color:
            config["color"]["enable_color_match"] = bool(user_color.get("enable_color_match"))
        if "match_strength" in user_color:
            config["color"]["match_strength"] = float(user_color.get("match_strength", config["color"]["match_strength"]))

    user_lighting = user_config.get("lighting", {})
    if isinstance(user_lighting, dict):
        if "shadow_strength" in user_lighting:
            config["lighting"]["shadow_strength"] = float(user_lighting.get("shadow_strength", config["lighting"]["shadow_strength"]))
        if "displacement_strength" in user_lighting:
            config["lighting"]["displacement_strength"] = float(
                user_lighting.get("displacement_strength", config["lighting"]["displacement_strength"])
            )
        if "specular_strength" in user_lighting:
            config["lighting"]["specular_strength"] = float(
                user_lighting.get("specular_strength", config["lighting"]["specular_strength"])
            )
        if "specular_threshold" in user_lighting:
            config["lighting"]["specular_threshold"] = int(
                user_lighting.get("specular_threshold", config["lighting"]["specular_threshold"])
            )

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
        "scale": float(user_warp.get("design_scale", config["design_transform"]["scale"])),
        "offset_x": float(user_warp.get("design_offset_x", config["design_transform"]["offset_x"])),
        "offset_y": float(user_warp.get("design_offset_y", config["design_transform"]["offset_y"])),
    }

    if has_manual_mesh:
        config["mesh"]["control_src"] = mesh_src
        config["mesh"]["control_dst"] = mesh_dst
        config["product_type"] = "clothes"

    return config


def _build_preview_design_canvas(size: int = 400) -> np.ndarray:
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    cell = max(16, size // 10)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            even = ((x // cell) + (y // cell)) % 2 == 0
            color = 240 if even else 160
            alpha = 100 if even else 55
            canvas[y : y + cell, x : x + cell, :3] = [color, color, color]
            canvas[y : y + cell, x : x + cell, 3] = alpha

    for v in range(0, size + 1, cell):
        cv2.line(canvas, (v, 0), (v, size - 1), (255, 255, 255, 170), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, v), (size - 1, v), (255, 255, 255, 170), 1, cv2.LINE_AA)
    return canvas


def _render_warp_preview_image(req: WarpPreviewRequest, design_canvas: np.ndarray) -> np.ndarray:
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
        target_width=canvas_w,
        target_height=canvas_h,
    )

    if req.warp_type == "cylinder":
        smile_val = float(req.curve)
        pitch_val = float(req.print_area.get("camera_elevation", 0))
        warped = cylindrical_warp(
            design_canvas,
            pa,
            (w, h),
            req.theta_max_deg,
            pitch_val,
            smile_val,
            req.curve_top,
            req.curve_bottom,
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
    design_image: UploadFile = File(...),
    template_id: str = Form(...),
    output_format: str = Form("jpg"),
    jpeg_quality: int = Form(None),
):
    request_id = f"req_{uuid.uuid4().hex[:8]}"
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

    design_bytes = await design_image.read()
    quality = jpeg_quality if jpeg_quality is not None else assets.config.get("output", {}).get("jpeg_quality", 90)

    try:
        image_bytes, meta = await asyncio.to_thread(run_pipeline, design_bytes, assets, output_format, quality)
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

    return Response(
        content=image_bytes,
        media_type=meta["content_type"],
        headers={
            "X-Processing-Time-Ms": str(meta["processing_time_ms"]),
            "X-Template-Id": template_id,
            "X-Request-Id": request_id,
        },
    )


@router.post("/mockup/render-adhoc")
async def renderAdhoc(
    mockup_image: UploadFile = File(...),
    design_image: UploadFile = File(...),
    output_format: str = Form("jpg"),
    config_json: str = Form(None),
):
    request_id = f"req_{uuid.uuid4().hex[:8]}"

    try:
        mockup_bytes = await mockup_image.read()
        design_bytes = await design_image.read()

        buf = np.frombuffer(mockup_bytes, dtype=np.uint8)
        mockup = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if mockup is None:
            raise ValueError("invalid_mockup")

        h, w = mockup.shape[:2]
        if max(h, w) > 3000:
            scale = 3000 / max(h, w)
            mockup = cv2.resize(mockup, (int(w * scale), int(h * scale)))
            h, w = mockup.shape[:2]

        mask = np.full((h, w), 255, dtype=np.uint8)

        theta_max_rad = np.deg2rad(52.0)
        cols = np.arange(w, dtype=np.float32)
        theta = (cols / w - 0.5) * 2.0 * theta_max_rad
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

        config = _build_default_adhoc_config(w, h)
        config = _merge_adhoc_user_config(config, config_json)
        lighting_cfg = config.get("lighting", {})
        specular_strength = float(lighting_cfg.get("specular_strength", 0.0))
        if specular_strength > 0:
            specular_threshold = int(lighting_cfg.get("specular_threshold", 220))
            specular_map = extract_specular_from_mockup(mockup, threshold=specular_threshold)
        else:
            specular_map = np.zeros((h, w), dtype=np.float32)
        logging.getLogger("mockup_service").info(
            "render-adhoc effective warp: theta_max_deg=%s curve_top=%s curve_bottom=%s pitch=%s",
            config.get("cylinder", {}).get("theta_max_deg"),
            config.get("cylinder", {}).get("curve_top"),
            config.get("cylinder", {}).get("curve_bottom"),
            config.get("cylinder", {}).get("pitch"),
        )

        product_type = config.get("product_type", "mug")
        if product_type == "mug":
            assets = MugAssets(
                mockup=mockup,
                shadow_map=shadow_map,
                normal_map=normal_map,
                mask=mask,
                specular_map=specular_map,
                config=config,
            )
        else:
            assets = ClothesAssets(
                mockup=mockup,
                wrinkle_map=normal_map,
                shadow_map=shadow_map,
                mask=mask,
                config=config,
            )

        image_bytes, meta = await asyncio.to_thread(run_pipeline, design_bytes, assets, output_format, 90)

        return Response(
            content=image_bytes,
            media_type=meta["content_type"],
            headers={
                "X-Processing-Time-Ms": str(meta["processing_time_ms"]),
                "X-Template-Id": "adhoc",
                "X-Request-Id": request_id,
            },
        )
    except Exception as exc:
        import traceback

        logger = logging.getLogger("mockup_service")
        logger.error(f"Render-adhoc failed: {traceback.format_exc()}")
        raise HTTPException(
            status_code=400,
            detail={"error": "render_failed", "message": str(exc), "request_id": request_id},
        ) from exc


@router.post("/mockup/warp-preview")
async def renderWarpPreview(req: WarpPreviewRequest):
    design_canvas = _build_preview_design_canvas(400)
    warped = _render_warp_preview_image(req, design_canvas)

    ok, buf = cv2.imencode(".png", warped)
    if not ok:
        raise HTTPException(status_code=500, detail="encode_failed")
    return Response(content=bytes(buf), media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.post("/mockup/warp-preview-file")
async def renderWarpPreviewFile(
    config_json: str = Form(...),
    design_image: UploadFile = File(...),
):
    try:
        payload = json.loads(config_json)
        req = WarpPreviewRequest(**payload)
        design_bytes = await design_image.read()
        design_canvas = decode_design(design_bytes)
        warped = _render_warp_preview_image(req, design_canvas)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    ok, buf = cv2.imencode(".png", warped)
    if not ok:
        raise HTTPException(status_code=500, detail="encode_failed")
    return Response(content=bytes(buf), media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.post("/mockup/detect-region")
async def detectRegion(mockup_image: UploadFile = File(...)):
    content = await mockup_image.read()
    buf = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid_mockup")

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


@router.post("/mockup/bake-normal")
async def bakeNormal(
    mockup_image: UploadFile = File(...),
    fold_map: UploadFile = File(None),
):
    content = await mockup_image.read()
    buf = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="invalid_mockup")

    f_map = None
    if fold_map:
        f_content = await fold_map.read()
        f_buf = np.frombuffer(f_content, dtype=np.uint8)
        raw = cv2.imdecode(f_buf, cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            f_map = (raw.astype(np.float32) - 128.0) / 128.0

    normal_map = bake_normal_map(img, f_map)
    ok, buf = cv2.imencode(".png", normal_map)
    if not ok:
        raise HTTPException(status_code=500, detail="encode_failed")
    return Response(content=bytes(buf), media_type="image/png")


@router.get("/templates")
async def listTemplates():
    async with async_session() as session:
        stmt = select(Template).where(Template.status == "active")
        result = await session.execute(stmt)
        templates = result.scalars().all()

    return {
        "templates": [
            {
                "id": t.slug,
                "name": t.name,
                "preview_url": f"/static/templates/{t.slug}/mockup.jpg",
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
