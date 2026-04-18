import cv2
import numpy as np

from .gpu_ops import gpuRemap
from .structured_mesh_warp import apply_structured_mesh_warp
from .liveview_cache import (
    make_tps_map_cache_key,
    get_tps_map_cache,
    set_tps_map_cache,
)


def _normalize_points(
    points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    if pts.ndim != 2 or pts.shape[1] != 2:
        return pts

    if (
        np.nanmax(pts[:, 0]) <= 1.5
        and np.nanmax(pts[:, 1]) <= 1.5
        and np.nanmin(pts[:, 0]) >= -0.5
        and np.nanmin(pts[:, 1]) >= -0.5
    ):
        pts[:, 0] *= max(width - 1, 1)
        pts[:, 1] *= max(height - 1, 1)
    return pts


def apply_smooth_mesh_warp(
    image: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    output_size: tuple[int, int] | None = None,
    interpolation: int = cv2.INTER_CUBIC,
    is_preview: bool = False,
) -> np.ndarray:
    """
    Smooth mesh warp using Thin Plate Spline.

    This is better suited for interactive mesh editing than per-cell warps
    because the displacement field stays continuous across the full surface.
    Falls back to the structured cell warp if TPS is unavailable.
    """
    if image is None or src_pts is None or dst_pts is None:
        return image

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 2:
        return image

    if len(src) < 3:
        return image

    if output_size is None:
        h, w = image.shape[:2]
    else:
        w, h = output_size

    ih, iw = image.shape[:2]

    # PREVIEW OPTIMIZATION (Task 0.1): Reduce resolution for TPS computation
    PREVIEW_MAX_PX = 512
    is_downscaling = is_preview and max(h, w) > PREVIEW_MAX_PX

    import logging
    import time
    logger = logging.getLogger('mockup_service')
    t_start = time.perf_counter()
    logger.info(f"[PERF] smooth_mesh_warp input shape: {image.shape}, is_preview={is_preview}, will_downscale={is_downscaling}")
    
    current_w, current_h = w, h
    current_iw, current_ih = iw, ih
    current_image = image
    scale = 1.0

    if is_downscaling:
        scale = PREVIEW_MAX_PX / max(h, w)
        current_w = int(w * scale)
        current_h = int(h * scale)
        current_iw = int(iw * scale)
        current_ih = int(ih * scale)
        current_image = cv2.resize(image, (current_iw, current_ih), interpolation=cv2.INTER_AREA)

    src_px = _normalize_points(src, iw, ih)
    dst_px = _normalize_points(dst, w, h)

    # Scale points if downscaling
    src_px_scaled = src_px * scale if is_downscaling else src_px
    dst_px_scaled = dst_px * scale if is_downscaling else dst_px

    # --- FAST CACHE LOOKUP ---
    tps_cache_key = make_tps_map_cache_key(
        src_px_scaled,
        dst_px_scaled,
        (current_iw, current_ih),
        (current_w, current_h),
        is_preview=is_preview,
    )
    cached_maps = get_tps_map_cache(tps_cache_key)
    if cached_maps is not None:
        map_x, map_y = cached_maps
        interp_method = cv2.INTER_LINEAR if (is_preview or is_downscaling) else interpolation
        warped_result = gpuRemap(
            current_image,
            map_x,
            map_y,
            interpolation=interp_method,
        )
        if is_downscaling:
            result = cv2.resize(warped_result, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            result = warped_result
        logger.info(f"[PERF] TPS Map: CACHE HIT for shape {map_x.shape}. Total time: {int((time.perf_counter() - t_start) * 1000)}ms")
        return result

    try:
        tps = cv2.createThinPlateSplineShapeTransformer()
    except AttributeError:
        # Fallback doesn't support downscaling yet, keep original behavior
        return apply_structured_mesh_warp(
            image,
            src_pts=src_px,
            dst_pts=dst_px,
            output_size=(w, h),
        )

    try:
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_px_scaled))]
        # Estimate inverse mapping (output -> input) for cv2.remap.
        tps.estimateTransformation(
            dst_px_scaled.reshape(1, -1, 2),
            src_px_scaled.reshape(1, -1, 2),
            matches,
        )

        TPS_GRID_SIZE = 128
        if is_preview:
            tps_w = min(current_w, TPS_GRID_SIZE)
            tps_h = min(current_h, TPS_GRID_SIZE)
            
            ys, xs = np.mgrid[0:tps_h, 0:tps_w].astype(np.float32)
            xs = (xs / max(tps_w - 1, 1)) * (current_w - 1)
            ys = (ys / max(tps_h - 1, 1)) * (current_h - 1)
            
            grid_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(1, -1, 2)
            _, mapped = tps.applyTransformation(grid_pts)
            mapped = mapped.reshape(tps_h, tps_w, 2)
            
            map_x = cv2.resize(mapped[:, :, 0], (current_w, current_h), interpolation=cv2.INTER_CUBIC)
            map_y = cv2.resize(mapped[:, :, 1], (current_w, current_h), interpolation=cv2.INTER_CUBIC)
            
            map_x = np.clip(map_x, 0, current_iw - 1)
            map_y = np.clip(map_y, 0, current_ih - 1)
        else:
            ys, xs = np.mgrid[0:current_h, 0:current_w].astype(np.float32)
            grid_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(1, -1, 2)
            _, mapped = tps.applyTransformation(grid_pts)
            mapped = mapped.reshape(current_h, current_w, 2)

            map_x = np.clip(mapped[:, :, 0], 0, current_iw - 1)
            map_y = np.clip(mapped[:, :, 1], 0, current_ih - 1)
            
            # Cache full-resolution maps
            set_tps_map_cache(tps_cache_key, (map_x, map_y))
        
        interp_method = cv2.INTER_LINEAR if (is_preview or is_downscaling) else interpolation
        warped_result = gpuRemap(
            current_image,
            map_x,
            map_y,
            interpolation=interp_method,
        )

        if is_downscaling:
            result = cv2.resize(warped_result, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            result = warped_result
        logger.info(f"[PERF] smooth_mesh_warp completed in {int((time.perf_counter() - t_start) * 1000)}ms (downscaled: {is_downscaling})")
        return result

    except cv2.error:
        result = apply_structured_mesh_warp(
            image,
            src_pts=src_px,
            dst_pts=dst_px,
            output_size=(w, h),
        )
        logger.info(f"[PERF] smooth_mesh_warp completed in {int((time.perf_counter() - t_start) * 1000)}ms (fallback to structured mesh)")
        return result
