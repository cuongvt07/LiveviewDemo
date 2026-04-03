import logging
import time

import cv2
import numpy as np

from .cylinder_math import apply_horizontal_squeeze, compute_uv_cylindrical
from ..shared.gpu_ops import gpuRemap
from ..shared.liveview_cache import (
    get_cylindrical_map_cache,
    make_cylindrical_map_cache_key,
    set_cylindrical_map_cache,
)

DEFAULT_CURVE_CORRECTION_ALPHA = 0.0
DEFAULT_CURVE_SNAP_THRESHOLD_PX = 2.0
DEFAULT_BOUNDARY_DENSITY_POWER = 2.0


def _compute_adaptive_boundary_samples(samples: int, density_power: float = DEFAULT_BOUNDARY_DENSITY_POWER) -> np.ndarray:
    count = max(4, int(samples))
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)
    power = max(1.0, float(density_power))
    if power <= 1.0 + 1e-8:
        return t

    left_mask = t <= 0.5
    out = np.empty_like(t)
    out[left_mask] = 0.5 * np.power(t[left_mask] / 0.5, power)
    out[~left_mask] = 1.0 - 0.5 * np.power((1.0 - t[~left_mask]) / 0.5, power)
    out[0] = 0.0
    out[-1] = 1.0
    return np.maximum.accumulate(np.clip(out, 0.0, 1.0))


def _extract_curve_edge_points(mockup_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = bbox
    h, w = mockup_bgr.shape[:2]
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, 0, w - 1))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    if x1 <= x0 or y1 <= y0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    gray = cv2.cvtColor(mockup_bgr, cv2.COLOR_BGR2GRAY) if mockup_bgr.ndim == 3 else mockup_bgr
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    roi = edges[y0 : y1 + 1, x0 : x1 + 1]

    contours, _ = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contour_sets: list[np.ndarray] = []
    for contour in sorted(contours, key=lambda cnt: len(cnt), reverse=True)[:12]:
        pts = contour.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 4:
            continue
        pts[:, 0] += x0
        pts[:, 1] += y0
        contour_sets.append(pts)

    if contour_sets:
        all_pts = np.concatenate(contour_sets, axis=0)
        return all_pts[:, 0], all_pts[:, 1]

    yy, xx = np.nonzero(roi)
    if xx.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return (xx.astype(np.float32) + x0), (yy.astype(np.float32) + y0)


def _collect_boundary_candidates(
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    base_x: np.ndarray,
    base_y: np.ndarray,
    search_band_px: float,
    x_window_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    sample_x: list[float] = []
    sample_y: list[float] = []

    for x0, y0 in zip(base_x, base_y):
        near_x = np.abs(edge_x - x0) <= x_window_px
        if not np.any(near_x):
            continue
        local_x = edge_x[near_x]
        local_y = edge_y[near_x]
        near_y = np.abs(local_y - y0) <= search_band_px
        if not np.any(near_y):
            continue
        local_x = local_x[near_y]
        local_y = local_y[near_y]
        idx = int(np.argmin(np.abs(local_y - y0)))
        sample_x.append(float(local_x[idx]))
        sample_y.append(float(local_y[idx]))

    if len(sample_x) < 4:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    xs = np.round(np.asarray(sample_x, dtype=np.float32)).astype(np.int32)
    ys = np.asarray(sample_y, dtype=np.float32)
    unique_x, inverse = np.unique(xs, return_inverse=True)
    mean_y = np.zeros_like(unique_x, dtype=np.float32)
    counts = np.zeros_like(unique_x, dtype=np.float32)
    np.add.at(mean_y, inverse, ys)
    np.add.at(counts, inverse, 1.0)
    mean_y /= np.maximum(counts, 1.0)
    return unique_x.astype(np.float32), mean_y


def _fit_boundary_curve(sample_x: np.ndarray, sample_y: np.ndarray, eval_x: np.ndarray, degree: int = 3) -> np.ndarray | None:
    if sample_x.size < 4 or sample_y.size < 4:
        return None
    if np.allclose(sample_x.max(), sample_x.min()):
        return None

    deg = min(int(degree), int(sample_x.size - 1))
    x_min = float(sample_x.min())
    x_span = float(sample_x.max() - sample_x.min())
    x_norm = ((sample_x - x_min) / max(x_span, 1e-6)) * 2.0 - 1.0
    eval_norm = ((eval_x - x_min) / max(x_span, 1e-6)) * 2.0 - 1.0

    coeffs = np.polyfit(x_norm, sample_y, deg=deg)
    fitted = np.polyval(coeffs, eval_norm)
    return fitted.astype(np.float32)


def _blend_and_snap_curve(
    base_y: np.ndarray,
    fitted_y: np.ndarray | None,
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    eval_x: np.ndarray,
    alpha: float,
    snap_threshold_px: float,
    x_window_px: float,
) -> np.ndarray:
    if fitted_y is None:
        return base_y

    corrected = (alpha * fitted_y + (1.0 - alpha) * base_y).astype(np.float32)
    if sample_x.size == 0:
        return corrected

    for idx, (x0, y0) in enumerate(zip(eval_x, corrected)):
        near_x = np.abs(sample_x - x0) <= x_window_px
        if not np.any(near_x):
            continue
        nearby_y = sample_y[near_x]
        nearest = float(nearby_y[np.argmin(np.abs(nearby_y - y0))])
        if abs(nearest - y0) <= snap_threshold_px:
            corrected[idx] = nearest
    return corrected


def _apply_mockup_curve_correction(
    mockup_bgr: np.ndarray,
    dst_corners: np.ndarray,
    px_top: np.ndarray,
    py_top: np.ndarray,
    px_bottom: np.ndarray,
    py_bottom: np.ndarray,
    alpha: float,
    snap_threshold_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    blend = float(np.clip(alpha, 0.0, 1.0))
    if blend <= 1e-8:
        return py_top, py_bottom

    xs = dst_corners[:, 0]
    ys = dst_corners[:, 1]
    margin_x = max(6, int((xs.max() - xs.min()) * 0.08))
    margin_y = max(6, int((ys.max() - ys.min()) * 0.12))
    bbox = (
        int(np.floor(xs.min() - margin_x)),
        int(np.floor(ys.min() - margin_y)),
        int(np.ceil(xs.max() + margin_x)),
        int(np.ceil(ys.max() + margin_y)),
    )

    edge_x, edge_y = _extract_curve_edge_points(mockup_bgr, bbox)
    if edge_x.size == 0:
        return py_top, py_bottom

    bbox_height = max(1.0, float(bbox[3] - bbox[1]))
    bbox_width = max(1.0, float(bbox[2] - bbox[0]))
    search_band_px = float(np.clip(bbox_height * 0.12, 6.0, 28.0))
    x_window_px = float(np.clip(bbox_width / 96.0, 2.0, 6.0))

    top_sample_x, top_sample_y = _collect_boundary_candidates(edge_x, edge_y, px_top, py_top, search_band_px, x_window_px)
    bottom_sample_x, bottom_sample_y = _collect_boundary_candidates(edge_x, edge_y, px_bottom, py_bottom, search_band_px, x_window_px)

    top_fit = _fit_boundary_curve(top_sample_x, top_sample_y, px_top)
    bottom_fit = _fit_boundary_curve(bottom_sample_x, bottom_sample_y, px_bottom)

    corrected_top = _blend_and_snap_curve(
        py_top,
        top_fit,
        top_sample_x,
        top_sample_y,
        px_top,
        blend,
        snap_threshold_px,
        x_window_px,
    )
    corrected_bottom = _blend_and_snap_curve(
        py_bottom,
        bottom_fit,
        bottom_sample_x,
        bottom_sample_y,
        px_bottom,
        blend,
        snap_threshold_px,
        x_window_px,
    )

    corrected_top = np.clip(corrected_top, 0, mockup_bgr.shape[0] - 1)
    corrected_bottom = np.clip(corrected_bottom, 0, mockup_bgr.shape[0] - 1)
    corrected_bottom = np.maximum(corrected_bottom, corrected_top + 1.0)
    return corrected_top.astype(np.float32), corrected_bottom.astype(np.float32)


def _local_warp_point(
    u: np.ndarray,
    v: np.ndarray,
    theta_max_deg: float,
    pitch: float,
    hr_ratio: float,
    smile_base: float,
    curve_top: float | None,
    curve_bottom: float | None,
    edge_squeeze: float,
    squeeze_power: float,
    center_focus_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta_max = np.radians(theta_max_deg)
    if abs(theta_max) < 1e-8:
        return u, v

    nx = (u - 0.5) * 2.0
    sin_theta_max = np.sin(theta_max)
    sin_theta = np.clip(nx * sin_theta_max, -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    t_final = apply_horizontal_squeeze(
        theta,
        theta_max,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
    )
    wx = (t_final + 1.0) / 2.0

    cos_displacement = np.cos(theta) - np.cos(theta_max)
    if curve_top is not None and curve_bottom is not None:
        curve_at_pixel = (curve_top / 100.0) * (1.0 - v) + (curve_bottom / 100.0) * v
        y_proj = (v - 0.5) * 2.0
        sign = -y_proj
        wy_canon = y_proj + sign * curve_at_pixel * hr_ratio * 0.15 * cos_displacement
        wy = (wy_canon + 1.0) / 2.0
    else:
        y_proj = (v - 0.5) * 2.0
        factor = -y_proj / 2.0
        curve_v = smile_base + factor * (pitch / 100.0) * hr_ratio * 0.15
        wy = v - curve_v * cos_displacement

    return wx, wy


def _build_curved_clip_mask(
    mockup_bgr: np.ndarray | None,
    dst_corners: np.ndarray,
    H: int,
    W: int,
    H_mat_output_to_canon: np.ndarray,
    theta_max_deg: float,
    pitch: float,
    hr_ratio: float,
    smile_base: float,
    curve_top: float | None,
    curve_bottom: float | None,
    edge_squeeze: float,
    squeeze_power: float,
    center_focus_width: float,
    curve_correction_alpha: float = DEFAULT_CURVE_CORRECTION_ALPHA,
    curve_snap_threshold_px: float = DEFAULT_CURVE_SNAP_THRESHOLD_PX,
    samples: int = 256,
) -> np.ndarray:
    H_inv = np.linalg.inv(H_mat_output_to_canon)

    u = _compute_adaptive_boundary_samples(samples)
    v_top = np.zeros_like(u)
    v_bot = np.ones_like(u)

    wx_top, wy_top = _local_warp_point(
        u,
        v_top,
        theta_max_deg,
        pitch,
        hr_ratio,
        smile_base,
        curve_top,
        curve_bottom,
        edge_squeeze,
        squeeze_power,
        center_focus_width,
    )
    wx_bot, wy_bot = _local_warp_point(
        u,
        v_bot,
        theta_max_deg,
        pitch,
        hr_ratio,
        smile_base,
        curve_top,
        curve_bottom,
        edge_squeeze,
        squeeze_power,
        center_focus_width,
    )

    x_canon = np.concatenate([wx_top * 2.0 - 1.0, (wx_bot[::-1] * 2.0 - 1.0)]).astype(np.float32)
    y_canon = np.concatenate([wy_top * 2.0 - 1.0, (wy_bot[::-1] * 2.0 - 1.0)]).astype(np.float32)
    ones = np.ones_like(x_canon)
    pts_canon = np.stack([x_canon, y_canon, ones], axis=0)

    proj = H_inv @ pts_canon
    proj /= proj[2:3, :]
    px = np.clip(proj[0], 0, W - 1)
    py = np.clip(proj[1], 0, H - 1)

    split = u.shape[0]
    px_top = px[:split]
    py_top = py[:split]
    px_bottom = px[split:][::-1]
    py_bottom = py[split:][::-1]

    if mockup_bgr is not None and float(curve_correction_alpha) > 1e-8:
        py_top, py_bottom = _apply_mockup_curve_correction(
            mockup_bgr=mockup_bgr,
            dst_corners=dst_corners,
            px_top=px_top,
            py_top=py_top,
            px_bottom=px_bottom,
            py_bottom=py_bottom,
            alpha=curve_correction_alpha,
            snap_threshold_px=curve_snap_threshold_px,
        )

    poly = np.stack(
        [
            np.concatenate([px_top, px_bottom[::-1]]),
            np.concatenate([py_top, py_bottom[::-1]]),
        ],
        axis=1,
    ).astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_AA)
    return mask


def compute_and_cache_cylindrical_map(
    print_area: dict,
    output_size: tuple[int, int],
    design_size: tuple[int, int],
    theta_max_deg: float,
    pitch: float,
    smile_base: float,
    curve_top: float | None,
    curve_bottom: float | None,
    edge_squeeze: float,
    squeeze_power: float,
    center_focus_width: float,
    reference_mockup: np.ndarray | None = None,
    curve_correction_alpha: float = 0.0,
    curve_snap_threshold_px: float = 2.0,
    is_preview: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute map_x/map_y (heavy math) and write to liveview cache.

    Returns (map_x, map_y, H_mat) where H_mat is the homography used to build
    curved mask (may be None if homography couldn't be estimated).
    """
    W, H = int(output_size[0]), int(output_size[1])
    dw, dh = int(design_size[0]), int(design_size[1])

    cache_key = make_cylindrical_map_cache_key(
        print_area=print_area,
        output_size=(W, H),
        design_size=(dw, dh),
        theta_max_deg=theta_max_deg,
        pitch=pitch,
        smile_base=smile_base,
        curve_top=curve_top,
        curve_bottom=curve_bottom,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
        is_preview=is_preview,
    )

    # If another thread already filled the cache, return it
    existing = get_cylindrical_map_cache(cache_key)
    if existing is not None:
        return existing["map_x"], existing["map_y"], None

    src_canonical = np.float32([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ])
    try:
        dst_corners = np.float32([
            print_area["top_left"],
            print_area["top_right"],
            print_area["bottom_right"],
            print_area["bottom_left"],
        ])
    except Exception:
        dst_corners = src_canonical

    H_mat, _ = cv2.findHomography(dst_corners, src_canonical)

    PREVIEW_GRID_SIZE = 512
    if is_preview and (W > PREVIEW_GRID_SIZE or H > PREVIEW_GRID_SIZE):
        sw = PREVIEW_GRID_SIZE if W >= H else int(W * PREVIEW_GRID_SIZE / H)
        sh = PREVIEW_GRID_SIZE if H >= W else int(H * PREVIEW_GRID_SIZE / W)
        
        ys_small, xs_small = np.mgrid[0:sh, 0:sw].astype(np.float32)
        # Map grid [0, sw-1] to [0, W-1]
        xs_full = (xs_small / max(sw - 1, 1)) * (W - 1)
        ys_full = (ys_small / max(sh - 1, 1)) * (H - 1)
        
        pts_small = np.stack([xs_full.ravel(), ys_full.ravel(), np.ones(sh * sw)], axis=0)
        proj_small = H_mat @ pts_small
        proj_small /= proj_small[2:3, :]
        X_proj_small = proj_small[0].reshape(sh, sw)
        Y_proj_small = proj_small[1].reshape(sh, sw)
        
        U_small, V_small = compute_uv_cylindrical(
            X_proj_small,
            Y_proj_small,
            theta_max_deg=theta_max_deg,
            pitch=pitch,
            hr_ratio=max(1e-6, float(abs(print_area.get("bottom_left", [0, 0])[1] - print_area.get("top_left", [0, 0])[1]) / (abs(print_area.get("top_right", [0, 0])[0] - print_area.get("top_left", [0, 0])[0]) + 1e-6))),
            smile_base=smile_base,
            curve_top=curve_top,
            curve_bottom=curve_bottom,
            edge_squeeze=edge_squeeze,
            squeeze_power=squeeze_power,
            center_focus_width=center_focus_width,
            clamp_v=False,
        )
        
        # Upscale U, V back to (W, H)
        U = cv2.resize(U_small, (W, H), interpolation=cv2.INTER_LINEAR)
        V = cv2.resize(V_small, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # We also need X_proj, Y_proj for the "outside" check later
        X_proj = cv2.resize(X_proj_small, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        pts = np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)], axis=0)
        proj = H_mat @ pts
        proj /= proj[2:3, :]
        X_proj = proj[0].reshape(H, W)
        Y_proj = proj[1].reshape(H, W)

        U, V = compute_uv_cylindrical(
            X_proj,
            Y_proj,
            theta_max_deg=theta_max_deg,
            pitch=pitch,
            hr_ratio=max(1e-6, float(abs(print_area.get("bottom_left", [0, 0])[1] - print_area.get("top_left", [0, 0])[1]) / (abs(print_area.get("top_right", [0, 0])[0] - print_area.get("top_left", [0, 0])[0]) + 1e-6))),
            smile_base=smile_base,
            curve_top=curve_top,
            curve_bottom=curve_bottom,
            edge_squeeze=edge_squeeze,
            squeeze_power=squeeze_power,
            center_focus_width=center_focus_width,
            clamp_v=False,
        )

    map_x = (np.clip(U, 0.0, 1.0) * (dw - 1)).astype(np.float32)
    map_y = (np.clip(V, 0.0, 1.0) * (dh - 1)).astype(np.float32)

    outside = (
        (X_proj < -1.0)
        | (X_proj > 1.0)
        | (U < 0.0)
        | (U > 1.0)
        | (V < 0.0)
        | (V > 1.0)
    )
    map_x[outside] = -1
    map_y[outside] = -1

    curved_mask = None
    if reference_mockup is not None and float(curve_correction_alpha) > 1e-8:
        try:
            curved_mask = _build_curved_clip_mask(
                mockup_bgr=reference_mockup,
                dst_corners=dst_corners,
                H=H,
                W=W,
                H_mat_output_to_canon=H_mat,
                theta_max_deg=theta_max_deg,
                pitch=pitch,
                hr_ratio=max(1e-6, float(abs(print_area.get("bottom_left", [0, 0])[1] - print_area.get("top_left", [0, 0])[1]) / (abs(print_area.get("top_right", [0, 0])[0] - print_area.get("top_left", [0, 0])[0]) + 1e-6))),
                smile_base=smile_base,
                curve_top=curve_top,
                curve_bottom=curve_bottom,
                edge_squeeze=edge_squeeze,
                squeeze_power=squeeze_power,
                center_focus_width=center_focus_width,
                curve_correction_alpha=curve_correction_alpha,
                curve_snap_threshold_px=curve_snap_threshold_px,
            )
        except Exception:
            curved_mask = None

    # Store into LRU cache
    set_cylindrical_map_cache(
        cache_key,
        {
            "map_x": map_x,
            "map_y": map_y,
            "curved_mask": curved_mask,
        },
    )

    return map_x, map_y, H_mat


def cylindrical_warp(
    design: np.ndarray,
    print_area: dict,
    output_size: tuple[int, int],
    reference_mockup: np.ndarray | None = None,
    theta_max_deg: float = 52.0,
    pitch: float = 0.0,
    smile_base: float = 0.08,
    curve_top: float | None = None,
    curve_bottom: float | None = None,
    edge_squeeze: float = 0.0,
    squeeze_power: float = 2.0,
    center_focus_width: float = 0.0,
    curve_correction_alpha: float = DEFAULT_CURVE_CORRECTION_ALPHA,
    curve_snap_threshold_px: float = DEFAULT_CURVE_SNAP_THRESHOLD_PX,
    is_preview: bool = False,
) -> np.ndarray:
    """
    Warp a flat design to mug cylindrical space.
    """
    W, H = output_size
    dh, dw = design.shape[:2]

    pa = print_area
    dst_corners = np.float32([
        pa["top_left"],
        pa["top_right"],
        pa["bottom_right"],
        pa["bottom_left"],
    ])

    bh_px = abs(pa["bottom_left"][1] - pa["top_left"][1])
    bw_px = abs(pa["top_right"][0] - pa["top_left"][0])
    hr_ratio = bh_px / (bw_px + 1e-6)

    cache_key = make_cylindrical_map_cache_key(
        print_area=pa,
        output_size=(W, H),
        design_size=(dw, dh),
        theta_max_deg=theta_max_deg,
        pitch=pitch,
        smile_base=smile_base,
        curve_top=curve_top,
        curve_bottom=curve_bottom,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
        is_preview=is_preview,
    )
    _cyl_logger = logging.getLogger('mockup_service')
    t_cyl = time.perf_counter()
    cached_bundle = get_cylindrical_map_cache(cache_key)
    map_x: np.ndarray
    map_y: np.ndarray
    H_mat: np.ndarray | None = None

    if cached_bundle is not None:
        map_x = cached_bundle["map_x"]
        map_y = cached_bundle["map_y"]
        _cyl_logger.info('[PERF]     warp map: CACHE HIT (%dms)', int((time.perf_counter() - t_cyl) * 1000))
    else:
        map_x, map_y, H_mat = compute_and_cache_cylindrical_map(
            print_area=pa,
            output_size=(W, H),
            design_size=(dw, dh),
            theta_max_deg=theta_max_deg,
            pitch=pitch,
            smile_base=smile_base,
            curve_top=curve_top,
            curve_bottom=curve_bottom,
            edge_squeeze=edge_squeeze,
            squeeze_power=squeeze_power,
            center_focus_width=center_focus_width,
            reference_mockup=reference_mockup,
            curve_correction_alpha=curve_correction_alpha,
            curve_snap_threshold_px=curve_snap_threshold_px,
            is_preview=is_preview,
        )
        _cyl_logger.info('[PERF]     warp map: CACHE MISS - computed (%dms)', int((time.perf_counter() - t_cyl) * 1000))
        # refresh cached bundle reference for curved_mask lookup
        cached_bundle = get_cylindrical_map_cache(cache_key)

    if is_preview:
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_LANCZOS4
        if W <= 1000:
            interpolation = cv2.INTER_LINEAR
        elif W <= 2000:
            interpolation = cv2.INTER_CUBIC

    t_remap = time.perf_counter()
    warped = gpuRemap(
        design,
        map_x,
        map_y,
        interpolation=interpolation,
        map_cache_key=str(cache_key),
    )
    _cyl_logger.info('[PERF]     gpuRemap: %dms (interp=%s)', int((time.perf_counter() - t_remap) * 1000), 'LANCZOS4' if interpolation == cv2.INTER_LANCZOS4 else ('CUBIC' if interpolation == cv2.INTER_CUBIC else 'LINEAR'))

    # Final clip by curved boundary so rendered print matches locked editor grid shape.
    if warped.ndim == 3 and warped.shape[2] == 4:
        can_reuse_cached_mask = reference_mockup is None or float(curve_correction_alpha) <= 1e-8
        curved_mask = cached_bundle.get("curved_mask") if (cached_bundle and can_reuse_cached_mask) else None
        if curved_mask is None:
            if H_mat is None:
                src_canonical = np.float32([
                    [-1, -1],
                    [1, -1],
                    [1, 1],
                    [-1, 1],
                ])
                H_mat, _ = cv2.findHomography(dst_corners, src_canonical)

            curved_mask = _build_curved_clip_mask(
                mockup_bgr=reference_mockup,
                dst_corners=dst_corners,
                H=H,
                W=W,
                H_mat_output_to_canon=H_mat,
                theta_max_deg=theta_max_deg,
                pitch=pitch,
                hr_ratio=hr_ratio,
                smile_base=smile_base,
                curve_top=curve_top,
                curve_bottom=curve_bottom,
                edge_squeeze=edge_squeeze,
                squeeze_power=squeeze_power,
                center_focus_width=center_focus_width,
                curve_correction_alpha=curve_correction_alpha,
                curve_snap_threshold_px=curve_snap_threshold_px,
            )
            if can_reuse_cached_mask:
                cached_bundle = set_cylindrical_map_cache(
                    cache_key,
                    {
                        "map_x": map_x,
                        "map_y": map_y,
                        "curved_mask": curved_mask,
                    },
                )
        warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], curved_mask)

    return warped
