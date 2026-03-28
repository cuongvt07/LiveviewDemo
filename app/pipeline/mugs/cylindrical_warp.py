import cv2
import numpy as np

from .cylinder_math import apply_horizontal_squeeze, compute_uv_cylindrical
from ..shared.gpu_ops import gpuRemap


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
    samples: int = 256,
) -> np.ndarray:
    H_inv = np.linalg.inv(H_mat_output_to_canon)

    u = np.linspace(0.0, 1.0, samples, dtype=np.float32)
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
    poly = np.stack([px, py], axis=1).astype(np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_AA)
    return mask


def cylindrical_warp(
    design: np.ndarray,
    print_area: dict,
    output_size: tuple[int, int],
    theta_max_deg: float = 52.0,
    pitch: float = 0.0,
    smile_base: float = 0.08,
    curve_top: float | None = None,
    curve_bottom: float | None = None,
    edge_squeeze: float = 0.0,
    squeeze_power: float = 2.0,
    center_focus_width: float = 0.0,
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
    src_canonical = np.float32([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ])
    H_mat, _ = cv2.findHomography(dst_corners, src_canonical)

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(H * W)], axis=0)
    proj = H_mat @ pts
    proj /= proj[2:3, :]
    X_proj = proj[0].reshape(H, W)
    Y_proj = proj[1].reshape(H, W)

    bh_px = abs(pa["bottom_left"][1] - pa["top_left"][1])
    bw_px = abs(pa["top_right"][0] - pa["top_left"][0])
    hr_ratio = bh_px / (bw_px + 1e-6)

    U, V = compute_uv_cylindrical(
        X_proj,
        Y_proj,
        theta_max_deg=theta_max_deg,
        pitch=pitch,
        hr_ratio=hr_ratio,
        smile_base=smile_base,
        curve_top=curve_top,
        curve_bottom=curve_bottom,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
        clamp_v=False,
    )

    # Keep sampling in range but invalidate out-of-domain pixels later.
    map_x = (np.clip(U, 0.0, 1.0) * (dw - 1)).astype(np.float32)
    map_y = (np.clip(V, 0.0, 1.0) * (dh - 1)).astype(np.float32)

    # Use curved domain validity so output boundary follows locked grid shape.
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

    warped = gpuRemap(
        design,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
    )

    # Final clip by curved boundary so rendered print matches locked editor grid shape.
    if warped.ndim == 3 and warped.shape[2] == 4:
        curved_mask = _build_curved_clip_mask(
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
        )
        warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], curved_mask)

    return warped
