import cv2
import numpy as np

from .cylinder_math import apply_horizontal_squeeze


def _normalize_kernel_size(kernel_size: int) -> int:
    k = max(1, int(kernel_size))
    return k if k % 2 == 1 else k + 1


def _to_mask_float(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask is None:
        return np.ones((h, w), dtype=np.float32)

    if mask.ndim == 3 and mask.shape[2] == 4:
        mask_arr = mask[:, :, 3]
    elif mask.ndim == 3:
        mask_arr = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_arr = mask

    if mask_arr.shape[:2] != (h, w):
        mask_arr = cv2.resize(mask_arr, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_float = mask_arr.astype(np.float32)
    if mask_float.max() > 1.0:
        mask_float /= 255.0
    return np.clip(mask_float, 0.0, 1.0)


def _mask_aware_blur(values: np.ndarray, mask_float: np.ndarray, blur_kernel: int) -> np.ndarray:
    k = _normalize_kernel_size(blur_kernel)
    weighted = values * mask_float
    smooth_num = cv2.GaussianBlur(weighted, (k, k), 0)
    smooth_den = cv2.GaussianBlur(mask_float, (k, k), 0)
    return np.divide(
        smooth_num,
        np.maximum(smooth_den, 1e-6),
        out=np.zeros_like(smooth_num),
        where=smooth_den > 1e-6,
    )


def normalize_masked_field(
    values: np.ndarray,
    mask: np.ndarray | None,
    *,
    empty_fill: float = 1.0,
    outside_fill: float = 0.0,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    mask_float = _to_mask_float(mask, values.shape[:2])
    active = mask_float > 1e-3
    if not np.any(active):
        return np.full(values.shape[:2], empty_fill, dtype=np.float32)

    sample = values[active].astype(np.float32)
    lo = float(np.percentile(sample, low_percentile))
    hi = float(np.percentile(sample, high_percentile))
    if hi - lo <= 1e-6:
        normalized = np.ones_like(values, dtype=np.float32)
    else:
        normalized = (values.astype(np.float32) - lo) / (hi - lo)

    return np.where(active, np.clip(normalized, 0.0, 1.0), outside_fill).astype(np.float32)


def build_light_direction(
    light_pos_x: float = 0.62,
    light_pos_y: float = 0.32,
    light_height: float = 55.0,
) -> np.ndarray:
    """
    Build a normalized directional light vector from UI-friendly controls.

    `light_height` is expected in [0..100] and is internally remapped to a
    non-zero z-range so the light does not collapse into a flat side-light.
    """
    pos_x = float(np.clip(light_pos_x, -1.0, 2.0))
    pos_y = float(np.clip(light_pos_y, -1.0, 2.0))
    height = float(np.clip(light_height, 0.0, 100.0))

    lx = (pos_x - 0.5) * 2.0
    # Screen Y grows downward; invert so dragging upward means light from above.
    ly = (0.5 - pos_y) * 2.0
    lz = 0.1 + (height / 100.0) * 2.0

    light_dir = np.array([lx, ly, lz], dtype=np.float32)
    norm = float(np.linalg.norm(light_dir))
    if norm <= 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return light_dir / norm


def extract_masked_lighting(
    mockup_bgr: np.ndarray,
    mask: np.ndarray | None,
    blur_kernel: int = 21,
) -> np.ndarray:
    """
    Extract a normalized lighting field from the mockup using LAB L channel.

    The blur is mask-aware so dark background pixels do not bleed into the mug
    region near the print boundary.
    """
    h, w = mockup_bgr.shape[:2]
    mask_float = _to_mask_float(mask, (h, w))
    active = mask_float > 1e-3
    if not np.any(active):
        return np.ones((h, w), dtype=np.float32)

    lab = cv2.cvtColor(mockup_bgr, cv2.COLOR_BGR2LAB)
    lighting = lab[:, :, 0].astype(np.float32) / 255.0

    smooth = _mask_aware_blur(lighting, mask_float, blur_kernel)

    values = smooth[active]
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1e-6:
        normalized = np.ones_like(smooth, dtype=np.float32)
    else:
        normalized = (smooth - lo) / (hi - lo)

    return np.where(active, np.clip(normalized, 0.0, 1.0), 0.0).astype(np.float32)


def derive_highlight_map(
    lighting: np.ndarray,
    threshold: float = 0.7,
    blur_kernel: int = 9,
) -> np.ndarray:
    threshold_f = float(threshold)
    if threshold_f > 1.0:
        threshold_f /= 255.0
    threshold_f = float(np.clip(threshold_f, 0.0, 0.99))

    denom = max(1e-6, 1.0 - threshold_f)
    highlight = np.clip((lighting.astype(np.float32) - threshold_f) / denom, 0.0, 1.0)

    k = _normalize_kernel_size(blur_kernel)
    if k > 1:
        highlight = cv2.GaussianBlur(highlight, (k, k), 0)
    return np.clip(highlight, 0.0, 1.0).astype(np.float32)


def extract_masked_highlight_detail(
    mockup_bgr: np.ndarray,
    mask: np.ndarray | None,
    diffuse_blur_kernel: int = 41,
    detail_blur_kernel: int = 9,
) -> np.ndarray:
    h, w = mockup_bgr.shape[:2]
    mask_float = _to_mask_float(mask, (h, w))
    active = mask_float > 1e-3
    if not np.any(active):
        return np.zeros((h, w), dtype=np.float32)

    lab = cv2.cvtColor(mockup_bgr, cv2.COLOR_BGR2LAB)
    lighting = lab[:, :, 0].astype(np.float32) / 255.0

    diffuse = _mask_aware_blur(lighting, mask_float, diffuse_blur_kernel)
    detail = np.clip(lighting - diffuse, 0.0, 1.0) * mask_float

    scale = float(np.percentile(detail[active], 95))
    if scale <= 1e-6:
        normalized = np.zeros_like(detail, dtype=np.float32)
    else:
        normalized = np.clip(detail / scale, 0.0, 1.0)

    k = _normalize_kernel_size(detail_blur_kernel)
    if k > 1:
        normalized = cv2.GaussianBlur(normalized, (k, k), 0)
    return np.where(active, np.clip(normalized, 0.0, 1.0), 0.0).astype(np.float32)


def _project_print_area_to_canonical(
    print_area: dict,
    output_size: tuple[int, int],
    mask: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    out_w, out_h = output_size
    mask_float = _to_mask_float(mask, (out_h, out_w))

    dst_corners = np.float32(
        [
            print_area["top_left"],
            print_area["top_right"],
            print_area["bottom_right"],
            print_area["bottom_left"],
        ]
    )
    src_canonical = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    homography, _ = cv2.findHomography(dst_corners, src_canonical)
    if homography is None:
        valid = np.zeros((out_h, out_w), dtype=bool)
        return None, None, valid, mask_float

    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(out_h * out_w, dtype=np.float32)], axis=0)
    proj = homography @ pts
    proj /= np.maximum(proj[2:3, :], 1e-8)

    x_proj = proj[0].reshape(out_h, out_w)
    y_proj = proj[1].reshape(out_h, out_w)
    valid = (
        (x_proj >= -2.0)
        & (x_proj <= 2.0)
        & (y_proj >= -2.0)
        & (y_proj <= 2.0)
        & (mask_float > 1e-3)
    )
    return x_proj, y_proj, valid, mask_float


def build_cylinder_surface_maps(
    print_area: dict,
    output_size: tuple[int, int],
    mask: np.ndarray | None,
    theta_max_deg: float = 52.0,
    edge_squeeze: float = 0.0,
    squeeze_power: float = 2.0,
    center_focus_width: float = 0.0,
) -> dict[str, np.ndarray]:
    out_w, out_h = output_size
    x_proj, y_proj, valid, mask_float = _project_print_area_to_canonical(print_area, output_size, mask)
    if x_proj is None or y_proj is None:
        normals = np.zeros((out_h, out_w, 3), dtype=np.float32)
        normals[:, :, 2] = 1.0
        return {
            "normals": normals,
            "surface_u": np.full((out_h, out_w), 0.5, dtype=np.float32),
            "surface_v": np.full((out_h, out_w), 0.5, dtype=np.float32),
            "valid_mask": valid,
            "mask_float": mask_float,
        }

    theta_max = np.radians(theta_max_deg)
    normals = np.zeros((out_h, out_w, 3), dtype=np.float32)
    normals[:, :, 2] = 1.0
    surface_u = np.full((out_h, out_w), 0.5, dtype=np.float32)
    surface_v = np.clip((y_proj + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    if abs(theta_max) < 1e-8:
        return {
            "normals": normals,
            "surface_u": surface_u,
            "surface_v": surface_v,
            "valid_mask": valid,
            "mask_float": mask_float,
        }

    sin_theta = np.clip(x_proj * np.sin(theta_max), -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    t_final = apply_horizontal_squeeze(
        theta,
        theta_max,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
    )
    theta_mapped = t_final * theta_max

    normals[:, :, 0] = np.sin(theta_mapped).astype(np.float32)
    normals[:, :, 1] = 0.0
    normals[:, :, 2] = np.clip(np.cos(theta_mapped), 0.0, 1.0).astype(np.float32)
    normals[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    surface_u = np.clip((t_final + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    return {
        "normals": normals,
        "surface_u": surface_u,
        "surface_v": surface_v,
        "valid_mask": valid,
        "mask_float": mask_float,
    }


def compute_directional_diffuse(
    normal_map: np.ndarray,
    light_dir: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    light = np.asarray(light_dir, dtype=np.float32)
    light_norm = float(np.linalg.norm(light))
    if light_norm <= 1e-6:
        light = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        light = light / light_norm

    diffuse = np.clip(np.sum(normal_map.astype(np.float32) * light.reshape(1, 1, 3), axis=-1), 0.0, 1.0)
    if valid_mask is None:
        return diffuse.astype(np.float32)
    return np.where(valid_mask, diffuse, 0.0).astype(np.float32)


def build_light_field(
    surface_u: np.ndarray,
    surface_v: np.ndarray,
    valid_mask: np.ndarray | None,
    light_pos_x: float = 0.62,
    light_pos_y: float = 0.32,
    softness: float = 55.0,
    contrast: float = 50.0,
    theta_max_deg: float = 52.0,
    edge_squeeze: float = 0.0,
    squeeze_power: float = 2.0,
    center_focus_width: float = 0.0,
) -> np.ndarray:
    """
    Build a Gaussian light field on the cylinder surface.

    light_pos_x/y are in design UV space [0..1].  They are converted
    to warped surface_u/v coordinates so the Gaussian peak lines up
    with the actual surface mapping used by the warp pipeline.
    """
    raw_x = float(np.clip(light_pos_x, -1.0, 2.0))
    pos_y = float(np.clip(light_pos_y, -1.0, 2.0))
    softness_t = float(np.clip(softness, 0.0, 100.0)) / 100.0
    contrast_t = float(np.clip(contrast, 0.0, 100.0)) / 100.0

    # Convert design UV X → surface_u (warped coordinate)
    theta_max = np.radians(float(np.clip(theta_max_deg, 0.0, 180.0)))
    if abs(theta_max) > 1e-8:
        nx = (raw_x - 0.5) * 2.0
        sin_theta = float(np.clip(nx * np.sin(theta_max), -1.0, 1.0))
        theta_scalar = float(np.arcsin(sin_theta))
        t_arr = np.array([theta_scalar], dtype=np.float32)
        t_final = apply_horizontal_squeeze(
            t_arr,
            theta_max,
            edge_squeeze=edge_squeeze,
            squeeze_power=squeeze_power,
            center_focus_width=center_focus_width,
        )
        pos_x = float(np.clip((t_final[0] + 1.0) * 0.5, -1.0, 2.0))
    else:
        pos_x = raw_x

    sigma_x = 0.08 + softness_t * 0.28
    sigma_y = 0.10 + softness_t * 0.34
    dx = (surface_u.astype(np.float32) - pos_x) / max(sigma_x, 1e-3)
    dy = (surface_v.astype(np.float32) - pos_y) / max(sigma_y, 1e-3)
    raw_field = np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)
    field = normalize_masked_field(
        raw_field,
        valid_mask.astype(np.uint8) if isinstance(valid_mask, np.ndarray) else valid_mask,
        empty_fill=1.0,
        outside_fill=0.0,
        low_percentile=0.0,
        high_percentile=99.0,
    )

    contrast_centered = (contrast_t - 0.5) * 2.0
    gamma = float(np.clip(1.0 - contrast_centered * 0.6, 0.4, 1.6))
    ambient = float(np.clip(0.42 - contrast_centered * 0.08, 0.28, 0.55))
    field = ambient + (1.0 - ambient) * np.power(np.clip(field, 0.0, 1.0), gamma)

    if valid_mask is None:
        return np.clip(field, 0.0, 1.0).astype(np.float32)
    return np.where(valid_mask, np.clip(field, 0.0, 1.0), 0.0).astype(np.float32)


def compute_directional_specular(
    normal_map: np.ndarray,
    light_dir: np.ndarray,
    valid_mask: np.ndarray | None = None,
    shininess: float = 32.0,
    blur_kernel: int = 0,
) -> np.ndarray:
    light = np.asarray(light_dir, dtype=np.float32)
    light_norm = float(np.linalg.norm(light))
    if light_norm <= 1e-6:
        light = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        light = light / light_norm

    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normals = normal_map.astype(np.float32)
    ndotl = np.clip(np.sum(normals * light.reshape(1, 1, 3), axis=-1, keepdims=True), 0.0, 1.0)
    reflect = 2.0 * ndotl * normals - light.reshape(1, 1, 3)
    reflect_norm = np.linalg.norm(reflect, axis=-1, keepdims=True)
    reflect = np.divide(reflect, np.maximum(reflect_norm, 1e-6), out=np.zeros_like(reflect), where=reflect_norm > 1e-6)
    specular = np.clip(np.sum(reflect * view_dir.reshape(1, 1, 3), axis=-1), 0.0, 1.0)
    specular = np.power(specular, max(float(shininess), 1.0)).astype(np.float32)

    k = _normalize_kernel_size(blur_kernel)
    if k > 1:
        specular = cv2.GaussianBlur(specular, (k, k), 0)

    if valid_mask is None:
        return np.clip(specular, 0.0, 1.0).astype(np.float32)
    return np.where(valid_mask, np.clip(specular, 0.0, 1.0), 0.0).astype(np.float32)


def build_cylinder_geometry_maps(
    print_area: dict,
    output_size: tuple[int, int],
    mask: np.ndarray | None,
    theta_max_deg: float = 52.0,
    edge_squeeze: float = 0.0,
    squeeze_power: float = 2.0,
    center_focus_width: float = 0.0,
    cylinder_strength: float = 0.28,
    edge_darkening_strength: float = 0.12,
    specular_position: float = 0.18,
    specular_sigma: float = 0.12,
    specular_blur_kernel: int = 11,
) -> tuple[np.ndarray, np.ndarray]:
    out_w, out_h = output_size
    mask_float = _to_mask_float(mask, (out_h, out_w))

    dst_corners = np.float32(
        [
            print_area["top_left"],
            print_area["top_right"],
            print_area["bottom_right"],
            print_area["bottom_left"],
        ]
    )
    src_canonical = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    homography, _ = cv2.findHomography(dst_corners, src_canonical)
    if homography is None:
        return np.ones((out_h, out_w), dtype=np.float32), np.zeros((out_h, out_w), dtype=np.float32)

    ys, xs = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(out_h * out_w, dtype=np.float32)], axis=0)
    proj = homography @ pts
    proj /= np.maximum(proj[2:3, :], 1e-8)

    x_proj = proj[0].reshape(out_h, out_w)
    valid = (x_proj >= -1.0) & (x_proj <= 1.0) & (mask_float > 1e-3)

    theta_max = np.radians(theta_max_deg)
    if abs(theta_max) < 1e-8:
        return np.ones((out_h, out_w), dtype=np.float32), np.zeros((out_h, out_w), dtype=np.float32)

    sin_theta = np.clip(x_proj * np.sin(theta_max), -1.0, 1.0)
    theta = np.arcsin(sin_theta)
    t_final = apply_horizontal_squeeze(
        theta,
        theta_max,
        edge_squeeze=edge_squeeze,
        squeeze_power=squeeze_power,
        center_focus_width=center_focus_width,
    )

    cos_profile = np.clip(np.cos(theta), 0.0, 1.0)
    base_shading = 1.0 - float(np.clip(cylinder_strength, 0.0, 1.0)) * (1.0 - cos_profile)
    edge_profile = np.clip(np.abs(t_final), 0.0, 1.0)
    edge_factor = 1.0 - float(np.clip(edge_darkening_strength, 0.0, 1.0)) * np.power(edge_profile, 2.0)
    cylinder_map = np.where(valid, np.clip(base_shading * edge_factor, 0.0, 1.0), 1.0).astype(np.float32)

    sigma = max(1e-3, float(specular_sigma))
    line = np.exp(-0.5 * np.square((t_final - float(specular_position)) / sigma)).astype(np.float32)
    k = _normalize_kernel_size(specular_blur_kernel)
    if k > 1:
        line = cv2.GaussianBlur(line, (k, k), 0)
    line = np.where(valid, np.clip(line, 0.0, 1.0), 0.0).astype(np.float32)

    return cylinder_map, line
