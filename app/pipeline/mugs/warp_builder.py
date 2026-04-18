import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from .mesh_config import MugMeshConfig
from .cosine_grid import cosine_spacing


def build_warp_map(config: MugMeshConfig, out_w: int, out_h: int, quality: str = "full") -> tuple[np.ndarray, np.ndarray]:
    pts = config.to_numpy()
    u_knots = np.array(cosine_spacing(config.cols))
    v_knots = np.array(cosine_spacing(config.rows))

    if quality == "preview":
        return _build_preview_smooth(pts, u_knots, v_knots, out_w, out_h)
    return _build_cubic_with_fairing(pts, u_knots, v_knots, out_w, out_h, config)


def _build_preview_smooth(pts, u_knots, v_knots, out_w, out_h):
    spline_x = RectBivariateSpline(v_knots, u_knots, pts[:, :, 0], kx=3, ky=3)
    spline_y = RectBivariateSpline(v_knots, u_knots, pts[:, :, 1], kx=3, ky=3)

    out_u = np.linspace(0.0, 1.0, out_w)
    out_v = np.linspace(0.0, 1.0, out_h)

    map_x = spline_x(out_v, out_u).astype(np.float32) * out_w
    map_y = spline_y(out_v, out_u).astype(np.float32) * out_h
    return map_x, map_y


def _build_cubic_with_fairing(pts, u_knots, v_knots, out_w, out_h, config: MugMeshConfig):
    spline_x = RectBivariateSpline(v_knots, u_knots, pts[:, :, 0], kx=3, ky=3)
    spline_y = RectBivariateSpline(v_knots, u_knots, pts[:, :, 1], kx=3, ky=3)

    out_u = np.linspace(0.0, 1.0, out_w)
    out_v = np.linspace(0.0, 1.0, out_h)

    map_x = spline_x(out_v, out_u).astype(np.float32) * out_w
    map_y = spline_y(out_v, out_u).astype(np.float32) * out_h

    map_x, map_y = apply_corner_fairing(map_x, map_y, config.corner_blend_radius)
    return map_x, map_y


def apply_corner_fairing(map_x: np.ndarray, map_y: np.ndarray, corner_radius: float = 0.08) -> tuple[np.ndarray, np.ndarray]:
    h, w = map_x.shape
    r = max(4, int(corner_radius * min(w, h)))
    sigma = max(0.5, r * 0.4)

    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for (cy, cx) in corners:
        y0 = max(0, cy - r); y1 = min(h, cy + r + 1)
        x0 = max(0, cx - r); x1 = min(w, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            continue

        yr = np.arange(y0, y1) - cy
        xr = np.arange(x0, x1) - cx
        xx, yy = np.meshgrid(xr, yr)
        dist = np.sqrt(xx ** 2 + yy ** 2)
        weight = np.exp(-dist ** 2 / (2.0 * sigma ** 2))

        blurred_x = gaussian_filter(map_x[y0:y1, x0:x1], sigma=sigma * 0.5)
        blurred_y = gaussian_filter(map_y[y0:y1, x0:x1], sigma=sigma * 0.5)

        map_x[y0:y1, x0:x1] = weight * blurred_x + (1.0 - weight) * map_x[y0:y1, x0:x1]
        map_y[y0:y1, x0:x1] = weight * blurred_y + (1.0 - weight) * map_y[y0:y1, x0:x1]

    return map_x, map_y
