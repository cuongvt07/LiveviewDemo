import cv2
import numpy as np

from .gpu_ops import gpuRemap
from .structured_mesh_warp import apply_structured_mesh_warp


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
    src_px = _normalize_points(src, iw, ih)
    dst_px = _normalize_points(dst, w, h)

    try:
        tps = cv2.createThinPlateSplineShapeTransformer()
    except AttributeError:
        return apply_structured_mesh_warp(
            image,
            src_pts=src_px,
            dst_pts=dst_px,
            output_size=(w, h),
        )

    try:
        matches = [cv2.DMatch(i, i, 0) for i in range(len(src_px))]
        # Estimate inverse mapping (output -> input) for cv2.remap.
        tps.estimateTransformation(
            dst_px.reshape(1, -1, 2),
            src_px.reshape(1, -1, 2),
            matches,
        )

        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        grid_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(1, -1, 2)
        _, mapped = tps.applyTransformation(grid_pts)
        mapped = mapped.reshape(h, w, 2)

        map_x = np.clip(mapped[:, :, 0], 0, iw - 1)
        map_y = np.clip(mapped[:, :, 1], 0, ih - 1)
        return gpuRemap(
            image,
            map_x,
            map_y,
            interpolation=interpolation,
        )
    except cv2.error:
        return apply_structured_mesh_warp(
            image,
            src_pts=src_px,
            dst_pts=dst_px,
            output_size=(w, h),
        )
