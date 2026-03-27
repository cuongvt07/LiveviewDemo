import cv2
import numpy as np
from .harris_control_points import extract_wrinkle_control_points
from ..shared.gpu_ops import gpuRemap

def tps_warp_design(
    design: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    output_size: tuple[int, int]
) -> np.ndarray:
    """
    TPS warp design theo nếp gấp vải.
    Chỉ dùng cho Clothes — không dùng cho Mugs.

    TPS (Thin Plate Spline) fit smooth surface qua tất cả control points.
    Kết quả: design biến dạng chính xác theo từng nếp gấp của vải.
    """
    W, H = output_size
    dh, dw = design.shape[:2]
    if len(src_pts) < 3:
        return design

    src_px = src_pts.astype(np.float32).copy()
    # Backward compatible:
    # - normalized [0..1] source points => convert to design pixels
    # - already pixel space => keep as-is
    if (
        np.nanmax(src_px[:, 0]) <= 1.5
        and np.nanmax(src_px[:, 1]) <= 1.5
        and np.nanmin(src_px[:, 0]) >= -0.5
        and np.nanmin(src_px[:, 1]) >= -0.5
    ):
        src_px[:, 0] *= (dw - 1)
        src_px[:, 1] *= (dh - 1)

    tps = cv2.createThinPlateSplineShapeTransformer()
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_px))]
    # We need inverse mapping for cv2.remap (output -> input), so estimate dst -> src.
    tps.estimateTransformation(
        dst_pts.reshape(1, -1, 2),
        src_px.reshape(1, -1, 2),
        matches,
    )

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    grid_pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(1, -1, 2)
    _, mapped = tps.applyTransformation(grid_pts)
    mapped = mapped.reshape(H, W, 2)

    map_x = np.clip(mapped[:, :, 0], 0, dw - 1)
    map_y = np.clip(mapped[:, :, 1], 0, dh - 1)

    return gpuRemap(
        design, map_x, map_y,
        interpolation=cv2.INTER_LANCZOS4,
    )
