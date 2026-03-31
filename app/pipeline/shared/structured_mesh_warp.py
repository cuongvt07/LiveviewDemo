import cv2
import numpy as np

from .gpu_ops import gpuRemap


def _triangle_area(tri: np.ndarray) -> float:
    return float(
        0.5
        * abs(
            (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
            - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0])
        )
    )


def _write_triangle_map(
    map_x: np.ndarray,
    map_y: np.ndarray,
    coverage: np.ndarray,
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
    width: int,
    height: int,
) -> None:
    if _triangle_area(src_tri) <= 1e-4 or _triangle_area(dst_tri) <= 1e-4:
        return

    int_dst = np.rint(dst_tri).astype(np.int32)
    cv2.fillConvexPoly(coverage, int_dst, 255, lineType=cv2.LINE_8)

    min_x = max(0, int(np.floor(np.min(dst_tri[:, 0]))))
    max_x = min(width - 1, int(np.ceil(np.max(dst_tri[:, 0]))))
    min_y = max(0, int(np.floor(np.min(dst_tri[:, 1]))))
    max_y = min(height - 1, int(np.ceil(np.max(dst_tri[:, 1]))))
    if min_x >= max_x or min_y >= max_y:
        return

    cell_w = max_x - min_x + 1
    cell_h = max_y - min_y + 1
    local_mask = np.zeros((cell_h, cell_w), dtype=np.uint8)
    local_poly = np.stack(
        [
            np.clip(dst_tri[:, 0] - min_x, 0, cell_w - 1),
            np.clip(dst_tri[:, 1] - min_y, 0, cell_h - 1),
        ],
        axis=1,
    ).astype(np.int32)
    cv2.fillConvexPoly(local_mask, local_poly, 255, lineType=cv2.LINE_8)
    if not np.any(local_mask):
        return

    affine = cv2.getAffineTransform(dst_tri, src_tri)
    local_ys, local_xs = np.mgrid[min_y : max_y + 1, min_x : max_x + 1].astype(np.float32)
    mapped_x = affine[0, 0] * local_xs + affine[0, 1] * local_ys + affine[0, 2]
    mapped_y = affine[1, 0] * local_xs + affine[1, 1] * local_ys + affine[1, 2]

    cell_mask = local_mask > 0
    map_x_slice = map_x[min_y : max_y + 1, min_x : max_x + 1]
    map_y_slice = map_y[min_y : max_y + 1, min_x : max_x + 1]
    map_x_slice[cell_mask] = mapped_x[cell_mask]
    map_y_slice[cell_mask] = mapped_y[cell_mask]


def _fill_triangle_coverage(coverage: np.ndarray, tri: np.ndarray) -> None:
    if _triangle_area(tri) <= 1e-4:
        return
    cv2.fillConvexPoly(coverage, np.rint(tri).astype(np.int32), 255, lineType=cv2.LINE_8)


def apply_structured_mesh_warp(
    image: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Coarse-to-fine structured mesh warp.

    - `src_pts` and `dst_pts` are expected in output pixel space.
    - Points must form a square lattice: 4x4, 5x5, ...
    - Each cell is split into two triangles and warped by affine maps.
      This keeps shared edges continuous so the preview does not tear
      when neighboring cells deform differently.
    """
    if image is None or src_pts is None or dst_pts is None:
        return image

    src = np.asarray(src_pts, dtype=np.float32)
    dst = np.asarray(dst_pts, dtype=np.float32)
    if src.ndim != 2 or dst.ndim != 2 or src.shape != dst.shape or src.shape[1] != 2:
        return image

    n = src.shape[0]
    side = int(round(np.sqrt(n)))
    if side < 2 or side * side != n:
        return image

    if output_size is None:
        h, w = image.shape[:2]
    else:
        w, h = output_size

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xs.copy()
    map_y = ys.copy()
    src_coverage = np.zeros((h, w), dtype=np.uint8)
    dst_coverage = np.zeros((h, w), dtype=np.uint8)

    for r in range(side - 1):
        for c in range(side - 1):
            i00 = r * side + c
            i01 = r * side + c + 1
            i11 = (r + 1) * side + c + 1
            i10 = (r + 1) * side + c

            src_quad = np.float32([src[i00], src[i01], src[i11], src[i10]])
            dst_quad = np.float32([dst[i00], dst[i01], dst[i11], dst[i10]])

            src_triangles = (
                np.float32([src_quad[0], src_quad[1], src_quad[2]]),
                np.float32([src_quad[0], src_quad[2], src_quad[3]]),
            )
            dst_triangles = (
                np.float32([dst_quad[0], dst_quad[1], dst_quad[2]]),
                np.float32([dst_quad[0], dst_quad[2], dst_quad[3]]),
            )

            for src_tri, dst_tri in zip(src_triangles, dst_triangles):
                _fill_triangle_coverage(src_coverage, src_tri)
                _write_triangle_map(map_x, map_y, dst_coverage, src_tri, dst_tri, w, h)

    ih, iw = image.shape[:2]
    map_x = np.clip(map_x, 0, iw - 1)
    map_y = np.clip(map_y, 0, ih - 1)

    warped = gpuRemap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LANCZOS4,
    )

    stale_mask = (src_coverage > 0) & (dst_coverage == 0)
    if np.any(stale_mask):
        warped = warped.copy()
        warped[stale_mask] = 0

    return warped
