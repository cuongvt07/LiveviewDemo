import cv2
import numpy as np


def apply_design_transform(
    design: np.ndarray,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    """
    Apply editor design transform before warp.
    - scale: zoom factor around image center
    - offset_x/offset_y: normalized shift relative to design width/height
    """
    scale = float(scale)
    offset_x = float(offset_x)
    offset_y = float(offset_y)

    if (
        abs(scale - 1.0) < 1e-6
        and abs(offset_x) < 1e-6
        and abs(offset_y) < 1e-6
    ):
        return design

    h, w = design.shape[:2]
    tx = (w * (1.0 - scale) * 0.5) + (offset_x * w)
    ty = (h * (1.0 - scale) * 0.5) + (offset_y * h)
    mat = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)

    return cv2.warpAffine(
        design,
        mat,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
