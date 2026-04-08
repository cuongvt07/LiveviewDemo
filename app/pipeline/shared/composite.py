import cv2
import numpy as np
from .edge import feather_mask

def composite(
    mockup: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray,
    feather_px: int = 6,
) -> np.ndarray:
    """
    Alpha composite warped design lên mockup.
    Dùng chung — chỉ khác feather_px:
      - Mug:     feather_px=6  (viền cứng hơn — men sứ)
      - Clothes: feather_px=10 (viền mềm hơn — vải)
    """
    h, w = mockup.shape[:2]
    warped_r = cv2.resize(warped, (w, h)) if warped.shape[:2] != (h, w) else warped
    mask_r   = cv2.resize(mask, (w, h))   if mask.shape[:2]   != (h, w) else mask

    if warped_r.shape[2] == 4:
        warped_alpha = warped_r[:, :, 3]
    else:
        warped_alpha = np.full(warped_r.shape[:2], 255, dtype=np.uint8)

    soft_mask         = feather_mask(mask_r, feather_px)
    soft_design_alpha = feather_mask(warped_alpha, feather_px)
    alpha             = (soft_mask * soft_design_alpha)[:, :, np.newaxis]

    base    = mockup.astype(np.float32)
    overlay = warped_r[:, :, :3].astype(np.float32)
    result  = base * (1 - alpha) + overlay * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
