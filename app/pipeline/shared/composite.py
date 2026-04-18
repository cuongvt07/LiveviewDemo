import cv2
import numpy as np
from .edge import feather_mask

def composite(
    mockup: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray,
    feather_px: int = 10,
    blur_design: bool = True,
) -> np.ndarray:
    """
    Alpha composite warped design lên mockup.
    Dùng chung — chỉ khác feather_px:
      - Mug:     feather_px=10 (viền mềm hơn — men sứ)
      - Clothes: feather_px=10 (viền mềm hơn — vải)
    """
    h, w = mockup.shape[:2]
    warped_r = cv2.resize(warped, (w, h)) if warped.shape[:2] != (h, w) else warped
    mask_r   = cv2.resize(mask, (w, h))   if mask.shape[:2]   != (h, w) else mask

    if warped_r.shape[2] == 4:
        warped_alpha = warped_r[:, :, 3]
        alpha_f = warped_alpha.astype(np.float32) / 255.0
        safe_alpha = np.where(alpha_f > 0, alpha_f, 1.0)
        overlay = (warped_r[:, :, :3].astype(np.float32) / safe_alpha[:, :, np.newaxis])
        overlay = np.clip(overlay, 0, 255)
    else:
        warped_alpha = np.full(warped_r.shape[:2], 255, dtype=np.uint8)
        overlay = warped_r[:, :, :3].astype(np.float32)

    soft_mask = feather_mask(mask_r, feather_px)
    if blur_design:
        soft_design_alpha = feather_mask(warped_alpha, feather_px)
    else:
        soft_design_alpha = warped_alpha.astype(np.float32) / 255.0
        
    alpha = (soft_mask * soft_design_alpha)[:, :, np.newaxis]

    base    = mockup.astype(np.float32)
    result  = base * (1 - alpha) + overlay * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
