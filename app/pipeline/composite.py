# app/pipeline/composite.py

import cv2
import numpy as np
from .edge import feather_mask


def composite(
    mockup: np.ndarray,
    warped: np.ndarray,
    mask: np.ndarray,
    feather_px: int = 6,
) -> np.ndarray:
    '''
    Ghép warped design lên mockup theo mask mềm.

    Args:
        mockup:  BGR (H, W, 3)
        warped:  BGRA (H, W, 4)
        mask:    Grayscale (H, W) — 255 = vùng print
        feather_px: độ mờ viền (pixels)

    Returns:
        BGR (H, W, 3)
    '''
    h, w = mockup.shape[:2]
    warped_r = cv2.resize(warped, (w, h)) if warped.shape[:2] != (h, w) else warped
    mask_r   = cv2.resize(mask,   (w, h)) if mask.shape[:2]   != (h, w) else mask

    # Alpha từ design (nếu có) × mask vùng print × feather
    soft_mask = feather_mask(mask_r, feather_px)        # [0..1]
    design_alpha = warped_r[:, :, 3].astype(np.float32) / 255.0  # [0..1]
    alpha = (soft_mask * design_alpha)[:, :, np.newaxis]          # [H,W,1]

    base    = mockup.astype(np.float32)
    overlay = warped_r[:, :, :3].astype(np.float32)

    result = base * (1 - alpha) + overlay * alpha
    return np.clip(result, 0, 255).astype(np.uint8)
