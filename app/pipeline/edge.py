# app/pipeline/edge.py

import cv2
import numpy as np


def feather_mask(mask: np.ndarray, feather_px: int = 6) -> np.ndarray:
    '''
    Làm mềm viền mask để composite trông tự nhiên hơn.

    Args:
        mask:       Grayscale uint8 (H, W) — 255 = print area
        feather_px: Bán kính blur (pixels). 4–8 cho cốc sứ.
                    0 = tắt feathering (giữ mask cứng).

    Returns:
        Grayscale float32 (H, W) trong khoảng [0, 1]
    '''
    if feather_px <= 0:
        return mask.astype(np.float32) / 255.0

    # kernel_size phải lẻ
    ksize = feather_px * 2 + 1
    blurred = cv2.GaussianBlur(
        mask.astype(np.float32),
        (ksize, ksize),
        sigmaX=feather_px / 2,
    )
    return blurred / 255.0   # [0..1]
