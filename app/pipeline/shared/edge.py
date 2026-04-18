import cv2
import numpy as np

def feather_mask(mask: np.ndarray, feather_px: int = 10) -> np.ndarray:
    """Làm mềm viền mask. Dùng chung."""
    if feather_px <= 0:
        return mask.astype(np.float32) / 255.0
    ksize = feather_px * 2 + 1
    blurred = cv2.GaussianBlur(
        mask.astype(np.float32), (ksize, ksize), sigmaX=feather_px / 2
    )
    return blurred / 255.0
