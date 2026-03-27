import cv2
import numpy as np
from .colorspace import to_linear, to_srgb

def apply_color_match(
    warped: np.ndarray,
    mockup: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.35,
) -> np.ndarray:
    """
    Von Kries chromatic adaptation — shift màu design theo ánh sáng ảnh gốc.
    Dùng chung cho mug + clothes. Chỉ khác strength:
      - Mug:     strength=0.40 (studio lighting mạnh hơn)
      - Clothes: strength=0.25 (natural light, nhẹ hơn)
    """
    region = mockup[mask > 128]
    if len(region) == 0:
        return warped
    gray_vals = region.mean(axis=1)
    threshold = np.percentile(gray_vals, 95)
    bright = region[gray_vals >= threshold].astype(np.float32)
    white_point = np.clip(bright.mean(axis=0) / 255.0, 0.5, 1.0)

    result = warped.copy().astype(np.float32)
    rgb = result[:, :, :3] / 255.0
    rgb_shifted = rgb * (1 + (white_point - 1.0) * strength)
    result[:, :, :3] = (np.clip(rgb_shifted, 0, 1) * 255).astype(np.uint8)
    return result.astype(np.uint8)
