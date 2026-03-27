import cv2
import numpy as np
from ..shared.colorspace import to_linear, to_srgb

def apply_specular_gloss(
    composited: np.ndarray,
    specular_map: np.ndarray,
    strength: float = 0.30,
    shininess: float = 20.0,
) -> np.ndarray:
    """
    Phong specular cho men sứ bóng.
    Chỉ dùng cho Mugs — không dùng cho Clothes.

    shininess=20 theo Phong shading model cho ceramic glaze.
    Screen blend đè lên composite — men sứ phủ trên design.
    """
    h, w = composited.shape[:2]
    spec_r = cv2.resize(specular_map, (w, h))
    spec   = (spec_r * strength)[:, :, np.newaxis]

    # Tính toán trong Linear space
    base_linear = to_linear(composited)
    result_linear = 1.0 - (1.0 - base_linear) * (1.0 - spec)
    return to_srgb(np.clip(result_linear, 0, 1))


def extract_specular_from_mockup(
    mockup: np.ndarray,
    threshold: int = 220,
) -> np.ndarray:
    """Extract highlight men sứ từ ảnh cốc trắng."""
    gray = cv2.cvtColor(mockup, cv2.COLOR_BGR2GRAY).astype(np.float32)
    specular = np.clip(gray - threshold, 0, 255) / (255 - threshold)
    specular = cv2.GaussianBlur(specular, (15, 15), sigmaX=5)
    return specular
