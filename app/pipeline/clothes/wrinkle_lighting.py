import cv2
import numpy as np
from ..shared.colorspace import to_linear, to_srgb

def apply_fabric_multiply(
    warped: np.ndarray,
    shirt_img: np.ndarray,
    shadow_strength: float = 0.40,
    specular_strength: float = 0.08,
) -> np.ndarray:
    """
    Multiply blend shadow cho vải trong Linear color space.
    Chỉ dùng cho Clothes — không dùng cho Mugs.

    Vải cotton: Multiply (không phải Overlay)
    - Vùng sáng áo (≈1.0) → design gần nguyên
    - Vùng tối áo (≈0.5) → design tối theo đúng nếp gấp
    - Specular rất thấp (0.05–0.10) vì cotton không bóng

    Linear color space bắt buộc — tránh màu bị shift khi blend.
    """
    h, w = warped.shape[:2]

    shirt_gray = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2GRAY)
    shirt_gray = cv2.GaussianBlur(shirt_gray, (0, 0), sigmaX=3)

    # Convert sang Linear trước blend
    design_linear = to_linear(warped[:, :, :3])
    shadow = to_linear(shirt_gray)[:, :, np.newaxis]

    # Multiply blend
    multiplied = design_linear * shadow
    shadowed = design_linear + (multiplied - design_linear) * shadow_strength
    shadowed = np.clip(shadowed, 0, 1)

    # Specular cực nhẹ cho cotton
    if specular_strength > 0:
        spec_raw = np.clip(shirt_gray.astype(np.float32) - 240, 0, 15) / 15.0
        spec_raw = cv2.GaussianBlur(spec_raw, (21, 21), sigmaX=7)
        spec = spec_raw[:, :, np.newaxis] * specular_strength
        shadowed = 1.0 - (1.0 - shadowed) * (1.0 - spec)
        shadowed = np.clip(shadowed, 0, 1)

    result = warped.copy()
    result[:, :, :3] = to_srgb(shadowed)
    return result
