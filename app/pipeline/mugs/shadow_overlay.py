import cv2
import numpy as np
from ..shared.colorspace import to_linear, to_srgb

def apply_shadow_overlay(
    warped: np.ndarray,
    shadow_map: np.ndarray,
    strength: float = 0.45,
) -> np.ndarray:
    """
    Overlay blend shadow cho men sứ.
    Chỉ dùng cho Mugs.

    Men sứ: dùng Overlay (không phải Multiply) vì bề mặt bóng
    phản xạ ánh sáng theo cả 2 chiều tối/sáng.
    Tính toán trong Linear color space.
    """
    h, w = warped.shape[:2]
    result = warped.copy()
    rgb_linear = to_linear(result[:, :, :3])

    s = cv2.resize(shadow_map, (w, h)).astype(np.float32) / 255.0
    s_linear = to_linear(s[:, :, np.newaxis])

    overlay = np.where(
        s_linear < 0.5,
        2 * rgb_linear * s_linear,
        1 - 2 * (1 - rgb_linear) * (1 - s_linear),
    )
    rgb_out = rgb_linear + (overlay - rgb_linear) * strength
    result[:, :, :3] = to_srgb(np.clip(rgb_out, 0, 1))
    return result
