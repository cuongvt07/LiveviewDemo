import cv2
import numpy as np

from ..shared.colorspace import to_linear, to_srgb


def apply_shadow_overlay(
    warped: np.ndarray,
    shadow_map: np.ndarray,
    strength: float = 0.45,
) -> np.ndarray:
    """
    Apply lighting-derived shadow to the warped design in linear space.

    `shadow_map` can be in [0..1] or [0..255]. Brighter values preserve more of
    the design, darker values attenuate it.
    """
    h, w = warped.shape[:2]
    result = warped.copy()
    rgb_linear = to_linear(result[:, :, :3])

    shadow = cv2.resize(shadow_map, (w, h)).astype(np.float32)
    if shadow.max() > 1.0:
        shadow /= 255.0
    shadow = np.clip(shadow, 0.0, 1.0)

    shadow_factor = 1.0 - (1.0 - shadow) * float(np.clip(strength, 0.0, 1.0))
    rgb_out = rgb_linear * shadow_factor[:, :, np.newaxis]
    result[:, :, :3] = to_srgb(np.clip(rgb_out, 0.0, 1.0))
    return result
