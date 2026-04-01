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


def apply_mug_lighting(
    warped: np.ndarray,
    light_map: np.ndarray,
    specular_mask: np.ndarray,
    specular_strength: float = 0.30,
    shininess: float = 20.0,
) -> np.ndarray:
    """Apply separated diffuse and specular lighting to a warped design image.

    - `warped` is expected to be an HxWx4 RGBA image (uint8) from the warp step.
    - `light_map` is a HxW float32 lighting multiplier in [0,1] (diffuse).
    - `specular_mask` is a HxW float32 map in [0,1] representing specular highlights.

    The function operates in linear color space: diffuse multiplies the linear
    color, specular is added in linear space, then converted back to sRGB.
    The alpha channel of `warped` is preserved.
    """
    if warped is None:
        return warped

    h, w = warped.shape[:2]
    lm = light_map.astype(np.float32)
    if lm.shape[:2] != (h, w):
        lm = cv2.resize(lm, (w, h), interpolation=cv2.INTER_LINEAR)

    sm = specular_mask.astype(np.float32)
    if sm.shape[:2] != (h, w):
        sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)

    rgb = warped[:, :, :3]
    alpha = warped[:, :, 3] if warped.ndim == 3 and warped.shape[2] == 4 else None

    base_linear = to_linear(rgb)

    diffuse = base_linear * lm[:, :, np.newaxis]

    spec_int = np.power(np.clip(sm, 0.0, 1.0), max(float(shininess), 1.0)) * float(specular_strength)
    spec = spec_int[:, :, np.newaxis]

    result_linear = np.clip(diffuse + spec, 0.0, 1.0)
    result_srgb = to_srgb(result_linear)

    if alpha is not None:
        out = np.dstack([result_srgb, alpha.astype(np.uint8)])
    else:
        out = result_srgb
    return out
