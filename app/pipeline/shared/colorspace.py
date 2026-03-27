import numpy as np

def to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """
    sRGB uint8 hoặc float → Linear float32.
    Bắt buộc convert trước mọi blend operation.
    Áp dụng cho CẢ mug lẫn clothes.
    """
    f = img_srgb.astype(np.float32)
    if f.max() > 1.0:
        f = f / 255.0
    return np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)

def to_srgb(img_linear: np.ndarray) -> np.ndarray:
    """Linear float32 → sRGB uint8."""
    c = np.clip(img_linear, 0, 1)
    srgb = np.where(c <= 0.0031308,
                    c * 12.92,
                    1.055 * (c ** (1 / 2.4)) - 0.055)
    return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)
