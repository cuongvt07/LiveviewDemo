import cv2
import numpy as np

MAX_BYTES = 10 * 1024 * 1024
MAX_DIM   = 8000

def decode_design(data: bytes) -> np.ndarray:
    """Decode bytes → BGRA ndarray. Dùng chung cho mug + clothes."""
    if len(data) > MAX_BYTES:
        raise ValueError("image_too_large")
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("invalid_image")
    if max(img.shape[:2]) > MAX_DIM:
        raise ValueError("invalid_image")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    h, w = img.shape[:2]
    if max(h, w) > 3000:
        scale = 3000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img
