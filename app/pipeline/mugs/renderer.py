import cv2
import numpy as np
from .warp_cache import WarpMapRegistry

registry = WarpMapRegistry()

def render_mug_preview(design: np.ndarray, config, out_w: int = 512, out_h: int = 512) -> np.ndarray:
    map_x, map_y = registry.get_or_build(config, out_w, out_h, quality="preview")
    warped = cv2.remap(design, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    return warped


def render_mug_final(design: np.ndarray, config, out_w: int = 2048, out_h: int = 2048) -> np.ndarray:
    map_x, map_y = registry.get_or_build(config, out_w, out_h, quality="full")
    warped = cv2.remap(design, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_TRANSPARENT)
    return warped
