import cv2
import numpy as np


def estimate_print_area_canvas_size(
    print_area: dict | None,
    fallback_width: int,
    fallback_height: int,
) -> tuple[int, int]:
    if not isinstance(print_area, dict):
        return max(1, int(fallback_width)), max(1, int(fallback_height))

    keys = ("top_left", "top_right", "bottom_right", "bottom_left")
    if not all(k in print_area for k in keys):
        return max(1, int(fallback_width)), max(1, int(fallback_height))

    try:
        tl = np.float32(print_area["top_left"])
        tr = np.float32(print_area["top_right"])
        br = np.float32(print_area["bottom_right"])
        bl = np.float32(print_area["bottom_left"])
        top = float(np.linalg.norm(tr - tl))
        bottom = float(np.linalg.norm(br - bl))
        left = float(np.linalg.norm(bl - tl))
        right = float(np.linalg.norm(br - tr))
        width = int(round(max(1.0, (top + bottom) * 0.5)))
        height = int(round(max(1.0, (left + right) * 0.5)))
        return width, height
    except Exception:
        return max(1, int(fallback_width)), max(1, int(fallback_height))


def apply_design_transform(
    design: np.ndarray,
    scale: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    fit_mode: str = "cover",
    target_width: int | None = None,
    target_height: int | None = None,
) -> np.ndarray:
    """
    Apply editor design transform before warp.
    - Preserves original artwork aspect ratio.
    - Uses a print-area viewport model instead of translating inside the raw image canvas.
      This avoids exposing transparent borders during normal drag/pan.
    - fit_mode:
        - "cover": artwork fills the print viewport, possibly cropping overflow.
        - "contain": artwork stays fully inside the print viewport with transparent margins if needed.
    - scale: user zoom factor relative to the auto-fit base size selected by fit_mode.
    - offset_x/offset_y: normalized pan inside the overflow area, expected in [-1, 1].
    """
    user_scale = max(0.05, float(scale))
    offset_x = float(np.clip(offset_x, -1.0, 1.0))
    offset_y = float(np.clip(offset_y, -1.0, 1.0))
    normalized_fit_mode = str(fit_mode or "cover").strip().lower()
    if normalized_fit_mode not in {"cover", "contain"}:
        normalized_fit_mode = "cover"

    src_h, src_w = design.shape[:2]
    dst_w = max(1, int(target_width or src_w))
    dst_h = max(1, int(target_height or src_h))

    if src_w <= 0 or src_h <= 0:
        return design

    width_scale = dst_w / max(src_w, 1)
    height_scale = dst_h / max(src_h, 1)
    if normalized_fit_mode == "contain":
        base_scale = min(width_scale, height_scale)
    else:
        base_scale = max(width_scale, height_scale)
    total_scale = base_scale * user_scale
    scaled_w = src_w * total_scale
    scaled_h = src_h * total_scale
    overflow_x = max(0.0, scaled_w - dst_w)
    overflow_y = max(0.0, scaled_h - dst_h)

    tx = ((dst_w - scaled_w) * 0.5) + (offset_x * overflow_x * 0.5)
    ty = ((dst_h - scaled_h) * 0.5) + (offset_y * overflow_y * 0.5)
    mat = np.array([[total_scale, 0.0, tx], [0.0, total_scale, ty]], dtype=np.float32)

    interpolation = cv2.INTER_LANCZOS4
    if dst_w <= 1000:
        interpolation = cv2.INTER_LINEAR
    elif dst_w <= 2000:
        interpolation = cv2.INTER_CUBIC

    return cv2.warpAffine(
        design,
        mat,
        (dst_w, dst_h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
