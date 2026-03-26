# app/pipeline/lighting.py

import cv2
import numpy as np


def apply_displacement(
    warped: np.ndarray,
    normal_map: np.ndarray,
    strength: float = 0.10,
) -> np.ndarray:
    '''
    Warp nhẹ design theo normal map để mô phỏng bề mặt cong men sứ.
    strength thấp hơn v1.1 (0.10 thay vì 0.15) vì perspective warp
    đã xử lý phần lớn geometry — displacement chỉ để thêm micro-detail.
    '''
    h, w = warped.shape[:2]
    nm = cv2.resize(normal_map, (w, h)).astype(np.float32)

    Nx = nm[:, :, 2] / 255.0 * 2 - 1   # R → X
    Ny = nm[:, :, 1] / 255.0 * 2 - 1   # G → Y
    max_shift = strength * 20

    gx, gy = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    map_x = np.clip(gx + Nx * max_shift, 0, w - 1)
    map_y = np.clip(gy + Ny * max_shift, 0, h - 1)

    return cv2.remap(warped, map_x, map_y,
                     cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def apply_shadow(
    warped: np.ndarray,
    shadow_map: np.ndarray,
    strength: float = 0.45,
) -> np.ndarray:
    '''
    Multiply shadow map lên design trong linear space.
    Sửa lỗi gamma: decode sRGB -> linear trước khi nhân, sau đó re-encode.
    '''
    h, w = warped.shape[:2]
    result = warped.copy().astype(np.float32)
    
    # 1. Decode sRGB to linear: color_lin = (color / 255.0) ** 2.2
    rgb_lin = (result[:, :, :3] / 255.0) ** 2.2
    
    # 2. Prepare shadow map in linear space
    s = cv2.resize(shadow_map, (w, h)).astype(np.float32) / 255.0
    s_lin = s ** 2.2
    s_lin = s_lin[:, :, np.newaxis]
    
    # 3. Multiply in linear space (physical lighting)
    # Lấy blend giữa nguyên bản và shadow theo strength
    shadowed_lin = rgb_lin * s_lin
    combined_lin = rgb_lin + (shadowed_lin - rgb_lin) * strength
    
    # 4. Re-encode to sRGB: color_srgb = color_lin ** (1/2.2) * 255
    rgb_out = np.clip(combined_lin, 0, 1) ** (1.0 / 2.2)
    
    result[:, :, :3] = (rgb_out * 255).astype(np.uint8)
    return result.astype(np.uint8)


BLEND_PRESETS = {
    "cylinder_ceramic": {"multiply": 0.85, "highlight": 0.15, "opacity": 0.90},
    "apparel_cotton":   {"multiply": 0.80, "highlight": 0.10, "opacity": 0.88},
    "plastic_case":     {"multiply": 0.70, "highlight": 0.25, "opacity": 0.95},
    "fabric_bag":       {"multiply": 0.90, "highlight": 0.05, "opacity": 0.85},
    "flat_print":       {"multiply": 0.00, "highlight": 0.00, "opacity": 1.00},
}

def composite_with_surface_lighting(design_warped, mockup, soft_mask, product_type="cylinder_ceramic"):
    """
    Composite thiết kế với mockup sử dụng Multiply blend + surface highlight.
    """
    preset = BLEND_PRESETS.get(product_type, BLEND_PRESETS["flat_print"])
    multiply_weight = preset["multiply"]
    hi_weight = preset["highlight"]
    opacity = preset["opacity"]

    # Chuyển sang float32 [0, 1]
    design_f = design_warped[:, :, :3].astype(np.float32) / 255.0
    mockup_f = mockup[:, :, :3].astype(np.float32) / 255.0
    alpha_f = soft_mask.astype(np.float32) / 255.0
    
    if design_warped.shape[2] == 4:
        # Nếu có alpha channel, nhân mask với alpha của thiết kế
        # Design alpha thường ở byte 3
        d_alpha = design_warped[:, :, 3].astype(np.float32) / 255.0
        alpha_f = alpha_f * d_alpha

    alpha_f = alpha_f[:, :, np.newaxis]
    
    # 1. Multiply blend (Shadows/Folds bleeds through)
    multiply = design_f * mockup_f
    
    # 2. Extract Highlights từ mockup
    # Lấy lấp lánh bề mặt từ mockup gốc
    mockup_gray = cv2.cvtColor(mockup, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    highlight_map = np.clip((mockup_gray - 0.5) * 2.0, 0, 1)[:, :, np.newaxis]
    
    # HIỆU CHỈNH: Công thức chuẩn cho Multiply + Specular Highlight
    # Specular highlight là lớp cộng dồn phía trên cùng
    # result = (design * shadow) + specular
    # Ở đây ta dùng trọng số hi_weight để kiểm soát độ mạnh của highlight
    base_weight = 1.0 - hi_weight
    surface_blend = multiply * base_weight + highlight_map * hi_weight
    surface_blend = np.clip(surface_blend, 0, 1)
    
    # 4. Final mix với mockup gốc dựa trên mask alpha
    # result = design_layer * alpha + mockup * (1-alpha)
    final_design_layer = surface_blend * opacity + design_f * (1 - opacity)
    
    result = final_design_layer * alpha_f + mockup_f * (1 - alpha_f)
    
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)
