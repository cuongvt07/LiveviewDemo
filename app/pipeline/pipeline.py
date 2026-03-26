# app/pipeline/pipeline.py

import cv2
import numpy as np
import time
from dataclasses import dataclass
from .warp import perspective_warp, cylinder_warp
from .color_match import apply_color_match
from .lighting import apply_displacement, apply_shadow
from .composite import composite
from .specular import extract_specular_from_mockup, apply_specular, apply_oren_nayar_diffuse


MAX_BYTES = 10 * 1024 * 1024  # 10MB
MAX_DIM   = 8000


@dataclass
class TemplateAssets:
    mockup:       np.ndarray   # BGR, pre-resized
    shadow_map:   np.ndarray   # Grayscale, pre-resized
    normal_map:   np.ndarray   # BGR, pre-resized
    mask:         np.ndarray   # Grayscale, pre-resized
    specular_map: np.ndarray   # float32, pre-computed từ mockup
    config:       dict


def decode_design(data: bytes) -> np.ndarray:
    '''
    Decode bytes → BGRA ndarray.
    Raise ValueError nếu không hợp lệ.
    '''
    if len(data) > MAX_BYTES:
        raise ValueError('image_too_large')

    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError('invalid_image')
    if max(img.shape[:2]) > MAX_DIM:
        raise ValueError('invalid_image')

    # Đảm bảo luôn có 4 channel (BGRA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Resize xuống nếu quá lớn (giảm ~30% thời gian warp)
    h, w = img.shape[:2]
    if max(h, w) > 3000:
        scale = 3000 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def run_pipeline(
    design_bytes: bytes,
    assets: TemplateAssets,
    output_format: str = 'jpg',
    jpeg_quality: int = 90,
) -> tuple[bytes, dict]:
    '''
    Main render function. Thread-safe (không có shared mutable state).

    Returns:
        (image_bytes, meta_dict)
    '''
    t0 = time.perf_counter()
    cfg = assets.config

    # [0] Decode
    design = decode_design(design_bytes)

    W = assets.mockup.shape[1]
    H = assets.mockup.shape[0]

    # [1] Phân tích config warp
    warp_config = cfg.get('warp', {})
    warp_type = warp_config.get('warp_type', 'cylinder')  # mặc định cylinder cho mug

    if warp_type == 'cylinder':
        theta_max_deg = float(warp_config.get('theta_max_deg', 52.0))
        smile = float(warp_config.get('curve', 0.15))
        pitch = float(cfg['print_area'].get('camera_elevation', 0.0))
        d_scale = float(warp_config.get('design_scale', 1.0))
        d_off_x = float(warp_config.get('design_offset_x', 0.0))
        d_off_y = float(warp_config.get('design_offset_y', 0.0))
        
        warped = cylinder_warp(
            design,
            print_area=cfg['print_area'],
            output_size=(W, H),
            theta_max_deg=theta_max_deg,
            smile=smile,
            pitch=pitch,
            design_scale=d_scale,
            design_offset_x=d_off_x,
            design_offset_y=d_off_y
        )
        current_normal = assets.normal_map
    elif warp_type == 'tps':
        # Đối với TPS, chúng ta warp cả normal map để reproject lighting logic
        warped, warped_normal = tps_warp(
            design,
            print_area=cfg['print_area'],
            output_size=(W, H),
            normal_map=assets.normal_map
        )
        # Update normal map used for lighting
        current_normal = warped_normal
    else:
        # Perspective
        warped = perspective_warp(
            design,
            print_area=cfg['print_area'],
            output_size=(W, H),
        )
        current_normal = assets.normal_map # Fallback

    # [2] Color match
    if cfg.get('color', {}).get('enable_color_match', True):
        warped = apply_color_match(
            warped,
            assets.mockup,
            assets.mask,
            strength=cfg['color'].get('match_strength', 0.40),
        )

    # [3] Lighting (Linear Space Chain)
    lighting = cfg.get('lighting', {})
    
    # Displacement vẫn chạy trên sRGB/pixel coordinates
    if lighting.get('displacement_strength', 0) > 0:
        warped = apply_displacement(
            warped, current_normal,
            strength=lighting.get('displacement_strength', 0.10),
        )

    # Chuyển sang linear space cho các bước tính toán ánh sáng vật lý
    warped_f = warped.astype(np.float32) / 255.0
    alpha = warped_f[:, :, 3:]
    warped_lin = warped_f[:, :, :3] ** 2.2
    
    # Shadow (Diffuse multiplication) - sRGB conversion moved to pipeline or kept in apply_shadow?
    # Để tối ưu, ta làm thẳng trong linear space ở đây.
    shadow_strength = lighting.get('shadow_strength', 0.45)
    if shadow_strength > 0:
        h_w, w_w = warped.shape[:2]
        s = cv2.resize(assets.shadow_map, (w_w, h_w)).astype(np.float32) / 255.0
        s_lin = (s ** 2.2)[:, :, np.newaxis]
        
        # Multiply design * shadow
        shadowed_lin = warped_lin * s_lin
        warped_lin = warped_lin + (shadowed_lin - warped_lin) * shadow_strength

    # Oren-Nayar Diffuse (Bề mặt vải/nhám)
    roughness = lighting.get('roughness', 0.8) # Mặc định 0.8 cho cotton
    if lighting.get('use_oren_nayar', False):
        warped_lin = apply_oren_nayar_diffuse(
            warped_lin, current_normal,
            roughness=roughness
        )

    # Specular (Screen blend)
    spec_strength = lighting.get('specular_strength', 0.30)
    if spec_strength > 0:
        warped_lin = apply_specular(
            warped_lin, assets.specular_map, current_normal,
            strength=spec_strength,
            shininess=lighting.get('shininess', 20.0)
        )

    # Quay lại sRGB
    warped_srgb = np.clip(warped_lin, 0, 1) ** (1.0 / 2.2)
    warped[:, :, :3] = (warped_srgb * 255).astype(np.uint8)

    # [4] + [5] Edge feather + Composite
    # --- Custom Polygon Mask Enhancement ---
    final_mask = assets.mask
    if 'mask_points' in cfg.get('print_area', {}) and cfg['print_area']['mask_points']:
        pts = np.array(cfg['print_area']['mask_points'], dtype=np.int32)
        poly_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [pts], 255)
        # Kết hợp Mask hệ thống và Mask người dùng vẽ (giao nhau)
        final_mask = cv2.bitwise_and(final_mask, poly_mask)

    result = composite(
        assets.mockup, warped, final_mask,
        feather_px=cfg.get('edge', {}).get('feather_px', 6),
    )


    # [7] Encode
    if output_format == 'png':
        ok, buf = cv2.imencode('.png', result)
        content_type = 'image/png'
    else:
        ok, buf = cv2.imencode(
            '.jpg', result,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
             cv2.IMWRITE_JPEG_OPTIMIZE, 1]
        )
        content_type = 'image/jpeg'

    if not ok:
        raise RuntimeError('encode_failed')

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    meta = {
        'processing_time_ms': elapsed_ms,
        'content_type': content_type,
        'output_size': [W, H],
    }
    return bytes(buf), meta
