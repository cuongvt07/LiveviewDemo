# app/pipeline/color_match.py

import cv2
import numpy as np


def extract_white_point(mockup: np.ndarray, mask: np.ndarray) -> np.ndarray:
    '''
    Tìm màu "trắng" thực tế của cốc trong vùng print area.
    Lấy top 5% pixel sáng nhất trong mask làm white reference.

    Returns: BGR float32 [0..1], shape (3,)
    '''
    # Chỉ lấy pixel trong print area (mask > 128)
    region = mockup[mask > 128]   # shape (N, 3)
    if len(region) == 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    gray_vals = region.mean(axis=1)
    threshold = np.percentile(gray_vals, 95)
    bright = region[gray_vals >= threshold].astype(np.float32)

    white_point = bright.mean(axis=0) / 255.0  # BGR, [0..1]
    # Clamp: tránh chia cho 0 hoặc bùng sáng
    white_point = np.clip(white_point, 0.5, 1.0)
    return white_point


def apply_color_match(
    warped: np.ndarray,
    mockup: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.40,
) -> np.ndarray:
    '''
    Shift màu của design để khớp với color temperature của mockup.
    Sử dụng Local Patch Transfer trong Lab space + Gaussian smoothing.
    '''
    h, w = warped.shape[:2]
    result = warped.copy()
    
    # 1. Convert to Lab: float32 for precision
    warped_lab = cv2.cvtColor(result[:, :, :3], cv2.COLOR_BGR2Lab).astype(np.float32)
    mockup_lab = cv2.cvtColor(mockup, cv2.COLOR_BGR2Lab).astype(np.float32)
    
    # 2. Local patch transfer
    PATCH = 64
    # Tạo map chứa delta L, a, b cho từng block
    h_p, w_p = (h + PATCH - 1) // PATCH, (w + PATCH - 1) // PATCH
    delta_map = np.zeros((h_p, w_p, 3), dtype=np.float32)
    
    for r in range(h_p):
        for c in range(w_p):
            r0, r1 = r * PATCH, min((r + 1) * PATCH, h)
            c0, c1 = c * PATCH, min((c + 1) * PATCH, w)
            
            # Chỉ tính trong vùng mask
            m_patch = mask[r0:r1, c0:c1]
            if np.mean(m_patch) < 20: # Block ngoài vùng in, dùng default
                 continue
                 
            p_w = warped_lab[r0:r1, c0:c1][m_patch > 128]
            p_m = mockup_lab[r0:r1, c0:c1][m_patch > 128]
            
            if len(p_w) > 0 and len(p_m) > 0:
                delta_map[r, c] = (np.mean(p_m, axis=0) - np.mean(p_w, axis=0))
    
    # 3. Upscale and smooth delta map
    delta_full = cv2.resize(delta_map, (w, h), interpolation=cv2.INTER_LINEAR)
    delta_full = cv2.GaussianBlur(delta_full, (PATCH*2+1, PATCH*2+1), sigmaX=PATCH)
    
    # 4. Apply shift
    warped_lab += delta_full * strength
    
    # 5. Convert back to BGR
    result[:, :, :3] = cv2.cvtColor(np.clip(warped_lab, 0, [100, 255, 255]).astype(np.uint8), cv2.COLOR_Lab2BGR)
    
    return result
