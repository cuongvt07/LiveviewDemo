# app/pipeline/specular.py

import cv2
import numpy as np


def extract_specular_from_mockup(
    mockup: np.ndarray,
    threshold: int = 220,
) -> np.ndarray:
    '''
    Extract vùng phản chiếu men sứ từ ảnh cốc trắng.
    Pixel rất sáng (> threshold) = highlight men sứ.

    Args:
        mockup:    BGR (H, W, 3) — ảnh cốc trắng gốc
        threshold: pixel value 0–255. 220 phù hợp cho cốc sứ bóng.

    Returns:
        Grayscale float32 (H, W) trong [0..1]
    '''
    gray = cv2.cvtColor(mockup, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Chỉ giữ phần vượt ngưỡng
    specular = np.clip(gray - threshold, 0, 255) / (255 - threshold)

    # Blur nhẹ để specular trông natural, không bị pixelated
    specular = cv2.GaussianBlur(specular, (15, 15), sigmaX=5)
    return specular   # [0..1]


def apply_oren_nayar_diffuse(
    diffuse_lin: np.ndarray,
    normal_map: np.ndarray,
    roughness: float = 0.8,
) -> np.ndarray:
    '''
    Oren-Nayar diffuse approximation (xấp xỉ cho vải/bề mặt nhám).
    A = 1.0 - 0.5 * roughness**2 / (roughness**2 + 0.33)
    B = 0.45 * roughness**2 / (roughness**2 + 0.09)
    '''
    h, w = diffuse_lin.shape[:2]
    nm = cv2.resize(normal_map, (w, h)).astype(np.float32) / 127.5 - 1.0
    
    # Nz tương ứng với cos(theta) nếu view vector là (0,0,1)
    cos_theta = np.clip(nm[:, :, 2], 0.001, 1.0)
    
    sigma2 = roughness ** 2
    A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33))
    B = 0.45 * (sigma2 / (sigma2 + 0.09))
    
    # Giả định đơn giản: sin_alpha * tan_beta xấp xỉ sqrt(1 - cos_theta^2)
    # Vì we don't have light direction vector for every pixel easily here.
    # User's formula: C_diffuse * (A + B * max(0, cos_theta_diff) * sin_alpha * tan_beta)
    # Ta dùng cos_theta (độ nghiêng bề mặt) để điều chỉnh độ sáng.
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta**2, 0, 1))
    tan_theta = sin_theta / cos_theta
    
    # Phụ thuộc vào góc nghiêng (vải ở mép tối hơn/sáng hơn tùy roughness)
    oren_nayar = A + B * sin_theta * tan_theta
    
    # Clip oren_nayar to avoid extreme values
    oren_nayar = np.clip(oren_nayar, 0.5, 1.5)
    
    return diffuse_lin * oren_nayar[:, :, np.newaxis]


def apply_specular(
    base_lin: np.ndarray,
    specular_map: np.ndarray,
    normal_map: np.ndarray,
    strength: float = 0.30,
    shininess: float = 20.0,
) -> np.ndarray:
    '''
    Screen-blend specular lên ảnh (trong linear space).
    Formula: C_spec = specular_map * strength * (1.0 / cos(theta))**shininess
    C_combined = 1.0 - (1.0 - C_diffuse) * (1.0 - C_spec)
    '''
    h, w = base_lin.shape[:2]
    spec_r = cv2.resize(specular_map, (w, h)).astype(np.float32)
    
    nm = cv2.resize(normal_map, (w, h)).astype(np.float32) / 127.5 - 1.0
    cos_theta = np.clip(nm[:, :, 2], 0.1, 1.0)
    
    # Fresnel-like specular highlight enhancement at grazing angles
    spec_term = (1.0 / cos_theta) ** (shininess / 10.0)
    spec_term = np.clip(spec_term, 1.0, 5.0)
    
    c_spec = spec_r * strength * spec_term
    c_spec = c_spec[:, :, np.newaxis]
    
    # Screen blend in linear space
    result_lin = 1.0 - (1.0 - base_lin) * (1.0 - c_spec)
    return np.clip(result_lin, 0, 1)
