#!/usr/bin/env python3
'''
Dùng khi thêm template mới, không có Blender.
Chạy: python scripts/generate_maps_from_photo.py <mockup.jpg> <output_dir>

Output:
  shadow_map.png  — bóng extract từ luminance ảnh thật
  normal_map.png  — hướng bề mặt từ Sobel gradient
  specular_map.png — highlight men sứ (pixel rất sáng)
  mask_placeholder.png — cần tạo thủ công hoặc qua Admin UI
'''

import cv2
import numpy as np
import sys
from pathlib import Path


def generate_all_maps(mockup_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(mockup_path)
    if img is None:
        raise FileNotFoundError(f'Không đọc được: {mockup_path}')

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Shadow map ---
    shadow = cv2.normalize(gray, None, 100, 220, cv2.NORM_MINMAX)
    shadow = cv2.GaussianBlur(shadow, (5, 5), 0)
    cv2.imwrite(str(out / 'shadow_map.png'), shadow)
    print(f'  shadow_map.png -> {shadow.min():.0f}-{shadow.max():.0f}')

    # --- Normal map ---
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=7)
    gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=7)

    gx_n = cv2.normalize(gx, None, -1, 1, cv2.NORM_MINMAX)
    gy_n = cv2.normalize(gy, None, -1, 1, cv2.NORM_MINMAX)
    gz   = np.sqrt(np.clip(1 - gx_n**2 - gy_n**2, 0, 1))

    nx = ((gx_n + 1) / 2 * 255).astype(np.uint8)
    ny = ((gy_n + 1) / 2 * 255).astype(np.uint8)
    nz = (gz * 255).astype(np.uint8)
    normal = cv2.merge([nx, ny, nz])
    cv2.imwrite(str(out / 'normal_map.png'), normal)
    print(f'  normal_map.png -> OK')

    # --- Specular map ---
    THRESHOLD = 220
    gray_f = gray.astype(np.float32)
    specular = np.clip(gray_f - THRESHOLD, 0, 255) / (255 - THRESHOLD)
    specular = cv2.GaussianBlur(specular, (21, 21), sigmaX=7)
    specular_u8 = (specular * 255).astype(np.uint8)
    cv2.imwrite(str(out / 'specular_map.png'), specular_u8)
    print(f'  specular_map.png -> max highlight: {specular.max():.3f}')

    # --- Mask placeholder ---
    mask = np.zeros((h, w), dtype=np.uint8)
    margin_x = int(w * 0.22)
    margin_y = int(h * 0.14)
    mask[margin_y:h - margin_y, margin_x:w - margin_x] = 255
    cv2.imwrite(str(out / 'mask_placeholder.png'), mask)
    print(f'  mask_placeholder.png -> ({margin_x},{margin_y}) -> ({w-margin_x},{h-margin_y})')
    print(f'  !!  Mask la placeholder - can chinh Print Area tren Admin UI!')

    print(f'\nDone -> {out}/')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python generate_maps_from_photo.py <mockup.jpg> <output_dir>')
        sys.exit(1)
    generate_all_maps(sys.argv[1], sys.argv[2])
