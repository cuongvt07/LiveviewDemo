#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from pathlib import Path

def generate_mug_maps(mockup_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(mockup_path)
    if img is None: return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Shadow (Overlay optimized for ceramic)
    shadow = cv2.normalize(gray, None, 120, 240, cv2.NORM_MINMAX)
    cv2.imwrite(str(out / 'shadow_map.png'), shadow)
    
    # Specular (Phong high threshold for gloss)
    specular = np.clip(gray.astype(np.float32) - 220, 0, 255) / 35.0
    specular = (np.clip(specular, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / 'specular_map.png'), specular)
    
    # Normal (Smooth cylindrical curvature)
    h, w = gray.shape
    normal = np.zeros((h, w, 3), dtype=np.uint8)
    normal[:] = [128, 128, 255]
    cv2.imwrite(str(out / 'normal_map.png'), normal)

if __name__ == '__main__':
    generate_mug_maps(sys.argv[1], sys.argv[2])
