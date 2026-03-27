#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from pathlib import Path

def generate_clothes_maps(mockup_path: str, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(mockup_path)
    if img is None: return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Wrinkle Map (High contrast grayscale for TPS points)
    wrinkle = cv2.equalizeHist(gray)
    cv2.imwrite(str(out / 'wrinkle_map.png'), wrinkle)
    
    # Shadow (Multiply optimized for fabric)
    shadow = cv2.normalize(gray, None, 50, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(out / 'shadow_map.png'), shadow)

if __name__ == '__main__':
    generate_clothes_maps(sys.argv[1], sys.argv[2])
