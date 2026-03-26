#!/usr/bin/env python3
'''
Test pipeline locally — không cần server, không cần DB.
Chạy: python scripts/test_render.py --mockup <> --maps <> --design <> --output <>
'''
import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.pipeline import (
    decode_design, run_pipeline, TemplateAssets
)
from app.pipeline.specular import extract_specular_from_mockup


def main():
    ap = argparse.ArgumentParser(
        description='Test render pipeline locally without server'
    )
    ap.add_argument('--mockup',  required=True, help='Ảnh cốc trắng')
    ap.add_argument('--maps',    required=True, help='Thư mục chứa shadow/normal/mask')
    ap.add_argument('--design',  required=True, help='Ảnh design (PNG/JPG)')
    ap.add_argument('--output',  default='output/result.jpg')
    ap.add_argument('--quality', type=int, default=90)
    args = ap.parse_args()

    maps_dir = Path(args.maps)

    # Load mockup
    mockup = cv2.imread(args.mockup)
    if mockup is None:
        print(f'[ERR] Không đọc được mockup: {args.mockup}')
        sys.exit(1)

    H, W = mockup.shape[:2]

    # Load maps (với fallback)
    def loadOrPlaceholder(path, flags, fallback_fn):
        p = maps_dir / path
        if p.exists():
            img = cv2.imread(str(p), flags)
            if img is not None:
                return cv2.resize(img, (W, H))
        print(f'  [WARN] {path} không tìm thấy — dùng placeholder')
        return fallback_fn(H, W)

    shadow = loadOrPlaceholder(
        'shadow_map.png', cv2.IMREAD_GRAYSCALE,
        lambda h, w: np.full((h, w), 180, dtype=np.uint8)
    )
    normal = loadOrPlaceholder(
        'normal_map.png', cv2.IMREAD_COLOR,
        lambda h, w: np.full((h, w, 3), [128, 128, 220], dtype=np.uint8)
    )
    mask = loadOrPlaceholder(
        'mask.png', cv2.IMREAD_GRAYSCALE,
        lambda h, w: _makeCenterMask(h, w)
    )

    # Specular
    spec_path = maps_dir / 'specular_map.png'
    if spec_path.exists():
        spec_raw = cv2.imread(str(spec_path), cv2.IMREAD_GRAYSCALE)
        specular = cv2.resize(spec_raw, (W, H)).astype(np.float32) / 255.0
    else:
        specular = extract_specular_from_mockup(mockup)

    # Cấu hình mặc định
    config = {
        'print_area': _estimatePrintArea(mask),
        'lighting': {
            'shadow_strength':       0.45,
            'highlight_strength':    0.55,
            'displacement_strength': 0.10,
            'specular_strength':     0.30,
            'specular_threshold':    220,
        },
        'color': {
            'enable_color_match': True,
            'match_strength':     0.40,
        },
        'edge': {
            'feather_px': 6,
        },
        'output': {
            'jpeg_quality': args.quality,
        },
    }

    assets = TemplateAssets(
        mockup=mockup,
        shadow_map=shadow,
        normal_map=normal,
        mask=mask,
        specular_map=specular,
        config=config,
    )

    # Load design
    with open(args.design, 'rb') as f:
        design_bytes = f.read()

    # Render
    print(f'\nRendering {args.design} -> {args.output} ...')
    t0 = time.perf_counter()
    result_bytes, meta = run_pipeline(
        design_bytes, assets,
        output_format='jpg',
        jpeg_quality=args.quality,
    )
    elapsed = int((time.perf_counter() - t0) * 1000)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'wb') as f:
        f.write(result_bytes)

    print(f'Done -> {args.output}')
    print(f'Time: {elapsed}ms  |  Size: {len(result_bytes)//1024}KB')
    print(f'\nParams đang dùng:')
    print(f'  print_area:  {config["print_area"]}')
    print(f'  shadow:      {config["lighting"]["shadow_strength"]}')
    print(f'  specular:    {config["lighting"]["specular_strength"]}')
    print(f'  color_match: {config["color"]["match_strength"]}')
    print(f'  feather:     {config["edge"]["feather_px"]}px')


def _makeCenterMask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    mx, my = int(w * 0.22), int(h * 0.14)
    mask[my:h - my, mx:w - mx] = 255
    return mask


def _estimatePrintArea(mask: np.ndarray) -> dict:
    '''Extract 4 góc từ mask có sẵn.'''
    coords = cv2.findNonZero(mask)
    if coords is None:
        h, w = mask.shape
        return {
            'top_left':     [int(w * 0.22), int(h * 0.14)],
            'top_right':    [int(w * 0.78), int(h * 0.14)],
            'bottom_right': [int(w * 0.78), int(h * 0.86)],
            'bottom_left':  [int(w * 0.22), int(h * 0.86)],
        }
    x, y, bw, bh = cv2.boundingRect(coords)
    return {
        'top_left':     [x,      y     ],
        'top_right':    [x + bw, y     ],
        'bottom_right': [x + bw, y + bh],
        'bottom_left':  [x,      y + bh],
    }


if __name__ == '__main__':
    main()
