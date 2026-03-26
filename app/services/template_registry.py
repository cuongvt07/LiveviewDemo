# app/services/template_registry.py

import cv2
import numpy as np
import threading
from app.pipeline.pipeline import TemplateAssets
from app.pipeline.specular import extract_specular_from_mockup


_registry: dict[str, TemplateAssets] = {}
_lock = threading.RLock()


def load_template(slug: str, record: dict) -> TemplateAssets:
    '''Load và pre-process tất cả assets cho 1 template.'''
    W = record['output_width']
    H = record['output_height']

    def load_img(path, flags=cv2.IMREAD_COLOR):
        if not path:
            return None
        img = cv2.imread(path, flags)
        if img is None:
            raise FileNotFoundError(f'Không tìm thấy: {path}')
        return img

    mockup = cv2.resize(load_img(record['mockup_path']), (W, H))
    
    # Shadow fallback: neutral gray (no shadow)
    s_raw = load_img(record.get('shadow_map_path'), cv2.IMREAD_GRAYSCALE)
    if s_raw is None:
        shadow = np.full((H, W), 255, dtype=np.uint8)
    else:
        shadow = cv2.resize(s_raw, (W, H))

    # Normal fallback: neutral blue [128, 128, 255] (pointing straight out)
    n_raw = load_img(record.get('normal_map_path'))
    if n_raw is None:
        normal = np.zeros((H, W, 3), dtype=np.uint8)
        normal[:] = [128, 128, 255]
    else:
        normal = cv2.resize(n_raw, (W, H))

    # Mask: required
    mask = cv2.resize(load_img(record['mask_path'], cv2.IMREAD_GRAYSCALE), (W, H))

    # Specular: dùng pre-computed file nếu có, không thì extract từ mockup
    if record.get('specular_path'):
        spec_raw = cv2.resize(
            load_img(record['specular_path'], cv2.IMREAD_GRAYSCALE), (W, H))
        specular = spec_raw.astype(np.float32) / 255.0
    else:
        specular = extract_specular_from_mockup(mockup)

    assets = TemplateAssets(
        mockup=mockup,
        shadow_map=shadow,
        normal_map=normal,
        mask=mask,
        specular_map=specular,
        config=record['config'],
    )

    set_(slug, assets)
    return assets


def get(slug: str) -> TemplateAssets | None:
    with _lock:
        return _registry.get(slug)


def set_(slug: str, assets: TemplateAssets):
    with _lock:
        _registry[slug] = assets


def invalidate(slug: str):
    with _lock:
        _registry.pop(slug, None)


def listSlugs() -> list[str]:
    with _lock:
        return list(_registry.keys())
