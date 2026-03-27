# app/pipeline/shared/asset_manager.py

import cv2
import numpy as np
import threading
from app.pipeline.mugs.mug_pipeline import MugAssets
from app.pipeline.clothes.clothes_pipeline import ClothesAssets
from app.pipeline.mugs.specular_gloss import extract_specular_from_mockup

_registry: dict[str, MugAssets | ClothesAssets] = {}
_mesh_cache: dict[str, tuple] = {}
_lock = threading.RLock()

def load_template_assets(slug: str, record: dict):
    '''Load và pre-process tất cả assets cho 1 template.'''
    W = record['output_width']
    H = record['output_height']

    def load_img(path, flags=cv2.IMREAD_COLOR):
        if not path: return None
        img = cv2.imread(path, flags)
        if img is None: raise FileNotFoundError(f'Không tìm thấy: {path}')
        return img

    mockup = cv2.resize(load_img(record['mockup_path']), (W, H))
    
    # Shadow fallback
    s_raw = load_img(record.get('shadow_map_path'), cv2.IMREAD_GRAYSCALE)
    shadow = cv2.resize(s_raw, (W, H)) if s_raw is not None else np.full((H, W), 255, dtype=np.uint8)

    # Normal fallback
    n_raw = load_img(record.get('normal_map_path'))
    if n_raw is None:
        normal = np.zeros((H, W, 3), dtype=np.uint8); normal[:] = [128, 128, 255]
    else:
        normal = cv2.resize(n_raw, (W, H))

    mask = cv2.resize(load_img(record['mask_path'], cv2.IMREAD_GRAYSCALE), (W, H))
    product_type = record['config'].get('product_type', 'mug')

    if product_type == 'mug':
        if record.get('specular_path'):
            spec_raw = cv2.resize(load_img(record['specular_path'], cv2.IMREAD_GRAYSCALE), (W, H))
            specular = spec_raw.astype(np.float32) / 255.0
        else:
            specular = extract_specular_from_mockup(mockup)

        assets = MugAssets(mockup=mockup, shadow_map=shadow, normal_map=normal, mask=mask, specular_map=specular, config=record['config'])
    else:
        wrinkle_raw = load_img(record.get('normal_map_path', None), cv2.IMREAD_GRAYSCALE)
        wrinkle = cv2.resize(wrinkle_raw, (W, H)) if wrinkle_raw is not None else np.full((H, W), 128, dtype=np.uint8)
        
        assets = ClothesAssets(mockup=mockup, wrinkle_map=wrinkle, shadow_map=shadow, mask=mask, config=record['config'], slug=slug)

    with _lock:
        _registry[slug] = assets
    return assets

def get_assets(slug: str):
    with _lock: return _registry.get(slug)

def invalidate_assets(slug: str):
    with _lock:
        _registry.pop(slug, None)
        _mesh_cache.pop(slug, None)

def get_tps_mesh_cache(slug: str):
    with _lock: return _mesh_cache.get(slug)

def set_tps_mesh_cache(slug: str, mesh_data: tuple):
    with _lock: _mesh_cache[slug] = mesh_data
