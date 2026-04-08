# app/pipeline/shared/gpu_ops.py
'''
GPU-accelerated image operations.
Sử dụng cv2.UMat cho các phép xử lý nặng khi GPU enabled.
Fallback về numpy khi GPU disabled.
'''

import cv2
import numpy as np
import threading
from collections import OrderedDict
from typing import Optional
from app.config import isGpuEnabled, toGpu, toCpu

# Cache UMat maps trên GPU — tránh upload lại mỗi request
_gpu_map_cache: OrderedDict[str, tuple[cv2.UMat, cv2.UMat]] = OrderedDict()
_GPU_MAP_CACHE_MAX = 30  # ~30 templates × 32MB = ~960MB VRAM tối đa
_gpu_cache_lock = threading.Lock()  # Bảo vệ cache khi multi-thread

def _get_or_upload_maps(
    cache_key: str,
    map_x: np.ndarray,
    map_y: np.ndarray
) -> tuple[cv2.UMat, cv2.UMat]:
    """Lấy UMat maps từ cache hoặc upload lần đầu. Thread-safe."""
    with _gpu_cache_lock:
        if cache_key in _gpu_map_cache:
            # LRU: move to end
            _gpu_map_cache.move_to_end(cache_key)
            return _gpu_map_cache[cache_key]

    # Upload lên GPU — chỉ 1 lần duy nhất per template
    # Thực hiện bên ngoài lock để tránh giữ lock quá lâu khi upload
    u_map_x = cv2.UMat(map_x.astype(np.float32))
    u_map_y = cv2.UMat(map_y.astype(np.float32))

    with _gpu_cache_lock:
        # Double-check: thread khác có thể đã upload trong lúc chờ
        if cache_key in _gpu_map_cache:
            _gpu_map_cache.move_to_end(cache_key)
            return _gpu_map_cache[cache_key]

        _gpu_map_cache[cache_key] = (u_map_x, u_map_y)

        # Evict nếu vượt giới hạn
        if len(_gpu_map_cache) > _GPU_MAP_CACHE_MAX:
            _gpu_map_cache.popitem(last=False)  # Xóa item cũ nhất

    return u_map_x, u_map_y



def gpuResize(img: np.ndarray, size: tuple) -> np.ndarray:
    '''Resize ảnh trên GPU nếu có.'''
    if isGpuEnabled():
        u = toGpu(img)
        result = cv2.resize(u, size)
        return toCpu(result)
    return cv2.resize(img, size)


def gpuGaussianBlur(img: np.ndarray, ksize: tuple, sigma: float = 0) -> np.ndarray:
    '''Gaussian blur trên GPU.'''
    if isGpuEnabled():
        u = toGpu(img)
        result = cv2.GaussianBlur(u, ksize, sigma)
        return toCpu(result)
    return cv2.GaussianBlur(img, ksize, sigma)


def gpuWarpAffine(img: np.ndarray, M, dsize: tuple, **kwargs) -> np.ndarray:
    '''Warp affine trên GPU.'''
    if isGpuEnabled():
        u = toGpu(img)
        result = cv2.warpAffine(u, M, dsize, **kwargs)
        return toCpu(result)
    return cv2.warpAffine(img, M, dsize, **kwargs)


def gpuRemap(
    img: np.ndarray,
    map1: np.ndarray,
    map2: np.ndarray,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0, 0),
    map_cache_key: Optional[str] = None,
) -> np.ndarray:
    '''cv2.remap trên GPU — dùng cho cả Cylindrical và TPS warp.'''
    if isGpuEnabled():
        u_img = toGpu(img)
        
        # Dùng GPU map cache nếu có key
        if map_cache_key is not None:
            u_map1, u_map2 = _get_or_upload_maps(map_cache_key, map1, map2)
        else:
            u_map1 = toGpu(map1.astype(np.float32))
            u_map2 = toGpu(map2.astype(np.float32))
            
        result = cv2.remap(u_img, u_map1, u_map2, interpolation, borderMode=borderMode, borderValue=borderValue)
        return toCpu(result)
    return cv2.remap(img, map1, map2, interpolation, borderMode=borderMode, borderValue=borderValue)


def gpuMultiply(a: np.ndarray, b: np.ndarray, scale: float = 1.0) -> np.ndarray:
    '''Multiply blend trên GPU.'''
    if isGpuEnabled():
        u_a = toGpu(a)
        u_b = toGpu(b)
        result = cv2.multiply(u_a, u_b, scale=scale)
        return toCpu(result)
    return cv2.multiply(a, b, scale=scale)


def gpuAddWeighted(a: np.ndarray, alpha: float, b: np.ndarray, beta: float, gamma: float = 0) -> np.ndarray:
    '''Weighted add trên GPU.'''
    if isGpuEnabled():
        u_a = toGpu(a)
        u_b = toGpu(b)
        result = cv2.addWeighted(u_a, alpha, u_b, beta, gamma)
        return toCpu(result)
    return cv2.addWeighted(a, alpha, b, beta, gamma)
