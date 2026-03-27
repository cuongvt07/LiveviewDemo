# app/pipeline/shared/gpu_ops.py
'''
GPU-accelerated image operations.
Sử dụng cv2.UMat cho các phép xử lý nặng khi GPU enabled.
Fallback về numpy khi GPU disabled.
'''

import cv2
import numpy as np
from app.config import isGpuEnabled, toGpu, toCpu


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


def gpuRemap(img: np.ndarray, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)) -> np.ndarray:
    '''cv2.remap trên GPU — dùng cho cả Cylindrical và TPS warp.'''
    if isGpuEnabled():
        u_img = toGpu(img)
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
