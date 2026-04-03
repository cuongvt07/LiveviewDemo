import logging
import time
import numpy as np

_logger = logging.getLogger('mockup_service')

_f256 = np.arange(256) / 255.0
_LUT_LINEAR_FLOAT32 = np.where(_f256 <= 0.04045, _f256 / 12.92, ((_f256 + 0.055) / 1.055) ** 2.4).astype(np.float32)

_f65536 = np.arange(65536) / 65535.0
_srgb65536 = np.where(_f65536 <= 0.0031308, _f65536 * 12.92, 1.055 * (_f65536 ** (1 / 2.4)) - 0.055)
_LUT_SRGB_UINT8 = (np.clip(_srgb65536, 0.0, 1.0) * 255.0).astype(np.uint8)

def to_linear(img_srgb: np.ndarray) -> np.ndarray:
    """
    sRGB uint8 or float → Linear float32.
    """
    t0 = time.perf_counter()
    if img_srgb.dtype == np.uint8:
        idx = img_srgb.astype(np.int32)
    else:
        if img_srgb.max() <= 1.0:
            idx = np.clip(img_srgb * 255.0, 0, 255).astype(np.int32)
        else:
            idx = np.clip(img_srgb, 0, 255).astype(np.int32)
            
    result = _LUT_LINEAR_FLOAT32[idx]
    _logger.info('[PERF]     to_linear: %dms (shape=%s)', int((time.perf_counter() - t0) * 1000), img_srgb.shape)
    return result

def to_srgb(img_linear: np.ndarray) -> np.ndarray:
    """Linear float32 → sRGB uint8."""
    t0 = time.perf_counter()
    idx = np.clip(img_linear * 65535.0, 0, 65535).astype(np.int32)
    result = np.take(_LUT_SRGB_UINT8, idx)
    _logger.info('[PERF]     to_srgb: %dms (shape=%s)', int((time.perf_counter() - t0) * 1000), img_linear.shape)
    return result
