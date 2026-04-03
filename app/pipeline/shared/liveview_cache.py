from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

import numpy as np


def _normalize_scalar(value: Any, digits: int = 4) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _quantize_scalar(value: Any, step: float) -> Any:
    """Làm tròn tham số theo bước (step) để gom các request gần giống nhau về chung cache key."""
    if value is None:
        return None
    from numbers import Number
    if isinstance(value, (np.floating, float)) or (isinstance(value, Number) and not isinstance(value, bool)):
        return float(round(round(float(value) / step) * step, 6))
    return value


def _normalize_points(points: list | tuple | None, quantize_step: float | None = None) -> tuple:
    if not isinstance(points, (list, tuple)):
        return ()
    normalized = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = float(point[0]), float(point[1])
            if quantize_step is not None and quantize_step > 0:
                x = round(x / quantize_step) * quantize_step
                y = round(y / quantize_step) * quantize_step
            normalized.append((int(round(x)), int(round(y))))
    return tuple(normalized)


class _ByteBoundLruCache:
    def __init__(self, max_bytes: int):
        self._max_bytes = max(1, int(max_bytes))
        self._entries: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.RLock()

    def _estimate_size(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(self._estimate_size(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value)
        return 0

    def get(self, key: Any) -> Any | None:
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return None
            self._entries[key] = entry
            return entry[0]

    def set(self, key: Any, value: Any) -> Any:
        size = self._estimate_size(value)
        if size <= 0 or size > self._max_bytes:
            return value

        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._current_bytes -= existing[1]

            while self._entries and self._current_bytes + size > self._max_bytes:
                _, (_, removed_size) = self._entries.popitem(last=False)
                self._current_bytes -= removed_size

            self._entries[key] = (value, size)
            self._current_bytes += size
        return value


_preview_canvas_cache = _ByteBoundLruCache(max_bytes=32 * 1024 * 1024)
_cylindrical_map_cache = _ByteBoundLruCache(max_bytes=160 * 1024 * 1024)
_tps_map_cache = _ByteBoundLruCache(max_bytes=400 * 1024 * 1024)


def make_preview_canvas_cache_key(width: int, height: int, mesh_density_strength: float) -> tuple:
    return (
        int(width),
        int(height),
        _normalize_scalar(mesh_density_strength, digits=3),
    )


def get_preview_canvas_cache(key: tuple) -> np.ndarray | None:
    return _preview_canvas_cache.get(key)


def set_preview_canvas_cache(key: tuple, canvas: np.ndarray) -> np.ndarray:
    return _preview_canvas_cache.set(key, canvas)


def make_cylindrical_map_cache_key(
    print_area: dict,
    output_size: tuple[int, int],
    design_size: tuple[int, int],
    theta_max_deg: float,
    pitch: float,
    smile_base: float,
    curve_top: float | None,
    curve_bottom: float | None,
    edge_squeeze: float,
    squeeze_power: float,
    center_focus_width: float,
    is_preview: bool = False,
) -> tuple:
    quantize_step = 4.0 if is_preview else None
    return (
        int(output_size[0]),
        int(output_size[1]),
        int(design_size[0]),
        int(design_size[1]),
        is_preview,
        _normalize_points(
            [
                print_area.get("top_left"),
                print_area.get("top_right"),
                print_area.get("bottom_right"),
                print_area.get("bottom_left"),
            ],
            quantize_step=quantize_step
        ),
        _quantize_scalar(theta_max_deg, 2.0),
        _quantize_scalar(pitch, 5.0),
        _quantize_scalar(smile_base, 0.02),
        _quantize_scalar(curve_top, 5.0),
        _quantize_scalar(curve_bottom, 5.0),
        _quantize_scalar(edge_squeeze, 0.05),
        _quantize_scalar(squeeze_power, 0.1),
        _quantize_scalar(center_focus_width, 0.05),
    )



def get_cylindrical_map_cache(key: tuple) -> dict | None:
    return _cylindrical_map_cache.get(key)


def set_cylindrical_map_cache(key: tuple, bundle: dict) -> dict:
    return _cylindrical_map_cache.set(key, bundle)


def make_tps_map_cache_key(
    src_pts: np.ndarray | list | tuple,
    dst_pts: np.ndarray | list | tuple,
    input_size: tuple[int, int],
    output_size: tuple[int, int],
    is_preview: bool = False,
) -> tuple:
    def _hash_pts(pts):
        pts_arr = np.asarray(pts, dtype=np.float32)
        if is_preview:
            # Quantize by 4px to increase cache hits during dragging
            quant_step = 4.0
            pts_arr = np.round(pts_arr / quant_step) * quant_step
        return hashlib.md5(pts_arr.tobytes()).hexdigest()
        
    return (
        _hash_pts(src_pts),
        _hash_pts(dst_pts),
        int(input_size[0]),
        int(input_size[1]),
        int(output_size[0]),
        int(output_size[1]),
        is_preview,
    )


def get_tps_map_cache(key: tuple) -> tuple[np.ndarray, np.ndarray] | None:
    return _tps_map_cache.get(key)


def set_tps_map_cache(key: tuple, tps_maps: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return _tps_map_cache.set(key, tps_maps)
