import hashlib
import json
import threading
import numpy as np
from typing import Dict, Tuple

from .mesh_config import MugMeshConfig
from .warp_builder import build_warp_map
from ..shared.liveview_cache import ByteBoundLruCache


class WarpMapRegistry:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Giới hạn 512MB RAM cho bộ đệm ma trận warp lưới (mỗi bộ ~32MB ở 2K)
                    cls._instance._cache = ByteBoundLruCache(max_bytes=512 * 1024 * 1024)
        return cls._instance

    @staticmethod
    def _config_hash(config: MugMeshConfig, out_w: int, out_h: int) -> str:
        pts = [(p.u, p.v) for p in config.points]
        key = {"pts": pts, "tension": config.tension, "corner_radius": config.corner_blend_radius, "w": out_w, "h": out_h}
        h = hashlib.md5(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        return h

    def get_or_build(self, config: MugMeshConfig, out_w: int, out_h: int, quality: str = "full") -> Tuple[np.ndarray, np.ndarray]:
        if quality == "preview":
            return build_warp_map(config, out_w, out_h, quality="preview")

        k = self._config_hash(config, out_w, out_h)
        cached = self._cache.get(k)
        if cached is not None:
            return cached

        map_x, map_y = build_warp_map(config, out_w, out_h, quality="full")
        self._cache.set(k, (map_x, map_y))
        return map_x, map_y

    def warmup(self, sku_configs: Dict[str, MugMeshConfig], sizes=[(512, 512), (2048, 2048)]) -> None:
        def _t():
            for sku, cfg in sku_configs.items():
                for (w, h) in sizes:
                    try:
                        self.get_or_build(cfg, w, h, quality="full")
                        print(f"[WarpCache] Warmed {sku} @ {w}x{h}")
                    except Exception as e:
                        print(f"[WarpCache] Warm failed {sku}: {e}")
        thread = threading.Thread(target=_t, daemon=True)
        thread.start()
