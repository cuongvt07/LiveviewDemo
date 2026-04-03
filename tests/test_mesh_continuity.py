import numpy as np
from app.pipeline.mugs.warp_builder import build_warp_map
from app.pipeline.mugs.mesh_config import MugMeshConfig


def test_c2_continuity():
    cfg = MugMeshConfig()
    map_x, map_y = build_warp_map(cfg, 512, 512, quality="full")

    d2x_dx2 = np.abs(np.diff(map_x, n=2, axis=1))
    med = np.median(d2x_dx2)
    thresh = max(1e-6, 10 * med)
    spikes = np.sum(d2x_dx2 > thresh)
    assert spikes < 500, f"Too many discontinuities: {spikes}"


def test_corner_smoothness():
    cfg = MugMeshConfig()
    map_x, map_y = build_warp_map(cfg, 512, 512, quality="full")
    H, W = map_x.shape
    r = max(4, int(0.08 * min(W, H)))
    for (cy, cx) in [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]:
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        region = map_x[y0:y1, x0:x1]
        gy, gx = np.gradient(region)
        max_grad = max(np.max(np.abs(gx)), np.max(np.abs(gy)))
        assert max_grad < 50.0, f"Corner gradient too large: {max_grad}"
