import numpy as np
from .mesh_config import MugMeshConfig
from .cosine_grid import cosine_spacing


def migrate_4x4_to_dense(old_config: dict) -> MugMeshConfig:
    new_config = MugMeshConfig(cols=12, rows=8)
    if not isinstance(old_config, dict):
        return new_config

    old_cols = int(old_config.get('cols', 4))
    old_rows = int(old_config.get('rows', 4))
    if old_cols != 4 or old_rows != 4:
        return new_config

    old_pts = np.array(old_config.get('points', [])).reshape((old_rows + 1, old_cols + 1, 2))
    new_u = np.array(cosine_spacing(new_config.cols))
    new_v = np.array(cosine_spacing(new_config.rows))

    # Bilinear interpolate from old grid into new grid
    for vi in range(new_config.rows + 1):
        for ui in range(new_config.cols + 1):
            u = new_u[ui] * old_cols
            v = new_v[vi] * old_rows
            x0 = int(np.floor(u)); x1 = min(int(np.ceil(u)), old_cols)
            y0 = int(np.floor(v)); y1 = min(int(np.ceil(v)), old_rows)
            sx = u - x0; sy = v - y0
            p00 = old_pts[y0, x0]
            p10 = old_pts[y0, x1]
            p01 = old_pts[y1, x0]
            p11 = old_pts[y1, x1]
            top = (1 - sx) * p00 + sx * p10
            bot = (1 - sx) * p01 + sx * p11
            pv = (1 - sy) * top + sy * bot
            idx = vi * (new_config.cols + 1) + ui
            new_config.points[idx].u = float(np.clip(pv[0], 0.0, 1.0))
            new_config.points[idx].v = float(np.clip(pv[1], 0.0, 1.0))

    return new_config
