# app/services/template_registry.py

from app.pipeline.shared.asset_manager import (
    load_template_assets,
    get_assets,
    invalidate_assets,
    get_tps_mesh_cache,
    set_tps_mesh_cache
)
from typing import Any

try:
    from app.pipeline.mugs.mesh_migration import migrate_4x4_to_dense
    from app.pipeline.mugs.mesh_config import MugMeshConfig
except Exception:
    migrate_4x4_to_dense = None
    MugMeshConfig = None

def load_template(slug: str, record: dict):
    """Load template assets; migrate old 4x4 mesh configs to dense mesh automatically."""
    cfg: dict[str, Any] = record.get('config', {}) if isinstance(record, dict) else {}
    # Detect legacy 4x4 grid: points list of length 25 (5x5)
    try:
        pts = cfg.get('points')
        if migrate_4x4_to_dense and isinstance(pts, list) and len(pts) == 25:
            new_cfg = migrate_4x4_to_dense({'cols': 4, 'rows': 4, 'points': pts})
            # attach new mesh under cfg['mesh'] for backward compatibility
            mesh_payload = {
                'enabled': True,
                'cols': new_cfg.cols,
                'rows': new_cfg.rows,
                'spacing': new_cfg.spacing,
                'tension': new_cfg.tension,
                'corner_blend_radius': new_cfg.corner_blend_radius,
                'symmetry_lock': new_cfg.symmetry_lock,
                'max_displacement': new_cfg.max_displacement,
                'points': [[p.u, p.v, bool(p.locked)] for p in new_cfg.points],
            }
            cfg['mesh'] = mesh_payload
            record['config'] = cfg
    except Exception:
        pass

    return load_template_assets(slug, record)

def get(slug: str):
    return get_assets(slug)

def invalidate(slug: str):
    invalidate_assets(slug)

def get_cached_mesh(slug: str, shirt_img, mask, cfg):
    mesh = get_tps_mesh_cache(slug)
    if mesh is None:
        from app.pipeline.clothes.harris_control_points import extract_wrinkle_control_points
        src_pts, dst_pts = extract_wrinkle_control_points(
            shirt_img, mask,
            n_points=cfg.get("mesh", {}).get("n_points", 80),
            strength=cfg.get("mesh", {}).get("displacement_strength", 6.0),
        )
        mesh = (src_pts, dst_pts)
        set_tps_mesh_cache(slug, mesh)
    return mesh

