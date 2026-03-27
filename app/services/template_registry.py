# app/services/template_registry.py

from app.pipeline.shared.asset_manager import (
    load_template_assets,
    get_assets,
    invalidate_assets,
    get_tps_mesh_cache,
    set_tps_mesh_cache
)

def load_template(slug: str, record: dict):
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

