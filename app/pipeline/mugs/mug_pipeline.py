import numpy as np
from ..shared.decode      import decode_design
from ..shared.design_transform import apply_design_transform
from ..shared.color_match import apply_color_match
from ..shared.composite   import composite
from .cylindrical_warp    import cylindrical_warp
from .shadow_overlay      import apply_shadow_overlay
from .specular_gloss      import apply_specular_gloss, extract_specular_from_mockup
from dataclasses import dataclass


@dataclass
class MugAssets:
    mockup:       np.ndarray
    shadow_map:   np.ndarray
    normal_map:   np.ndarray
    mask:         np.ndarray
    specular_map: np.ndarray
    config:       dict


def run_mug_pipeline(
    design_bytes: bytes,
    assets: MugAssets,
) -> np.ndarray:
    """
    Pipeline riêng cho Mugs.

    Thứ tự:
    [1] Cylindrical warp (sin/arcsin)
    [2] Color match
    [3] Shadow Overlay (men sứ)
    [4] Edge feather + Composite
    [5] Specular Gloss (Phong, screen blend)
    """
    cfg = assets.config
    l   = cfg.get("lighting", {})
    W, H = assets.mockup.shape[1], assets.mockup.shape[0]

    design = decode_design(design_bytes)
    dt = cfg.get("design_transform", {})
    design = apply_design_transform(
        design,
        scale=dt.get("scale", 1.0),
        offset_x=dt.get("offset_x", 0.0),
        offset_y=dt.get("offset_y", 0.0),
    )

    # [1] Cylindrical warp — đặc thù mug
    cyl = cfg.get("cylinder", {})
    warped = cylindrical_warp(
        design,
        print_area=cfg["print_area"],
        output_size=(W, H),
        theta_max_deg=cyl.get("theta_max_deg", 52.0),
        pitch=cyl.get("pitch", 0.0),
        smile_base=cyl.get("smile_base", 0.08),
        curve_top=cyl.get("curve_top"),
        curve_bottom=cyl.get("curve_bottom"),
    )

    # [2] Color match
    if cfg.get("color", {}).get("enable_color_match", True):
        warped = apply_color_match(
            warped, assets.mockup, assets.mask,
            strength=cfg["color"].get("match_strength", 0.40),
        )

    # [3] Shadow — Overlay cho men sứ
    warped = apply_shadow_overlay(
        warped, assets.shadow_map,
        strength=l.get("shadow_strength", 0.45),
    )

    # [4] Composite
    result = composite(
        assets.mockup, warped, assets.mask,
        feather_px=cfg.get("edge", {}).get("feather_px", 6),
    )

    # [5] Specular Phong — chỉ mug
    if l.get("specular_strength", 0) > 0:
        result = apply_specular_gloss(
            result, assets.specular_map,
            strength=l.get("specular_strength", 0.30),
            shininess=l.get("shininess", 20.0),
        )

    return result
