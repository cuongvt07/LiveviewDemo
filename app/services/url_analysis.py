from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
from contextlib import suppress
from dataclasses import asdict, dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen

try:
    import httpx
except Exception:  # pragma: no cover - optional fallback
    httpx = None  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUTS_DIR = REPO_ROOT / "inputs"
PUBLIC_DIR = REPO_ROOT / "public"
logger = logging.getLogger("mockup_service")

STATIC_ASSET_PREFIXES: tuple[tuple[str, Path], ...] = (
    ("/static/bases/", INPUTS_DIR / "bases"),
    ("/static/artworks/", INPUTS_DIR / "artworks"),
    ("/static/mockups/", PUBLIC_DIR / "mockups"),
    ("/static/templates/", REPO_ROOT / "templates"),
)

RESOLUTION_RE = re.compile(r"^\d+x\d+$", re.IGNORECASE)
HEX_COLOR_RE = re.compile(r"^[0-9a-fA-F]{6}$")
SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")
DEFAULT_MOCKUP_VIEW_ORDER: tuple[str, ...] = ("left", "right", "center")
DEFAULT_SECTION_URL_TEMPLATES: dict[str, str] = {
    "print": "https://asset.prtvstatic.com/{folder_path}/{name}.png",
    "cus": "https://asset.prtvstatic.com/{folder_path}/{name}.png",
}

_HTTPX_CLIENT: Optional["httpx.AsyncClient"] = None
_HTTPX_CLIENT_LOCK = asyncio.Lock()
_HTTP2_AVAILABLE = find_spec("h2") is not None
_COMMON_IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".avif")

MUG_MOCKUP_POLICIES: dict[tuple[str, str], dict] = {
    ("11oz", "white"): {
        "mockup_url": "/static/mockups/mug/mug-white-left-11oz.png",
        "product_type": "cylinder_ceramic",
        "print_area": {
            "quad": [[261, 260], [1320, 260], [1320, 1251], [261, 1251]],
            "mask_points": [[203, 205], [1379, 205], [1379, 1307], [203, 1307]],
        },
        "views": {
            "left": {
                "mockup_url": "/static/mockups/mug/mug-white-left-11oz.png",
                "print_area": {
                    "quad": [[261, 260], [1320, 260], [1320, 1251], [261, 1251]],
                    "mask_points": [[203, 205], [1379, 205], [1379, 1307], [203, 1307]],
                },
            },
            "right": {
                "mockup_url": "/static/mockups/mug/mug-white-right-11oz.png",
                "print_area": {
                    "quad": [[178, 179], [1238, 179], [1238, 1246], [178, 1246]],
                    "mask_points": [[120, 120], [1296, 120], [1296, 1305], [120, 1305]],
                },
            },
            "center": {
                "mockup_url": "/static/mockups/mug/mug_white_11oz_center.jpeg",
                "print_area": {
                    "quad": [[440, 280], [1560, 280], [1560, 1720], [440, 1720]],
                    "mask_points": [[440, 280], [1560, 280], [1560, 1720], [440, 1720]],
                },
            },
        },
        "warp": {
            "warp_type": "cylinder",
            "theta_max_deg": 52,
            "curve_pct": 58,
            "curve_top": 16,
            "curve_bottom": 6,
            "edge_squeeze": 0.15,
            "squeeze_power": 2,
            "center_focus_width": 0.0,
            "mesh_density_strength": 2,
        },
    },
    ("11oz", "black"): {
        "mockup_url": "/static/mockups/mug/mug-black-left-11oz.png",
        "product_type": "cylinder_ceramic",
        "print_area": {
            "quad": [[281, 263], [1321, 263], [1321, 1252], [281, 1252]],
            "mask_points": [[224, 209], [1379, 209], [1379, 1307], [224, 1307]],
        },
        "views": {
            "left": {
                "mockup_url": "/static/mockups/mug/mug-black-left-11oz.png",
                "print_area": {
                    "quad": [[281, 263], [1321, 263], [1321, 1252], [281, 1252]],
                    "mask_points": [[224, 209], [1379, 209], [1379, 1307], [224, 1307]],
                },
            },
            "right": {
                "mockup_url": "/static/mockups/mug/mug-black-right-11oz.png",
                "print_area": {
                    "quad": [[177, 267], [1217, 267], [1217, 1320], [177, 1320]],
                    "mask_points": [[120, 209], [1274, 209], [1274, 1378], [120, 1378]],
                },
            },
        },
        "warp": {
            "warp_type": "cylinder",
            "theta_max_deg": 52,
            "curve_pct": 58,
            "curve_top": 16,
            "curve_bottom": 6,
            "edge_squeeze": 0.15,
            "squeeze_power": 2,
            "center_focus_width": 0.0,
            "mesh_density_strength": 2,
        },
    },
    ("15oz", "white"): {
        "mockup_url": "/static/mockups/mug/mug-white-left-15oz.png",
        "product_type": "cylinder_ceramic",
        "print_area": {
            "quad": [[370, 254], [1321, 240], [1337, 1266], [385, 1280]],
            "mask_points": [[316, 198], [1374, 182], [1391, 1322], [333, 1338]],
        },
        "views": {
            "left": {
                "mockup_url": "/static/mockups/mug/mug-white-left-15oz.png",
                "print_area": {
                    "quad": [[370, 254], [1321, 240], [1337, 1266], [385, 1280]],
                    "mask_points": [[316, 198], [1374, 182], [1391, 1322], [333, 1338]],
                },
            },
            "right": {
                "mockup_url": "/static/mockups/mug/mug-white-right-15oz.png",
                "print_area": {
                    "quad": [[90, 318], [1194, 176], [1334, 1264], [230, 1406]],
                    "mask_points": [[21, 266], [1248, 108], [1403, 1316], [176, 1474]],
                },
            },
        },
        "warp": {
            "warp_type": "cylinder",
            "theta_max_deg": 54,
            "curve_pct": 60,
            "curve_top": 18,
            "curve_bottom": 7,
            "edge_squeeze": 0.18,
            "squeeze_power": 2,
            "center_focus_width": 0.0,
            "mesh_density_strength": 2,
        },
    },
    ("15oz", "black"): {
        "mockup_url": "/static/mockups/mug/mug-black-left-15oz.png",
        "product_type": "cylinder_ceramic",
        "print_area": {
            "quad": [[361, 260], [1320, 246], [1334, 1241], [375, 1255]],
            "mask_points": [[307, 205], [1373, 190], [1388, 1296], [323, 1311]],
        },
        "views": {
            "left": {
                "mockup_url": "/static/mockups/mug/mug-black-left-15oz.png",
                "print_area": {
                    "quad": [[361, 260], [1320, 246], [1334, 1241], [375, 1255]],
                    "mask_points": [[307, 205], [1373, 190], [1388, 1296], [323, 1311]],
                },
            },
            "right": {
                "mockup_url": "/static/mockups/mug/mug-black-right-15oz.png",
                "print_area": {
                    "quad": [[182, 259], [1316, 259], [1316, 1320], [182, 1320]],
                    "mask_points": [[120, 201], [1378, 201], [1378, 1378], [120, 1378]],
                },
            },
        },
        "warp": {
            "warp_type": "cylinder",
            "theta_max_deg": 54,
            "curve_pct": 60,
            "curve_top": 18,
            "curve_bottom": 7,
            "edge_squeeze": 0.18,
            "squeeze_power": 2,
            "center_focus_width": 0.0,
            "mesh_density_strength": 2,
        },
    },
}


@dataclass
class ParsedLiveviewUrl:
    source_url: str
    hostname: str
    resolution: Optional[str]
    slug: str
    extension: str
    template: str
    color_slug: Optional[str]
    design_slug: str
    color_hex: Optional[str]
    section: str

    def to_dict(self) -> dict:
        return asdict(self)


def _sanitize_filename(value: str) -> str:
    sanitized = SAFE_NAME_RE.sub("-", value).strip("-._")
    return sanitized or "asset"


def _hex_to_rgb(color_hex: str) -> tuple[int, int, int]:
    return tuple(int(color_hex[i : i + 2], 16) for i in range(0, 6, 2))


def _infer_color_variant(color_slug: Optional[str], color_hex: Optional[str]) -> str:
    if isinstance(color_slug, str):
        lowered = color_slug.strip().lower()
        if "black" in lowered:
            return "black"
        if "white" in lowered:
            return "white"

    if isinstance(color_hex, str) and HEX_COLOR_RE.fullmatch(color_hex):
        r, g, b = _hex_to_rgb(color_hex)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
        return "white" if luminance >= 0.55 else "black"

    return "white"


def _infer_mug_size(template_slug: str) -> str:
    lowered = template_slug.lower()
    if "15oz" in lowered:
        return "15oz"
    if "11oz" in lowered:
        return "11oz"
    return "11oz"


def _infer_product_type(template_slug: str) -> str:
    lowered = template_slug.lower()
    if lowered.startswith("mugs"):
        return "cylinder_ceramic"
    if lowered.startswith("hoodie"):
        return "apparel_hoodie"
    if lowered.startswith("totebag") or lowered.startswith("tote"):
        return "apparel_totebag"
    return "cylinder_ceramic"


def build_url_lookup_context(source_url: str) -> dict:
    parsed = parse_printerval_liveview_url(source_url)
    color_variant = _infer_color_variant(parsed.color_slug, parsed.color_hex)
    mug_size = _infer_mug_size(parsed.template)
    product_type = _infer_product_type(parsed.template)
    return {
        "parsed": parsed,
        "color_variant": color_variant,
        "mug_size": mug_size,
        "product_type": product_type,
        "mockup_family_key": f"{mug_size}:{color_variant}",
        "design_lookup_key": f"{parsed.template}|{color_variant}|{parsed.design_slug}",
    }


def _get_policy_views(policy: dict) -> dict[str, dict]:
    views = policy.get("views")
    if isinstance(views, dict):
        normalized: dict[str, dict] = {}
        for key, value in views.items():
            if isinstance(key, str) and isinstance(value, dict):
                normalized[key.lower()] = value
        if normalized:
            return normalized

    fallback_mockup_url = str(policy.get("mockup_url", "")).strip()
    fallback_print_area = policy.get("print_area")
    if fallback_mockup_url:
        return {
            "right": {
                "mockup_url": fallback_mockup_url,
                "print_area": fallback_print_area,
            }
        }
    return {}


def list_available_mockup_views(parsed: ParsedLiveviewUrl) -> list[dict]:
    color_variant = _infer_color_variant(parsed.color_slug, parsed.color_hex)
    mug_size = _infer_mug_size(parsed.template)
    policy = MUG_MOCKUP_POLICIES.get((mug_size, color_variant), {})
    views = _get_policy_views(policy)

    available: list[dict] = []
    for view_name in DEFAULT_MOCKUP_VIEW_ORDER:
        view = views.get(view_name)
        if not isinstance(view, dict):
            continue
        mockup_url = str(view.get("mockup_url", "")).strip()
        if not mockup_url:
            continue
        if resolve_local_asset_path(mockup_url).exists():
            available.append(
                {
                    "view": view_name,
                    "mockup_url": mockup_url,
                    "print_area": view.get("print_area"),
                    "product_type": policy.get("product_type", _infer_product_type(parsed.template)),
                }
            )

    if available:
        return available

    mockup_dir = PUBLIC_DIR / "mockups" / "mug"
    fallback_patterns = [
        ("left", f"mug-{color_variant}-left-{mug_size}.png"),
        ("right", f"mug-{color_variant}-right-{mug_size}.png"),
        ("center", f"mug_{color_variant}_{mug_size}_center.jpeg"),
        ("center", f"mug_{color_variant}_{mug_size}_center.jpg"),
        ("center", f"mug_{color_variant}_{mug_size}_center.png"),
    ]
    seen_views: set[str] = set()
    for view_name, candidate in fallback_patterns:
        path = mockup_dir / candidate
        if path.exists() and view_name not in seen_views:
            available.append(
                {
                    "view": view_name,
                    "mockup_url": f"/static/mockups/mug/{candidate}",
                    "print_area": None,
                    "product_type": _infer_product_type(parsed.template),
                }
            )
            seen_views.add(view_name)

    return available


def resolve_public_mockup_variant(parsed: ParsedLiveviewUrl, preferred_view: Optional[str] = None) -> dict:
    available_views = list_available_mockup_views(parsed)
    if not available_views:
        color_variant = _infer_color_variant(parsed.color_slug, parsed.color_hex)
        mug_size = _infer_mug_size(parsed.template)
        raise FileNotFoundError(
            f"KhÃƒÂ´ng tÃƒÂ¬m thÃ¡ÂºÂ¥y mockup local cho template={parsed.template}, color={color_variant}, size={mug_size}"
        )

    normalized_preferred = str(preferred_view or "").strip().lower()
    if normalized_preferred:
        for item in available_views:
            if item["view"] == normalized_preferred:
                return item

    return available_views[0]


def parse_printerval_liveview_url(source_url: str) -> ParsedLiveviewUrl:
    parsed = urlparse(source_url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL phÃ¡ÂºÂ£i bÃ¡ÂºÂ¯t Ã„â€˜Ã¡ÂºÂ§u bÃ¡ÂºÂ±ng http:// hoÃ¡ÂºÂ·c https://")

    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2 or path_parts[0] != "image":
        raise ValueError("URL khÃƒÂ´ng Ã„â€˜ÃƒÂºng format /image/[resolution]/[slug].[ext]")

    resolution = None
    file_part = unquote(path_parts[-1])
    if len(path_parts) >= 3 and RESOLUTION_RE.fullmatch(path_parts[-2]):
        resolution = path_parts[-2]

    if "." not in file_part:
        raise ValueError("URL khÃƒÂ´ng cÃƒÂ³ phÃ¡ÂºÂ§n mÃ¡Â»Å¸ rÃ¡Â»â„¢ng Ã¡ÂºÂ£nh hÃ¡Â»Â£p lÃ¡Â»â€¡")

    slug, extension = file_part.rsplit(".", 1)
    slug_parts = [part.strip() for part in slug.split(",") if part.strip()]
    if len(slug_parts) == 4:
        template_slug, color_slug, design_slug, color_hex = slug_parts
    elif len(slug_parts) == 3:
        template_slug, design_slug, color_hex = slug_parts
        color_slug = None
    else:
        raise ValueError("Slug URL phÃ¡ÂºÂ£i cÃƒÂ³ dÃ¡ÂºÂ¡ng 3 hoÃ¡ÂºÂ·c 4 phÃ¡ÂºÂ§n ngÃ„Æ’n bÃ¡Â»Å¸i dÃ¡ÂºÂ¥u phÃ¡ÂºÂ©y")

    section = design_slug.split("-", 1)[0].lower()
    return ParsedLiveviewUrl(
        source_url=source_url,
        hostname=parsed.netloc.lower(),
        resolution=resolution,
        slug=slug,
        extension=extension.lower(),
        template=template_slug,
        color_slug=color_slug,
        design_slug=design_slug,
        color_hex=color_hex.lower() if isinstance(color_hex, str) else None,
        section=section,
    )


def _build_design_source_url(parsed: ParsedLiveviewUrl) -> tuple[Optional[str], Optional[str]]:
    mapping_raw = os.getenv("PRINTERVAL_SECTION_URL_TEMPLATES", "").strip()
    mapping: dict[str, str] = dict(DEFAULT_SECTION_URL_TEMPLATES)
    configured_sections: set[str] = set()
    if mapping_raw:
        try:
            loaded = json.loads(mapping_raw)
            if isinstance(loaded, dict):
                normalized_loaded = {str(key).lower(): str(value) for key, value in loaded.items()}
                mapping.update(normalized_loaded)
                configured_sections.update(normalized_loaded.keys())
        except json.JSONDecodeError:
            pass

    explicit_section_template = os.getenv(
        f"PRINTERVAL_{parsed.section.upper()}_URL_TEMPLATE",
        "",
    ).strip()
    section_template = explicit_section_template or mapping.get(parsed.section)
    if not section_template:
        return None, None

    mode = "configured_template" if explicit_section_template or parsed.section in configured_sections else "builtin_section_rule"

    remainder = parsed.design_slug
    prefix = f"{parsed.section}-"
    if remainder.startswith(prefix):
        remainder = remainder[len(prefix) :]

    split_positions = [pos for pos in (remainder.find("_"), remainder.find("+"), remainder.find(" ")) if pos > 0]
    split_at = min(split_positions) if split_positions else -1
    if split_at > 0:
        folder = remainder[:split_at]
        asset_name = remainder[split_at + 1 :]
    else:
        folder = remainder
        asset_name = remainder

    folder_path = folder.replace("-", "/")
    random_str = hashlib.sha1(parsed.source_url.encode("utf-8")).hexdigest()[:12]
    return (
        section_template.format(
            section=parsed.section,
            design_slug=parsed.design_slug,
            remainder=remainder,
            folder=folder,
            folder_path=folder_path,
            name=asset_name,
            base_color=parsed.color_hex or (parsed.color_slug or ""),
            random_str=random_str,
        ),
        mode,
    )


def resolve_public_mockup_url(parsed: ParsedLiveviewUrl, preferred_view: Optional[str] = None) -> str:
    return str(resolve_public_mockup_variant(parsed, preferred_view=preferred_view).get("mockup_url"))


def resolve_local_asset_path(raw_path: str) -> Path:
    for prefix, base_dir in STATIC_ASSET_PREFIXES:
        if raw_path.startswith(prefix):
            relative = raw_path[len(prefix) :].replace("/", os.sep)
            return (base_dir / relative).resolve()

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return path.resolve()


def is_remote_http_url(raw_path: Optional[str]) -> bool:
    if not isinstance(raw_path, str):
        return False
    normalized = raw_path.strip()
    if not normalized:
        return False
    parsed = urlparse(normalized)
    return parsed.scheme.lower() in {"http", "https"} and bool(parsed.netloc)


def _infer_remote_preferred_name(remote_url: str, fallback: str = "asset") -> str:
    parsed = urlparse(remote_url)
    stem = Path(unquote(parsed.path)).stem.strip()
    if not stem:
        stem = fallback
    sanitized = _sanitize_filename(stem)
    if len(sanitized) > 120:
        sanitized = sanitized[:120].rstrip("-._")
    return sanitized or fallback


async def resolve_or_download_remote_asset(
    raw_path: str,
    *,
    target_dir: Path,
    preferred_name: Optional[str] = None,
    reuse_local: bool = True,
) -> Path:
    normalized = str(raw_path or "").strip()
    if not normalized:
        raise ValueError("asset_url_empty")

    if not is_remote_http_url(normalized):
        return resolve_local_asset_path(normalized)

    target_dir = target_dir.resolve()
    resolved_preferred_name = _sanitize_filename(
        preferred_name or _infer_remote_preferred_name(normalized)
    )

    if reuse_local:
        cached = _find_cached_downloaded_asset(
            normalized,
            target_dir,
            preferred_name=resolved_preferred_name,
        )
        if cached is not None:
            return cached

    return await download_remote_asset_async(
        normalized,
        target_dir,
        preferred_name=resolved_preferred_name,
    )


def _build_default_print_area(width: int, height: int, product_type: str) -> dict:
    if str(product_type).startswith("cylinder"):
        top_left = [int(width * 0.22), int(height * 0.14)]
        top_right = [int(width * 0.78), int(height * 0.14)]
        bottom_right = [int(width * 0.78), int(height * 0.86)]
        bottom_left = [int(width * 0.22), int(height * 0.86)]
    else:
        top_left = [int(width * 0.24), int(height * 0.20)]
        top_right = [int(width * 0.76), int(height * 0.20)]
        bottom_right = [int(width * 0.76), int(height * 0.80)]
        bottom_left = [int(width * 0.24), int(height * 0.80)]

    quad = [top_left, top_right, bottom_right, bottom_left]
    return {
        "top_left": top_left,
        "top_right": top_right,
        "bottom_right": bottom_right,
        "bottom_left": bottom_left,
        "quad": quad,
        "base_points_raw": quad,
        "mask_points": quad,
        "product_type": product_type,
    }


def _build_policy_print_area(policy: dict, product_type: str, preferred_view: Optional[str] = None) -> Optional[dict]:
    view_name = str(preferred_view or "").strip().lower()
    view_config = _get_policy_views(policy).get(view_name) if view_name else None
    print_area_source = view_config.get("print_area") if isinstance(view_config, dict) else policy.get("print_area")
    quad = print_area_source.get("quad") if isinstance(print_area_source, dict) else None
    if not isinstance(quad, list) or len(quad) != 4:
        return None

    mask_points = print_area_source.get("mask_points") if isinstance(print_area_source, dict) else None
    if not isinstance(mask_points, list) or len(mask_points) < 4:
        mask_points = quad
    return {
        "top_left": quad[0],
        "top_right": quad[1],
        "bottom_right": quad[2],
        "bottom_left": quad[3],
        "quad": quad,
        "base_points_raw": quad,
        "mask_points": mask_points,
        "product_type": product_type,
    }


def _detect_print_area_preset(mockup_static_url: str, product_type: str) -> tuple[dict, float, str]:
    path = resolve_local_asset_path(mockup_static_url)
    try:
        import cv2  # type: ignore
        from app.services.vision import auto_detect_print_area_v2
    except Exception:
        cv2 = None
        auto_detect_print_area_v2 = None

    if cv2 is None or auto_detect_print_area_v2 is None:
        return _build_default_print_area(1500, 1500, product_type), 0.0, "default"

    image = cv2.imread(str(path))
    if image is None:
        return _build_default_print_area(1500, 1500, product_type), 0.0, "default"

    height, width = image.shape[:2]
    detected = auto_detect_print_area_v2(image)
    confidence = float(detected.get("confidence", 0.0))
    quad = detected.get("quad")
    clip_mask = detected.get("clip_mask") or quad
    if isinstance(quad, list) and len(quad) == 4:
        return {
            "top_left": quad[0],
            "top_right": quad[1],
            "bottom_right": quad[2],
            "bottom_left": quad[3],
            "quad": quad,
            "base_points_raw": quad,
            "mask_points": clip_mask,
            "product_type": product_type,
        }, confidence, "auto_detect"

    return _build_default_print_area(width, height, product_type), confidence, "default"


def _build_warp_preset(parsed: ParsedLiveviewUrl, color_variant: str) -> dict:
    mug_size = _infer_mug_size(parsed.template)
    policy = MUG_MOCKUP_POLICIES.get((mug_size, color_variant), {})
    warp = dict(policy.get("warp") or {})
    if not warp:
        warp = {
            "warp_type": "cylinder",
            "theta_max_deg": 52 if mug_size == "11oz" else 54,
            "curve_pct": 58 if mug_size == "11oz" else 60,
            "curve_top": 16 if mug_size == "11oz" else 18,
            "curve_bottom": 6 if mug_size == "11oz" else 7,
            "edge_squeeze": 0.15 if mug_size == "11oz" else 0.18,
            "squeeze_power": 2,
            "center_focus_width": 0.0,
            "mesh_density_strength": 2,
        }

    warp.setdefault("product_type", _infer_product_type(parsed.template))
    warp.setdefault("light_pos_x", 0.62)
    warp.setdefault("light_pos_y", 0.32)
    warp.setdefault("light_height", 55)
    warp.setdefault("light_contrast", 50)
    warp.setdefault("light_highlight", 60)
    warp.setdefault("light_softness", 55)
    warp.setdefault("feather_radius", 3)
    return warp


def _guess_extension(remote_url: str, content_type: str) -> str:
    parsed = urlparse(remote_url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip()) if content_type else None
    return guessed or ".png"


def _build_cached_filename(preferred_name: str, remote_url: str, suffix: str) -> str:
    safe_suffix = suffix if str(suffix).startswith(".") else f".{suffix}"
    safe_preferred_name = _sanitize_filename(preferred_name)
    url_hash = hashlib.sha1(remote_url.encode("utf-8")).hexdigest()[:12]
    return f"{safe_preferred_name}-{url_hash}{safe_suffix.lower()}"


async def _get_async_http_client() -> Optional["httpx.AsyncClient"]:
    global _HTTPX_CLIENT
    if httpx is None:
        return None
    if _HTTPX_CLIENT is not None:
        return _HTTPX_CLIENT

    async with _HTTPX_CLIENT_LOCK:
        if _HTTPX_CLIENT is not None:
            return _HTTPX_CLIENT
            
        # Optimization: use pre-configured singleton for high-volume concurrent downloads
        _HTTPX_CLIENT = httpx.AsyncClient(
            http2=_HTTP2_AVAILABLE,
            limits=httpx.Limits(
                max_connections=100, 
                max_keepalive_connections=20,
                keepalive_expiry=30.0
            ),
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0),
            follow_redirects=True,
            headers={
                "Accept-Encoding": "gzip, br",
                "Connection": "keep-alive"
            }
        )
        if not _HTTP2_AVAILABLE:
            logger.warning("HTTP/2 disabled for URL downloads because package 'h2' is not installed.")
        return _HTTPX_CLIENT
async def close_async_http_client() -> None:
    global _HTTPX_CLIENT
    if _HTTPX_CLIENT is None:
        return
    try:
        await _HTTPX_CLIENT.aclose()
    except Exception:
        logger.warning("Failed to close async HTTP client", exc_info=True)
    finally:
        _HTTPX_CLIENT = None


def download_remote_asset(remote_url: str, target_dir: Path, preferred_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    request = Request(
        remote_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/*,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=30) as response:
        content = response.read()
        content_type = response.headers.get("Content-Type", "")

    if not content:
        raise ValueError(f"URL khÃƒÂ´ng trÃ¡ÂºÂ£ vÃ¡Â»Â dÃ¡Â»Â¯ liÃ¡Â»â€¡u Ã¡ÂºÂ£nh: {remote_url}")

    suffix = _guess_extension(remote_url, content_type)
    file_name = _build_cached_filename(preferred_name, remote_url, suffix)
    destination = target_dir / file_name
    destination.write_bytes(content)
    return destination


async def download_remote_asset_async(remote_url: str, target_dir: Path, preferred_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    client = await _get_async_http_client()

    if client is None:
        return await asyncio.to_thread(download_remote_asset, remote_url, target_dir, preferred_name)

    try:
        response = await client.get(
            remote_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "image/*,*/*;q=0.8",
            },
        )
    except Exception as exc:
        if "h2" in str(exc).lower() and "http2" in str(exc).lower():
            logger.warning("HTTP/2 transport unavailable, fallback to urllib downloader for %s", remote_url)
            return await asyncio.to_thread(download_remote_asset, remote_url, target_dir, preferred_name)
        raise
    response.raise_for_status()
    content = response.content
    content_type = response.headers.get("Content-Type", "")

    if not content:
        raise ValueError(f"URL khÃƒÂ´ng trÃ¡ÂºÂ£ vÃ¡Â»Â dÃ¡Â»Â¯ liÃ¡Â»â€¡u Ã¡ÂºÂ£nh: {remote_url}")

    suffix = _guess_extension(remote_url, content_type)
    file_name = _build_cached_filename(preferred_name, remote_url, suffix)
    destination = target_dir / file_name
    await asyncio.to_thread(destination.write_bytes, content)
    return destination


def _find_cached_downloaded_asset(remote_url: str, target_dir: Path, preferred_name: str) -> Optional[Path]:
    if not remote_url:
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_preferred_name = _sanitize_filename(preferred_name)
    url_hash = hashlib.sha1(remote_url.encode("utf-8")).hexdigest()[:12]
    file_prefix = f"{safe_preferred_name}-{url_hash}"

    # Fast path: deterministic direct-file existence checks.
    for suffix in _COMMON_IMAGE_SUFFIXES:
        candidate = target_dir / f"{file_prefix}{suffix}"
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate

    # Fallback: preserve compatibility for uncommon extensions.
    for candidate in target_dir.glob(f"{file_prefix}.*"):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def check_local_asset_exists(raw_path: Optional[str]) -> bool:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False
    try:
        path = resolve_local_asset_path(raw_path.strip())
    except Exception:
        return False
    return path.exists() and path.is_file() and path.stat().st_size > 0


def analyze_and_ingest_url(
    source_url: str,
    preferred_view: Optional[str] = None,
    reuse_local_artwork: bool = True,
) -> dict:
    context = build_url_lookup_context(source_url)
    parsed = context["parsed"]
    color_variant = context["color_variant"]
    mug_size = context["mug_size"]
    product_type = context["product_type"]
    policy = MUG_MOCKUP_POLICIES.get((mug_size, color_variant), {})
    mockup_variant = resolve_public_mockup_variant(parsed, preferred_view=preferred_view)
    mockup_url = str(mockup_variant["mockup_url"])
    selected_view = str(mockup_variant["view"])
    print_area_preset, print_area_confidence, print_area_source = _detect_print_area_preset(
        mockup_url,
        product_type,
    )
    if print_area_source == "default":
        policy_print_area = _build_policy_print_area(policy, product_type, preferred_view=selected_view)
        if policy_print_area:
            print_area_preset = policy_print_area
            print_area_confidence = 0.98
            print_area_source = "policy"
    warp_preset = _build_warp_preset(parsed, color_variant)

    design_source_url, design_source_mode = _build_design_source_url(parsed)
    warning: Optional[str] = None
    if not design_source_url:
        design_source_url = source_url
        design_source_mode = "original_url_fallback"
        warning = (
            "ChÃ†Â°a cÃ¡ÂºÂ¥u hÃƒÂ¬nh PRINTERVAL_SECTION_URL_TEMPLATES nÃƒÂªn hÃ¡Â»â€¡ thÃ¡Â»â€˜ng Ã„â€˜ang tÃ¡ÂºÂ£i chÃƒÂ­nh URL CDN gÃ¡Â»â€˜c vÃ¡Â»Â local lÃƒÂ m artwork fallback."
        )

    preferred_name = f"{parsed.template}-{parsed.design_slug}"
    artwork_dir = INPUTS_DIR / "artworks"
    downloaded: Optional[Path] = None

    if reuse_local_artwork:
        downloaded = _find_cached_downloaded_asset(
            design_source_url,
            artwork_dir,
            preferred_name=preferred_name,
        )
        if downloaded is not None:
            design_source_mode = f"{design_source_mode}_local_cache"

    if downloaded is None:
        try:
            downloaded = download_remote_asset(
                design_source_url,
                artwork_dir,
                preferred_name=preferred_name,
            )
        except Exception:
            if design_source_mode == "configured_template":
                design_source_mode = "original_url_fallback"
                warning = (
                    "KhÃƒÂ´ng tÃ¡ÂºÂ£i Ã„â€˜Ã†Â°Ã¡Â»Â£c design source URL Ã„â€˜ÃƒÂ£ cÃ¡ÂºÂ¥u hÃƒÂ¬nh, hÃ¡Â»â€¡ thÃ¡Â»â€˜ng fallback sang chÃƒÂ­nh URL CDN gÃ¡Â»â€˜c."
                )
                if reuse_local_artwork:
                    downloaded = _find_cached_downloaded_asset(
                        parsed.source_url,
                        artwork_dir,
                        preferred_name=preferred_name,
                    )
                    if downloaded is not None:
                        design_source_mode = "original_url_fallback_local_cache"
                if downloaded is None:
                    downloaded = download_remote_asset(
                        parsed.source_url,
                        artwork_dir,
                        preferred_name=preferred_name,
                    )
                design_source_url = parsed.source_url
            else:
                raise

    return {
        "source_url": parsed.source_url,
        "parsed": parsed.to_dict(),
        "product_type": product_type,
        "mockup_url": mockup_url,
        "mockup_policy_key": f"{mug_size}:{color_variant}",
        "mockup_family_key": context["mockup_family_key"],
        "design_lookup_key": context["design_lookup_key"],
        "mockup_view": selected_view,
        "available_views": [item["view"] for item in list_available_mockup_views(parsed)],
        "design_url": f"/static/artworks/{downloaded.name}",
        "design_source_url": design_source_url,
        "design_source_mode": design_source_mode,
        "print_area_preset": print_area_preset,
        "print_area_preset_confidence": print_area_confidence,
        "print_area_preset_source": print_area_source,
        "warp_config_preset": warp_preset,
        "warning": warning,
    }


async def analyze_and_ingest_url_async(
    source_url: str,
    preferred_view: Optional[str] = None,
    reuse_local_artwork: bool = True,
    include_presets: bool = True,
) -> dict:
    context = build_url_lookup_context(source_url)
    parsed = context["parsed"]
    color_variant = context["color_variant"]
    mug_size = context["mug_size"]
    product_type = context["product_type"]
    policy = MUG_MOCKUP_POLICIES.get((mug_size, color_variant), {})
    mockup_variant = resolve_public_mockup_variant(parsed, preferred_view=preferred_view)
    mockup_url = str(mockup_variant["mockup_url"])
    selected_view = str(mockup_variant["view"])

    detect_task = None
    warp_preset: Optional[dict] = None
    if include_presets:
        # Run print-area detection in parallel with network download.
        detect_task = asyncio.create_task(
            asyncio.to_thread(_detect_print_area_preset, mockup_url, product_type)
        )
        warp_preset = _build_warp_preset(parsed, color_variant)

    design_source_url, design_source_mode = _build_design_source_url(parsed)
    warning: Optional[str] = None
    if not design_source_url:
        design_source_url = source_url
        design_source_mode = "original_url_fallback"
        warning = (
            "PRINTERVAL_SECTION_URL_TEMPLATES is not configured; fallback to downloading from original source URL."
        )

    preferred_name = f"{parsed.template}-{parsed.design_slug}"
    artwork_dir = INPUTS_DIR / "artworks"
    downloaded: Optional[Path] = None

    if reuse_local_artwork:
        downloaded = _find_cached_downloaded_asset(
            design_source_url,
            artwork_dir,
            preferred_name=preferred_name,
        )
        if downloaded is not None:
            design_source_mode = f"{design_source_mode}_local_cache"

    try:
        if downloaded is None:
            try:
                downloaded = await download_remote_asset_async(
                    design_source_url,
                    artwork_dir,
                    preferred_name=preferred_name,
                )
            except Exception:
                if design_source_mode == "configured_template":
                    design_source_mode = "original_url_fallback"
                    warning = "Configured design source failed; fallback to original URL."
                    if reuse_local_artwork:
                        downloaded = _find_cached_downloaded_asset(
                            parsed.source_url,
                            artwork_dir,
                            preferred_name=preferred_name,
                        )
                        if downloaded is not None:
                            design_source_mode = "original_url_fallback_local_cache"
                    if downloaded is None:
                        downloaded = await download_remote_asset_async(
                            parsed.source_url,
                            artwork_dir,
                            preferred_name=preferred_name,
                        )
                    design_source_url = parsed.source_url
                else:
                    raise
    except Exception:
        if detect_task is not None and not detect_task.done():
            detect_task.cancel()
            with suppress(BaseException):
                await detect_task
        raise

    print_area_preset: Optional[dict] = None
    print_area_confidence: Optional[float] = None
    print_area_preset_source: Optional[str] = None
    if include_presets and detect_task is not None:
        print_area_preset, print_area_confidence, print_area_preset_source = await detect_task
        if print_area_preset_source == "default":
            policy_print_area = _build_policy_print_area(policy, product_type, preferred_view=selected_view)
            if policy_print_area:
                print_area_preset = policy_print_area
                print_area_confidence = 0.98
                print_area_preset_source = "policy"

    return {
        "source_url": parsed.source_url,
        "parsed": parsed.to_dict(),
        "product_type": product_type,
        "mockup_url": mockup_url,
        "mockup_policy_key": f"{mug_size}:{color_variant}",
        "mockup_family_key": context["mockup_family_key"],
        "design_lookup_key": context["design_lookup_key"],
        "mockup_view": selected_view,
        "available_views": [item["view"] for item in list_available_mockup_views(parsed)],
        "design_url": f"/static/artworks/{downloaded.name}",
        "design_source_url": design_source_url,
        "design_source_mode": design_source_mode,
        "print_area_preset": print_area_preset,
        "print_area_preset_confidence": print_area_confidence,
        "print_area_preset_source": print_area_preset_source,
        "warp_config_preset": warp_preset,
        "warning": warning,
    }
