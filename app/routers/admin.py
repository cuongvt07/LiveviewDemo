# app/routers/admin.py

import os
import base64
import uuid
import json
import copy
import shutil
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from fastapi import (
    APIRouter,
    Request,
    Response,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Header,
)
from sqlalchemy import select

from app.db.database import async_session, getSession
from app.db.models import Template, TemplateConfigHistory, RenderedResult
from app.schemas import (
    TemplateCreate,
    TemplateConfigUpdate,
    LibraryItem,
    TemplateSaveAdhoc,
    UrlAnalysisImportRequest,
)
from app.services import template_registry
from app.services.url_analysis import (
    DEFAULT_MOCKUP_VIEW_ORDER,
    analyze_and_ingest_url_async,
    build_url_lookup_context,
    check_local_asset_exists,
    list_available_mockup_views,
    resolve_local_asset_path,
)

router = APIRouter(tags=['admin'])
logger = logging.getLogger('mockup_service')

ASSET_BASE_DIR = os.getenv('ASSET_BASE_DIR', './templates')
ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', 'change-me-in-production')


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default


URL_ANALYSIS_PREVIEW_WIDTH = max(1, _read_int_env('URL_ANALYSIS_PREVIEW_WIDTH', 960))
URL_ANALYSIS_PREVIEW_HEIGHT = max(1, _read_int_env('URL_ANALYSIS_PREVIEW_HEIGHT', 960))
URL_ANALYSIS_PREVIEW_QUALITY = max(1, min(100, _read_int_env('URL_ANALYSIS_PREVIEW_QUALITY', 90)))
URL_ANALYSIS_PREVIEW_FIT = str(os.getenv('URL_ANALYSIS_PREVIEW_FIT', 'contain') or 'contain').strip().lower()
URL_ANALYSIS_RENDER_QUALITY = max(1, min(100, _read_int_env('URL_ANALYSIS_RENDER_QUALITY', 100)))


# NOTE: admin auth removed per request — endpoints are now public in this development instance.


def _normalize_saved_template_config(
    raw_config: dict,
    output_width: int,
    output_height: int,
    product_type_hint: str,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> dict:
    from app.routers.render import _build_default_adhoc_config, _merge_adhoc_user_config

    payload = copy.deepcopy(raw_config) if isinstance(raw_config, dict) else {}
    if not isinstance(payload.get('print_area'), dict):
        payload['print_area'] = {}
    if not isinstance(payload.get('warp'), dict):
        payload['warp'] = {}

    if abs(scale_x - 1.0) > 1e-8 or abs(scale_y - 1.0) > 1e-8:
        pa = payload.get('print_area', {})

        def _scale_point(point):
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                return [float(point[0]) * scale_x, float(point[1]) * scale_y]
            return point

        for pt_key in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            if pt_key in pa:
                pa[pt_key] = _scale_point(pa[pt_key])

        for list_key in ['mask_points', 'mesh_control_src', 'mesh_control_dst', 'quad', 'base_points_raw']:
            if list_key in pa and isinstance(pa[list_key], list):
                pa[list_key] = [_scale_point(p) for p in pa[list_key]]

    # If frontend provided mesh.points (normalized [0..1]), convert to
    # print_area.mesh_control_dst so downstream normalization/merge handles it.
    try:
        mesh = payload.get('mesh')
        if isinstance(mesh, dict) and isinstance(mesh.get('points'), list) and len(mesh['points']) > 0:
            pts = []
            for p in mesh['points']:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    x = float(p[0])
                    y = float(p[1])
                    # If values look normalized (<=1.5), scale to pixel coords
                    if abs(x) <= 1.5 and abs(y) <= 1.5:
                        pts.append([x * output_width, y * output_height])
                    else:
                        pts.append([x, y])
            if pts:
                payload.setdefault('print_area', {})
                # store as mesh_control_dst by default
                payload['print_area']['mesh_control_dst'] = pts
    except Exception:
        # Best-effort conversion; ignore on failure
        pass

    payload.setdefault('product_type', product_type_hint)
    payload['print_area'].setdefault('product_type', product_type_hint)
    payload['warp'].setdefault('product_type', product_type_hint)

    default_config = _build_default_adhoc_config(output_width, output_height)
    merged = _merge_adhoc_user_config(default_config, json.dumps(payload))
    if isinstance(payload.get('url_analysis'), dict):
        merged['url_analysis'] = copy.deepcopy(payload['url_analysis'])
    return merged


def _template_preview_url(slug: str) -> str:
    template_dir = Path(ASSET_BASE_DIR) / slug
    preview_path = template_dir / 'preview.jpg'
    if preview_path.exists():
        return f'/static/templates/{slug}/preview.jpg'
    legacy_preview_path = template_dir / 'preview.png'
    if legacy_preview_path.exists():
        return f'/static/templates/{slug}/preview.png'
    return f'/static/templates/{slug}/mockup.jpg'


def _normalize_preview_fit(raw_fit: str | None) -> str:
    fit = str(raw_fit or 'contain').strip().lower()
    return fit if fit in {'cover', 'contain'} else 'contain'


def _template_preview_source_path(slug: str) -> Path | None:
    template_dir = Path(ASSET_BASE_DIR) / slug
    for filename in (
        'preview.jpg',
        'preview.jpeg',
        'preview.png',
        'mockup.jpg',
        'mockup.jpeg',
        'mockup.png',
        'mockup.webp',
    ):
        candidate = template_dir / filename
        if candidate.exists():
            return candidate
    return None


def _template_preview_image_url(
    slug: str,
    width: int | None = None,
    height: int | None = None,
    fit: str | None = None,
    quality: int | None = None,
) -> str:
    query: dict[str, str | int] = {}
    if width is not None:
        query['width'] = max(1, int(width))
    if height is not None:
        query['height'] = max(1, int(height))
    if quality is not None:
        query['quality'] = max(1, min(100, int(quality)))
    query['fit'] = _normalize_preview_fit(fit or URL_ANALYSIS_PREVIEW_FIT)
    base_path = f'/admin/templates/{slug}/preview-image'
    return f'{base_path}?{urlencode(query)}'


def _to_full_url(path: str, request: Request | None = None) -> str:
    if not path:
        return path
    if path.startswith('http://') or path.startswith('https://'):
        return path
    if request is not None:
        return f"{str(request.base_url).rstrip('/')}{path}"
    public_base = str(os.getenv('PUBLIC_BASE_URL', '') or '').strip().rstrip('/')
    if public_base:
        return f'{public_base}{path}'
    return path


def _extract_template_url_analysis_meta(template: Template) -> dict:
    cfg = template.config if isinstance(template.config, dict) else {}
    meta = cfg.get('url_analysis')
    return meta if isinstance(meta, dict) else {}


def _template_url_analysis_json_expr(meta_key: str):
    return Template.config['url_analysis'][meta_key].as_string()


async def _query_active_templates_by_url_analysis_meta(meta_key: str, meta_value: str) -> list[Template]:
    if not meta_value:
        return []
    async with async_session() as session:
        try:
            stmt = select(Template).where(
                Template.status == 'active',
                _template_url_analysis_json_expr(meta_key) == meta_value,
            )
            result = await session.execute(stmt)
            return result.scalars().all()
        except Exception:
            # Fallback for non-PostgreSQL dev environments.
            result = await session.execute(select(Template).where(Template.status == 'active'))
            active_templates = result.scalars().all()
            return [
                template
                for template in active_templates
                if _extract_template_url_analysis_meta(template).get(meta_key) == meta_value
            ]


def _build_lookup_items(
    templates: list[Template],
    request: Request,
    preview_width: int,
    preview_height: int,
    preview_fit: str,
    preview_quality: int,
) -> list[dict]:
    return _sort_template_lookup_items(
        [
            _build_template_lookup_item(
                template,
                request,
                preview_width=preview_width,
                preview_height=preview_height,
                preview_fit=preview_fit,
                preview_quality=preview_quality,
            )
            for template in templates
        ]
    )


async def _update_templates_design_meta(
    templates: list[Template],
    design_url: str,
    design_source_url: str | None,
    design_source_mode: str | None,
) -> None:
    if not templates or not design_url:
        return
    ids = [template.id for template in templates if getattr(template, 'id', None) is not None]
    if not ids:
        return
    async with async_session() as session:
        result = await session.execute(select(Template).where(Template.id.in_(ids)))
        rows = result.scalars().all()
        for row in rows:
            config = copy.deepcopy(row.config) if isinstance(row.config, dict) else {}
            meta = config.get('url_analysis')
            if not isinstance(meta, dict):
                meta = {}
                config['url_analysis'] = meta
            meta['design_url'] = design_url
            if design_source_url is not None:
                meta['design_source_url'] = design_source_url
            if design_source_mode is not None:
                meta['design_source_mode'] = design_source_mode
            row.config = config
            session.add(row)
        await session.commit()


def _build_template_lookup_item(
    template: Template,
    request: Request | None = None,
    preview_width: int | None = None,
    preview_height: int | None = None,
    preview_fit: str | None = None,
    preview_quality: int | None = None,
) -> dict:
    meta = _extract_template_url_analysis_meta(template)
    width = max(1, int(preview_width)) if preview_width is not None else URL_ANALYSIS_PREVIEW_WIDTH
    height = max(1, int(preview_height)) if preview_height is not None else URL_ANALYSIS_PREVIEW_HEIGHT
    quality = max(1, min(100, int(preview_quality))) if preview_quality is not None else URL_ANALYSIS_PREVIEW_QUALITY
    fit = _normalize_preview_fit(preview_fit or URL_ANALYSIS_PREVIEW_FIT)
    preview_url = _template_preview_image_url(
        template.slug,
        width=width,
        height=height,
        fit=fit,
        quality=quality,
    )
    return {
        'slug': template.slug,
        'name': template.name,
        'template_id': template.slug,
        'preview_url': preview_url,
        'preview_url_full': _to_full_url(preview_url, request),
        'mockup_view': meta.get('mockup_view'),
        'mockup_url': meta.get('mockup_url'),
        'design_url': meta.get('design_url'),
        'design_lookup_key': meta.get('design_lookup_key'),
        'mockup_family_key': meta.get('mockup_family_key'),
        'mockup_policy_key': meta.get('mockup_policy_key'),
    }


def _sort_template_lookup_items(items: list[dict]) -> list[dict]:
    view_order = {view: index for index, view in enumerate(DEFAULT_MOCKUP_VIEW_ORDER)}
    return sorted(
        items,
        key=lambda item: (
            view_order.get(str(item.get('mockup_view') or '').lower(), len(view_order)),
            str(item.get('slug') or ''),
        ),
    )


def _sort_templates_by_lookup_meta(templates: list[Template]) -> list[Template]:
    view_order = {view: index for index, view in enumerate(DEFAULT_MOCKUP_VIEW_ORDER)}
    return sorted(
        templates,
        key=lambda template: (
            view_order.get(
                str(_extract_template_url_analysis_meta(template).get('mockup_view') or '').lower(),
                len(view_order),
            ),
            str(template.slug or ''),
        ),
    )


async def _load_active_template_assets(slug: str):
    assets = template_registry.get(slug)
    if assets is not None:
        return assets

    async with async_session() as session:
        stmt = select(Template).where(Template.slug == slug, Template.status == 'active')
        result = await session.execute(stmt)
        template = result.scalar_one_or_none()

    if template is None:
        raise FileNotFoundError(f"Template '{slug}' not found or inactive")

    record = {
        'mockup_path': template.mockup_path,
        'shadow_map_path': template.shadow_map_path,
        'normal_map_path': template.normal_map_path,
        'specular_path': template.specular_path,
        'mask_path': template.mask_path,
        'config': template.config,
        'output_width': template.output_width,
        'output_height': template.output_height,
    }
    return template_registry.load_template(slug, record)


async def _find_cached_rendered_result_path(source_url: str) -> str | None:
    from app.services.render_persist import extractSlug

    slug = extractSlug(source_url)
    async with async_session() as session:
        stmt = (
            select(RenderedResult.image_path)
            .where(RenderedResult.url_slug == slug)
            .order_by(RenderedResult.created_at.desc(), RenderedResult.id.desc())
            .limit(1)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def _persist_rendered_result_and_get_path(source_url: str, image_bytes: bytes, output_format: str = 'jpg') -> str:
    from app.services.render_persist import buildRenderFilename, extractSlug, saveRenderedImage

    slug = extractSlug(source_url)
    filename = buildRenderFilename(slug, output_format)
    saveRenderedImage(image_bytes, filename)

    image_path = f'/static/renders/{filename}'
    async with async_session() as session:
        session.add(RenderedResult(url_slug=slug, image_path=image_path))
        await session.commit()
    return image_path


def _write_preview_data_url(
    preview_data_url: str,
    destination: Path,
    resize_to: tuple[int, int] | None = None,
    jpeg_quality: int = 90,
) -> None:
    if not isinstance(preview_data_url, str) or not preview_data_url.startswith('data:'):
        raise ValueError('preview_data_url invalid')

    header, _, payload = preview_data_url.partition(',')
    if ';base64' not in header or not payload:
        raise ValueError('preview_data_url must be base64 data URL')

    import cv2
    import numpy as np

    binary = base64.b64decode(payload)
    buf = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError('preview_data_url decode failed')

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0, dtype=np.float32)
        img = np.clip(rgb * alpha + white * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)

    if resize_to:
        target_w = max(1, int(resize_to[0]))
        target_h = max(1, int(resize_to[1]))
        if img.shape[1] != target_w or img.shape[0] != target_h:
            interpolation = cv2.INTER_AREA if (img.shape[1] > target_w or img.shape[0] > target_h) else cv2.INTER_LINEAR
            img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    ok = cv2.imwrite(
        str(destination),
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(1, min(100, jpeg_quality)))],
    )
    if not ok:
        raise ValueError('preview_data_url save failed')


@router.get('/templates/{slug}/preview-image')
async def getTemplatePreviewImage(
    slug: str,
    width: int | None = None,
    height: int | None = None,
    fit: str = URL_ANALYSIS_PREVIEW_FIT,
    quality: int = URL_ANALYSIS_PREVIEW_QUALITY,
):
    import cv2
    import numpy as np

    if '..' in Path(slug).parts:
        raise HTTPException(status_code=400, detail={'message': 'invalid_slug'})

    source_path = _template_preview_source_path(slug)
    if source_path is None:
        raise HTTPException(status_code=404, detail={'message': f"Template '{slug}' preview not found"})

    buf = np.frombuffer(source_path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=500, detail={'message': 'failed_to_decode_preview'})

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0, dtype=np.float32)
        image = np.clip(rgb * alpha + white * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)

    src_h, src_w = image.shape[:2]
    target_w = int(width) if width is not None else None
    target_h = int(height) if height is not None else None

    if target_w is None and target_h is None:
        target_w, target_h = src_w, src_h
    elif target_w is None:
        target_h = max(1, target_h)
        target_w = max(1, int(round(src_w * (target_h / max(src_h, 1)))))
    elif target_h is None:
        target_w = max(1, target_w)
        target_h = max(1, int(round(src_h * (target_w / max(src_w, 1)))))
    else:
        target_w = max(1, target_w)
        target_h = max(1, target_h)

    resized = image
    fit_mode = _normalize_preview_fit(fit)
    if target_w != src_w or target_h != src_h:
        if fit_mode == 'cover':
            scale = max(target_w / max(src_w, 1), target_h / max(src_h, 1))
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            interpolation = cv2.INTER_AREA if new_w < src_w or new_h < src_h else cv2.INTER_LINEAR
            expanded = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            start_x = max(0, (new_w - target_w) // 2)
            start_y = max(0, (new_h - target_h) // 2)
            resized = expanded[start_y:start_y + target_h, start_x:start_x + target_w]
        else:
            scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
            interpolation = cv2.INTER_AREA if new_w < src_w or new_h < src_h else cv2.INTER_LINEAR
            contained = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            resized = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2
            resized[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = contained

    encode_quality = max(1, min(100, int(quality)))
    ok, encoded = cv2.imencode(
        '.jpg',
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), encode_quality],
    )
    if not ok:
        raise HTTPException(status_code=500, detail={'message': 'failed_to_encode_preview'})

    return Response(
        content=encoded.tobytes(),
        media_type='image/jpeg',
        headers={'Cache-Control': 'public, max-age=300'},
    )


@router.post('/templates')
async def createTemplate(
    slug: str = Form(...),
    name: str = Form(...),
    description: str = Form(None),
    config: str = Form(...),
    output_width: int = Form(1500),
    output_height: int = Form(1500),
    mockup_file: UploadFile = File(...),
    mask_file: UploadFile = File(...),
    shadow_map_file: UploadFile = File(None),
    normal_map_file: UploadFile = File(None),
    specular_file: UploadFile = File(None),
):
    '''
    Tạo template mới. Upload mockup + mask (bắt buộc),
    shadow/normal/specular (tùy chọn).
    '''
    import json

    try:
        config_dict = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail={
            'error': 'invalid_config',
            'message': 'Config JSON không hợp lệ',
        })

    # Kiểm tra slug trùng
    async with async_session() as session:
        exists = await session.execute(
            select(Template).where(Template.slug == slug)
        )
        if exists.scalar_one_or_none():
            raise HTTPException(status_code=409, detail={
                'error': 'slug_exists',
                'message': f'Slug \'{slug}\' đã tồn tại',
            })

    # Lưu files
    template_dir = Path(ASSET_BASE_DIR) / slug
    maps_dir = template_dir / 'maps'
    maps_dir.mkdir(parents=True, exist_ok=True)

    async def saveFile(upload: UploadFile, dest: Path):
        content = await upload.read()
        dest.write_bytes(content)

    mockup_path = template_dir / 'mockup.jpg'
    mask_path = maps_dir / 'mask.jpg'

    await saveFile(mockup_file, mockup_path)
    await saveFile(mask_file, mask_path)

    shadow_path = None
    normal_path = None
    specular_path = None

    if shadow_map_file:
        shadow_path = maps_dir / 'shadow_map.png'
        await saveFile(shadow_map_file, shadow_path)

    if normal_map_file:
        normal_path = maps_dir / 'normal_map.png'
        await saveFile(normal_map_file, normal_path)

    if specular_file:
        specular_path = maps_dir / 'specular_map.png'
        await saveFile(specular_file, specular_path)

    # Tạo DB record
    template = Template(
        slug=slug,
        name=name,
        description=description,
        mockup_path=str(mockup_path),
        shadow_map_path=str(shadow_path) if shadow_path else None,
        normal_map_path=str(normal_path) if normal_path else None,
        specular_path=str(specular_path) if specular_path else None,
        mask_path=str(mask_path),
        config=config_dict,
        output_width=output_width,
        output_height=output_height,
    )

    async with async_session() as session:
        session.add(template)
        await session.commit()
        await session.refresh(template)

    return {
        'id': str(template.id),
        'slug': template.slug,
        'status': template.status,
        'message': 'Template created successfully',
    }


@router.put('/templates/{slug}/config')
async def updateConfig(
    slug: str,
    body: TemplateConfigUpdate,
):
    '''Cập nhật config cho template. Lưu lịch sử.'''
    async with async_session() as session:
        result = await session.execute(
            select(Template).where(Template.slug == slug)
        )
        template = result.scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail={
                'error': 'template_not_found',
                'message': f'Template \'{slug}\' không tồn tại',
            })

        # Lưu config cũ vào history
        history = TemplateConfigHistory(
            template_id=template.id,
            config=template.config,
            change_note=body.change_note,
        )
        session.add(history)

        # Cập nhật config mới
        template.config = body.config.model_dump()
        await session.commit()

    # Invalidate cache
    template_registry.invalidate(slug)

    return {'message': 'Config updated', 'slug': slug}


@router.post('/templates/{slug}/publish')
async def publishTemplate(slug: str):
    '''Chuyển template từ draft → active.'''
    async with async_session() as session:
        result = await session.execute(
            select(Template).where(Template.slug == slug)
        )
        template = result.scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail={
                'error': 'template_not_found',
                'message': f'Template \'{slug}\' không tồn tại',
            })

        template.status = 'active'
        await session.commit()

    return {'message': 'Template published', 'slug': slug, 'status': 'active'}


@router.post('/templates/{slug}/archive')
async def archiveTemplate(slug: str):
    '''Ẩn template (active → archived).'''
    async with async_session() as session:
        result = await session.execute(
            select(Template).where(Template.slug == slug)
        )
        template = result.scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail={
                'error': 'template_not_found',
                'message': f'Template \'{slug}\' không tồn tại',
            })

        template.status = 'archived'
        await session.commit()

    # Invalidate cache
    template_registry.invalidate(slug)

    return {'message': 'Template archived', 'slug': slug, 'status': 'archived'}


@router.post('/templates/{slug}/rollback/{history_id}')
async def rollbackConfig(slug: str, history_id: int):
    '''Khôi phục config từ lịch sử.'''
    async with async_session() as session:
        result = await session.execute(
            select(Template).where(Template.slug == slug)
        )
        template = result.scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail={'message': f'Template \'{slug}\' không tồn tại'})

        history_result = await session.execute(
            select(TemplateConfigHistory)
            .where(TemplateConfigHistory.id == history_id, TemplateConfigHistory.template_id == template.id)
        )
        history_record = history_result.scalar_one_or_none()
        if not history_record:
            raise HTTPException(status_code=404, detail={'message': 'History record not found'})

        # Lưu config hiện tại vào history trước khi rollback
        new_history = TemplateConfigHistory(
            template_id=template.id,
            config=template.config,
            change_note=f"Auto-backup before rollback to #{history_id}",
        )
        session.add(new_history)

        # Cập nhật config
        template.config = history_record.config
        await session.commit()

    # Invalidate cache
    template_registry.invalidate(slug)

    return {'message': 'Config rolled back successfully', 'slug': slug, 'history_id': history_id}


@router.get('/templates/{slug}/config-history')
async def getConfigHistory(slug: str):
    '''Lịch sử thay đổi config.'''
    async with async_session() as session:
        result = await session.execute(
            select(Template).where(Template.slug == slug)
        )
        template = result.scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail={
                'error': 'template_not_found',
                'message': f'Template \'{slug}\' không tồn tại',
            })

        history_result = await session.execute(
            select(TemplateConfigHistory)
            .where(TemplateConfigHistory.template_id == template.id)
            .order_by(TemplateConfigHistory.changed_at.desc())
        )
        history = history_result.scalars().all()

    return {
        'slug': slug,
        'history': [
            {
                'id': h.id,
                'config': h.config,
                'changed_by': h.changed_by,
                'changed_at': str(h.changed_at) if h.changed_at else None,
                'change_note': h.change_note,
            }
            for h in history
        ],
    }


@router.post('/templates/{slug}/generate-maps')
async def generateMaps(slug: str):
    '''
    Tự tạo shadow/normal/specular maps từ ảnh mockup.
    Chạy script generate_maps_from_photo.
    '''
    async with async_session() as session:
        result = await session.execute(
            select(Template).where(Template.slug == slug)
        )
        template = result.scalar_one_or_none()
        if not template:
            raise HTTPException(status_code=404, detail={
                'error': 'template_not_found',
                'message': f'Template \'{slug}\' không tồn tại',
            })

    mockup_path = template.mockup_path
    template_dir = Path(ASSET_BASE_DIR) / slug
    maps_dir = template_dir / 'maps'

    # Import và chạy trong threadpool
    from scripts.generate_maps_from_photo import generate_all_maps

    try:
        await asyncio.to_thread(
            generate_all_maps,
            mockup_path,
            str(maps_dir),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            'error': 'render_failed',
            'message': f'Generate maps failed: {str(e)}',
        })

    # Cập nhật paths trong DB
    async with async_session() as session:
        result = await session.execute(
            select(Template).where(Template.slug == slug)
        )
        template = result.scalar_one_or_none()

        shadow_path = maps_dir / 'shadow_map.png'
        normal_path = maps_dir / 'normal_map.png'
        specular_path = maps_dir / 'specular_map.png'

        if shadow_path.exists():
            template.shadow_map_path = str(shadow_path)
        if normal_path.exists():
            template.normal_map_path = str(normal_path)
        if specular_path.exists():
            template.specular_path = str(specular_path)

        await session.commit()

    # Invalidate cache
    template_registry.invalidate(slug)

    return {
        'message': 'Maps generated successfully',
        'slug': slug,
        'maps': str(maps_dir),
    }


# --- Library Endpoints ---

@router.get('/library/{lib_type}', response_model=list[LibraryItem])
async def listLibrary(lib_type: str):
    '''Liệt kê file trong thư viện (bases hoặc artworks).'''
    if lib_type not in ['bases', 'artworks']:
        raise HTTPException(status_code=400, detail='Invalid library type')
    
    lib_dir = Path(f'inputs/{lib_type}')
    if not lib_dir.exists():
        lib_dir.mkdir(parents=True, exist_ok=True)
    
    items = []
    for f in lib_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
            stat = f.stat()
            items.append(LibraryItem(
                name=f.name,
                url=f'/static/{lib_type}/{f.name}',
                size_bytes=stat.st_size,
                modified_at=str(datetime.fromtimestamp(stat.st_mtime))
            ))
    
    # Sắp xếp theo thời gian mới nhất
    items.sort(key=lambda x: x.modified_at, reverse=True)
    return items


@router.post('/library/{lib_type}/upload')
async def uploadToLibrary(
    lib_type: str,
    file: UploadFile = File(...),
):
    '''Upload file vào thư viện.'''
    if lib_type not in ['bases', 'artworks']:
        raise HTTPException(status_code=400, detail='Invalid library type')
    
    lib_dir = Path(f'inputs/{lib_type}')
    lib_dir.mkdir(parents=True, exist_ok=True)
    
    # Tránh trùng tên bằng cách thêm uuid nếu cần, hoặc ghi đè
    dest = lib_dir / file.filename
    content = await file.read()
    dest.write_bytes(content)
    
    return {
        'name': file.filename,
        'url': f'/static/{lib_type}/{file.filename}',
        'message': f'Uploaded to {lib_type} library'
    }


@router.post('/url-analysis/import')
async def importFromAnalyzedUrl(
    body: UrlAnalysisImportRequest,
    request: Request,
    preview_width: int = URL_ANALYSIS_PREVIEW_WIDTH,
    preview_height: int = URL_ANALYSIS_PREVIEW_HEIGHT,
    preview_fit: str = URL_ANALYSIS_PREVIEW_FIT,
    preview_quality: int = URL_ANALYSIS_PREVIEW_QUALITY,
):
    try:
        context = build_url_lookup_context(body.source_url)
        exact_templates = _sort_templates_by_lookup_meta(
            await _query_active_templates_by_url_analysis_meta(
                'design_lookup_key',
                context['design_lookup_key'],
            )
        )
        if exact_templates:
            matching_templates = _build_lookup_items(
                exact_templates,
                request,
                preview_width=preview_width,
                preview_height=preview_height,
                preview_fit=preview_fit,
                preview_quality=preview_quality,
            )
            existing_template_item = matching_templates[0] if matching_templates else None
            existing_meta = _extract_template_url_analysis_meta(exact_templates[0])
            existing_design_url = str(existing_meta.get('design_url') or '')

            if existing_design_url and check_local_asset_exists(existing_design_url):
                return {
                    'template_found': True,
                    'source_url': body.source_url,
                    'parsed': context['parsed'].to_dict(),
                    'product_type': context['product_type'],
                    'design_lookup_key': context['design_lookup_key'],
                    'mockup_family_key': context['mockup_family_key'],
                    'design_url': existing_design_url,
                    'existing_template': existing_template_item,
                    'matching_templates': matching_templates,
                    'template_count': len(matching_templates),
                }

            analyzed = await analyze_and_ingest_url_async(body.source_url, reuse_local_artwork=True)
            refreshed_design_url = str(analyzed.get('design_url') or '')
            if refreshed_design_url:
                await _update_templates_design_meta(
                    exact_templates,
                    refreshed_design_url,
                    analyzed.get('design_source_url'),
                    analyzed.get('design_source_mode'),
                )
                for item in matching_templates:
                    item['design_url'] = refreshed_design_url
                if existing_template_item is not None:
                    existing_template_item['design_url'] = refreshed_design_url

            return {
                'template_found': True,
                'source_url': body.source_url,
                'parsed': context['parsed'].to_dict(),
                'product_type': context['product_type'],
                'design_lookup_key': context['design_lookup_key'],
                'mockup_family_key': context['mockup_family_key'],
                'design_url': refreshed_design_url or existing_design_url or None,
                'design_source_url': analyzed.get('design_source_url'),
                'design_source_mode': analyzed.get('design_source_mode'),
                'existing_template': existing_template_item,
                'matching_templates': matching_templates,
                'template_count': len(matching_templates),
            }

        family_templates = await _query_active_templates_by_url_analysis_meta(
            'mockup_family_key',
            context['mockup_family_key'],
        )
        available_views = [item['view'] for item in list_available_mockup_views(context['parsed'])]
        preferred_view = None
        if available_views:
            preferred_view = available_views[len(family_templates) % len(available_views)]

        analyzed = await analyze_and_ingest_url_async(
            body.source_url,
            preferred_view=preferred_view,
            reuse_local_artwork=True,
        )
        analyzed['template_found'] = False
        analyzed['rotation_index'] = len(family_templates)
        analyzed['preferred_view'] = preferred_view
        return analyzed
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={'message': str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={'message': str(exc)}) from exc
    except Exception as exc:
        logger.exception('URL analysis import failed')
        raise HTTPException(status_code=502, detail={'message': str(exc)}) from exc


@router.post('/url-analysis/template-matches')
async def lookupTemplatesFromUrl(
    body: UrlAnalysisImportRequest,
    request: Request,
    preview_width: int = URL_ANALYSIS_PREVIEW_WIDTH,
    preview_height: int = URL_ANALYSIS_PREVIEW_HEIGHT,
    preview_fit: str = URL_ANALYSIS_PREVIEW_FIT,
    preview_quality: int = URL_ANALYSIS_PREVIEW_QUALITY,
):
    try:
        context = build_url_lookup_context(body.source_url)
        matched_templates = _sort_templates_by_lookup_meta(
            await _query_active_templates_by_url_analysis_meta(
                'design_lookup_key',
                context['design_lookup_key'],
            )
        )
        matching_templates = _build_lookup_items(
            matched_templates,
            request,
            preview_width=preview_width,
            preview_height=preview_height,
            preview_fit=preview_fit,
            preview_quality=preview_quality,
        )
        first_match = matching_templates[0] if matching_templates else None
        first_meta = _extract_template_url_analysis_meta(matched_templates[0]) if matched_templates else {}

        analyzed: dict | None = None
        design_url = str(first_meta.get('design_url') or '')
        design_source_url = first_meta.get('design_source_url')
        design_source_mode = first_meta.get('design_source_mode')
        warning = None

        if matching_templates:
            if not (design_url and check_local_asset_exists(design_url)):
                analyzed = await analyze_and_ingest_url_async(body.source_url, reuse_local_artwork=True)
                design_url = str(analyzed.get('design_url') or '')
                design_source_url = analyzed.get('design_source_url')
                design_source_mode = analyzed.get('design_source_mode')
                warning = analyzed.get('warning')
                if design_url:
                    await _update_templates_design_meta(
                        matched_templates,
                        design_url,
                        design_source_url,
                        design_source_mode,
                    )
                    for item in matching_templates:
                        item['design_url'] = design_url
        else:
            analyzed = await analyze_and_ingest_url_async(body.source_url, reuse_local_artwork=True)
            design_url = str(analyzed.get('design_url') or '')
            design_source_url = analyzed.get('design_source_url')
            design_source_mode = analyzed.get('design_source_mode')
            warning = analyzed.get('warning')

        return {
            'template_found': bool(matching_templates),
            'source_url': body.source_url,
            'parsed': analyzed.get('parsed') if analyzed else context['parsed'].to_dict(),
            'product_type': analyzed.get('product_type') if analyzed else context['product_type'],
            'design_lookup_key': context['design_lookup_key'],
            'mockup_family_key': context['mockup_family_key'],
            'design_url': design_url or None,
            'design_source_url': design_source_url,
            'design_source_mode': design_source_mode,
            'available_views': analyzed.get('available_views', []) if analyzed else [item['view'] for item in list_available_mockup_views(context['parsed'])],
            'warning': warning,
            'matching_templates': matching_templates,
            'template_count': len(matching_templates),
            'selected_template_id': first_match.get('template_id') if first_match else None,
            'selected_preview_url': first_match.get('preview_url') if first_match else None,
            'selected_preview_url_full': first_match.get('preview_url_full') if first_match else None,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail={'message': str(exc)}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={'message': str(exc)}) from exc
    except Exception as exc:
        logger.exception('Template URL lookup failed')
        raise HTTPException(status_code=502, detail={'message': str(exc)}) from exc


@router.get('/url-analysis/design-link')
async def getDesignLinkFromUrl(url: str, request: Request):
    from fastapi.responses import PlainTextResponse

    try:
        analyzed = await analyze_and_ingest_url_async(url)
        design_url = str(analyzed.get('design_url') or '')
        if not design_url:
            return PlainTextResponse(content="", status_code=404)
        return PlainTextResponse(content=_to_full_url(design_url, request))
    except Exception as exc:
        logger.error(f'Failed to get design link for url={url}: {exc}', exc_info=True)
        return PlainTextResponse(content="", status_code=404)


@router.get('/url-analysis/mockup-link')
@router.get('/url-analysis/template-preview-link')
async def getTemplatePreviewLinkFromUrl(
    url: str,
    request: Request,
    preview_width: int = URL_ANALYSIS_PREVIEW_WIDTH,
    preview_height: int = URL_ANALYSIS_PREVIEW_HEIGHT,
    preview_fit: str = URL_ANALYSIS_PREVIEW_FIT,
    preview_quality: int = URL_ANALYSIS_PREVIEW_QUALITY,
):
    from fastapi.responses import PlainTextResponse
    try:
        context = build_url_lookup_context(url)
        async with async_session() as session:
            result = await session.execute(
                select(Template).where(Template.status == 'active')
            )
            active_templates = result.scalars().all()

        exact_templates: list[Template] = []
        family_templates: list[Template] = []
        for template in active_templates:
            meta = _extract_template_url_analysis_meta(template)
            if meta.get('design_lookup_key') == context['design_lookup_key']:
                exact_templates.append(template)
            if meta.get('mockup_family_key') == context['mockup_family_key']:
                family_templates.append(template)

        # Exact same template+artwork was already imported => return template image immediately.
        if exact_templates:
            selected_template = _sort_templates_by_lookup_meta(exact_templates)[0]
            preview_url = _template_preview_url(selected_template.slug)
            return PlainTextResponse(content=_to_full_url(preview_url, request))

        # Cache hit for previously rendered URL => return existing rendered link.
        cached_render_path = await _find_cached_rendered_result_path(url)
        if cached_render_path:
            return PlainTextResponse(content=_to_full_url(cached_render_path, request))

        # No exact match: pick corresponding family template and render new mockup using extracted artwork.
        if not family_templates:
            return PlainTextResponse(content="", status_code=404)

        selected_template = _sort_templates_by_lookup_meta(family_templates)[0]
        selected_meta = _extract_template_url_analysis_meta(selected_template)
        preferred_view = str(selected_meta.get('mockup_view') or '').strip().lower() or None

        analyzed = await analyze_and_ingest_url_async(url, preferred_view=preferred_view)
        design_url = str(analyzed.get('design_url') or '')
        if not design_url:
            return PlainTextResponse(content="", status_code=404)

        design_path = resolve_local_asset_path(design_url)
        if not design_path.exists():
            return PlainTextResponse(content="", status_code=404)
        design_bytes = design_path.read_bytes()

        assets = await _load_active_template_assets(selected_template.slug)
        from app.pipeline.pipeline import run_pipeline

        image_bytes, _ = await asyncio.to_thread(
            run_pipeline,
            design_bytes,
            assets,
            'jpg',
            URL_ANALYSIS_RENDER_QUALITY,
        )
        rendered_path = await _persist_rendered_result_and_get_path(url, image_bytes, 'jpg')
        return PlainTextResponse(content=_to_full_url(rendered_path, request))
    except Exception as exc:
        logger.error(f'Failed to get template preview link for url={url}: {exc}', exc_info=True)
        return PlainTextResponse(content="", status_code=404)


@router.post('/templates/save-adhoc')
async def saveAdhocTemplate(
    body: TemplateSaveAdhoc,
    request: Request,
):
    """Create template from ad-hoc calibration payload."""
    target_w = max(1, int(body.output_width))
    target_h = max(1, int(body.output_height))

    # 1. Check duplicated slug
    async with async_session() as session:
        exists = await session.execute(
            select(Template).where(Template.slug == body.slug)
        )
        if exists.scalar_one_or_none():
            raise HTTPException(status_code=409, detail={'message': f"Slug '{body.slug}' already exists"})

    # 2. Resolve source mockup
    mockup_path = resolve_local_asset_path(body.mockup_url)
    if not mockup_path.exists():
        raise HTTPException(status_code=400, detail={'message': f"Mockup path '{body.mockup_url}' not found"})

    # 3. Prepare template directory
    template_dir = Path(ASSET_BASE_DIR) / body.slug
    maps_dir = template_dir / 'maps'
    maps_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import numpy as np

    src_img = cv2.imread(str(mockup_path), cv2.IMREAD_COLOR)
    if src_img is None:
        raise HTTPException(status_code=400, detail={'message': 'invalid mockup image'})
    src_h, src_w = src_img.shape[:2]

    scale_x = float(target_w) / float(max(1, src_w))
    scale_y = float(target_h) / float(max(1, src_h))

    # 4. Normalize config with coordinate scaling to target output size
    normalized_config = _normalize_saved_template_config(
        body.config,
        target_w,
        target_h,
        body.product_type,
        scale_x=scale_x,
        scale_y=scale_y,
    )

    # 5. Save canonical mockup resized exactly to requested size
    resized_mockup = cv2.resize(
        src_img,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA if (src_w > target_w or src_h > target_h) else cv2.INTER_LINEAR,
    )
    canonical_mockup_path = template_dir / 'mockup.jpg'
    ok_mockup = cv2.imwrite(
        str(canonical_mockup_path),
        resized_mockup,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not ok_mockup:
        raise HTTPException(status_code=500, detail={'message': 'failed to save mockup.jpg'})

    from scripts.generate_maps_from_photo import generate_all_maps

    # 6. Build mask from scaled config
    mask_path = maps_dir / 'mask.jpg'
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    mask_points = normalized_config.get('print_area', {}).get('mask_points')
    if mask_points:
        pts = np.array(mask_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    else:
        quad = normalized_config.get('print_area', {}).get('quad')
        if quad:
            pts = np.array(quad, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            mask.fill(255)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    ok_mask = cv2.imwrite(str(mask_path), mask_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok_mask:
        raise HTTPException(status_code=500, detail={'message': 'failed to save mask.jpg'})

    # 7. Generate maps
    try:
        await asyncio.to_thread(generate_all_maps, str(canonical_mockup_path), str(maps_dir))
    except Exception as e:
        logger.error(f'Failed to generate maps: {e}')

    # 8. Save preview as JPEG (resized to requested output size)
    preview_path = template_dir / 'preview.jpg'
    if body.preview_data_url:
        try:
            _write_preview_data_url(
                body.preview_data_url,
                preview_path,
                resize_to=(target_w, target_h),
                jpeg_quality=95,
            )
        except Exception as e:
            logger.warning(f'Failed to save preview image for {body.slug}: {e}')
    if not preview_path.exists():
        shutil.copy2(canonical_mockup_path, preview_path)

    # 9. Save DB row
    template = Template(
        slug=body.slug,
        name=body.name,
        product_type=body.product_type,
        mockup_path=str(canonical_mockup_path),
        mask_path=str(mask_path),
        shadow_map_path=str(maps_dir / 'shadow_map.png') if (maps_dir / 'shadow_map.png').exists() else None,
        normal_map_path=str(maps_dir / 'normal_map.png') if (maps_dir / 'normal_map.png').exists() else None,
        specular_path=str(maps_dir / 'specular_map.png') if (maps_dir / 'specular_map.png').exists() else None,
        config=normalized_config,
        output_width=target_w,
        output_height=target_h,
        status='active',
    )

    async with async_session() as session:
        session.add(template)
        await session.commit()

    preview_url = _template_preview_url(body.slug)
    return {
        'message': 'Template saved from adhoc',
        'slug': body.slug,
        'preview_url': preview_url,
        'preview_url_full': _to_full_url(preview_url, request),
    }
