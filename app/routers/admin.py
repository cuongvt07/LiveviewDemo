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
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    Header,
)
from sqlalchemy import select

from app.db.database import async_session, getSession
from app.db.models import Template, TemplateConfigHistory
from app.schemas import (
    TemplateCreate,
    TemplateConfigUpdate,
    LibraryItem,
    TemplateSaveAdhoc,
    UrlAnalysisImportRequest,
)
from app.services import template_registry
from app.services.url_analysis import (
    analyze_and_ingest_url,
    build_url_lookup_context,
    list_available_mockup_views,
    resolve_local_asset_path,
)

router = APIRouter(tags=['admin'])
logger = logging.getLogger('mockup_service')

ASSET_BASE_DIR = os.getenv('ASSET_BASE_DIR', './templates')
ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', 'change-me-in-production')


def verifyAdmin(authorization: str = Header(None)):
    '''Kiểm tra admin token.'''
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing token')
    token = authorization.split(' ', 1)[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail='Invalid token')


def _normalize_saved_template_config(
    raw_config: dict,
    output_width: int,
    output_height: int,
    product_type_hint: str,
) -> dict:
    from app.routers.render import _build_default_adhoc_config, _merge_adhoc_user_config

    payload = copy.deepcopy(raw_config) if isinstance(raw_config, dict) else {}
    if not isinstance(payload.get('print_area'), dict):
        payload['print_area'] = {}
    if not isinstance(payload.get('warp'), dict):
        payload['warp'] = {}

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
    preview_path = template_dir / 'preview.png'
    if preview_path.exists():
        return f'/static/templates/{slug}/preview.png'
    return f'/static/templates/{slug}/mockup.jpg'


def _extract_template_url_analysis_meta(template: Template) -> dict:
    cfg = template.config if isinstance(template.config, dict) else {}
    meta = cfg.get('url_analysis')
    return meta if isinstance(meta, dict) else {}


def _write_preview_data_url(preview_data_url: str, destination: Path) -> None:
    if not isinstance(preview_data_url, str) or not preview_data_url.startswith('data:'):
        raise ValueError('preview_data_url không hợp lệ')

    header, _, payload = preview_data_url.partition(',')
    if ';base64' not in header or not payload:
        raise ValueError('preview_data_url phải là base64 data URL')

    binary = base64.b64decode(payload)
    destination.write_bytes(binary)


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
    _admin=Depends(verifyAdmin),
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
    _admin=Depends(verifyAdmin),
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
async def publishTemplate(slug: str, _admin=Depends(verifyAdmin)):
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
async def archiveTemplate(slug: str, _admin=Depends(verifyAdmin)):
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
async def rollbackConfig(slug: str, history_id: int, _admin=Depends(verifyAdmin)):
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
async def getConfigHistory(slug: str, _admin=Depends(verifyAdmin)):
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
async def generateMaps(slug: str, _admin=Depends(verifyAdmin)):
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
async def importFromAnalyzedUrl(body: UrlAnalysisImportRequest):
    try:
        context = build_url_lookup_context(body.source_url)

        async with async_session() as session:
            result = await session.execute(
                select(Template).where(Template.status == 'active')
            )
            active_templates = result.scalars().all()

        existing_template: Template | None = None
        family_templates: list[Template] = []
        for template in active_templates:
            meta = _extract_template_url_analysis_meta(template)
            if meta.get('design_lookup_key') == context['design_lookup_key']:
                existing_template = template
                break
            if meta.get('mockup_family_key') == context['mockup_family_key']:
                family_templates.append(template)

        if existing_template is not None:
            meta = _extract_template_url_analysis_meta(existing_template)
            return {
                'template_found': True,
                'source_url': body.source_url,
                'parsed': context['parsed'].to_dict(),
                'design_lookup_key': context['design_lookup_key'],
                'mockup_family_key': context['mockup_family_key'],
                'existing_template': {
                    'slug': existing_template.slug,
                    'name': existing_template.name,
                    'template_id': existing_template.slug,
                    'preview_url': _template_preview_url(existing_template.slug),
                    'mockup_view': meta.get('mockup_view'),
                },
            }

        available_views = [item['view'] for item in list_available_mockup_views(context['parsed'])]
        preferred_view = None
        if available_views:
            preferred_view = available_views[len(family_templates) % len(available_views)]

        analyzed = analyze_and_ingest_url(body.source_url, preferred_view=preferred_view)
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


@router.post('/templates/save-adhoc')
async def saveAdhocTemplate(
    body: TemplateSaveAdhoc,
):
    '''Tạo template mới từ kết quả calibrate adhoc.'''
    # 1. Kiểm tra slug trùng
    async with async_session() as session:
        exists = await session.execute(
            select(Template).where(Template.slug == body.slug)
        )
        if exists.scalar_one_or_none():
            raise HTTPException(status_code=409, detail={'message': f'Slug \'{body.slug}\' đã tồn tại'})

    # 2. Xác định ảnh gốc (từ library hoặc path)
    mockup_path = resolve_local_asset_path(body.mockup_url)
    if not mockup_path.exists():
         raise HTTPException(status_code=400, detail={'message': f'Mockup path \'{body.mockup_url}\' không tồn tại trên server'})

    # 3. Chuẩn bị thư mục template
    template_dir = Path(ASSET_BASE_DIR) / body.slug
    maps_dir = template_dir / 'maps'
    maps_dir.mkdir(parents=True, exist_ok=True)

    normalized_config = _normalize_saved_template_config(
        body.config,
        body.output_width,
        body.output_height,
        body.product_type,
    )

    # Copy ảnh mockup vào thư mục template để làm "bản chính"
    shutil.copy2(mockup_path, template_dir / 'mockup.jpg')
    
    # 4. Tạo mặt nạ (mask) từ config nếu có mask_points
    # (Tạm thời giả định frontend đã calibrate xong và gởi config đầy đủ)
    # Chúng ta sẽ cần tạo file mask.png. Nếu không có mask_points, copy mask trắng
    # Nhưng quy trình này thường cần mask.png. 
    # TODO: Implement auto-mask generation from mask_points if missing.
    # Hiện tại giả định frontend gởi config chứa print_area.
    
    # Để đơn giản và "render nhanh", chúng ta sẽ chạy generate-maps ngay
    from scripts.generate_maps_from_photo import generate_all_maps
    
    # Tạo mask.png (giả định mask trắng toàn bộ nếu chưa có logic tách nền)
    # Hoặc nếu user đã vẽ mask_points, chúng ta dùng nó để tạo mask.png
    mask_path = maps_dir / 'mask.jpg'
    
    # Dummy mask generator (trắng) - logic thật nên dùng cv2 vẽ mask_points
    import cv2
    import numpy as np
    img = cv2.imread(str(mockup_path))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    mask_points = normalized_config.get('print_area', {}).get('mask_points')
    if mask_points:
        pts = np.array(mask_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    else:
        # Nếu ko vẽ mask, coi như cả vùng in là mask (hoặc lấy quad)
        quad = normalized_config.get('print_area', {}).get('quad')
        if quad:
             pts = np.array(quad, dtype=np.int32)
             cv2.fillPoly(mask, [pts], 255)
        else:
             mask.fill(255) # Fallback trắng xóa
             
    # Convert to 3-channel before saving JPEG (grayscale -> BGR)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Save with reasonable quality
    cv2.imwrite(str(mask_path), mask_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # 5. Sinh Maps
    try:
        await asyncio.to_thread(generate_all_maps, str(template_dir / 'mockup.jpg'), str(maps_dir))
    except Exception as e:
        logger.error(f'Failed to generate maps: {e}')

    preview_path = template_dir / 'preview.png'
    if body.preview_data_url:
        try:
            _write_preview_data_url(body.preview_data_url, preview_path)
        except Exception as e:
            logger.warning(f'Failed to save preview image for {body.slug}: {e}')

    # 6. Lưu DB
    template = Template(
        slug=body.slug,
        name=body.name,
        product_type=body.product_type,
        mockup_path=str(template_dir / 'mockup.jpg'),
        mask_path=str(mask_path),
        shadow_map_path=str(maps_dir / 'shadow_map.png') if (maps_dir / 'shadow_map.png').exists() else None,
        normal_map_path=str(maps_dir / 'normal_map.png') if (maps_dir / 'normal_map.png').exists() else None,
        specular_path=str(maps_dir / 'specular_map.png') if (maps_dir / 'specular_map.png').exists() else None,
        config=normalized_config,
        output_width=body.output_width,
        output_height=body.output_height,
        status='active'
    )

    async with async_session() as session:
        session.add(template)
        await session.commit()
    
    return {
        'message': 'Template saved from adhoc',
        'slug': body.slug,
        'preview_url': _template_preview_url(body.slug),
    }
