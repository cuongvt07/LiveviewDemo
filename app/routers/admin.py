# app/routers/admin.py

import os
import uuid
import shutil
import asyncio
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
from app.schemas import TemplateCreate, TemplateConfigUpdate
from app.services import template_registry

router = APIRouter(tags=['admin'])

ASSET_BASE_DIR = os.getenv('ASSET_BASE_DIR', './templates')
ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', 'change-me-in-production')


def verifyAdmin(authorization: str = Header(None)):
    '''Kiểm tra admin token.'''
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing token')
    token = authorization.split(' ', 1)[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail='Invalid token')


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
    mask_path = maps_dir / 'mask.png'

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
