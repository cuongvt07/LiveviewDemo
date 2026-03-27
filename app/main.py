# app/main.py

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from app.routers import render, admin

logger = logging.getLogger('mockup_service')
logging.basicConfig(level=logging.INFO)

ASSET_BASE_DIR = os.getenv('ASSET_BASE_DIR', './templates')


@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Startup: init GPU + load tất cả active templates vào cache.'''
    # Khởi tạo GPU nếu RENDER_DEVICE=gpu
    from app.config import initGpu, getRenderDevice
    gpu_ok = initGpu()
    logger.info(f'Render device: {getRenderDevice()} (GPU active: {gpu_ok})')

    # Tạo thư mục templates nếu chưa có
    os.makedirs(ASSET_BASE_DIR, exist_ok=True)

    # Load templates từ DB vào cache
    try:
        from sqlalchemy import select
        from app.db.database import async_session
        from app.db.models import Template
        from app.services import template_registry

        async with async_session() as session:
            stmt = select(Template).where(Template.status == 'active')
            result = await session.execute(stmt)
            templates = result.scalars().all()

        loaded = 0
        for t in templates:
            try:
                record = {
                    'mockup_path': t.mockup_path,
                    'shadow_map_path': t.shadow_map_path,
                    'normal_map_path': t.normal_map_path,
                    'specular_path': t.specular_path,
                    'mask_path': t.mask_path,
                    'config': t.config,
                    'output_width': t.output_width,
                    'output_height': t.output_height,
                }
                template_registry.load_template(t.slug, record)
                loaded += 1
                logger.info(f'Loaded template: {t.slug}')
            except Exception as e:
                logger.warning(f'Failed to load template {t.slug}: {e}')

        logger.info(f'Loaded {loaded}/{len(templates)} templates')
    except Exception as e:
        logger.warning(f'Could not load templates from DB: {e}')
        logger.info('Running without DB — use scripts/test_render.py for local testing')

    yield

    logger.info('Shutting down...')


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title='Mug Mockup Service',
    description='POD Liveview — Render realistic mug mockups with perspective warp, '
                'color matching, shadow/displacement, specular overlay.',
    version='1.2.0',
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Static files for template previews
if os.path.isdir(ASSET_BASE_DIR):
    app.mount(
        '/static/templates',
        StaticFiles(directory=ASSET_BASE_DIR),
        name='templates',
    )

from fastapi import Request
import time

@app.middleware('http')
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f'{request.method} {request.url.path} - {response.status_code} ({duration:.2f}s)')
    return response

# Routers
app.include_router(render.router, prefix='/v1')
app.include_router(admin.router, prefix='/admin')


@app.get('/health')
async def healthCheck():
    from app.config import getRenderDevice, isGpuEnabled
    return {
        'status': 'ok',
        'version': '1.2.0',
        'render_device': getRenderDevice(),
        'gpu_active': isGpuEnabled(),
    }
