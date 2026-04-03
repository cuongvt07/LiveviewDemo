# app/main.py

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from app.routers import render, admin, resolve

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

    # Tạo thư mục templates và library nếu chưa có
    os.makedirs(ASSET_BASE_DIR, exist_ok=True)
    os.makedirs('inputs/bases', exist_ok=True)
    os.makedirs('inputs/artworks', exist_ok=True)
    os.makedirs('public/mockups', exist_ok=True)

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

            # Warm-up: precompute cylindrical warp maps for mug templates
            try:
                assets = template_registry.get(t.slug)
                if assets is not None and assets.config.get('product_type', 'mug') == 'mug':
                    from app.pipeline.mugs.cylindrical_warp import _compute_and_cache_cylindrical_map
                    from app.pipeline.shared.design_transform import estimate_print_area_canvas_size

                    canvas_w, canvas_h = estimate_print_area_canvas_size(
                        assets.config.get('print_area'),
                        fallback_width=assets.mockup.shape[1],
                        fallback_height=assets.mockup.shape[0],
                    )
                    cyl = assets.config.get('cylinder', {})
                    try:
                        _compute_and_cache_cylindrical_map(
                            print_area=assets.config.get('print_area', {}),
                            output_size=(assets.mockup.shape[1], assets.mockup.shape[0]),
                            design_size=(canvas_w, canvas_h),
                            theta_max_deg=cyl.get('theta_max_deg', 52.0),
                            pitch=cyl.get('pitch', 0.0),
                            smile_base=cyl.get('smile_base', 0.08),
                            curve_top=cyl.get('curve_top'),
                            curve_bottom=cyl.get('curve_bottom'),
                            edge_squeeze=cyl.get('edge_squeeze', 0.0),
                            squeeze_power=cyl.get('squeeze_power', 2.0),
                            center_focus_width=cyl.get('center_focus_width', 0.0),
                            reference_mockup=assets.mockup,
                            curve_correction_alpha=cyl.get('curve_correction_alpha', 0.0),
                            curve_snap_threshold_px=cyl.get('curve_snap_threshold_px', 2.0),
                        )
                        logger.info(f'Warmed warp maps for template: {t.slug}')
                    except Exception as e:
                        logger.info(f'Could not warm warp maps for {t.slug}: {e}')
            except Exception:
                pass

        # Warm-up mesh-based warp maps for templates that provide mesh config
        try:
            from app.pipeline.mugs.warp_cache import WarpMapRegistry
            from app.pipeline.mugs.mesh_config import MugMeshConfig, MeshPoint

            sku_configs = {}
            for t in templates:
                try:
                    assets = template_registry.get(t.slug)
                    if not assets:
                        continue
                    mesh_cfg = assets.config.get('mesh', {})
                    if isinstance(mesh_cfg, dict) and mesh_cfg.get('enabled', False):
                        mcfg = MugMeshConfig(
                            cols=int(mesh_cfg.get('cols', 12)),
                            rows=int(mesh_cfg.get('rows', 8)),
                            spacing=str(mesh_cfg.get('spacing', 'cosine')),
                            tension=float(mesh_cfg.get('tension', 0.35)),
                            corner_blend_radius=float(mesh_cfg.get('corner_blend_radius', 0.08)),
                            symmetry_lock=bool(mesh_cfg.get('symmetry_lock', True)),
                            max_displacement=float(mesh_cfg.get('max_displacement', 0.15)),
                        )
                        pts = mesh_cfg.get('points')
                        if isinstance(pts, list) and len(pts) == (mcfg.rows + 1) * (mcfg.cols + 1):
                            mcfg.points = [MeshPoint(u=float(p[0]), v=float(p[1]), locked=bool(p[2]) if len(p) > 2 else False) for p in pts]
                        sku_configs[t.slug] = mcfg
                except Exception:
                    continue

            if sku_configs:
                try:
                    WarpMapRegistry().warmup(sku_configs)
                    logger.info(f'Started mesh warp warmup for {len(sku_configs)} templates')
                except Exception as e:
                    logger.warning(f'Failed to start mesh warp warmup: {e}')
        except Exception:
            pass

        logger.info(f'Loaded {loaded}/{len(templates)} templates')
    except Exception as e:
        logger.warning(f'Could not load templates from DB: {e}')
        logger.info('Running without DB — use scripts/test_render.py for local testing')

    yield

    logger.info('Shutting down...')
    try:
        from app.services.url_analysis import close_async_http_client
        await close_async_http_client()
    except Exception:
        logger.warning('Failed to close URL analysis HTTP client', exc_info=True)


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

# Static files for libraries
if os.path.exists('inputs/bases'):
    app.mount('/static/bases', StaticFiles(directory='inputs/bases'), name='bases')
if os.path.exists('inputs/artworks'):
    app.mount('/static/artworks', StaticFiles(directory='inputs/artworks'), name='artworks')
if os.path.exists('public/mockups'):
    app.mount('/static/mockups', StaticFiles(directory='public/mockups'), name='mockups')
os.makedirs('output/renders', exist_ok=True)
app.mount('/static/renders', StaticFiles(directory='output/renders'), name='renders')

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
app.include_router(resolve.router, prefix='/v1')
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
