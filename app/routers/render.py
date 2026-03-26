# app/routers/render.py

import uuid
import asyncio
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from sqlalchemy import select
from pydantic import BaseModel

from app.db.database import async_session
from app.db.models import Template
from app.pipeline.pipeline import run_pipeline
from app.services import template_registry
from app.pipeline.warp import perspective_from_camera_angle, cylinder_warp, tps_warp
from app.services.vision import auto_detect_print_area_v2, create_soft_mask
from app.pipeline.lighting import composite_with_surface_lighting

router = APIRouter(tags=['render'])


@router.post('/mockup/render')
async def renderMockup(
    design_image: UploadFile = File(...),
    template_id: str = Form(...),
    output_format: str = Form('jpg'),
    jpeg_quality: int = Form(None),
):
    '''
    Render design lên template mockup.
    Trả về binary image (JPEG/PNG).
    '''
    request_id = f'req_{uuid.uuid4().hex[:8]}'

    # Kiểm tra template trong cache
    assets = template_registry.get(template_id)

    if assets is None:
        # Thử load từ DB
        async with async_session() as session:
            stmt = select(Template).where(
                Template.slug == template_id,
                Template.status == 'active',
            )
            result = await session.execute(stmt)
            template = result.scalar_one_or_none()

        if template is None:
            raise HTTPException(
                status_code=404,
                detail={
                    'error': 'template_not_found',
                    'message': f'Template \'{template_id}\' không tồn tại hoặc chưa active',
                    'request_id': request_id,
                },
            )

        # Load assets vào cache
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
        try:
            assets = template_registry.load_template(template_id, record)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={
                    'error': 'render_failed',
                    'message': str(e),
                    'request_id': request_id,
                },
            )

    # Đọc design image
    design_bytes = await design_image.read()

    # Xác định quality
    quality = jpeg_quality
    if quality is None:
        quality = assets.config.get('output', {}).get('jpeg_quality', 90)

    # Chạy pipeline trong threadpool (CPU-bound)
    try:
        image_bytes, meta = await asyncio.to_thread(
            run_pipeline,
            design_bytes,
            assets,
            output_format,
            quality,
        )
    except ValueError as e:
        error_code = str(e)
        status = 400
        if error_code == 'image_too_large':
            msg = 'File > 10MB'
        elif error_code == 'invalid_image':
            msg = 'File không đọc được hoặc sai định dạng'
        else:
            msg = error_code
        raise HTTPException(
            status_code=status,
            detail={
                'error': error_code,
                'message': msg,
                'request_id': request_id,
            },
        )
    except Exception as e:
        import logging
        import traceback
        logger = logging.getLogger('mockup_service')
        logger.error(f'Render failed: {traceback.format_exc()}')
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'render_failed',
                'message': str(e),
                'request_id': request_id,
            },
        )

    return Response(
        content=image_bytes,
        media_type=meta['content_type'],
        headers={
            'X-Processing-Time-Ms': str(meta['processing_time_ms']),
            'X-Template-Id': template_id,
            'X-Request-Id': request_id,
        },
    )


@router.post('/mockup/render-adhoc')
async def renderAdhoc(
    mockup_image: UploadFile = File(...),
    design_image: UploadFile = File(...),
    output_format: str = Form('jpg'),
    config_json: str = Form(None),
):
    '''
    Render nhanh dùng ảnh mockup tùy chỉnh tải lên, không cần tạo Template.
    Tự động tạo mask và print_area mặc định ở giữa ảnh.
    '''
    import cv2
    import numpy as np
    from app.pipeline.pipeline import decode_design, TemplateAssets
    from app.pipeline.specular import extract_specular_from_mockup
    import time

    request_id = f'req_{uuid.uuid4().hex[:8]}'

    try:
        mockup_bytes = await mockup_image.read()
        design_bytes = await design_image.read()

        # Decode Mockup (BGR)
        buf = np.frombuffer(mockup_bytes, dtype=np.uint8)
        mockup = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if mockup is None:
            raise ValueError('Ảnh mockup không hợp lệ')
            
        H, W = mockup.shape[:2]
        if max(H, W) > 3000:
            scale = 3000 / max(H, W)
            mockup = cv2.resize(mockup, (int(W * scale), int(H * scale)))
            H, W = mockup.shape[:2]

        # Tạo mask full trắng rỗng (vì adhoc không có mask xịn, ta phó thác cho Alpha channel của warped image)
        mask = np.full((H, W), 255, dtype=np.uint8)
        mx, my = int(W * 0.22), int(H * 0.14)

        # --- Tự động sinh Normal Map hình trụ cho cốc (tốt hơn placeholder phẳng) ---
        # Normal trên bề mặt trụ: Nx = sin(theta), Ny = 0, Nz = cos(theta)
        # theta biến thiên theo chiều ngang từ -theta_max đến +theta_max
        theta_max_rad = np.deg2rad(52.0)
        cols = np.arange(W, dtype=np.float32)
        theta = (cols / W - 0.5) * 2.0 * theta_max_rad  # [-theta_max, +theta_max]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        Nx = np.broadcast_to(sin_t[None, :], (H, W))
        Nz = np.broadcast_to(cos_t[None, :], (H, W))
        Ny = np.zeros((H, W), dtype=np.float32)

        normal_map = np.dstack([
            ((Nx * 0.5 + 0.5) * 255).astype(np.uint8),   # R = X
            ((Ny * 0.5 + 0.5) * 255).astype(np.uint8),   # G = Y
            ((Nz * 0.5 + 0.5) * 255).astype(np.uint8),   # B = Z
        ])

        # --- Tự động sinh Shadow Map hình trụ ---
        # Viền 2 bên tối hơn (cos falloff), tạo bóng tự nhiên
        shadow_intensity = np.broadcast_to(cos_t[None, :], (H, W))
        # Boost giữa lên sáng, viền tối hơn
        shadow_map = np.clip(shadow_intensity * 0.6 + 0.4, 0, 1)
        shadow_map = (shadow_map * 255).astype(np.uint8)

        # Extract specular map (tự động)
        specular_map = extract_specular_from_mockup(mockup)

        # Config mặc định — BẬT ĐẦY ĐỦ lighting pipeline
        config = {
            'warp': {
                'warp_type': 'cylinder',
                'theta_max_deg': 52.0,
                'curve': 0.15,
            },
            'print_area': {
                'top_left':     [mx, my],
                'top_right':    [W - mx, my],
                'bottom_right': [W - mx, H - my],
                'bottom_left':  [mx, H - my],
            },
            'lighting': {
                'shadow_strength':       0.35,
                'displacement_strength': 0.08,
                'specular_strength':     0.30,
                'specular_threshold':    220,
            },
            'color': {
                'enable_color_match': True,
                'match_strength':     0.40,
            },
            'edge': {
                'feather_px': 4,
            },
            'output': {
                'jpeg_quality': 90,
            },
        }

        # Thử override config nếu user có truyền lên
        if config_json:
            import json
            try:
                user_config = json.loads(config_json)
                # Override các tuỳ chỉnh an toàn trong ad-hoc mode
                if 'print_area' in user_config:
                    # Merge print_area fields individually to avoid losing default fields if user only sends some
                    for key in ['top_left', 'top_right', 'bottom_right', 'bottom_left', 'mask_points', 'camera_elevation']:
                        if key in user_config['print_area']:
                            config['print_area'][key] = user_config['print_area'][key]
                if 'warp' in user_config:
                    config['warp'].update(user_config['warp'])
            except Exception as e:
                pass

        assets = TemplateAssets(
            mockup=mockup,
            shadow_map=shadow_map,
            normal_map=normal_map,
            mask=mask,
            specular_map=specular_map,
            config=config,
        )

        image_bytes, meta = await asyncio.to_thread(
            run_pipeline,
            design_bytes,
            assets,
            output_format,
            90,
        )

        return Response(
            content=image_bytes,
            media_type=meta['content_type'],
            headers={
                'X-Processing-Time-Ms': str(meta['processing_time_ms']),
                'X-Template-Id': 'adhoc',
                'X-Request-Id': request_id,
            },
        )

    except Exception as e:
        import traceback
        import logging
        logger = logging.getLogger('mockup_service')
        logger.error(f'Render-adhoc failed: {traceback.format_exc()}')
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'render_failed',
                'message': str(e),
                'request_id': request_id,
            },
        )


class WarpPreviewRequest(BaseModel):
    mockup_width: int
    mockup_height: int
    print_area: dict
    warp_type: str = 'cylinder'
    theta_max_deg: float = 52.0
    curve: float = 0.0
    design_scale: float = 1.0
    design_offset_x: float = 0.0
    design_offset_y: float = 0.0
    mask_points: Optional[list[list[int]]] = None
    template_id: Optional[str] = None

@router.post('/mockup/warp-preview')
async def renderWarpPreview(req: WarpPreviewRequest):
    '''
    Endpoint sinh preview đã được composite (Multiply + Highlight).
    Đảm bảo Editor nhìn thấy chính xác những gì sẽ render ra file thật.
    '''
    import numpy as np
    import cv2
    from app.pipeline.warp import perspective_warp, cylinder_warp, tps_warp, perspective_from_camera_angle
    from app.pipeline.lighting import composite_with_surface_lighting
    from app.services.vision import create_soft_mask

    W, H = req.mockup_width, req.mockup_height
    
    # 1. Tạo Design Layer cho Preview
    # Nếu không có template_id (chỉ xem grid), ta vẫn dùng grid nhưng mờ hơn
    # Nếu CÓ template_id, ta dùng một lớp bán trong suốt để thấy được vùng in trên Mockup 
    # mà không bị rối mắt bởi 2 bộ grid.
    design_canvas = np.zeros((400, 400, 4), dtype=np.uint8)
    
    design_canvas[:, :, :3] = [255, 255, 255]
    design_canvas[:, :, 3] = 10 # Rất mờ, gần như trong suốt

    # 2. Warp Geometry
    # We use the quad points directly from the frontend (already calibrated)
    pa = {
        'top_left': req.print_area['top_left'], 
        'top_right': req.print_area['top_right'],
        'bottom_right': req.print_area['bottom_right'], 
        'bottom_left': req.print_area['bottom_left']
    }

    if req.warp_type == 'cylinder':
        # Pass both smile and pitch for differential curve
        smile_val = req.curve
        pitch_val = req.print_area.get('camera_elevation', 0)
        warped = cylinder_warp(design_canvas, pa, (W, H), req.theta_max_deg, smile_val, pitch_val, 
                               req.design_scale, req.design_offset_x, req.design_offset_y)
    elif req.warp_type == 'tps':
        src_pts = req.print_area.get('mesh_src', [[0,0], [400,0], [400,400], [0,400]])
        dst_pts = req.print_area.get('mesh_dst', [pa['top_left'], pa['top_right'], pa['bottom_right'], pa['bottom_left']])
        warped = tps_warp(design_canvas, src_pts, dst_pts, (W, H))
    else:
        warped = cv2.warpPerspective(design_canvas, cv2.getPerspectiveTransform(
            np.float32([[0,0], [400,0], [400,400], [0,400]]), 
            np.float32([pa['top_left'], pa['top_right'], pa['bottom_right'], pa['bottom_left']])), (W, H))

    # 3. Apply Alpha Masking
    mask_to_use = np.full((H, W), 255, dtype=np.uint8)
    if req.mask_points:
        mask_to_use = create_soft_mask(req.mask_points, (H, W), feather_radius=2)
        if warped.shape[2] == 4:
            warped[:, :, 3] = cv2.bitwise_and(warped[:, :, 3], mask_to_use)

    # 4. Physical Composite (Tạm thời bỏ qua để ảnh sáng rõ theo yêu cầu User)
    final_view = warped
    # if req.template_id:
    #     ... lighting logic ...

    ok, buf = cv2.imencode('.png', final_view)
    return Response(content=bytes(buf), media_type='image/png', headers={"Cache-Control": "no-cache"})


@router.post('/mockup/detect-region')
async def detectRegion(mockup_image: UploadFile = File(...)):
    '''
    Tự động nhận diện vùng in (quad) và phân loại sản phẩm.
    '''
    import cv2
    import numpy as np
    from app.services.vision import auto_detect_print_area

    content = await mockup_image.read()
    buf = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh mockup")

    # Resize nếu ảnh quá lớn để processing nhanh
    H, W = img.shape[:2]
    max_dim = 1000
    if max(H, W) > max_dim:
        scale = max_dim / max(H, W)
        img_small = cv2.resize(img, (int(W * scale), int(H * scale)))
        result = auto_detect_print_area_v2(img_small)
        # Scale lại tọa độ
        for key in ['quad', 'clip_mask']:
            if result.get(key):
                result[key] = [[int(p[0]/scale), int(p[1]/scale)] for p in result[key]]
    else:
        result = auto_detect_print_area_v2(img)


def apply_gamma_correction(img, target_gamma=2.2):
    """ Áp dụng gamma correction để màu chuẩn POD. """
    lut = np.array([((i / 255.0) ** (1.0 / target_gamma)) * 255 for i in range(256)], dtype=np.uint8)
    if len(img.shape) == 3:
        res = img.copy()
        res[:, :, :3] = cv2.LUT(img[:, :, :3], lut)
        return res
    return cv2.LUT(img, lut)


def unsharp_mask(img, amount=0.3, radius=1.0):
    """ Sharpen nhẹ để bù softness từ interpolation. """
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


@router.post('/mockup/bake-normal')
async def bakeNormal(
    mockup_image: UploadFile = File(...),
    fold_map: UploadFile = File(None),
):
    '''
    Bake normal map từ ảnh mockup và fold map.
    '''
    import cv2
    import numpy as np
    from app.services.vision import bake_normal_map

    content = await mockup_image.read()
    buf = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    f_map = None
    if fold_map:
        f_content = await fold_map.read()
        f_buf = np.frombuffer(f_content, dtype=np.uint8)
        f_map = cv2.imdecode(f_buf, cv2.IMREAD_GRAYSCALE)
        # Normalize fold_map về [-1, 1] if needed? 
        # Typically brush is 0-255. 128 = neutral.
        f_map = (f_map.astype(np.float32) - 128.0) / 128.0

    normal_map = bake_normal_map(img, f_map)
    ok, buf = cv2.imencode('.png', normal_map)
    
    return Response(content=bytes(buf), media_type='image/png')


@router.get('/templates')
async def listTemplates():
    '''Liệt kê tất cả template đang active.'''
    async with async_session() as session:
        stmt = select(Template).where(Template.status == 'active')
        result = await session.execute(stmt)
        templates = result.scalars().all()

    return {
        'templates': [
            {
                'id': t.slug,
                'name': t.name,
                'preview_url': f'/static/templates/{t.slug}/mockup.jpg',
                'output_size': [t.output_width, t.output_height],
            }
            for t in templates
        ]
    }
