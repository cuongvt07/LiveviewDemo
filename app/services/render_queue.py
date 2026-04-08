import threading
import time
import uuid
import os
from typing import Any, Dict

from pathlib import Path

from app.pipeline.pipeline import run_pipeline

_jobs: Dict[str, Dict[str, Any]] = {}
_queue: list[str] = []
_lock = threading.Lock()
_worker_started = False

OUT_DIR = Path('output') / 'renders'
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_MAX_DIM = 512

# --- Giới hạn bộ nhớ ---
_JOB_TTL_SECONDS = 3600       # Xóa job sau 1 giờ
_MAX_JOBS_IN_MEMORY = 200     # Giới hạn tối đa 200 job trong RAM
_LAST_CLEANUP_TIME = 0.0
_CLEANUP_INTERVAL = 300       # Dọn dẹp mỗi 5 phút


def _cleanup_expired_jobs():
    '''Xóa các job đã quá hạn TTL ra khỏi bộ nhớ.'''
    global _LAST_CLEANUP_TIME
    now = time.time()
    if now - _LAST_CLEANUP_TIME < _CLEANUP_INTERVAL:
        return
    _LAST_CLEANUP_TIME = now

    expired_ids = []
    for job_id, job in _jobs.items():
        age = now - job.get('created_at', now)
        is_terminal = job.get('status') in ('done', 'failed')
        if is_terminal and age > _JOB_TTL_SECONDS:
            expired_ids.append(job_id)

    for job_id in expired_ids:
        _jobs.pop(job_id, None)

    # Nếu vẫn vượt giới hạn, xóa thêm job cũ nhất đã hoàn thành
    if len(_jobs) > _MAX_JOBS_IN_MEMORY:
        terminal_jobs = [
            (jid, j.get('created_at', 0))
            for jid, j in _jobs.items()
            if j.get('status') in ('done', 'failed')
        ]
        terminal_jobs.sort(key=lambda x: x[1])
        excess = len(_jobs) - _MAX_JOBS_IN_MEMORY
        for jid, _ in terminal_jobs[:excess]:
            _jobs.pop(jid, None)


def _release_job_payload(job: Dict[str, Any]):
    '''Giải phóng dữ liệu nặng (bytes ảnh) sau khi job xong.'''
    payload = job.get('payload')
    if isinstance(payload, dict):
        payload.pop('design_bytes', None)
        payload.pop('mockup_bytes', None)


def _start_worker():
    global _worker_started
    if _worker_started:
        return
    t = threading.Thread(target=_worker_loop, daemon=True)
    t.start()
    _worker_started = True


def enqueue_adhoc_render(design_bytes: bytes, mockup_bytes: bytes, config: dict, output_format: str = 'jpg') -> str:
    job_id = f'job_{uuid.uuid4().hex[:12]}'
    job = {
        'id': job_id,
        'status': 'queued',
        'phase': 'queued',
        'created_at': time.time(),
        'updated_at': time.time(),
        'payload': {
            'design_bytes': design_bytes,
            'mockup_bytes': mockup_bytes,
            'config': config,
            'output_format': output_format,
        },
        'result': {},
    }
    with _lock:
        _cleanup_expired_jobs()
        _jobs[job_id] = job
        _queue.append(job_id)
    _start_worker()
    return job_id


def get_job(job_id: str) -> Dict[str, Any] | None:
    with _lock:
        return _jobs.get(job_id)


def _worker_loop():
    while True:
        job_id = None
        with _lock:
            if _queue:
                job_id = _queue.pop(0)
        if job_id is None:
            time.sleep(0.2)
            continue

        job = _jobs.get(job_id)
        if job is None:
            continue
        try:
            job['status'] = 'processing'
            job['phase'] = 'preparing'
            job['updated_at'] = time.time()

            payload = job['payload']
            design_bytes = payload['design_bytes']
            mockup_bytes = payload['mockup_bytes']
            config = payload['config']
            requested_output_format = str(payload.get('output_format', 'jpg')).strip().lower()
            output_format = 'jpg'
            if requested_output_format not in {'jpg', 'jpeg'}:
                # Queue worker enforces JPEG output for lighter payloads.
                pass

            # Build minimal assets similar to render-adhoc
            import cv2
            import numpy as np
            from app.pipeline.mugs.mug_pipeline import MugAssets
            from app.pipeline.mugs.specular_gloss import extract_specular_from_mockup

            job['phase'] = 'decode'
            job['updated_at'] = time.time()

            buf = np.frombuffer(mockup_bytes, dtype=np.uint8)
            mockup = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if mockup is None:
                raise RuntimeError('invalid_mockup')

            h, w = mockup.shape[:2]
            mask = np.full((h, w), 255, dtype=np.uint8)

            lighting_cfg = config.get('lighting', {})
            specular_strength = float(lighting_cfg.get('specular_strength', 0.0))
            if specular_strength > 0:
                specular_map = extract_specular_from_mockup(mockup, threshold=int(lighting_cfg.get('specular_threshold', 220)))
            else:
                specular_map = np.zeros((h, w), dtype=np.float32)

            product_type = config.get('product_type', 'mug')
            if product_type == 'mug':
                assets = MugAssets(mockup=mockup, shadow_map=np.zeros((h, w), dtype=np.uint8), normal_map=np.zeros((h, w, 3), dtype=np.uint8), mask=mask, specular_map=specular_map, config=config)
            else:
                # fallback to mug assets shape
                assets = MugAssets(mockup=mockup, shadow_map=np.zeros((h, w), dtype=np.uint8), normal_map=np.zeros((h, w, 3), dtype=np.uint8), mask=mask, specular_map=specular_map, config=config)

            job['phase'] = 'rendering_lowres'
            job['updated_at'] = time.time()
            # Produce a quick preview with max dimension pinned to 512px.
            max_side = max(int(w), int(h), 1)
            scale = min(1.0, float(PREVIEW_MAX_DIM) / float(max_side))
            preview_w = max(1, int(round(w * scale)))
            preview_h = max(1, int(round(h * scale)))
            try:
                # Keep render output logic simple: render once, then downscale for preview.
                result_bytes, meta = run_pipeline(design_bytes, assets, output_format, 90, False)
                # write preview and final
                preview_path = OUT_DIR / f"{job_id}_preview.jpg"
                final_path = OUT_DIR / f"{job_id}.jpg"
                with open(final_path, 'wb') as f:
                    f.write(result_bytes)
                # create small preview by resizing final
                try:
                    img = cv2.imdecode(np.frombuffer(result_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        small = cv2.resize(img, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
                        ok_preview, preview_buf = cv2.imencode(
                            '.jpg',
                            small,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 85],
                        )
                        if ok_preview:
                            preview_buf.tofile(str(preview_path))
                        else:
                            raise RuntimeError('preview_encode_failed')
                except Exception:
                    # fallback: copy final to preview
                    from shutil import copyfile
                    copyfile(final_path, preview_path)

                job['result'] = {
                    'final_path': str(final_path),
                    'preview_path': str(preview_path),
                    'content_type': meta.get('content_type', 'image/jpeg'),
                    'processing_time_ms': meta.get('processing_time_ms', 0),
                }
                job['phase'] = 'done'
                job['status'] = 'done'
                job['updated_at'] = time.time()
                _release_job_payload(job)
            except Exception as e:
                job['status'] = 'failed'
                job['phase'] = 'error'
                job['error'] = str(e)
                job['updated_at'] = time.time()
                _release_job_payload(job)
        except Exception as exc:
            with _lock:
                job['status'] = 'failed'
                job['phase'] = 'error'
                job['error'] = str(exc)
                job['updated_at'] = time.time()
                _release_job_payload(job)
        finally:
            # loop continues
            pass
