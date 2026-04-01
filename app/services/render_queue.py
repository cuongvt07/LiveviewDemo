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


def _start_worker():
    global _worker_started
    if _worker_started:
        return
    t = threading.Thread(target=_worker_loop, daemon=True)
    t.start()
    _worker_started = True


def enqueue_adhoc_render(design_bytes: bytes, mockup_bytes: bytes, config: dict, output_format: str = 'png') -> str:
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
            output_format = payload.get('output_format', 'png')

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
            # Produce a low-res quick preview (512px) to show fast
            preview_w, preview_h = (512, int(512 * h / max(w, 1)))
            try:
                # Try to generate a preview by resizing design and calling run_pipeline
                small_design = design_bytes
                # run_pipeline expects full design bytes; we'll pass original but set output size via assets.mockup size? Keep simple: render full and resize result
                result_bytes, meta = run_pipeline(design_bytes, assets, output_format, 90, False)
                # write preview and final
                preview_path = OUT_DIR / f"{job_id}_preview.png"
                final_path = OUT_DIR / f"{job_id}.png"
                with open(final_path, 'wb') as f:
                    f.write(result_bytes)
                # create small preview by resizing final
                try:
                    img = cv2.imdecode(np.frombuffer(result_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        small = cv2.resize(img, (preview_w, preview_h), interpolation=cv2.INTER_LINEAR)
                        cv2.imencode('.png', small)[1].tofile(str(preview_path))
                except Exception:
                    # fallback: copy final to preview
                    from shutil import copyfile
                    copyfile(final_path, preview_path)

                job['result'] = {
                    'final_path': str(final_path),
                    'preview_path': str(preview_path),
                    'content_type': meta.get('content_type', 'image/png'),
                    'processing_time_ms': meta.get('processing_time_ms', 0),
                }
                job['phase'] = 'done'
                job['status'] = 'done'
                job['updated_at'] = time.time()
            except Exception as e:
                job['status'] = 'failed'
                job['phase'] = 'error'
                job['error'] = str(e)
                job['updated_at'] = time.time()
        except Exception as exc:
            with _lock:
                job['status'] = 'failed'
                job['phase'] = 'error'
                job['error'] = str(exc)
                job['updated_at'] = time.time()
        finally:
            # loop continues
            pass
