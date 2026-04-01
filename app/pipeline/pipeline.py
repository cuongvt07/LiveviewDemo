# app/pipeline/pipeline.py

import cv2
import time
import numpy as np
from .mugs.mug_pipeline        import run_mug_pipeline,     MugAssets
from .clothes.clothes_pipeline import run_clothes_pipeline,  ClothesAssets


def run_pipeline(
    design_bytes: bytes,
    assets,                    # MugAssets | ClothesAssets
    output_format: str = "jpg",
    jpeg_quality: int = 90,
    optimize_jpeg: bool = True,
) -> tuple[bytes, dict]:
    """
    Entry point chung — dispatch theo product_type trong config.
    """
    t0 = time.perf_counter()
    product_type = assets.config.get("product_type", "mug")

    # Dispatch
    if product_type == "mug":
        result_bgr = run_mug_pipeline(design_bytes, assets)
    elif product_type in ("tshirt", "hoodie", "tote", "clothes"):
        result_bgr = run_clothes_pipeline(design_bytes, assets)
    else:
        raise ValueError(f"Unknown product_type: {product_type}")

    # Encode
    if output_format == "png":
        ok, buf = cv2.imencode(".png", result_bgr)
        content_type = "image/png"
    else:
        ok, buf = cv2.imencode(
            ".jpg", result_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality,
             cv2.IMWRITE_JPEG_OPTIMIZE, 1 if optimize_jpeg else 0]
        )
        content_type = "image/jpeg"

    if not ok:
        raise RuntimeError("encode_failed")

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return bytes(buf), {
        "processing_time_ms": elapsed_ms,
        "content_type": content_type,
        "product_type": product_type,
    }
