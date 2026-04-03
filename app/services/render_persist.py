# app/services/render_persist.py

"""
Persist rendered mockup results to disk and database for fast URL resolution.
Designed for fire-and-forget background usage so it never blocks the render response.
"""

import hashlib
import logging
import os
import time
from pathlib import Path

from app.db.database import async_session
from app.db.models import RenderedResult
from app.services.url_analysis import parse_printerval_liveview_url

logger = logging.getLogger('mockup_service')

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERS_DIR = REPO_ROOT / 'output' / 'renders'

# Ensure output directory exists at import time
RENDERS_DIR.mkdir(parents=True, exist_ok=True)


def extractSlug(source_url: str) -> str:
    """Extract the normalized slug from a Printerval CDN URL.

    The slug is the comma-separated identifier portion of the URL path,
    e.g. 'mugs-11oz,White,print-123_456,ffffff' — used as the DB lookup key.
    """
    if not source_url:
        return "adhoc_no_url"
        
    if not source_url.startswith(('http://', 'https://')):
        return "adhoc_" + hashlib.md5(source_url.encode('utf-8')).hexdigest()[:12]
        
    parsed = parse_printerval_liveview_url(source_url)
    return parsed.slug


def buildRenderFilename(slug: str, output_format: str) -> str:
    """Build a deterministic, collision-free filename for a rendered result."""
    slug_hash = hashlib.sha256(slug.encode('utf-8')).hexdigest()[:16]
    timestamp_ms = int(time.time() * 1000)
    extension = 'png' if output_format == 'png' else 'jpg'
    return f'{slug_hash}_{timestamp_ms}.{extension}'


def saveRenderedImage(image_bytes: bytes, filename: str) -> Path:
    """Write rendered image bytes to the renders output directory."""
    destination = RENDERS_DIR / filename
    destination.write_bytes(image_bytes)
    return destination


async def persistRenderResult(
    source_url: str,
    image_bytes: bytes,
    output_format: str = 'jpg',
) -> None:
    """Save a rendered mockup image to disk and record the mapping in the DB.

    This function is designed to be called via asyncio.create_task() so it
    runs in the background without blocking the HTTP response.
    """
    try:
        slug = extractSlug(source_url)
        filename = buildRenderFilename(slug, output_format)
        saveRenderedImage(image_bytes, filename)

        image_path = f'/static/renders/{filename}'

        async with async_session() as session:
            record = RenderedResult(
                url_slug=slug,
                image_path=image_path,
            )
            session.add(record)
            await session.commit()

        logger.info(
            'Persisted render result: slug=%s path=%s',
            slug,
            image_path,
        )
    except Exception:
        logger.warning(
            'Failed to persist render result for URL: %s',
            source_url,
            exc_info=True,
        )
