# app/routers/resolve.py

"""
Fast resolution endpoint: maps a Printerval CDN URL to a previously rendered
mockup image URL.  Target latency < 1 s — achieved via indexed DB lookup with
no image processing.
"""

import logging

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.db.database import async_session
from app.db.models import RenderedResult
from app.schemas import UrlResolveResponse
from app.services.url_analysis import parse_printerval_liveview_url

logger = logging.getLogger('mockup_service')

router = APIRouter(tags=['resolve'])


@router.get(
    '/mockup/resolve',
    response_model=UrlResolveResponse,
    summary='Resolve a Printerval URL to a rendered mockup image',
    description=(
        'Accepts a full Printerval CDN liveview URL and returns the URL of '
        'the oldest previously rendered mockup image for that design, if one '
        'exists in the database.  Designed for sub-second responses.'
    ),
)
async def resolveMockupUrl(
    url: str = Query(
        ...,
        min_length=8,
        max_length=2000,
        description='Full Printerval CDN liveview URL to resolve',
    ),
):
    # 1. Parse URL → extract slug (the unique design identifier)
    try:
        parsed = parse_printerval_liveview_url(url)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'invalid_url',
                'message': str(exc),
            },
        ) from exc

    slug = parsed.slug

    # 2. Query the oldest rendered result for this slug
    async with async_session() as session:
        stmt = (
            select(RenderedResult)
            .where(RenderedResult.url_slug == slug)
            .order_by(RenderedResult.created_at.asc())
            .limit(1)
        )
        result = await session.execute(stmt)
        record = result.scalar_one_or_none()

    # 3. Return response
    if record is None:
        return UrlResolveResponse(found=False)

    return UrlResolveResponse(
        found=True,
        url=record.image_path,
        slug=slug,
        created_at=record.created_at.isoformat() if record.created_at else None,
    )


@router.post(
    '/mockup/resolve-batch',
    summary='Resolve multiple Printerval URLs in a single request',
    description=(
        'Accepts a list of Printerval CDN URLs and returns resolved mockup '
        'image URLs for each.  Returns oldest result per slug for historical '
        'accuracy.'
    ),
)
async def resolveMockupUrlBatch(
    urls: list[str],
):
    """Batch resolution — parse all slugs up front, then do a single DB query."""
    if len(urls) > 100:
        raise HTTPException(
            status_code=400,
            detail={
                'error': 'too_many_urls',
                'message': 'Maximum 100 URLs per batch request',
            },
        )

    # Parse all URLs → slugs
    slug_map: dict[str, str] = {}  # slug → original URL
    errors: dict[str, str] = {}
    for source_url in urls:
        try:
            parsed = parse_printerval_liveview_url(source_url)
            slug_map[parsed.slug] = source_url
        except ValueError as exc:
            errors[source_url] = str(exc)

    # Single DB query for all slugs
    resolved: dict[str, dict] = {}
    if slug_map:
        async with async_session() as session:
            stmt = (
                select(RenderedResult)
                .where(RenderedResult.url_slug.in_(list(slug_map.keys())))
                .order_by(RenderedResult.created_at.asc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        # Keep only the oldest per slug
        for row in rows:
            if row.url_slug not in resolved:
                resolved[row.url_slug] = {
                    'found': True,
                    'url': row.image_path,
                    'slug': row.url_slug,
                    'created_at': row.created_at.isoformat() if row.created_at else None,
                }

    # Build final response keyed by original URL
    results: dict[str, dict] = {}
    for slug, source_url in slug_map.items():
        if slug in resolved:
            results[source_url] = resolved[slug]
        else:
            results[source_url] = {'found': False}

    return {
        'results': results,
        'errors': errors,
        'total': len(urls),
        'resolved_count': len(resolved),
    }
