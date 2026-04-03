import asyncio
import os
import sys
from sqlalchemy import select
from pathlib import Path
from urllib.parse import urlparse, unquote

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

from app.db.database import async_session
from app.db.models import Template
from app.services.url_analysis import build_url_lookup_context, parse_printerval_liveview_url

async def check():
    raw_url = 'https://cdn.printerval.com/image/960x960/mugs-15oz%2Cwhite%2Cprint-seller-2026-03-30_small_2034r-decorated-with-beautiful-colorful-flowers-56aa34d4f04559230f214e0c56298d3d%2Cffffff.jpeg'
    
    print(f'Testing with RAW URL: {raw_url}')
    
    try:
        context = build_url_lookup_context(raw_url)
        target_key = context['design_lookup_key']
        print(f'Target key: {target_key}')
        
        async with async_session() as session:
            result = await session.execute(select(Template))
            templates = result.scalars().all()
            print(f'Found {len(templates)} templates in DB')
            
            for t in templates:
                config = t.config if isinstance(t.config, dict) else {}
                meta = config.get('url_analysis', {})
                design_key = meta.get('design_lookup_key')
                
                match = (design_key == target_key)
                print(f'Slug: {t.slug}, Status: {t.status}, Match: {match}')
                if match:
                    print(f'  Metadata: {meta}')
                    print(f'  Preview URL check: {_template_preview_url(t.slug)}')
                
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    asyncio.run(check())
