import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import select
from app.db.database import async_session
from app.db.models import Template

async def seed():
    async with async_session() as session:
        # Check if exists
        result = await session.execute(select(Template).where(Template.slug == 'mug-test'))
        existing = result.scalar_one_or_none()
        
        if existing:
            print("Template 'mug-test' already exists.")
            return

        print("Seeding 'mug-test' template...")
        t = Template(
            slug='mug-test',
            name='Cốc Test 1500px',
            description='Template sinh tự động để test giao diện',
            status='active',
            mockup_path='templates/mug-test/mockup.jpg',
            mask_path='templates/mug-test/maps/mask.png',
            config={
                'print_area': {
                    'top_left': [300, 200],
                    'top_right': [1200, 200],
                    'bottom_right': [1200, 1300],
                    'bottom_left': [300, 1300]
                },
                'lighting': {
                    'shadow_strength': 0.0,
                    'displacement_strength': 0.0,
                    'specular_strength': 0.3,
                    'specular_threshold': 220
                },
                'color': {
                    'enable_color_match': True,
                    'match_strength': 0.4
                },
                'edge': {
                    'feather_px': 6
                },
                'output': {
                    'jpeg_quality': 90
                }
            },
            output_width=1500,
            output_height=1500,
            created_by='system'
        )
        session.add(t)
        await session.commit()
        print("Seeded successfully!")

if __name__ == '__main__':
    asyncio.run(seed())
