import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine
from app.db.database import DATABASE_URL
from app.db.models import Base

async def reset_db():
    print(f"Connecting to {DATABASE_URL}...")
    engine = create_async_engine(DATABASE_URL, echo=False)
    
    # Check if we are sure
    print("Dropping all tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        
    print("Creating all tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    print("Database reset successfully.")

if __name__ == "__main__":
    asyncio.run(reset_db())
    
    print("Running seed.py...")
    import subprocess
    subprocess.run([sys.executable, "seed.py"], check=True)
