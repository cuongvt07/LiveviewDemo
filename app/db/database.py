# app/db/database.py

import os
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
)
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql+asyncpg://postgres:postgres@localhost:5432/mockup_service'
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def getSession() -> AsyncSession:
    '''Dependency injection cho FastAPI.'''
    async with async_session() as session:
        yield session
