import asyncio
import sys
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# psycopg3's async mode refuses to run under Windows' default
# ProactorEventLoop - it requires a selector-based loop instead. Must be
# set before any event loop is created, so this runs at import time.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# pool_size + max_overflow is the max connections *one worker process*
# can hold; multiply by worker count and check against Postgres's
# max_connections (SHOW max_connections) before scaling to more workers.
engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)


SessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        await db.close()
