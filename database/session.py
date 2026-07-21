import os
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

logger = logging.getLogger(__name__)

class DatabaseSessionManager:
    """
    Manages sync and async database engines, session factories, connection pools,
    and fallback mechanisms for PostgreSQL/TimescaleDB and SQLite.
    """
    def __init__(self, dsn: Optional[str] = None, sqlite_path: Optional[str] = None):
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if sqlite_path is None:
            from pathlib import Path
            sqlite_path = str(Path(__file__).resolve().parent.parent / "alphalens_local.db")
        self.sqlite_path = sqlite_path
        self.is_sqlite = True
        
        # Determine paths and drivers
        self._init_dsn_paths()
        
        # Init Engines & Session Factories
        self._init_engines()

        # Auto-create tables for SQLite to ensure robust schema presence
        if self.is_sqlite:
            try:
                self.create_tables()
            except Exception as e:
                logger.warning(f"DatabaseSessionManager: Auto table creation failed: {e}")

    def _init_dsn_paths(self):
        if self.dsn:
            # Check if using postgres
            if self.dsn.startswith("postgresql://") or self.dsn.startswith("postgresql+asyncpg://") or self.dsn.startswith("postgres://"):
                self.is_sqlite = False
                # Convert raw postgres DSNs to appropriate drivers
                clean_dsn = self.dsn.replace("postgres://", "postgresql://")
                self.sync_dsn = clean_dsn
                if "postgresql+asyncpg://" not in clean_dsn:
                    self.async_dsn = clean_dsn.replace("postgresql://", "postgresql+asyncpg://")
                else:
                    self.async_dsn = clean_dsn
                logger.info("DatabaseSessionManager: Configured for PostgreSQL/TimescaleDB.")
                return
        
        # SQLite fallback
        self.is_sqlite = True
        self.sync_dsn = f"sqlite:///{self.sqlite_path}"
        self.async_dsn = f"sqlite+aiosqlite:///{self.sqlite_path}"
        logger.info(f"DatabaseSessionManager: Configured for SQLite at {self.sqlite_path}.")

    def _init_engines(self):
        # Setup Connection Pools for Postgres
        if not self.is_sqlite:
            # Sync Engine
            self.sync_engine = create_engine(
                self.sync_dsn,
                pool_size=20,
                max_overflow=10,
                pool_recycle=1800,
                pool_pre_ping=True
            )
            # Async Engine
            self.async_engine = create_async_engine(
                self.async_dsn,
                pool_size=20,
                max_overflow=10,
                pool_recycle=1800,
                pool_pre_ping=True
            )
        else:
            # SQLite engines (disable pooling limitations, but use check_same_thread=False)
            self.sync_engine = create_engine(
                self.sync_dsn,
                connect_args={"check_same_thread": False}
            )
            self.async_engine = create_async_engine(
                self.async_dsn,
                connect_args={"check_same_thread": False}
            )

        # Session Makers
        self.sync_session_factory = sessionmaker(
            bind=self.sync_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

    @asynccontextmanager
    async def async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Async session context manager with automatic rollback and close."""
        session: AsyncSession = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session rollback due to exception: {e}")
            raise
        finally:
            await session.close()

    @contextmanager
    def sync_session(self) -> Generator[Session, None, None]:
        """Sync session context manager with automatic rollback and close."""
        session: Session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Sync database session rollback due to exception: {e}")
            raise
        finally:
            session.close()

    async def close_all(self):
        """Disposes sync and async engines."""
        if hasattr(self, 'sync_engine'):
            self.sync_engine.dispose()
        if hasattr(self, 'async_engine'):
            await self.async_engine.dispose()
        logger.info("DatabaseSessionManager engines disposed.")

    def create_tables(self):
        """Creates all registered SQLAlchemy schema tables."""
        from database.base import Base
        import database.models  # Imports all models to register them on Base
        Base.metadata.create_all(self.sync_engine)
        logger.info("DatabaseSessionManager: All ORM tables initialized successfully.")

# Global session manager instance
db_manager = DatabaseSessionManager()

