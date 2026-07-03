from typing import Generic, TypeVar, Type, Optional, List, Any
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")

class BaseRepository(Generic[T]):
    """
    Base Repository class implementing the Repository Pattern for async database operations.
    """
    def __init__(self, model: Type[T], session: AsyncSession):
        self.model = model
        self.session = session

    async def create(self, **kwargs) -> T:
        """Create a new model instance and insert it into the session."""
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def get_by_id(self, id_val: Any) -> Optional[T]:
        """Fetch a model instance by its primary key (single or composite)."""
        return await self.session.get(self.model, id_val)

    async def update(self, id_val: Any, **kwargs) -> Optional[T]:
        """Update fields of an existing model instance and flush the session."""
        instance = await self.get_by_id(id_val)
        if instance:
            for key, val in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, val)
                else:
                    logger.warning(f"Repository update: attribute '{key}' not found on {self.model.__name__} instance.")
            await self.session.flush()
        return instance

    async def delete(self, id_val: Any) -> bool:
        """Delete an instance by its primary key from the session."""
        instance = await self.get_by_id(id_val)
        if instance:
            await self.session.delete(instance)
            await self.session.flush()
            return True
        return False

    async def list_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """List all records for this model with limit and offset."""
        stmt = select(self.model).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
