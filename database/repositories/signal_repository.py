from typing import List
from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from database.models.signal import Signal
from database.repositories.base import BaseRepository

class SignalRepository(BaseRepository[Signal]):
    """
    Repository for handling queries related to Signal model.
    """
    def __init__(self, session: AsyncSession):
        super().__init__(Signal, session)

    async def get_by_hypothesis(self, hypothesis_id: str) -> List[Signal]:
        """Fetch all signals related to a specific hypothesis."""
        stmt = select(Signal).where(Signal.hypothesis_id == hypothesis_id).order_by(Signal.timestamp.desc())
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
