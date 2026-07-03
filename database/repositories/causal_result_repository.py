from typing import Optional
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from database.models.causal_result import CausalResult
from database.repositories.base import BaseRepository

class CausalResultRepository(BaseRepository[CausalResult]):
    """
    Repository for handling queries related to CausalResult model.
    """
    def __init__(self, session: AsyncSession):
        super().__init__(CausalResult, session)

    async def get_latest_by_hypothesis(self, hypothesis_id: str) -> Optional[CausalResult]:
        """Fetch the most recent causal result for a hypothesis."""
        stmt = select(CausalResult).where(CausalResult.hypothesis_id == hypothesis_id).order_by(CausalResult.created_at.desc()).limit(1)
        result = await self.session.execute(stmt)
        return result.scalars().first()
