from sqlalchemy.ext.asyncio import AsyncSession
from database.models.hypothesis import Hypothesis
from database.repositories.base import BaseRepository

class HypothesisRepository(BaseRepository[Hypothesis]):
    """
    Repository for handling queries related to Hypothesis model.
    """
    def __init__(self, session: AsyncSession):
        super().__init__(Hypothesis, session)
