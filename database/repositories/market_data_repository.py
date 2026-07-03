from typing import List, Optional, Union
from datetime import datetime
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from database.models.market_data import MarketData
from database.repositories.base import BaseRepository

class MarketDataRepository(BaseRepository[MarketData]):
    """
    Repository for MarketData providing look-ahead bias prevention.
    """
    def __init__(self, session: AsyncSession):
        super().__init__(MarketData, session)

    async def get_market_data(
        self,
        symbol: str,
        timestamp: Union[datetime, str],
        limit: int = 100
    ) -> List[MarketData]:
        """
        Gets market data for a symbol up to the given timestamp.
        Crucial for preventing look-ahead bias: NEVER returns records AFTER the timestamp.
        """
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp)
        else:
            dt = timestamp

        stmt = (
            select(MarketData)
            .where(MarketData.symbol == symbol)
            .where(MarketData.timestamp <= dt)
            .order_by(MarketData.timestamp.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        # Return chronologically ascending (oldest first)
        records = list(result.scalars().all())
        return list(reversed(records))

    async def get_latest_market_data(
        self,
        symbol: str,
        timestamp: Union[datetime, str]
    ) -> Optional[MarketData]:
        """
        Retrieves the single latest price bar as of (at or before) the specified timestamp.
        """
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp)
        else:
            dt = timestamp

        stmt = (
            select(MarketData)
            .where(MarketData.symbol == symbol)
            .where(MarketData.timestamp <= dt)
            .order_by(MarketData.timestamp.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()
