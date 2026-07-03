from datetime import datetime
from sqlalchemy import String, Float, DateTime, Index, PrimaryKeyConstraint
from sqlalchemy.orm import Mapped, mapped_column
from database.base import Base

class MarketData(Base):
    """
    Time-series asset price bars. Configured as a TimescaleDB hypertable.
    """
    __tablename__ = "market_data"

    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "symbol"),
        Index("idx_market_data_symbol_timestamp", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<MarketData(symbol={self.symbol}, time={self.timestamp}, close={self.close})>"
