from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, DateTime, Index, PrimaryKeyConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from database.base import Base

class PortfolioAllocation(Base):
    """
    Tracks portfolio asset allocation weights over time.
    """
    __tablename__ = "portfolio_allocations"

    allocation_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    portfolio_id: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("idx_portfolio_allocations_port_time", "portfolio_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<PortfolioAllocation(portfolio={self.portfolio_id}, symbol={self.symbol}, weight={self.weight})>"


class PortfolioReturn(Base):
    """
    Timeseries portfolio performance metrics. Configured as a TimescaleDB hypertable.
    """
    __tablename__ = "portfolio_returns"

    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    portfolio_id: Mapped[str] = mapped_column(String(50), nullable=False)
    returns: Mapped[float] = mapped_column(Float, nullable=False)
    equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "portfolio_id"),
    )

    def __repr__(self) -> str:
        return f"<PortfolioReturn(portfolio={self.portfolio_id}, time={self.timestamp}, returns={self.returns})>"
