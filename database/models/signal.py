from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Float, DateTime, ForeignKey, Index, PrimaryKeyConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from database.base import Base

class Signal(Base):
    """
    Represents generated trading signals. Compatible with TimescaleDB hypertable requirements.
    """
    __tablename__ = "signals"

    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    signal_id: Mapped[str] = mapped_column(String(50), nullable=False)
    hypothesis_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("hypotheses.hypothesis_id", ondelete="CASCADE"),
        nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    information_coefficient: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    information_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    half_life_days: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("timestamp", "signal_id"),
        Index("idx_signals_hypothesis_id", "hypothesis_id"),
        Index("idx_signals_symbol_timestamp", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<Signal(id={self.signal_id}, hypothesis={self.hypothesis_id}, symbol={self.symbol}, val={self.value})>"
