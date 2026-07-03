from datetime import datetime, timezone
from typing import Optional, Dict
from sqlalchemy import String, Float, ForeignKey, JSON, DateTime, Index, func
from sqlalchemy.orm import Mapped, mapped_column
from database.base import Base

class CausalResult(Base):
    """
    Represents causal inference analysis results.
    """
    __tablename__ = "causal_results"

    result_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    hypothesis_id: Mapped[str] = mapped_column(
        String(50),
        ForeignKey("hypotheses.hypothesis_id", ondelete="CASCADE"),
        nullable=False
    )
    p_value: Mapped[float] = mapped_column(Float, nullable=False)
    ate_magnitude: Mapped[float] = mapped_column(Float, nullable=False) # Average Treatment Effect
    confidence_interval_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_interval_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confounders: Mapped[Optional[Dict[str, float]]] = mapped_column(JSON, nullable=True) # JSON object
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now())

    __table_args__ = (
        Index("idx_causal_results_hypothesis_id", "hypothesis_id"),
    )

    def __repr__(self) -> str:
        return f"<CausalResult(id={self.result_id}, hypothesis={self.hypothesis_id}, p_value={self.p_value}, ate={self.ate_magnitude})>"
