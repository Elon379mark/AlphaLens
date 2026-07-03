from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import String, Float, Text, JSON, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from database.base import Base

class Hypothesis(Base):
    """
    Represents an economic or statistical financial hypothesis.
    """
    __tablename__ = "hypotheses"

    hypothesis_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    predictor_variable: Mapped[str] = mapped_column(String(100), nullable=False)
    target_asset_class: Mapped[str] = mapped_column(String(100), nullable=False)
    predicted_direction: Mapped[str] = mapped_column(String(20), nullable=False) # e.g. "positive" or "negative"
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    theoretical_mechanism: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_references: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True) # JSON array of citations
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now())

    def __repr__(self) -> str:
        return f"<Hypothesis(id={self.hypothesis_id}, predictor={self.predictor_variable}, asset_class={self.target_asset_class})>"
