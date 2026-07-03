from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class PredictedDirection(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"

class HypothesisSchema(BaseModel):
    hypothesis_id: str = Field(..., description="Unique index key identifying the hypothesis")
    predictor_variable: str = Field(..., description="Name of the predictor variable matching global features")
    target_asset_class: str = Field(..., description="Target asset class name")
    predicted_direction: PredictedDirection = Field(..., description="Direction of predicted impact: positive or negative")
    confidence: float = Field(..., ge=0.00, le=1.00, description="Confidence of the hypothesis bounded between 0.00 and 1.00")
    theoretical_mechanism: str = Field(..., description="Theoretical economic mechanism explaining the link")
    source_references: List[str] = Field(default_factory=list, description="Citations or reference source strings (e.g. arXiv IDs)")

class AgentStateSchema(BaseModel):
    """
    Representation of the LangGraph shared state memory block.
    """
    hypothesis: Optional[HypothesisSchema] = None
    p_value: Optional[float] = Field(default=None, description="p-value from causal DAG validation / ATE estimation")
    ate_magnitude: Optional[float] = Field(default=None, description="Average Treatment Effect magnitude from DML")
    sharpe_ratio: Optional[float] = Field(default=None, description="Backtested Sharpe ratio from simulation framework")
    information_coefficient: Optional[float] = Field(default=None, description="Information Coefficient (IC)")
    information_ratio: Optional[float] = Field(default=None, description="Information Ratio (ICIR)")
    half_life_days: Optional[float] = Field(default=None, description="Signal half-life in days")
    current_node: str = Field(default="start", description="Name of the currently active agent/node")
    error_message: Optional[str] = Field(default=None, description="Details of any execution or validation error")
