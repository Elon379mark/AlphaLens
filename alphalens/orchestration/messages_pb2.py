import json
from enum import IntEnum
from typing import List, Optional

class Direction(IntEnum):
    POSITIVE = 0;
    NEGATIVE = 1;

class HypothesisPayload:
    def __init__(self, 
                 hypothesis_id: str = "", 
                 predictor_variable: str = "", 
                 target_asset_class: str = "", 
                 predicted_direction: Direction = Direction.POSITIVE, 
                 confidence: float = 0.0, 
                 theoretical_mechanism: str = "", 
                 source_references: Optional[List[str]] = None):
        self.hypothesis_id = hypothesis_id
        self.predictor_variable = predictor_variable
        self.target_asset_class = target_asset_class
        self.predicted_direction = predicted_direction
        self.confidence = confidence
        self.theoretical_mechanism = theoretical_mechanism
        self.source_references = source_references or []

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "predictor_variable": self.predictor_variable,
            "target_asset_class": self.target_asset_class,
            "predicted_direction": int(self.predicted_direction),
            "confidence": self.confidence,
            "theoretical_mechanism": self.theoretical_mechanism,
            "source_references": self.source_references
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'HypothesisPayload':
        return cls(
            hypothesis_id=data.get("hypothesis_id", ""),
            predictor_variable=data.get("predictor_variable", ""),
            target_asset_class=data.get("target_asset_class", ""),
            predicted_direction=Direction(data.get("predicted_direction", 0)),
            confidence=data.get("confidence", 0.0),
            theoretical_mechanism=data.get("theoretical_mechanism", ""),
            source_references=data.get("source_references", [])
        )


class AgentMessage:
    def __init__(self, 
                 sender: str = "", 
                 recipient: str = "", 
                 timestamp: float = 0.0, 
                 priority: int = 0, 
                 hypothesis: Optional[HypothesisPayload] = None, 
                 p_value: float = 0.0, 
                 ate_magnitude: float = 0.0, 
                 sharpe_ratio: float = 0.0, 
                 information_coefficient: float = 0.0, 
                 information_ratio: float = 0.0, 
                 half_life_days: float = 0.0, 
                 error_message: str = ""):
        self.sender = sender
        self.recipient = recipient
        self.timestamp = timestamp
        self.priority = priority
        self.hypothesis = hypothesis or HypothesisPayload()
        self.p_value = p_value
        self.ate_magnitude = ate_magnitude
        self.sharpe_ratio = sharpe_ratio
        self.information_coefficient = information_coefficient
        self.information_ratio = information_ratio
        self.half_life_days = half_life_days
        self.error_message = error_message

    def SerializeToString(self) -> bytes:
        """
        Serializes the message to bytes using UTF-8 JSON.
        """
        data = {
            "sender": self.sender,
            "recipient": self.recipient,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "hypothesis": self.hypothesis.to_dict(),
            "p_value": self.p_value,
            "ate_magnitude": self.ate_magnitude,
            "sharpe_ratio": self.sharpe_ratio,
            "information_coefficient": self.information_coefficient,
            "information_ratio": self.information_ratio,
            "half_life_days": self.half_life_days,
            "error_message": self.error_message
        }
        return json.dumps(data).encode("utf-8")

    def ParseFromString(self, serialized_data: bytes):
        """
        Parses the message from bytes.

        .. deprecated::
            Use the :meth:`from_string` classmethod instead, which returns
            a new ``AgentMessage`` instance rather than mutating in place.
        """
        data = json.loads(serialized_data.decode("utf-8"))
        self.sender = data.get("sender", "")
        self.recipient = data.get("recipient", "")
        self.timestamp = data.get("timestamp", 0.0)
        self.priority = data.get("priority", 0)
        
        hyp_data = data.get("hypothesis", {})
        self.hypothesis = HypothesisPayload.from_dict(hyp_data)
        
        self.p_value = data.get("p_value", 0.0)
        self.ate_magnitude = data.get("ate_magnitude", 0.0)
        self.sharpe_ratio = data.get("sharpe_ratio", 0.0)
        self.information_coefficient = data.get("information_coefficient", 0.0)
        self.information_ratio = data.get("information_ratio", 0.0)
        self.half_life_days = data.get("half_life_days", 0.0)
        self.error_message = data.get("error_message", "")
        return self

    @classmethod
    def from_string(cls, data: bytes) -> 'AgentMessage':
        """Create a new AgentMessage by deserializing *data*.

        This is the preferred alternative to :meth:`ParseFromString`
        because it returns a fresh instance instead of mutating an
        existing one.

        Args:
            data: UTF-8 encoded JSON bytes produced by
                :meth:`SerializeToString`.

        Returns:
            A fully populated ``AgentMessage``.
        """
        msg = cls()
        msg.ParseFromString(data)
        return msg
