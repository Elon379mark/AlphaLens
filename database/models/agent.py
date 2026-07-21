from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import String, Text, JSON, DateTime, LargeBinary, Index, func
from sqlalchemy.orm import Mapped, mapped_column
from database.base import Base

class AgentLog(Base):
    """
    Agent execution logs for system auditing.
    Stores both human-readable messages and Protobuf-serialized binary payloads
    per Section 5.1 of the AlphaLens specification (Protocol Buffer event log).
    """
    __tablename__ = "agent_logs"

    log_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    workflow_id: Mapped[str] = mapped_column(String(50), nullable=False)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    log_level: Mapped[str] = mapped_column(String(10), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    # §5.1: Raw Protobuf-serialized AgentMessage binary for replay capability
    payload_binary: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now())

    __table_args__ = (
        Index("idx_agent_logs_workflow_id", "workflow_id"),
        Index("idx_agent_logs_agent_name", "agent_name"),
    )

    def __repr__(self) -> str:
        return f"<AgentLog(agent={self.agent_name}, level={self.log_level}, workflow={self.workflow_id})>"


class WorkflowState(Base):
    """
    Stores structured LangGraph shared states and run execution status.
    """
    __tablename__ = "workflow_states"

    workflow_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    state_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False) # e.g. "RUNNING", "COMPLETED", "FAILED"
    current_node: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("idx_workflow_states_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<WorkflowState(id={self.workflow_id}, status={self.status}, current_node={self.current_node})>"


class AgentMemory(Base):
    """
    Long-term agent memory storage.
    """
    __tablename__ = "agent_memory"

    memory_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False)
    memory_key: Mapped[str] = mapped_column(String(100), nullable=False)
    memory_value: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("idx_agent_memory_agent_key", "agent_name", "memory_key", unique=True),
    )

    def __repr__(self) -> str:
        return f"<AgentMemory(agent={self.agent_name}, key={self.memory_key})>"
