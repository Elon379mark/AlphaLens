from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, LargeBinary, Integer, DateTime, PrimaryKeyConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from database.base import Base

class CheckpointSave(Base):
    """
    Saves state checkpoints of LangGraph agent executions.
    """
    __tablename__ = "checkpoint_saves"

    thread_id: Mapped[str] = mapped_column(String(100), nullable=False)
    checkpoint_ns: Mapped[str] = mapped_column(String(100), default="", server_default="")
    checkpoint_id: Mapped[str] = mapped_column(String(100), nullable=False)
    parent_checkpoint_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    checkpoint_type: Mapped[str] = mapped_column(String(100), nullable=False)
    checkpoint_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    metadata_type: Mapped[str] = mapped_column(String(100), nullable=False)
    metadata_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), server_default=func.now())

    __table_args__ = (
        PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id"),
    )

    def __repr__(self) -> str:
        return f"<CheckpointSave(thread={self.thread_id}, ns={self.checkpoint_ns}, id={self.checkpoint_id})>"


class CheckpointBlob(Base):
    """
    Saves individual channel state versions of LangGraph execution paths.
    """
    __tablename__ = "checkpoint_blobs"

    thread_id: Mapped[str] = mapped_column(String(100), nullable=False)
    checkpoint_ns: Mapped[str] = mapped_column(String(100), default="", server_default="")
    channel: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(100), nullable=False)
    blob_type: Mapped[str] = mapped_column(String(100), nullable=False)
    blob_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("thread_id", "checkpoint_ns", "channel", "version"),
    )

    def __repr__(self) -> str:
        return f"<CheckpointBlob(thread={self.thread_id}, ns={self.checkpoint_ns}, ch={self.channel}, ver={self.version})>"


class CheckpointWrite(Base):
    """
    Stores pending task writes before commitment in LangGraph.
    """
    __tablename__ = "checkpoint_writes"

    thread_id: Mapped[str] = mapped_column(String(100), nullable=False)
    checkpoint_ns: Mapped[str] = mapped_column(String(100), default="", server_default="")
    checkpoint_id: Mapped[str] = mapped_column(String(100), nullable=False)
    task_id: Mapped[str] = mapped_column(String(100), nullable=False)
    idx: Mapped[int] = mapped_column(Integer, nullable=False)
    channel: Mapped[str] = mapped_column(String(100), nullable=False)
    value_type: Mapped[str] = mapped_column(String(100), nullable=False)
    value_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    task_path: Mapped[str] = mapped_column(String(255), default="", server_default="")

    __table_args__ = (
        PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"),
    )

    def __repr__(self) -> str:
        return f"<CheckpointWrite(thread={self.thread_id}, ns={self.checkpoint_ns}, id={self.checkpoint_id}, task={self.task_id}, idx={self.idx})>"
