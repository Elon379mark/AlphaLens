import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Sequence, Iterator, AsyncIterator, Tuple
from typing_extensions import TypedDict
from enum import Enum

from sqlalchemy import select, and_, or_, update as sql_update, delete as sql_delete
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

# LangGraph checkpoint imports
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
    get_checkpoint_id,
)
from langgraph.checkpoint.base import WRITES_IDX_MAP
from langchain_core.runnables import RunnableConfig

# Database imports
from database.session import DatabaseSessionManager, db_manager
from database.models.hypothesis import Hypothesis
from database.models.signal import Signal
from database.models.causal_result import CausalResult
from database.models.portfolio import PortfolioAllocation, PortfolioReturn
from database.models.agent import AgentLog, WorkflowState, AgentMemory
from database.models.checkpoint import CheckpointSave, CheckpointBlob, CheckpointWrite

# Repositories
from database.repositories.hypothesis_repository import HypothesisRepository
from database.repositories.signal_repository import SignalRepository
from database.repositories.causal_result_repository import CausalResultRepository
from database.repositories.workflow_repository import WorkflowRepository
from database.repositories.market_data_repository import MarketDataRepository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph Shared State Schema (Requirement 12)
# ---------------------------------------------------------------------------
class AlphaLensState(TypedDict):
    hypothesis_id: str
    predictor_variable: str
    target_asset_class: str
    predicted_direction: str # "positive" | "negative"
    confidence: float
    p_value: float
    ate_magnitude: float
    sharpe_ratio: float


# ---------------------------------------------------------------------------
# Agent Database Service Layer (Requirement 11)
# ---------------------------------------------------------------------------
class DatabaseAgentService:
    """
    Coordinates repository operations and transaction management for the Agentic workflow.
    """
    def __init__(self, session_manager: Optional[DatabaseSessionManager] = None):
        self.db = session_manager or db_manager

    async def save_hypothesis(self, hypothesis: Union[dict, Any]) -> Hypothesis:
        """
        Creates or updates a hypothesis record.
        """
        if hasattr(hypothesis, "model_dump"):
            data = hypothesis.model_dump()
        elif hasattr(hypothesis, "dict"):
            data = hypothesis.dict()
        else:
            data = dict(hypothesis)

        # Handle predicted direction enum
        if isinstance(data.get("predicted_direction"), Enum):
            data["predicted_direction"] = data["predicted_direction"].value

        async with self.db.async_session() as session:
            repo = HypothesisRepository(session)
            h_id = data.get("hypothesis_id")
            existing = await repo.get_by_id(h_id) if h_id else None
            
            if existing:
                updated = await repo.update(h_id, **data)
                logger.info(f"DatabaseAgentService: Updated hypothesis {h_id}")
                return updated
            else:
                if not h_id:
                    data["hypothesis_id"] = f"H-{uuid.uuid4().hex[:8].upper()}"
                new_h = await repo.create(**data)
                logger.info(f"DatabaseAgentService: Saved new hypothesis {data['hypothesis_id']}")
                return new_h

    async def save_signal(self, signal_data: dict) -> Signal:
        """
        Creates or updates a generated signal.
        """
        async with self.db.async_session() as session:
            repo = SignalRepository(session)
            sig_id = signal_data.get("signal_id")
            timestamp = signal_data.get("timestamp") or datetime.now(timezone.utc)
            
            # Check composite PK (timestamp, signal_id)
            existing = None
            if sig_id and timestamp:
                existing = await repo.get_by_id((timestamp, sig_id))
            
            if existing:
                updated = await repo.update((timestamp, sig_id), **signal_data)
                logger.info(f"DatabaseAgentService: Updated signal {sig_id}")
                return updated
            else:
                if not sig_id:
                    signal_data["signal_id"] = f"S-{uuid.uuid4().hex[:8].upper()}"
                if "timestamp" not in signal_data:
                    signal_data["timestamp"] = timestamp
                new_sig = await repo.create(**signal_data)
                logger.info(f"DatabaseAgentService: Saved new signal {signal_data['signal_id']}")
                return new_sig

    async def save_causal_result(self, causal_data: dict) -> CausalResult:
        """
        Saves or updates causal validation results.
        """
        async with self.db.async_session() as session:
            repo = CausalResultRepository(session)
            r_id = causal_data.get("result_id")
            existing = await repo.get_by_id(r_id) if r_id else None
            
            if existing:
                updated = await repo.update(r_id, **causal_data)
                logger.info(f"DatabaseAgentService: Updated causal result {r_id}")
                return updated
            else:
                if not r_id:
                    causal_data["result_id"] = f"R-{uuid.uuid4().hex[:8].upper()}"
                new_r = await repo.create(**causal_data)
                logger.info(f"DatabaseAgentService: Saved new causal result {causal_data['result_id']}")
                return new_r

    async def save_portfolio(
        self,
        portfolio_id: str,
        timestamp: datetime,
        allocations: List[dict],
        returns_val: Optional[float] = None,
        equity: Optional[float] = None
    ) -> None:
        """
        Saves portfolio allocations and returns in a single transaction.
        """
        async with self.db.async_session() as session:
            # Save allocations
            for alloc in allocations:
                alloc_id = f"A-{uuid.uuid4().hex[:8].upper()}"
                allocation = PortfolioAllocation(
                    allocation_id=alloc_id,
                    portfolio_id=portfolio_id,
                    symbol=alloc["symbol"],
                    weight=alloc["weight"],
                    timestamp=timestamp
                )
                session.add(allocation)
            
            # Save returns if provided
            if returns_val is not None:
                p_return = PortfolioReturn(
                    timestamp=timestamp,
                    portfolio_id=portfolio_id,
                    returns=returns_val,
                    equity=equity
                )
                session.add(p_return)
            
            await session.flush()
            logger.info(f"DatabaseAgentService: Saved portfolio {portfolio_id} data at {timestamp}")

    async def load_agent_memory(self, agent_name: str, key: str) -> Optional[dict]:
        """Loads specific agent long-term memory."""
        async with self.db.async_session() as session:
            stmt = select(AgentMemory).where(
                and_(AgentMemory.agent_name == agent_name, AgentMemory.memory_key == key)
            )
            result = await session.execute(stmt)
            memory = result.scalars().first()
            return memory.memory_value if memory else None

    async def save_agent_memory(self, agent_name: str, key: str, value: dict) -> None:
        """Saves or updates agent long-term memory."""
        async with self.db.async_session() as session:
            stmt = select(AgentMemory).where(
                and_(AgentMemory.agent_name == agent_name, AgentMemory.memory_key == key)
            )
            result = await session.execute(stmt)
            memory = result.scalars().first()
            
            if memory:
                memory.memory_value = value
                memory.updated_at = datetime.now(timezone.utc)
            else:
                memory_id = f"M-{uuid.uuid4().hex[:8].upper()}"
                new_memory = AgentMemory(
                    memory_id=memory_id,
                    agent_name=agent_name,
                    memory_key=key,
                    memory_value=value
                )
                session.add(new_memory)
            await session.flush()
            logger.info(f"DatabaseAgentService: Saved agent memory for {agent_name} key {key}")

    async def save_workflow_state(
        self,
        workflow_id: str,
        state_data: dict,
        status: str,
        current_node: Optional[str] = None
    ) -> None:
        """Saves or updates active workflow state."""
        async with self.db.async_session() as session:
            repo = WorkflowRepository(session)
            existing = await repo.get_by_id(workflow_id)
            
            if existing:
                await repo.update(
                    workflow_id,
                    state_data=state_data,
                    status=status,
                    current_node=current_node,
                    updated_at=datetime.now(timezone.utc)
                )
                logger.info(f"DatabaseAgentService: Updated workflow {workflow_id} state to {status}")
            else:
                await repo.create(
                    workflow_id=workflow_id,
                    state_data=state_data,
                    status=status,
                    current_node=current_node
                )
                logger.info(f"DatabaseAgentService: Created new workflow state {workflow_id}")


# ---------------------------------------------------------------------------
# PostgreSQL backed LangGraph Checkpoint Saver (Requirement 13)
# ---------------------------------------------------------------------------
class PostgresCheckpointSaver(BaseCheckpointSaver):
    """
    Production-grade LangGraph Checkpoint Saver using PostgreSQL + TimescaleDB.
    Gracefully supports sync and async query contexts on top of DatabaseSessionManager.
    """
    def __init__(self, session_manager: Optional[DatabaseSessionManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.db = session_manager or db_manager

    # -----------------------------------------------------------------------
    # Helper loader method for versioned channel blobs
    # -----------------------------------------------------------------------
    def _load_blobs(self, session: Union[Session, AsyncSession], thread_id: str, checkpoint_ns: str, versions: ChannelVersions) -> dict:
        """Loads versioned channel blobs synchronously."""
        if not versions:
            return {}
        
        conditions = [
            and_(CheckpointBlob.channel == k, CheckpointBlob.version == ver)
            for k, ver in versions.items()
        ]
        
        # Load from sync session
        stmt = select(CheckpointBlob).where(
            and_(
                CheckpointBlob.thread_id == thread_id,
                CheckpointBlob.checkpoint_ns == checkpoint_ns,
                or_(*conditions)
            )
        )
        
        blobs = session.execute(stmt).scalars().all()
        
        result = {}
        for blob in blobs:
            if blob.blob_type != "empty":
                result[blob.channel] = self.serde.loads_typed((blob.blob_type, blob.blob_bytes))
        return result

    async def _aload_blobs(self, session: AsyncSession, thread_id: str, checkpoint_ns: str, versions: ChannelVersions) -> dict:
        """Loads versioned channel blobs asynchronously."""
        if not versions:
            return {}
        
        conditions = [
            and_(CheckpointBlob.channel == k, CheckpointBlob.version == ver)
            for k, ver in versions.items()
        ]
        
        stmt = select(CheckpointBlob).where(
            and_(
                CheckpointBlob.thread_id == thread_id,
                CheckpointBlob.checkpoint_ns == checkpoint_ns,
                or_(*conditions)
            )
        )
        
        result_set = await session.execute(stmt)
        blobs = result_set.scalars().all()
        
        result = {}
        for blob in blobs:
            if blob.blob_type != "empty":
                result[blob.channel] = self.serde.loads_typed((blob.blob_type, blob.blob_bytes))
        return result

    # -----------------------------------------------------------------------
    # Synchronous API implementation
    # -----------------------------------------------------------------------
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Retrieve a checkpoint tuple from database storage."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        with self.db.sync_session() as session:
            # Query CheckpointSave
            if checkpoint_id:
                stmt = select(CheckpointSave).filter_by(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id
                )
            else:
                stmt = select(CheckpointSave).where(
                    and_(
                        CheckpointSave.thread_id == thread_id,
                        CheckpointSave.checkpoint_ns == checkpoint_ns
                    )
                ).order_by(CheckpointSave.checkpoint_id.desc()).limit(1)

            checkpoint_save = session.execute(stmt).scalars().first()
            if not checkpoint_save:
                return None

            checkpoint_id = checkpoint_save.checkpoint_id
            
            # Reconstruct Checkpoint Dict
            checkpoint_data = self.serde.loads_typed(
                (checkpoint_save.checkpoint_type, checkpoint_save.checkpoint_bytes)
            )
            
            # Reconstruct Metadata
            metadata = self.serde.loads_typed(
                (checkpoint_save.metadata_type, checkpoint_save.metadata_bytes)
            )

            # Reconstruct channel values
            channel_values = self._load_blobs(
                session, thread_id, checkpoint_ns, checkpoint_data["channel_versions"]
            )
            checkpoint_data["channel_values"] = channel_values

            # Retrieve Writes
            write_stmt = select(CheckpointWrite).filter_by(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id
            )
            writes = session.execute(write_stmt).scalars().all()
            pending_writes = [
                (w.task_id, w.channel, self.serde.loads_typed((w.value_type, w.value_bytes)))
                for w in writes
            ]

            parent_config = None
            if checkpoint_save.parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_save.parent_checkpoint_id,
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint_data,
                metadata=metadata,
                pending_writes=pending_writes,
                parent_config=parent_config
            )

    def list(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints matching the filter criteria."""
        thread_id = config["configurable"]["thread_id"] if config else None
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") if config else None
        config_checkpoint_id = get_checkpoint_id(config) if config else None

        with self.db.sync_session() as session:
            stmt = select(CheckpointSave)
            
            filters = []
            if thread_id is not None:
                filters.append(CheckpointSave.thread_id == thread_id)
            if checkpoint_ns is not None:
                filters.append(CheckpointSave.checkpoint_ns == checkpoint_ns)
            if config_checkpoint_id is not None:
                filters.append(CheckpointSave.checkpoint_id == config_checkpoint_id)
            if before is not None:
                before_id = get_checkpoint_id(before)
                if before_id:
                    filters.append(CheckpointSave.checkpoint_id < before_id)
            
            if filters:
                stmt = stmt.where(and_(*filters))
            
            stmt = stmt.order_by(CheckpointSave.checkpoint_id.desc())
            if limit:
                stmt = stmt.limit(limit)

            saves = session.execute(stmt).scalars().all()

            for save in saves:
                # filter by metadata
                meta = self.serde.loads_typed((save.metadata_type, save.metadata_bytes))
                if filter and not all(meta.get(k) == v for k, v in filter.items()):
                    continue

                checkpoint_data = self.serde.loads_typed((save.checkpoint_type, save.checkpoint_bytes))
                channel_values = self._load_blobs(session, save.thread_id, save.checkpoint_ns, checkpoint_data["channel_versions"])
                checkpoint_data["channel_values"] = channel_values

                write_stmt = select(CheckpointWrite).filter_by(
                    thread_id=save.thread_id,
                    checkpoint_ns=save.checkpoint_ns,
                    checkpoint_id=save.checkpoint_id
                )
                writes = session.execute(write_stmt).scalars().all()
                pending_writes = [
                    (w.task_id, w.channel, self.serde.loads_typed((w.value_type, w.value_bytes)))
                    for w in writes
                ]

                parent_config = None
                if save.parent_checkpoint_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": save.thread_id,
                            "checkpoint_ns": save.checkpoint_ns,
                            "checkpoint_id": save.parent_checkpoint_id,
                        }
                    }

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": save.thread_id,
                            "checkpoint_ns": save.checkpoint_ns,
                            "checkpoint_id": save.checkpoint_id,
                        }
                    },
                    checkpoint=checkpoint_data,
                    metadata=meta,
                    pending_writes=pending_writes,
                    parent_config=parent_config
                )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions
    ) -> RunnableConfig:
        """Save a new state checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        c = checkpoint.copy()
        values = c.pop("channel_values", {})

        with self.db.sync_session() as session:
            # 1. Upsert blobs for new versions
            for k, ver in new_versions.items():
                if k in values:
                    b_type, b_bytes = self.serde.dumps_typed(values[k])
                else:
                    b_type, b_bytes = "empty", b""
                
                # Check existing
                stmt = select(CheckpointBlob).filter_by(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    channel=k,
                    version=ver
                )
                existing_blob = session.execute(stmt).scalars().first()
                if not existing_blob:
                    session.add(
                        CheckpointBlob(
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            channel=k,
                            version=ver,
                            blob_type=b_type,
                            blob_bytes=b_bytes
                        )
                    )

            # 2. Save CheckpointSave
            c_type, c_bytes = self.serde.dumps_typed(c)
            m_type, m_bytes = self.serde.dumps_typed(metadata)
            
            stmt = select(CheckpointSave).filter_by(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id
            )
            existing_save = session.execute(stmt).scalars().first()
            if existing_save:
                existing_save.checkpoint_type = c_type
                existing_save.checkpoint_bytes = c_bytes
                existing_save.metadata_type = m_type
                existing_save.metadata_bytes = m_bytes
                existing_save.parent_checkpoint_id = parent_checkpoint_id
            else:
                session.add(
                    CheckpointSave(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        parent_checkpoint_id=parent_checkpoint_id,
                        checkpoint_type=c_type,
                        checkpoint_bytes=c_bytes,
                        metadata_type=m_type,
                        metadata_bytes=m_bytes
                    )
                )
            
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = ""
    ) -> None:
        """Persist pending writes."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self.db.sync_session() as session:
            for idx, (channel, val) in enumerate(writes):
                v_type, v_bytes = self.serde.dumps_typed(val)
                stmt = select(CheckpointWrite).filter_by(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx=WRITES_IDX_MAP.get(channel, idx)
                )
                existing = session.execute(stmt).scalars().first()
                if not existing:
                    session.add(
                        CheckpointWrite(
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            checkpoint_id=checkpoint_id,
                            task_id=task_id,
                            idx=WRITES_IDX_MAP.get(channel, idx),
                            channel=channel,
                            value_type=v_type,
                            value_bytes=v_bytes,
                            task_path=task_path
                        )
                    )

    # -----------------------------------------------------------------------
    # Asynchronous API implementation
    # -----------------------------------------------------------------------
    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Async retrieve a checkpoint tuple."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        async with self.db.async_session() as session:
            if checkpoint_id:
                stmt = select(CheckpointSave).filter_by(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id
                )
            else:
                stmt = select(CheckpointSave).where(
                    and_(
                        CheckpointSave.thread_id == thread_id,
                        CheckpointSave.checkpoint_ns == checkpoint_ns
                    )
                ).order_by(CheckpointSave.checkpoint_id.desc()).limit(1)

            res = await session.execute(stmt)
            checkpoint_save = res.scalars().first()
            if not checkpoint_save:
                return None

            checkpoint_id = checkpoint_save.checkpoint_id
            checkpoint_data = self.serde.loads_typed(
                (checkpoint_save.checkpoint_type, checkpoint_save.checkpoint_bytes)
            )
            metadata = self.serde.loads_typed(
                (checkpoint_save.metadata_type, checkpoint_save.metadata_bytes)
            )

            channel_values = await self._aload_blobs(
                session, thread_id, checkpoint_ns, checkpoint_data["channel_versions"]
            )
            checkpoint_data["channel_values"] = channel_values

            write_stmt = select(CheckpointWrite).filter_by(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id
            )
            write_res = await session.execute(write_stmt)
            writes = write_res.scalars().all()
            pending_writes = [
                (w.task_id, w.channel, self.serde.loads_typed((w.value_type, w.value_bytes)))
                for w in writes
            ]

            parent_config = None
            if checkpoint_save.parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_save.parent_checkpoint_id,
                    }
                }

            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint=checkpoint_data,
                metadata=metadata,
                pending_writes=pending_writes,
                parent_config=parent_config
            )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None
    ) -> AsyncIterator[CheckpointTuple]:
        """Async list checkpoints matching the filter criteria."""
        thread_id = config["configurable"]["thread_id"] if config else None
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "") if config else None
        config_checkpoint_id = get_checkpoint_id(config) if config else None

        async with self.db.async_session() as session:
            stmt = select(CheckpointSave)
            
            filters = []
            if thread_id is not None:
                filters.append(CheckpointSave.thread_id == thread_id)
            if checkpoint_ns is not None:
                filters.append(CheckpointSave.checkpoint_ns == checkpoint_ns)
            if config_checkpoint_id is not None:
                filters.append(CheckpointSave.checkpoint_id == config_checkpoint_id)
            if before is not None:
                before_id = get_checkpoint_id(before)
                if before_id:
                    filters.append(CheckpointSave.checkpoint_id < before_id)
            
            if filters:
                stmt = stmt.where(and_(*filters))
            
            stmt = stmt.order_by(CheckpointSave.checkpoint_id.desc())
            if limit:
                stmt = stmt.limit(limit)

            res = await session.execute(stmt)
            saves = res.scalars().all()

            for save in saves:
                meta = self.serde.loads_typed((save.metadata_type, save.metadata_bytes))
                if filter and not all(meta.get(k) == v for k, v in filter.items()):
                    continue

                checkpoint_data = self.serde.loads_typed((save.checkpoint_type, save.checkpoint_bytes))
                channel_values = await self._aload_blobs(session, save.thread_id, save.checkpoint_ns, checkpoint_data["channel_versions"])
                checkpoint_data["channel_values"] = channel_values

                write_stmt = select(CheckpointWrite).filter_by(
                    thread_id=save.thread_id,
                    checkpoint_ns=save.checkpoint_ns,
                    checkpoint_id=save.checkpoint_id
                )
                write_res = await session.execute(write_stmt)
                writes = write_res.scalars().all()
                pending_writes = [
                    (w.task_id, w.channel, self.serde.loads_typed((w.value_type, w.value_bytes)))
                    for w in writes
                ]

                parent_config = None
                if save.parent_checkpoint_id:
                    parent_config = {
                        "configurable": {
                            "thread_id": save.thread_id,
                            "checkpoint_ns": save.checkpoint_ns,
                            "checkpoint_id": save.parent_checkpoint_id,
                        }
                    }

                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": save.thread_id,
                            "checkpoint_ns": save.checkpoint_ns,
                            "checkpoint_id": save.checkpoint_id,
                        }
                    },
                    checkpoint=checkpoint_data,
                    metadata=meta,
                    pending_writes=pending_writes,
                    parent_config=parent_config
                )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions
    ) -> RunnableConfig:
        """Async save a new state checkpoint."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        c = checkpoint.copy()
        values = c.pop("channel_values", {})

        async with self.db.async_session() as session:
            for k, ver in new_versions.items():
                if k in values:
                    b_type, b_bytes = self.serde.dumps_typed(values[k])
                else:
                    b_type, b_bytes = "empty", b""
                
                stmt = select(CheckpointBlob).filter_by(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    channel=k,
                    version=ver
                )
                res = await session.execute(stmt)
                existing_blob = res.scalars().first()
                if not existing_blob:
                    session.add(
                        CheckpointBlob(
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            channel=k,
                            version=ver,
                            blob_type=b_type,
                            blob_bytes=b_bytes
                        )
                    )

            c_type, c_bytes = self.serde.dumps_typed(c)
            m_type, m_bytes = self.serde.dumps_typed(metadata)
            
            stmt = select(CheckpointSave).filter_by(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id
            )
            res = await session.execute(stmt)
            existing_save = res.scalars().first()
            if existing_save:
                existing_save.checkpoint_type = c_type
                existing_save.checkpoint_bytes = c_bytes
                existing_save.metadata_type = m_type
                existing_save.metadata_bytes = m_bytes
                existing_save.parent_checkpoint_id = parent_checkpoint_id
            else:
                session.add(
                    CheckpointSave(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        parent_checkpoint_id=parent_checkpoint_id,
                        checkpoint_type=c_type,
                        checkpoint_bytes=c_bytes,
                        metadata_type=m_type,
                        metadata_bytes=m_bytes
                    )
                )
            
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = ""
    ) -> None:
        """Async persist pending writes."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]

        async with self.db.async_session() as session:
            for idx, (channel, val) in enumerate(writes):
                v_type, v_bytes = self.serde.dumps_typed(val)
                stmt = select(CheckpointWrite).filter_by(
                    thread_id=thread_id,
                    checkpoint_ns=checkpoint_ns,
                    checkpoint_id=checkpoint_id,
                    task_id=task_id,
                    idx=WRITES_IDX_MAP.get(channel, idx)
                )
                res = await session.execute(stmt)
                existing = res.scalars().first()
                if not existing:
                    session.add(
                        CheckpointWrite(
                            thread_id=thread_id,
                            checkpoint_ns=checkpoint_ns,
                            checkpoint_id=checkpoint_id,
                            task_id=task_id,
                            idx=WRITES_IDX_MAP.get(channel, idx),
                            channel=channel,
                            value_type=v_type,
                            value_bytes=v_bytes,
                            task_path=task_path
                        )
                    )
