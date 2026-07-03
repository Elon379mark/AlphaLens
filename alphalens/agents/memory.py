import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy import and_
from sqlalchemy.future import select
from database.session import DatabaseSessionManager, db_manager
from database.models.agent import AgentLog, AgentMemory, WorkflowState

logger = logging.getLogger(__name__)

class AgentMemoryEngine:
    """
    Implements Episodic (logs and historical transitions) and Semantic (associative facts)
    memory stores for LangGraph agents. Working memory is persisted via CheckpointSaver.
    """
    def __init__(self, session_manager: Optional[DatabaseSessionManager] = None):
        self.db = session_manager or db_manager

    # --- EPISODIC MEMORY (Agent Logs & Action Trails) ---

    async def add_episode_log(
        self,
        workflow_id: str,
        agent_name: str,
        log_level: str,
        message: str
    ) -> AgentLog:
        """Adds a new execution log entry representing an agent episode/action."""
        async with self.db.async_session() as session:
            log_id = f"L-{uuid.uuid4().hex[:8].upper()}"
            log_entry = AgentLog(
                log_id=log_id,
                workflow_id=workflow_id,
                agent_name=agent_name,
                log_level=log_level,
                message=message,
                timestamp=datetime.now(timezone.utc)
            )
            session.add(log_entry)
            await session.commit()
            logger.debug(f"Episodic: Logged episode {log_id} for agent {agent_name}")
            return log_entry

    async def get_episodes_by_workflow(self, workflow_id: str) -> List[AgentLog]:
        """Retrieves all logged episodes for a specific workflow run."""
        async with self.db.async_session() as session:
            stmt = (
                select(AgentLog)
                .where(AgentLog.workflow_id == workflow_id)
                .order_by(AgentLog.timestamp.asc())
            )
            res = await session.execute(stmt)
            return list(res.scalars().all())

    async def get_episodes_by_agent(self, agent_name: str) -> List[AgentLog]:
        """Retrieves all logged episodes for a specific agent across all runs."""
        async with self.db.async_session() as session:
            stmt = (
                select(AgentLog)
                .where(AgentLog.agent_name == agent_name)
                .order_by(AgentLog.timestamp.desc())
            )
            res = await session.execute(stmt)
            return list(res.scalars().all())

    # --- SEMANTIC MEMORY (Associative Key-Value Facts) ---

    async def store_semantic_fact(
        self,
        agent_name: str,
        memory_key: str,
        memory_value: Dict[str, Any]
    ) -> AgentMemory:
        """Stores or updates a long-term fact/concept inside agent memory."""
        async with self.db.async_session() as session:
            stmt = select(AgentMemory).where(
                and_(
                    AgentMemory.agent_name == agent_name,
                    AgentMemory.memory_key == memory_key
                )
            )
            res = await session.execute(stmt)
            existing = res.scalars().first()

            if existing:
                existing.memory_value = memory_value
                existing.updated_at = datetime.now(timezone.utc)
                await session.commit()
                logger.debug(f"Semantic: Updated fact for agent {agent_name} key: {memory_key}")
                return existing
            else:
                memory_id = f"M-{uuid.uuid4().hex[:8].upper()}"
                new_fact = AgentMemory(
                    memory_id=memory_id,
                    agent_name=agent_name,
                    memory_key=memory_key,
                    memory_value=memory_value,
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(new_fact)
                await session.commit()
                logger.debug(f"Semantic: Created fact {memory_id} for agent {agent_name} key: {memory_key}")
                return new_fact

    async def get_semantic_fact(self, agent_name: str, memory_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single long-term fact by key."""
        async with self.db.async_session() as session:
            stmt = select(AgentMemory).where(
                and_(
                    AgentMemory.agent_name == agent_name,
                    AgentMemory.memory_key == memory_key
                )
            )
            res = await session.execute(stmt)
            fact = res.scalars().first()
            return fact.memory_value if fact else None

    async def list_agent_semantic_memory(self, agent_name: str) -> List[AgentMemory]:
        """Lists all semantic memory entries for a given agent."""
        async with self.db.async_session() as session:
            stmt = select(AgentMemory).where(AgentMemory.agent_name == agent_name)
            res = await session.execute(stmt)
            return list(res.scalars().all())
