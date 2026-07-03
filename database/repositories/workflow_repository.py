from typing import List, Optional
from datetime import datetime
import uuid
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from database.models.agent import WorkflowState, AgentLog
from database.repositories.base import BaseRepository

class WorkflowRepository(BaseRepository[WorkflowState]):
    """
    Repository for managing workflow states and agent execution logs.
    """
    def __init__(self, session: AsyncSession):
        super().__init__(WorkflowState, session)

    async def add_log(self, workflow_id: str, agent_name: str, log_level: str, message: str) -> AgentLog:
        """Create and persist an agent log record."""
        log = AgentLog(
            log_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            agent_name=agent_name,
            log_level=log_level,
            message=message,
            timestamp=datetime.utcnow()
        )
        self.session.add(log)
        await self.session.flush()
        return log

    async def get_logs(self, workflow_id: str, limit: int = 100) -> List[AgentLog]:
        """Fetch all logs associated with a workflow session."""
        stmt = select(AgentLog).where(AgentLog.workflow_id == workflow_id).order_by(AgentLog.timestamp.asc()).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
