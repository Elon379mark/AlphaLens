import os
import sys
import unittest
import tempfile
import asyncio
from datetime import datetime

# Adjust path to find alphalens
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.session import DatabaseSessionManager
from alphalens.agents.memory import AgentMemoryEngine

class TestMemoryIntegration(unittest.TestCase):
    def setUp(self):
        # Create temp file for SQLite DB fallback
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.db_manager = DatabaseSessionManager(sqlite_path=self.db_path)
        self.db_manager.create_tables()
        self.memory = AgentMemoryEngine(session_manager=self.db_manager)

    def tearDown(self):
        asyncio.run(self.db_manager.close_all())

        try:
            os.close(self.db_fd)
            os.unlink(self.db_path)
        except OSError:
            pass

    def test_episodic_memory(self):
        async def run_test():
            # Add logs
            await self.memory.add_episode_log("W-TEST-1", "literature_agent", "INFO", "Generated hypothesis H-01")
            await self.memory.add_episode_log("W-TEST-1", "signal_gen_agent", "INFO", "Computed factor matrix")
            await self.memory.add_episode_log("W-TEST-2", "literature_agent", "WARNING", "Failed to decode JSON")

            # Query by workflow
            w1_logs = await self.memory.get_episodes_by_workflow("W-TEST-1")
            self.assertEqual(len(w1_logs), 2)
            self.assertEqual(w1_logs[0].agent_name, "literature_agent")
            self.assertEqual(w1_logs[1].agent_name, "signal_gen_agent")

            # Query by agent
            lit_logs = await self.memory.get_episodes_by_agent("literature_agent")
            self.assertEqual(len(lit_logs), 2)

        asyncio.run(run_test())

    def test_semantic_memory(self):
        async def run_test():
            fact = {"reason": "p-value 0.12 >= 0.05", "suggestions": ["Consider shorter lags"]}
            await self.memory.store_semantic_fact("literature_agent", "refinement", fact)

            # Get fact
            retrieved = await self.memory.get_semantic_fact("literature_agent", "refinement")
            self.assertEqual(retrieved, fact)

            # List facts
            facts = await self.memory.list_agent_semantic_memory("literature_agent")
            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0].memory_key, "refinement")

        asyncio.run(run_test())
