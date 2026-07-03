import os
import sys
import unittest
import tempfile
import asyncio
from datetime import datetime, timedelta, timezone

# Adjust path to find alphalens / database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.session import DatabaseSessionManager
from database.repositories.hypothesis_repository import HypothesisRepository
from database.repositories.signal_repository import SignalRepository
from database.repositories.causal_result_repository import CausalResultRepository
from database.repositories.market_data_repository import MarketDataRepository
from database.repositories.workflow_repository import WorkflowRepository
from database.service import DatabaseAgentService, PostgresCheckpointSaver

class TestDatabaseLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temp database file path
        cls.db_fd, cls.db_path = tempfile.mkstemp()
        cls.db = DatabaseSessionManager(sqlite_path=cls.db_path)
        cls.db.create_tables()
        cls.service = DatabaseAgentService(session_manager=cls.db)
        cls.checkpointer = PostgresCheckpointSaver(session_manager=cls.db)

    @classmethod
    def tearDownClass(cls):
        # Close engine connection and delete temp file
        # We must run the async close in an event loop
        asyncio.run(cls.db.close_all())
        
        try:
            os.close(cls.db_fd)
            os.unlink(cls.db_path)
        except OSError:
            pass

    def run_async(self, coro):
        """Helper to run async coroutines in sync test context."""
        return asyncio.run(coro)

    def test_hypothesis_repository_crud(self):
        """Verifies CRUD operations on Hypothesis repository."""
        async def run_test():
            async with self.db.async_session() as session:
                repo = HypothesisRepository(session)
                
                # Create
                h = await repo.create(
                    hypothesis_id="H-UNIT-TEST",
                    predictor_variable="fed_rates",
                    target_asset_class="bonds",
                    predicted_direction="positive",
                    confidence=0.75,
                    theoretical_mechanism="Fed rate changes drive yields.",
                    source_references=["arXiv:2020"]
                )
                self.assertEqual(h.hypothesis_id, "H-UNIT-TEST")
                
                # Read
                retrieved = await repo.get_by_id("H-UNIT-TEST")
                self.assertIsNotNone(retrieved)
                self.assertEqual(retrieved.predictor_variable, "fed_rates")
                
                # Update
                updated = await repo.update("H-UNIT-TEST", confidence=0.80)
                self.assertEqual(updated.confidence, 0.80)
                
                # List
                all_h = await repo.list_all()
                self.assertTrue(len(all_h) >= 1)
                
                # Delete
                deleted = await repo.delete("H-UNIT-TEST")
                self.assertTrue(deleted)
                
                # Verify deleted
                retrieved_deleted = await repo.get_by_id("H-UNIT-TEST")
                self.assertIsNone(retrieved_deleted)

        self.run_async(run_test())

    def test_temporal_data_no_look_ahead_bias(self):
        """Verifies that the market data temporal layer prevents look-ahead bias."""
        async def run_test():
            base_time = datetime(2025, 6, 1, 10, 0, 0)
            async with self.db.async_session() as session:
                repo = MarketDataRepository(session)
                
                # Add historical price bars
                await repo.create(timestamp=base_time, symbol="MSFT", open=300.0, high=301.0, low=299.0, close=300.5, volume=100.0)
                await repo.create(timestamp=base_time + timedelta(minutes=5), symbol="MSFT", open=300.5, high=303.0, low=300.0, close=302.5, volume=150.0)
                await repo.create(timestamp=base_time + timedelta(minutes=10), symbol="MSFT", open=302.5, high=305.0, low=302.0, close=304.0, volume=200.0)
                
                # Query as of base_time + 5 mins.
                # Must NEVER return the bar from + 10 mins.
                query_time = base_time + timedelta(minutes=5)
                results = await repo.get_market_data(symbol="MSFT", timestamp=query_time)
                
                self.assertEqual(len(results), 2)
                timestamps = [r.timestamp for r in results]
                self.assertIn(base_time, timestamps)
                self.assertIn(base_time + timedelta(minutes=5), timestamps)
                self.assertNotIn(base_time + timedelta(minutes=10), timestamps)
                
                # Test get_latest_market_data
                latest = await repo.get_latest_market_data(symbol="MSFT", timestamp=query_time)
                self.assertIsNotNone(latest)
                self.assertEqual(latest.timestamp, query_time)
                self.assertEqual(latest.close, 302.5)

        self.run_async(run_test())

    def test_database_agent_service(self):
        """Verifies transactional save and load actions on DatabaseAgentService."""
        async def run_test():
            # Save hypothesis
            h_data = {
                "hypothesis_id": "H-SERV-TEST",
                "predictor_variable": "inflation",
                "target_asset_class": "commodities",
                "predicted_direction": "positive",
                "confidence": 0.90
            }
            await self.service.save_hypothesis(h_data)
            
            # Save signal
            sig_data = {
                "signal_id": "S-SERV-TEST",
                "hypothesis_id": "H-SERV-TEST",
                "timestamp": datetime.now(timezone.utc),
                "symbol": "GLD",
                "value": 1.8,
                "status": "PENDING"
            }
            await self.service.save_signal(sig_data)
            
            # Save memory
            await self.service.save_agent_memory("test_agent", "key_1", {"foo": "bar"})
            mem = await self.service.load_agent_memory("test_agent", "key_1")
            self.assertEqual(mem, {"foo": "bar"})
            
            # Save workflow state
            await self.service.save_workflow_state(
                workflow_id="WF-SERV-TEST",
                state_data={"node": "start"},
                status="COMPLETED"
            )

        self.run_async(run_test())

    def test_langgraph_checkpoint_saver(self):
        """Verifies the PostgresCheckpointSaver sync and async persistence API."""
        async def run_test():
            config = {"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}
            checkpoint = {
                "v": 1,
                "id": "chk-point-1",
                "ts": datetime.now(timezone.utc).isoformat(),
                "channel_values": {"p_value": 0.04},
                "channel_versions": {"p_value": "v-p-1"},
                "versions_seen": {}
            }
            metadata = {"source": "test-suite"}
            
            # Sync put & get
            self.checkpointer.put(config, checkpoint, metadata, {"p_value": "v-p-1"})
            retrieved = self.checkpointer.get_tuple(config)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.checkpoint["channel_values"]["p_value"], 0.04)
            self.assertEqual(retrieved.metadata["source"], "test-suite")
            
            # Async aput & aget_tuple
            config_async = {"configurable": {"thread_id": "t-async-1", "checkpoint_ns": ""}}
            checkpoint_async = {
                "v": 1,
                "id": "chk-point-async-1",
                "ts": datetime.now(timezone.utc).isoformat(),
                "channel_values": {"sharpe_ratio": 2.1},
                "channel_versions": {"sharpe_ratio": "v-s-1"},
                "versions_seen": {}
            }
            await self.checkpointer.aput(config_async, checkpoint_async, metadata, {"sharpe_ratio": "v-s-1"})
            retrieved_async = await self.checkpointer.aget_tuple(config_async)
            self.assertIsNotNone(retrieved_async)
            self.assertEqual(retrieved_async.checkpoint["channel_values"]["sharpe_ratio"], 2.1)

        self.run_async(run_test())

if __name__ == "__main__":
    unittest.main()
