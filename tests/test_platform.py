import os
import sys
import unittest
import tempfile
import asyncio
from typing import Dict, Any

# Adjust path to find alphalens
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.session import DatabaseSessionManager
from database.repositories.hypothesis_repository import HypothesisRepository
from alphalens.storage.cache import CacheManager
from alphalens.streaming.kafka_bus import KafkaMessageBus
from alphalens.orchestration.messages_pb2 import AgentMessage, HypothesisPayload, Direction
from alphalens.orchestration.graph import AlphaLensGraph
from alphalens.simulation.backtest import BacktestEngine
from alphalens.contracts.schemas import HypothesisSchema, PredictedDirection

class TestAlphaLensPlatform(unittest.TestCase):
    def setUp(self):
        # Create temp file for SQLite DB fallback
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.db_manager = DatabaseSessionManager(sqlite_path=self.db_path)
        self.db_manager.create_tables()
        
        self.cache = CacheManager()
        self.bus = KafkaMessageBus()

    def tearDown(self):
        asyncio.run(self.db_manager.close_all())

        try:
            os.close(self.db_fd)
            os.unlink(self.db_path)
        except OSError:
            pass
        self.bus.shutdown()

    def test_database_and_schemas(self):
        """
        Verifies database insertion and queries.
        """
        async def run_test():
            async with self.db_manager.async_session() as session:
                repo = HypothesisRepository(session)
                
                # Insert a hypothesis
                await repo.create(
                    hypothesis_id="H-TEST",
                    predictor_variable="vix_regime",
                    target_asset_class="US_equities",
                    predicted_direction="negative",
                    confidence=0.85,
                    theoretical_mechanism="High volatility depresses prices",
                    source_references=[]
                )
                
                # Query
                res = await repo.get_by_id("H-TEST")
                self.assertIsNotNone(res)
                self.assertEqual(res.predictor_variable, "vix_regime")
                self.assertEqual(res.predicted_direction, "negative")

        asyncio.run(run_test())

    def test_cache_manager(self):
        """
        Verifies Redis in-memory cache fallback.
        """
        matrix = {"factor_A": [0.1, -0.2, 0.5], "factor_B": [1.0, 0.5, -0.1]}
        self.cache.set_matrix("test_matrix", matrix)
        retrieved = self.cache.get_matrix("test_matrix")
        self.assertEqual(retrieved, matrix)

    def test_protobuf_serialization(self):
        """
        Verifies Python protobuf serialization and parsing.
        """
        msg = AgentMessage(
            sender="literature_agent",
            recipient="signal_gen_agent",
            timestamp=1717387200.0,
            priority=1,
            hypothesis=HypothesisPayload(
                hypothesis_id="H-01",
                predictor_variable="credit_spread_slope",
                target_asset_class="US_HY_bonds",
                predicted_direction=Direction.NEGATIVE,
                confidence=0.87,
                theoretical_mechanism="Term structure dynamics",
                source_references=["arXiv:2301"]
            )
        )
        
        serialized = msg.SerializeToString()
        parsed = AgentMessage().ParseFromString(serialized)
        
        self.assertEqual(parsed.sender, "literature_agent")
        self.assertEqual(parsed.hypothesis.hypothesis_id, "H-01")
        self.assertEqual(parsed.hypothesis.predicted_direction, Direction.NEGATIVE)

    def test_backtest_cost_modeling(self):
        """
        Verifies transaction cost calculation (Kyle's lambda market impact).
        """
        engine = BacktestEngine(commission_rate=0.0005, bid_ask_spread=0.0010)
        
        # Scenario: order 10,000 shares of stock with price $100, V_ADV 1,000,000, volatility 0.02 daily
        q = 10000.0
        v_adv = 1000000.0
        volatility = 0.02
        price = 100.0
        
        tc = engine.compute_transaction_costs(q, v_adv, volatility, price)
        
        # Kyle's lambda impact = 0.02 * sqrt(10,000 / 1,000,000) * 100 = 0.02 * 0.1 * 100 = 0.2 per share
        # Spread + commission = (0.001/2 + 0.0005) * 100 = (0.0005 + 0.0005) * 100 = 0.1 per share
        # Total cost per share = 0.3
        # Total absolute cost = 10,000 * 0.3 = $3,000
        self.assertAlmostEqual(tc, 3000.0)

    def test_langgraph_routing(self):
        """
        Verifies state machine loops and conditional branching.
        """
        # Node handlers that mock agent executions
        def lit_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"hypothesis": "mock_hypothesis", "current_node": "literature_agent"}

        def signal_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"current_node": "signal_gen_agent"}

        # Custom counter to verify refinement loop
        class Counter:
            def __init__(self):
                self.calls = 0
        
        causal_counter = Counter()

        def causal_node(state: Dict[str, Any]) -> Dict[str, Any]:
            causal_counter.calls += 1
            # First call fails validation (p-value = 0.10) to trigger loop refinement
            # Second call passes (p-value = 0.02) to proceed to backtest
            p_val = 0.10 if causal_counter.calls == 1 else 0.02
            return {"p_value": p_val, "current_node": "causal_validation_agent"}

        def backtest_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Returns Sharpe ratio >= 1.0 to proceed to portfolio agent
            return {"sharpe_ratio": 1.5, "current_node": "backtest_agent"}

        def portfolio_node(state: Dict[str, Any]) -> Dict[str, Any]:
            return {"current_node": "portfolio_agent"}

        graph = AlphaLensGraph(
            literature_agent_fn=lit_node,
            signal_gen_agent_fn=signal_node,
            causal_validation_agent_fn=causal_node,
            backtest_agent_fn=backtest_node,
            portfolio_agent_fn=portfolio_node
        )

        final_state = graph.run("test query")
        
        # Check that we visited causal_validation_agent twice (1 loop back to signal gen and back)
        self.assertEqual(causal_counter.calls, 2)
        # Check that we reached the portfolio node successfully
        self.assertEqual(final_state["current_node"], "portfolio_agent")
        self.assertEqual(final_state["sharpe_ratio"], 1.5)
        self.assertEqual(final_state["p_value"], 0.02)

if __name__ == "__main__":
    unittest.main()
