import os
import sys
import unittest
import tempfile
import asyncio
from datetime import datetime
from typing import Dict, Any

# Adjust path to find alphalens / database
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.session import DatabaseSessionManager
from database.service import DatabaseAgentService, PostgresCheckpointSaver
from alphalens.orchestration.graph import AlphaLensGraph
from alphalens.contracts.schemas import HypothesisSchema, PredictedDirection

class TestAgentDatabaseIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create temporary database file for integration test
        cls.db_fd, cls.db_path = tempfile.mkstemp()
        cls.db = DatabaseSessionManager(sqlite_path=cls.db_path)
        cls.db.create_tables()
        cls.checkpointer = PostgresCheckpointSaver(session_manager=cls.db)

    @classmethod
    def tearDownClass(cls):
        # Close connection and cleanup db file
        asyncio.run(cls.db.close_all())
        
        try:
            os.close(cls.db_fd)
            os.unlink(cls.db_path)
        except OSError:
            pass

    def run_async(self, coro):
        return asyncio.run(coro)

    def test_agent_graph_checkpointing_flow(self):
        """
        Verifies that when the LangGraph agent workflow runs,
        the PostgresCheckpointSaver successfully captures and persists state updates
        at each step in the agent state machine.
        """
        # Define mock agent node functions (so we don't require LLM calls for flow validation)
        def mock_lit(state: Dict[str, Any]) -> Dict[str, Any]:
            hyp = HypothesisSchema(
                hypothesis_id="H-INT-001",
                predictor_variable="credit_slope",
                target_asset_class="bonds",
                predicted_direction=PredictedDirection.NEGATIVE,
                confidence=0.88,
                theoretical_mechanism="Term structure dynamics",
                source_references=["sourceA"]
            )
            return {"hypothesis": hyp, "current_node": "literature_agent"}

        def mock_sig(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "information_coefficient": 0.065,
                "information_ratio": 0.95,
                "half_life_days": 4.2,
                "current_node": "signal_gen_agent"
            }

        def mock_causal(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "p_value": 0.02,
                "ate_magnitude": 0.12,
                "current_node": "causal_validation_agent"
            }

        def mock_backtest(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "sharpe_ratio": 1.45,
                "current_node": "backtest_agent"
            }

        def mock_portfolio(state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "portfolio_weights": [0.25, 0.25, 0.25, 0.25],
                "current_node": "portfolio_agent"
            }

        # Initialize the graph with the checkpointer and mock agent handlers
        graph = AlphaLensGraph(
            literature_agent_fn=mock_lit,
            signal_gen_agent_fn=mock_sig,
            causal_validation_agent_fn=mock_causal,
            backtest_agent_fn=mock_backtest,
            portfolio_agent_fn=mock_portfolio,
            checkpointer=self.checkpointer
        )

        config = {"configurable": {"thread_id": "thread-integration-test-999"}}
        
        # Run the workflow
        final_state = graph.run("credit spread slope", config=config)
        self.assertEqual(final_state["current_node"], "portfolio_agent")
        self.assertEqual(final_state["sharpe_ratio"], 1.45)

        # Retrieve the checkpoint state directly from the checkpointer database to verify persistence
        saved_state = self.checkpointer.get_tuple(config)
        self.assertIsNotNone(saved_state, "No checkpoint was persisted for the thread!")
        
        checkpoint = saved_state.checkpoint
        self.assertEqual(checkpoint["channel_values"]["current_node"], "portfolio_agent")
        self.assertEqual(checkpoint["channel_values"]["sharpe_ratio"], 1.45)
        
        # Check that we can access the saved hypothesis object
        hyp_obj = checkpoint["channel_values"]["hypothesis"]
        self.assertEqual(hyp_obj.hypothesis_id, "H-INT-001")
        self.assertEqual(hyp_obj.predictor_variable, "credit_slope")

        print("Integration Test Passed: Agent workflow persisted all checkpoints successfully!")

if __name__ == "__main__":
    unittest.main()
